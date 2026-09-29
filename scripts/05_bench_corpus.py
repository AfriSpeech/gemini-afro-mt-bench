"""Stage 3 (v2) - the corpus-grounded benchmark.

For every language that has a YouVersion Bible in AfriSpeech/africa-corpus and
every word in the v2 word list, ask Gemini for the word and then look that
answer up in the target language's own Bible text:

    English noun --translate--> <target term>
                                     |
                        does it occur in that language's Bible?
                        - in the verse the English noun was taken from
                        - anywhere in the corpus

No back-translation and no model-graded judgement: the reference is human
translation work, and the model never grades itself. A word passes when the term
Gemini gave is a term real speakers of that language actually write.

Writes results/v2/bench.partial.<shard>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

from common import (
    SCREEN_MIN_SCORABLE,
    SCREEN_N,
    SCREEN_TIER,
    CorpusIndex,
    DATA,
    MODEL,
    RESULTS,
    TRANSLATION_SCHEMA,
    assert_no_leak,
    call_json,
    client,
    forward_prompt,
    normalise,
    prompt_collision,
)

WORDS = DATA / "wordlist.json"
MANIFEST = DATA / "corpus_manifest.json"
# Mirrors common.MAX_VERSES in 03_build_bands.py: the number of aligned verses
# a single (language, word) pair is allowed to be judged on.
MAX_VERSES = 20
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "96"))


def screen_set(words: list[dict], lang: dict) -> list[dict]:
    """The numeral probe for this language: its most frequent numerals.

    A word whose own spelling is visible in the prompt (a handful of ISO codes
    and language names are also English words) is skipped here: a collision
    would hand the model the answer and turn a genuine failure into a pass.
    """
    cand = [w for w in words
            if w["pos"] == SCREEN_TIER and not prompt_collision(w["word"], lang)]
    cand.sort(key=lambda w: (-w["bible_count"], w["word"]))
    return cand[:SCREEN_N]


def snapshot_dir() -> str:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"))
    if hits:
        return hits[0]
    hits = glob.glob(os.path.expanduser(
        "/mnt/volume_d2wey28/hf_cache/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"))
    if not hits:
        raise SystemExit("africa-corpus snapshot not found")
    return hits[0]


def read_verses(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["verse_key"]: r.get("local") or "" for r in csv.DictReader(f)
                if r.get("verse_key")}


def score(index: CorpusIndex, term: str, verses: list[str]) -> dict:
    """Look `term` up in the target-language Bible.

    `verses` is every English verse the noun occurs in. Which of them are usable
    depends on the target version, so they are filtered here rather than being
    fixed up front: a YouVersion file may be a single Gospel or the New
    Testament only, and a word that occurs only in the Pentateuch would then
    have nothing to align against.
    """
    needle = index.term(term)
    if not needle.usable:
        return {"scorable": False, "reason": "unmatchable_term", "n_aligned": 0}
    present = [k for k in verses if k in index.script][:MAX_VERSES]
    if not present:
        # No shared verse with the English side: the version is too partial to
        # test this word, so it is dropped rather than scored as a failure.
        return {"scorable": False, "reason": "no_aligned_verses", "n_aligned": 0}
    hits = [k for k in present if index.in_verse(needle, k)]
    f_count, s_count, mode = index.count(needle)
    return {
        "scorable": True,
        "n_aligned": len(present),
        "n_hits": len(hits),
        "hit": bool(hits),
        "hit_rate": len(hits) / len(present),
        "hit_majority": len(hits) * 2 >= len(present),
        "corpus_hits": f_count + s_count,
        "corpus_mode": mode,
        "match_mode": mode if hits else "none",
        "example_verse": hits[0] if hits else None,
        # A term written in a script the target Bible never uses can never be
        # found. That is a different failure from giving the wrong word in the
        # right script, and it is a real one -- the model is told the target
        # language and still answers in a script nobody writes it in -- so it
        # is scored as a miss but counted separately in the aggregate.
        "script_match": needle.latin == (index.latin_share > 0.5),
    }


async def run_language(aio, lang: dict, words: list[dict], part: Path) -> dict:
    verses = read_verses(Path(snapshot_dir()) / lang["file"])
    t0 = time.time()
    index = CorpusIndex(verses)
    index_secs = time.time() - t0
    lang["latin_share"] = round(index.latin_share, 3)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(w: dict) -> dict:
        src = w["word"]
        row = {
            "iso639_3": lang["iso639_3"],
            "name": lang["name"],
            "family": lang.get("family"),
            "region": lang.get("region"),
            "version_id": lang["version_id"],
            "version_title": lang.get("version_title"),
            "corpus_verses": lang["n_verses"],
            "corpus_latin_share": lang.get("latin_share"),
            "word": src,
            "pos": w["pos"],
            "band": w["band"],
            "ghana_count": w["ghana_count"],
            "bible_count": w["bible_count"],
            "bible_n_verses": w["bible_n_verses"],
            "core": w.get("core", False),
            "prompt_collision": prompt_collision(src, lang),
            "term": "",
            "scorable": False,
        }
        async with sem:
            fwd = await call_json(aio, forward_prompt(src, lang), TRANSLATION_SCHEMA)
        term = ((fwd or {}).get("translation") or "").strip()
        row["term"] = term
        row["echoed_english"] = bool(term) and normalise(term) == normalise(src)
        row["target_has_source_token"] = assert_no_leak(src, term)
        if not term:
            row["reason"] = "empty_translation"
            return row
        row.update(score(index, term, w["verses"]))
        return row

    def flush(rows: list[dict]) -> None:
        with open(part, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- phase 1: numeral probe (kept as an informative diagnostic)
    probes = screen_set(words, lang)
    screen_rows = await asyncio.gather(*[one(w) for w in probes])
    good_screen = [r for r in screen_rows if r["scorable"] and not r["prompt_collision"]]
    screen_hits = sum(1 for r in good_screen if r["hit"])
    screen_enough = len(good_screen) >= SCREEN_MIN_SCORABLE
    screen_str = (f"{screen_hits}/{len(good_screen)}" if screen_enough
                  else f"skipped, {len(good_screen)} scorable")
    for r in screen_rows:
        r["screen"] = True

    # ---- phase 2: the rest of the words. Evaluated for EVERY language.
    rest = [w for w in words if w["word"] not in {p["word"] for p in probes}]
    rows = await asyncio.gather(*[one(w) for w in rest])
    allrows = screen_rows + rows
    flush(allrows)
    good = [r for r in allrows if r["scorable"] and not r["prompt_collision"]]
    return {
        "iso639_3": lang["iso639_3"],
        "name": lang["name"],
        "n": len(good),
        "hits": sum(1 for r in good if r["hit"]),
        "screen": screen_str,
        "index_secs": round(index_secs, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    langs = manifest["languages"]
    if args.limit:
        langs = langs[: args.limit]
    langs = [l for i, l in enumerate(langs) if i % args.nshards == args.shard]

    words = json.loads(WORDS.read_text(encoding="utf-8"))["words"]
    part = RESULTS / f"bench.partial.{args.shard}.jsonl"

    done_langs = set()
    if part.exists():
        with open(part, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if not r.get("screen"):
                            done_langs.add(r["iso639_3"])
                    except Exception:
                        pass
        if done_langs:
            print(f"resuming: found {len(done_langs)} already-completed languages in {part.name}",
                  flush=True)

    print(f"shard {args.shard}/{args.nshards} | model={MODEL} | "
          f"languages={len(langs)} words={len(words)} calls={len(langs)*len(words)}",
          flush=True)

    aio = client().aio
    t0 = time.time()
    n = 0
    for i, lang in enumerate(langs, 1):
        if lang["iso639_3"] in done_langs:
            n += 1
            continue
        try:
            st = await run_language(aio, lang, words, part)
        except Exception as e:  # noqa: BLE001 - one bad file must not kill the shard
            print(f"  !! {lang['name']} ({lang['iso639_3']}) failed: {e}", flush=True)
            continue
        n += 1
        rate = 100 * st["hits"] / max(st["n"], 1)
        el = time.time() - t0
        eta = (len(langs) - i) * el / max(i, 1) / 60
        tag = f"hit={rate:.1f}%"
        print(f"  [{i}/{len(langs)}] {st['name']} ({st['iso639_3']}) "
              f"{tag} ({st['hits']}/{st['n']}) screen={st['screen']} "
              f"idx={st['index_secs']}s | {el/60:.1f}m elapsed, eta {eta:.0f}m",
              flush=True)

    print(f"shard {args.shard} done: {n} languages in {(time.time()-t0)/60:.1f} min",
          flush=True)
    closer = getattr(aio, "aclose", None)
    if closer is not None:
        try:
            await closer()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    asyncio.run(main())
