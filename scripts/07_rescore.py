"""Re-score existing benchmark translation rows across ALL Bible versions and occurrences.

Takes the Gemini translations from results/details.jsonl.gz, indexes every
available Bible version for each language (instead of only a single version),
and checks every single verse occurrence where the English word appears (no 20-verse cap).
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import DATA, RESULTS, CorpusIndex


def snapshot_dir() -> Path:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"))
    if hits:
        return Path(hits[0])
    hits = glob.glob(os.path.expanduser(
        "/mnt/volume_d2wey28/hf_cache/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"))
    if not hits:
        raise SystemExit("africa-corpus snapshot not found")
    return Path(hits[0])


def read_verses(snap: Path, files: list[str]) -> dict[str, str]:
    combined = defaultdict(list)
    for fname in files:
        p = snap / fname
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vk = r.get("verse_key")
                loc = r.get("local")
                if vk and loc:
                    combined[vk].append(loc)
    return {k: " \n ".join(vs) for k, vs in combined.items()}


def score(index: CorpusIndex, term: str, verses: list[str]) -> dict:
    needle = index.term(term)
    if not needle.usable:
        return {"scorable": False, "reason": "unmatchable_term", "n_aligned": 0}
    present = [k for k in verses if k in index.script]
    if not present:
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
        "script_match": needle.latin == (index.latin_share > 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    snap = snapshot_dir()
    manifest = json.loads((DATA / "corpus_manifest.json").read_text(encoding="utf-8"))
    langs = manifest["languages"]
    langs_dict = {l["iso639_3"]: l for l in langs}

    words_data = json.loads((DATA / "wordlist.json").read_text(encoding="utf-8"))["words"]
    word_verses = {w["word"]: w["verses"] for w in words_data}

    # Partition languages for this shard
    shard_langs = [l for i, l in enumerate(langs) if i % args.nshards == args.shard]
    shard_isos = {l["iso639_3"] for l in shard_langs}
    print(f"shard {args.shard}/{args.nshards}: re-scoring {len(shard_langs)} languages across all versions...")

    # Load all existing rows for this shard's languages
    rows_by_lang: dict[str, list[dict]] = defaultdict(list)
    src_details = RESULTS / "details.jsonl.gz"
    if not src_details.exists():
        sys.exit(f"{src_details} not found")

    with gzip.open(src_details, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            iso = r["iso639_3"]
            if iso in shard_isos:
                rows_by_lang[iso].append(r)

    part_out = RESULTS / f"bench.partial.{args.shard}.jsonl"
    t0 = time.time()
    n_total_hits = 0
    n_total_scorable = 0

    with open(part_out, "w", encoding="utf-8") as out:
        for i, lang in enumerate(shard_langs, 1):
            iso = lang["iso639_3"]
            files = lang.get("files") or [lang["file"]]
            verses = read_verses(snap, files)
            index = CorpusIndex(verses)
            latin_share = round(index.latin_share, 3)

            rs = rows_by_lang.get(iso, [])
            lang_hits = 0
            lang_scorable = 0

            for r in rs:
                term = r.get("term", "")
                r["corpus_verses"] = len(verses)
                r["corpus_latin_share"] = latin_share
                r["n_versions"] = len(files)

                if term:
                    res = score(index, term, word_verses.get(r["word"], []))
                    r.update(res)
                else:
                    r["scorable"] = False
                    r["hit"] = False
                    r["reason"] = "empty_translation"

                if r.get("scorable") and not r.get("prompt_collision"):
                    lang_scorable += 1
                    if r.get("hit"):
                        lang_hits += 1

                out.write(json.dumps(r, ensure_ascii=False) + "\n")

            n_total_hits += lang_hits
            n_total_scorable += lang_scorable
            rate = 100 * lang_hits / max(lang_scorable, 1)
            el = time.time() - t0
            eta = (len(shard_langs) - i) * el / max(i, 1) / 60
            print(f"  [{i}/{len(shard_langs)}] {lang['name']} ({iso}) "
                  f"hit={rate:.1f}% ({lang_hits}/{lang_scorable}) "
                  f"versions={len(files)} | {el:.1f}s elapsed, eta {eta:.1f}m", flush=True)

    print(f"shard {args.shard} finished in {(time.time()-t0)/60:.1f}m. Wrote {part_out}")


if __name__ == "__main__":
    main()
