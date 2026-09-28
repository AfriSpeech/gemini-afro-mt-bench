"""Stage 2 - the full word-level benchmark.

For every language that survived screening, every BU 200 Word English term is
round-tripped:

    English --translate--> <target language> --translate back--> English

Each direction is a separate, stateless `generate_content` call, so the
back-translator never sees the English source word. A word passes when the
back-translation normalises to the source word.

Writes results/benchmark.jsonl (resumable) and results/benchmark.json (summary).
"""
from __future__ import annotations

import asyncio
import json
import time

from common import (
    BACKWARD_SCHEMA,
    CONCURRENCY,
    DATA,
    MODEL,
    RESULTS,
    TRANSLATION_SCHEMA,
    assert_no_leak,
    backward_prompt,
    call_json,
    client,
    forward_prompt,
    normalise,
    prompt_collision,
    score_match,
)

SCREENING = RESULTS / "screening.json"
PARTIAL = RESULTS / "benchmark.partial.jsonl"
WORDS_JSON = DATA / "wordlist.json"
OUT = RESULTS / "benchmark.json"

# BU 200 label -> afriso ISO 639-3, so Gemini's output can be compared against
# the project's human-curated translation for the 9 published languages.
BU_LANG_TO_ISO = {
    "isiXhosa": "xho",
    "isiZulu": "zul",
    "Amharic": "amh",
    "Wolof": "wol",
    "Hausa": "hau",
    "kiSwahili": "swh",
    "Igbo": "ibo",
    "Akan Twi": "twi",
    "Mandinka": "mnk",
}
ISO_TO_BU = {v: k for k, v in BU_LANG_TO_ISO.items()}


def load_resume() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["iso639_3"], r["word"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def one_word(aio, lang: dict, w: dict) -> dict:
    src = w["english"]
    fwd = await call_json(aio, forward_prompt(src, lang), TRANSLATION_SCHEMA)
    target = ((fwd or {}).get("translation") or "").strip()

    res = {
        "iso639_3": lang["iso639_3"],
        "word": src,
        "target": target,
        "back_translation": "",
        "readable": False,
        "exact": False,
        "close": False,
        "echoed_english": bool(target) and normalise(target) == normalise(src),
        # The ISO 639-3 code is required in the prompt, but a few codes are also
        # English words (bus, box, hoe, man, two) and a few language names
        # contain one. Those pairs are not evidence of translation.
        "prompt_collision": prompt_collision(src, lang),
        # Gemini reproduced the source token inside its "translation" (plural
        "target_has_source_token": assert_no_leak(src, target),
    }
    # An echoed English "translation" is an automatic fail and there is nothing
    # to back-translate, so it never reaches the back-translation call.
    if not target or res["echoed_english"]:
        return res

    bwd = await call_json(aio, backward_prompt(target, lang), BACKWARD_SCHEMA)
    if bwd and bwd.get("is_translation") is not False:
        back = (bwd.get("back_translation") or "").strip()
        res["back_translation"] = back
        res["readable"] = True
        res["exact"], res["close"] = score_match(src, back)
    return res


async def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap languages (0 = all)")
    ap.add_argument("--words", type=int, default=0, help="cap words per language")
    args = ap.parse_args()

    screening = json.load(open(SCREENING, encoding="utf-8"))
    langs = [r for r in screening["results"] if r["passed_screening"]]
    langs = [r for r in langs if not r.get("error")]
    if args.limit:
        langs = langs[: args.limit]
    langs.sort(key=lambda r: (-r["screen_score"], r["name"]))

    wl = json.load(open(WORDS_JSON, encoding="utf-8"))
    words = wl["words"]
    if args.words:
        words = words[: args.words]

    done = load_resume()
    print(f"model={MODEL} concurrency={CONCURRENCY}", flush=True)
    print(f"languages={len(langs)} words={len(words)} "
          f"calls={len(langs)*len(words)*2}", flush=True)
    print(f"already done={len(done)}", flush=True)

    aio = client().aio
    call_sem = asyncio.Semaphore(CONCURRENCY)
    lang_sem = asyncio.Semaphore(max(8, CONCURRENCY // 12))
    fh = open(PARTIAL, "a", encoding="utf-8")
    stats = {"n": 0, "exact": 0, "t0": time.time()}
    lock = asyncio.Lock()

    async def guarded(coro_fn, *a):
        async with call_sem:
            return await coro_fn(*a)

    async def run_lang(lang: dict) -> None:
        async with lang_sem:
            todo = [w for w in words
                    if (lang["iso639_3"], w["english"]) not in done]
            if not todo:
                return
            results = await asyncio.gather(
                *[guarded(one_word, aio, lang, w) for w in todo]
            )
            bu_label = ISO_TO_BU.get(lang["iso639_3"])
            for w, r in zip(todo, results):
                r["category"] = w["category"]
                r["frequency_band"] = w.get("frequency_band")
                src_ref = (w.get("reference_translations") or {}).get(bu_label or "")
                if src_ref:
                    r["bu_reference"] = src_ref
                    r["matches_reference"] = normalise(r["target"]) in {
                        normalise(x) for x in str(src_ref).split("/")
                    }
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                async with lock:
                    stats["n"] += 1
                    stats["exact"] += int(r["exact"])
            async with lock:
                fh.flush()
                n, ex = stats["n"], stats["exact"]
                el = time.time() - stats["t0"]
                rate = n / max(el, 1e-9)
                left = (len(langs) * len(words) - n) / max(rate, 1e-9)
                print(
                    f"  {n}/{len(langs)*len(words)} words "
                    f"({100*n/(len(langs)*len(words)):.1f}%) | "
                    f"running pass={100*ex/max(n,1):.1f}% | "
                    f"{rate:.0f} w/s | eta {left/60:.0f}m | "
                    f"{lang['name']}",
                    flush=True,
                )

    await asyncio.gather(*[run_lang(l) for l in langs])
    fh.close()
    print(f"\ndone in {(time.time()-stats['t0'])/60:.1f} min", flush=True)
    print(f"aggregate exact pass: {100*stats['exact']/max(stats['n'],1):.1f}%", flush=True)
    # The genai client keeps a pooled httpx connection alive; without this the
    # pool is torn down at interpreter exit and logs a spurious
    # "Event loop is closed" traceback after the results are already written.
    closer = getattr(aio, "aclose", None)
    if closer is not None:
        try:
            await closer()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    asyncio.run(main())
