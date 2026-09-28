"""Stage 1 - screening: 3 very common words per language.

Each African language is round-tripped through 3 near-universal concrete nouns
(head / water / mother). Languages that score 0/3 are not real translation
targets for Gemini - the model is echoing English, transliterating, or inventing
text - and are dropped before the expensive 250-word benchmark.

Writes results/screening.json.
"""
from __future__ import annotations

import asyncio
import json
import time

from common import (
    BACKWARD_SCHEMA,
    DATA,
    MODEL,
    RESULTS,
    SCREEN_CONCURRENCY,
    SCREEN_WORDS,
    TRANSLATION_SCHEMA,
    backward_prompt,
    call_json,
    client,
    forward_prompt,
    score_match,
    write_json,
)

OUT = RESULTS / "screening.json"
RESUME = RESULTS / "screening.partial.jsonl"


def load_langs() -> list[dict]:
    return json.load(open(DATA / "languages.json", encoding="utf-8"))


def load_done() -> dict:
    done: dict[str, dict] = {}
    if RESUME.exists():
        for line in RESUME.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["iso639_3"]] = r
                except json.JSONDecodeError:
                    pass
    return done


async def screen_lang(aio, lang: dict) -> dict:
    """Round-trip the 3 screen words for one language. Words run concurrently."""
    words = SCREEN_WORDS

    async def fwd(w):
        r = await call_json(aio, forward_prompt(w, lang), TRANSLATION_SCHEMA)
        t = (r or {}).get("translation") or ""
        return w, t.strip()

    fwd_res = await asyncio.gather(*[fwd(w) for w in words])

    async def bwd(item):
        w, t = item
        if not t:
            return w, t, "", False
        r = await call_json(aio, backward_prompt(t, lang), BACKWARD_SCHEMA)
        if not r:
            return w, t, "", False
        if r.get("is_translation") is False:
            return w, t, "", False
        return w, t, (r.get("back_translation") or "").strip(), True

    bwd_res = await asyncio.gather(*[bwd(i) for i in fwd_res])

    items, passes = [], 0
    for (w, fwd_txt), (w2, fwd_txt2, back, ok) in zip(fwd_res, bwd_res):
        exact = close = False
        if ok and back:
            exact, close = score_match(w, back)
        if exact:
            passes += 1
        items.append(
            {
                "word": w,
                "forward": fwd_txt,
                "back_translation": back,
                "readable": ok,
                "exact": exact,
                "close": close,
            }
        )

    return {
        "iso639_3": lang["iso639_3"],
        "name": lang["name"],
        "family": lang["family"],
        "region": lang["region"],
        "screen_score": passes,
        "screen_max": len(words),
        "passed_screening": passes > 0,
        "echoed_english": sum(
            1 for it in items if it["forward"] and it["forward"].strip().lower() == it["word"]
        ),
        "items": items,
    }


async def main() -> None:
    langs = load_langs()
    done = load_done()
    todo = [l for l in langs if l["iso639_3"] not in done]
    print(f"languages={len(langs)} already_done={len(done)} todo={len(todo)}", flush=True)
    print(f"model={MODEL} words={SCREEN_WORDS} concurrency={SCREEN_CONCURRENCY}", flush=True)

    aio = client().aio
    sem = asyncio.Semaphore(SCREEN_CONCURRENCY)
    results: list[dict] = list(done.values())
    t0 = time.time()
    fh = open(RESUME, "a", encoding="utf-8")
    n = [0]

    async def run(lang):
        async with sem:
            try:
                r = await screen_lang(aio, lang)
            except Exception as e:  # noqa: BLE001
                r = {
                    "iso639_3": lang["iso639_3"], "name": lang["name"],
                    "family": lang["family"], "region": lang["region"],
                    "screen_score": 0, "screen_max": len(SCREEN_WORDS),
                    "passed_screening": False, "echoed_english": 0,
                    "items": [], "error": str(e)[:200],
                }
            results.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                fh.flush()
                el = time.time() - t0
                ok = sum(1 for x in results if x["passed_screening"])
                print(
                    f"  {n[0]}/{len(todo)} done | {el:.0f}s | "
                    f"{n[0]/max(el,1e-9):.1f} lang/s | survivors={ok}",
                    flush=True,
                )

    await asyncio.gather(*[run(l) for l in todo])
    fh.close()

    results.sort(key=lambda r: (-r["screen_score"], r["name"]))
    survivors = [r for r in results if r["passed_screening"]]
    write_json(
        OUT,
        {
            "model": MODEL,
            "screen_words": SCREEN_WORDS,
            "n_languages": len(results),
            "n_survivors": len(survivors),
            "elapsed_seconds": round(time.time() - t0, 1),
            "results": results,
        },
    )

    print(f"\nscreened {len(results)} languages in {time.time()-t0:.0f}s")
    print(f"survivors (score > 0/3): {len(survivors)}")
    from collections import Counter

    for k, v in sorted(Counter(r["screen_score"] for r in results).items()):
        print(f"  score {k}/3 : {v} languages")
    print("\ntop 25 survivors:")
    for r in survivors[:25]:
        print(
            f"  {r['screen_score']}/3  {r['name']:22s} {r['iso639_3']:4s} "
            f"{r['region']:14s} {[i['forward'] for i in r['items']]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
