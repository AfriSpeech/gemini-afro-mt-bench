"""Build the v2 word list: Bible English words in three part-of-speech tiers,
each split into frequency bands.

Nouns, adjectives and numbers are separate tiers because they fail differently.
A noun has one ordinary equivalent; an adjective often has a handful of
synonyms and a natural periphrastic rendering; a numeral is a closed, often
irregular set. Blending them into one score hides which of those is failing, so
each tier is reported on its own.

Banding axis, per tier:

  noun       GhanaNouns news+research+speech count. The word must be real
             vocabulary Ghanaian text actually uses, and its frequency there is
             an independent measure of how much support a translator can lean on.
  adjective  English-Bible token count.
  number     English-Bible token count.

GhanaNouns cannot band adjectives or numerals: it is a noun inventory, so only
147 of the 1,779 spaCy adjective lemmas in this text appear in it, and almost
all of those are nouns spaCy mis-tagged.

Output: data/wordlist.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA,
    MIN_VERSES,
    N_BANDS,
    PER_BAND,
    SEED,
    TIERS,
    read_csv,
    write_json,
)

BAND_NAMES = ["frequent", "mid", "rare"]
# Verses actually searched per (language, word). scripts/13_bench_corpus.py
# applies the same cap, after intersecting with the target version.
MAX_VERSES = 20


def ghana_counts() -> dict[str, float]:
    """Single-word GhanaNouns entries -> news+research+speech count."""
    counts: dict[str, float] = {}
    for row in read_csv(DATA / "ghana-nouns.csv"):
        phrase = row["phrase"].strip().lower()
        if not phrase.isalpha():  # drops multiword phrases and hyphenates
            continue
        try:
            n = (float(row["news_count"]) + float(row["research_count"])
                 + float(row["speech_count"]))
        except (TypeError, ValueError):
            continue
        counts[phrase] = counts.get(phrase, 0.0) + n
    return counts


def main() -> None:
    lex = json.loads((DATA / "bible_lexicon.json").read_text(encoding="utf-8"))
    ghana = ghana_counts()
    rng = random.Random(SEED)

    words: list[dict] = []
    report: dict[str, dict] = {}

    for pos, cfg in TIERS.items():
        source = cfg["band_source"]
        pool = [w for w, rec in lex.items()
                if rec["pos"] == pos and not rec["proper"] and rec["n_verses"] >= MIN_VERSES]
        if source == "ghana_count":
            # Membership in GhanaNouns is the eligibility test for this tier,
            # not just the banding axis: the point of the tier is that the word
            # is vocabulary Ghanaian text uses.
            pool = [w for w in pool if ghana.get(w, 0) > 0]
            key = lambda w: ghana[w]                       # noqa: E731
        else:
            key = lambda w: float(lex[w]["count"])          # noqa: E731
        pool.sort(key=lambda w: (-key(w), w))  # most frequent first

        if not pool:
            raise SystemExit(f"no eligible {pos}s")

        size = len(pool) // N_BANDS
        n_per_band = min(PER_BAND, size)
        tier_rows = []
        for b in range(N_BANDS):
            cell = pool[b * size:(b + 1) * size] if b < N_BANDS - 1 else pool[b * size:]
            picked = rng.sample(cell, n_per_band)
            for lemma in sorted(picked):
                rec = lex[lemma]
                # Every verse the word occurs in is kept, not a fixed sample: a
                # YouVersion file often covers only part of the Bible (one
                # Gospel, or the New Testament), so which verses are usable
                # differs per language and can only be decided once the version
                # is known. 13_bench_corpus.py takes the first MAX_VERSES that
                # the target version actually has.
                tier_rows.append({
                    "word": lemma,
                    "pos": pos,
                    "band": BAND_NAMES[b],
                    "band_source": source,
                    "band_value": key(lemma),
                    "ghana_count": ghana.get(lemma, 0.0),
                    "bible_count": rec["count"],
                    "bible_n_verses": rec["n_verses"],
                    "forms": rec["forms"],
                    "verses": rec["verses"],
                })
        words.extend(tier_rows)
        report[pos] = {
            "eligible_pool": len(pool),
            "per_band": n_per_band,
            "sampled": len(tier_rows),
            "shortfall": max(0, PER_BAND * N_BANDS - len(tier_rows)),
        }
        vals = sorted(r["band_value"] for r in tier_rows)
        print(f"{pos:10s} pool={len(pool):4d} sampled={len(tier_rows):3d} "
              f"({n_per_band}/band)  {source} {vals[0]:.0f}–{vals[-1]:.0f} "
              f"(median {vals[len(vals)//2]:.0f})", file=sys.stderr)

    out = {
        "meta": {
            "min_verses": MIN_VERSES,
            "per_band_target": PER_BAND,
            "n_bands": N_BANDS,
            "max_verses": MAX_VERSES,
            "seed": SEED,
            "n_words": len(words),
            "tiers": report,
        },
        "words": words,
    }
    write_json(DATA / "wordlist.json", out)


if __name__ == "__main__":
    main()
