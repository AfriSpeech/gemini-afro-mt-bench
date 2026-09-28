"""Build the benchmark word list from the BU 200 Word Project English side.

Grouping is by the 16 categories inherent to the BU 200 Word Project (body
parts, food and drink, animals, numbers, ...), which is the structure the
dataset was designed around. GhanaNouns supplies an independent frequency count
for 134 of the 204 words, giving a second, orthogonal axis: how well Gemini does
on frequent vs rare vocabulary.

Writes data/wordlist.json
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request

from common import DATA, write_json

BU = DATA / "bu200.json"
NOUNS_CSV = DATA / "ghana-nouns.csv"
NOUNS_URL = (
    "https://raw.githubusercontent.com/GhanaNLP/GhanaNouns/main/data/ghana-nouns.csv"
)
OUT = DATA / "wordlist.json"

MIN_REF_LANGS = 2
SINGLE_WORD = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
N_FREQ_BANDS = 5


def ghana_nouns_counts() -> dict[str, float]:
    """word -> news+research+speech total occurrence count."""
    if not NOUNS_CSV.exists():
        print(f"downloading {NOUNS_URL}", flush=True)
        urllib.request.urlretrieve(NOUNS_URL, NOUNS_CSV)
    counts: dict[str, float] = {}
    with open(NOUNS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = (r.get("phrase") or "").strip()
            if SINGLE_WORD.match(p):
                counts[p.lower()] = (
                    float(r["news_count"])
                    + float(r["research_count"])
                    + float(r["speech_count"])
                )
    return counts


def main() -> None:
    bu = json.load(open(BU, encoding="utf-8"))
    counts = ghana_nouns_counts()

    words = [
        w
        for w in bu["words"]
        if SINGLE_WORD.match(w["english"]) and w["n_reference_languages"] >= MIN_REF_LANGS
    ]
    words.sort(key=lambda w: w["english"].lower())
    print(f"benchmark words: {len(words)} (single-word, >={MIN_REF_LANGS} editions)")

    for w in words:
        w["ghana_nouns_total"] = counts.get(w["english_lower"])

    matched = [w for w in words if w["ghana_nouns_total"] is not None]
    print(f"with GhanaNouns frequency: {len(matched)}/{len(words)}")

    # Frequency bands over the matched subset only, so the axis is honest about
    # its own coverage rather than padding ranks with absent words.
    ranked = sorted(matched, key=lambda w: -w["ghana_nouns_total"])
    for i, w in enumerate(ranked):
        w["frequency_band"] = min(N_FREQ_BANDS, i * N_FREQ_BANDS // len(ranked)) + 1

    by_cat: dict[str, list] = {}
    for w in words:
        by_cat.setdefault(w["category"], []).append(w)

    write_json(
        OUT,
        {
            "source": bu["source"],
            "source_description": bu["description"],
            "frequency_source": NOUNS_URL,
            "n_words": len(words),
            "n_with_frequency": len(matched),
            "n_categories": len(by_cat),
            "min_reference_languages": MIN_REF_LANGS,
            "categories": [
                {
                    "category": c,
                    "n_words": len(v),
                    "words": [w["english"] for w in v],
                }
                for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
            ],
            "frequency_band_sizes": {
                str(b): sum(1 for w in matched if w.get("frequency_band") == b)
                for b in range(1, N_FREQ_BANDS + 1)
            },
            "words": words,
        },
    )

    print(f"\n{'category':22s} {'words':>5s}")
    for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:20s} {len(v):5d}   e.g. {[w['english'] for w in v[:6]]}")
    bands = {
        b: sum(1 for w in matched if w.get("frequency_band") == b)
        for b in range(1, N_FREQ_BANDS + 1)
    }
    print("\nfrequency bands:", bands)


if __name__ == "__main__":
    main()
