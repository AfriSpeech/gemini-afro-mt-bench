"""Aggregate the raw benchmark into the JSON the dashboard reads.

Outputs
  results/summary.json  per-language scores, per-category and per-band
                        breakdowns, and region/family/category rollups
  results/examples.json a handful of concrete round-trips per language, so the
                        dashboard can show real translations without shipping
                        all 331k rows
  results/details.jsonl.gz  every row, for download and secondary analysis

Scoring rules
  * `prompt_collision` rows are excluded - the ISO 639-3 code in the prompt is
    itself an English word for a few languages, so the model is handed the
    answer for that item.
  * `pass` is the normalised-exact back-translation match.
  * `close` additionally allows one side's tokens to be a subset of the
    other's, so a bare "big head" still credits "head". Reported, not scored.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict

import common
from common import DATA, MODEL, RESULTS, write_json

PARTIAL = RESULTS / "benchmark.partial.jsonl"
SCREENING = RESULTS / "screening.json"
SUMMARY = RESULTS / "summary.json"
EXAMPLES = RESULTS / "examples.json"
DETAILS = RESULTS / "details.jsonl.gz"

# Tier cut-points on the exact back-translation pass rate.
STRONG = 0.60
MEDIUM = 0.30

# Words sampled per language for the dashboard drill-down, one per category
# where possible, so the sample spans the whole list rather than one corner.
EXAMPLE_TARGET = 10


def tier_of(p: float) -> str:
    if p >= STRONG:
        return "strong"
    if p >= MEDIUM:
        return "medium"
    return "weak"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rate(flags: list[bool]) -> float:
    return 100.0 * sum(1 for f in flags if f) / len(flags) if flags else 0.0


def main() -> None:
    screening = json.load(open(SCREENING, encoding="utf-8"))
    screen_by_iso = {r["iso639_3"]: r for r in screening["results"]}
    wordlist = json.load(open(DATA / "wordlist.json", encoding="utf-8"))
    categories = [c["category"] for c in wordlist["categories"]]

    rows: list[dict] = []
    with open(PARTIAL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Reference agreement is re-derived here rather than trusted
                # from the run, because it depends on how two spellings of the
                # same word are compared. BU 200 writes "Bulukondiŋo" where
                # Gemini writes "bulukondingo"; without orthographic folding the
                # metric measures spelling convention, not translation quality.
                #
                # The fold is Latin-only, so a non-Latin script (Amharic is
                # written in Ge'ez) cannot be folded. Those rows fall back to an
                # exact script-aware comparison, and only text with no
                # comparable content is marked incomparable. See
                # common.compare_reference.
                r["matches_reference"] = None
                r["reference_mode"] = common.REF_MODE_NONE
                if r.get("bu_reference"):
                    hit, mode = common.compare_reference(
                        r.get("target", ""), r["bu_reference"]
                    )
                    r["matches_reference"] = hit
                    r["reference_mode"] = mode
                r["reference_comparable"] = r["matches_reference"] is not None
                rows.append(r)
    print(f"raw rows: {len(rows)}")

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lang[r["iso639_3"]].append(r)

    languages = []
    n_collision = n_echo = n_unreadable = 0

    for iso, rs in by_lang.items():
        meta = screen_by_iso.get(iso, {})
        scorable = [r for r in rs if not r.get("prompt_collision")]
        n_collision += len(rs) - len(scorable)
        if not scorable:
            continue

        n_echo += sum(1 for r in scorable if r.get("echoed_english"))
        n_unreadable += sum(1 for r in scorable if not r.get("readable"))

        by_cat: dict[str, list[bool]] = defaultdict(list)
        by_band: dict[str, list[bool]] = defaultdict(list)
        for r in scorable:
            by_cat[r.get("category") or "unknown"].append(r["exact"])
            if r.get("frequency_band"):
                by_band[str(r["frequency_band"])].append(r["exact"])

        ref_rows = [r for r in scorable if r.get("reference_comparable")]
        pr = rate([r["exact"] for r in scorable])

        languages.append(
            {
                "iso639_3": iso,
                "name": meta.get("name", iso),
                "family": meta.get("family", "Unknown"),
                "region": meta.get("region", "Unknown"),
                "screen_score": meta.get("screen_score", 0),
                "n": len(scorable),
                "n_excluded": len(rs) - len(scorable),
                "pass": round(pr, 2),
                "close": round(rate([r["close"] for r in scorable]), 2),
                "tier": tier_of(pr / 100),
                "echoed": round(rate([r.get("echoed_english", False) for r in scorable]), 2),
                "unreadable": round(rate([not r.get("readable") for r in scorable]), 2),
                "by_category": {c: round(rate(by_cat[c]), 1) for c in categories if c in by_cat},
                "by_band": {b: round(rate(by_band[b]), 1) for b in sorted(by_band)},
                "bu_reference_agreement": (
                    round(rate([r["matches_reference"] for r in ref_rows]), 1)
                    if ref_rows
                    else None
                ),
                "n_bu_reference": len(ref_rows),
                "reference_mode": (
                    max(
                        (r["reference_mode"] for r in ref_rows),
                        key=[r["reference_mode"] for r in ref_rows].count,
                    )
                    if ref_rows
                    else None
                ),
                "reference_modes": {
                    m: sum(1 for r in ref_rows if r["reference_mode"] == m)
                    for m in sorted({r["reference_mode"] for r in ref_rows})
                },
            }
        )

    languages.sort(key=lambda x: (-x["pass"], x["name"]))

    def rollup(key: str) -> dict:
        g: dict[str, list] = defaultdict(list)
        for l in languages:
            g[l[key]].append(l)
        return {
            k: {
                "n": len(v),
                "mean_pass": round(mean([x["pass"] for x in v]), 1),
                "max_pass": round(max(x["pass"] for x in v), 1),
                "min_pass": round(min(x["pass"] for x in v), 1),
            }
            for k, v in sorted(g.items(), key=lambda kv: -mean([x["pass"] for x in kv[1]]))
        }

    # Mean pass rate per category, pooled over every language.
    cat_pool: dict[str, list[float]] = defaultdict(list)
    for l in languages:
        for c, v in l["by_category"].items():
            cat_pool[c].append(v)
    n_words_by_cat = {c["category"]: c["n_words"] for c in wordlist["categories"]}
    by_category = {
        cat: {
            "mean_pass": round(mean(cat_pool[cat]), 1),
            "median_pass": round(sorted(cat_pool[cat])[len(cat_pool[cat]) // 2], 1),
            "n_languages": len(cat_pool[cat]),
            "n_words": n_words_by_cat.get(cat, 0),
        }
        for cat in categories
        if cat in cat_pool
    }

    # Second, orthogonal axis: GhanaNouns frequency quintile. Only the 134 of the
    # 204 words that appear in the GhanaNouns inventory carry a count, so bands
    # are built over that subset and are reported as such.
    band_pool: dict[str, list[float]] = defaultdict(list)
    for l in languages:
        for b, v in l["by_band"].items():
            if v is not None:
                band_pool[b].append(v)
    by_band = {
        b: {
            "mean_pass": round(mean(band_pool[b]), 1),
            "n_languages": len(band_pool[b]),
            "n_words": wordlist["frequency_band_sizes"].get(b, 0),
        }
        for b in sorted(band_pool, key=int)
    }

    tier_counts = defaultdict(int)
    for l in languages:
        tier_counts[l["tier"]] += 1

    write_json(
        SUMMARY,
        {
            "meta": {
                "model": MODEL,
                "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                "elapsed_minutes": 55.8,
                "api_calls": 2 * len(rows),
                "n_languages_screened": screening["n_languages"],
                "n_languages_tested": len(languages),
                "n_words": wordlist["n_words"],
                "n_categories": len(categories),
                "n_rows": len(rows),
                "n_excluded_prompt_collision": n_collision,
                "n_echoed_english": n_echo,
                "n_unreadable": n_unreadable,
                "tiers": {"strong": STRONG, "medium": MEDIUM},
                "source": wordlist["source"],
                "source_description": wordlist["source_description"],
                "frequency_source": wordlist["frequency_source"],
                "screen_words": screening["screen_words"],
            },
            "categories": categories,
            "tier_counts": dict(tier_counts),
            "by_category": by_category,
            "by_frequency_band": by_band,
            "by_region": rollup("region"),
            "by_family": rollup("family"),
            "languages": languages,
        },
    )

    # ---- examples: a few concrete round-trips per language -------------------
    examples = {}
    for iso, rs in by_lang.items():
        scorable = [r for r in rs if not r.get("prompt_collision")]
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in scorable:
            by_cat[r.get("category") or "unknown"].append(r)
        picked: list[dict] = []
        cats = sorted(by_cat)
        i = 0
        while len(picked) < EXAMPLE_TARGET and cats:
            c = cats[i % len(cats)]
            if by_cat[c]:
                picked.append(
                    {
                        "word": by_cat[c][i // len(cats) % len(by_cat[c])]["word"],
                        "category": c,
                    }
                )
            i += 1
        chosen = []
        for p in picked:
            m = next(
                (r for r in by_cat[p["category"]] if r["word"] == p["word"]), None
            )
            if m and m not in chosen:
                chosen.append(m)
        ex = [m for m in scorable if m.get("echoed_english") or not m.get("readable")][:2]
        examples[iso] = [
            {
                "word": r["word"],
                "category": r.get("category"),
                "gemini": r.get("target", ""),
                "back": r.get("back_translation", ""),
                "pass": r.get("exact", False),
                "bu_reference": r.get("bu_reference"),
                "matches_reference": r.get("matches_reference"),
                "reference_comparable": r.get("reference_comparable", False),
            }
            for r in chosen + ex
        ]
    write_json(EXAMPLES, examples)

    with gzip.open(DETAILS, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {DETAILS.relative_to(RESULTS.parent)} "
          f"({DETAILS.stat().st_size/1024/1024:.1f} MB gz)")

    print(f"\nlanguages tested: {len(languages)}")
    print(f"tiers: {dict(tier_counts)}")
    print(f"rows excluded for prompt collision: {n_collision}")
    print("\npass rate by category:")
    for c, v in sorted(by_category.items(), key=lambda kv: -kv[1]["mean_pass"]):
        print(f"  {c:20s} {v['mean_pass']:5.1f}%  (n_words={v['n_words']})")
    print("\npass rate by GhanaNouns frequency band (quintile, 1 = most frequent):")
    for b, v in by_band.items():
        print(f"  band {b}  {v['mean_pass']:5.1f}%  "
              f"(n_words={v['n_words']}, n_languages={v['n_languages']})")
    print("\ntop 20 languages:")
    for l in languages[:20]:
        print(f"  {l['pass']:5.1f}%  {l['name']:24s} {l['iso639_3']:4s} "
              f"{l['family']:22s} {l['region']}")
    print("\nBU reference agreement (human-curated 200 Word data):")
    for l in sorted(
        (l for l in languages if l["bu_reference_agreement"] is not None),
        key=lambda l: -l["bu_reference_agreement"],
    ):
        print(f"  {l['bu_reference_agreement']:5.1f}%  {l['name']:14s} "
              f"(n={l['n_bu_reference']}, {l['reference_mode']})  "
              f"round-trip pass={l['pass']}%")


if __name__ == "__main__":
    main()
