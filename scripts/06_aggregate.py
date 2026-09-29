"""Aggregate the raw v2 corpus benchmark into summary, examples, and details.

Outputs
  results/v2/summary.json      per-language scores, breakdowns by POS, band, region, family
  results/v2/examples.json     sample translation lookups per language (hits and misses)
  results/v2/details.jsonl.gz  gzipped row-level data for download and secondary analysis
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import DATA, MODEL, RESULTS, write_json

SUMMARY = RESULTS / "summary.json"
EXAMPLES = RESULTS / "examples.json"
DETAILS = RESULTS / "details.jsonl.gz"
MANIFEST = DATA / "corpus_manifest.json"
WORDS = DATA / "wordlist.json"

STRONG = 60.0
MEDIUM = 30.0
EXAMPLE_TARGET = 10


def tier_of(score: float, screened_out: bool) -> str:
    if screened_out or score <= 0.0:
        return "unsupported"
    if score >= STRONG:
        return "strong"
    if score >= MEDIUM:
        return "medium"
    return "weak"


def rate(flags: list[bool]) -> float:
    return round(100.0 * sum(1 for f in flags if f) / len(flags), 2) if flags else 0.0


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=str(RESULTS),
                    help="Directory containing bench.partial.*.jsonl files")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    langs_meta = {l["iso639_3"]: l for l in manifest["languages"]}
    wordlist = json.loads(WORDS.read_text(encoding="utf-8"))
    core_words = {w["word"] for w in wordlist["words"] if w.get("core")}

    part_files = sorted(results_dir.glob("bench.partial.*.jsonl"))
    if not part_files:
        print(f"no bench.partial.*.jsonl found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"reading {len(part_files)} partial files from {results_dir}...")
    rows_by_lang: dict[str, list[dict]] = defaultdict(list)
    total_raw_rows = 0

    with gzip.open(DETAILS, "wt", encoding="utf-8") as out_details:
        for p in part_files:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total_raw_rows += 1
                    out_details.write(json.dumps(r, ensure_ascii=False) + "\n")
                    rows_by_lang[r["iso639_3"]].append(r)

    print(f"loaded {total_raw_rows:,} rows across {len(rows_by_lang)} languages; wrote {DETAILS.name}")

    languages = []
    examples: dict[str, list[dict]] = {}

    for iso, rs in rows_by_lang.items():
        meta = langs_meta.get(iso, {})
        screen_rows = [r for r in rs if r.get("screen")]
        screened_out = any(r.get("screened_out") for r in screen_rows)

        # Usable rows: scorable, not prompt collision
        scorable = [r for r in rs if r.get("scorable") and not r.get("prompt_collision")]
        screen_scorable = [r for r in screen_rows if r.get("scorable") and not r.get("prompt_collision")]
        screen_hits = sum(1 for r in screen_scorable if r.get("hit"))

        if screened_out:
            score = 0.0
            score_majority = 0.0
            score_core = 0.0
            tier = "unsupported"
            by_pos = {}
            by_band = {}
            script_match_rate = 0.0
            echoed_rate = 0.0
        else:
            hits = [r for r in scorable if r.get("hit")]
            score = rate([r.get("hit", False) for r in scorable])
            score_majority = rate([r.get("hit_majority", False) for r in scorable])

            core_scorable = [r for r in scorable if r["word"] in core_words]
            score_core = rate([r.get("hit", False) for r in core_scorable])
            tier = tier_of(score_core, screened_out)

            # Breakdowns
            by_p: dict[str, list[bool]] = defaultdict(list)
            by_b: dict[str, list[bool]] = defaultdict(list)
            for r in scorable:
                by_p[r.get("pos", "noun")].append(r.get("hit", False))
                by_b[r.get("band", "mid")].append(r.get("hit", False))

            by_pos = {p: rate(flags) for p, flags in sorted(by_p.items())}
            by_band = {b: rate(flags) for b, flags in sorted(by_b.items())}
            script_match_rate = rate([r.get("script_match", True) for r in scorable])
            echoed_rate = rate([r.get("echoed_english", False) for r in scorable])

        first = rs[0]
        lang_res = {
            "iso639_3": iso,
            "name": meta.get("name", first.get("name", iso)),
            "family": meta.get("family", first.get("family", "Unknown")),
            "region": meta.get("region", first.get("region", "Unknown")),
            "version_id": meta.get("version_id", first.get("version_id")),
            "version_title": meta.get("version_title", first.get("version_title")),
            "corpus_verses": meta.get("n_verses", first.get("corpus_verses")),
            "corpus_latin_share": first.get("corpus_latin_share"),
            "screen": f"{screen_hits}/{len(screen_scorable)}" if screen_scorable else "-",
            "screened_out": screened_out,
            "v1_screen_passed": meta.get("v1_screen_passed", False),
            "n_scorable": len(scorable),
            "n_hits": sum(1 for r in scorable if r.get("hit")),
            "score": score,
            "score_core": score_core,
            "score_majority": score_majority,
            "tier": tier,
            "by_pos": by_pos,
            "by_band": by_band,
            "script_match_rate": script_match_rate,
            "echoed_rate": echoed_rate,
        }
        languages.append(lang_res)

        # Sample examples (balanced between hits and misses)
        hits = [r for r in scorable if r.get("hit")]
        misses = [r for r in scorable if not r.get("hit")]
        ex_hits = hits[:EXAMPLE_TARGET // 2]
        ex_misses = misses[:EXAMPLE_TARGET - len(ex_hits)]
        chosen = ex_hits + ex_misses
        examples[iso] = [
            {
                "word": r["word"],
                "pos": r.get("pos"),
                "band": r.get("band"),
                "term": r.get("term"),
                "hit": r.get("hit", False),
                "n_aligned": r.get("n_aligned", 0),
                "n_hits": r.get("n_hits", 0),
                "example_verse": r.get("example_verse"),
                "script_match": r.get("script_match", True),
                "echoed_english": r.get("echoed_english", False),
            }
            for r in chosen
        ]

    # Sort leaderboard by standardized core score descending, then general score
    languages.sort(key=lambda x: (-x["score_core"], -x["score"], x["name"]))

    # Rollups
    def rollup(key: str) -> dict:
        g: dict[str, list] = defaultdict(list)
        for l in languages:
            if not l["screened_out"]:
                g[l[key]].append(l)
        return {
            k: {
                "n": len(v),
                "mean_core": mean([x["score_core"] for x in v]),
                "mean_score": mean([x["score"] for x in v]),
                "max_core": round(max((x["score_core"] for x in v), default=0.0), 2),
                "min_core": round(min((x["score_core"] for x in v), default=0.0), 2),
            }
            for k, v in sorted(g.items(), key=lambda kv: -mean([x["score_core"] for x in kv[1]]))
        }

    # Pos & Band pools across all supported languages
    pos_pool: dict[str, list[float]] = defaultdict(list)
    band_pool: dict[str, list[float]] = defaultdict(list)
    for l in languages:
        if not l["screened_out"]:
            for p, v in l["by_pos"].items():
                pos_pool[p].append(v)
            for b, v in l["by_band"].items():
                band_pool[b].append(v)

    by_pos = {
        p: {
            "mean": mean(vals),
            "median": round(sorted(vals)[len(vals) // 2], 2) if vals else 0.0,
            "n_languages": len(vals),
        }
        for p, vals in sorted(pos_pool.items())
    }

    by_band = {
        b: {
            "mean": mean(vals),
            "median": round(sorted(vals)[len(vals) // 2], 2) if vals else 0.0,
            "n_languages": len(vals),
        }
        for b, vals in sorted(band_pool.items())
    }

    tier_counts = defaultdict(int)
    for l in languages:
        tier_counts[l["tier"]] += 1

    summary = {
        "meta": {
            "model": MODEL,
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "n_languages_manifest": len(langs_meta),
            "n_languages_evaluated": len(languages),
            "n_languages_supported": sum(1 for l in languages if not l["screened_out"]),
            "n_languages_screened_out": sum(1 for l in languages if l["screened_out"]),
            "n_words": len(wordlist["words"]),
            "n_core_words": len(core_words),
            "core_threshold": wordlist["meta"].get("core_threshold", 0.90),
            "tier_thresholds": {"strong": STRONG, "medium": MEDIUM},
            "total_rows": total_raw_rows,
        },
        "tier_counts": dict(tier_counts),
        "by_pos": by_pos,
        "by_band": by_band,
        "by_region": rollup("region"),
        "by_family": rollup("family"),
        "languages": languages,
    }

    write_json(SUMMARY, summary)
    write_json(EXAMPLES, examples)

    print(f"wrote {SUMMARY}")
    print(f"wrote {EXAMPLES}")
    print("\n--- Summary Highlights ---")
    print(f"Languages evaluated: {len(languages)} (Supported: {summary['meta']['n_languages_supported']}, Screened out: {summary['meta']['n_languages_screened_out']})")
    print(f"Tiers: {dict(tier_counts)}")
    print(f"By POS (mean %): { {k: v['mean'] for k, v in by_pos.items()} }")
    print(f"By Band (mean %): { {k: v['mean'] for k, v in by_band.items()} }")
    print("\nTop 10 Languages (by Core Score):")
    for l in languages[:10]:
        print(f"  {l['name']:25s} ({l['iso639_3']}) core={l['score_core']:5.1f}% raw={l['score']:5.1f}% "
              f"maj={l['score_majority']:5.1f}% [{l['tier']:11s}]")


if __name__ == "__main__":
    main()
