"""Map each afriso language to its best YouVersion Bible file in
AfriSpeech/africa-corpus, so the v2 benchmark can check Gemini's translation
against real target-language text.

Only languages that actually have a Bible can be scored by the corpus-grounded
metric -- there is nothing to check a translation against otherwise -- so this
manifest is also the v2 language set. Where several versions exist for one
language the most complete one wins, because a translation present in a partial
New Testament may simply be missing from a truncated Old Testament.

Output: data/corpus_manifest.json
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, write_json  # noqa: E402

SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"
)
SNAPSHOT_GLOB_ALT = os.path.expanduser(
    "/mnt/volume_d2wey28/hf_cache/hub/datasets--AfriSpeech--africa-corpus/snapshots/*"
)
FILENAME_RE = re.compile(r"^(?P<name>.+)_(?P<code>[a-z]{3})_v(?P<vid>\d+)\.csv$")

# A word joins the "core" set when this fraction of languages has a version
# containing at least one of its verses. The core set is what makes the
# leaderboard a like-for-like comparison: outside it, each language is judged
# on a different handful of words, so a high score could just mean the words it
# happened to be given were easy.
CORE_THRESHOLD = 0.90


def english_keys() -> set[str]:
    """Verse keys available on the English pivot side."""
    from importlib import import_module

    mod = import_module("02_extract_lexicon")
    keys, _ = mod.load_english()
    return set(keys)


def main() -> None:
    snaps = glob.glob(SNAPSHOT_GLOB) or glob.glob(SNAPSHOT_GLOB_ALT)
    if not snaps:
        raise SystemExit("africa-corpus snapshot not found in the HF cache")
    snap = snaps[0]
    english = english_keys()
    print(f"{len(english)} English verse keys", file=sys.stderr)

    # verse_key -> indices of the benchmark words occurring in it, so that a
    # version's coverage of the word list is a single pass over its own keys
    # rather than a scan of every word's verse list.
    words = json.loads((DATA / "wordlist.json").read_text(encoding="utf-8"))["words"]
    by_verse: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        for k in w["verses"]:
            by_verse.setdefault(k, []).append(i)

    versions_by_iso: dict[str, list[dict]] = defaultdict(list)
    keys_by_iso: dict[str, set[str]] = defaultdict(set)
    coverage: dict[str, set[int]] = defaultdict(set)

    for fname in os.listdir(snap):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        iso = m.group("code")
        path = os.path.join(snap, fname)
        with open(path, newline="", encoding="utf-8") as f:
            n = n_align = 0
            for row in csv.DictReader(f):
                key = row.get("verse_key") or ""
                n += 1
                if key in english:
                    n_align += 1
                    keys_by_iso[iso].add(key)
                    idx = by_verse.get(key)
                    if idx:
                        coverage[iso].update(idx)
        versions_by_iso[iso].append({
            "file": fname,
            "version_id": int(m.group("vid")),
            "version_title": os.path.splitext(fname)[0].rsplit("_v", 1)[0],
            "n_verses": n,
            "n_aligned": n_align,
        })

    languages = {l["iso639_3"]: l for l in
                 json.loads((DATA / "languages.json").read_text(encoding="utf-8"))}

    rows = []
    for iso, v_list in versions_by_iso.items():
        if iso not in languages:  # not in afriso's African-country set
            continue
        v_list.sort(key=lambda x: (x["n_aligned"], x["n_verses"]), reverse=True)
        primary = v_list[0]
        row = dict(languages[iso])
        row["version_id"] = primary["version_id"]
        row["version_title"] = primary["version_title"]
        row["file"] = primary["file"]
        row["files"] = [v["file"] for v in v_list]
        row["n_versions"] = len(v_list)
        row["n_verses"] = sum(v["n_verses"] for v in v_list)
        row["n_aligned"] = len(keys_by_iso[iso])
        row["n_scorable_words"] = len(coverage[iso])
        rows.append(row)
    rows.sort(key=lambda r: (r["name"], r["iso639_3"]))

    # A language must have at least one aligned verse for a word before the
    # benchmark can test it, so that is the count the core threshold is applied
    # to. Bands are kept in proportion so the core is not all-easy or all-rare.
    n = len(rows) or 1
    alignable = [0] * len(words)
    for r in rows:
        iso = r["iso639_3"]
        for i in coverage[iso]:
            alignable[i] += 1
    core = {words[i]["word"] for i in range(len(words))
            if alignable[i] >= CORE_THRESHOLD * n}
    bands = {w["word"]: w["band"] for w in words}
    wordlist = json.loads((DATA / "wordlist.json").read_text(encoding="utf-8"))
    for i, w in enumerate(wordlist["words"]):
        w["core"] = w["word"] in core
        w["n_langs_alignable"] = alignable[i]
    wordlist["meta"]["core_threshold"] = CORE_THRESHOLD
    wordlist["meta"]["n_core"] = len(core)
    write_json(DATA / "wordlist.json", wordlist)

    sc = [r["n_scorable_words"] for r in rows]

    screening = RESULTS / "screening.json"
    if screening.exists():
        scr = json.loads(screening.read_text(encoding="utf-8"))
        passed = {r["iso639_3"]: bool(r.get("passed_screening"))
                  for r in scr.get("results", scr)}
        # Recorded, not filtered on. v2 scores by corpus presence only, so
        # pre-filtering on a v1 round-trip score would smuggle the old metric
        # into the new one; the flag is kept for cross-checking the two runs.
        for r in rows:
            r["v1_screen_passed"] = passed.get(r["iso639_3"], False)

    out = {
        "meta": {"snapshot": os.path.basename(snap), "n_languages": len(rows),
                 "n_candidates": len(versions_by_iso), "core_threshold": CORE_THRESHOLD,
                 "n_core_words": len(core)},
        "languages": rows,
    }
    write_json(DATA / "corpus_manifest.json", out)

    aligned = sorted(r["n_aligned"] for r in rows)
    print(f"verses/language: min {aligned[0]} median {aligned[len(aligned)//2]} "
          f"max {aligned[-1]}", file=sys.stderr)
    print(f"scorable words/language: min {min(sc)} median {sorted(sc)[len(sc)//2]} "
          f"max {max(sc)}", file=sys.stderr)
    print(f"core ({CORE_THRESHOLD:.0%} of languages): {len(core)} of {len(words)} words, "
          f"{dict(collections.Counter(bands[w] for w in core))}", file=sys.stderr)


if __name__ == "__main__":
    main()
