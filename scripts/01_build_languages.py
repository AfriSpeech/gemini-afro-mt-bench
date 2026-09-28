"""Build the African language list from afriso.

Uses only the afriso *main* name plus the ISO 639-3 code (never `alt_names`),
because aliases for a single language are frequently shared with unrelated
languages (e.g. "Guang" is an alt-name of eight separate Ghanaian languages) and
would make the prompt ambiguous.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

from common import DATA, write_json

AFRISO = "https://raw.githubusercontent.com/AfriSpeech/afriso/main/src/afriso/data"
LANG_CSV = DATA / "afriso_languages.csv"
COUNTRIES_CSV = DATA / "afriso_countries.csv"
OUT = DATA / "languages.json"


def fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        print(f"downloading {url}", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def main() -> None:
    fetch(f"{AFRISO}/languages.csv", LANG_CSV)
    fetch(f"{AFRISO}/countries.csv", COUNTRIES_CSV)

    regions: dict[str, str] = {}
    for r in csv.DictReader(open(COUNTRIES_CSV, encoding="utf-8")):
        regions[r["code2"]] = r["region"]

    langs = []
    for r in csv.DictReader(open(LANG_CSV, encoding="utf-8")):
        countries = [c.strip() for c in (r.get("countries") or "").split(";") if c.strip()]
        afr = [c for c in countries if c in regions]
        if not afr:
            continue
        if r["type"] != "living":
            continue
        if "sign language" in r["name"].lower():
            continue  # not a spoken MT target
        name = r["name"].strip()
        if "(" in name and name.endswith(")"):
            continue  # disambiguated duplicates like "Buli (Ghana)"
        langs.append(
            {
                "iso639_3": r["iso639_3"].strip(),
                "name": name,
                "family": (r.get("family") or "Unknown").strip() or "Unknown",
                "countries": afr,
                "region": regions[afr[0]],
                "scope": (r.get("scope") or "").strip(),
            }
        )

    langs.sort(key=lambda x: x["name"])
    write_json(OUT, langs)
    print(f"{len(langs)} African living spoken languages", flush=True)
    from collections import Counter

    for k, v in Counter(x["region"] for x in langs).most_common():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
