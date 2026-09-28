"""Scrape the BU 200 Word Project and extract the English word side.

Each page of https://www.bu.edu/200word/ lists its words as an <h4> of the form

    <target word> (<English gloss>)

e.g. "Aso (Ear)". The English side is the project-wide 200-word core list,
identical across the 9 participating languages, so we scrape all 9 and keep the
words that several languages agree on.

The 9 human-curated target translations are retained as a reference set, which
the benchmark can score Gemini against.

Categories come from the site navigation, whose link titles are
"<target category> (<English category>)" - far more reliable than guessing from
the per-language URL slug.

Writes data/bu200.json
"""
from __future__ import annotations

import html
import re
import urllib.request
from collections import Counter, defaultdict

from common import DATA, write_json

BASE = "https://www.bu.edu/200word"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}
OUT = DATA / "bu200.json"
CACHE = DATA / "bu200_cache"

LANGS = {
    "isixhosa": "isiXhosa",
    "isizulu": "isiZulu",
    "amharic": "Amharic",
    "wolof": "Wolof",
    "hausa": "Hausa",
    "kiswahili": "kiSwahili",
    "igbo": "Igbo",
    "akan-twi": "Akan Twi",
    "mandinka": "Mandinka",
}

# <h4> text -> (target, english). The gloss is the LAST parenthesised group, so
# entries like "Nnumre (Akan) (Numbers)" resolve to target "Nnumre (Akan)".
H4 = re.compile(r"<h4[^>]*>(.*?)</h4>", re.S | re.I)
PAIR = re.compile(r"^(?P<tgt>.*?)\s*\(\s*(?P<eng>[^()]{1,60}?)\s*\)\s*$", re.S)
NAV = re.compile(
    r'<a class="level_2" href="https://www\.bu\.edu/200word/'
    r'(?P<lang>[a-z-]+)/(?P<slug>[a-z0-9%\-]+)/"[^>]*title="Navigate to: ([^"]*)"',
    re.I,
)
NAV_CAT = re.compile(r"\(([^()]{2,40})\)\s*$")

# Category names must never be treated as vocabulary.
CATEGORY_WORDS = {
    "transportation", "body parts", "people", "professions", "animals",
    "natural world", "food & drink", "food and drink", "cooking", "clothing",
    "places", "house", "farm", "school", "days of the week", "days",
    "numbers", "colors", "tools", "family", "jobs",
}
CANON = {
    "food & drink": "food and drink", "food and drink": "food and drink",
    "days of the week": "days of the week", "days": "days of the week",
    "jobs": "professions", "colors": "colors",
}

# Singular head form for every attested irregular/singular-plural pair on the
# site. Built by hand from the extracted glosses; the benchmark needs one
# canonical noun per concept, and a round-trip test on "Finger" and "Fingers"
# measures the same thing twice.
PLURAL_TO_SINGULAR = {
    "clouds": "cloud", "eggs": "egg", "fingers": "finger", "lips": "lip",
    "mountains": "mountain", "nails": "nail", "plates": "plate", "shoes": "shoe",
    "stars": "star", "thousands": "thousand", "toes": "toe", "parents": "parent",
    "siblings": "sibling", "grapes": "grape", "oxen": "ox",
}

# Glosses that are category titles or transliterated section headings, not
# vocabulary. Each is attested in exactly one language and is a target-language
# string, not English.
ARTIFACTS = {"bala kuŋolu", "taransipooroo", "tuloo"}


def norm_cat(s: str) -> str:
    s = html.unescape(s).strip().lower()
    return CANON.get(s, s)


def canonical_key(gloss: str) -> str:
    """Collapse case, plural and slash variants of a gloss to one key."""
    g = gloss.strip().lower()
    if g in ARTIFACTS:
        return ""
    if "/" in g:
        g = g.split("/")[0].strip()  # "Box/Chest" -> "box"
    g = PLURAL_TO_SINGULAR.get(g, g)
    return g


def display_form(gloss: str, key: str) -> str:
    """Prefer the site's own Capitalised spelling of the canonical key."""
    return key.capitalize()



def get(url: str, key: str) -> str:
    """Cached GET. `key` must be language-qualified - slugs collide across
    languages ("bodyparts" exists in seven of them)."""
    cached = CACHE / f"{key}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        txt = r.read().decode("utf-8", errors="replace")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(txt, encoding="utf-8")
    return txt


def nav_categories(home: str) -> dict[str, str]:
    """slug -> canonical English category, from the site navigation."""
    out: dict[str, str] = {}
    for m in NAV.finditer(home):
        title = html.unescape(m.group(3))
        c = NAV_CAT.search(title)
        if c:
            out.setdefault(m.group("slug"), norm_cat(c.group(1)))
    return out


def category_slugs(home: str, lang: str) -> list[str]:
    seen, out = set(), []
    for m in NAV.finditer(home):
        if m.group("lang").lower() == lang.lower() and m.group("slug") not in seen:
            seen.add(m.group("slug"))
            out.append(m.group("slug"))
    return out


def parse_page(text: str) -> list[dict]:
    body = text.split("<article", 1)[-1]
    rows = []
    for raw in H4.findall(body):
        t = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        m = PAIR.match(t)
        if not m:
            continue
        tgt, eng = m.group("tgt").strip(), m.group("eng").strip()
        if not eng or not re.search(r"[A-Za-z]", eng):
            continue
        if norm_cat(eng) in CATEGORY_WORDS:
            continue  # category title, not a vocabulary item
        if tgt.lower() == eng.lower():
            continue  # untranslated entry
        rows.append({"target": tgt, "english": eng})
    return rows


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    home = get(f"{BASE}/", "site_index")
    cat_of_slug = nav_categories(home)

    per_lang: dict[str, dict[str, str]] = {}
    cat_of_english: dict[str, str] = {}
    all_english: set[str] = set()

    for slug, label in LANGS.items():
        slugs = category_slugs(home, slug)
        got: dict[str, str] = {}
        for s in slugs:
            cat = cat_of_slug.get(s, "unknown")
            for row in parse_page(get(f"{BASE}/{slug}/{s}/", f"{slug}__{s}")):
                all_english.add(row["english"])
                if row["english"] not in got:
                    got[row["english"]] = row["target"]
                cat_of_english.setdefault(row["english"], cat)
        per_lang[label] = got
        print(
            f"{label:12s} {len(slugs):2d} pages -> {len(got):3d} english words",
            flush=True,
        )

    # Reference translations keyed by the canonical English concept.
    refs: dict[str, dict[str, str]] = defaultdict(dict)
    for label, got in per_lang.items():
        for eng, tgt in got.items():
            k = canonical_key(eng)
            if k:
                refs[k][label] = tgt

    # Re-attribute categories onto canonical keys, by majority vote.
    key_cats: dict[str, Counter] = defaultdict(Counter)
    for eng, cat in cat_of_english.items():
        k = canonical_key(eng)
        if k:
            key_cats[k][cat] += 1

    words = []
    for k in sorted(refs):
        words.append(
            {
                "english": display_form(k, k),
                "english_lower": k,
                "category": key_cats[k].most_common(1)[0][0] if key_cats[k] else "unknown",
                "reference_translations": refs[k],
                "n_reference_languages": len(refs[k]),
                "site_spellings": sorted({g for g in all_english if canonical_key(g) == k}),
            }
        )

    strong = [w for w in words if w["n_reference_languages"] >= 3]
    core = [w for w in words if w["n_reference_languages"] >= 2]
    single = [w for w in core if " " not in w["english"]]
    out = {
        "source": BASE,
        "description": (
            "BU 200 Word Project - specialized, picturable words in 16 "
            "categories, developed by the BU African Language Program and the "
            "Geddes Language Center. English side extracted from all 9 "
            "published language editions and cross-checked across them."
        ),
        "languages": list(LANGS.values()),
        "n_words_all": len(words),
        "n_words_ge2": len(core),
        "n_words_ge3": len(strong),
        "n_single_word": len(single),
        "categories": sorted({w["category"] for w in words if w["category"] != "unknown"}),
        "words": words,
    }
    write_json(OUT, out)

    print(f"\nunique English concepts (canonicalised): {len(words)}")
    print(f"attested in >=2 languages: {len(core)}")
    print(f"attested in >=3 languages: {len(strong)}")
    print(f"single-word (benchmark-ready): {len(single)}")
    print(f"all 9 languages: {sum(1 for w in words if w['n_reference_languages'] == 9)}")
    print("\nmerged variants (site spellings collapsed):")
    for w in words:
        if len(w["site_spellings"]) > 1:
            print(f"  {w['english']:18s} <- {w['site_spellings']}")
    print("\nby category:")
    for c, n in Counter(w["category"] for w in words).most_common():
        print(f"  {c:20s} {n}")
    print("\nsample (english | twi | swahili | amharic | xhosa):")
    for w in core[:14]:
        r = w["reference_translations"]
        print(
            f"  {w['english']:14s} {r.get('Akan Twi','-'):16s} "
            f"{r.get('kiSwahili','-'):16s} {r.get('Amharic','-'):14s} "
            f"{r.get('isiXhosa','-')}"
        )


if __name__ == "__main__":
    main()
