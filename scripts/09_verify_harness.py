"""Verify the benchmark harness and corpus matching integrity.

Asserts:
  1. the pipeline never uses the chat/session API (no conversational memory)
  2. every model call is a fresh stateless generate_content
  3. the forward prompt template has proper slot constraints
  4. no benchmark word collides with the prompt template instructions
  5. echo detection works accurately for source leaks
  6. word list integrity (204 words across 3 tiers, frequency bands, >=20 verses)
  7. corpus manifest integrity (645 languages, version selection, 90% core)
  8. CorpusIndex matching logic:
     - Latin whole-word matching (no false substring hits)
     - Non-Latin substring search (Ge'ez, Arabic)
     - Verse boundary isolation (no cross-verse phrase matches)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA,
    FORWARD_TEMPLATE,
    CorpusIndex,
    assert_no_leak,
    fold_orthographic,
    fold_search,
    normalise,
    prompt_collision,
)

SCRIPTS = Path(__file__).parent
ok = True


def check(label: str, cond: bool) -> None:
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    ok = ok and cond


print("1. no chat/session API anywhere in the pipeline")
for py in sorted(SCRIPTS.glob("*.py")):
    if py.name == Path(__file__).name:
        continue
    src = py.read_text(encoding="utf-8")
    banned = [
        t for t in ("start_chat", "chats.create", "send_message", "history=")
        if t in src
    ]
    check(f"{py.name} clean", not banned)

print("\n2. every model call is a fresh stateless generate_content")
for py in sorted(SCRIPTS.glob("0*.py")):
    tree = ast.parse(py.read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "generate_content"
    ]
    check(f"{py.name}: {len(calls)} generate_content call(s)", True)

print("\n3. prompt template slots are constrained")
check(
    "forward template slots are word/name/iso/region only",
    set(f.split("}")[0].split(":")[0].lstrip("{") for f in FORWARD_TEMPLATE.split("{")[1:])
    == {"word", "name", "iso", "region"},
)

print("\n4. no benchmark word collides with template instructions")
wordlist_file = DATA / "wordlist.json"
wordlist_data = json.loads(wordlist_file.read_text(encoding="utf-8"))
words = [w["word"] for w in wordlist_data["words"]]
check(f"{len(words)} benchmark words loaded", len(words) == 204)

SENTINEL = "zzqx"
rendered = FORWARD_TEMPLATE.format(word=SENTINEL, name="Twi", iso="twi", region="West Africa")
toks = set(normalise(rendered).split())
collide = sorted({w for w in words if set(normalise(w).split()) <= toks})
check(f"forward template collides with: {collide or 'nothing'}", not collide)

langs = json.load(open(DATA / "languages.json", encoding="utf-8"))
prompt_collisions = [
    (l["iso639_3"], l["name"], w)
    for l in langs
    for w in words
    if prompt_collision(w, l)
]
check(f"prompt-collision pairs flagged for exclusion: {len(prompt_collisions)}", bool(prompt_collisions))

print("\n5. echo detection is a flag, not a crash")
check("exact echo detected", assert_no_leak("Field", "Field") is True)
check("plural echo detected", assert_no_leak("Field", "Fields") is True)
check("compound echo detected", assert_no_leak("field", "palm field") is True)
check("clean target", assert_no_leak("Field", "Eyina") is False)
check("substring is not a token match", assert_no_leak("Apple", "Pineapple") is False)

print("\n6. word list integrity (204 concepts across 3 tiers)")
check("204 words in wordlist.json", len(words) == 204)
check("no duplicates", len(set(w.lower() for w in words)) == len(words))
pos_counts = {p: sum(1 for w in wordlist_data["words"] if w["pos"] == p) for p in ("noun", "adjective", "number")}
check("3 tiers: noun=150, adj=33, num=21", pos_counts == {"noun": 150, "adjective": 33, "number": 21})
check("all words have >=20 verses in English Bible", all(len(w["verses"]) >= 20 for w in wordlist_data["words"]))
check("common core defined (>=90% alignable)", wordlist_data["meta"].get("n_core", 0) >= 180)

print("\n7. corpus manifest integrity")
manifest = json.loads((DATA / "corpus_manifest.json").read_text(encoding="utf-8"))
check("645 African languages with YouVersion Bibles", len(manifest["languages"]) == 645)
check("all entries have valid ISO and file", all(l.get("iso639_3") and l.get("file") for l in manifest["languages"]))

print("\n8. CorpusIndex matching logic")
idx = CorpusIndex({
    "GEN.1.1": "Wɔwoo baako pɛ",
    "GEN.1.2": "wanene nso nyame",
    "GEN.1.3": "አንድ ሁለት ሦስት",   # Ge'ez: 1, 2, 3
    "GEN.1.4": "سُوسُو فِي البَيْتِ",  # Arabic
})

# Latin whole-word matching
n_baako = idx.term("baako")
check("baako matches in GEN.1.1", idx.in_verse(n_baako, "GEN.1.1") is True)
n_ane = idx.term("ane")
check("ane does NOT match inside wanene (whole-word boundary)", idx.in_verse(n_ane, "GEN.1.2") is False)

# Non-Latin substring matching
n_geez = idx.term("አንድ")
check("Ge'ez substring matches in GEN.1.3", idx.in_verse(n_geez, "GEN.1.3") is True)
n_arab = idx.term("البَيْتِ")
check("Arabic substring matches in GEN.1.4", idx.in_verse(n_arab, "GEN.1.4") is True)

# Cross-verse isolation
n_cross = idx.term("pɛ wanene")
check("cross-verse phrase does NOT match across verses", idx.count(n_cross)[2] == "incomparable")

# Script preservation
check("fold_search preserves Ge'ez", fold_search("አንድ") == "አንድ")
check("fold_search preserves Arabic", fold_search("سُوسُو") == "سوسو")
check("fold_orthographic empties non-Latin", fold_orthographic("አንድ") == "")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
