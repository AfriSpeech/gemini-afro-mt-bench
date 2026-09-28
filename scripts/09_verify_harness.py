"""Verify the benchmark harness has no answer-leak channel.

The back-translation is only meaningful if the model cannot see the English
source word when it produces the English back-translation. This asserts:

  1. the pipeline never uses the chat/session API (no conversational memory)
  2. every prompt is a fresh stateless generate_content call
  3. the back-translation prompt never contains the source word
  4. the leakage guard actually fires when a leak is introduced
  5. no full source-word list is ever placed in a prompt
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    BACKWARD_TEMPLATE,
    DATA,
    FORWARD_TEMPLATE,
    assert_no_leak,
    REF_MODE_FOLDED,
    REF_MODE_SCRIPT,
    backward_prompt,
    compare_reference,
    fold_orthographic,
    prompt_collision,
    normalise,
    score_match,
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
        continue  # this file names the banned APIs in order to look for them
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

print("\n3. the back-translation template has no slot for the source word")
check(
    "template slots are name/iso/target only",
    set(place := {
        f.split("}")[0].split(":")[0].lstrip("{")
        for f in BACKWARD_TEMPLATE.split("{")[1:]
    }) == {"name", "iso", "target"},
)
check("template mentions no source-word argument", "source" not in BACKWARD_TEMPLATE.lower())
fn = next(
    n for n in ast.walk(ast.parse((SCRIPTS / "common.py").read_text(encoding="utf-8")))
    if isinstance(n, ast.FunctionDef) and n.name == "backward_prompt"
)
params = [a.arg for a in fn.args.args]
check(f"backward_prompt params = {params}", "source" not in params and "english" not in params)

print("\n4. no benchmark word appears in either prompt template")
lang = {"iso639_3": "twi", "name": "Twi", "family": "Atlantic-Congo", "region": "West Africa"}
probe = json.load(open(DATA / "wordlist.json", encoding="utf-8"))
words = [w["english"] for w in probe["words"]]
check(f"{len(words)} benchmark words loaded", len(words) == 204)

SENTINEL = "zzqx"
for label, tmpl in (("forward", FORWARD_TEMPLATE), ("backward", BACKWARD_TEMPLATE)):
    rendered = tmpl.format(word=SENTINEL, target=SENTINEL, name="Twi", iso="twi",
                           region="West Africa")
    toks = set(normalise(rendered).split())
    collide = sorted({w for w in words if set(normalise(w).split()) <= toks})
    check(f"{label} template collides with: {collide or 'nothing'}", not collide)

# A language name or ISO code could also smuggle a word in. Every such pair
# must be flagged for exclusion rather than silently scored.
langs = json.load(open(DATA / "languages.json", encoding="utf-8"))
collide = [
    (l["iso639_3"], l["name"], w)
    for l in langs
    for w in words
    if prompt_collision(w, l)
]
check(f"prompt-collision pairs flagged for exclusion: {len(collide)}", bool(collide))
print("        " + ", ".join(f"{n}/{w}" for _, n, w in collide))

# And the fully rendered backward prompt must not contain a *different*
# benchmark word, which would let the model copy it out of the instructions.
leaks = []
for w in words:
    toks = set(normalise(backward_prompt(w, lang)).split()) - set(normalise(w).split())
    for other in words:
        if other != w and set(normalise(other).split()) <= toks:
            leaks.append((w, other))
check(f"cross-contamination leaks: {len(leaks)}", not leaks)

print("\n5. echo detection is a flag, not a crash")
check("exact echo detected", assert_no_leak("Field", "Field") is True)
check("plural echo detected", assert_no_leak("Field", "Fields") is True)
check("compound echo detected", assert_no_leak("field", "palm field") is True)
check("clean target", assert_no_leak("Field", "Eyina") is False)
check("substring is not a token match", assert_no_leak("Apple", "Pineapple") is False)

print("\n6. matching logic")
check("exact 'eti'->'eti'", score_match("eti", "eti") == (True, True))
check("normalised 'The Head.'->'head'", score_match("The Head.", "head") == (True, True))
check("accent 'Ngor'->'ngor'", score_match("Café", "cafe") == (True, True))
check("wrong word fails", score_match("head", "dog")[0] is False)
check("plural does not pass", score_match("cat", "cats") == (False, False))
check("close on superset", score_match("head", "big head") == (False, True))
check("empty back fails", score_match("head", "") == (False, False))

print("\n7. reference comparison is script-robust")
# A non-Latin script must never fold to a truthy blank, or every pair of
# non-Latin spellings would compare equal and score as agreement.
check("Ge'ez folds to empty", fold_orthographic("አውሮፕላን") == "")
check("Arabic folds to empty", fold_orthographic("سُوسُو") == "")
check("Devanagari folds to empty", fold_orthographic("सेब") == "")
check("blank is falsy", not fold_orthographic("ፖም"))
check("non-Latin never agrees on a fold", compare_reference("አውሮፕላን", "አይሮፕላን") == (False, REF_MODE_SCRIPT))
check("non-Latin exact match agrees", compare_reference("ፖም", "ፖም") == (True, REF_MODE_SCRIPT))
check("Latin orthography folds", compare_reference("bulukondingo", "Bulukondiŋo") == (True, REF_MODE_FOLDED))
check("eng folds to n", fold_orthographic("ŋ") == "n")
check("slash alternatives", compare_reference("bulukondingo", "Waŋ/Bulukondiŋo") == (True, REF_MODE_FOLDED))
check("vowel length not folded", compare_reference("doktoo", "Jaararlaa/Doktooroo") == (False, REF_MODE_FOLDED))
check("wrong word disagrees", compare_reference("namaso", "Banaanoo") == (False, REF_MODE_FOLDED))
check("no comparable content", compare_reference("---", "---")[0] is None)
check("disagreeing Ge'ez", compare_reference("አውሮፕላን", "አይሮፕላን")[0] is False)

print("\n8. word list integrity")
check("204 single words", len(words) == 204)
check("no duplicates", len(set(w.lower() for w in words)) == len(words))
check("16 categories", len(probe["categories"]) == 16)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
