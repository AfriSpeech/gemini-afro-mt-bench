"""Shared config, Gemini client and scoring helpers for the word-MT benchmark."""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDCCdgeH9KAje_GiT_PDzg-lk5exJl7sEM")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SEED = 20260928
WORDS_PER_BAND = 50
N_BANDS = 5
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "64"))
SCREEN_CONCURRENCY = int(os.environ.get("SCREEN_CONCURRENCY", "96"))

# 3 very common, near-universal concrete nouns used to screen language support.
SCREEN_WORDS = ["head", "water", "mother"]

TRANSLATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"translation": types.Schema(type=types.Type.STRING)},
    required=["translation"],
)

_STOP = {"the", "a", "an", "of"}


def client() -> genai.Client:
    return genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------- normalisation
# Character-level orthographic folding. West African languages are written with
# IPA-ish letters that a single regular orthography is free to spell as plain
# Latin: eta/eng, ny/nn, sh/s, gb/b, o/open-o, e/epsilon. Gemini and the BU 200
# Word Project routinely pick *different conventions for the same word*
# (Bulukondiŋo vs bulukondingo), so folding is required before the two can be
# compared. This is letter-level only - it never merges distinct words, and it
# does not affect the pass metric, which compares English to English.
_ORTHOGRAPHIC_FOLD = {
    "ŋ": "n",    # eng
    "ɲ": "ny",   # n with left hook
    "ʃ": "sh",   # esh
    "ʒ": "zh",   # ezh
    "ɣ": "gh",   # gamma
    "ɡ": "g",    # script g
    "ɔ": "o",    # open o
    "ɞ": "o",    # phi
    "ɛ": "e",    # epsilon
    "ɩ": "i",    # barred i
    "ɓ": "b",    # b with hook
    "ɗ": "d",    # d with hook
    "ɖ": "d",    # retroflex d
    "ʈ": "t",    # retroflex t
    "ɳ": "n",    # retroflex n
    "Ɔ": "o",    # capital open o
    "Ɛ": "e",    # capital epsilon
    "Ʋ": "v",    # capital v with hook
    "Ʒ": "z",    # capital z with stroke
    "ʔ": "",     # glottal stop
    "'": "'",    # modifier letter apostrophe -> plain apostrophe
    "’": "'",    # right single quote -> plain apostrophe
    "ʼ": "'",    # modifier letter apostrophe
    "·": "",     # middle dot
    "=": "",     # equals sign used as tone marker
}

# Geminates that are pure orthographic convention rather than sound, collapsed
# only when comparing two spellings of the same language.
_GEMINATE_FOLD = (
    ("ngh", "n"), ("ng", "n"), ("ny", "n"), ("ŋ", "n"),
    ("sh", "s"), ("ch", "c"), ("gb", "b"), ("kp", "p"),
    ("gy", "y"),
)


def fold_orthographic(text: str) -> str:
    """Fold IPA-ish letters and orthographic geminates to plain Latin.

    Returns "" for anything with no Latin content, which is how non-Latin
    scripts (Ge'ez, N'Ko, Tifinagh, Cyrillic, ...) are detected: they are not
    comparable by letter folding. Whitespace is collapsed so that a script
    stripped to nothing yields "" rather than a run of spaces -- a string of
    spaces is truthy, and comparing one against another would score every pair
    as agreeing.
    """
    t = unicodedata.normalize("NFC", str(text)).lower()
    for a, b in _ORTHOGRAPHIC_FOLD.items():
        t = t.replace(a, b)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    for a, b in _GEMINATE_FOLD:
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


def fold_script(text: str) -> str:
    """Script-agnostic normalisation: strip punctuation/whitespace, casefold.

    Used to compare two spellings written in the same non-Latin script, where
    letter-level folding does not apply but exact identity still means
    something.
    """
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"[\s\W_]+", " ", t, flags=re.UNICODE)
    return " ".join(t.split())


# How a spelling comparison was made: orthographic folding for Latin-script
# languages, exact script-aware match otherwise, or not comparable at all.
REF_MODE_FOLDED = "folded"
REF_MODE_SCRIPT = "exact_script"
REF_MODE_NONE = "incomparable"


def compare_reference(target: str, reference: str) -> tuple[bool | None, str]:
    """Compare Gemini's output against a human BU 200 translation.

    The reference may list several accepted spellings separated by "/". Returns
    (matched_or_None, mode). Comparison is done in two passes so that no
    language is silently dropped and none is silently inflated: Latin scripts
    are folded (Bulukondiŋo == bulukondingo), other scripts are compared
    exactly after normalisation, and only text with no comparable content at all
    is reported as incomparable.
    """
    candidates = str(reference).split("/")
    got = fold_orthographic(target)
    if got:
        folded = {fold_orthographic(c) for c in candidates}
        folded.discard("")
        if folded:
            return got in folded, REF_MODE_FOLDED
    got_s = fold_script(target)
    if got_s:
        exact = {fold_script(c) for c in candidates}
        exact.discard("")
        if exact:
            return got_s in exact, REF_MODE_SCRIPT
    return None, REF_MODE_NONE


def normalise(text: str) -> str:
    """Lowercase, de-accent, drop punctuation/articles, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP]
    return " ".join(toks)


def score_match(source: str, back: str) -> tuple[bool, bool]:
    """Return (exact_pass, close_pass) for a back-translation vs the source word.

    exact: normalised back-translation equals normalised source.
    close: every content token of the source survives in the back-translation
           (tolerates an extra qualifier, but not a different head noun).
    """
    ns, nb = normalise(source), normalise(back)
    if not ns or not nb:
        return False, False
    if ns == nb:
        return True, True
    st, bt = set(ns.split()), set(nb.split())
    close = st.issubset(bt) or bt.issubset(st)
    return False, close


# ------------------------------------------------------------------- prompting
def lang_label(row: dict) -> str:
    """Prompt-facing language label: afriso main name + ISO 639-3 code.

    Deliberately uses only the main reference name (never `alt_names`) so the
    model is disambiguated by code rather than by competing aliases.
    """
    return f"{row['name']} (ISO 639-3: {row['iso639_3']})"


def prompt_collision(source_word: str, row: dict) -> bool:
    """Would the language label hand the model this source word?

    The ISO 639-3 code is required in the prompt to disambiguate the language,
    but a few codes are also English words (bus, box, hoe, man, two) and a few
    language names contain one (Mango, Cross River Mbembe). For those pairs the
    word is visible in the back-translation instructions, so the round-trip
    result is not evidence of translation. Flagged here and dropped from scoring.
    """
    label = set(normalise(lang_label(row)).split())
    return set(normalise(source_word).split()) <= label


FORWARD_TEMPLATE = (
    "Translate the English noun \"{word}\" into {name} (ISO 639-3: {iso}), "
    "a language spoken in {region}.\n"
    "Rules:\n"
    "- Output the everyday, native term for this noun in {name}.\n"
    "- Keep the native orthography and diacritics. Do not romanise.\n"
    "- If the noun has a distinct everyday usage, prefer that usage.\n"
    "- Output the single term only, with no gloss, no explanation, no quotes."
)


def forward_prompt(word: str, row: dict) -> str:
    return FORWARD_TEMPLATE.format(
        word=word,
        name=row["name"],
        iso=row["iso639_3"],
        region=row.get("region") or "Africa",
    )


def backward_prompt(translation: str, row: dict) -> str:
    return BACKWARD_TEMPLATE.format(
        name=row["name"], iso=row["iso639_3"], target=translation
    )


BACKWARD_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "is_translation": types.Schema(
            type=types.Type.BOOLEAN,
            description="false if the input was not actually in the target language",
        ),
        "back_translation": types.Schema(type=types.Type.STRING),
    },
    required=["is_translation", "back_translation"],
)


class AnswerLeak(AssertionError):
    """Raised if a source word can reach the back-translation prompt."""


# The back-translation prompt is assembled from this template, which has no slot
# for the English source word at all. Every forward/backward pair is a fresh,
# stateless `generate_content` call - the pipeline never uses the chat/session
# API - so the model has no conversational memory of the forward pass. The
# template is the only place a leak could originate, so it is checked here once.
#
# Wording is constrained: no word from the 204-word benchmark list may appear in
# this template (or in the forward template), or the model would be handed the
# answer for that item. scripts/09_verify_harness.py asserts this.
BACKWARD_TEMPLATE = (
    "Translate the following {name} (ISO 639-3: {iso}) term into English.\n"
    "Term: {target}\n"
    "Rules:\n"
    "- First check whether the term above is genuinely in {name} rather than "
    "already English. If it is already English, or you cannot read it at all, "
    'set is_translation to false and leave back_translation empty.\n'
    "- Otherwise set is_translation to true and give the plain English noun it "
    "denotes (lowercase, singular, no article, no explanation).\n"
    "- Output the single English term only."
)


def assert_no_leak(source_word: str, prompt: str) -> bool:
    """Diagnostic: did the model's own output smuggle the source word back?

    This is *not* a harness bug - a target like "Fields" for "Field" is Gemini
    echoing English, which the round-trip test scores as a failure. Callers
    record the flag and carry on rather than aborting.

    Stem-aware so a plural echo counts, token-aware so "pineapple" is not
    reported as containing "apple".
    """
    def stem(t: str) -> str:
        return t[:-1] if len(t) > 3 and t.endswith("s") else t

    sw = {stem(t) for t in normalise(source_word).split()}
    if not sw:
        return False
    return any(stem(t) in sw for t in normalise(prompt).split())




def _config(schema, thinking_budget: int = 0) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )


_RETRYABLE = (genai_errors.APIError, asyncio.TimeoutError, ConnectionError)


async def call_json(aio, prompt: str, schema, attempts: int = 5) -> dict | None:
    """Single structured call with exponential backoff. Returns parsed dict or None."""
    for i in range(attempts):
        try:
            r = await aio.models.generate_content(
                model=MODEL, contents=[prompt], config=_config(schema)
            )
            parsed = r.parsed
            if isinstance(parsed, dict):
                return parsed
            if r.text:
                try:
                    return json.loads(r.text)
                except json.JSONDecodeError:
                    pass
        except asyncio.CancelledError:
            raise
        except _RETRYABLE:
            pass
        except Exception as e:  # noqa: BLE001 - surface non-retryable as failure
            if not _is_retryable(e):
                return None
        if i < attempts - 1:
            await asyncio.sleep(min(2**i * 0.75 + random.random(), 20))
    return None


def _is_retryable(e: Exception) -> bool:
    s = str(e).lower()
    return any(
        k in s
        for k in ("429", "500", "502", "503", "504", "timeout", "overloaded",
                  "rate limit", "unavailable", "resource_exhausted", "deadline")
    )


def read_csv(path: Path) -> list[dict]:
    import csv

    csv.field_size_limit(10**9)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size/1024:.0f} KB)", file=sys.stderr)
