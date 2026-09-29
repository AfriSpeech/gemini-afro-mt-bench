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

# No default key: GEMINI_API_KEY must be exported. Committing a key would put a
# live credential in the repo, and a revoked one in a fallback silently turns
# every call into an empty translation rather than an error.
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SEED = 20260928
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "64"))
SCREEN_CONCURRENCY = int(os.environ.get("SCREEN_CONCURRENCY", "96"))

# ------------------------------------------------- v2 lexicon / band settings
# A word needs this many distinct English verses before it can be benchmarked:
# after intersecting with a version that covers only part of the Bible, there
# has to be an aligned verse left to test the target term against.
MIN_VERSES = 20
# Sampled words per (part-of-speech, frequency band) cell. The numeral pool is
# only ~25 strong, so its cells are whatever the data allows -- see
# 11_build_bands.py, which reports the shortfall instead of hiding it.
PER_BAND = 50
N_BANDS = 3
# Part-of-speech tiers, each with its own frequency bands. "band_source" names
# the axis the bands are cut on: Ghana frequency is only defined for words in
# the GhanaNouns inventory, which is a noun list.
TIERS = {
    "noun": {"band_source": "ghana_count"},
    "adjective": {"band_source": "bible_count"},
    "number": {"band_source": "bible_count"},
}

# Language screen. A language is dropped from the full evaluation only if
# Gemini scores zero on this many numerals, drawn from the most frequent ones in
# the English text. Numerals make the best probe available: the set is closed,
# the referents are unambiguous, and it is disjoint from the noun and adjective
# metrics the leaderboard is built on, so screening on it cannot favour the
# languages that are good at the thing being measured.
SCREEN_TIER = "number"
SCREEN_N = 5
# A language needs at least this many *scorable* screen words before the gate is
# allowed to drop it. Below that the zero would be explained by a thin Bible
# rather than by a model that cannot translate, and dropping it would quietly
# reintroduce the coverage confound the per-language denominator exists to avoid.
SCREEN_MIN_SCORABLE = 3

TRANSLATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"translation": types.Schema(type=types.Type.STRING)},
    required=["translation"],
)

_STOP = {"the", "a", "an", "of"}


def client() -> genai.Client:
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set")
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


# ------------------------------------------------------- v2 corpus-grounded match
# The v2 benchmark does not ask the model to translate anything back. It asks
# whether the term Gemini produced is the term real speakers of the target
# language actually use, by searching that language's own Bible text. That needs
# a normalisation that keeps non-Latin scripts intact -- fold_orthographic
# deliberately erases them -- plus a way to search a whole corpus cheaply.

# Character-level fold only: IPA-ish letters, no geminates. Geminates are pure
# orthographic convention (ng/n, sh/s) and collapsing them is what makes
# "Bulukondiŋo" and "bulukondingo" comparable, but it also maps distinct words
# onto each other ("wash" -> "was"), so the loosest pass is only ever reported
# alongside the exact-script pass, never instead of it.
_FOLD_CHARS = str.maketrans({ord(k): v for k, v in _ORTHOGRAPHIC_FOLD.items()})
_ASCII_RE = re.compile(r"[a-z0-9 ]*")


def fold_search(text: str) -> str:
    """Casefold, de-accent and fold IPA-ish letters, keeping other scripts.

    Unlike fold_orthographic this preserves Ge'ez, N'Ko, Tifinagh, Cyrillic and
    Arabic, so a folded term can be searched for inside text written in any
    script. Geminates are left alone -- see _FOLD_CHARS.
    """
    t = unicodedata.normalize("NFC", str(text)).lower().translate(_FOLD_CHARS)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\s\W_]+", " ", t, flags=re.UNICODE).strip()


def is_latinish(folded: str) -> bool:
    """True when a folded string is plain Latin, i.e. word-delimited search is safe."""
    return bool(folded) and _ASCII_RE.fullmatch(folded) is not None


def _substring_present(needle: str, hay: str) -> bool:
    """Substring search for scripts that are not space-delimited."""
    if not needle:
        return False
    if needle in hay:
        return True
    # Some orthographies write a noun and its enclitic without a space; compare
    # with all whitespace removed as well.
    tight = needle.replace(" ", "")
    return len(tight) >= 2 and tight in hay.replace(" ", "")


def _phrase_present(needle: str, hay: str) -> bool:
    """Whole-word phrase search, for space-delimited (Latin) text."""
    padded = f" {hay} "
    return f" {needle} " in padded


# Verse separator inside the pre-joined corpus blobs. It has to be a character
# that fold_search and fold_orthographic cannot emit, so that a whole-word probe
# can be counted by searching for SEP + term + SEP.
SEP = "␟"


class CorpusIndex:
    """A target-language Bible, indexed for "is this term in here?" lookups.

    Two views of the same text, mirroring compare_reference's folded-then-exact
    policy:

      * `folded`  - fold_orthographic, for Latin-script text
      * `script`  - fold_search, used for non-Latin scripts and as the loose
                    fallback for Latin text

    A term matches when either view contains it. Latin text is searched by whole
    words, so "ane" cannot match inside "wanene"; other scripts are searched by
    substring, because a large part of Africa is written without word spacing.
    """

    __slots__ = ("script", "folded", "_blob_script", "_blob_script_squeezed",
                 "_blob_folded", "n_verses", "latin_share")

    def __init__(self, verses: dict[str, str]) -> None:
        self.n_verses = len(verses)
        self.script = {k: fold_search(v) for k, v in verses.items()}
        self.folded: dict[str, str] = {}
        for k, v in verses.items():
            f = fold_orthographic(v)  # empty means non-Latin: not foldable
            if f:
                self.folded[k] = f
        # Share of verses written in Latin script. A minority-language Bible in
        # Ge'ez or N'Ko will be near 0, a Latin-orthography one near 1. Used to
        # tell "wrong script" apart from "wrong word" without screening the
        # language out, since that is a real finding about the model rather than
        # a reason to drop the row.
        self.latin_share = len(self.folded) / max(self.n_verses, 1)
        # Space-padded verses separated by SEP: ensures whole-word matching
        # works anywhere in a verse while stopping matches across verse boundaries.
        self._blob_folded = SEP.join(f" {v} " for v in self.folded.values())
        self._blob_script = SEP.join(f" {v} " for v in self.script.values())
        self._blob_script_squeezed = SEP.join(
            v.replace(" ", "") for v in self.script.values())

    # ------------------------------------------------------------------ matching
    def term(self, term: str) -> "TermNeedle":
        return TermNeedle(term)

    def count(self, needle: "TermNeedle") -> tuple[int, int, str]:
        """Corpus-wide occurrences of `needle` -> (folded_count, script_count, mode).

        The counting rule has to agree with in_verse() exactly. When it did not,
        a term could be reported absent from every verse it was judged on while
        the corpus-wide count claimed hundreds of occurrences: for Adhola,
        "three" -> "dek" returned 913 hits because the script view was counted
        by raw substring and so matched inside "adek" and "ndek".
        """
        f = s = 0
        if needle.latin:
            # Space-delimited: whole-word only, mirroring in_verse().
            if needle.folded:
                f = self._blob_folded.count(f" {needle.folded} ")
            if needle.script:
                s = self._blob_script.count(f" {needle.script} ")
                tight = needle.script.replace(" ", "")
                if not s and len(tight) >= 2:
                    s = self._blob_script_squeezed.count(tight)
        else:
            # Not space-delimited: substring, mirroring _substring_present().
            if needle.folded:
                f = self._blob_folded.count(needle.folded)
            if needle.script:
                s = self._blob_script.count(needle.script)
                tight = needle.script.replace(" ", "")
                if not s and len(tight) >= 2:
                    s = self._blob_script_squeezed.count(tight)
        if f and s:
            return f, s, REF_MODE_FOLDED
        if f:
            return f, 0, REF_MODE_FOLDED
        if s:
            return 0, s, REF_MODE_SCRIPT
        return 0, 0, REF_MODE_NONE

    def in_verse(self, needle: "TermNeedle", key: str) -> bool:
        """Does verse `key` contain `needle`, under either view?"""
        if needle.latin:
            # Space-delimited: whole-word only, so "ane" cannot match "wanene".
            text = self.folded.get(key)
            if text is not None and _phrase_present(needle.folded, text):
                return True
            text = self.script.get(key)
            if text is not None and _phrase_present(needle.script, text):
                return True
            return False
        for view, probe in ((self.folded, needle.folded), (self.script, needle.script)):
            text = view.get(key)
            if text and _substring_present(probe, text):
                return True
        return False


class TermNeedle:
    """A search term reduced to the two views CorpusIndex looks for."""

    __slots__ = ("raw", "folded", "script", "latin")

    def __init__(self, term: str) -> None:
        self.raw = str(term).strip()
        self.script = fold_search(self.raw)
        folded = fold_orthographic(self.raw)
        self.folded = folded or self.script
        # Whether the term is written in Latin script at all. A term in a
        # script the corpus never uses can never be found, which is a different
        # failure from giving the wrong word in the right script.
        self.latin = bool(folded) and is_latinish(folded)

    @property
    def usable(self) -> bool:
        return bool(self.folded or self.script)


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
    word is visible in the prompt, so the result is dropped from scoring.
    """
    label = set(normalise(lang_label(row)).split())
    return set(normalise(source_word).split()) <= label


FORWARD_TEMPLATE = (
    "Translate the English word \"{word}\" into {name} (ISO 639-3: {iso}), "
    "a language spoken in {region}.\n"
    "Rules:\n"
    "- Output the everyday, native term for this word in {name}.\n"
    "- Keep the native orthography and diacritics. Do not romanise.\n"
    "- If the word has a distinct everyday usage, prefer that usage.\n"
    "- Output the single term only, with no gloss, no explanation, no quotes."
)


def forward_prompt(word: str, row: dict) -> str:
    return FORWARD_TEMPLATE.format(
        word=word,
        name=row["name"],
        iso=row["iso639_3"],
        region=row.get("region") or "Africa",
    )


def assert_no_leak(source_word: str, prompt: str) -> bool:
    """Diagnostic: did the model's own output smuggle the source word back?

    A target like "Fields" for "Field" is Gemini echoing English.
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
