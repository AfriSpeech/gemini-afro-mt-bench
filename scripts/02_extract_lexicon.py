"""Extract the English-side lexicon from the Bible parallel corpus: the nouns
and the concrete adjectives that appear in it, with the verses they occur in.

The v1 benchmark scored words from the BU 200 Word Project, an isolated
concrete-noun list with no sentence context. v2 instead takes its vocabulary
from the English pivot of AfriSpeech/africa-corpus (the Common English Bible),
so every word carries the verse it was seen in and can later be checked against
the aligned verse on the target-language side.

Adjectives are restricted to a hand-checked list of concrete, translatable
qualities (scripts/CONCRETE_ADJECTIVES). spaCy tags 1,779 adjective lemmas in
this text, but most are not lexical adjectives at all: quantifiers ("other",
"many", "same"), comparatives ("greater", "older"), and domain jargon
("ephphatha"). Those measure something other than a target language's
vocabulary, and a comparative in particular has no single-word equivalent in
most African languages, so scoring them would mostly measure the metric's
blindness.

Output: data/bible_lexicon.json
    {lemma: {pos, count, verses, n_verses, proper, forms}}
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MIN_VERSES, write_json  # noqa: E402

ARROW = (
    Path.home()
    / ".cache/huggingface/datasets/AfriSpeech___africa-corpus/english/0.0.0"
    / "ded7566ef3e2b9579bac18d042acb5cd42deac5b/africa-corpus-train.arrow"
)

# Concrete, translatable qualities. Deliberately excludes comparatives and
# superlatives (no single-word equivalent in most target languages), quantifiers
# and pronouns that spaCy tags as adjectives ("other", "own", "same", "many",
# "such"), and abstract religious or moral terms that have no single-word
# equivalent either ("eternal", "divine", "spiritual"). Everything kept here
# names a property a speaker could check by looking at something.
CONCRETE_ADJECTIVES = {
    # size, age, quantity-as-adjective
    "big", "small", "large", "great", "little", "long", "short", "tall",
    "young", "old", "new", "high", "low", "deep", "thick", "thin", "wide",
    "narrow", "fat", "heavy",
    # physical condition
    "strong", "weak", "hard", "soft", "smooth", "rough", "sharp", "dry",
    "wet", "hot", "cold", "warm", "cool", "clean", "dirty", "empty", "full",
    "blind", "deaf", "lame", "sick", "healthy", "dead", "broken", "whole",
    "sound", "blind", "naked", "bare", "blind",
    # colour
    "white", "black", "red", "green", "yellow", "blue", "grey", "purple",
    # taste, sound
    "sweet", "bitter", "sour", "salty", "loud", "quiet", "sharp",
    # directly perceivable states
    "hungry", "thirsty", "tired", "sleepy", "afraid", "angry", "glad",
    "sad", "brave", "calm", "patient", "ashamed", "proud", "grateful",
    "sorry", "lonely", "free", "busy", "ready", "sure", "visible", "blind",
    # qualities with a concrete referent in ordinary usage
    "good", "bad", "evil", "holy", "righteous", "wicked", "faithful",
    "unfaithful", "honest", "kind", "cruel", "gentle", "humble", "wise",
    "foolish", "true", "false", "precious", "harmless", "perfect",
    "unclean", "clean", "noble", "gentle",
}

# Cardinals, kept in: numerals are among the most concrete things a language
# can name, and they are a known weak spot in low-resource MT, so they earn
# their place in the word list. spaCy usually tags them NUM or PRON rather than
# NOUN, so they are captured from the surface form as well. Ordinals
# ("first", "third") are left out -- in this text they behave as adjectives or
# adverbs far more often than as counts, and "first" is also a temporal
# adverbial, which would make the test ambiguous.
CONCRETE_NUMERALS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
}

# Lemmas that are artefacts of parsing archaic or translated English rather
# than vocabulary: CEB's measure words, and bare ordinals.
_STOP_LEMMAS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
}


def load_english() -> tuple[list[str], list[str]]:
    with pa.memory_map(str(ARROW), "rb") as src:
        try:
            reader = ipc.open_stream(src)
        except pa.ArrowInvalid:
            src.seek(0)
            reader = ipc.open_file(src)
        table = reader.read_all()
    return table.column("verse_key").to_pylist(), table.column("eng").to_pylist()


def main() -> None:
    keys, texts = load_english()
    print(f"{len(texts)} English verses", file=sys.stderr)

    nlp = spacy.load("en_core_web_sm")
    count: collections.Counter = collections.Counter()
    pos_of: dict[str, str] = {}
    verses: dict[str, list[str]] = collections.defaultdict(list)
    proper: dict[str, bool] = collections.defaultdict(bool)
    forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for key, doc in zip(keys, nlp.pipe(texts, batch_size=256, n_process=4)):
        for tok in doc:
            if not tok.is_alpha:
                continue
            surface = tok.text.lower()
            lemma = tok.lemma_.lower().strip()
            if not re.fullmatch(r"[a-z]+", lemma):
                continue
            if surface in CONCRETE_NUMERALS:
                pos = "number"
            elif tok.pos_ == "ADJ":
                if lemma not in CONCRETE_ADJECTIVES:
                    continue
                pos = "adjective"
            elif tok.pos_ == "NOUN":
                pos = "noun"
            elif tok.pos_ == "PROPN":
                pos = "proper"
            else:
                continue
            if pos != "number" and (len(lemma) < 3 or lemma in _STOP_LEMMAS):
                continue
            # A lemma tagged more than one way keeps the noun reading: the noun
            # is the more clearly translatable unit, and several of the
            # adjectives above ("bitter", "free") are common nouns too.
            if lemma in pos_of and pos_of[lemma] != pos and "adjective" in (pos_of[lemma], pos):
                if pos_of[lemma] == "noun":
                    continue
            pos_of[lemma] = pos
            count[lemma] += 1
            forms[lemma][surface] += 1
            if pos == "proper":
                proper[lemma] = True
            verses[lemma].append(key)

    # spaCy leaves "God's" lemmatised to "gods"; merge it into "god".
    for bogus in ("gods", "lords", "mens", "womens", "childrens"):
        if bogus in count and bogus[:-1] in count:
            count[bogus[:-1]] += count[bogus]
            for f, c in forms[bogus].items():
                forms[bogus[:-1]][f] += c
            verses[bogus[:-1]] = sorted(set(verses[bogus[:-1]]) | set(verses[bogus]))
            del count[bogus], verses[bogus], forms[bogus]

    # A word can occur more than once in a verse; the verse list is a set of
    # places the word appears, and a duplicate would inflate both the distinct
    # verse count used for eligibility and the aligned-verse sample.
    for lemma in verses:
        verses[lemma] = sorted(set(verses[lemma]))

    out = {
        lemma: {
            "pos": pos_of[lemma],
            "count": n,
            "verses": verses[lemma],
            "n_verses": len(verses[lemma]),
            "proper": proper[lemma],
            "forms": [f for f, _ in forms[lemma].most_common(6)],
        }
        for lemma, n in count.most_common()
    }
    write_json(DATA / "bible_lexicon.json", out)

    n_adj = sum(1 for v in out.values() if v["pos"] == "adjective")
    n_num = sum(1 for v in out.values() if v["pos"] == "number")
    n_proper = sum(1 for v in out.values() if v["proper"])
    for pos in ("noun", "adjective", "number"):
        ready = sorted(k for k, v in out.items()
                       if v["pos"] == pos and not v["proper"] and v["n_verses"] >= MIN_VERSES)
        print(f"{pos:10s} {len(ready):3d} with >={MIN_VERSES} verses: "
              f"{', '.join(ready) if len(ready) <= 60 else ', '.join(ready[:60]) + ' …'}",
              file=sys.stderr)
    print(f"{len(out)} lemmas total ({n_proper} ever-tagged proper, "
          f"{n_adj} concrete adjectives, {n_num} numerals)", file=sys.stderr)


if __name__ == "__main__":
    main()
