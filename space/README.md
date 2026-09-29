---
title: Gemini African Word-MT Benchmark
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: cc-by-4.0
short_description: Corpus-grounded word MT benchmark for Gemini across 645 African languages
---

# Gemini × African Languages — Word-Level MT Benchmark

How well does Google's **Gemini** translate words into African languages?

This benchmark evaluates Gemini on **645 African languages** by checking whether its translations appear in verified target-language Bible text from the YouVersion parallel corpus ([AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus)).

* **Ground Truth** — 950 African language Bible versions aligned at verse level to the English pivot (31,082 verses). A word passes only if Gemini's translation appears in the **exact aligned target verses** where the source English word occurs.
* **Vocabulary** — 204 concepts across 3 Part-of-Speech tiers:
  * **Nouns** (150 words): Extracted from the English Bible with spaCy, intersected with GhanaNouns, split into 3 frequency bands based on real usage.
  * **Adjectives** (33 words): Curated concrete physical properties (e.g. `big`, `clean`, `old`, `strong`), banded by English Bible frequency.
  * **Numerals** (21 words): Cardinal numbers 1–20, tens through 90, hundred, thousand, banded by Bible frequency.
* **Early-Exit Gate** — Each language is first probed on 5 frequent numerals. If Gemini scores 0 across all scorable numerals, it is dropped early to avoid hallucinated noise and burning quota.
* **Common Core** — 195 words are alignable in ≥90% of languages with a Bible, providing an unbiased basis for cross-lingual comparison without Bible length bias.

The app reads its data from the GitHub repository at runtime, so updates appear immediately after a push. Full per-word rows are published as `details.jsonl.gz`.
