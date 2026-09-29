---
title: Gemini Afro MT Benchmark
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: cc-by-4.0
short_description: Corpus-grounded MT benchmark for Gemini, African languages
---

# Gemini × African Languages Corpus-Grounded MT Benchmark

How well does Google's **Gemini** translate vocabulary into African languages?

This benchmark evaluates Gemini on **645 African languages** by checking whether its translations appear in verified target-language parallel Bible texts from the YouVersion corpus ([AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus)) across **877 language editions**.

* **GitHub Repository** — [https://github.com/AfriSpeech/gemini-word-mt-bench](https://github.com/AfriSpeech/gemini-word-mt-bench)
* **Ground Truth** — 877 African language Bible versions aligned at verse level to the English pivot (31,082 verses). Translations are verified across all parallel verse occurrences where the concept appears.
* **Vocabulary** — 204 concepts across 3 Part-of-Speech tiers:
  * **Nouns** (150 words): Extracted from the English Bible with spaCy, intersected with GhanaNouns, split into 3 frequency bands.
  * **Adjectives** (33 words): Curated concrete physical properties (`big`, `clean`, `old`, `strong`), banded by English Bible frequency.
  * **Numerals** (21 words): Cardinal numbers 1–20, tens through 90, hundred, thousand, banded by Bible frequency.
* **Common Core** — 195 concepts are alignable in ≥90% of languages with a Bible, providing an unbiased basis for cross-lingual comparison.

The dashboard reads its data from the GitHub repository at runtime. Full per-word rows are published as `details.jsonl.gz`.
