---
title: Gemini African Word-MT Benchmark
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: cc-by-4.0
short_description: Round-trip word MT benchmark for Gemini, African languages
---

# Gemini × African Languages — Word-Level MT Benchmark

How well does Gemini translate **nouns** into African languages?

Each English word is translated **English → target language → back to English**.
A word passes only if the back-translation returns the original word.

* **Vocabulary** — the [BU 200 Word Project](https://www.bu.edu/200word/) English
  side, 204 single words across 16 categories, extracted from all 9 published
  language editions and cross-checked across them.
* **Languages** — 2,035 living spoken languages of African countries, from
  [afriso](https://github.com/AfriSpeech/afriso). Only the afriso *main* name and
  ISO 639-3 code are shown to the model, never the aliases, because aliases are
  often shared between unrelated languages.
* **Screening** — every language is first round-tripped through 3 very common
  words. Languages scoring 0/3 are dropped as not being real translation targets.

The app reads its data from the GitHub repository at runtime, so a new run
appears here after a push with no redeploy. Raw per-word rows are published as
`details.jsonl.gz`.
