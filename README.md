# Gemini × African Languages — Corpus-Grounded Word MT Benchmark

How well does Google's **Gemini** translate vocabulary into African languages?

This benchmark evaluates Gemini across **645 African languages** by checking whether its translations appear in verified target-language parallel Bible texts from the YouVersion corpus ([AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus)).

An interactive dashboard lives in [`space/`](space/) and is deployed as a static Hugging Face Space.

---

## Benchmark Results

* **645 languages** evaluated · **204 words** (150 nouns, 33 concrete adjectives, 21 numerals) · **82,427 rows** scored · model **gemini-3.6-flash**.
* **398 languages supported** (scored > 0) · **247 languages screened out** (0 hits on the 5-numeral gate).
* **Tiers (Standardized 90% Common Core)**:
  * **Strong (≥60%)**: **4 languages** — Afrikaans (68.2%), Ewe (65.1%), Lingala (64.1%), Congo Swahili (61.5%).
  * **Medium (30–60%)**: **69 languages** — including Amharic (58.0%), Yoruba (56.9%), West Central Oromo (55.9%), Krio (55.4%), Chadian Arabic (54.9%), Plateau Malagasy (53.3%), Akan (52.8%), Chichewa (51.1%), Acoli (50.8%), Algerian Arabic (50.8%), Wolof (47.7%), Tigrinya (47.2%), Adangme (42.6%), Bassa (39.5%), Somali (37.9%), Dyula (36.9%), Bini (34.9%), Borana-Arsi-Guji Oromo (32.3%), Baoulé (31.3%).
  * **Weak (<30%)**: **325 languages**.
  * **Unsupported**: **247 languages** (failed the early-exit numeral gate).

### Performance Breakdowns

* **By Part of Speech (POS)**:
  * **Numerals**: **23.9%** mean match rate. Closed set, least ambiguous referents.
  * **Nouns**: **17.7%** mean match rate across everyday concrete objects, nature, people, and society.
  * **Adjectives**: **16.7%** mean match rate across physical, perceptible properties.
* **By Frequency Band**:
  * **Frequent**: **21.5%** mean score.
  * **Mid**: **17.5%** mean score.
  * **Rare**: **15.4%** mean score.
* **Common Core (195 words)**: 195 of the 204 words are alignable in ≥90% of languages with a Bible, providing an unbiased basis for cross-lingual comparison without Bible length bias.

Full results are in [`results/summary.json`](results/summary.json), [`results/examples.json`](results/examples.json), and [`results/details.jsonl.gz`](results/details.jsonl.gz).

---

## Methodology

### 1. Ground Truth Target Bibles
We index **950 YouVersion African language Bible versions** aligned at verse level to the English pivot text (31,082 verses). To maximize coverage, the pipeline selects the single best version per language based on aligned verses and word coverage.

### 2. Lexicon & Part-of-Speech Tiers
The vocabulary consists of **204 concepts** across three distinct linguistic tiers, each requiring at least 20 distinct English verse occurrences:
* **Nouns (150 words)**: Extracted from the English Bible with spaCy (`en_core_web_sm`), intersected with real-world Ghanaian usage from [GhanaNLP/GhanaNouns](https://github.com/GhanaNLP/GhanaNouns), and divided into 3 frequency bands (50 frequent, 50 mid, 50 rare).
* **Adjectives (33 words)**: Curated allow-list of physical, concrete properties (`big`, `clean`, `old`, `strong`, `hungry`, etc.), banded into 3 tiers based on English Bible frequency.
* **Numerals (21 words)**: Cardinal numbers 1–20, tens through 90, hundred, thousand, banded by Bible frequency.

### 3. Five-Numeral Early-Exit Gate
Before running the full 204-word evaluation, each language is probed on 5 frequent numerals. If Gemini scores 0 hits across all scorable numerals, the language is dropped early. This saves ~200 API calls per unsupported language and prevents hallucinated noise from entering the evaluation.

### 4. Verse-Parallel Verification
A translation only passes if the term appears in the **exact aligned target verses** where the source English word occurs:
* **Latin script**: Whole-word phrase matching with space-delimited boundaries to prevent substring false-positives (e.g., `ane` cannot match inside `wanene`).
* **Non-Latin scripts**: Unicode substring search (`fold_search`) that preserves Ge'ez, Arabic, Tifinagh, and Cyrillic scripts while normalizing combining accents and whitespace.
* **Boundary Isolation**: Verse boundaries are sentinel-isolated (`␟`) so multi-word terms cannot cross verse boundaries.

---

## Pipeline

```bash
cd scripts
python3 01_build_languages.py     # afriso → data/languages.json (2,035 languages)
python3 02_extract_lexicon.py     # English Bible arrow + spaCy → data/bible_lexicon.json
python3 03_build_bands.py         # GhanaNouns + Bible frequency → data/wordlist.json
python3 04_build_corpus_index.py  # Select best Bible per language → data/corpus_manifest.json
python3 09_verify_harness.py      # Assert harness and matcher integrity (run before bench)
python3 05_bench_corpus.py        # Sharded Gemini evaluation → results/bench.partial.<shard>.jsonl
python3 06_aggregate.py           # Rollups → results/summary.json, examples.json, details.jsonl.gz
```

### Running with Concurrency and Shards
`05_bench_corpus.py` supports parallel sharding and is fully resumable:
```bash
# Example: running shard 0 of 4
export GEMINI_API_KEY="your-api-key"
export BENCH_CONCURRENCY=32
python3 scripts/05_bench_corpus.py --shard 0 --nshards 4
```

---

## Dashboard

An interactive dashboard (Plotly, static HTML/JS) lives in [`space/`](space/) and is deployed as a static Hugging Face Space.

```bash
cd space && python3 -m http.server 8000   # open http://localhost:8000
```

| Tab | Answers |
|---|---|
| **Overview** | Match rates by POS (nouns, adjectives, numbers), frequency bands, regions, and tier counts |
| **Leaderboard** | 645 languages with region/family/tier filters, search, core score, and raw hit rates |
| **POS & Bands** | Heatmap of top 80 languages across parts of speech |
| **Language Detail** | In-depth breakdown per language, Bible version metadata, and aligned verse lookups |

---

## Data Files

| Path | Contents |
|---|---|
| `data/languages.json` | 2,035 African languages from afriso |
| `data/corpus_manifest.json` | 645 languages with matched YouVersion Bibles and coverage statistics |
| `data/wordlist.json` | 204 benchmark words (150 nouns, 33 adjectives, 21 numerals) with verse alignments and core flags |
| `data/bible_lexicon.json` | Full English Bible lexicon extracted with spaCy |
| `data/ghana-nouns.csv` | GhanaNouns frequency inventory for noun frequency bands |
| `results/summary.json` | Per-language scores, POS/band breakdowns, rollups |
| `results/examples.json` | Sample aligned verse lookups per language |
| `results/details.jsonl.gz` | All 82,427 scored rows |

---

## Sources

* [AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus) — Parallel YouVersion African Bible texts
* [GhanaNLP/GhanaNouns](https://github.com/GhanaNLP/GhanaNouns) — Ghanaian noun frequency inventory
* [AfriSpeech/afriso](https://github.com/AfriSpeech/afriso) — African language names and ISO 639-3 codes
* Google Gemini via the `google-genai` SDK
