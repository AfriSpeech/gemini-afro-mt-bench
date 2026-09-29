# Gemini × African Languages Corpus-Grounded MT Benchmark

How well does Google's **Gemini** translate vocabulary into African languages?

This benchmark evaluates Gemini across **645 African languages** by checking whether its translations appear in verified target-language parallel Bible texts from the YouVersion corpus ([AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus)).

An interactive dashboard lives in [`space/`](space/) and is deployed as a static Hugging Face Space at [AfriSpeech/gemini-afro-mt-bench](https://huggingface.co/spaces/AfriSpeech/gemini-afro-mt-bench).

---

## Benchmark Results

* **645 languages** evaluated across **877 parallel Bible versions** · **204 words** (150 nouns, 33 concrete adjectives, 21 numerals) · **131,580 rows** scored · model **gemini-3.6-flash**.
* **643 languages supported** (scored > 0) · **only 2 languages scored 0%** across the entire parallel corpus.
* **Tiers (Standardized 90% Common Core)**:
  * **Strong (≥60%)**: **23 languages** — Amharic (87.7%), Afrikaans (86.2%), Hausa (73.8%), Twi (73.8%), Ewe (73.3%), Plateau Malagasy (71.3%), Kinyarwanda (69.7%), Lingala (69.2%), Igbo (68.7%), Yoruba (68.2%), Ganda/Luganda (67.2%), Congo Swahili (64.6%), Chichewa (64.1%), West Central Oromo (63.6%), Krio (63.1%), Wolof (61.0%), Chadian Arabic (60.0%), and others.
  * **Medium (30–60%)**: **90 languages** — including Somali (55.9%), Sudanese Arabic (53.8%), Zulu (53.3%), Morisyen (51.3%), Algerian Arabic (50.8%), Fanti (49.7%), Pedi/Northern Sotho (49.2%), Tigrinya (47.2%), Dan (44.1%), Swati (42.1%), Dagbani (36.9%), Bini (34.9%), Ngbaka (34.3%), and others.
  * **Weak (<30%)**: **530 languages**.
  * **Zero / Unsupported**: **2 languages**.

### Performance Breakdowns

* **By Part of Speech (POS)**:
  * **Numerals**: **21.5%** mean match rate.
  * **Adjectives**: **17.3%** mean match rate across concrete, physical properties.
  * **Nouns**: **17.1%** mean match rate across everyday concrete concepts.
* **By Frequency Band**:
  * **Frequent**: **21.1%** mean score.
  * **Mid**: **16.9%** mean score.
  * **Rare**: **14.6%** mean score.
* **Common Core (195 words)**: 195 of the 204 words are alignable in ≥90% of languages with a Bible, providing an unbiased basis for cross-lingual comparison without Bible length bias.

Full results are in [`results/summary.json`](results/summary.json), [`results/examples.json`](results/examples.json), and [`results/details.jsonl.gz`](results/details.jsonl.gz).

---

## Methodology

### 1. Multi-Version Parallel Bible Corpus
We index **877 YouVersion African language Bible versions** aligned at verse level to the English pivot text (31,082 verses). For languages with multiple translations (e.g. Swahili has 13, Afrikaans 9, Shona 8, Oromo 8, Twi 5, Amharic 5), all versions are pooled simultaneously to capture orthographic and dialectal variation.

### 2. Lexicon & Part-of-Speech Tiers
The vocabulary consists of **204 concepts** across three distinct linguistic tiers, each requiring at least 20 distinct English verse occurrences:
* **Nouns (150 words)**: Extracted from the English Bible with spaCy (`en_core_web_sm`), intersected with real-world Ghanaian usage from [GhanaNLP/GhanaNouns](https://github.com/GhanaNLP/GhanaNouns), and divided into 3 frequency bands (50 frequent, 50 mid, 50 rare).
* **Adjectives (33 words)**: Curated allow-list of physical, concrete properties (`big`, `clean`, `old`, `strong`, `hungry`, etc.), banded into 3 tiers based on English Bible frequency.
* **Numerals (21 words)**: Cardinal numbers 1–20, tens through 90, hundred, thousand, banded by Bible frequency.

### 3. Comprehensive Verse-Parallel Verification
Every word is evaluated across **all parallel verse occurrences** where the English word appears in the Bible (no artificial verse cap):
* **Multi-Version Search**: If Gemini's translation appears in any of the available versions for that language in that verse, it is credited.
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
