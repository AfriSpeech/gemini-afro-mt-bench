# Gemini × African Languages — Word-Level MT Benchmark

How well does Gemini translate **nouns** into African languages?

Every English word is round-tripped:

```
English  ──translate──▶  <target language>  ──translate back──▶  English
```

A word **passes** when the back-translation normalises back to the original word.
There is no conversational session: each direction is a separate, stateless
`generate_content` call, and the back-translation prompt has no slot for the
English source word — so the back-translator is never handed the answer.

An interactive dashboard lives in [`space/`](space/) and is deployed as a static
Gradio Space. It reads the published JSON from this repository at runtime, so
re-running the benchmark updates the dashboard after a push with no redeploy.

## Results

**1,622** languages tested (of 2,035 screened) · **204** words · **330,888**
scored rows · model **gemini-3.6-flash**.

* Mean pass rate across languages: **24.0%** (median 17.7%).
* Tiers: **63 strong** (≥60%), **413 medium** (30–60%), **1,146 weak** (<30%).
* Best performers are almost all high-resource: Arabic 88.7%, Standard Arabic
  87.3%, Egyptian Arabic 85.8%, Spanish 85.3%, Afrikaans 84.8%, Yiddish 82.8%,
  Amharic 81.9%.
* Against the 8 languages with an independent BU 200 human reference
  translation, agreement ranges from 75.8% (Amharic) down to 18.6%
  (Mandinka) — Gemini's own back-translation grading and an external
  reference broadly agree on relative ordering.
* Pass rate rises with word frequency (GhanaNouns quintile 1, most frequent:
  36.7%) and falls for rarer words (quintile 3: 23.5%), and is highest for
  concrete everyday categories (professions 41.4%, days of the week 39.6%)
  and lowest for colors (9.6%) and farm (11.9%) vocabulary.

Full breakdowns (by category, region, family, frequency band, and every
language) are in [`results/summary.json`](results/summary.json) and the
[dashboard](space/).

## Method

### Vocabulary — the BU 200 Word Project

`scripts/00_scrape_bu200.py` scrapes <https://www.bu.edu/200word/>, whose pages
list each word as `<target word> (<English gloss>)`. The English side is the
project-wide core list, so all 9 published editions (isiXhosa, isiZulu, Amharic,
Wolof, Hausa, kiSwahili, Igbo, Akan Twi, Mandinka) are scraped and cross-checked
against each other.

The scrape yields **204 single-word concepts** after collapsing case variants,
singular/plural pairs and slash-compounds (`Box/Chest` → `Box`), and dropping
section headings that are category titles rather than vocabulary. Each word
carries the 16-category label inherent to the dataset (body parts, food and
drink, animals, numbers, …), which is the primary grouping axis.

The 9 human-curated target translations are kept in `data/bu200.json` as a
reference set. For those languages the benchmark also reports **agreement with
the BU reference**, an independent check that does not rely on Gemini grading
itself.

### Languages — afriso

`scripts/01_build_languages.py` selects from
[afriso](https://github.com/AfriSpeech/afriso) the **2,035 living, non-sign
languages** spoken in the 58 African countries in its country table. Prompts use
only the afriso **main name and ISO 639-3 code**:

> Translate the English noun "banana" into Twi (ISO 639-3: twi), a language
> spoken in West Africa.

Aliases are deliberately withheld. In the afriso data, `Guang` is an alternative
name of eight separate Ghanaian languages and `Moba` is shared across several
more, so supplying aliases would make the prompt ambiguous rather than precise.

> **Caveat.** The filter is "spoken in an African country", not "African
> language". This admits Spanish, French, Afrikaans, Ladino, Yiddish and
> Portuguese creoles, which afriso lists for Morocco, Algeria and South Africa.
> They are included because they are genuinely in scope for the registry, and the
> family rollup on the dashboard keeps them visible rather than mixed in. Set
> `families` filtering on the dashboard to exclude Indo-European.

> **Coverage gap: Swahili.** afriso attributes countries to specific varieties,
> not to the macrolanguage row (`swa`, "Swahili (macrolanguage)"), which has zero
> countries listed and so never enters the benchmark's language set — even
> though standalone Swahili is one of the 9 BU 200 reference languages. Swahili
> is represented only through its country-attributed varieties, chiefly
> `swc` Congo Swahili (76.5%, strong tier). The same gap applies to any of
> afriso's other 17 macrolanguage rows (Akan, Arabic, Fulah, Oromo, …) that lack
> a country attribution of their own.

### Screening

`scripts/03_screen.py` round-trips 3 near-universal concrete nouns — **head**,
**water**, **mother** — through every language first. Languages that score
**0/3** are not real translation targets: the model is echoing English,
transliterating, or inventing text. Those are dropped before the expensive run.
1,622 of 2,035 languages survive.

### Scoring

A pass requires the **normalised exact** back-translation to equal the source
word. Normalisation lowercases, strips diacritics and punctuation, and drops
articles. So `Ŋmeɛ` → `nmee` and `The Head.` → `head` pass, while `cats` for
`cat` does not.

Each row also records:

| field | meaning |
|---|---|
| `close` | one side's tokens are a subset of the other's, so `big head` still credits `head` — reported, not scored |
| `echoed_english` | the forward "translation" was just the English source word |
| `readable` | the back-translator judged the input genuinely in the target language |
| `matches_reference` | Gemini's output equals the BU 200 human-curated translation |
| `prompt_collision` | **excluded from scoring** — see below |

### Prompt-collision exclusions

The ISO 639-3 code is required in the prompt to disambiguate the language, but a
few codes are also English words and a few language names contain one. For those
8 (language, word) pairs the model is handed the answer, so they are flagged and
dropped from the denominator rather than scored as a win:

`Bokobaru/Bus` · `Buamu/Box` · `Cross River Mbembe/River` · `Horom/Hoe` ·
`Mandingo/Man` · `Mango/Mango` · `Sekpele/Lip` · `Tswapong/Two`

`scripts/09_verify_harness.py` additionally asserts that no word from the
204-word list appears in either prompt template. It caught two real confounds
during development: the instruction *"give the plain English **head** noun"*
handed the model the answer for the word **head**, and *"return the value
UNPARSEABLE in the is_translation **field**"* did the same for **field**.

### Tiers

Pass rate is bucketed so the dashboard is readable at a glance:

| tier | rule |
|---|---|
| **strong** | ≥ 60% |
| **medium** | 30–60% |
| **weak** | < 30% |

## Pipeline

```bash
cd scripts
python3 00_scrape_bu200.py     # BU 200 Word Project → data/bu200.json
python3 01_build_languages.py  # afriso → data/languages.json  (2,035 languages)
python3 02_build_wordlist.py   # → data/wordlist.json  (204 words, 16 categories)
python3 09_verify_harness.py   # assert no answer-leak channel  ← run before benchmarking
python3 03_screen.py           # 3-word screen → results/screening.json
python3 04_bench.py            # full run    → results/benchmark.partial.jsonl
python3 05_aggregate.py        # → results/summary.json, examples.json, details.jsonl.gz
```

`04_bench.py` is resumable — it appends to `results/benchmark.partial.jsonl` and
skips `(language, word)` pairs already present, so an interrupted run continues
where it stopped. Tune `BENCH_CONCURRENCY` (default 64) to trade rate limits
against throughput.

## Dashboard

A static HTML/JS page (Plotly, no backend) — deployed as a static
[Hugging Face Space](space/) at [`space/index.html`](space/index.html). It
fetches the JSON straight from this repo's `raw.githubusercontent.com` URLs at
page load, so a new benchmark run appears with a page refresh and no redeploy.

```bash
cd space && python3 -m http.server 8000   # then open http://localhost:8000
```

| tab | what it answers |
|---|---|
| **Overview** | pass rate by BU category, by region, by tier, and against screening score |
| **Leaderboard** | every language with region/family/tier filters and search, sorted and tier-coloured |
| **Category heatmap** | top 120 languages × 16 categories at a glance |
| **Language detail** | one language's category profile plus its actual round-trips, including BU reference comparison |

Data source, overridable with `?repo=` / `?branch=` query params:

```
https://raw.githubusercontent.com/AfriSpeech/gemini-word-mt-bench/main/results/
```

## Data

| path | contents |
|---|---|
| `data/bu200.json` | scraped BU 200 Word Project, 281 canonical concepts with the 9 reference translations |
| `data/wordlist.json` | the 204 benchmark words with category and GhanaNouns frequency |
| `data/languages.json` | 2,035 African-country languages from afriso |
| `results/screening.json` | 3-word screen for all 2,035 languages |
| `results/summary.json` | per-language scores, category/band breakdowns, rollups — what the dashboard reads |
| `results/examples.json` | concrete round-trips per language for the drill-down |
| `results/details.jsonl.gz` | all ~331k per-word rows |

`data/ghana-nouns.csv` is symlinked from
[GhanaNLP/GhanaNouns](https://github.com/GhanaNLP/GhanaNouns) and is used only to
attach an independent frequency count to 134 of the 204 words, giving the
dashboard a second axis (frequent vs rare vocabulary) alongside the dataset's own
categories.

## Sources

* [BU 200 Word Project](https://www.bu.edu/200word/) — Boston University African
  Language Program and Geddes Language Center
* [GhanaNLP/GhanaNouns](https://github.com/GhanaNLP/GhanaNouns) — Ghanaian noun
  frequency inventory
* [AfriSpeech/afriso](https://github.com/AfriSpeech/afriso) — African language
  names → ISO 639-3 codes, from SIL ISO 639-3 and Glottolog (CC-BY 4.0)
* Google Gemini via the `google-genai` SDK
