"""Static Hugging Face Space dashboard for the Gemini African word-MT benchmark.

Results are not bundled with the app. Every load reads the published JSON from
the GitHub repo over raw.githubusercontent.com, so a new benchmark run shows up
here after a push with no redeploy. A local results/ directory is used instead
when one is present, which makes local development work.

Run locally:  GRADIO_SERVER_NAME=0.0.0.0 python space/app.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OWNER = os.environ.get("BENCH_REPO", "michsethowusu/gemini-word-mt-bench")
BRANCH = os.environ.get("BENCH_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{OWNER}/{BRANCH}"
LOCAL = Path(__file__).resolve().parent.parent / "results"

TIER_COLOR = {"strong": "#1a936f", "medium": "#e9a13b", "weak": "#d1495b"}
TIER_ORDER = {"strong": 0, "medium": 1, "weak": 2}

_cache: dict[str, tuple[float, object]] = {}
TTL = 600  # seconds


def load(name: str):
    """Fetch <name>.json from the local results dir, else from GitHub raw."""
    local = LOCAL / f"{name}.json"
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    hit = _cache.get(name)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    url = f"{RAW}/results/{name}.json"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            _cache[name] = (time.time(), data)
            return data
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt == 2:
                return {"__error__": f"could not load {name}.json: {e}"}
            time.sleep(2 * (attempt + 1))


def frame(summary: dict) -> pd.DataFrame:
    df = pd.DataFrame(summary["languages"])
    for c in ("by_category", "by_band"):
        df[c] = df[c].map(lambda d: "; ".join(f"{k} {v}" for k, v in d.items()))
    return df


def fmt(n: float) -> str:
    return f"{n:.1f}" if pd.notna(n) else "-"


# ------------------------------------------------------------------ leaderboard
def leaderboard(
    summary: dict,
    region: str,
    family: str,
    tier: str,
    search: str,
    min_words: int,
    sort_by: str,
    top_n: int,
):
    if "__error__" in summary:
        return None, None, f"**Error:** {summary['__error__']}", None

    df = frame(summary)
    if region != "All":
        df = df[df["region"] == region]
    if family != "All":
        df = df[df["family"] == family]
    if tier != "All":
        df = df[df["tier"] == tier]
    if search:
        q = search.lower()
        df = df[
            df["name"].str.lower().str.contains(q, na=False)
            | df["iso639_3"].str.lower().str.contains(q, na=False)
        ]
    df = df[df["n"] >= min_words]

    if not len(df):
        return None, None, "No languages match these filters.", None

    key = {
        "pass rate": "pass",
        "close rate": "close",
        "name": "name",
        "screen score": "screen_score",
        "region": "region",
    }[sort_by]
    ascending = sort_by in ("name", "region")
    view = df.sort_values(key, ascending=ascending).head(top_n)

    chart = px.bar(
        view.sort_values("pass"),
        x="pass",
        y="name",
        orientation="h",
        color="tier",
        color_discrete_map=TIER_COLOR,
        category_orders={"tier": ["weak", "medium", "strong"]},
        hover_data={
            "iso639_3": True,
            "family": True,
            "region": True,
            "close": ":.1f",
            "screen_score": True,
            "n": True,
        },
        labels={"pass": "Round-trip pass rate (%)", "name": ""},
        title=f"{len(df)} languages match",
    )
    chart.update_layout(height=max(320, 22 * len(view) + 140), margin=dict(l=10, r=10, t=50, b=10))
    chart.update_yaxes(title="")

    counts = df["tier"].value_counts().to_dict()
    stat = "  ".join(
        f"**{k.title()}**: {counts.get(k, 0)}" for k in ("strong", "medium", "weak")
    )
    return chart, view, stat, df


# -------------------------------------------------------------------- overview
def overview_charts(summary: dict):
    if "__error__" in summary:
        return None, None, None, None, None, summary["__error__"]
    meta, bc = summary["meta"], summary["by_category"]

    cat = pd.DataFrame(
        [{"category": k, **v} for k, v in bc.items()]
    ).sort_values("mean_pass")
    fig_cat = px.bar(
        cat,
        x="mean_pass",
        y="category",
        orientation="h",
        color="mean_pass",
        color_continuous_scale=["#d1495b", "#e9a13b", "#1a936f"],
        range_color=(0, 100),
        labels={"mean_pass": "Mean pass rate (%)"},
        title="Pass rate by BU 200 Word category (pooled over all languages)",
    )
    fig_cat.update_layout(height=520, coloraxis_showscale=False)
    fig_cat.update_yaxes(title="")

    reg = pd.DataFrame([{"region": k, **v} for k, v in summary["by_region"].items()])
    fig_reg = px.bar(
        reg.sort_values("mean_pass", ascending=False),
        x="region",
        y="mean_pass",
        color="mean_pass",
        color_continuous_scale=["#d1495b", "#e9a13b", "#1a936f"],
        range_color=(0, 100),
        labels={"mean_pass": "Mean pass rate (%)", "region": ""},
        title="Pass rate by African region",
    )
    fig_reg.update_layout(height=380, coloraxis_showscale=False)

    fig_tier = go.Figure(
        go.Pie(
            labels=[k.title() for k in ("strong", "medium", "weak")],
            values=[summary["tier_counts"].get(k, 0) for k in ("strong", "medium", "weak")],
            marker=dict(
                colors=[TIER_COLOR[k] for k in ("strong", "medium", "weak")],
                line=dict(color="#fff", width=2),
            ),
            hole=0.55,
            textinfo="label+percent",
        )
    )
    fig_tier.update_layout(
        height=380, title="Languages per tier", showlegend=False
    )

    band = summary.get("by_frequency_band") or {}
    if band:
        bf = pd.DataFrame(
            [
                {
                    "band": f"Q{int(k)}",
                    "label": ["most frequent", "frequent", "mid", "low", "rarest"][int(k) - 1]
                    if 1 <= int(k) <= 5
                    else k,
                    "mean_pass": v["mean_pass"],
                    "n_words": v["n_words"],
                }
                for k, v in sorted(band.items(), key=lambda kv: int(kv[0]))
            ]
        )
        fig_band = px.bar(
            bf,
            x="band",
            y="mean_pass",
            color="mean_pass",
            color_continuous_scale=["#d1495b", "#e9a13b", "#1a936f"],
            range_color=(0, 100),
            text="n_words",
            custom_data=["label", "n_words"],
            labels={
                "mean_pass": "Mean pass rate (%)",
                "band": "Frequency quintile",
            },
            title="Pass rate by GhanaNouns frequency band (2nd axis; 134/204 words have a count)",
        )
        fig_band.update_traces(
            texttemplate="%{customdata[1]} words<br>%{y:.0f}%", textposition="outside"
        )
        fig_band.update_layout(height=380, coloraxis_showscale=False, yaxis_range=[0, 105])
    else:
        fig_band = go.Figure()

    fig_scatter = px.scatter(
        frame(summary),
        x="screen_score",
        y="pass",
        color="region",
        hover_name="name",
        hover_data={"iso639_3": True, "family": True, "tier": True, "n": True},
        labels={"screen_score": "Screening score (of 3 common words)", "pass": "Pass rate (%)"},
        title="Pass rate vs screening score",
    )
    fig_scatter.update_layout(height=380)

    head = (
        f"**Model:** `{meta['model']}` &nbsp;&nbsp; **Generated:** {meta['generated']} &nbsp;&nbsp; "
        f"**Languages:** {meta['n_languages_tested']:,} tested / {meta['n_languages_screened']:,} screened "
        f"&nbsp;&nbsp; **Words each:** {meta['n_words']} &nbsp;&nbsp; "
        f"**Rows:** {meta['n_rows']:,} &nbsp;&nbsp; "
        f"**Calls:** {meta['n_rows']*2:,}"
    )
    return fig_cat, fig_reg, fig_tier, fig_scatter, fig_band, head


def category_heatmap(summary: dict):
    if "__error__" in summary:
        return None
    cats = summary["categories"]
    langs = summary["languages"]
    # Top languages by pass rate keep the heatmap legible.
    langs = sorted(langs, key=lambda l: -l["pass"])[:120]
    z = [
        [l["by_category"].get(c) for c in cats] for l in langs
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cats,
            y=[l["name"] for l in langs],
            zmin=0,
            zmax=100,
            colorscale=[[0, "#d1495b"], [0.5, "#e9a13b"], [1, "#1a936f"]],
            colorbar=dict(title="pass %"),
        )
    )
    fig.update_layout(
        height=900,
        title="Top 120 languages x BU 200 Word category",
        xaxis=dict(tickangle=-45, title=""),
        yaxis=dict(tickfont=dict(size=8), title=""),
    )
    return fig


# ---------------------------------------------------------------- detail view
def language_detail(summary: dict, examples: dict, iso: str):
    if "__error__" in summary:
        return None, None, summary["__error__"]
    rec = next((l for l in summary["languages"] if l["iso639_3"] == iso), None)
    if rec is None:
        return None, None, "Language not found."

    cats = list(rec["by_category"].items())
    fig = px.bar(
        pd.DataFrame(cats, columns=["category", "pass"]),
        x="pass",
        y="category",
        orientation="h",
        color="pass",
        color_continuous_scale=["#d1495b", "#e9a13b", "#1a936f"],
        range_color=(0, 100),
        labels={"pass": "Pass rate (%)"},
        title=f"{rec['name']} by category",
    )
    fig.update_layout(height=440, coloraxis_showscale=False)
    fig.update_yaxes(title="", autorange="reversed")

    ex = (examples or {}).get(iso) or []
    if ex:
        df = pd.DataFrame(ex)[
            ["word", "category", "gemini", "back", "pass", "bu_reference", "matches_reference"]
        ].rename(
            columns={
                "word": "English", "gemini": f"{rec['name']} (Gemini)",
                "back": "back-translation", "pass": "pass",
                "bu_reference": "BU 200 reference", "matches_reference": "matches BU",
            }
        )
        df = df.sort_values(["pass", "category"], ascending=[False, True])
    else:
        df = pd.DataFrame()

    head = (
        f"### {rec['name']} `{rec['iso639_3']}`\n"
        f"**{rec['family']}** · {rec['region']} · screened **{rec['screen_score']}/3**\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| tier | **{rec['tier'].upper()}** |\n"
        f"| round-trip pass | **{fmt(rec['pass'])}%** ({rec['n']} words) |\n"
        f"| close match | {fmt(rec['close'])}% |\n"
        f"| echoed English | {fmt(rec['echoed'])}% |\n"
        f"| unreadable output | {fmt(rec['unreadable'])}% |\n"
        f"| BU reference agreement | "
        f"{fmt(rec['bu_reference_agreement']) if rec['bu_reference_agreement'] is not None else 'n/a'}% |\n"
    )
    if rec.get("n_excluded"):
        head += f"| excluded (prompt collision) | {rec['n_excluded']} |\n"
    return fig, df, head


# ------------------------------------------------------------------------ app
def build() -> gr.Blocks:
    summary = load("summary")
    examples = load("examples")
    fig_cat, fig_reg, fig_tier, fig_scatter, fig_band, head = overview_charts(summary)

    regions = ["All"] + sorted(summary.get("by_region", {}))
    families = ["All"] + sorted({l["family"] for l in summary.get("languages", [])})
    isos = [l["iso639_3"] for l in summary.get("languages", [])]
    iso_names = {
        l["iso639_3"]: f"{l['name']} ({l['iso639_3']}) - {fmt(l['pass'])}% [{l['tier']}]"
        for l in summary.get("languages", [])
    }

    with gr.Blocks(title="Gemini African Word-MT Benchmark", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Gemini × African Languages — Word-Level MT Benchmark")
        gr.Markdown(
            "Every word is translated **English → target language → back to English** and "
            "scored on whether the round-trip returns the original word. Each direction is a "
            "separate, stateless call, so the back-translator never sees the source word. "
            "Vocabulary is the "
            "[BU 200 Word Project](https://www.bu.edu/200word/) English side, cross-checked "
            "across all 9 published editions.\n\n"
            f"**Tiers** — strong ≥ {summary.get('meta', {}).get('tiers', {}).get('strong', 0)*100:.0f}% "
            f"· medium ≥ {summary.get('meta', {}).get('tiers', {}).get('medium', 0)*100:.0f}% "
            "· weak below that."
        )
        gr.Markdown(head or "")

        with gr.Tab("Overview"):
            with gr.Row():
                gr.Plot(fig_scatter, scale=1)
                gr.Plot(fig_tier, scale=1)
            gr.Plot(fig_cat)
            with gr.Row():
                gr.Plot(fig_reg, scale=1)
                gr.Plot(fig_band, scale=1)

        with gr.Tab("Leaderboard"):
            with gr.Row():
                s_region = gr.Dropdown(regions, value="All", label="Region")
                s_family = gr.Dropdown(families, value="All", label="Family")
                s_tier = gr.Dropdown(["All", "strong", "medium", "weak"], value="All", label="Tier")
                s_search = gr.Textbox(placeholder="search name or ISO code", label="Search")
            with gr.Row():
                s_min = gr.Slider(0, 204, value=204, step=1, label="min words scored")
                s_sort = gr.Dropdown(
                    ["pass rate", "name", "close rate", "screen score", "region"],
                    value="pass rate",
                    label="Sort by",
                )
                s_top = gr.Slider(10, 400, value=120, step=10, label="rows / bars")
            l_chart = gr.Plot()
            l_stat = gr.Markdown()
            l_table = gr.Dataframe(
                headers=["Language", "ISO", "Family", "Region", "Tier", "Pass %", "Close %",
                         "Screen", "Words"],
                wrap=True,
                interactive=True,
            )

            def run(rg, fm, tr, sc, mw, so, tn):
                ch, view, stat, full = leaderboard(summary, rg, fm, tr, sc, mw, so, tn)
                if full is None:
                    return None, "No languages match these filters.", stat, None
                show = full.sort_values("pass", ascending=False).copy()
                show = show[
                    ["name", "iso639_3", "family", "region", "tier", "pass", "close",
                     "screen_score", "n"]
                ]
                return ch, show, stat, ch

            for ctl in (s_region, s_family, s_tier, s_search, s_min, s_sort, s_top):
                ctl.change(run, [s_region, s_family, s_tier, s_search, s_min, s_sort, s_top],
                           [l_chart, l_table, l_stat, l_chart])
            s_search.submit(run, [s_region, s_family, s_tier, s_search, s_min, s_sort, s_top],
                            [l_chart, l_table, l_stat, l_chart])

        with gr.Tab("Category heatmap"):
            gr.Plot(category_heatmap(summary))

        with gr.Tab("Language detail"):
            d_pick = gr.Dropdown(
                choices=[iso_names[i] for i in isos], value=iso_names[isos[0]] if isos else None,
                label="Language",
            )
            d_info = gr.Markdown()
            d_chart = gr.Plot()
            d_table = gr.Dataframe(wrap=True, interactive=True)
            d_pick.change(lambda v: language_detail(summary, examples, v.split(" (")[1].split(")")[0]),
                          [d_pick], [d_chart, d_table, d_info])
            if isos:
                demo.load(lambda: language_detail(summary, examples, isos[0]),
                          None, [d_chart, d_table, d_info])

        gr.Markdown(
            f"<small>Data: <a href='{RAW}/results/summary.json'>{OWNER}</a> "
            f"@ <code>{BRANCH}</code> · per-word rows: "
            f"<a href='{RAW}/results/details.jsonl.gz'>details.jsonl.gz</a> · "
            f"refreshes every {TTL // 60} min</small>"
        )
    return demo


if __name__ == "__main__":
    build().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
