"""Streamlit dashboard — GitHub Hiring Repository Intelligence."""

import json
import re
import sys
import os
import warnings
from pathlib import Path

# Must be set before torch is imported to avoid OMP duplicate-library crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
import torch
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).resolve().parent / ".env")

sys.path.insert(0, "src")
from summarization import build_text_summary

# ── constants ──────────────────────────────────────────────────────────────────
LABELS = ["intern", "junior", "senior", "lead_architect",
          "template_boilerplate", "low_value"]

LABEL_COLORS = {
    "intern":               "#9e9e9e",
    "junior":               "#4caf50",
    "senior":               "#2196f3",
    "lead_architect":       "#ff9800",
    "template_boilerplate": "#9c27b0",
    "low_value":            "#f44336",
}

LABEL_EMOJI = {
    "intern":               "🟤",
    "junior":               "🟢",
    "senior":               "🔵",
    "lead_architect":       "🟡",
    "template_boilerplate": "🟣",
    "low_value":            "🔴",
}

MODELS_DIR  = Path("models/trained_models")
FIGURES_DIR = Path("output/figures")
METRICS_DIR = Path("output/metrics")
DATA_DIR    = Path("data/labeled")

st.set_page_config(
    page_title="Repository Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-left: 3rem;
            padding-right: 3rem;
            padding-top: 2rem;
        }
        p { line-height: 1.7; margin-bottom: 0.6rem; }
        li { line-height: 1.8; margin-bottom: 0.2rem; }
        h2 { margin-top: 1.6rem !important; margin-bottom: 0.6rem !important; }
        h3 { margin-top: 1.2rem !important; margin-bottom: 0.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    path = DATA_DIR / "repos_labeled.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_eval_report() -> dict:
    path = METRICS_DIR / "evaluation_report.json"
    if not path.exists():
        return {}
    return json.load(open(path))


@st.cache_resource
def load_bert():
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    if not (MODELS_DIR / "model.safetensors").exists():
        return None, None
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODELS_DIR))
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODELS_DIR))
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


# ── prediction helpers ─────────────────────────────────────────────────────────

def predict_text(text: str, tokenizer, model) -> tuple[str, float, np.ndarray]:
    label2id = json.load(open(MODELS_DIR / "label2id.json"))["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    device = next(model.parameters()).device
    inputs = tokenizer(
        text, padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], float(probs[pred_id]), probs


def fetch_github_repo(owner: str, name: str) -> dict | None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    base = "https://api.github.com"

    def _get(url):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            return r if r.status_code == 200 else None
        except Exception:
            return None

    repo_r = _get(f"{base}/repos/{owner}/{name}")
    if repo_r is None:
        return None
    repo = repo_r.json()

    # root contents for file signals
    contents_r = _get(f"{base}/repos/{owner}/{name}/contents/")
    root_items = contents_r.json() if (contents_r and isinstance(contents_r.json(), list)) else []

    _CI  = {".travis.yml", "jenkinsfile", ".gitlab-ci.yml", ".circleci"}
    _TST = {"tests", "test", "__tests__", "spec", "specs"}
    _REQ = {"requirements.txt", "pyproject.toml", "setup.cfg",
            "package.json", "pom.xml", "go.mod", "cargo.toml"}

    signals = {
        "has_readme": False, "readme_size_bytes": 0,
        "has_docker": False, "has_requirements": False,
        "has_tests_folder": False, "has_ci_cd": False,
        "root_file_count": len(root_items),
    }
    for item in root_items:
        n = item["name"].lower()
        t = item.get("type", "")
        if n.startswith("readme"):
            signals["has_readme"] = True
            signals["readme_size_bytes"] = item.get("size", 0)
        if n == "dockerfile":
            signals["has_docker"] = True
        if n in _REQ:
            signals["has_requirements"] = True
        if n in _TST and t == "dir":
            signals["has_tests_folder"] = True
        if n == ".github" and t == "dir":
            signals["has_ci_cd"] = True
        if n in _CI:
            signals["has_ci_cd"] = True

    # contributor count via Link header
    cont_r = _get(f"{base}/repos/{owner}/{name}/contributors?per_page=1&anon=false")
    contributors = 0
    if cont_r:
        link = cont_r.headers.get("Link", "")
        m = re.search(r'page=(\d+)>; rel="last"', link)
        contributors = int(m.group(1)) if m else len(cont_r.json() or [])

    from datetime import datetime, timezone
    def _days(iso):
        if not iso:
            return 0
        now = datetime.now(timezone.utc)
        dt  = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (now - dt).days

    return pd.Series({
        "full_name":            repo["full_name"],
        "description":          (repo.get("description") or "")[:300],
        "language":             repo.get("language") or "unknown",
        "topics":               ",".join(repo.get("topics", [])),
        "stars":                repo["stargazers_count"],
        "forks":                repo["forks_count"],
        "open_issues":          repo["open_issues_count"],
        "contributor_count":    contributors,
        "size_kb":              repo["size"],
        "is_fork":              repo["fork"],
        "is_archived":          repo.get("archived", False),
        "last_commit_days_ago": _days(repo.get("pushed_at")),
        "repo_age_days":        _days(repo.get("created_at")),
        **signals,
    })


def parse_github_url(raw: str) -> tuple[str | None, str | None]:
    raw = raw.strip().rstrip("/")
    m = re.search(r'github\.com/([^/\s]+)/([^/\s]+)', raw)
    if m:
        return m.group(1), m.group(2)
    parts = raw.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Problem & Methodology",
    "Exploratory Analysis",
    "Model Results",
    "Interactive Exploration",
])


# ── Tab 1: Problem & Methodology ───────────────────────────────────────────────

with tab1:
    st.title("GitHub Hiring Repository Intelligence")
    st.markdown(
        "**Weak-supervision NLP pipeline** that classifies GitHub repositories "
        "by engineering maturity level to assist technical recruiters."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("The Problem")
        st.markdown("""
Hiring a software engineer costs **$15,000–$30,000** in recruiter time,
interviews, and onboarding. Most of that cost comes from screening candidates
who don't meet the required level.

**GitHub repositories are the most objective signal of engineering maturity**
— but no tool automatically classifies them.

**Who needs this:**
- Technical recruiters doing pre-screening
- HR teams evaluating hundreds of profiles
- Hiring platforms (LinkedIn, Toptal, TripleByte)
        """)

    with col2:
        st.subheader("Pipeline")
        _pipe_style = (
            "background:rgba(128,128,128,0.08);border:1px solid rgba(128,128,128,0.25);"
            "border-radius:6px;padding:18px 22px;font-family:'Courier New',monospace;"
            "font-size:0.82em;line-height:2.0;white-space:pre;"
            "overflow-x:auto;color:inherit;"
        )
        st.markdown(
            f'<div style="{_pipe_style}">'
            "GitHub API  ──►  data/raw/          ──►  preprocessing.py\n"
            "                 (per-class CSVs)         data/processed/\n"
            "\n"
            "DeepSeek V3 ──►  data/labeled/      ──►  split.py\n"
            "                 (weak labels)            data/splits/ (70/15/15)\n"
            "\n"
            "baseline.py  ──►  TF-IDF + LogReg   (comparison)\n"
            "train.py     ──►  DistilBERT        ──►  evaluation.py\n"
            "\n"
            "Streamlit   ──►  app.py  (4 tabs)"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Engineering Maturity Labels")
    labels_info = [
        ("🟡", "lead_architect",       "Large-scale systems, complex architecture, 50+ contributors, advanced CI/CD"),
        ("🔵", "senior",               "Well-organised, comprehensive tests, CI/CD, solid documentation"),
        ("🟢", "junior",               "Functional but simple, minimal tests, little CI/CD or design patterns"),
        ("🟤", "intern",               "Single-file scripts, no tests, minimal structure, no docs"),
        ("🟣", "template_boilerplate", "Designed to be copied — cookiecutters, starters, scaffolds"),
        ("🔴", "low_value",            "Abandoned, empty, unchanged fork, or no engineering signal"),
    ]
    cards_html = ""
    for emoji, label, desc in labels_info:
        color = LABEL_COLORS[label]
        cards_html += (
            f"<div style='background:{color}18;border-left:4px solid {color};"
            f"border-radius:6px;padding:10px 14px;margin-bottom:8px;line-height:1.5;color:inherit;'>"
            f"<strong>{emoji} <code>{label}</code></strong> — {desc}"
            f"</div>"
        )
    st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("Signals Collected (17 per repository)")
    signals_df = pd.DataFrame([
        ("stars, forks, open_issues",             "Community signal & interest"),
        ("contributor_count",                     "Collaboration & team size"),
        ("size_kb",                               "Codebase volume"),
        ("is_fork, is_archived",                  "Repository status"),
        ("last_commit_days_ago, repo_age_days",   "Activity vs abandonment"),
        ("has_readme, readme_size_bytes",          "Documentation quality"),
        ("has_tests_folder, has_ci_cd",           "Engineering culture"),
        ("has_docker, has_requirements",           "Reproducibility & dependency mgmt"),
        ("root_file_count",                       "Structural complexity"),
    ], columns=["Signal(s)", "What it captures"])
    st.dataframe(signals_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Weak Labeling Strategy")
    st.markdown("""
No public dataset of repos labelled by engineering maturity exists.
We use **DeepSeek-V3** as a virtual senior recruiter to generate proxy labels:

- **Cost:** ~$0.20 for 645 repos (vs ~$3–5 with GPT-4o)
- **temperature=0.1** → deterministic, not creative
- **response_format: json_object** → structured, parseable output
- **Checkpoint every 10 repos** → resilient to network failures

After labeling: 614 clean repos (31 ambiguous cases discarded).
    """)

    st.divider()
    st.subheader("Repository Selection Methodology")
    st.markdown("""
Repositories were selected from **curated, hand-verified lists** organised by maturity class —
not via the GitHub Search API, which triggered GitHub's undocumented secondary rate limit
(HTTP 403 abuse detection) regardless of primary quota. All collection uses direct
`GET /repos/{owner}/{name}` calls with hardcoded lists.

A supplementary script (`github_collect_missing.py`) extended the lists to fill under-populated classes
while skipping already-collected entries.
    """)
    _sel_df = pd.DataFrame([
        ("lead_architect",       "linux, pytorch, tensorflow, kubernetes …",    "130+", "~139"),
        ("senior",               "fastapi, requests, flask, scikit-learn …",    "200+", "~236"),
        ("junior",               "small personal / course repos …",             "120+", "~119"),
        ("intern",               "first scripts, single-file utilities …",       "60+",  "~60"),
        ("template_boilerplate", "cookiecutter templates, scaffold starters …",  "100",  "~100"),
        ("low_value",            "abandoned forks, empty repos, dotfiles …",    "100",  "~100"),
    ], columns=["Class", "Source examples", "Target", "Collected"])
    st.dataframe(_sel_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Prompt Design")
    st.markdown("""
The prompt is the most critical design decision: it defines what the LLM understands as
"engineering maturity" and determines label quality for the entire downstream pipeline.
    """)
    with st.expander("System prompt sent to DeepSeek-V3"):
        st.text(
            "You are a senior software engineering recruiter with deep technical expertise.\n"
            "You assess GitHub repositories to determine the engineering maturity level\n"
            "of their authors, helping companies make faster and more accurate hiring decisions."
        )
    with st.expander("User prompt structure (abbreviated — one per repository)"):
        st.text(
            "Classify this GitHub repository into EXACTLY ONE of the following\n"
            "six engineering maturity categories:\n\n"
            "  intern          — Single-file scripts, no tests, no docs ...\n"
            "  junior          — Basic folder structure, minimal tests, simple logic ...\n"
            "  senior          — Well-organised, CI/CD, comprehensive tests ...\n"
            "  lead_architect  — Large-scale multi-module, 50+ contributors ...\n"
            "  template_boiler — Primary purpose is to be copied or scaffolded ...\n"
            "  low_value       — Abandoned, empty, unchanged fork ...\n\n"
            "Repository profile:\n"
            "  Repository: {full_name}\n"
            "  Description: ...\n"
            "  Language: ... | Topics: ...\n"
            "  Stars: ... | Forks: ... | Contributors: ...\n"
            "  Size: ... KB | Repo age: ... days | Last commit: ... days ago\n"
            "  README: ... | Tests: ... | CI/CD: ... | Docker: ... | Dependencies: ...\n"
            "  Root file count: ... | Status: original / fork\n\n"
            'Respond ONLY with valid JSON:\n'
            '{"label": "...", "confidence": 0.0-1.0, "reasoning": "2-3 sentences"}'
        )
    st.markdown("""
**Key design decisions:**
- `temperature=0.1` — deterministic classifications; minimises label variance across runs
- `response_format: json_object` — forces structured parseable output
- **Recruiter persona** — grounds the LLM in hiring context, not abstract code quality
- **Negative examples** in each category definition reduce boundary confusion
- **Confidence + reasoning fields** — unknowns (confidence=0 or unparseable) are discarded
    """)

    st.divider()
    st.subheader("Dataset Construction")
    _train_p = Path("data/splits/train.csv")
    _val_p   = Path("data/splits/val.csv")
    _test_p  = Path("data/splits/test.csv")
    if _train_p.exists() and _val_p.exists() and _test_p.exists():
        _tr_lbl   = pd.read_csv(_train_p)["llm_label"].value_counts()
        _va_lbl   = pd.read_csv(_val_p)["llm_label"].value_counts()
        _te_lbl   = pd.read_csv(_test_p)["llm_label"].value_counts()
        _split_df = pd.DataFrame({
            "Label": LABELS,
            "Train": [int(_tr_lbl.get(l, 0)) for l in LABELS],
            "Val":   [int(_va_lbl.get(l, 0)) for l in LABELS],
            "Test":  [int(_te_lbl.get(l, 0)) for l in LABELS],
        })
        _split_df["Total"] = _split_df["Train"] + _split_df["Val"] + _split_df["Test"]
        st.markdown("**Stratified 70 / 15 / 15 split — class proportions preserved in every fold:**")
        st.dataframe(_split_df, width="stretch", hide_index=True)
        st.caption(
            f"Total: {int(_split_df['Total'].sum())} repos  ·  "
            f"Train: {int(_split_df['Train'].sum())}  ·  "
            f"Val: {int(_split_df['Val'].sum())}  ·  "
            f"Test: {int(_split_df['Test'].sum())}"
        )
    else:
        st.info("Split files not found. Run `python src/split.py` to generate them.")

    st.divider()
    st.subheader("Key Signals per Maturity Level")
    st.markdown("Which repository signals are most associated with each engineering level — and why:")
    _sig_df = pd.DataFrame([
        ("🟡 lead_architect",       "stars >5k, contributors >50, CI/CD, tests, Docker, size >10 MB",
         "Scale is the differentiator: multi-module architecture + large community"),
        ("🔵 senior",               "CI/CD, tests, requirements file, README >2 KB, size 100–5000 KB",
         "Production practices present; may have low stars but high structural quality"),
        ("🟢 junior",               "README, requirements file, no CI/CD, no tests, size 10–500 KB",
         "Functional code with some structure; testing and automation absent"),
        ("🟤 intern",               "root_file_count <5, no tests, no CI, no Docker, size <50 KB",
         "Single-file or minimal scripts; no packaging, no documentation"),
        ("🟣 template_boilerplate", "README present, generic topics, low contributors despite stars",
         "Scaffold structure designed to be copied, not grown"),
        ("🔴 low_value",            "last_commit >365 days, is_archived, stars <2, size <10 KB",
         "No recent activity, no content, or unchanged fork"),
    ], columns=["Level", "Defining signals", "Recruiting interpretation"])
    st.dataframe(_sig_df, width="stretch", hide_index=True)
    st.markdown("""
**Hardest separation:** `senior` vs `lead_architect` — both have tests and CI/CD.
The discriminating factor is **scale**: contributors >50 and stars >5k
push a repo from senior to lead_architect. The BERT model confuses these 15 times in the test set.

**Most ambiguous pair:** `junior` vs `low_value` — a junior repo with no recent commits
looks identical to an abandoned project in raw signals. Context from description and topics
is the only separator, and BERT only receives a structured text summary.
    """)

    st.divider()
    st.subheader("Low-Value & Template Distinction Logic")
    _dist_col1, _dist_col2 = st.columns(2)
    with _dist_col1:
        st.markdown("""
**`low_value`** — no meaningful hiring signal:
- `is_archived = True` OR `last_commit > 730 days`
- Stars < 2 AND size < 20 KB AND no tests AND no CI/CD
- Fork of a popular repo with zero original contributions
- Personal config files (dotfiles, editor settings)

*A `low_value` repo should not penalise a candidate — it is noise, not evidence.*
        """)
    with _dist_col2:
        st.markdown("""
**`template_boilerplate`** — designed to be copied:
- Topics include "template", "boilerplate", "scaffold", "starter"
- README explains how to clone/use, but code is placeholder
- Often has cookiecutter.json or generator tooling
- Low contributors despite high stars (people star but don't contribute)

*A `template_boilerplate` repo should not count as engineering experience —
copying a scaffold shows tooling familiarity, not design skill.*
        """)

    st.divider()
    st.subheader("Limitations & Ethical Considerations")
    _lim_col1, _lim_col2 = st.columns(2)
    with _lim_col1:
        st.markdown("**Methodological limitations**")
        st.markdown("""
- **Weak labels:** DeepSeek-V3 is a proxy annotator, not a human expert panel.
  Label noise propagates to the fine-tuned model.
- **Small dataset:** 614 repos is insufficient for DistilBERT to generalise.
  The TF-IDF baseline outperforms BERT precisely because the text template is
  highly structured and repetitive — keyword matching dominates.
- **Selection bias:** curated lists over-represent popular English-language
  open-source projects. Private-sector work is entirely invisible to this system.
- **Class imbalance:** `senior` is ≈38% of data; `intern` is <5%.
  WeightedTrainer compensates at training time but the model still struggles
  with rare classes (intern F1 = 0.00 on test set).
- **Text-only input:** the model reads a structured summary, not actual code.
  Code quality metrics (cyclomatic complexity, test coverage) are absent.
        """)
    with _lim_col2:
        st.markdown("**Ethical considerations**")
        st.markdown("""
- **Career impact:** outputs influence hiring decisions. A false negative
  (strong engineer classified as `junior`) could unfairly screen them out.
- **Language and cultural bias:** repos written in English with Western
  open-source conventions score systematically higher than equivalent work
  in other languages or in private enterprise repositories.
- **Stars as proxy:** star count rewards visibility and social networks,
  not engineering skill. An excellent engineer working on internal tooling
  will never accumulate stars.
- **Screening aid, not a gate:** intended use is to surface candidates for
  human review, never to replace a technical interview.
- **Transparency:** candidates should be informed that automated repository
  analysis is used and have the right to contest the classification.
        """)


# ── Tab 2: Exploratory Analysis ───────────────────────────────────────────────

with tab2:
    st.title("Exploratory Analysis")
    df = load_dataset()

    if df is None:
        st.info(
            "Dataset not found (`data/labeled/repos_labeled.csv`). "
            "Run the pipeline to generate it:\n\n"
            "```\npython src/github_collector.py\n"
            "python src/preprocessing.py\n"
            "python src/summarization.py\n"
            "python src/llm_labeling.py\n```"
        )
        st.stop()

    # ── class distribution ──────────────────────────────────────────────────
    st.subheader("Class Distribution")
    counts = df["llm_label"].value_counts().reindex(LABELS).fillna(0).astype(int)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [LABEL_COLORS[l] for l in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=20, ha="right")
        ax.set_ylabel("Number of repositories")
        ax.set_title("Repository count per maturity class", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        st.markdown("**Counts**")
        count_html = ""
        for label in LABELS:
            emoji = LABEL_EMOJI.get(label, "")
            n = int(counts[label])
            pct = n / len(df) * 100
            color = LABEL_COLORS[label]
            count_html += (
                f"<div style='padding:4px 0 4px 8px;border-left:3px solid {color};"
                f"margin-bottom:6px;color:inherit;'>"
                f"{emoji} <strong>{label}</strong>: {n} ({pct:.1f}%)"
                f"</div>"
            )
        st.markdown(count_html, unsafe_allow_html=True)
        st.markdown("---")
        st.metric("Total repos", len(df))
        st.metric("Classes", df["llm_label"].nunique())

    st.markdown("""
**Insight:** `senior` dominates the dataset (≈38%), reflecting the curated selection of popular
production libraries. `intern` and `template_boilerplate` are minority classes — this imbalance
is the root cause of poor recall on rare labels and is compensated by `WeightedTrainer` during
fine-tuning, but remains a limitation on a 614-repo dataset.
    """)

    # ── key signals by class ────────────────────────────────────────────────
    st.subheader("Key Signal Distributions by Class")
    col1, col2 = st.columns(2)

    for col, signal, title in [
        (col1, "log_stars",         "Log Stars by class"),
        (col2, "contributor_count", "Contributor Count by class (capped at 200)"),
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_df = df[df["llm_label"].isin(LABELS)].copy()
            if signal == "contributor_count":
                plot_df[signal] = plot_df[signal].clip(upper=200)
            order  = LABELS
            colors = [LABEL_COLORS[l] for l in order]
            data   = [plot_df[plot_df["llm_label"] == l][signal].dropna() for l in order]
            bp = ax.boxplot(data, labels=order, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=2))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.markdown("""
**Insight:** `log_stars` clearly separates `lead_architect` (high median) from `intern` and
`low_value` (near-zero). `contributor_count` is the strongest single signal for `lead_architect`
vs all others, with an outlier tail driven by repos like linux and pytorch (thousands of contributors).
`senior` shows high variance in both signals — it spans solo engineers and large teams — making
it the hardest class to isolate with a single numerical threshold.
    """)

    # ── signal correlation heatmap ──────────────────────────────────────────
    st.subheader("Signal Correlation Heatmap")
    num_cols = ["stars", "forks", "contributor_count", "size_kb",
                "readme_size_bytes", "root_file_count",
                "last_commit_days_ago", "repo_age_days",
                "has_readme", "has_tests_folder", "has_ci_cd",
                "has_docker", "has_requirements"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 7}, linewidths=0.3,
    )
    ax.set_title("Pearson correlation between repository signals", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        "Boolean signals (has_readme, has_tests_folder, has_ci_cd, has_docker, "
        "has_requirements) are treated as 0/1 integers for correlation."
    )
    st.markdown("""
**Reading the heatmap:** `stars` and `forks` are nearly collinear (r ≈ 0.97) — they capture
the same community signal and could be reduced to one feature without information loss.
`has_ci_cd` and `has_tests_folder` correlate moderately (r ≈ 0.40): teams that test tend
to also automate. `last_commit_days_ago` shows low correlation with all quality signals,
confirming that recent activity alone is insufficient for maturity classification.
    """)


# ── Tab 3: Model Results ───────────────────────────────────────────────────────

with tab3:
    st.title("Model Results")
    report = load_eval_report()

    if not report:
        st.warning("evaluation_report.json not found. Run `python src/evaluation.py` first.")
    else:
        bert = report["bert"]
        base = report["baseline"]

        # ── summary metrics ─────────────────────────────────────────────────
        st.subheader("Summary — Test Set (n=93)")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Baseline Accuracy",    f"{base['accuracy']:.1%}")
        c2.metric("Baseline F1 macro",    f"{base['f1_macro']:.3f}")
        c3.metric("Baseline F1 weighted", f"{base['f1_weighted']:.3f}")
        c4.metric("BERT Accuracy",        f"{bert['accuracy']:.1%}")
        c5.metric("BERT F1 macro",        f"{bert['f1_macro']:.3f}")
        c6.metric("BERT F1 weighted",     f"{bert['f1_weighted']:.3f}")

        st.info(
            "The TF-IDF baseline outperforms DistilBERT on this dataset. "
            "With only 614 labeled samples and a structured text template, "
            "keyword matching is highly effective. BERT would require more data "
            "to leverage its contextual advantage."
        )

        # ── F1 per class ─────────────────────────────────────────────────────
        st.subheader("F1 Score per Class")
        f1_img = FIGURES_DIR / "f1_comparison.png"
        if f1_img.exists():
            st.image(str(f1_img), width="stretch")

        f1_table = pd.DataFrame({
            "Label":    LABELS,
            "Baseline": [base["f1_per_class"].get(l, 0) for l in LABELS],
            "BERT":     [bert["f1_per_class"].get(l, 0) for l in LABELS],
        })
        f1_table["Winner"] = f1_table.apply(
            lambda r: "BERT" if r["BERT"] > r["Baseline"] else "Baseline", axis=1
        )
        st.dataframe(f1_table, width="stretch", hide_index=True)

        # ── confusion matrices ───────────────────────────────────────────────
        st.subheader("Confusion Matrices (normalised)")
        col1, col2 = st.columns(2)
        with col1:
            img = FIGURES_DIR / "confusion_matrix_bert.png"
            if img.exists():
                st.image(str(img), caption="DistilBERT", width="stretch")
        with col2:
            img = FIGURES_DIR / "confusion_matrix_baseline.png"
            if img.exists():
                st.image(str(img), caption="Baseline TF-IDF + LR", width="stretch")

        st.markdown(
            "<div style='background:rgba(249,168,37,0.12); border-left:4px solid #f9a825;"
            " border-radius:4px; padding:14px 18px; margin-top:12px; color:inherit;'>"
            "<strong>Main confusion patterns (BERT)</strong><br><br>"
            "• <code>senior</code> → <code>lead_architect</code> (15 cases):"
            " repos with high stars and CI/CD are pushed toward lead_architect<br><br>"
            "• <code>junior</code> → <code>low_value</code> (9 cases):"
            " low-activity repos without clear junior signals fall through<br><br>"
            "• <code>intern</code> is never correctly identified (F1 = 0.00):"
            " only 3 test samples — too few to learn from"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── sensitivity analysis ─────────────────────────────────────────────
        st.subheader("Sensitivity Analysis")
        sens_img = FIGURES_DIR / "sensitivity_analysis.png"
        if sens_img.exists():
            st.image(str(sens_img), width="stretch")

        sens = report.get("sensitivity_analysis", {})
        if sens:
            sens_df = pd.DataFrame([
                {"Variant": k, "F1 macro": v["f1_macro"], "Accuracy": v["accuracy"]}
                for k, v in sens.items()
            ])
            baseline_f1 = sens.get("All signals", {}).get("f1_macro", 0)
            sens_df["F1 drop"] = (baseline_f1 - sens_df["F1 macro"]).round(4)
            st.dataframe(sens_df, width="stretch", hide_index=True)
            st.caption(
                "CI/CD and Stars/Forks are the most discriminating signals. "
                "Removing both drops F1 macro by "
                f"{sens_df[sens_df['Variant']=='Without CI/CD + Stars']['F1 drop'].values[0]:.4f}."
            )

        # ── error analysis ───────────────────────────────────────────────────
        st.subheader("Error Analysis — BERT Misclassifications")
        _errors_path = METRICS_DIR / "bert_errors.csv"
        if _errors_path.exists():
            _err_df = pd.read_csv(_errors_path)
            st.markdown(
                f"DistilBERT made **{len(_err_df)} errors** on the test set of {report.get('n_test', 93)} repos "
                f"({len(_err_df)/report.get('n_test', 93):.0%} error rate). "
                "Most frequent confusion pairs:"
            )
            _pairs = (
                _err_df.groupby(["llm_label", "bert_pred"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(8)
            )
            _pairs.columns = ["True label", "BERT predicted", "Cases"]
            st.dataframe(_pairs, width="stretch", hide_index=True)
            with st.expander("Full misclassified repository list"):
                _display_cols = [c for c in ["full_name", "llm_label", "bert_pred",
                                              "baseline_pred", "stars",
                                              "has_ci_cd", "has_tests_folder"] if c in _err_df.columns]
                st.dataframe(_err_df[_display_cols], width="stretch", hide_index=True)
        else:
            st.info("Error analysis file not found. Run `python src/evaluation.py` first.")


# ── Tab 4: Interactive Exploration ────────────────────────────────────────────

with tab4:
    st.title("Interactive Repository Exploration")
    st.markdown(
        "Paste a GitHub repository URL and get its predicted engineering "
        "maturity level from the trained DistilBERT model."
    )

    model_available = (MODELS_DIR / "model.safetensors").exists()
    if not model_available:
        st.warning(
            "Model not found at `models/trained_models/`. "
            "Run `python src/train.py` to train and save the model first."
        )

    url_input = st.text_input(
        "GitHub repository URL or owner/repo",
        placeholder="e.g. https://github.com/tiangolo/fastapi  or  tiangolo/fastapi",
    )

    demo_col1, demo_col2, demo_col3 = st.columns(3)
    with demo_col1:
        if st.button("Demo: tiangolo/fastapi"):
            url_input = "tiangolo/fastapi"
    with demo_col2:
        if st.button("Demo: torvalds/linux"):
            url_input = "torvalds/linux"
    with demo_col3:
        if st.button("Demo: octocat/hello-world"):
            url_input = "octocat/hello-world"

    if url_input and url_input.strip():
        owner, repo_name = parse_github_url(url_input)

        if not owner or not repo_name:
            st.error("Could not parse the URL. Use format: `owner/repo` or a full GitHub URL.")
        else:
            with st.spinner(f"Fetching `{owner}/{repo_name}` from GitHub API ..."):
                row = fetch_github_repo(owner, repo_name)

            if row is None:
                st.error(f"Repository `{owner}/{repo_name}` not found or GitHub API error.")
            else:
                text_summary = build_text_summary(row)

                # ── signals card ────────────────────────────────────────────
                st.subheader("Repository Signals")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Stars",        f"{int(row['stars']):,}")
                s2.metric("Forks",        f"{int(row['forks']):,}")
                s3.metric("Contributors", int(row["contributor_count"]))
                s4.metric("Size (KB)",    f"{int(row['size_kb']):,}")
                s5.metric("Last commit",  f"{int(row['last_commit_days_ago'])} days ago")

                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("README",       "Yes" if row["has_readme"]       else "No")
                b2.metric("Tests",        "Yes" if row["has_tests_folder"] else "No")
                b3.metric("CI/CD",        "Yes" if row["has_ci_cd"]        else "No")
                b4.metric("Docker",       "Yes" if row["has_docker"]       else "No")
                b5.metric("Deps file",    "Yes" if row["has_requirements"] else "No")

                # ── prediction ──────────────────────────────────────────────
                if model_available:
                    with st.spinner("Running DistilBERT prediction ..."):
                        tokenizer, model = load_bert()
                        if tokenizer is None:
                            st.error("Model files present but could not be loaded.")
                        else:
                            pred_label, confidence, probs = predict_text(
                                text_summary, tokenizer, model
                            )

                    color = LABEL_COLORS.get(pred_label, "#607d8b")
                    emoji = LABEL_EMOJI.get(pred_label, "")

                    st.subheader("Prediction")
                    st.markdown(
                        f"<div style='background:{color}22; border-left:6px solid {color}; "
                        f"padding:16px; border-radius:6px; font-size:1.3em; font-weight:bold;'>"
                        f"{emoji} &nbsp; {pred_label} &nbsp;&nbsp;"
                        f"<span style='font-size:0.8em; font-weight:normal; color:#555;'>"
                        f"confidence {confidence:.1%}</span></div>",
                        unsafe_allow_html=True,
                    )

                    # confidence bar chart
                    fig, ax = plt.subplots(figsize=(7, 3))
                    colors  = [LABEL_COLORS[l] for l in LABELS]
                    ax.barh(LABELS, probs, color=colors, alpha=0.85)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.set_title("Class probabilities (DistilBERT)", fontweight="bold")
                    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
                    ax.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                else:
                    st.info("Train the model to see predictions.")

                # ── text summary used ───────────────────────────────────────
                with st.expander("Text summary sent to the model"):
                    st.text(text_summary)
