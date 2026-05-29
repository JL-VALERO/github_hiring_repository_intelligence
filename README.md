# GitHub Hiring Repository Intelligence

**Track A — Engineering Maturity Classification**
Course: Data Science with Python | HW_04_202601

Weak-supervision NLP pipeline that classifies GitHub repositories by engineering
maturity level to assist technical recruiters in screening candidates.

---

## Classification Labels

| Label | Description |
|---|---|
| `lead_architect` | Large, complex systems with clear architectural patterns |
| `senior` | Well-structured, tested, documented production libraries |
| `junior` | Functional but simple: tutorials, basic projects |
| `intern` | Single-file scripts, minimal structure, no tests |
| `template_boilerplate` | Repos designed to be copied as starting points |
| `low_value` | Abandoned, empty, or low-signal repositories |

---

## Pipeline

```
GitHub API ──► data/raw/          ──► preprocessing.py
               (per-class CSVs)       data/processed/

LLM (DeepSeek) ──► data/labeled/  ──► train/val/test split
                   (weak labels)       data/splits/

DistilBERT ──► models/            ──► evaluation.py
               fine-tuned             output/

Streamlit ──► app.py  (4 tabs)
```

---

## Business Applications

This system targets technical hiring contexts where manual repository review is too
slow or inconsistent at scale.

| Stakeholder | Use case |
|---|---|
| Technical recruiters | Pre-screen candidates by estimated engineering maturity before interview scheduling |
| Startups & hiring platforms | Automated scoring layer in candidate pipelines (LinkedIn, Toptal, TripleByte) |
| Engineering managers | Objective comparison of candidates' public portfolios |
| Accelerators | Fast evaluation of founding-team technical depth |

**Important constraints:**
- Intended as a **screening aid**, not a binary gate — human review is mandatory before any hiring decision.
- Evaluates only public GitHub repositories; private enterprise work is entirely invisible.
- Results should be disclosed to candidates, who retain the right to contest the classification.

---

## Data Collection Methodology

### Initial collection — GitHub REST API (direct fetch)

Repository signals are collected via `GET /repos/{owner}/{name}` (GitHub REST API v3).
Each repository contributes 17 signals covering community activity, code structure,
DevOps practices, and time-based decay.

Repositories are selected using **curated predefined lists** organized by maturity
class, using stars, repository size, and known organizational provenance as proxies.
The GitHub Search API was intentionally avoided (see limitation below).

| Class | Collection target | Final count |
|---|---|---|
| `senior` | 200+ | ~236 |
| `lead_architect` | 130 | ~139 |
| `junior` | 120+ | ~119 |
| `template_boilerplate` | 100 | ~100 |
| `low_value` | 100 | ~100 |
| `intern` | 60+ | ~60 |
| **Total (after LLM filtering)** | — | **614** |

### Minority-class supplement — `src/github_collect_missing.py`

For `lead_architect`, `template_boilerplate`, and `low_value`, the initial collection
yielded fewer repositories than the target (130 / 100 / 100 respectively).
A supplementary script appends new repositories from extended hardcoded lists,
reading existing CSVs and skipping already-collected entries to avoid duplication.

### Known limitation — GitHub Secondary Rate Limit

Early collection attempts used `GET /search/repositories` to query GitHub dynamically.
This triggered GitHub's **secondary rate limit** (abuse detection), which returns
`HTTP 403` independently of the primary quota (`X-RateLimit-Remaining`).
The secondary limit is not documented with a fixed threshold and cannot be predicted.

**Design decision:** All collection was migrated to direct `GET /repos/{owner}/{name}`
calls with hardcoded lists. This eliminates search-based 403 errors at the cost of
reduced discovery flexibility.

**Area of improvement:** A production system would use GitHub's GraphQL API or
a pre-built dataset (e.g., GH Archive, GHTorrent) to achieve dynamic, large-scale
collection without rate-limit constraints.

---

## Requisitos del entorno

- **Python 3.10.20 es obligatorio.** Versiones anteriores no soportan la sintaxis
  de union types (`X | Y`) usada en las anotaciones de tipo del proyecto.

- Instalar PyTorch con soporte CUDA:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- Luego instalar el resto de dependencias:
  ```bash
  pip install -r requirements.txt
  ```

---

## Signals Collected (17 per repository)

| Signal | Type | Captures |
|---|---|---|
| `stars`, `forks`, `open_issues` | int | Community signal |
| `contributor_count` | int | Collaboration |
| `size_kb` | int | Codebase size |
| `is_fork`, `is_archived` | bool | Repository status |
| `last_commit_days_ago`, `repo_age_days` | int | Activity / abandonment |
| `has_readme`, `readme_size_bytes` | bool / int | Documentation quality |
| `has_tests_folder`, `has_ci_cd` | bool | Engineering culture |
| `has_docker`, `has_requirements` | bool | Dependency management |
| `root_file_count` | int | Structure complexity |

---

## Repository Summary Creation

Each repository is converted into a structured plain-text profile by `src/summarization.py`.
This single format serves as input to both the LLM labeling step and DistilBERT fine-tuning.

**Output format (one block per repository):**

```
Repository: {owner}/{name}
Description: "{description}"
Language: {language} | Topics: {topics}
Stars: {stars} | Forks: {forks} | Contributors: {contributor_count}
Size: {size_kb} KB | Repo age: {repo_age_days} days | Last commit: {last_commit_days_ago} days ago
README: present ({readme_size_bytes} bytes) / missing
Tests: test folder present / no test folder
CI/CD: configured / not configured
Docker: Dockerfile present / absent
Dependencies: managed / no dependency file
Root file count: {root_file_count}
Status: original / fork | archived
```

**Justification:** A fixed template ensures every repository uses the same vocabulary and
signal ordering. This makes LLM classification more consistent (no free-form hallucination
about absent signals) and gives DistilBERT a predictable positional structure — keywords
like "CI/CD: configured" appear at the same offset across all inputs. Average summary
length is ~350 characters (~90 tokens), well within DistilBERT's 512-token limit.

---

## Prompt Design

The weak labeling step sends two-component prompts to DeepSeek-V3 (`deepseek-chat`):

**System prompt** — establishes recruiter persona:
> *"You are a senior software engineering recruiter with deep technical expertise.
> You assess GitHub repositories to determine the engineering maturity level
> of their authors, helping companies make faster and more accurate hiring decisions."*

**User prompt** — one per repository:
- Explicit definitions for all 6 categories with positive **and** negative examples
- Structured text summary appended as the "Repository profile"
- Required JSON output: `{"label": "...", "confidence": 0.0–1.0, "reasoning": "2–3 sentences"}`

**Key design decisions:**

| Decision | Rationale |
|---|---|
| `temperature=0.1` | Deterministic output; minimises label variance across runs |
| `response_format: json_object` | Forces parseable output; no free-text wrapping |
| Recruiter persona | Grounds the LLM in hiring context, not abstract code aesthetics |
| Negative examples per class | Reduces boundary confusion between adjacent levels |
| Confidence + reasoning fields | Labels with `"unknown"` or parse failures are discarded |
| Checkpoint every 10 repos | Resilient to network failures; avoids re-labeling from scratch |

Cost: **~$0.20** for 645 repositories (vs ~$3–5 with GPT-4o).

---

## Setup

```bash
# 1. Clone and install
git clone <repo-url>

# Install PyTorch with CUDA first (see Requisitos del entorno above)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt

# 2. Add API keys
cp .env.example .env
# Edit .env with your GITHUB_TOKEN and DEEPSEEK_API_KEY

# 3. Collect data
python src/github_collector.py
python src/github_collect_missing.py   # fill under-populated classes

# 4. Run pipeline (activate conda geo environment first)
python src/preprocessing.py
python src/summarization.py
python src/llm_labeling.py
python src/split.py
python src/baseline.py      # TF-IDF + LogisticRegression baseline
python src/train.py         # DistilBERT fine-tuning (requires CUDA)
python src/evaluation.py    # metrics, confusion matrix, error analysis

# 5. Launch app
streamlit run app.py
```

> **Note — Tab 4 (Interactive Prediction):** `model.safetensors` (255 MB) exceeds GitHub's file size limit and is not included in the repository. Tabs 1–3 work fully on a fresh clone. To enable BERT predictions in Tab 4, run `python src/train.py` (requires CUDA) after completing steps 3–4 above.

---

## Final Results

Evaluated on a held-out test set of **93 repositories** (stratified 15% split, unseen during training).

| Model | Test Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| TF-IDF + LogisticRegression (baseline) | 62.4% | 0.603 | 0.614 |
| DistilBERT fine-tuned | 52.7% | 0.424 | 0.468 |

The **baseline outperforms DistilBERT** on this dataset. With only 614 labeled samples
and a highly structured text template, keyword matching is highly effective. DistilBERT
would require significantly more data (~10× minimum) to leverage its contextual advantage.

**F1 per class — test set:**

| Class | Baseline F1 | DistilBERT F1 |
|---|---|---|
| `intern` | 0.00 | 0.00 |
| `junior` | 0.44 | 0.26 |
| `senior` | 0.80 | 0.68 |
| `lead_architect` | 0.72 | 0.57 |
| `template_boilerplate` | 0.62 | 0.47 |
| `low_value` | 0.56 | 0.38 |

**Key finding — sensitivity analysis:** Removing CI/CD signals from the text summary
drops baseline F1 macro by ~0.06; removing Stars/Forks drops it by ~0.04.
These two signal groups are the most discriminating for maturity classification.

---

## Limitations

### Methodological

- **Weak labels:** DeepSeek-V3 is a proxy annotator, not a human expert panel. Label noise
  propagates to the fine-tuned model and is unquantifiable without a gold-standard benchmark.
- **Small dataset:** 614 repositories is insufficient for DistilBERT to generalise; the
  structured text format makes TF-IDF surprisingly competitive at this scale.
- **Selection bias:** curated lists over-represent popular, English-language, open-source
  projects. Private-sector work and non-English repositories are entirely absent.
- **Class imbalance:** `senior` accounts for ~38% of data; `intern` for <5%. `WeightedTrainer`
  compensates during training but `intern` still achieves F1 = 0.00 on the test set (only
  3 test samples — too few to evaluate reliably).
- **Text-only model:** DistilBERT reads a structured text summary, not actual code. Metrics
  such as cyclomatic complexity, test coverage %, and code duplication are absent.
- **GitHub Search API exclusion:** dynamic discovery was abandoned due to secondary rate
  limits. Hardcoded curated lists reduce collection diversity and introduce selection bias.

### Ethical

- **Career impact:** false negatives (a strong engineer classified as `junior`) could
  unfairly screen out qualified candidates from consideration.
- **Language and cultural bias:** Western open-source conventions score systematically higher
  than equivalent work in other languages or private enterprise settings.
- **Stars as a proxy for skill:** star count rewards visibility and social network effects,
  not engineering quality. Engineers working on internal tooling are systematically penalised.
- **Transparency requirement:** candidates should be informed when automated repository
  analysis is used in their evaluation and retain the right to contest the outcome.

---

## Project Structure

```
├── app.py                    # Streamlit dashboard (4 tabs)
├── requirements.txt
├── .env.example              # API key template
├── src/
│   ├── github_collector.py   # GitHub API extraction
│   ├── github_collect_missing.py    # Supplement minority classes
│   ├── preprocessing.py      # Cleaning and transformations
│   ├── summarization.py      # Text representation for LLM/BERT
│   ├── llm_labeling.py       # Weak labeling with DeepSeek
│   ├── baseline.py           # TF-IDF + LogisticRegression baseline
│   ├── train.py              # DistilBERT fine-tuning (local GPU, fp16)
│   ├── evaluation.py         # Metrics, confusion matrix, error analysis
│   ├── visualization.py      # Plots for Streamlit
│   └── utils.py              # Shared helpers
├── data/
│   ├── raw/                  # Per-class CSVs + combined
│   ├── processed/            # Cleaned dataset
│   ├── labeled/              # LLM-labeled dataset
│   └── splits/               # Train / val / test
├── models/trained_models/    # Fine-tuned DistilBERT
├── output/
│   ├── figures/              # Confusion matrix, distributions
│   ├── tables/               # Metrics tables
│   └── metrics/              # JSON evaluation results
└── video/link.txt            # Presentation video link
```
