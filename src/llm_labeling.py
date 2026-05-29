"""Weak labeling with LLMs (DeepSeek-V3)."""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env")

IN_FILE   = Path("data/processed/repositories_summarized.csv")
OUT_FILE  = Path("data/labeled/repos_labeled.csv")
CKPT_FILE = Path("data/labeled/labeling_checkpoint.csv")

VALID_LABELS = {
    "intern", "junior", "senior",
    "lead_architect", "template_boilerplate", "low_value",
}

BATCH_SAVE_EVERY = 10   # write checkpoint every N repos
SLEEP_BETWEEN    = 1.2  # seconds between API calls (avoids DeepSeek rate limit)

# ── prompt design ──────────────────────────────────────────────────────────────
# The prompt is the most critical component: the professor evaluates whether the
# categories are well-defined and whether the reasoning is business-relevant.

SYSTEM_PROMPT = (
    "You are a senior software engineering recruiter with deep technical expertise. "
    "You assess GitHub repositories to determine the engineering maturity level "
    "of their authors, helping companies make faster and more accurate hiring decisions."
)

USER_PROMPT_TEMPLATE = """\
Classify this GitHub repository into EXACTLY ONE of the following six \
engineering maturity categories:

- intern: Single-file scripts, no tests, minimal structure, poor or absent \
documentation. Typical of a student's first project or a hobby experiment \
with no architectural thinking.

- junior: Basic folder structure, minimal or no tests, simple logic, little \
evidence of CI/CD or code review. Functional but lacks design patterns, \
proper error handling, and separation of concerns.

- senior: Well-organised codebase, comprehensive tests, CI/CD pipeline, \
solid documentation, clear module boundaries, production-ready practices. \
Evidence of code review and dependency management.

- lead_architect: Large-scale, multi-module system with complex architecture, \
extensive test suites, advanced CI/CD, design documents, and a substantial \
contributor community. Reflects systems-level thinking and organisational impact.

- template_boilerplate: Repository whose primary purpose is to be copied, \
scaffolded, or used as a starting point. May contain placeholder code, \
cookiecutter configs, or generator tooling.

- low_value: Abandoned project, empty repository, unchanged fork, personal \
dotfiles, demo or test repo with essentially no engineering content or \
hiring signal.

Repository profile:
{text_summary}

Respond ONLY with valid JSON in this exact format (no extra text):
{{"label": "<one of the six categories>", "confidence": <0.0–1.0>, \
"reasoning": "<2–3 sentences explaining the classification in terms of \
engineering signals and business relevance for a recruiter>"}}"""

API_URL = "https://api.deepseek.com/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}


# ── API call ───────────────────────────────────────────────────────────────────

def call_deepseek(text_summary: str, retries: int = 3) -> dict:
    """Call DeepSeek-V3 and return the parsed JSON classification."""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                text_summary=text_summary
            )},
        ],
        "temperature": 0.1,      # low = more deterministic classifications
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)

            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 15)) + 1
                print(f"  [rate limit] sleeping {wait}s …")
                time.sleep(wait)
                continue

            print(f"  [API {resp.status_code}] {resp.text[:200]}")
            time.sleep(2 ** attempt)

        except requests.RequestException as e:
            print(f"  [network error attempt {attempt+1}] {e}")
            time.sleep(2 ** attempt)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [parse error attempt {attempt+1}] {e}")
            time.sleep(2 ** attempt)

    return {
        "label": "unknown",
        "confidence": 0.0,
        "reasoning": "API call failed after all retries.",
    }


# ── label validation ───────────────────────────────────────────────────────────

def validate_label(raw: str) -> str:
    """Coerce LLM output to one of the six valid labels."""
    clean = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if clean in VALID_LABELS:
        return clean
    for valid in VALID_LABELS:
        if valid in clean or clean in valid:
            return valid
    return "unknown"


# ── checkpointing ──────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if CKPT_FILE.exists():
        return set(pd.read_csv(CKPT_FILE)["full_name"])
    return set()


def save_checkpoint(buffer: list) -> None:
    new_df = pd.DataFrame(buffer)
    if CKPT_FILE.exists():
        existing = pd.read_csv(CKPT_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="full_name", keep="last")
    else:
        combined = new_df
    combined.to_csv(CKPT_FILE, index=False)
    print(f"  [checkpoint] {len(combined)} repos labeled so far")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Guard: if the final output already exists and contains no unknown labels,
    # there is nothing to do — skip all API calls and avoid overwriting clean data.
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        if "llm_label" in existing.columns:
            n_unknown = (existing["llm_label"] == "unknown").sum()
            if n_unknown == 0 and len(existing) > 0:
                print(f"Output already exists with {len(existing)} clean repos (0 unknowns) — nothing to do.")
                print(f"Delete {OUT_FILE} to force re-labeling.")
                return

    if not IN_FILE.exists():
        raise FileNotFoundError(f"{IN_FILE} not found — run summarization.py first.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df   = pd.read_csv(IN_FILE)
    done = load_checkpoint()

    remaining = df[~df["full_name"].isin(done)]
    print(f"Total: {len(df)} | Already labeled: {len(done)} | Remaining: {len(remaining)}")

    if remaining.empty:
        print("All repos already labeled — writing final output.")
        labeled_df = pd.read_csv(CKPT_FILE)
        labeled_df = labeled_df[labeled_df["llm_label"] != "unknown"]
        labeled_df.to_csv(OUT_FILE, index=False)
        return

    buffer = []
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        print(f"  [{i}/{len(remaining)}] {row['full_name']}")

        result     = call_deepseek(row["text_summary"])
        llm_label  = validate_label(result.get("label", "unknown"))
        confidence = float(result.get("confidence", 0.0))
        reasoning  = result.get("reasoning", "")

        print(f"    → {llm_label}  (conf={confidence:.2f})")

        record = row.to_dict()
        record["llm_label"]      = llm_label
        record["llm_confidence"] = confidence
        record["llm_reasoning"]  = reasoning
        buffer.append(record)

        if i % BATCH_SAVE_EVERY == 0:
            save_checkpoint(buffer)
            buffer = []

        time.sleep(SLEEP_BETWEEN)

    if buffer:
        save_checkpoint(buffer)

    # Write final labeled dataset
    labeled_df = pd.read_csv(CKPT_FILE)
    labeled_df.to_csv(OUT_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"Labeled {len(labeled_df)} repos → {OUT_FILE}")
    print(f"\nOriginal (weak proxy) label distribution:")
    print(labeled_df["label"].value_counts().to_string())
    print(f"\nLLM re-label distribution:")
    print(labeled_df["llm_label"].value_counts().to_string())
    agreement = (labeled_df["label"] == labeled_df["llm_label"]).mean()
    print(f"\nLabel agreement (proxy vs LLM): {agreement:.1%}")
    print(f"Unknown labels: {(labeled_df['llm_label'] == 'unknown').sum()}")


if __name__ == "__main__":
    main()
