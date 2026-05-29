"""Cleaning and transformations of raw repository data."""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUT_FILE      = PROCESSED_DIR / "repositories_clean.csv"

# Expected per-class checkpoint files from github_collector.py
CLASSES = ["lead_architect", "senior", "junior", "intern",
           "template_boilerplate", "low_value"]

BOOL_COLS = ["is_fork", "is_archived", "has_readme",
             "has_docker", "has_requirements", "has_tests_folder", "has_ci_cd"]

INT_COLS  = ["stars", "forks", "open_issues", "contributor_count",
             "size_kb", "readme_size_bytes", "root_file_count"]

# Columns where NaN is filled with the column median (time-based signals)
MEDIAN_FILL_COLS = ["last_commit_days_ago", "repo_age_days"]


def load_raw_csvs() -> pd.DataFrame:
    """Read all available per-class CSVs and concatenate them."""
    frames = []
    missing = []

    for label in CLASSES:
        path = RAW_DIR / f"{label}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            print(f"  loaded {label:25s} → {len(df):4d} rows")
        else:
            missing.append(label)
            print(f"  MISSING {label:23s} (collector may still be running)")

    if not frames:
        raise FileNotFoundError("No class CSVs found in data/raw/. Run github_collector.py first.")

    if missing:
        print(f"\n  Warning: {len(missing)} class(es) missing — {missing}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined)} rows, {combined['label'].nunique()} classes\n")
    return combined


def report_nulls(df: pd.DataFrame, stage: str) -> None:
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print(f"  [{stage}] No nulls found.")
    else:
        print(f"  [{stage}] Columns with nulls:")
        for col, count in null_counts.items():
            print(f"    {col:30s} {count:4d} ({count/len(df)*100:.1f}%)")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Null report BEFORE cleaning ===")
    report_nulls(df, "before")

    # --- Deduplication ---
    before = len(df)
    df = df.drop_duplicates(subset="repo_id")
    dropped = before - len(df)
    if dropped:
        print(f"\n  Dropped {dropped} duplicate repo_id rows")

    # --- String columns ---
    df["description"] = df["description"].fillna("").str.strip()
    df["language"]    = df["language"].fillna("unknown").str.strip()
    df["topics"]      = df["topics"].fillna("")
    df["full_name"]   = df["full_name"].fillna("")

    # --- Boolean columns ---
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # --- Integer columns ---
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # --- Median-fill time columns ---
    for col in MEDIAN_FILL_COLS:
        if col in df.columns:
            median = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(median).astype(int)
            if n_filled:
                print(f"  Filled {n_filled} nulls in '{col}' with median={median:.0f}")

    # --- Label column ---
    df["label"] = df["label"].str.strip()

    # --- Derived features ---
    # Ratio of open issues to forks (signals project activity relative to interest)
    df["issues_per_fork"] = (
        df["open_issues"] / df["forks"].replace(0, 1)
    ).round(4)

    # Log-scale stars to reduce skew from very popular repos
    df["log_stars"] = np.log1p(df["stars"]).round(4)

    print("\n=== Null report AFTER cleaning ===")
    report_nulls(df, "after")

    return df


LABELED_FILE = Path("data/labeled/repos_labeled.csv")

# Keywords that override signal rules and force template_boilerplate
_TEMPLATE_KW = {
    "template", "boilerplate", "starter", "scaffold",
    "cookiecutter", "skeleton", "generator", "archetype", "blueprint",
}


def _rule_based_label(row: pd.Series) -> str:
    """
    Signal-based fallback for repos where the LLM returned 'unknown'.

    Priority order matters: template keywords fire first (they are
    name-encoded and unambiguous), then size/community signals discriminate
    the remaining maturity levels.
    """
    searchable = " ".join([
        str(row.get("full_name", "")),
        str(row.get("description", "")),
        str(row.get("topics", "")),
    ]).lower()

    if any(kw in searchable for kw in _TEMPLATE_KW):
        return "template_boilerplate"

    stars        = int(row.get("stars", 0))
    contributors = int(row.get("contributor_count", 0))
    size_kb      = int(row.get("size_kb", 0))
    last_commit  = int(row.get("last_commit_days_ago", 9999))
    is_fork      = bool(row.get("is_fork", False))
    has_ci       = bool(row.get("has_ci_cd", False))
    has_tests    = bool(row.get("has_tests_folder", False))
    has_readme   = bool(row.get("has_readme", False))
    has_req      = bool(row.get("has_requirements", False))

    # low_value: no community signal + abandoned or trivially small
    if stars == 0 and (is_fork or last_commit > 730 or size_kb < 10):
        return "low_value"

    # lead_architect: massive popularity OR large collaborative codebase
    if stars > 5_000 or (stars > 1_000 and contributors > 50 and has_ci):
        return "lead_architect"

    # senior: solid engineering signals across all dimensions
    if (stars > 100 and has_ci and has_tests and has_readme) or \
       (stars > 50  and has_ci and has_readme and has_req):
        return "senior"

    # junior: some structure but not fully mature
    if stars > 5 and has_readme and not is_fork:
        return "junior"

    # intern: default fallback
    return "intern"


def resolve_unknown_labels(path: Path = LABELED_FILE) -> None:
    """
    Replace 'unknown' llm_labels in the labeled CSV using signal-based rules.
    Overwrites the file in place and prints a before/after summary.
    """
    if not path.exists():
        print(f"  {path} not found — skipping resolve step.")
        return

    df = pd.read_csv(path)
    mask = df["llm_label"] == "unknown"
    n_unknown = mask.sum()

    if n_unknown == 0:
        print("  No unknown labels found — nothing to resolve.")
        return

    print(f"  Resolving {n_unknown} unknown labels using signal-based rules …")

    assigned = df[mask].apply(_rule_based_label, axis=1)
    df.loc[mask, "llm_label"] = assigned

    # Mark these so the analyst knows they were rule-assigned, not LLM-assigned
    if "llm_confidence" in df.columns:
        df.loc[mask, "llm_confidence"] = 0.5
    if "llm_reasoning" in df.columns:
        df.loc[mask, "llm_reasoning"] = "Rule-based fallback (API call failed)."

    df.to_csv(path, index=False)

    print(f"  Resolved distribution:")
    print(assigned.value_counts().to_string())
    print(f"  Saved → {path}")
    print(f"  Remaining unknowns: {(df['llm_label'] == 'unknown').sum()}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading raw CSVs ===")
    df = load_raw_csvs()

    print("=== Cleaning ===")
    df = clean(df)

    print("\n=== Class distribution ===")
    print(df["label"].value_counts().to_string())

    print("\n=== Final shape ===")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")

    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")

    print("\n=== Resolving unknown LLM labels (if any) ===")
    resolve_unknown_labels()


if __name__ == "__main__":
    main()
