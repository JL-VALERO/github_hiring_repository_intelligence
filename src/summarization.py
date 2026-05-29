"""Repository summary generation for LLM input and BERT tokenization."""

import pandas as pd
from pathlib import Path

IN_FILE  = Path("data/processed/repositories_clean.csv")
OUT_FILE = Path("data/processed/repositories_summarized.csv")


def build_text_summary(row: pd.Series) -> str:
    """
    Build a structured plain-text profile of a repository.
    Compact enough to fit inside an LLM prompt; rich enough for BERT.
    """
    readme_info = (
        f"present ({row['readme_size_bytes']:,} bytes)"
        if row["has_readme"]
        else "missing"
    )
    topics = row["topics"] if str(row["topics"]).strip() else "none"

    return (
        f"Repository: {row['full_name']}\n"
        f"Description: \"{row['description']}\"\n"
        f"Language: {row['language']} | Topics: {topics}\n"
        f"Stars: {int(row['stars']):,} | Forks: {int(row['forks']):,} "
        f"| Contributors: {int(row['contributor_count'])}\n"
        f"Size: {int(row['size_kb']):,} KB "
        f"| Repo age: {int(row['repo_age_days'])} days "
        f"| Last commit: {int(row['last_commit_days_ago'])} days ago\n"
        f"README: {readme_info}\n"
        f"Tests: {'test folder present' if row['has_tests_folder'] else 'no test folder'}\n"
        f"CI/CD: {'configured' if row['has_ci_cd'] else 'not configured'}\n"
        f"Docker: {'Dockerfile present' if row['has_docker'] else 'absent'}\n"
        f"Dependencies: {'managed' if row['has_requirements'] else 'no dependency file'}\n"
        f"Root file count: {int(row['root_file_count'])}\n"
        f"Status: {'fork' if row['is_fork'] else 'original'}"
        f"{' | archived' if row['is_archived'] else ''}"
    )


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"{IN_FILE} not found. Run preprocessing.py first."
        )

    df = pd.read_csv(IN_FILE)
    print(f"Loaded {len(df)} repos from {IN_FILE}")

    df["text_summary"] = df.apply(build_text_summary, axis=1)

    # Approximate token count (1 token ≈ 4 chars) — useful sanity check for BERT's 512-token limit
    df["summary_char_len"]   = df["text_summary"].str.len()
    df["summary_approx_tok"] = (df["summary_char_len"] / 4).astype(int)

    over_512 = (df["summary_approx_tok"] > 512).sum()
    if over_512:
        print(f"  Warning: {over_512} summaries may exceed BERT's 512-token limit "
              f"(tokenizer will truncate automatically)")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} summarized repos → {OUT_FILE}")
    print(f"  avg chars/summary : {df['summary_char_len'].mean():.0f}")
    print(f"  avg approx tokens : {df['summary_approx_tok'].mean():.0f}")
    print(f"\nSample summary for first repo:\n{'-'*60}")
    print(df['text_summary'].iloc[0])


if __name__ == "__main__":
    main()
