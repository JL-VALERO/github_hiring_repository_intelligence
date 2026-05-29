"""Stratified 70/15/15 split of labeled data into train/val/test sets."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_FILE   = Path("data/labeled/repos_labeled.csv")
SPLIT_DIR = Path("data/splits")


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"{IN_FILE} not found — run llm_labeling.py first.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_FILE)
    print(f"Loaded {len(df)} repos | {df['llm_label'].nunique()} classes")
    print("\nClass distribution:")
    print(df["llm_label"].value_counts().to_string())

    # 70% train, 30% temp — then split temp into 50/50 → 15% val / 15% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["llm_label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["llm_label"],
    )

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR  / "val.csv",   index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv",  index=False)

    print(f"\nSplit sizes:")
    print(f"  train : {len(train_df):4d}  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  val   : {len(val_df):4d}  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  test  : {len(test_df):4d}  ({len(test_df)/len(df)*100:.1f}%)")

    print("\nClass distribution per split:")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  [{name}]")
        print(split["llm_label"].value_counts().to_string())

    print(f"\nSaved -> {SPLIT_DIR}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
