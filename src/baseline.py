"""Baseline classifier: TF-IDF + LogisticRegression for comparison against DistilBERT."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path("data/splits")
MODELS_DIR  = Path("models/trained_models")
RESULTS_DIR = Path("models/results")

LABELS = ["intern", "junior", "senior", "lead_architect",
          "template_boilerplate", "low_value"]

PIPELINE_PATH = MODELS_DIR / "baseline_pipeline.pkl"
RESULTS_PATH  = RESULTS_DIR / "baseline_results.json"


def load_split(filename: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(SPLITS_DIR / filename)
    return df["text_summary"].tolist(), df["llm_label"].tolist()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits ...")
    X_train, y_train = load_split("train.csv")
    X_val,   y_val   = load_split("val.csv")
    X_test,  y_test  = load_split("test.csv")
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # ── pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("\nTraining TF-IDF + LogisticRegression ...")
    pipeline.fit(X_train, y_train)

    # ── evaluate on all splits ─────────────────────────────────────────────────
    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                               ("val",   X_val,   y_val),
                               ("test",  X_test,  y_test)]:
        preds = pipeline.predict(X)
        acc        = round(accuracy_score(y, preds), 4)
        f1_macro   = round(f1_score(y, preds, average="macro",    labels=LABELS, zero_division=0), 4)
        f1_weighted= round(f1_score(y, preds, average="weighted", labels=LABELS, zero_division=0), 4)
        f1_per_class = {
            label: round(score, 4)
            for label, score in zip(
                LABELS,
                f1_score(y, preds, average=None, labels=LABELS, zero_division=0)
            )
        }
        results[split_name] = {
            "accuracy":     acc,
            "f1_macro":     f1_macro,
            "f1_weighted":  f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": confusion_matrix(y, preds, labels=LABELS).tolist(),
        }

        print(f"\n  [{split_name}]")
        print(f"    accuracy   : {acc:.4f}")
        print(f"    f1_macro   : {f1_macro:.4f}")
        print(f"    f1_weighted: {f1_weighted:.4f}")

    # ── classification report (test) ───────────────────────────────────────────
    test_preds = pipeline.predict(X_test)
    report = classification_report(y_test, test_preds, labels=LABELS, zero_division=0)
    print(f"\nClassification report (test):\n{report}")
    results["classification_report_test"] = report

    # ── save ───────────────────────────────────────────────────────────────────
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved -> {PIPELINE_PATH}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved  -> {RESULTS_PATH}")

    # ── summary vs BERT ────────────────────────────────────────────────────────
    bert_path = RESULTS_DIR / "training_results.json"
    if bert_path.exists():
        bert = json.load(open(bert_path))
        bert_f1  = bert.get("test", {}).get("test_f1_macro", "n/a")
        bert_acc = bert.get("test", {}).get("test_accuracy", "n/a")
        bl_f1    = results["test"]["f1_macro"]
        bl_acc   = results["test"]["accuracy"]

        print("\n" + "=" * 45)
        print(f"{'':25s}  {'Baseline':>8}  {'BERT':>8}")
        print(f"{'Test accuracy':25s}  {bl_acc:>8.4f}  {bert_acc if bert_acc == 'n/a' else f'{bert_acc:>8.4f}'}")
        print(f"{'Test F1 macro':25s}  {bl_f1:>8.4f}  {bert_f1 if bert_f1 == 'n/a' else f'{bert_f1:>8.4f}'}")
        print("=" * 45)


if __name__ == "__main__":
    main()
