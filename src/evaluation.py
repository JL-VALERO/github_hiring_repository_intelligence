"""Metrics, confusion matrix, error analysis, and sensitivity analysis."""

import json
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path("data/splits")
MODELS_DIR  = Path("models/trained_models")
RESULTS_DIR = Path("models/results")
FIGURES_DIR = Path("output/figures")
METRICS_DIR = Path("output/metrics")

LABELS = ["intern", "junior", "senior", "lead_architect",
          "template_boilerplate", "low_value"]
SHORT  = ["Intern", "Junior", "Senior", "Lead Arch.",
          "Template", "Low Value"]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# ── BERT inference ─────────────────────────────────────────────────────────────

def bert_predict(texts: list[str]) -> np.ndarray:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODELS_DIR))
    model     = DistilBertForSequenceClassification.from_pretrained(str(MODELS_DIR))
    model.to(device).eval()

    all_preds = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())

    return np.array(all_preds)


# ── metrics helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":     round(f1_score(y_true, y_pred, average="macro",
                                        labels=LABELS, zero_division=0), 4),
        "f1_weighted":  round(f1_score(y_true, y_pred, average="weighted",
                                        labels=LABELS, zero_division=0), 4),
        "f1_per_class": {
            label: round(s, 4)
            for label, s in zip(
                LABELS,
                f1_score(y_true, y_pred, average=None,
                         labels=LABELS, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


# ── figures ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=SHORT, yticklabels=SHORT,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_f1_comparison(bert_pc: dict, base_pc: dict, path: Path) -> None:
    x       = np.arange(len(LABELS))
    width   = 0.35
    bert_v  = [bert_pc.get(l, 0) for l in LABELS]
    base_v  = [base_pc.get(l, 0) for l in LABELS]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base_v, width, label="Baseline (TF-IDF + LR)", color="#6baed6")
    ax.bar(x + width / 2, bert_v, width, label="DistilBERT",             color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=15, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_title("F1 per class: Baseline vs DistilBERT (test set)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_sensitivity(results: dict, path: Path) -> None:
    variants = list(results.keys())
    f1s      = [results[v]["f1_macro"] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(variants, f1s, color=["#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"])
    ax.set_xlabel("F1 Macro (test)")
    ax.set_xlim(0, 0.85)
    ax.set_title("Sensitivity Analysis — signal removal (Baseline)", fontweight="bold")
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── sensitivity analysis ───────────────────────────────────────────────────────

def _remove_line(texts: list[str], pattern: str) -> list[str]:
    """Strip lines matching a regex pattern from every summary."""
    rx = re.compile(pattern)
    return [
        "\n".join(l for l in t.split("\n") if not rx.search(l))
        for t in texts
    ]


def sensitivity_analysis(
    X_train: list[str], y_train: list[str],
    X_test:  list[str], y_test:  list[str],
) -> dict:
    variants = {
        "All signals":          (X_train, X_test),
        "Without CI/CD":        (
            _remove_line(X_train, r"^CI/CD:"),
            _remove_line(X_test,  r"^CI/CD:"),
        ),
        "Without Stars/Forks":  (
            _remove_line(X_train, r"^Stars:"),
            _remove_line(X_test,  r"^Stars:"),
        ),
        "Without CI/CD + Stars": (
            _remove_line(_remove_line(X_train, r"^CI/CD:"), r"^Stars:"),
            _remove_line(_remove_line(X_test,  r"^CI/CD:"), r"^Stars:"),
        ),
    }

    results = {}
    for name, (Xtr, Xte) in variants.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                                       sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1_000, class_weight="balanced",
                                          C=1.0, solver="lbfgs", random_state=42)),
        ])
        pipe.fit(Xtr, y_train)
        preds = pipe.predict(Xte)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_macro": round(f1_score(y_test, preds, average="macro",
                                        labels=LABELS, zero_division=0), 4),
        }
        print(f"  {name:30s}  f1_macro={results[name]['f1_macro']:.4f}")

    return results


# ── error analysis ─────────────────────────────────────────────────────────────

def error_analysis(
    test_df: pd.DataFrame,
    bert_preds: list[str],
    base_preds: list[str],
) -> pd.DataFrame:
    df = test_df[["full_name", "llm_label", "stars",
                  "has_ci_cd", "has_tests_folder"]].copy()
    df["bert_pred"]     = bert_preds
    df["baseline_pred"] = base_preds
    df["bert_correct"]  = df["llm_label"] == df["bert_pred"]
    df["base_correct"]  = df["llm_label"] == df["baseline_pred"]

    errors = df[~df["bert_correct"]].copy()
    errors = errors.sort_values("full_name")
    return errors


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load splits ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

    X_train = train_df["text_summary"].tolist()
    y_train = train_df["llm_label"].tolist()
    X_test  = test_df["text_summary"].tolist()
    y_test  = test_df["llm_label"].tolist()

    # ── BERT predictions ───────────────────────────────────────────────────────
    print("\n[1/5] BERT predictions ...")
    bert_ids   = bert_predict(X_test)
    bert_preds = [ID2LABEL[i] for i in bert_ids]
    bert_m     = compute_metrics(y_test, bert_preds)

    print(f"  accuracy   : {bert_m['accuracy']:.4f}")
    print(f"  f1_macro   : {bert_m['f1_macro']:.4f}")
    print(f"  f1_weighted: {bert_m['f1_weighted']:.4f}")
    print("\n  Classification report (BERT):")
    print(classification_report(y_test, bert_preds, labels=LABELS, zero_division=0))

    # ── baseline predictions ───────────────────────────────────────────────────
    print("[2/5] Baseline predictions ...")
    with open(MODELS_DIR / "baseline_pipeline.pkl", "rb") as f:
        base_pipe = pickle.load(f)
    base_preds = base_pipe.predict(X_test).tolist()
    base_m     = compute_metrics(y_test, base_preds)

    print(f"  accuracy   : {base_m['accuracy']:.4f}")
    print(f"  f1_macro   : {base_m['f1_macro']:.4f}")

    # ── confusion matrices ─────────────────────────────────────────────────────
    print("\n[3/5] Plotting confusion matrices ...")
    plot_confusion_matrix(
        np.array(bert_m["confusion_matrix"]),
        "Confusion Matrix — DistilBERT (test set, normalised)",
        FIGURES_DIR / "confusion_matrix_bert.png",
    )
    plot_confusion_matrix(
        np.array(base_m["confusion_matrix"]),
        "Confusion Matrix — Baseline TF-IDF + LR (test set, normalised)",
        FIGURES_DIR / "confusion_matrix_baseline.png",
    )
    plot_f1_comparison(
        bert_m["f1_per_class"],
        base_m["f1_per_class"],
        FIGURES_DIR / "f1_comparison.png",
    )

    # ── sensitivity analysis ───────────────────────────────────────────────────
    print("\n[4/5] Sensitivity analysis ...")
    sensitivity = sensitivity_analysis(X_train, y_train, X_test, y_test)
    plot_sensitivity(sensitivity, FIGURES_DIR / "sensitivity_analysis.png")

    # ── error analysis ─────────────────────────────────────────────────────────
    print("\n[5/5] Error analysis (BERT misclassifications) ...")
    errors = error_analysis(test_df, bert_preds, base_preds)
    print(f"  BERT errors: {len(errors)}/{len(test_df)}")
    print(f"  Most common confusion pairs:")
    pairs = errors.groupby(["llm_label", "bert_pred"]).size().sort_values(ascending=False).head(6)
    print(pairs.to_string())
    errors.to_csv(METRICS_DIR / "bert_errors.csv", index=False)
    print(f"  Saved -> {METRICS_DIR}/bert_errors.csv")

    # ── save full report ───────────────────────────────────────────────────────
    report = {
        "bert": {
            **bert_m,
            "classification_report": classification_report(
                y_test, bert_preds, labels=LABELS, zero_division=0, output_dict=True
            ),
        },
        "baseline": {
            **base_m,
            "classification_report": classification_report(
                y_test, base_preds, labels=LABELS, zero_division=0, output_dict=True
            ),
        },
        "sensitivity_analysis": sensitivity,
        "n_test": len(y_test),
        "n_bert_errors": int(len(errors)),
    }
    with open(METRICS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report -> {METRICS_DIR}/evaluation_report.json")

    # ── final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"{'Metric':30s}  {'Baseline':>9}  {'BERT':>9}")
    print("-" * 52)
    print(f"{'Test accuracy':30s}  {base_m['accuracy']:>9.4f}  {bert_m['accuracy']:>9.4f}")
    print(f"{'Test F1 macro':30s}  {base_m['f1_macro']:>9.4f}  {bert_m['f1_macro']:>9.4f}")
    print(f"{'Test F1 weighted':30s}  {base_m['f1_weighted']:>9.4f}  {bert_m['f1_weighted']:>9.4f}")
    print("-" * 52)
    print(f"\nF1 per class:")
    for label in LABELS:
        bv = base_m["f1_per_class"][label]
        dv = bert_m["f1_per_class"][label]
        winner = "<<" if dv > bv else ("  " if dv == bv else ">>")
        print(f"  {label:25s}  {bv:.4f}  {dv:.4f}  {winner}")
    print("=" * 52)


if __name__ == "__main__":
    main()
