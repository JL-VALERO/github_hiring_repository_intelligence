"""Fine-tune DistilBERT for 6-class repository maturity classification."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ── paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path("data/splits")
MODELS_DIR  = Path("models/trained_models")
RESULTS_DIR = Path("models/results")

# ── hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 256      # summaries average ~113 tokens; 256 leaves safe margin
BATCH_SIZE   = 8
EPOCHS       = 3
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED         = 42

# ── label registry ─────────────────────────────────────────────────────────────
LABELS   = ["intern", "junior", "senior", "lead_architect",
            "template_boilerplate", "low_value"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# ── weighted trainer ───────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to handle label imbalance."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── data loading ───────────────────────────────────────────────────────────────

def load_split(filename: str, tokenizer: DistilBertTokenizerFast) -> Dataset:
    df = pd.read_csv(SPLITS_DIR / filename)
    df = df.copy()
    df["label_id"] = df["llm_label"].map(LABEL2ID)

    before = len(df)
    df = df.dropna(subset=["label_id"])
    if len(df) < before:
        print(f"  Warning: dropped {before - len(df)} rows with unmapped labels in {filename}")

    df["label_id"] = df["label_id"].astype(int)

    ds = Dataset.from_dict({
        "text":  df["text_summary"].tolist(),
        "label": df["label_id"].tolist(),
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    ds = ds.map(tokenize, batched=True, desc=f"Tokenizing {filename}")
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_macro":    round(f1_score(labels, preds, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # ── device info ────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    device   = "cuda" if use_cuda else "cpu"
    print(f"\nDevice : {device.upper()}")
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available — training on CPU will be slow.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ──────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer  ({MODEL_NAME}) ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── datasets ───────────────────────────────────────────────────────────────
    print("Tokenizing splits ...")
    train_ds = load_split("train.csv", tokenizer)
    val_ds   = load_split("val.csv",   tokenizer)
    test_ds  = load_split("test.csv",  tokenizer)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── class weights ──────────────────────────────────────────────────────────
    train_df    = pd.read_csv(SPLITS_DIR / "train.csv")
    y_train     = train_df["llm_label"].map(LABEL2ID).astype(int).values
    raw_weights = compute_class_weight(
        "balanced", classes=np.arange(len(LABELS)), y=y_train
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)

    print("\nClass weights (balanced):")
    for label, w in zip(LABELS, raw_weights):
        print(f"  {label:25s}  {w:.4f}")

    # ── model ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model ({MODEL_NAME}) ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=use_cuda,                # only enable fp16 when CUDA is available
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=10,
        dataloader_num_workers=0,     # avoids Windows multiprocessing issues
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ── train ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Training ...")
    print("=" * 55)
    train_result = trainer.train()

    # ── save model ─────────────────────────────────────────────────────────────
    trainer.save_model(str(MODELS_DIR))
    tokenizer.save_pretrained(str(MODELS_DIR))

    label_map = {
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    with open(MODELS_DIR / "label2id.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved -> {MODELS_DIR}")

    # ── evaluate on test set ───────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # ── save results ───────────────────────────────────────────────────────────
    val_metrics = {
        k: v for k, v in trainer.state.log_history[-1].items()
        if k.startswith("eval_")
    }
    results = {
        "train": train_result.metrics,
        "val":   val_metrics,
        "test":  test_metrics,
    }
    with open(RESULTS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── summary ────────────────────────────────────────────────────────────────
    best_f1 = max(
        log["eval_f1_macro"]
        for log in trainer.state.log_history
        if "eval_f1_macro" in log
    )
    print("\n" + "=" * 55)
    print(f"Train loss  : {train_result.metrics['train_loss']:.4f}")
    print(f"Best val F1 (macro) : {best_f1:.4f}")
    print(f"Test accuracy       : {test_metrics.get('test_accuracy', 'n/a')}")
    print(f"Test F1 macro       : {test_metrics.get('test_f1_macro', 'n/a')}")
    print(f"Results saved -> {RESULTS_DIR}/training_results.json")
    print("=" * 55)


if __name__ == "__main__":
    main()
