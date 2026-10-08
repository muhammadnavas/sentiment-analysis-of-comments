"""
baseline_model.py
-----------------
Baseline ML model: TF-IDF + Logistic Regression
Provides the performance benchmark before deep learning.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR     = os.path.join(os.path.dirname(__file__), "..", "plots")
VECTORIZER_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_lr_model.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. TF-IDF Vectorization
# ─────────────────────────────────────────────

def build_tfidf_vectorizer(train_texts: pd.Series) -> TfidfVectorizer:
    """
    Builds and fits a TF-IDF vectorizer on training text.
    Uses unigrams + bigrams, top 20K features, and sublinear TF scaling.

    Args:
        train_texts: Series of cleaned training comments.

    Returns:
        TfidfVectorizer: Fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),       # Unigrams + bigrams
        sublinear_tf=True,        # Replace TF with log(TF) — helps with long docs
        strip_accents="unicode",
        analyzer="word",
        min_df=2,                 # Ignore very rare terms
    )
    vectorizer.fit(train_texts)
    print(f"[Baseline] TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    return vectorizer


def vectorize_splits(
    vectorizer: TfidfVectorizer,
    train_texts: pd.Series,
    val_texts: pd.Series,
    test_texts: pd.Series,
):
    """Transforms all three splits using the fitted vectorizer."""
    X_train = vectorizer.transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_val, X_test


# ─────────────────────────────────────────────
# 2. Logistic Regression Classifier
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Trains a Logistic Regression classifier.
    Uses L2 regularization with the lbfgs solver (handles sparse data well).

    Args:
        X_train: TF-IDF feature matrix (sparse).
        y_train: Binary label array.

    Returns:
        LogisticRegression: Trained model.
    """
    print("[Baseline] Training Logistic Regression...")
    model = LogisticRegression(
        C=1.0,           # Inverse regularization strength
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[Baseline] Training complete ✓")
    return model


# ─────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────

def evaluate_baseline(
    model: LogisticRegression,
    X_test,
    y_test: np.ndarray,
    split_name: str = "Test",
) -> dict:
    """
    Evaluates the baseline model and returns metrics.

    Args:
        model: Trained classifier.
        X_test: Feature matrix.
        y_test: True labels.
        split_name: Label for print output (e.g. 'Validation', 'Test').

    Returns:
        dict: Accuracy, F1, precision, recall.
    """
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[Baseline] ── {split_name} Evaluation ──")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return {"accuracy": acc, "f1": f1, "y_pred": y_pred}


# ─────────────────────────────────────────────
# 4. Confusion Matrix Plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Baseline (LR)",
    filename: str = "06_confusion_matrix_baseline.png",
) -> None:
    """Plots and saves a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)

    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Baseline] Confusion matrix saved: {path}")


# ─────────────────────────────────────────────
# 5. Save / Load Helpers
# ─────────────────────────────────────────────

def save_baseline_artifacts(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    """Saves the model and vectorizer to disk."""
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[Baseline] Model saved: {BASELINE_MODEL_PATH}")
    print(f"[Baseline] Vectorizer saved: {VECTORIZER_PATH}")


def load_baseline_artifacts():
    """Loads saved baseline model and vectorizer."""
    with open(BASELINE_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ─────────────────────────────────────────────
# 6. Full Baseline Pipeline
# ─────────────────────────────────────────────

def run_baseline(data: dict) -> dict:
    """
    End-to-end baseline pipeline.

    Args:
        data: Dict from preprocessing.run_preprocessing() containing
              train_df, val_df, test_df, y_train, y_val, y_test.

    Returns:
        dict: Evaluation results and predictions.
    """
    train_df = data["train_df"]
    val_df   = data["val_df"]
    test_df  = data["test_df"]

    # Vectorize
    vectorizer = build_tfidf_vectorizer(train_df["cleaned_comment"])
    X_train, X_val, X_test = vectorize_splits(
        vectorizer,
        train_df["cleaned_comment"],
        val_df["cleaned_comment"],
        test_df["cleaned_comment"],
    )

    # Train
    model = train_logistic_regression(X_train, data["y_train"])

    # Evaluate
    val_results  = evaluate_baseline(model, X_val,  data["y_val"],  "Validation")
    test_results = evaluate_baseline(model, X_test, data["y_test"], "Test")

    # Plot confusion matrix
    plot_confusion_matrix(data["y_test"], test_results["y_pred"])

    # Save
    save_baseline_artifacts(model, vectorizer)

    return {
        "val_accuracy":  val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
        "test_f1":       test_results["f1"],
        "model":         model,
        "vectorizer":    vectorizer,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_data
    from preprocessing import run_preprocessing

    df = load_data()
    data = run_preprocessing(df)
    results = run_baseline(data)
    print(f"\n[Baseline] Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"[Baseline] Final Test F1 Score : {results['test_f1']:.4f}")
