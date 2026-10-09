"""
train.py
--------
Main training script — orchestrates the full pipeline:
  1. Data Ingestion
  2. EDA
  3. Preprocessing
  4. Deep Learning (BiLSTM)
  5. Evaluation & Comparison

Run from project root:
  python src/train.py
"""

import os
import sys
import io
import time
import argparse
import pandas as pd

# Ensure UTF-8 output encoding for Windows command prompt / powershell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_ingestion import load_data
from preprocessing  import run_preprocessing
from eda            import run_eda
from dl_model       import run_dl_training
from evaluation     import evaluate_dl_model, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="E-Consultation Sentiment Analysis Training Pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of records to use (default: 10000 for balanced high accuracy & fast CPU training)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs for BiLSTM training (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()
    print("\n" + "🔬 " * 20)
    print("  E-CONSULTATION SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("🔬 " * 20 + "\n")

    # ────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ────────────────────────────────────────────────────
    print("━" * 60)
    print("STEP 1 / 5 : Data Ingestion")
    print("━" * 60)
    df = load_data(use_huggingface=False)

    if args.sample_size and args.sample_size < len(df):
        print(f"[INFO] Sampling {args.sample_size:,} balanced records for training...")
        half = args.sample_size // 2
        pos = df[df["sentiment"] == "positive"].sample(n=min(half, (df["sentiment"] == "positive").sum()), random_state=42)
        neg = df[df["sentiment"] == "negative"].sample(n=min(half, (df["sentiment"] == "negative").sum()), random_state=42)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled dataset ready: {len(df):,} samples.")

    # ────────────────────────────────────────────────────
    # STEP 2: EDA
    # ────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 2 / 5 : Exploratory Data Analysis")
    print("━" * 60)
    run_eda(df)

    # ────────────────────────────────────────────────────
    # STEP 3: Preprocessing
    # ────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 3 / 5 : Preprocessing")
    print("━" * 60)
    data = run_preprocessing(df)

    # ────────────────────────────────────────────────────
    # STEP 4: Deep Learning Model
    # ────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 4 / 4 : Deep Learning Model (Bidirectional LSTM)")
    print("━" * 60)
    dl_result = run_dl_training(data, epochs=args.epochs, batch_size=args.batch_size)
    dl_model   = dl_result["model"]

    # Full evaluation on test set
    dl_metrics = evaluate_dl_model(dl_model, data["X_test"], data["y_test"])

    elapsed = time.time() - total_start
    print(f"\n✅  Full pipeline complete in {elapsed/60:.1f} minutes.\n")
    print("┌─────────────────────────────────────────┐")
    print("│         FINAL RESULTS SUMMARY           │")
    print("├──────────────────────────┬──────────────┤")
    print(f"│ BiLSTM Accuracy          │  {dl_metrics['accuracy']*100:>6.2f}%    │")
    print(f"│ BiLSTM F1 Score          │  {dl_metrics['f1']:>8.4f}    │")
    if dl_metrics.get("roc_auc"):
        print(f"│ BiLSTM ROC-AUC           │  {dl_metrics['roc_auc']:>8.4f}    │")
    print("└──────────────────────────┴──────────────┘")
    print("\n📁 All plots saved to: ./plots/")
    print("💾 Trained model saved to: ./models/bilstm_sentiment_model.keras")
    print("🚀 To launch the web app: streamlit run app.py\n")


if __name__ == "__main__":
    main()
