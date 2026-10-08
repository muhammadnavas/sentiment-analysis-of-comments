"""
data_ingestion.py
-----------------
Handles dataset downloading, loading, and initial validation.
Dataset: IMDB 50K Movie Reviews (used as proxy for E-Consultation comments)
Source: Kaggle / HuggingFace datasets
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_FILE = os.path.join(DATA_DIR, "raw_reviews.csv")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_reviews.csv")


def download_dataset_from_huggingface() -> pd.DataFrame:
    """
    Downloads the IMDB dataset from HuggingFace Datasets as a proxy
    for E-Consultation user feedback/comments.

    Returns:
        pd.DataFrame: Raw dataframe with 'text' and 'label' columns.
    """
    try:
        from datasets import load_dataset
        print("[INFO] Loading IMDB dataset from HuggingFace...")
        dataset = load_dataset("imdb")
        train_df = pd.DataFrame(dataset["train"])
        test_df  = pd.DataFrame(dataset["test"])
        df = pd.concat([train_df, test_df], ignore_index=True)
        # Rename to match our schema
        df.rename(columns={"text": "comment", "label": "sentiment"}, inplace=True)
        # Map numeric labels to readable strings
        df["sentiment"] = df["sentiment"].map({0: "negative", 1: "positive"})
        print(f"[INFO] Dataset loaded: {len(df):,} samples.")
        return df
    except ImportError:
        print("[WARN] `datasets` library not found. Falling back to CSV load.")
        return load_from_csv()


def load_from_csv(filepath: str = RAW_FILE) -> pd.DataFrame:
    """
    Loads dataset from a local CSV file. The CSV must have at minimum
    a 'comment' and 'sentiment' (positive/negative) column.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Raw dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at: {filepath}\n"
            "Please either:\n"
            "  1. Run download_dataset_from_huggingface() to auto-download, OR\n"
            "  2. Place a CSV file with 'comment' and 'sentiment' columns at the above path."
        )
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} samples with columns: {list(df.columns)}")
    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the dataframe has required columns and drops rows
    with missing critical values.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Validated and cleaned dataframe.
    """
    required_cols = {"comment", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    initial_len = len(df)
    df = df.dropna(subset=["comment", "sentiment"])
    df = df[df["comment"].str.strip().str.len() > 0]
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with missing/empty values.")

    # Normalize sentiment labels
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    valid_sentiments = {"positive", "negative"}
    df = df[df["sentiment"].isin(valid_sentiments)]
    print(f"[INFO] Valid samples after validation: {len(df):,}")
    return df.reset_index(drop=True)


def save_raw_data(df: pd.DataFrame, filepath: str = RAW_FILE) -> None:
    """Saves validated raw data to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[INFO] Raw data saved to: {filepath}")


def load_data(use_huggingface: bool = True) -> pd.DataFrame:
    """
    Main entry point: loads, validates, and optionally saves the dataset.

    Args:
        use_huggingface: If True, attempts HuggingFace download first.

    Returns:
        pd.DataFrame: Clean, validated dataframe ready for EDA.
    """
    if os.path.exists(RAW_FILE):
        df = load_from_csv(RAW_FILE)
    elif use_huggingface:
        try:
            df = download_dataset_from_huggingface()
        except Exception as e:
            print(f"[WARN] HuggingFace download failed: {e}. Falling back to CSV load.")
            df = load_from_csv(RAW_FILE)
    else:
        df = load_from_csv(RAW_FILE)

    df = validate_dataframe(df)
    save_raw_data(df)
    return df


if __name__ == "__main__":
    df = load_data(use_huggingface=True)
    print(df.head())
    print(f"\nClass Distribution:\n{df['sentiment'].value_counts()}")
