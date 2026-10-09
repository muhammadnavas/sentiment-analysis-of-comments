"""
data_ingestion.py
-----------------
Loads dataset for E-Consultation Sentiment Analysis.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "raw_reviews.csv")


def load_data(filepath: str = RAW_CSV_PATH, use_huggingface: bool = False) -> pd.DataFrame:
    """
    Loads raw review data from local CSV.
    """
    if not os.path.exists(filepath):
        # Fallback relative to project root
        alt_path = os.path.join("data", "raw_reviews.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"[DATA] Loading dataset from: {os.path.abspath(filepath)}")
    df = pd.read_csv(filepath)
    print(f"[DATA] Successfully loaded {len(df):,} total records.")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nSentiment value counts:")
    print(df["sentiment"].value_counts())
