"""
preprocessing.py
----------------
Text preprocessing pipeline: cleaning, tokenization, padding,
and train/val/test splitting for sentiment analysis.
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
# Hyperparameters / Config
# ─────────────────────────────────────────────
MAX_VOCAB_SIZE  = 20_000   # Top N most frequent words to keep
MAX_SEQ_LEN     = 250      # Truncate/pad all sequences to this length
EMBEDDING_DIM   = 128      # Embedding dimension (used in DL model)
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
RANDOM_SEED     = 42

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
TOKENIZER_PATH  = os.path.join(DATA_DIR, "tokenizer.pkl")
PROCESSED_FILE  = os.path.join(DATA_DIR, "processed_reviews.csv")


# ─────────────────────────────────────────────
# 1. Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans a raw comment string:
      - Lowercases
      - Removes HTML tags (e.g., <br />)
      - Removes URLs
      - Removes non-alphabetic characters (keeps spaces)
      - Strips extra whitespace

    Args:
        text: Raw text string.

    Returns:
        str: Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)              # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)        # Remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # Collapse whitespace
    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning to the 'comment' column and adds a
    'cleaned_comment' column.

    Args:
        df: Dataframe with 'comment' column.

    Returns:
        pd.DataFrame: Dataframe with added 'cleaned_comment' column.
    """
    print("[INFO] Cleaning text...")
    df = df.copy()
    df["cleaned_comment"] = df["comment"].apply(clean_text)
    # Drop rows where cleaning results in empty strings
    df = df[df["cleaned_comment"].str.len() > 2].reset_index(drop=True)
    print(f"[INFO] Text cleaning complete. {len(df):,} samples remaining.")
    return df


# ─────────────────────────────────────────────
# 2. Label Encoding
# ─────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Converts sentiment labels to binary integers:
        positive → 1
        negative → 0

    Args:
        df: Dataframe with 'sentiment' column.

    Returns:
        Tuple[pd.DataFrame, dict]: Updated dataframe and label mapping dict.
    """
    label_map = {"negative": 0, "positive": 1}
    df = df.copy()
    df["label"] = df["sentiment"].map(label_map)
    print(f"[INFO] Labels encoded. Distribution:\n{df['label'].value_counts()}")
    return df, label_map


# ─────────────────────────────────────────────
# 3. Train / Val / Test Split
# ─────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train, validation, and test sets
    using stratified sampling to preserve class balance.

    Split ratio: 80% train / 10% val / 10% test

    Args:
        df: Full dataframe with 'cleaned_comment' and 'label' columns.

    Returns:
        Tuple: (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,   # 50% of remaining → 10% each
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )
    print(
        f"[INFO] Split → Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. Tokenization & Padding (for DL model)
# ─────────────────────────────────────────────

def build_tokenizer(train_texts: pd.Series) -> Tokenizer:
    """
    Fits a Keras Tokenizer on training text. Only the top
    MAX_VOCAB_SIZE words are kept; others are mapped to OOV token.

    Args:
        train_texts: Series of cleaned training comments.

    Returns:
        Tokenizer: Fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(train_texts)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"[INFO] Tokenizer built. Vocabulary size: {vocab_size:,}")
    return tokenizer


def texts_to_padded_sequences(
    texts: pd.Series, tokenizer: Tokenizer
) -> np.ndarray:
    """
    Converts text to integer sequences and pads/truncates them
    to MAX_SEQ_LEN.

    Args:
        texts: Series of cleaned text comments.
        tokenizer: Fitted Keras Tokenizer.

    Returns:
        np.ndarray: Padded integer sequence array of shape (N, MAX_SEQ_LEN).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    return padded


def save_tokenizer(tokenizer: Tokenizer, path: str = TOKENIZER_PATH) -> None:
    """Serializes the tokenizer to disk for inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Tokenizer saved to: {path}")


def load_tokenizer(path: str = TOKENIZER_PATH) -> Tokenizer:
    """Loads a previously saved tokenizer from disk."""
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"[INFO] Tokenizer loaded from: {path}")
    return tokenizer


# ─────────────────────────────────────────────
# 5. Main Pipeline
# ─────────────────────────────────────────────

def run_preprocessing(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Clean text
      2. Encode labels
      3. Split into train/val/test
      4. Tokenize and pad sequences
      5. Save tokenizer

    Args:
        df: Raw validated dataframe from data_ingestion.

    Returns:
        dict: Contains all split arrays + metadata needed for training.
    """
    df = apply_cleaning(df)
    df, label_map = encode_labels(df)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"[INFO] Processed data saved to: {PROCESSED_FILE}")

    train_df, val_df, test_df = split_data(df)

    # Build tokenizer on training data ONLY (no data leakage)
    tokenizer = build_tokenizer(train_df["cleaned_comment"])
    save_tokenizer(tokenizer)

    # Convert to padded sequences
    X_train = texts_to_padded_sequences(train_df["cleaned_comment"], tokenizer)
    X_val   = texts_to_padded_sequences(val_df["cleaned_comment"],   tokenizer)
    X_test  = texts_to_padded_sequences(test_df["cleaned_comment"],  tokenizer)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    print("[INFO] Preprocessing complete ✓")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "train_df": train_df,
        "val_df":   val_df,
        "test_df":  test_df,
    }


if __name__ == "__main__":
    from data_ingestion import load_data
    df = load_data()
    result = run_preprocessing(df)
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_val shape:   {result['X_val'].shape}")
    print(f"X_test shape:  {result['X_test'].shape}")
