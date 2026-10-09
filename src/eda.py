"""
eda.py
------
Exploratory Data Analysis for the E-Consultation Sentiment Dataset.
Generates and saves plots to the /plots directory.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
STYLE       = "seaborn-v0_8-darkgrid"
PALETTE     = {"positive": "#2ecc71", "negative": "#e74c3c"}
FIG_DPI     = 150

os.makedirs(PLOTS_DIR, exist_ok=True)
plt.style.use(STYLE)


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved: {path}")


# ─────────────────────────────────────────────
# 1. Class Distribution
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart showing count of positive vs negative comments."""
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[PALETTE.get(s, "#95a5a6") for s in counts.index],
        edgecolor="white",
        linewidth=1.5,
        width=0.4,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_title("Class Distribution — E-Consultation Sentiments", fontsize=13, pad=12)
    ax.set_xlabel("Sentiment Class", fontsize=11)
    ax.set_ylabel("Number of Comments", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, "01_class_distribution.png")


# ─────────────────────────────────────────────
# 2. Comment Length Distribution
# ─────────────────────────────────────────────

def plot_length_distribution(df: pd.DataFrame) -> None:
    """Histogram of comment character lengths by sentiment class."""
    df = df.copy()
    df["char_len"] = df["comment"].str.len()
    df["word_len"] = df["comment"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, label in zip(
        axes,
        ["char_len", "word_len"],
        ["Character Length", "Word Count"],
    ):
        for sentiment, color in PALETTE.items():
            subset = df[df["sentiment"] == sentiment][col]
            ax.hist(
                subset, bins=60, alpha=0.65, color=color,
                label=sentiment.capitalize(), edgecolor="none"
            )
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Distribution of {label}", fontsize=12)
        ax.legend()
        # Clip extreme tails for readability
        ax.set_xlim(0, df[col].quantile(0.98))

    fig.suptitle("Comment Length Analysis", fontsize=14, y=1.02)
    _save(fig, "02_length_distribution.png")


# ─────────────────────────────────────────────
# 3. Missing Data Heatmap
# ─────────────────────────────────────────────

def plot_missing_data(df: pd.DataFrame) -> None:
    """Heatmap showing missing values across all columns."""
    fig, ax = plt.subplots(figsize=(6, 3))
    missing = df.isnull().sum().to_frame(name="Missing Values")
    missing["% Missing"] = (missing["Missing Values"] / len(df) * 100).round(2)
    sns.heatmap(
        df.isnull(),
        cbar=False, yticklabels=False,
        cmap=["#2ecc71", "#e74c3c"],
        ax=ax
    )
    ax.set_title("Missing Data Map (Red = Missing)", fontsize=12)
    _save(fig, "03_missing_data_heatmap.png")
    print(f"[EDA] Missing Values:\n{missing}")


# ─────────────────────────────────────────────
# 4. Word Clouds
# ─────────────────────────────────────────────

def plot_wordclouds(df: pd.DataFrame) -> None:
    """Separate word clouds for positive and negative comments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sentiment, bg, colormap in zip(
        axes,
        ["positive", "negative"],
        ["#0d1117", "#0d1117"],
        ["Greens", "Reds"],
    ):
        text = " ".join(df[df["sentiment"] == sentiment]["comment"].fillna(""))
        wc = WordCloud(
            width=700, height=400,
            background_color=bg,
            colormap=colormap,
            max_words=150,
            collocations=False,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"{sentiment.capitalize()} Comments",
            fontsize=13, color="#f0f0f0",
            pad=8
        )
        ax.set_facecolor(bg)

    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Word Clouds — E-Consultation Sentiments", fontsize=15, color="white", y=1.01)
    _save(fig, "02_wordclouds.png")


# ─────────────────────────────────────────────
# 5. Top N Most Common Words
# ─────────────────────────────────────────────

def plot_top_words(df: pd.DataFrame, n: int = 20) -> None:
    """Bar charts of top-N most frequent words per sentiment class."""
    stop_words = {
        "the", "a", "an", "is", "in", "it", "of", "and", "to",
        "was", "for", "that", "this", "with", "on", "are", "at",
        "be", "by", "i", "he", "she", "they", "we", "you", "my",
        "br", "film", "movie", "one", "not", "but", "have", "had",
        "has", "as", "from", "or", "his", "her", "their", "its",
        "so", "do", "did", "no", "if", "up", "me", "all", "just",
        "been", "can", "about", "what", "when", "which", "who", "more",
        "out", "than", "would", "there", "will", "very", "into", "also",
        "some", "were", "him", "them", "its", "how", "after", "even"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, sentiment, color in zip(axes, ["positive", "negative"], ["#2ecc71", "#e74c3c"]):
        text = " ".join(df[df["sentiment"] == sentiment]["comment"].fillna(""))
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        words = [w for w in words if w not in stop_words]
        common = Counter(words).most_common(n)
        words_list, counts = zip(*common)

        ax.barh(
            list(reversed(words_list)),
            list(reversed(counts)),
            color=color, alpha=0.8, edgecolor="white"
        )
        ax.set_title(f"Top {n} Words — {sentiment.capitalize()}", fontsize=12)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Most Frequent Words by Sentiment Class", fontsize=14, y=1.02)
    _save(fig, "05_top_words.png")


# ─────────────────────────────────────────────
# 6. Summary Statistics Table
# ─────────────────────────────────────────────

def print_summary_stats(df: pd.DataFrame) -> None:
    """Prints a concise summary of the dataset."""
    df = df.copy()
    df["word_count"] = df["comment"].str.split().str.len()

    print("\n" + "=" * 55)
    print("   E-CONSULTATION SENTIMENT DATASET — EDA SUMMARY")
    print("=" * 55)
    print(f"  Total Samples       : {len(df):,}")
    print(f"  Positive Comments   : {(df['sentiment']=='positive').sum():,}")
    print(f"  Negative Comments   : {(df['sentiment']=='negative').sum():,}")
    print(f"  Avg Words/Comment   : {df['word_count'].mean():.1f}")
    print(f"  Max Words/Comment   : {df['word_count'].max():,}")
    print(f"  Min Words/Comment   : {df['word_count'].min():,}")
    print(f"  Duplicates          : {df.duplicated(subset=['comment']).sum():,}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
# Main EDA Runner
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """
    Runs the essential EDA suite and saves plots to /plots directory.

    Args:
        df: Validated raw dataframe with 'comment' and 'sentiment' columns.
    """
    print("[EDA] Starting Exploratory Data Analysis...\n")
    print_summary_stats(df)
    plot_class_distribution(df)
    print(f"\n[EDA] Essential plots saved to: {os.path.abspath(PLOTS_DIR)}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_data
    df = load_data()
    run_eda(df)
