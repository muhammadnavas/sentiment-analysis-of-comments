"""
dl_model.py
-----------
Deep Learning model: Bidirectional LSTM with Embedding layer.
Built with TensorFlow / Keras.

Architecture:
  Embedding → SpatialDropout → BiLSTM → LSTM → Dense → Dropout → Output
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D,
    GlobalMaxPooling1D, Conv1D, MaxPooling1D, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "plots")
MODEL_PATH     = os.path.join(MODELS_DIR, "bilstm_sentiment_model.keras")
LOG_DIR        = os.path.join(os.path.dirname(__file__), "..", "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters
VOCAB_SIZE     = 20_000
EMBEDDING_DIM  = 128
MAX_SEQ_LEN    = 250
LSTM_UNITS_1   = 128
LSTM_UNITS_2   = 64
DENSE_UNITS    = 64
DROPOUT_RATE   = 0.4
SPATIAL_DROP   = 0.25
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 128
EPOCHS         = 20        # EarlyStopping will stop before this if val loss plateaus


# ─────────────────────────────────────────────
# 1. Model Architecture
# ─────────────────────────────────────────────

def build_bilstm_model(
    vocab_size: int     = VOCAB_SIZE,
    embedding_dim: int  = EMBEDDING_DIM,
    max_seq_len: int    = MAX_SEQ_LEN,
    lstm_units_1: int   = LSTM_UNITS_1,
    lstm_units_2: int   = LSTM_UNITS_2,
    dense_units: int    = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM model for binary sentiment classification.

    Architecture:
      1. Embedding(vocab_size, embedding_dim, input_length=max_seq_len)
      2. SpatialDropout1D — drops entire embedding dimensions (more effective than regular dropout)
      3. Bidirectional LSTM (return_sequences=True) — captures context from both directions
      4. LSTM — extracts the final temporal summary
      5. BatchNormalization — stabilizes training
      6. Dense(64, relu) + Dropout
      7. Dense(1, sigmoid) — binary output

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        embedding_dim: Dimensionality of word embeddings.
        max_seq_len: Padded sequence length.
        lstm_units_1: Units in the BiLSTM layer.
        lstm_units_2: Units in the second LSTM layer.
        dense_units: Units in the dense hidden layer.
        dropout_rate: Dropout fraction for regularization.
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential([
        # ── Layer 1: Word Embeddings ──────────────────────────────
        Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            name="word_embeddings",
        ),

        # ── Layer 2: Spatial Dropout on embeddings ────────────────
        SpatialDropout1D(SPATIAL_DROP, name="spatial_dropout"),

        # ── Layer 3: Bidirectional LSTM ───────────────────────────
        Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.2),
            name="bilstm_layer",
        ),

        # ── Layer 4: Global Max Pooling (extracts strongest signals) ──
        GlobalMaxPooling1D(name="global_max_pool"),

        # ── Layer 5: Dense hidden layer ───────────────────────────
        Dense(dense_units, activation="relu", name="dense_hidden"),
        Dropout(dropout_rate, name="dropout"),

        # ── Layer 6: Output ───────────────────────────────────────
        Dense(1, activation="sigmoid", name="output"),
    ], name="BiLSTM_Sentiment_Classifier")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_model_summary(model: tf.keras.Model, seq_len: int = MAX_SEQ_LEN) -> None:
    """Prints a formatted model summary."""
    print("\n" + "=" * 65)
    print("  BIDIRECTIONAL LSTM — MODEL ARCHITECTURE")
    print("=" * 65)
    # Build the model with a dummy input shape so count_params works
    model.build(input_shape=(None, seq_len))
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total Parameters: {total_params:,}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# 2. Callbacks
# ─────────────────────────────────────────────

def get_callbacks(model_path: str = MODEL_PATH) -> list:
    """
    Returns a list of Keras callbacks:
      - EarlyStopping: halts training if val_loss doesn't improve for 4 epochs
      - ModelCheckpoint: saves the best model (lowest val_loss)
      - ReduceLROnPlateau: reduces LR by 50% if val_loss plateaus for 3 epochs
      - TensorBoard: logs metrics for visualization

    Args:
        model_path: File path to save the best model weights.

    Returns:
        list: List of Keras callbacks.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
        ),
    ]


# ─────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    batch_size: int = BATCH_SIZE,
    epochs: int     = EPOCHS,
) -> tf.keras.callbacks.History:
    """
    Trains the model with the given data.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training sequences and labels.
        X_val, y_val: Validation sequences and labels.
        batch_size: Mini-batch size.
        epochs: Maximum number of epochs (EarlyStopping may stop earlier).

    Returns:
        History: Keras training history object.
    """
    callbacks = get_callbacks()

    print(f"[DL] Training BiLSTM — batch_size={batch_size}, max_epochs={epochs}")
    print(f"[DL] Train samples: {len(X_train):,} | Val samples: {len(X_val):,}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n[DL] Training complete. Best val_loss: {min(history.history['val_loss']):.4f}")
    return history


# ─────────────────────────────────────────────
# 4. Training Curves Plot
# ─────────────────────────────────────────────

def plot_training_curves(
    history: tf.keras.callbacks.History,
    filename: str = "02_training_curves.png",
) -> None:
    """
    Plots and saves training vs. validation loss and accuracy curves.

    Args:
        history: Keras History object returned by model.fit().
        filename: Output filename for the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs_range, history.history["loss"],     "b-o", label="Train Loss",      markersize=4)
    ax.plot(epochs_range, history.history["val_loss"], "r-o", label="Validation Loss", markersize=4)
    ax.fill_between(
        epochs_range,
        history.history["loss"],
        history.history["val_loss"],
        alpha=0.08, color="purple"
    )
    ax.set_title("Training vs. Validation Loss", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Accuracy ──
    ax = axes[1]
    ax.plot(epochs_range, history.history["accuracy"],     "b-s", label="Train Accuracy",      markersize=4)
    ax.plot(epochs_range, history.history["val_accuracy"], "r-s", label="Validation Accuracy", markersize=4)
    ax.fill_between(
        epochs_range,
        history.history["accuracy"],
        history.history["val_accuracy"],
        alpha=0.08, color="purple"
    )
    ax.set_title("Training vs. Validation Accuracy", fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.5, 1.01)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("BiLSTM Model — Learning Curves", fontsize=15, y=1.02)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DL] Training curves saved: {path}")


# ─────────────────────────────────────────────
# 5. Save / Load
# ─────────────────────────────────────────────

def save_model(model: tf.keras.Model, path: str = MODEL_PATH) -> None:
    """Saves the full Keras model (architecture + weights)."""
    model.save(path)
    print(f"[DL] Model saved: {path}")


def load_trained_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """Loads a previously saved Keras model."""
    model = load_model(path)
    print(f"[DL] Model loaded: {path}")
    return model


# ─────────────────────────────────────────────
# 6. Full DL Pipeline
# ─────────────────────────────────────────────

def run_dl_training(data: dict, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> dict:
    """
    End-to-end DL training pipeline.

    Args:
        data: Dict from preprocessing.run_preprocessing() with
              X_train, y_train, X_val, y_val, X_test, y_test.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        dict: Trained model and history.
    """
    model = build_bilstm_model()
    print_model_summary(model)

    history = train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        batch_size=batch_size,
        epochs=epochs,
    )

    plot_training_curves(history)
    save_model(model)

    return {"model": model, "history": history}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_data
    from preprocessing import run_preprocessing

    df   = load_data()
    data = run_preprocessing(df)
    result = run_dl_training(data)
