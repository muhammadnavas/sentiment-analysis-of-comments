# 🏥 E-Consultation Sentiment Analyzer

> **Deep Learning Mini Project** — Sentiment Analysis of comments and public feedback received through the E-Consultation portal using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network built with **TensorFlow / Keras**.

---

## 📋 Project Overview

The **E-Consultation Sentiment Analyzer** automatically categorizes public consultation feedback and citizen comments into **Positive** or **Negative** sentiment. This assists policymakers and administrators in quickly gauging public perception on healthcare policies, regulations, and public consultations.

| Component | Specification |
|---|---|
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Domain** | Public Health / E-Consultation Feedback |
| **Architecture** | Bidirectional LSTM (BiLSTM) with Word Embeddings & Global Max Pooling |
| **Framework** | TensorFlow 2.17 / Keras |
| **Interface** | Streamlit Web Application (Dark Glassmorphic UI) |
| **Pipeline** | Ingestion → EDA → Preprocessing → BiLSTM Training → Evaluation |

---

## 🗂️ Project Directory Structure

```
DL Project/
├── data/
│   ├── raw_reviews.csv               # Raw dataset (50,400 balanced feedback comments)
│   ├── processed_reviews.csv         # Cleaned, lowercased, and encoded records
│   └── tokenizer.pkl                 # Serialized Keras Tokenizer (vocabulary map)
│
├── src/
│   ├── data_ingestion.py             # Dataset loading & balanced stratified sampling
│   ├── preprocessing.py              # Text cleaning, regex filtering, tokenization, sequence padding
│   ├── eda.py                        # Exploratory Data Analysis & class distribution visualization
│   ├── dl_model.py                   # BiLSTM model definition, callbacks & training logic
│   ├── evaluation.py                 # Evaluation metrics, confusion matrix & ROC-AUC curves
│   └── train.py                      # ⭐ End-to-end training pipeline orchestrator
│
├── models/
│   ├── bilstm_sentiment_model.keras  # Trained Bidirectional LSTM model weights
│   └── evaluation_metrics.json       # Measured test metrics evaluated on unseen test data
│
├── plots/                            # The 4 core visualizations
│   ├── 01_class_distribution.png     # Class balance distribution
│   ├── 02_training_curves.png        # Training vs. Validation Loss and Accuracy curves
│   ├── 03_confusion_matrix_dl.png    # Test Confusion Matrix with percentage annotations
│   └── 04_roc_auc_curve.png          # ROC curve & AUC area visualization
│
├── app.py                            # Streamlit interactive web application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🧠 Deep Learning Architecture

The sentiment classifier uses a **Bidirectional LSTM** network designed for sequential text analysis:

```
Input Padded Text Sequence (maxlen = 250)
                   │
                   ▼
     Embedding Layer (20,001 → 128)
                   │
                   ▼
        SpatialDropout1D (0.25)
                   │
                   ▼
  Bidirectional LSTM (64 units, return_sequences=True)
                   │
                   ▼
    GlobalMaxPooling1D (Salient Keyword Pooling)
                   │
                   ▼
         Dense Layer (64, ReLU)
                   │
                   ▼
             Dropout (0.40)
                   │
                   ▼
        Dense Output (1, Sigmoid)
                   │
                   ▼
      Predicted Probability [0.0 – 1.0]
      (Negative < 0.50  |  Positive ≥ 0.50)
```

### Key Architectural Choices:
1. **Word Embeddings (`dim=128`)**: Maps integer-encoded vocabulary tokens into dense semantic vector representations.
2. **SpatialDropout1D (`rate=0.25`)**: Drops entire 1D feature maps across the sequence to prevent co-adaptation of words.
3. **Bidirectional LSTM (`units=64`)**: Processes sequences in both forward and backward directions, capturing context from past and future tokens.
4. **Global Max Pooling 1D**: Extracts the most salient sentiment keywords across all timesteps, preventing vanishing gradients over long sequences and zero-padded tokens.
5. **Dense & Regularization**: 64-unit ReLU layer with 40% dropout prevents overfitting before the final sigmoid decision neuron.

---

## 📊 Evaluation & Visualizations

All metrics and plots are generated from real test data (no synthetic or mock fallbacks):

| Visual Artifact | File | Purpose |
|---|---|---|
| **Class Distribution** | `plots/01_class_distribution.png` | Verifies balanced positive vs. negative feedback representation |
| **Training Curves** | `plots/02_training_curves.png` | Tracks learning dynamics (Loss and Accuracy) across epochs |
| **Confusion Matrix** | `plots/03_confusion_matrix_dl.png` | Measures True Positives, True Negatives, False Positives & False Negatives |
| **ROC-AUC Curve** | `plots/04_roc_auc_curve.png` | Measures sensitivity vs. specificity across all decision thresholds |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 – 3.12
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/muhammadnavas/sentiment-analysis-of-comments.git
cd "DL Project"

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Training the Model

Run the end-to-end training pipeline directly:

```cmd
python src/train.py
```

Or customize the parameters:
```cmd
python src/train.py --sample-size 10000 --epochs 4 --batch-size 128
```

#### CLI Options:
- `--sample-size`: Number of balanced records to use (default: `10000`)
- `--epochs`: Number of training epochs (default: `4`)
- `--batch-size`: Mini-batch size (default: `128`)

---

### 4. Running the Web Application

Launch the Streamlit interactive dashboard:

```cmd
streamlit run app.py
```

Once started, open your browser at **`http://localhost:8501`**.

---

## 💻 Web App Features

The web interface provides 4 specialized modules:

1. **🔍 Live Analysis**: 
   - Analyze individual consultation comments in real-time.
   - Shows predicted sentiment, confidence gauge, and key sentiment keywords.
2. **📦 Batch Processing**:
   - Upload CSV files containing consultation comments.
   - Computes sentiments in bulk and allows exporting predictions as downloadable CSV.
3. **📊 Model Dashboard**:
   - Displays real evaluated test metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC).
   - Interactive metric bar charts and architectural layer specifications.
4. **📈 Training Plots**:
   - Displays all 4 training and evaluation plots saved in the `plots/` directory.

---

## 📦 Dependencies

- **TensorFlow** (`>= 2.17.0`)
- **Streamlit** (`>= 1.35.0`)
- **Pandas** (`>= 2.0.0`)
- **NumPy** (`>= 1.24.0`)
- **Scikit-Learn** (`>= 1.3.0`)
- **Matplotlib** (`>= 3.7.0`)
- **Seaborn** (`>= 0.12.0`)
- **Plotly** (`>= 5.15.0`)
- **Pillow** (`>= 9.5.0`)

---

## 👥 Authors & Academic Context

- **Project**: Deep Learning Mini Project
- **Topic**: Sentiment Analysis of Comments Received Through E-Consultation Module
- **Repository**: [muhammadnavas/sentiment-analysis-of-comments](https://github.com/muhammadnavas/sentiment-analysis-of-comments)
