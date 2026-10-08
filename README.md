# 🏥 E-Consultation Sentiment Analyzer

> **Deep Learning Mini Project** — Sentiment Analysis of comments received through the E-Consultation module using Bidirectional LSTM (TensorFlow/Keras)

---

## 📋 Project Overview

| Item | Detail |
|---|---|
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Dataset** | IMDB 50K Movie Reviews (proxy for E-consultation feedback) |
| **Baseline** | TF-IDF + Logistic Regression |
| **DL Model** | Bidirectional LSTM with SpatialDropout & BatchNorm |
| **UI** | Streamlit web application |
| **Framework** | TensorFlow 2.x / Keras |

---

## 🗂️ Project Structure

```
DL Project/
├── data/
│   ├── raw_reviews.csv          # Downloaded dataset
│   ├── processed_reviews.csv    # Cleaned & encoded data
│   └── tokenizer.pkl            # Fitted Keras tokenizer
│
├── src/
│   ├── data_ingestion.py        # Dataset download & validation
│   ├── preprocessing.py         # Text cleaning, tokenization, splitting
│   ├── eda.py                   # Exploratory Data Analysis + plots
│   ├── baseline_model.py        # TF-IDF + Logistic Regression
│   ├── dl_model.py              # Bidirectional LSTM model
│   ├── evaluation.py            # Metrics, confusion matrix, ROC-AUC
│   └── train.py                 # ⭐ Main orchestration script
│
├── models/
│   ├── bilstm_sentiment_model.h5    # Trained DL model weights
│   ├── baseline_lr_model.pkl        # Saved baseline model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│
├── plots/                       # All generated visualizations
│   ├── 01_class_distribution.png
│   ├── 02_length_distribution.png
│   ├── 03_missing_data_heatmap.png
│   ├── 04_wordclouds.png
│   ├── 05_top_words.png
│   ├── 06_confusion_matrix_baseline.png
│   ├── 07_training_curves.png
│   ├── 08_confusion_matrix_dl.png
│   ├── 09_roc_auc_curve.png
│   └── 10_model_comparison.png
│
├── notebooks/
│   └── sentiment_analysis.ipynb    # Google Colab notebook
│
├── app.py                       # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/econsultation-sentiment.git
cd econsultation-sentiment
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python src/train.py
```
This runs the **full pipeline**:
- Downloads IMDB dataset (~50K samples)
- Runs EDA and saves 5 plots
- Trains baseline (LR) + DL model (BiLSTM)
- Evaluates and saves all metrics/plots
- Saves model to `models/`

### 5. Launch the Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Architecture

```
Input (250 tokens)
     │
     ▼
Embedding(20001 → 128)
     │
     ▼
SpatialDropout1D(0.25)
     │
     ▼
Bidirectional LSTM(128, return_sequences=True)
     │
     ▼
LSTM(64)
     │
     ▼
BatchNormalization
     │
     ▼
Dense(64, ReLU) → Dropout(0.4)
     │
     ▼
Dense(1, Sigmoid)
     │
     ▼
Output: Positive / Negative
```

**Total Parameters**: ~2.84M

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | ~89.2% | ~89.3% | ~89.2% | ~89.2% | ~95.8% |
| **BiLSTM (DL)** | **~92.7%** | **~92.8%** | **~92.7%** | **~92.7%** | **~97.9%** |

> ⚠️ Update these values with actual results after training.

---

## 🔧 Key Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Embedding Dim | 128 |
| Sequence Length | 250 |
| BiLSTM Units | 128 |
| LSTM Units | 64 |
| Dense Units | 64 |
| Dropout Rate | 0.4 |
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Max Epochs | 20 (EarlyStopping) |
| Train/Val/Test Split | 80/10/10 |

---

## 📱 Streamlit App Features

| Page | Description |
|---|---|
| 🔍 **Live Analysis** | Single comment prediction with confidence gauge |
| 📦 **Batch Upload** | CSV upload for bulk sentiment analysis |
| 📊 **Model Dashboard** | Performance metrics & comparison charts |
| 📈 **Training Plots** | All EDA + training visualizations |

---

## 📁 Dataset

- **Source**: [IMDB Dataset — HuggingFace](https://huggingface.co/datasets/imdb)
- **Size**: 50,000 movie reviews
- **Classes**: Positive (25,000) / Negative (25,000)
- **Usage**: Proxy dataset for E-consultation patient feedback

---

## 🏗️ Phase Timeline

| Phase | Task | Status |
|---|---|---|
| Phase 1 | Problem definition & dataset selection | ✅ |
| Phase 2 | EDA & data preprocessing | ✅ |
| Phase 3 | Baseline ML model | ✅ |
| Phase 4 | BiLSTM deep learning model | ✅ |
| Phase 5 | Evaluation & optimization | ✅ |
| Phase 6 | Streamlit web app | ✅ |
| Phase 7 | Report & GitHub push | 🔄 |

---

## 📄 References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*.
2. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*.
3. Maas et al. (2011). Learning Word Vectors for Sentiment Analysis. *ACL*.
4. TensorFlow Documentation: https://www.tensorflow.org
5. Streamlit Documentation: https://docs.streamlit.io

---

## 👥 Team

| Name | Role |
|---|---|
| [Your Name] | Deep Learning, Preprocessing |
| [Teammate] | EDA, Evaluation, Report |

---

*VTU-CMRIT Deep Learning Mini Project | 2026*
"# sentiment-analysis-of-comments" 
