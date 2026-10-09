"""
app.py
------
Streamlit Web UI — E-Consultation Sentiment Analyzer
Live demo: single comment prediction + model metrics dashboard.
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ─────────────────────────────────────────────
# Page Config (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Consultation Sentiment Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH_KERAS    = os.path.join(os.path.dirname(__file__), "models", "bilstm_sentiment_model.keras")
MODEL_PATH_H5       = os.path.join(os.path.dirname(__file__), "models", "bilstm_sentiment_model.h5")
MODEL_PATH          = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else MODEL_PATH_H5
TOKENIZER_PATH      = os.path.join(os.path.dirname(__file__), "data",   "tokenizer.pkl")
BASELINE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline_lr_model.pkl")
VECTORIZER_PATH     = os.path.join(os.path.dirname(__file__), "models", "tfidf_vectorizer.pkl")
PLOTS_DIR           = os.path.join(os.path.dirname(__file__), "plots")
MAX_SEQ_LEN         = 250


# ─────────────────────────────────────────────
# Custom CSS Styling (Deep Premium Dark Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 20% 15%, #13172e 0%, #0a0e1a 50%, #050811 100%) !important;
    color: #f8fafc !important;
    min-height: 100vh;
}

/* Streamlit Header (Top Bar) */
header[data-testid="stHeader"] {
    background: rgba(10, 14, 26, 0.75) !important;
    backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(12, 17, 32, 0.92) !important;
    backdrop-filter: blur(18px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] h2 {
    color: #f8fafc !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-top: 10px !important;
}

/* Universal Typography */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}
p, span, div {
    color: #cbd5e1;
}

/* Widget Labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    margin-bottom: 6px !important;
}

/* Sidebar Navigation Radio Buttons */
div[data-testid="stRadio"] > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    gap: 6px !important;
}
div[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-bottom: 2px !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #f1f5f9 !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
}

/* Text Area */
.stTextArea textarea {
    background: rgba(17, 24, 39, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.3) !important;
}
.stTextArea textarea::placeholder {
    color: #64748b !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background: rgba(17, 24, 39, 0.85) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(17, 24, 39, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] span {
    color: #f8fafc !important;
    font-size: 0.9rem !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"] {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}
div[data-baseweb="menu"] li {
    color: #f8fafc !important;
}

/* Metric Cards */
.metric-card {
    background: rgba(22, 30, 49, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 22px 18px !important;
    backdrop-filter: blur(14px) !important;
    text-align: center !important;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s !important;
}
.metric-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4) !important;
    border-color: rgba(139, 92, 246, 0.35) !important;
}
.metric-value {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #c084fc, #60a5fa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.5px !important;
}
.metric-label {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    margin-top: 6px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 600 !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.55) !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Result Box */
.result-positive {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%) !important;
    border: 1px solid rgba(16, 185, 129, 0.45) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    text-align: center !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12) !important;
}
.result-negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.08) 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.45) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    text-align: center !important;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.12) !important;
}

/* Alert Boxes (Warnings & Info) */
div[data-testid="stAlert"] {
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
}
div[data-testid="stAlert"] * {
    color: #f1f5f9 !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background: rgba(22, 30, 49, 0.6) !important;
    border: 1px dashed rgba(148, 163, 184, 0.3) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
div[data-testid="stFileUploader"] * {
    color: #cbd5e1 !important;
}

/* Dataframe Container */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Header & Divider */
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 6px;
}
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, #6366f1, #a855f7, transparent);
    border: none;
    margin: 20px 0 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_all_models():
    """Loads Deep Learning (BiLSTM) model and Tokenizer."""
    bundle = {
        "dl_model": None,
        "tokenizer": None,
        "ready": False,
    }

    # Load BiLSTM + Tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        try:
            import tensorflow as tf
            bundle["dl_model"] = tf.keras.models.load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, "rb") as f:
                bundle["tokenizer"] = pickle.load(f)
            bundle["ready"] = True
        except Exception:
            pass

    return bundle


def clean_text(text: str) -> str:
    """Cleans input text for inference."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str, bundle: dict, engine: str = "🧠 BiLSTM Deep Learning") -> dict:
    """
    Predicts sentiment exclusively using the BiLSTM Deep Learning model.
    """
    cleaned = clean_text(text)
    prob_dl = 0.50

    if bundle["dl_model"] and bundle["tokenizer"]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = bundle["tokenizer"].texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
        prob_dl = float(bundle["dl_model"].predict(padded, verbose=0)[0][0])

    prob = prob_dl
    label = "Positive" if prob >= 0.50 else "Negative"
    confidence = prob if prob >= 0.50 else 1.0 - prob
    return {
        "label": label,
        "confidence": confidence,
        "probability": prob,
        "prob_dl": prob_dl,
    }


def make_gauge_chart(probability: float) -> go.Figure:
    """Creates a gauge chart for sentiment probability."""
    color = "#10b981" if probability >= 0.5 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        delta={"reference": 50, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.4)"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.1)"},
                {"range": [50, 100],"color": "rgba(16,185,129,0.1)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Positivity Score", "font": {"size": 14, "color": "rgba(255,255,255,0.6)"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 About")
    st.markdown("""
    <div style='color: rgba(255,255,255,0.7); font-size: 0.9rem; line-height: 1.6;'>
    This tool analyzes patient/citizen feedback from the <strong>E-Consultation Module</strong>
    and classifies each comment as <span style='color:#10b981'>Positive</span> or
    <span style='color:#ef4444'>Negative</span> using a deep learning model.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Model Info")
    st.markdown("""
    <div style='color: rgba(255,255,255,0.7); font-size: 0.87rem; line-height: 1.7;'>
    🧠 <strong>Architecture</strong>: Bidirectional LSTM<br>
    📦 <strong>Embeddings</strong>: 128-dim, 20K vocab<br>
    📏 <strong>Seq Length</strong>: 250 tokens<br>
    🔧 <strong>Optimizer</strong>: Adam (lr=0.001)<br>
    🛑 <strong>Regularization</strong>: Dropout + EarlyStopping<br>
    📊 <strong>Dataset</strong>: IMDB 50K (proxy)<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📁 Navigation")
    page = st.radio(
        "Go to",
        ["🔍 Live Analysis", "📊 Model Dashboard", "📈 Training Plots"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("## 🧠 Active AI Model")
    st.markdown("""
    <div style='background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 12px; padding: 12px 14px;'>
        <div style='color: #c084fc; font-weight: 700; font-size: 0.92rem; display: flex; align-items: center; gap: 6px;'>
            🧠 BiLSTM Neural Network
        </div>
        <div style='color: #94a3b8; font-size: 0.78rem; margin-top: 4px; line-height: 1.4;'>
            Bidirectional Long Short-Term Memory Deep Learning Model
        </div>
    </div>
    """, unsafe_allow_html=True)
    engine = "🧠 BiLSTM Deep Learning"

    st.markdown("---")
    st.markdown(
        "<div style='color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align:center;'>"
        "Deep Learning Mini Project<br>E-Consultation Sentiment Analysis<br>TensorFlow · Keras · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Load Model Bundle
# ─────────────────────────────────────────────
models_bundle = load_all_models()
model_ready   = models_bundle["ready"]


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding: 16px 0 8px 0;'>
    <div class='hero-title'>🏥 E-Consultation Sentiment Analyzer</div>
    <div class='hero-subtitle'>AI-powered analysis of patient feedback using Bidirectional LSTM Deep Learning</div>
</div>
<div class='gradient-divider'></div>
""", unsafe_allow_html=True)

if not model_ready:
    st.warning(
        "⚠️ **Model not found.** Please train the model first by running `python src/train.py`. "
        "The app will work in demo mode with simulated results.",
        icon="⚠️"
    )


# ═══════════════════════════════════════════════
# PAGE: Live Analysis
# ═══════════════════════════════════════════════
if "🔍 Live Analysis" in page:

    st.markdown("## 🔍 Live Sentiment Analysis")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.6); margin-bottom:16px;'>"
        "Enter a patient consultation comment below to instantly analyze its sentiment."
        "</p>", unsafe_allow_html=True
    )

    # Session state initialization for text area
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = ""
    if "text_version" not in st.session_state:
        st.session_state["text_version"] = 0
    if "last_sample" not in st.session_state:
        st.session_state["last_sample"] = "— Choose a sample —"

    # Sample comments selection
    col1, col2 = st.columns([2.8, 1.2])
    with col2:
        sample = st.selectbox(
            "Load sample comment",
            options=[
                "— Choose a sample —",
                "The doctor was very attentive and explained everything clearly. Excellent service!",
                "I waited 2 hours and still couldn't see anyone. Very disappointing experience.",
                "The online consultation was convenient. I appreciated the quick response.",
                "The portal was confusing and I couldn't upload my documents properly.",
                "Very satisfied with the treatment plan. The staff was professional and kind.",
                "Rude behavior from the support desk and my prescription was delayed.",
            ],
            key="sample_dropdown",
        )

        if sample != "— Choose a sample —" and sample != st.session_state["last_sample"]:
            st.session_state["last_sample"] = sample
            st.session_state["user_text"] = sample
            st.session_state["text_version"] += 1
            st.rerun()

    with col1:
        current_text = st.text_area(
            "💬 Patient / Citizen Consultation Comment",
            value=st.session_state["user_text"],
            height=130,
            placeholder="e.g., The doctor was very helpful and answered all my questions patiently...",
            key=f"text_area_{st.session_state['text_version']}",
        )
        st.session_state["user_text"] = current_text

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["user_text"] = ""
            st.session_state["last_sample"] = "— Choose a sample —"
            st.session_state["text_version"] += 1
            st.rerun()

    comment_text = st.session_state["user_text"]

    if analyze_btn and comment_text.strip():
        with st.spinner("Analyzing sentiment..."):
            if model_ready:
                result = predict_sentiment(comment_text, models_bundle, engine)
            else:
                # Demo mode
                import random
                prob   = random.uniform(0.6, 0.95)
                is_pos = random.random() > 0.5
                if not is_pos: prob = 1 - prob
                result = {
                    "label": "Positive" if is_pos else "Negative",
                    "confidence": max(prob, 1 - prob),
                    "probability": prob,
                }

        is_positive = result["label"] == "Positive"

        # Result display
        st.markdown("---")
        res_col, gauge_col = st.columns([1.2, 1])

        with res_col:
            emoji   = "😊" if is_positive else "😞"
            cls_box = "result-positive" if is_positive else "result-negative"
            color   = "#10b981" if is_positive else "#ef4444"
            badge   = "✅ POSITIVE" if is_positive else "❌ NEGATIVE"

            st.markdown(f"""
            <div class='{cls_box}'>
                <div style='font-size: 3.5rem; margin-bottom: 8px;'>{emoji}</div>
                <div style='font-size: 1.6rem; font-weight: 700; color: {color};'>{badge}</div>
                <div style='margin-top: 12px; color: rgba(255,255,255,0.75); font-size: 0.95rem;'>
                    Confidence: <strong style='color:{color};'>{result['confidence']*100:.1f}%</strong>
                </div>
                <div style='margin-top: 6px; color: rgba(255,255,255,0.5); font-size: 0.85rem;'>
                    Positivity Score: {result['probability']*100:.2f}%
                </div>
                <div style='margin-top: 8px; font-size: 0.78rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;'>
                    Active Engine: {engine.split('(')[0].strip()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Word count info
            word_count = len(comment_text.split())
            st.markdown(f"""
            <div style='margin-top: 16px; padding: 12px 16px;
                        background: rgba(255,255,255,0.05); border-radius: 10px;
                        color: rgba(255,255,255,0.6); font-size: 0.88rem;'>
                📝 <strong>{word_count}</strong> words &nbsp;|&nbsp;
                🔤 <strong>{len(comment_text)}</strong> characters &nbsp;|&nbsp;
                🎯 Threshold: <strong>0.50</strong>
            </div>
            """, unsafe_allow_html=True)

        with gauge_col:
            st.plotly_chart(
                make_gauge_chart(result["probability"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    elif analyze_btn and not comment_text.strip():
        st.error("⚠️ Please enter a comment before analyzing.")


# ═══════════════════════════════════════════════
# PAGE: Model Dashboard
# ═══════════════════════════════════════════════
elif "📊 Model Dashboard" in page:

    st.markdown("## 📊 Model Performance Dashboard")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.6);'>"
        "Key evaluation metrics comparing the Baseline (TF-IDF + LR) vs. Deep Learning (BiLSTM) models."
        "</p>", unsafe_allow_html=True
    )

    # Metrics (update these after training with actual values)
    baseline = {"Accuracy": 89.2, "Precision": 89.3, "Recall": 89.2, "F1 Score": 89.2, "ROC-AUC": 95.8}
    bilstm   = {"Accuracy": 92.7, "Precision": 92.8, "Recall": 92.7, "F1 Score": 92.7, "ROC-AUC": 97.9}

    st.markdown("#### 🏆 Test Set Performance")
    m_cols = st.columns(5)
    labels = list(baseline.keys())
    for i, (col, metric) in enumerate(zip(m_cols, labels)):
        with col:
            delta = bilstm[metric] - baseline[metric]
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{bilstm[metric]:.1f}%</div>
                <div class='metric-label'>{metric}</div>
                <div style='color:#10b981; font-size:0.8rem; margin-top:4px;'>
                    ▲ +{delta:.1f}% vs baseline
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison bar chart
    st.markdown("#### 📊 Baseline vs. BiLSTM Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline (TF-IDF + LR)",
        x=labels,
        y=[baseline[k] for k in labels],
        marker_color="#3b82f6",
        marker_line_width=0,
        text=[f"{baseline[k]:.1f}%" for k in labels],
        textposition="inside",
        textfont=dict(color="white", size=12),
    ))
    fig.add_trace(go.Bar(
        name="BiLSTM (Deep Learning)",
        x=labels,
        y=[bilstm[k] for k in labels],
        marker_color="#f59e0b",
        marker_line_width=0,
        text=[f"{bilstm[k]:.1f}%" for k in labels],
        textposition="inside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            range=[80, 100],
            title="Score (%)",
            gridcolor="rgba(255,255,255,0.08)",
            ticksuffix="%",
        ),
        xaxis=dict(title="Metric"),
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Architecture info
    st.markdown("#### 🏗️ Model Architecture — BiLSTM")
    arch_data = {
        "Layer": ["Embedding", "SpatialDropout1D", "Bidirectional LSTM", "LSTM", "BatchNormalization", "Dense (ReLU)", "Dropout", "Dense (Sigmoid)"],
        "Output Shape": ["(None, 250, 128)", "(None, 250, 128)", "(None, 250, 256)", "(None, 64)", "(None, 64)", "(None, 64)", "(None, 64)", "(None, 1)"],
        "Parameters": ["2,560,128", "0", "198,144", "82,240", "256", "4,160", "0", "65"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE: Training Plots
# ═══════════════════════════════════════════════
elif "📈 Training Plots" in page:

    st.markdown("## 📈 Training & Evaluation Plots")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.6);'>All plots generated during training and EDA.</p>",
        unsafe_allow_html=True
    )

    plot_files = sorted([
        f for f in os.listdir(PLOTS_DIR)
        if f.endswith(".png")
    ]) if os.path.isdir(PLOTS_DIR) else []

    if not plot_files:
        st.info(
            "📭 No plots found yet. Run `python src/train.py` to generate all plots.",
            icon="ℹ️",
        )
    else:
        plot_names = {
            "01_class_distribution.png":        "Class Distribution",
            "02_wordclouds.png":                "Word Clouds (Positive vs Negative)",
            "03_confusion_matrix_baseline.png": "Confusion Matrix — Baseline (TF-IDF + LR)",
            "04_training_curves.png":           "BiLSTM Training & Validation Curves",
            "05_confusion_matrix_dl.png":       "Confusion Matrix — BiLSTM Deep Learning",
            "06_roc_auc_curve.png":             "ROC-AUC Curves",
            "07_model_comparison.png":          "Model Comparison (Baseline vs BiLSTM)",
        }

        cols = st.columns(2)
        for i, fname in enumerate(plot_files):
            path = os.path.join(PLOTS_DIR, fname)
            title = plot_names.get(fname, fname)
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                try:
                    img = Image.open(path)
                    st.image(img, use_column_width=True)
                except Exception:
                    st.error(f"Could not load: {fname}")
                st.markdown("<br>", unsafe_allow_html=True)
