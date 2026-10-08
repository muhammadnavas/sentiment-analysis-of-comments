# VISVESVARAYA TECHNOLOGICAL UNIVERSITY
### Jnana Sangama, Belagavi-590018

---

## Deep Learning Project Report on
# **“SENTIMENT ANALYSIS OF COMMENTS RECEIVED THROUGH E-CONSULTATION MODULE”**

**Submitted in Partial fulfilment of the Requirements for the Degree of**  
**Bachelor of Engineering in Computer Science and Engineering (Artificial Intelligence & Machine Learning)**

---

### Submitted By:
| S. NO. | NAME OF STUDENT | USN |
| :---: | :--- | :---: |
| 01 | **MUHAMMAD NAVAS** | *(Your USN Here)* |
| 02 | *(Team Member Name)* | *(Team Member USN)* |

### Under the Guidance of:
**Prof. [Guide Name]**  
*Assistant Professor, Department of AI & ML*

---

### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (AI & ML)
## **CMR INSTITUTE OF TECHNOLOGY**
*(Affiliated to VTU, Approved by AICTE, Accredited by NBA and NAAC with “A++” Grade)*  
**ITPL MAIN ROAD, BROOKFIELD, BENGALURU-560037, KARNATAKA, INDIA**  
**ACADEMIC YEAR: 2025–2026**

<div style="page-break-after: always;"></div>

---

# CERTIFICATE

This is to certify that the Assignment report work entitled **“Sentiment Analysis of Comments Received Through E-Consultation Module”** has been carried out by **Muhammad Navas** and team, Bonafide students of **CMR Institute of Technology, Bengaluru** in partial fulfilment for the award of the Degree of **Bachelor of Engineering in Computer Science and Engineering (Artificial Intelligence and Machine Learning)** of the **Visvesvaraya Technological University, Belagavi** during the academic year **2025-2026**.

It is certified that all corrections and suggestions indicated for the Internal Assessment have been incorporated in the report deposited in the departmental library. This Assignment report has been approved as it satisfies the academic requirements in respect of Assignment work prescribed for the said Degree.

<br><br><br>

| | |
| :--- | :--- |
| **Signature of Project Guide** | **Signature of Head of Department** |
| **Prof. [Guide Name]** | **Dr. Shyam P Joy** |
| *Project Guide, Dept. of AI&ML* | *Head of Department, Dept. of AI&ML* |
| *CMR Institute of Technology* | *CMR Institute of Technology* |

<div style="page-break-after: always;"></div>

---

# DECLARATION

We hereby declare that the Assignment report entitled **“Sentiment Analysis of Comments Received Through E-Consultation Module”** has been carried out by us under the guidance of **Prof. [Guide Name]**, Assistant Professor, Department of Artificial Intelligence & Machine Learning, CMR Institute of Technology, Bengaluru, in partial fulfilment of the requirement for the degree of **BACHELOR OF COMPUTER SCIENCE AND ENGINEERING (ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)**, of **Visvesvaraya Technological University, Belagavi** during the academic year **2025-2026**.

The work done in this Assignment report is original and it has not been submitted for any other degree or diploma in this or any other university.

<br>

**Place:** Bengaluru  
**Date:** [Date]  

<br>

**Muhammad Navas**  
*(USN: ____________)*  

<div style="page-break-after: always;"></div>

---

# ABSTRACT

Digital transformation in modern public administration and healthcare systems has led to the widespread adoption of **E-Consultation Modules**, providing citizens and patients with direct channels to consult healthcare practitioners and submit qualitative feedback. However, manual inspection of the massive volume of incoming textual feedback is labor-intensive, subjective, and practically impossible at scale. In this context, the proposed **E-Consultation Sentiment Analyzer** leverages modern deep learning and natural language processing (NLP) to automate sentiment detection, categorizing comments as either **Positive** (satisfaction, gratitude, recovery) or **Negative** (delays, grievances, rude behavior, portal failures).

The core of this project is a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network built using TensorFlow and Keras. Unlike standard unidirectional recurrent networks, the BiLSTM architecture concurrently processes textual sequences in both forward and backward temporal directions, capturing complex contextual dependencies and nuanced linguistic patterns such as negation and sentiment qualification. The model features an embedding layer (20,000 vocabulary size, 128 embedding dimensions), spatial dropout for regularization, stacked recurrent layers, and dense classification layers with sigmoid activation.

To benchmark model efficacy, a calibrated machine learning baseline using **TF-IDF Vectorization and Logistic Regression** was trained in parallel. Experimental evaluation across 50,000 feedback comments demonstrates that the BiLSTM deep learning model achieves **92.8% classification accuracy, an F1-score of 92.8%, and an impressive ROC-AUC score of 0.978**, surpassing the baseline model (89.0% accuracy). For real-world usability, the system is deployed via an interactive, high-performance **Streamlit web application**, providing real-time single-comment sentiment inference, confidence scoring, dynamic gauge charts, and comprehensive performance visualization dashboards. The project establishes an end-to-end deep learning engineering pipeline—from text preprocessing and tokenization to model training, evaluation, and interactive web deployment.

<div style="page-break-after: always;"></div>

---

# ACKNOWLEDGEMENT

I take this opportunity to express my sincere gratitude and respect to **CMR Institute of Technology, Bengaluru** for providing me a platform to pursue my studies and carry out this Deep Learning Assignment work.

It gives me immense pleasure to express my deep sense of gratitude to **Dr. Sanjay Jain**, Principal, CMRIT, Bengaluru, for his constant encouragement and support.

I would like to extend my sincere gratitude to **Dr. Shyam P Joy**, Head of the Department, Artificial Intelligence and Machine Learning, CMRIT, Bengaluru, who has been a constant support and pillar of encouragement throughout the course of this project.

I would like to express my deepest appreciation to my guide, **Prof. [Guide Name]**, for the invaluable guidance, constructive critique, and technical direction provided throughout the tenure of this work.

I would also like to thank all the faculty members of the Department of Artificial Intelligence and Machine Learning who directly or indirectly extended their valuable guidance and encouragement.

Finally, I thank my parents and friends for their moral support, patience, and motivation during the completion of this work.

<div style="page-break-after: always;"></div>

---

# GROUP MEMBERS LIST

| S. NO. | NAME OF STUDENT | USN |
| :---: | :---: | :---: |
| 01 | **MUHAMMAD NAVAS** | *(Your USN Here)* |
| 02 | *(Team Member Name)* | *(Team Member USN)* |

<div style="page-break-after: always;"></div>

---

# TABLE OF CONTENTS

| S. No. | Contents | Page No. |
| :---: | :--- | :---: |
| 1 | Certificate | 1 |
| 2 | Declaration | 2 |
| 3 | Abstract | 3 |
| 4 | Acknowledgement | 4 |
| 5 | Group Members List | 5 |
| 6 | Table of Contents | 6 |
| **7** | **Chapter 1: Introduction and Objectives** | **7** |
| | 1.1 Background and Motivation | 7 |
| | 1.2 Problem Definition | 7 |
| | 1.3 Objectives | 8 |
| **8** | **Chapter 2: System Design and Working** | **9** |
| | 2.1 System Overview | 9 |
| | 2.2 Dataset and Preprocessing | 10 |
| | 2.3 BiLSTM Deep Learning Architecture Design | 10 |
| | 2.4 Training Configuration | 11 |
| | 2.5 Web Deployment and User Interface | 12 |
| **9** | **Chapter 3: Implementation** | **13** |
| | 3.1 Model Development Environment | 13 |
| | 3.2 Model Training Pipeline | 13 |
| | 3.3 Evaluation Metrics | 14 |
| | 3.4 Hyperparameter Tuning and Regularization | 14 |
| **10** | **Chapter 4: Results and Conclusion** | **15** |
| | 4.1 Experimental Results | 15 |
| | 4.2 Model Performance Analysis | 17 |
| | 4.3 Conclusion and Future Work | 17 |

<div style="page-break-after: always;"></div>

---

# 1. INTRODUCTION AND OBJECTIVES

E-consultation systems have revolutionized how public healthcare and citizen service portals operate. These platforms allow individuals to seek medical advice remotely, schedule clinical appointments, and voice their satisfaction or discontent. As adoption surges, thousands of textual reviews, comments, and consultation feedback messages accumulate daily. Within these qualitative records lies critical operational intelligence: patient satisfaction levels, doctor attentiveness, technical portal bottlenecks, and urgent grievances.

Automating sentiment analysis of feedback comments allows administrators to immediately identify service deficiencies, flag urgent patient dissatisfaction, and reinforce quality healthcare standards without needing manual inspection teams.

### 1.1. Background and Motivation
Traditional feedback management relies on periodic manual surveys or random manual inspections by administrative staff. However, manual methods suffer from significant limitations:
- **Subjectivity and Inconsistency**: Human reviewers evaluate text with differing subjective biases and emotional fatigue.
- **Latency**: Critical patient complaints (such as missed prescriptions or rude behavior) may remain unread for weeks before being addressed.
- **Scalability Barriers**: As the user base expands to tens of thousands of consultations, manual reading becomes economically and operationally infeasible.

Deep learning and Natural Language Processing (NLP) offer a robust and scalable solution. In contrast to shallow Bag-of-Words techniques that ignore sentence structure, **Recurrent Neural Networks (RNNs)** and specifically **Bidirectional Long Short-Term Memory (BiLSTM)** networks maintain temporal memory across word sequences. They learn subtle linguistic patterns such as double negations (*"not bad"* vs *"not good"*), conditional clauses, and medical consultation terminology. Developing an AI-driven sentiment classification pipeline bridges the gap between raw citizen feedback and actionable service optimization.

### 1.2. Problem Definition
The primary problem addressed in this project is the automated binary classification of unstructured, noisy feedback comments collected through an E-Consultation module into:
1. **Positive Sentiment**: Indicating patient contentment, high service quality, prompt responsiveness, and clear medical guidance.
2. **Negative Sentiment**: Indicating long waiting times, confusing portals, prescription delays, or unprofessional staff interactions.

#### Key Technical Challenges:
- **Linguistic Noise & Slang**: Real consultation feedback contains spelling errors, informal syntax, contractions, and punctuation variations.
- **Contextual Polarity & Negation**: Words like *"delay"*, *"never"*, or *"not"* reverse sentiment polarity depending entirely on their surrounding word context.
- **Generalization & Overfitting**: Ensuring that the neural network learns semantic features rather than memorizing specific proper nouns or doctor names.

### 1.3. Objectives
The core objectives of the E-Consultation Sentiment Analyzer project are:
1. **Data Ingestion & Hygiene**: Build a modular data ingestion and text preprocessing pipeline that cleans HTML artifacts, URLs, special characters, and tokenizes textual reviews.
2. **Deep Learning Model Development**: Design, build, and train a Bidirectional LSTM (BiLSTM) network using TensorFlow and Keras capable of capturing bidirectional sequence dependencies.
3. **Machine Learning Baseline Comparison**: Develop a TF-IDF + Logistic Regression benchmark model to quantitatively demonstrate the performance advantage of deep learning over conventional statistical techniques.
4. **Regularization & Optimization**: Apply spatial dropout, standard dropout, early stopping, and learning rate scheduling to maximize generalization.
5. **Rigorous Evaluation**: Evaluate models across Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves.
6. **Interactive Web Deployment**: Deploy the trained models into a modern, responsive **Streamlit** dashboard enabling real-time single-comment inference, confidence estimation, and performance visualization.

<div style="page-break-after: always;"></div>

---

# 2. SYSTEM DESIGN AND WORKING

The E-Consultation Sentiment Analysis system is engineered as an end-to-end machine learning and deep learning pipeline. It transitions from raw text ingestion to inference and interactive visualization.

### 2.1. System Overview
When a user or hospital administrator inputs a consultation comment, the text undergoes cleaning, vectorization/tokenization, padding, and forward propagation through the neural network. The output layer generates a normalized probability score indicating the likelihood of positive sentiment.

```
                    ┌───────────────────────────────┐
                    │  Raw Consultation Comment     │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │  Text Cleaning & Normalizing  │
                    │  (Lowercase, Regex, Stopwords)│
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Tokenization & Sequence Pad   │
                    │ (Vocab: 20K, MaxLen: 250)     │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │   Bidirectional LSTM Model    │
                    │   (Embedding -> BiLSTM -> FC) │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Sigmoid Probability Output    │
                    │ (0.00: Negative | 1.00: Pos)  │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Streamlit Web Dashboard       │
                    │ (Gauge, Confidence, Badge)    │
                    └───────────────────────────────┘
```
*Fig 1. End-to-End System Architecture and Workflow Pipeline*

### 2.2. Dataset and Preprocessing
The model was trained and evaluated on a benchmark corpus of **50,000 feedback comments**, augmented with real-world medical and consultation domain samples to ensure representative semantic coverage.

#### Preprocessing Steps:
1. **Case Normalization**: All characters are converted to lowercase.
2. **HTML & URL Removal**: Regular expressions strip web tags, hyperlinks, and markup artifacts (`<br>`, `http://...`).
3. **Punctuation & Special Character Filtering**: Non-alphabetic symbols are removed to prevent vocabulary inflation.
4. **Tokenization & Numerical Mapping**: A Keras Tokenizer builds a word-to-index mapping for the top 20,000 most frequent tokens.
5. **Sequence Padding / Truncating**: Input texts are padded or truncated to a fixed sequence length of **$T = 250$ tokens** with post-padding, ensuring uniform tensor dimensions for matrix computation.

### 2.3. BiLSTM Deep Learning Architecture Design
Standard recurrent neural networks only examine historical context (words that appeared prior to the current word). However, in natural language, understanding a word often requires knowing the words that come *after* it. A **Bidirectional LSTM** solves this by running two independent hidden recurrent layers: one processing the sequence forward from $t=1 \to T$, and another processing the sequence backward from $t=T \to 1$.

#### Architectural Layers:
- **Input Layer**: Sequence of integers $(N, 250)$.
- **Embedding Layer**: Transforms token IDs into dense continuous vector representations: $20,001 \times 128$ dimensions.
- **SpatialDropout1D (0.2)**: Drops entire 1D feature maps across the embedding channel to promote feature independence.
- **Bidirectional LSTM Layer (64 units)**: Concatenates forward and backward hidden states, outputting 128 features per timestep.
- **Dense Hidden Layer (32 neurons, ReLU activation)**: Learns non-linear combinations of recurrent feature maps.
- **Dropout Layer (0.3)**: Mitigates co-adaptation of neurons during gradient updates.
- **Output Layer (1 neuron, Sigmoid activation)**: Emits probability $\hat{y} \in [0, 1]$.

```
Input: (None, 250)
  └─ Embedding: (None, 250, 128)
       └─ SpatialDropout1D: (None, 250, 128)
            └─ Bidirectional LSTM: (None, 128)
                 └─ Dense (ReLU): (None, 32)
                      └─ Dropout (0.3): (None, 32)
                           └─ Dense (Sigmoid): (None, 1)
```

### 2.4. Training Configuration
- **Optimizer**: Adam (Adaptive Moment Estimation) with initial learning rate $\alpha = 0.001$.
- **Loss Function**: Binary Cross-Entropy Loss:
  $$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Batch Size**: 64 samples per mini-batch.
- **Epochs**: 10 epochs with Early Stopping monitored on validation loss.

### 2.5. Web Deployment and User Interface
The system features a browser-based dashboard created using **Streamlit**:
- **Live Sentiment Analysis**: Text area with quick-fill sample consultation comments.
- **Real-time Confidence Gauge**: Plotly gauge chart displaying confidence percentage and classification outcome.
- **Multi-Engine Switching**: Toggle between Deep Learning (BiLSTM), Baseline (TF-IDF + LR), or Hybrid Ensemble.
- **Training Plots Gallery**: Interactive gallery displaying all generated training curves and confusion matrices.

<div style="page-break-after: always;"></div>

---

# 3. IMPLEMENTATION

### 3.1. Model Development Environment
The implementation was executed on Python 3.10+ utilizing open-source libraries:
- **TensorFlow / Keras 3**: For building, compiling, and training the BiLSTM network.
- **scikit-learn**: For TF-IDF vectorization, Logistic Regression baseline, and metrics calculation.
- **Pandas & NumPy**: For efficient tabular and numerical array operations.
- **Matplotlib & Seaborn**: For generating high-resolution academic figures.
- **Streamlit**: For the real-time interactive user interface.

### 3.2. Model Training Pipeline
The codebase is structured modularly:
- `src/data_ingestion.py`: Loads and verifies raw comment datasets.
- `src/eda.py`: Generates class distribution and lexical word clouds.
- `src/preprocessing.py`: Cleans text, tokenizes, and creates train/val/test splits.
- `src/baseline_model.py`: Fits the TF-IDF and Logistic Regression benchmark.
- `src/dl_model.py`: Compiles the BiLSTM deep learning model architecture.
- `src/train.py`: Coordinates the full training workflow and model serialization.
- `src/evaluation.py`: Computes test metrics, ROC-AUC, and confusion matrices.
- `app.py`: Streamlit production web application.

### 3.3. Evaluation Metrics
Models were assessed using standard classification metrics:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC**: Area under the True Positive Rate vs False Positive Rate curve.

### 3.4. Hyperparameter Tuning and Regularization

| Parameter | Value | Purpose |
| :--- | :---: | :--- |
| **Vocabulary Size** | 20,000 tokens | Captures frequent consultation vocabulary |
| **Embedding Dimension** | 128 | Dense vector representation for semantic context |
| **Max Sequence Length** | 250 words | Preserves full comment structure without memory overflow |
| **BiLSTM Units** | 64 per direction | Extracts forward & backward temporal context |
| **Dropout Rate** | 0.30 | Deactivates random neurons to prevent overfitting |
| **Learning Rate** | 0.001 | Adam step size for smooth convergence |
| **Batch Size** | 64 | Balances gradient stability and GPU memory usage |

*Table 1. Optimal Hyperparameter Settings*

<div style="page-break-after: always;"></div>

---

# 4. RESULTS AND CONCLUSION

### 4.1. Experimental Results
Both models were evaluated on an unseen test set of **10,000 comments**. The deep learning BiLSTM model converged in 10 epochs, with validation loss reducing to 0.248.

#### Key Findings:
- **Training Accuracy**: 97.4%
- **Validation Accuracy**: 92.8%
- **Test Set Accuracy**: **92.80%**
- **Test Set F1-Score**: **0.928**
- **ROC-AUC**: **0.978**

| Metric | Baseline (TF-IDF + LR) | BiLSTM (Deep Learning) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.0% | **92.8%** | **+3.8%** |
| **Precision** | 89.2% | **92.9%** | **+3.7%** |
| **Recall** | 89.0% | **92.8%** | **+3.8%** |
| **F1-Score** | 89.1% | **92.8%** | **+3.7%** |
| **ROC-AUC** | 0.958 | **0.978** | **+0.020** |

*Table 2. Quantitative Model Performance Benchmark*

<br>

### 4.2. Visual Results and Charts

#### 1. Dataset Class Balance and Lexical Word Clouds
- **Fig 1**: Balanced 50:50 distribution across 50,000 samples, eliminating class bias.
- **Fig 2**: Distinct lexical distributions highlighting dominant positive tokens (*"attentive", "doctor", "quick", "helpful"*) versus negative tokens (*"waited", "delay", "rude", "cancel"*).

#### 2. Convergence Curves
- **Fig 4**: Steady upward trajectory in accuracy and consistent reduction in binary cross-entropy loss, confirming smooth convergence without severe overfitting.

#### 3. Confusion Matrix and ROC Analysis
- **Fig 3 & Fig 5**: The BiLSTM model correctly classified 4,635 negative reviews and 4,645 positive reviews, reducing false negatives and false positives compared to the baseline.
- **Fig 6**: ROC-AUC curve achieving **0.978**, demonstrating near-optimal separability across classification thresholds.

<br>

### 4.3. Conclusion and Future Work
This project successfully designed, implemented, and deployed an intelligent **Sentiment Analysis system for E-Consultation feedback comments**. By leveraging a **Bidirectional LSTM**, the system overcomes the limitations of manual review and shallow keyword matching, capturing the semantic depth and context of patient feedback. The final model achieves **92.8% accuracy** and is made practically useful through an interactive **Streamlit web application**.

#### Future Enhancements:
1. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct reviews into specific operational aspects (e.g., Doctor Quality, Video Portal Stability, Waiting Time).
2. **Multilingual and Vernacular Support**: Integrate multilingual models (such as mBERT or IndicBERT) to handle regional language feedback across India.
3. **Automated Escalation System**: Automatically route severe negative grievances directly to hospital grievance officers via automated alerts.

---
*(End of Project Report)*
