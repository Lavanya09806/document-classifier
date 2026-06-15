# AI-Powered Document Classification System

An advanced document classification system built with **Python**, **Scikit-learn**, **Pandas**, **NumPy**, and **Streamlit**. It automatically parses document text from **PDF**, **DOCX**, and **TXT** files and uses Natural Language Processing (NLP) with Machine Learning classifiers to classify them into one of six categories:
1. **Resume**
2. **Invoice**
3. **Medical Report**
4. **Legal Document**
5. **News Article**
6. **Research Paper**

---

## Key Features

- **Multi-Format Parsing:** Support for `.txt` (UTF-8/raw), `.docx` (via `python-docx`), and `.pdf` (via `pypdf`) files.
- **NLP Pipelines:** TF-IDF feature extraction (`TfidfVectorizer`) with custom stop words, n-gram ranges, and max feature configurations.
- **Machine Learning Models:** Fully-functional classifiers for:
  - **Naive Bayes (MultinomialNB)**
  - **Logistic Regression (L2 regularized)**
  - **Support Vector Machine (Linear Kernel SVC)**
- **Interactive Analytics Dashboard:** Real-time metrics visualization including Accuracy, Precision, Recall, F1 Score, interactive Confusion Matrix, and model benchmark comparisons.
- **Feature Importance Visualization:** Inspect the highest-weighted terms representing the vocabulary of each document category.
- **Hyperparameter Console:** Dynamic model tuning from the interface. Allows configuring train-test splits, smoothing factors, and regularization strengths with immediate retraining.
- **Sample Document Generator:** Automatic creation of mock PDF, DOCX, and TXT files containing category-specific text to download and use for upload testing.

---

## Directory Architecture

```
aiml/
├── requirements.txt         # Project package dependencies
├── app.py                   # Main Streamlit web application frontend
├── README.md                # Project documentation and user guide
├── src/                     # Core python libraries and pipeline logic
│   ├── __init__.py
│   ├── text_extractor.py    # Reads PDF, DOCX, TXT and cleans text
│   ├── dataset_generator.py # Generates synthetic dataset and sample documents
│   └── model_pipeline.py    # Trains, evaluates, saves, and queries models
├── data/
│   ├── document_dataset.csv # Automatically generated dataset (~360 samples)
│   └── samples/             # Downloadable sample test documents
└── models/                  # Saved pickled models and TF-IDF vectorizers
```

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Open a terminal in the project directory and install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit server:
```bash
streamlit run app.py
```
After running, Streamlit will open a new window in your browser (typically at `http://localhost:8501`).

---

## Machine Learning Pipeline

### Data Preprocessing
- Letters are lowercased and special characters/emojis are removed.
- Consecutive spaces are squashed.
- `TfidfVectorizer` transforms documents using term frequency-inverse document frequency weighting, capturing unigrams and bigrams.
- Custom stop words are removed to filters out boilerplate terms like "and", "the", etc.

### Classifiers
1. **Naive Bayes:** Fast, baseline probabilistic classifier. Great for handling sparse text arrays.
2. **Logistic Regression:** Softmax-calibrated classifier. Provides clean, reliable probability confidence outputs.
3. **Support Vector Machine:** Fits a linear kernel hyper-plane with probability estimation enabled. Great at scaling to high-dimensional classification boundaries.
