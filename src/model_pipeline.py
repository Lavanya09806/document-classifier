import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODELS_DIR = "models"
DATASET_PATH = "data/document_dataset.csv"

def load_dataset():
    """Loads the dataset from the CSV file."""
    if not os.path.exists(DATASET_PATH):
        # Fallback: import generator and create dataset if not exists
        from src.dataset_generator import generate_dataset_csv, generate_all_samples
        generate_dataset_csv(DATASET_PATH, num_samples_per_class=60)
        generate_all_samples("data/samples")
    
    return pd.read_csv(DATASET_PATH)

def train_and_evaluate_models(train_ratio=0.8, nb_alpha=1.0, lr_c=1.0, svm_c=1.0):
    """
    Trains Naive Bayes, Logistic Regression, and SVM models on the dataset.
    Returns metrics and plots data for the Streamlit UI.
    """
    df = load_dataset()
    
    X = df['text']
    y = df['category']
    
    # Split dataset
    test_size = 1.0 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define models
    models = {
        "Naive Bayes": MultinomialNB(alpha=nb_alpha),
        "Logistic Regression": LogisticRegression(C=lr_c, max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(C=svm_c, kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the vectorizer
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    for name, model in models.items():
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate confusion matrix
        classes = sorted(y.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        # Save model
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump(model, f)
            
        results[name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classes": classes
        }
        
    return results

def get_loaded_model(model_name):
    """Loads a saved model by its display name."""
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        # If model doesn't exist, train models first
        train_and_evaluate_models()
        
    with open(filepath, "rb") as f:
        return pickle.load(f)

def get_loaded_vectorizer():
    """Loads the saved TfidfVectorizer."""
    filepath = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(filepath):
        train_and_evaluate_models()
        
    with open(filepath, "rb") as f:
        return pickle.load(f)

def predict_document(text, model_name):
    """
    Cleans text, vectorizes it, and runs inference.
    Returns: (predicted_category, confidence_scores_dict)
    """
    vectorizer = get_loaded_vectorizer()
    model = get_loaded_model(model_name)
    
    # Transform text
    vec_text = vectorizer.transform([text])
    
    # Predict category
    predicted_category = model.predict(vec_text)[0]
    
    # Get probability confidence scores
    classes = model.classes_
    probabilities = model.predict_proba(vec_text)[0]
    
    confidence_scores = {cl: float(prob) for cl, prob in zip(classes, probabilities)}
    
    # Sort confidence scores by highest first
    sorted_confidence = dict(sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True))
    
    return predicted_category, sorted_confidence

def get_top_features_per_class(model_name, num_top_words=10):
    """
    Extracts the top terms per class based on model weights.
    Returns a dictionary mapping class name -> list of (term, weight) tuples.
    """
    vectorizer = get_loaded_vectorizer()
    model = get_loaded_model(model_name)
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = model.classes_
    
    top_features = {}
    
    if model_name == "Naive Bayes":
        # MultinomialNB has feature_log_prob_ (n_classes, n_features)
        for i, class_label in enumerate(classes):
            top_indices = np.argsort(model.feature_log_prob_[i])[-num_top_words:][::-1]
            top_features[class_label] = [(feature_names[idx], float(model.feature_log_prob_[i][idx])) for idx in top_indices]
            
    elif model_name == "Logistic Regression":
        # LogisticRegression has coef_ (n_classes, n_features)
        if hasattr(model, 'coef_') and model.coef_.shape[0] == len(classes):
            coef = model.coef_
            for i, class_label in enumerate(classes):
                row = coef[i].toarray()[0] if hasattr(coef[i], 'toarray') else coef[i]
                top_indices = np.argsort(row)[-num_top_words:][::-1]
                top_features[class_label] = [(feature_names[idx], float(row[idx])) for idx in top_indices]
                
    # Fallback/Default for SVM or if not populated: use mean TF-IDF features for the class
    if not top_features:
        df = load_dataset()
        for class_label in classes:
            class_texts = df[df['category'] == class_label]['text']
            vec_class = vectorizer.transform(class_texts)
            mean_tfidf = np.asarray(vec_class.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_tfidf)[-num_top_words:][::-1]
            top_features[class_label] = [(feature_names[idx], float(mean_tfidf[idx])) for idx in top_indices]
            
    return top_features
