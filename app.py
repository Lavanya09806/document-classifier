import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.text_extractor import extract_text
from src.model_pipeline import (
    train_and_evaluate_models, 
    predict_document, 
    get_top_features_per_class,
    get_loaded_model,
    load_dataset
)
from src.dataset_generator import generate_dataset_csv, generate_all_samples

# Page configurations
st.set_page_config(
    page_title="DocuClassify AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #f8fafc;
    }
    
    /* Title bar gradient */
    .title-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom CSS for KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result prediction card */
    .predict-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 25px;
        color: #1e3a8a;
        margin-bottom: 20px;
    }
    .predict-title {
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 15px;
    }
    .predict-class {
        font-size: 2.5rem;
        font-weight: 900;
        color: #1e40af;
        margin-bottom: 5px;
    }
    .predict-confidence {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2563eb;
    }
    
    /* Styled buttons */
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        transition: background-color 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# App directories paths
DATASET_CSV = "data/document_dataset.csv"
SAMPLES_DIR = "data/samples"
MODELS_DIR = "models"

# Automatically initialize dataset and sample documents if missing
@st.cache_resource
def initialize_app_data():
    if not os.path.exists(DATASET_CSV):
        os.makedirs("data", exist_ok=True)
        generate_dataset_csv(DATASET_CSV, num_samples_per_class=60)
        generate_all_samples(SAMPLES_DIR)
        # Pre-train models
        train_and_evaluate_models()

initialize_app_data()

# App header
st.markdown("""
<div class="title-container">
    <h1 style="margin: 0; font-size: 2.5rem;">📄 DocuClassify AI</h1>
    <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 300;">
        An intelligent classification engine utilizing NLP TF-IDF features and Machine Learning models.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown("<h2 style='text-align: center; color: #1e293b;'>Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Go To Page",
    [
        "📊 Analytics Dashboard",
        "🚀 Document Classifier",
        "⚙️ Model Training & Tuning",
        "📥 Sample Document Generator"
    ]
)

# Active Model Selection in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Active Classifier")

# Read existing models
available_models = ["Support Vector Machine", "Logistic Regression", "Naive Bayes"]
active_model_name = st.sidebar.selectbox(
    "Select Model for Inference",
    available_models,
    index=0
)

# Load details about active model to show in sidebar
try:
    # Trigger model load test
    get_loaded_model(active_model_name)
    st.sidebar.success(f"✓ {active_model_name} loaded")
except Exception:
    st.sidebar.warning("Model needs to be trained first")

# Load Dataset for stats
df_dataset = load_dataset()
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Statistics")
st.sidebar.write(f"**Total Samples:** {len(df_dataset)}")
st.sidebar.write(f"**Categories:** {len(df_dataset['category'].unique())}")
for cat, count in df_dataset['category'].value_counts().items():
    st.sidebar.caption(f"- {cat}: {count} samples")


# ==============================================================================
# TABS LOGIC
# ==============================================================================

if nav_selection == "📊 Analytics Dashboard":
    st.header("Analytics & Evaluation Dashboard")
    st.markdown("Inspect performance metrics, model comparison benchmarks, confusion matrices, and representative vocabularies.")
    
    # Run evaluation of saved models to display metrics
    # We will run a quick check of model scores
    # To keep it fast, we can cache the metrics or run it once
    @st.cache_data
    def get_cached_metrics():
        # Retrains or returns scores on test split
        # We can extract training/test split performance
        return train_and_evaluate_models()
        
    try:
        all_metrics = get_cached_metrics()
        active_metrics = all_metrics[active_model_name]
        
        # 1. KPI Metrics Cards for the Active Model
        st.markdown(f"### {active_model_name} Metrics (Test Set)")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #10b981;">
                <div class="kpi-value">{active_metrics['accuracy']:.2%}</div>
                <div class="kpi-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #3b82f6;">
                <div class="kpi-value">{active_metrics['precision']:.2%}</div>
                <div class="kpi-label">Precision (Weighted)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #f59e0b;">
                <div class="kpi-value">{active_metrics['recall']:.2%}</div>
                <div class="kpi-label">Recall (Weighted)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #8b5cf6;">
                <div class="kpi-value">{active_metrics['f1_score']:.2%}</div>
                <div class="kpi-label">F1-Score (Weighted)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 2. Split Charts: Confusion Matrix & Comparison
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Confusion Matrix")
            # Build annotated confusion matrix heatmap using Plotly
            cm = active_metrics['confusion_matrix']
            classes = active_metrics['classes']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale='Blues',
                hoverongaps=False,
                text=cm,
                texttemplate="%{text}",
                showscale=True
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                margin=dict(l=40, r=40, t=20, b=40),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_right:
            st.markdown("### Model Benchmarking Comparison")
            # Build comparison bar chart for all models
            model_comparison_data = []
            for m_name, m_metrics in all_metrics.items():
                model_comparison_data.append({
                    "Model": m_name,
                    "Accuracy": m_metrics['accuracy'],
                    "Precision": m_metrics['precision'],
                    "Recall": m_metrics['recall'],
                    "F1-Score": m_metrics['f1_score']
                })
            df_compare = pd.DataFrame(model_comparison_data)
            
            # Melt dataframe for long format plotting
            df_compare_melted = df_compare.melt(id_vars="Model", var_name="Metric", value_name="Score")
            
            fig_compare = px.bar(
                df_compare_melted,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Safe,
                range_y=[0, 1.05]
            )
            fig_compare.update_layout(
                yaxis_tickformat='.0%',
                margin=dict(l=40, r=40, t=20, b=40),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
        st.markdown("---")
        
        # 3. Keyword Importance / Feature Importances per Class
        st.markdown("### Class Vocabulary & Feature Importance")
        st.markdown("Discover the most distinct keywords used by the classifier to identify each document category.")
        
        # Selectbox for category
        selected_category = st.selectbox(
            "Select Document Category to inspect vocabulary",
            classes,
            index=0
        )
        
        # Get top features
        top_features = get_top_features_per_class(active_model_name, num_top_words=15)
        
        if selected_category in top_features:
            cat_features = top_features[selected_category]
            words = [item[0] for item in cat_features]
            scores = [item[1] for item in cat_features]
            
            # Normalize scores for display purposes if negative (Naive Bayes has log probabilities)
            display_scores = scores
            x_title = "Feature Importance Weight"
            if active_model_name == "Naive Bayes":
                # Convert log probs to relative probabilities or positive values for display
                # relative score = exp(log_prob)
                display_scores = np.exp(scores)
                x_title = "Relative Probability Score"
                
            fig_features = px.bar(
                x=display_scores,
                y=words,
                orientation='h',
                labels={'x': x_title, 'y': 'Keywords'},
                color=display_scores,
                color_continuous_scale='GnBu'
            )
            fig_features.update_layout(
                yaxis={'categoryorder':'total ascending'},
                margin=dict(l=40, r=40, t=20, b=40),
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_features, use_container_width=True)
        else:
            st.warning("No features extracted for this category")
            
    except Exception as e:
        st.error(f"Error rendering dashboard: {str(e)}")
        st.info("Try retraining the models in the 'Model Training & Tuning' page.")


elif nav_selection == "🚀 Document Classifier":
    st.header("Document Classification Engine")
    st.markdown("Upload a document (**PDF, DOCX, or TXT**) to analyze its text content, classify its category, and see the confidence breakdown.")
    
    uploaded_file = st.file_uploader(
        "Upload a document file",
        type=["pdf", "docx", "txt"],
        help="Drag and drop or select a file. Maximum size 200MB."
    )
    
    if uploaded_file is not None:
        st.markdown("### Document Inspection & Extracted Content")
        
        # Save uploaded file to temp file to read
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Extracting and cleaning text from document..."):
            try:
                extracted_text = extract_text(temp_file_path)
                
                # Layout for preview and classification
                col_preview, col_classification = st.columns([1, 1])
                
                with col_preview:
                    st.markdown("#### Text Content Preview")
                    # Show first 1000 characters in an expander or container
                    if len(extracted_text) == 0:
                        st.warning("Warning: No text could be extracted from this document. Please check if the document is scanned/image-only.")
                    else:
                        st.info(f"Extracted {len(extracted_text)} characters.")
                        preview_length = min(1200, len(extracted_text))
                        st.text_area(
                            label="Document Text Segment",
                            value=extracted_text[:preview_length] + ("..." if len(extracted_text) > preview_length else ""),
                            height=350,
                            disabled=True
                        )
                        
                with col_classification:
                    st.markdown("#### Classification Prediction")
                    if len(extracted_text) > 0:
                        with st.spinner(f"Running inference using {active_model_name}..."):
                            predicted_category, confidence_scores = predict_document(extracted_text, active_model_name)
                            
                            # Custom icon mapping
                            icons = {
                                "Resume": "💼",
                                "Invoice": "🧾",
                                "Medical Report": "🏥",
                                "Legal Document": "⚖️",
                                "News Article": "📰",
                                "Research Paper": "🔬"
                            }
                            icon = icons.get(predicted_category, "📄")
                            
                            primary_confidence = confidence_scores.get(predicted_category, 0.0)
                            
                            # Render prediction card
                            st.markdown(f"""
                            <div class="predict-card">
                                <div class="predict-title">Predicted Category ({active_model_name})</div>
                                <div class="predict-class">{icon} {predicted_category}</div>
                                <div class="predict-confidence">Confidence Score: {primary_confidence:.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Draw horizontal bar chart for all classes confidence
                            st.markdown("#### Confidence Breakdown")
                            df_conf = pd.DataFrame([
                                {"Category": cat, "Probability": prob} 
                                for cat, prob in confidence_scores.items()
                            ])
                            
                            fig_conf = px.bar(
                                df_conf,
                                x="Probability",
                                y="Category",
                                orientation='h',
                                color="Probability",
                                color_continuous_scale='Blues',
                                range_x=[0, 1.05]
                            )
                            fig_conf.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                xaxis_tickformat='.0%',
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=280,
                                showlegend=False,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)
                    else:
                        st.error("Cannot classify empty text extraction.")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    else:
        st.info("Please upload a file to run the classification engine.")
        st.markdown("💡 *Need a sample document? Go to the **Sample Document Generator** tab to download test files.*")


elif nav_selection == "⚙️ Model Training & Tuning":
    st.header("Model Hyperparameters & Training Console")
    st.markdown("Customize your machine learning parameters, trigger a complete retraining process on the synthetic dataset, and review updated metrics.")
    
    # Inputs block
    col_params, col_info = st.columns([3, 2])
    
    with col_params:
        st.subheader("Hyperparameters Tuning")
        
        train_ratio = st.slider(
            "Train / Test Split Ratio",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Fraction of the dataset used to train models. The remainder is used for testing."
        )
        
        st.markdown("---")
        st.markdown("#### 1. Naive Bayes (MultinomialNB)")
        nb_alpha = st.slider(
            "Smoothing Parameter (Alpha)",
            min_value=0.01,
            max_value=5.0,
            value=1.0,
            step=0.05,
            help="Additive smoothing parameter (alpha=0 for no smoothing)."
        )
        
        st.markdown("---")
        st.markdown("#### 2. Logistic Regression")
        lr_c = st.number_input(
            "Regularization Strength C (Logistic Regression)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.5,
            help="Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization."
        )
        
        st.markdown("---")
        st.markdown("#### 3. Support Vector Machine (Linear Kernel SVC)")
        svm_c = st.number_input(
            "Regularization Strength C (SVM)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.5,
            help="Regularization parameter. The strength of the regularization is inversely proportional to C."
        )
        
        st.markdown("---")
        
        # Trigger button
        train_button = st.button("🚀 Train & Save Models", use_container_width=True)
        
    with col_info:
        st.subheader("Model Architectures")
        st.markdown("""
        **Multinomial Naive Bayes**
        - A probabilistic model based on Bayes' Theorem.
        - Calculates the probability of each class based on the frequencies of the words.
        - High speed, handles sparse feature vectors extremely well.
        
        **Logistic Regression**
        - A linear classifier that fits a logistic function to predict class probabilities.
        - Regularized using L2 penalty by default.
        - Provides strong calibrated probability estimates (via Softmax in multiclass settings).
        
        **Support Vector Machine (Linear Kernel)**
        - Finds the optimal separating hyperplane that maximizes the margin between classes.
        - Employs a linear kernel to handle high-dimensional TF-IDF vectors.
        - We train with probability estimation enabled (`probability=True`) to fetch confidence outputs.
        """)
        
    if train_button:
        with st.spinner("Retraining vectorizer and classifiers..."):
            try:
                # Clear cached metrics
                st.cache_data.clear()
                
                # Execute pipeline training
                new_results = train_and_evaluate_models(
                    train_ratio=train_ratio,
                    nb_alpha=nb_alpha,
                    lr_c=lr_c,
                    svm_c=svm_c
                )
                
                st.success("🎉 Models successfully trained, evaluated, and saved to disk!")
                
                # Display metrics summary
                st.markdown("### Retraining Performance Summary")
                st.markdown("The metrics show accuracy, precision, and recall evaluated on the holdout test set.")
                
                metrics_data = []
                for m_name, m_metrics in new_results.items():
                    metrics_data.append({
                        "Model": m_name,
                        "Accuracy": f"{m_metrics['accuracy']:.2%}",
                        "Precision (W)": f"{m_metrics['precision']:.2%}",
                        "Recall (W)": f"{m_metrics['recall']:.2%}",
                        "F1-Score (W)": f"{m_metrics['f1_score']:.2%}"
                    })
                st.table(pd.DataFrame(metrics_data))
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")


elif nav_selection == "📥 Sample Document Generator":
    st.header("Document Corpus & Sample Generator")
    st.markdown("""
    To test the classifier, you can generate fresh, realistic sample documents (PDF, DOCX, TXT) mapping to each of our 6 classes.
    Download these files locally, then upload them in the **🚀 Document Classifier** tab to test our engine.
    """)
    
    st.markdown("### Generate Fresh Documents")
    st.markdown("Clicking the button below regenerates and saves 6 new mock documents in the local storage, ready for download.")
    
    gen_button = st.button("Generate Fresh Sample Files")
    
    if gen_button:
        with st.spinner("Generating sample files..."):
            try:
                generate_all_samples(SAMPLES_DIR)
                st.success("🎉 Sample documents generated successfully!")
            except Exception as e:
                st.error(f"Error generating samples: {str(e)}")
                
    st.markdown("---")
    st.markdown("### Available Sample Documents for Download")
    
    # List available files
    if not os.path.exists(SAMPLES_DIR) or len(os.listdir(SAMPLES_DIR)) == 0:
        st.warning("No sample files found. Please click 'Generate Fresh Sample Files' above.")
    else:
        files = os.listdir(SAMPLES_DIR)
        
        # Categorized lists
        doc_cols = st.columns(3)
        
        for idx, file_name in enumerate(sorted(files)):
            file_path = os.path.join(SAMPLES_DIR, file_name)
            _, ext = os.path.splitext(file_name)
            
            # Format display details
            # e.g., sample_resume.pdf -> Resume (PDF)
            category_display = file_name.replace("sample_", "").split(".")[0].replace("_", " ").title()
            
            col_target = doc_cols[idx % 3]
            
            with col_target:
                st.markdown(f"""
                <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h5 style="margin-top: 0; margin-bottom: 5px; color: #1e293b;">{category_display}</h5>
                    <p style="margin: 0 0 10px 0; font-size: 0.8rem; color: #64748b;">Format: {ext.upper()[1:]} | Size: {os.path.getsize(file_path)/1024:.2f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add Streamlit native download button
                with open(file_path, "rb") as f:
                    file_data = f.read()
                    
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="application/octet-stream",
                    key=f"dl_{file_name}"
                )
                st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>DocuClassify AI | Built with Streamlit, Scikit-learn, and reportlab | 2026</p>", unsafe_allow_html=True)
