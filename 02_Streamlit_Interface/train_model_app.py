import streamlit as st
import json
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Get the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_DIR = os.path.join(CURRENT_DIR, 'head_gestures')
GESTURE_JSON = os.path.join(GESTURE_DIR, 'head_gestures.json')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')

# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    """Load and prepare gesture data"""
    with open(GESTURE_JSON, 'r') as f:
        gesture_db = json.load(f)
    X = []
    y = []
    timestamps = []
    for gesture in gesture_db:
        pose = gesture['head_pose']
        features = [pose['yaw'], pose['pitch'], pose['roll']]
        X.append(features)
        y.append(gesture['name'])
        timestamps.append(gesture['timestamp'])
    return np.array(X), np.array(y), timestamps

def plot_confusion_matrix(y_true, y_pred, classes):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

def plot_feature_distribution(X, y, feature_names):
    """Create feature distribution plots"""
    df = pd.DataFrame(X, columns=feature_names)
    df['Gesture'] = y
    
    figs = []
    for feature in feature_names:
        fig = px.box(df, x='Gesture', y=feature, title=f'{feature} Distribution by Gesture')
        figs.append(fig)
    return figs

def main():
    st.title("🤖 Head Gesture Model Training")
    st.markdown("""
    This application trains an SVM model to recognize head gestures for Subway Surfers control.
    The model uses head pose angles (yaw, pitch, roll) as features.
    """)

    # Load data
    try:
        X, y, timestamps = load_data()
        st.success(f"Successfully loaded {len(X)} samples from the dataset!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Please make sure you have captured gesture data first!")
        return

    # Data Overview
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Number of Features", X.shape[1])
    with col3:
        st.metric("Unique Gestures", len(np.unique(y)))

    # Show class distribution
    st.subheader("Class Distribution")
    class_counts = pd.Series(y).value_counts()
    fig = px.bar(x=class_counts.index, y=class_counts.values,
                labels={'x': 'Gesture', 'y': 'Count'},
                title='Number of Samples per Gesture')
    st.plotly_chart(fig)

    # Feature Distribution
    st.subheader("Feature Distributions")
    feature_names = ['Yaw', 'Pitch', 'Roll']
    dist_figs = plot_feature_distribution(X, y, feature_names)
    for fig in dist_figs:
        st.plotly_chart(fig)

    # Training Parameters
    st.header("🎯 Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        n_folds = st.slider("Number of Cross-validation Folds", min_value=2, max_value=10, value=5)
        kernel = st.selectbox("SVM Kernel", options=['rbf', 'linear', 'poly'], index=0)
    
    with col2:
        random_state = st.number_input("Random State", min_value=0, value=42)
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    # Train Model Button
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            fold_accuracies = []
            fold_reports = []
            all_y_true = []
            all_y_pred = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded), 1):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                
                model = SVC(kernel=kernel, probability=True, random_state=random_state)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                fold_accuracies.append(acc)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                
                report = classification_report(y_test, y_pred, 
                                            target_names=label_encoder.classes_,
                                            output_dict=True)
                fold_reports.append((fold, acc, report))
                
                progress_bar.progress(fold / n_folds)
                status_text.text(f"Processing fold {fold}/{n_folds}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Display Results
            st.header("📈 Training Results")
            
            # Overall Metrics
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Accuracy", f"{mean_acc:.2%}")
            with col2:
                st.metric("Standard Deviation", f"±{std_acc:.2%}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(all_y_true, all_y_pred, label_encoder.classes_)
            st.pyplot(cm_fig)
            
            # Detailed Results per Fold
            st.subheader("Results per Fold")
            for fold, acc, report in fold_reports:
                with st.expander(f"Fold {fold} (Accuracy: {acc:.2%})"):
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
            
            # Train final model on all data
            final_model = SVC(kernel=kernel, probability=True, random_state=random_state)
            final_model.fit(X_scaled, y_encoded)
            
            # Save models
            joblib.dump(final_model, os.path.join(MODELS_DIR, 'gesture_svm.pkl'))
            joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
            joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
            
            st.success("✅ Model training completed! Models saved in the 'models' directory.")

if __name__ == "__main__":
    st.set_page_config(page_title="Head Gesture Model Training", page_icon="🤖", layout="wide")
    main() 