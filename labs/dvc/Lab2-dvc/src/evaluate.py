"""
Model Evaluation Script
Evaluates the trained model and generates metrics and visualizations.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import importlib.util

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_model_module(model_path):
    """Dynamically load the model module."""
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module

def load_data(data_path):
    """Load processed data and split into test set."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']
    
    # Split data (same split as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set shape: {X_test.shape}")
    return X_test, y_test

def load_model(model_path):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")
    return model

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    print("\n Calculating metrics...")
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred))
    }
    
    print("\n" + "=" * 50)
    print(" EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.5f}")
    print(f"Precision: {metrics['precision']:.5f}")
    print(f"Recall:    {metrics['recall']:.5f}")
    print(f"F1 Score:  {metrics['f1']:.5f}")
    print("=" * 50)
    
    return metrics

def save_metrics(metrics, output_path):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n Metrics saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {output_path}")

def plot_precision_recall_curve(y_true, y_pred_proba, output_path):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Precision-Recall curve saved to {output_path}")

def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("\n" + "=" * 50)
    print(" DETAILED CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, 
                                target_names=['No Rain', 'Rain']))
    print("=" * 50)

def main():
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <data_path> <model_def_path> <model_path>")
        print("Example: python evaluate.py ./data/weatherAUS_processed.csv ./src/model.py ./models/model.joblib")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_def_path = sys.argv[2]
    model_path = sys.argv[3]
    
    # Output paths
    metrics_path = "./results/metrics.json"
    confusion_matrix_path = "./results/confusion_matrix.png"
    roc_curve_path = "./results/roc_curve.png"
    pr_curve_path = "./results/precision_recall_curve.png"
    
    print("=" * 50)
    print(" MODEL EVALUATION")
    print("=" * 50)
    
    # Load data and model
    X_test, y_test = load_data(data_path)
    model = load_model(model_path)
    
    # Make predictions
    print("\n Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate and save metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    save_metrics(metrics, metrics_path)
    
    # Print detailed report
    print_classification_report(y_test, y_pred)
    
    # Generate visualizations
    print("\n Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, confusion_matrix_path)
    plot_roc_curve(y_test, y_pred_proba, roc_curve_path)
    plot_precision_recall_curve(y_test, y_pred_proba, pr_curve_path)
    
    print("\n" + "=" * 50)
    print(" Evaluation completed successfully!")
    print("=" * 50)
    print(f"Metrics saved: {metrics_path}")
    print(f"Confusion Matrix: {confusion_matrix_path}")
    print(f"ROC Curve: {roc_curve_path}")
    print(f"Precision-Recall Curve: {pr_curve_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()