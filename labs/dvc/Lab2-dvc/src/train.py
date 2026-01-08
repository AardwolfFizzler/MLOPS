"""
Model Training Script
Trains a classification model to predict rain tomorrow.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import importlib.util

def load_model_module(model_path):
    """Dynamically load the model module."""
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module

def load_and_split_data(data_path, test_size=0.2, random_state=42):
    """Load processed data and split into train/test sets."""
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Separate features and target
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    """Train the model."""
    print("\n Starting model training...")
    print("=" * 50)
    
    model.fit(X_train, y_train)
    
    print("=" * 50)
    print(" Training completed!")
    
    return model

def save_model(model, output_path):
    """Save the trained model."""
    print(f"\nSaving model to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    joblib.dump(model, output_path)
    print(f"Model saved successfully!")
    
    # Print model info
    print(f"\nModel type: {type(model).__name__}")
    print(f"Model parameters: {model.get_params()}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python train.py <data_path> <model_path> <n_estimators>")
        print("Example: python train.py ./data/weatherAUS_processed.csv ./src/model.py 200")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    n_estimators = int(sys.argv[3])
    
    output_path = "./models/model.joblib"
    
    print("=" * 50)
    print("  WEATHER PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # Load model module
    print("\n Loading model module...")
    model_module = load_model_module(model_path)
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = model_module.get_model(
        model_type='random_forest',
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42
    )
    
    # Train model
    trained_model = train_model(model, X_train, y_train)
    
    # Quick evaluation on training set
    train_score = trained_model.score(X_train, y_train)
    test_score = trained_model.score(X_test, y_test)
    
    print(f"\nQuick Evaluation:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Save model
    save_model(trained_model, output_path)
    
    print("\n" + "=" * 50)
    print(" Training pipeline completed successfully!")
    print("=" * 50)
    print(f"Input data: {data_path}")
    print(f"Model saved: {output_path}")
    print(f"Model: Random Forest with {n_estimators} trees")
    print("=" * 50)

if __name__ == "__main__":
    main()