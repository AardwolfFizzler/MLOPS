import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def create_and_train_model():
    """
    Create and train a TensorFlow model for Iris classification.
    Save it in TensorFlow SavedModel format for TF Serving.
    """
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test accuracy: {test_acc:.4f}")
    
    # Save model in TensorFlow SavedModel format
    model_path = "../models/iris_classifier/1"
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("✅ Model saved successfully!")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = create_and_train_model()
