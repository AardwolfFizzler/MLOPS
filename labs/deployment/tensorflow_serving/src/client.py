import requests
import json
import numpy as np
from sklearn.datasets import load_iris

def predict_with_tf_serving(features, server_url="http://localhost:8501"):
    """
    Send prediction request to TensorFlow Serving.
    
    Args:
        features: List of feature values [sepal_length, sepal_width, petal_length, petal_width]
        server_url: TensorFlow Serving REST API URL
    
    Returns:
        Prediction result
    """
    # Prepare the request payload
    data = {
        "instances": [features]
    }
    
    # Send POST request
    url = f"{server_url}/v1/models/iris_classifier:predict"
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        predictions = response.json()['predictions'][0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {predictions}")
        
        return predicted_class, predictions
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def test_multiple_samples():
    """Test with multiple samples from the Iris dataset."""
    iris = load_iris()
    
    # Test with a few samples
    test_indices = [0, 50, 100]  # One from each class
    
    for idx in test_indices:
        features = iris.data[idx].tolist()
        true_class = iris.target_names[iris.target[idx]]
        
        print(f"\n{'='*50}")
        print(f"Testing sample {idx}")
        print(f"Features: {features}")
        print(f"True class: {true_class}")
        print(f"{'='*50}")
        
        predict_with_tf_serving(features)

if __name__ == "__main__":
    print("Testing TensorFlow Serving predictions...\n")
    test_multiple_samples()
