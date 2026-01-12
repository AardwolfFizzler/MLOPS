from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# TensorFlow Serving URL (uses Docker Compose service name)
TF_SERVING_URL = os.getenv('TF_SERVING_URL', 'http://tensorflow-serving:8501')

# Iris class names
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get features from request
        data = request.get_json()
        
        sepal_length = float(data.get('sepal_length', 0))
        sepal_width = float(data.get('sepal_width', 0))
        petal_length = float(data.get('petal_length', 0))
        petal_width = float(data.get('petal_width', 0))
        
        # Prepare request for TensorFlow Serving
        tf_request = {
            "instances": [[sepal_length, sepal_width, petal_length, petal_width]]
        }
        
        # Call TensorFlow Serving
        url = f"{TF_SERVING_URL}/v1/models/iris_classifier:predict"
        response = requests.post(url, json=tf_request)
        
        if response.status_code == 200:
            predictions = response.json()['predictions'][0]
            predicted_class = predictions.index(max(predictions))
            
            return jsonify({
                'success': True,
                'predicted_class': CLASS_NAMES[predicted_class],
                'predicted_index': predicted_class,
                'probabilities': {
                    CLASS_NAMES[i]: round(prob * 100, 2)
                    for i, prob in enumerate(predictions)
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'TensorFlow Serving error: {response.status_code}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if TF Serving is accessible
        url = f"{TF_SERVING_URL}/v1/models/iris_classifier"
        response = requests.get(url)
        
        if response.status_code == 200:
            return jsonify({
                'status': 'healthy',
                'tf_serving': 'connected'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'tf_serving': 'disconnected'
            }), 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
