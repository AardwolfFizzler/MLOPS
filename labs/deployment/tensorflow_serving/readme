# Lab 2: TensorFlow Serving Deployment 

## Overview

In this lab, you'll learn how to deploy TensorFlow models using TensorFlow Serving - Google's production-grade serving system. You'll train a neural network, serve it through TF Serving, and build a web interface for making predictions.

**What you'll learn:**
-  Training neural networks with TensorFlow/Keras
-  Saving models in TensorFlow SavedModel format
-  Deploying models with TensorFlow Serving
-  Multi-container applications with Docker Compose
-  Building web UIs for ML applications
-  Production-grade model serving

**Time estimate:** 60-90 minutes

---

## Prerequisites

Before starting, ensure you have:
-  Python 3.9+ installed
-  Docker Desktop installed and running
-  Basic understanding of neural networks
-  Completion of Lab 1 (recommended)

---

## Lab Structure

```
tensorflow_serving/
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/
│   ├── models.config         # TF Serving configuration
│   └── iris_classifier/
│       └── 1/                # Model version 1
│           ├── saved_model.pb
│           ├── variables/
│           └── assets/
├── src/
│   ├── __init__.py
│   ├── train_tf_model.py    # Neural network training
│   └── client.py            # Python client for testing
└── web_ui/
    ├── Dockerfile
    ├── requirements.txt
    ├── app.py              # Flask backend
    ├── static/
    │   └── style.css       # UI styling
    └── templates/
        └── index.html      # Web interface
```

---

## Architecture Overview

This lab uses a microservices architecture:

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Browser   │ ───▶ │   Flask Web UI   │ ───▶ │  TF Serving API │
│  (Port 5000)│ ◀─── │   (Port 5000)    │ ◀─── │  (Port 8501)    │
└─────────────┘      └──────────────────┘      └─────────────────┘
                              │                          │
                              └──────────────────────────┘
                                 Docker Network
```

**Components:**
1. **TensorFlow Serving**: Serves the trained model via REST API
2. **Flask Web UI**: Provides user-friendly interface
3. **Docker Network**: Connects the services

---

## Step 1: Understand the Code

### 1.1 Review the Training Script (`src/train_tf_model.py`)

This script creates and trains a neural network:

**Architecture:**
```
Input (4 features) 
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Dense Layer (8 neurons, ReLU)
    ↓
Output (3 classes, Softmax)
```

**Key steps:**
1. Load and preprocess Iris dataset
2. Standardize features (mean=0, std=1)
3. Build neural network with Keras
4. Train for 50 epochs
5. Save in TensorFlow SavedModel format

**SavedModel format** is TensorFlow's standard for serving models. It includes:
- Model architecture
- Trained weights
- Computation graph
- Serving signatures

### 1.2 Review the Client Script (`src/client.py`)

Tests the TF Serving API:
- Sends POST requests to TF Serving REST endpoint
- Parses predictions and displays results
- Tests with samples from each Iris class

### 1.3 Review the Flask Application (`web_ui/app.py`)

Provides three endpoints:
- `GET /`: Renders the web interface
- `POST /predict`: Handles prediction requests
- `GET /health`: Health check for both services

---

## Step 2: Set Up Your Environment

### 2.1 Navigate to the Lab Directory

```bash
cd tensorflow_serving
```

### 2.2 Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv tf_serving_env

# Activate it
# On macOS/Linux:
source tf_serving_env/bin/activate

# On Windows:
tf_serving_env\Scripts\activate
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `tensorflow`: Deep learning framework
- `scikit-learn`: For data loading and preprocessing
- `numpy`: Numerical computations
- `requests`: HTTP client for testing

---

## Step 3: Train the TensorFlow Model

### 3.1 Verify Model Directory Structure

```bash
# Check if models directory exists
ls -la models/
```

You should see `models.config` but no `iris_classifier/` yet.

### 3.2 Run the Training Script

```bash
cd src
python train_tf_model.py
```

**Expected output:**
```
Training model...
Epoch 1/50
7/7 [==============================] - 0s 3ms/step - loss: 1.0234 - accuracy: 0.3571
Epoch 2/50
7/7 [==============================] - 0s 2ms/step - loss: 0.9234 - accuracy: 0.5714
...
Epoch 50/50
7/7 [==============================] - 0s 2ms/step - loss: 0.1234 - accuracy: 0.9643

 Test accuracy: 0.9778

Saving model to ../models/iris_classifier/1...
 Model saved successfully!
```

### 3.3 Verify the SavedModel

```bash
cd ..
ls -lh models/iris_classifier/1/
```

**Expected output:**
```
saved_model.pb      # Protocol buffer with model graph
variables/          # Model weights
  variables.data-00000-of-00001
  variables.index
assets/            # Additional assets (may be empty)
```

### 3.4 Inspect the SavedModel (Optional)

```bash
# Install saved_model_cli (comes with TensorFlow)
saved_model_cli show --dir models/iris_classifier/1 --all
```

This shows the model's input/output signatures.

---

## Step 4: Understanding TensorFlow Serving Configuration

### 4.1 Review `models/models.config`

```
model_config_list {
  config {
    name: 'iris_classifier'              # Model name
    base_path: '/models/iris_classifier' # Path inside container
    model_platform: 'tensorflow'          # Platform type
    model_version_policy {
      all: {}                            # Serve all versions
    }
  }
}
```

**Key concepts:**
- **Model versioning**: Each version in a numbered folder (1, 2, 3...)
- **base_path**: Points to the folder containing versions
- **model_version_policy**: Controls which versions to serve
  - `all: {}`: Serves all versions
  - `specific: {versions: [1, 3]}`: Serves only versions 1 and 3
  - `latest: {num_versions: 2}`: Serves the 2 most recent versions

### 4.2 Review `docker-compose.yml`

Two services defined:

**TensorFlow Serving:**
```yaml
tensorflow-serving:
  image: tensorflow/serving:latest
  ports:
    - "8501:8501"  
    - "8500:8500"  
  volumes:
    - ./models:/models  # Mount models directory
  environment:
    - MODEL_NAME=iris_classifier
```

**Web UI:**
```yaml
web-ui:
  build: ./web_ui
  ports:
    - "5000:5000"
  environment:
    - TF_SERVING_URL=http://tensorflow-serving:8501
  depends_on:
    - tensorflow-serving
```

**Key points:**
- Services communicate via Docker network
- Web UI uses service name `tensorflow-serving` (Docker DNS)

---

## Step 5: Start TensorFlow Serving

### 5.1 Build and Start All Services

```bash
# Make sure you're in tensorflow_serving/
docker-compose up --build -d
```

**Flags explained:**
- `--build`: Build images before starting
- `-d`: Run in detached mode (background)

**Expected output:**
```
[+] Building 15.2s (10/10) FINISHED
[+] Running 3/3
 ✔ Network tensorflow_serving_mlops_network    Created
 ✔ Container tf_serving_iris                   Started
 ✔ Container iris_web_ui                       Started
```

### 5.2 Check Container Status

```bash
docker-compose ps
```

**Expected output:**
```
NAME              IMAGE                        STATUS        PORTS
tf_serving_iris   tensorflow/serving:latest    Up 30 seconds 0.0.0.0:8500-8501->8500-8501/tcp
iris_web_ui       tensorflow_serving-web-ui    Up 28 seconds 0.0.0.0:5000->5000/tcp
```

### 5.3 View Logs

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service
docker-compose logs tensorflow-serving
docker-compose logs web-ui
```

**Look for in TF Serving logs:**
```
Successfully loaded servable version {name: iris_classifier version: 1}
```

---

## Step 6: Test TensorFlow Serving API

### 6.1 Check Model Status

```bash
curl http://localhost:8501/v1/models/iris_classifier
```

**Expected response:**
```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

### 6.2 Get Model Metadata

```bash
curl http://localhost:8501/v1/models/iris_classifier/metadata
```

This shows input/output tensor specifications.

### 6.3 Test Prediction via REST API

```bash
curl -X POST http://localhost:8501/v1/models/iris_classifier:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [5.1, 3.5, 1.4, 0.2]
    ]
  }'
```

**Expected response:**
```json
{
  "predictions": [
    [0.98, 0.01, 0.01]
  ]
}
```

This means: 98% Setosa, 1% Versicolor, 1% Virginica

### 6.4 Test with the Python Client

```bash
# Activate virtual environment if not already
source tf_serving_env/bin/activate

cd src
python client.py
```

**Expected output:**
```
Testing TensorFlow Serving predictions...

==================================================
Testing sample 0
Features: [5.1, 3.5, 1.4, 0.2]
True class: setosa
==================================================
Predicted Class: Setosa
Confidence: 0.9823
All probabilities: [0.9823, 0.0089, 0.0088]

==================================================
Testing sample 50
Features: [7.0, 3.2, 4.7, 1.4]
True class: versicolor
==================================================
Predicted Class: Versicolor
Confidence: 0.9456
All probabilities: [0.0123, 0.9456, 0.0421]

==================================================
Testing sample 100
Features: [6.3, 3.3, 6.0, 2.5]
True class: virginica
==================================================
Predicted Class: Virginica
Confidence: 0.9789
All probabilities: [0.0067, 0.0144, 0.9789]
```

---

## Step 7: Use the Web Interface

### 7.1 Open the Web UI

Open your browser and go to:
```
http://localhost:5000
```

### 7.2 Test the Interface

**Try these sample inputs:**

**Setosa:**
- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.4
- Petal Width: 0.2

**Versicolor:**
- Sepal Length: 6.4
- Sepal Width: 3.2
- Petal Length: 4.5
- Petal Width: 1.5

**Virginica:**
- Sepal Length: 6.3
- Sepal Width: 3.3
- Petal Length: 6.0
- Petal Width: 2.5

### 7.3 Understanding the Results

The UI displays:
1. **Predicted Class**: The most likely flower type
2. **Confidence Bars**: Probability for each class
3. **Color Coding**: Green for high confidence, yellow for medium, red for low

### 7.4 Check Health Endpoint

```bash
curl http://localhost:5000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "tf_serving": "connected"
}
```
---

## Step 8: Container Management

### 8.1 View Container Logs

```bash
# All logs
docker-compose logs

# Specific service
docker-compose logs web-ui
docker-compose logs tensorflow-serving

# Follow logs (Ctrl+C to stop)
docker-compose logs -f
```

### 8.2 Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart web-ui
```

### 8.3 Stop Services

```bash
# Stop all services (keeps containers)
docker-compose stop

# Start again
docker-compose start
```

### 8.4 Tear Down Everything

```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes
docker-compose down -v

# Remove images too
docker-compose down --rmi all
```

---

## Troubleshooting

### Issue: "Model not found" or "Model not available"

**Solution:**
```bash
# Check if model exists
ls -la models/iris_classifier/1/

# Ensure saved_model.pb is present
# If not, retrain the model

# Check TF Serving logs
docker-compose logs tensorflow-serving | grep -i error
```

### Issue: "Web UI can't connect to TF Serving"

**Solution:**
```bash
# Check if both containers are running
docker-compose ps

# Test TF Serving directly
curl http://localhost:8501/v1/models/iris_classifier

# Check network connectivity
docker-compose exec web-ui ping tensorflow-serving
```

### Issue: "Port already in use"

**Solution:**
```bash
# Find and kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or modify docker-compose.yml to use different ports
ports:
  - "5001:5000"  # Change host port to 5001
```

### Issue: "TensorFlow installation issues"

**Solution:**
```bash
# Use specific TensorFlow version
pip install tensorflow==2.15.0

# Or use TensorFlow CPU-only (smaller)
pip install tensorflow-cpu==2.15.0
```

### Issue: "Containers keep restarting"

**Solution:**
```bash
# Check logs for errors
docker-compose logs

# Remove and rebuild
docker-compose down
docker-compose up --build
```

---

## Performance Considerations

### Model Loading Time

TensorFlow Serving loads models on startup:
- **Small models** (< 10 MB): ~1-2 seconds
- **Medium models** (10-100 MB): ~5-10 seconds
- **Large models** (> 100 MB): ~30+ seconds

---

## Clean Up

When you're done:

```bash
# Stop and remove containers
docker-compose down

# Remove images (optional)
docker rmi tensorflow_serving-web-ui
docker rmi tensorflow/serving:latest

# Deactivate virtual environment
deactivate

# Remove virtual environment (optional)
rm -rf tf_serving_env
```

---

## Key Takeaways

 **TensorFlow Serving** is production-grade serving system  
 **SavedModel format** is standard for TensorFlow deployment  
 **Model versioning** enables safe updates and A/B testing  
 **Docker Compose** simplifies multi-container applications  
 **REST and gRPC APIs** offer flexibility and performance  
 **Microservices architecture** enables scalable ML systems  

---


## Resources

- [TensorFlow Serving Documentation](https://www.tensorflow.org/tfx/guide/serving)
- [TensorFlow SavedModel Guide](https://www.tensorflow.org/guide/saved_model)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [gRPC in TF Serving](https://www.tensorflow.org/tfx/serving/api_rest)

---

**Happy serving! **