# Lab 1: FastAPI Docker Deployment 

**What you'll learn:**
-  Building a REST API with FastAPI
-  Training and saving ML models with scikit-learn
-  Containerizing applications with Docker
-  Testing API endpoints
-  Best practices for ML model deployment

---

## Prerequisites

Before starting, ensure you have:
-  Python 3.9+ installed
-  Docker Desktop installed and running
-  Basic understanding of REST APIs
-  Terminal/command line familiarity

---

## Lab Structure

```
fastapi_docker/
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model/                 # Trained model storage
│   └── iris_model.pkl    # Trained model (you'll create this)
└── src/                   # Source code
    ├── __init__.py       # Package initialization
    ├── data.py           # Data loading utilities
    ├── train.py          # Model training script
    ├── predict.py        # Prediction logic
    └── main.py           # FastAPI application
```

---

## Step 1: Understand the Code

### 1.1 Review the Data Module (`src/data.py`)

This module handles loading and splitting the Iris dataset:
- `load_data()`: Loads the Iris dataset
- `split_data()`: Splits data into train/test sets (70/30 split)

### 1.2 Review the Training Script (`src/train.py`)

Trains a Decision Tree classifier and saves it:
- Uses max_depth=3 to prevent overfitting
- Saves model to `../model/iris_model.pkl`
- Uses joblib for serialization

### 1.3 Review the Prediction Module (`src/predict.py`)

Loads the trained model and makes predictions:
- Loads model from `../model/iris_model.pkl`
- Returns predicted class labels

### 1.4 Review the FastAPI Application (`src/main.py`)

The main API with two endpoints:
- `GET /`: Health check endpoint
- `POST /predict`: Prediction endpoint

**Input format:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Output format:**
```json
{
  "response": 0
}
```

Where `0` = Setosa, `1` = Versicolor, `2` = Virginica

---

## Step 2: Set Up Your Environment

### 2.1 Navigate to the Lab Directory

```bash
cd fastapi_docker
```

### 2.2 Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv fastapi_docker_env

# Activate it
# On macOS/Linux:
source fastapi_docker_env/bin/activate

# On Windows:
fastapi_docker_env\Scripts\activate
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `fastapi`: Web framework for building APIs
- `uvicorn`: ASGI server for running FastAPI
- `pydantic`: Data validation
- `scikit-learn`: Machine learning library
- `joblib`: Model serialization
- `numpy`: Numerical computations

---

## Step 3: Train the Model

### 3.1 Create the Model Directory

```bash
# Make sure you're in fastapi_docker/
mkdir -p model
```

### 3.2 Run the Training Script

```bash
cd src
python train.py
```

**Expected output:**
```
 Model trained and saved to ../model/iris_model.pkl
```

### 3.3 Verify the Model Was Created

```bash
cd ..
ls -lh model/
```

You should see `iris_model.pkl` (approximately 2-5 KB)

---

## Step 4: Test the API Locally

### 4.1 Start the Development Server

```bash
# Make sure you're in fastapi_docker/ with virtual environment active
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 4.2 Test the Health Endpoint

Open a new terminal and run:

```bash
curl http://localhost:8000/
```

**Expected response:**
```json
{"status":"healthy"}
```

### 4.3 Test the Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Expected response:**
```json
{"response":0}
```

This predicts class 0 (Setosa) - which is correct for these measurements!

### 4.4 Try Different Flower Types

**Versicolor (should predict class 1):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.4,
    "sepal_width": 3.2,
    "petal_length": 4.5,
    "petal_width": 1.5
  }'
```

**Virginica (should predict class 2):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 6.0,
    "petal_width": 2.5
  }'
```

### 4.5 Test with Interactive Docs

FastAPI automatically generates interactive API documentation!

1. Open your browser to: `http://localhost:8000/docs`
2. You'll see Swagger UI with your API endpoints
3. Click on `POST /predict`
4. Click "Try it out"
5. Enter values and click "Execute"

**Stop the server:** Press `Ctrl+C` in the terminal

---

## Step 5: Containerize with Docker

### 5.1 Review the Dockerfile

Open `Dockerfile` and understand each instruction:

```dockerfile
FROM python:3.9-slim          # Base image
WORKDIR /app                  # Set working directory
COPY requirements.txt .       # Copy dependencies
RUN pip install ...           # Install dependencies
COPY src/ ./src/              # Copy source code
COPY model/ ./model/          # Copy trained model
EXPOSE 8000                   # Expose port
CMD ["uvicorn", ...]          # Run the server
```

### 5.2 Build the Docker Image

```bash
# Make sure you're in fastapi_docker/
docker build -t iris-fastapi:v1 .
```

**What's happening:**
- `-t iris-fastapi:v1`: Tags the image with name and version
- `.`: Uses current directory as build context

**Expected output:**
```
[+] Building 12.3s (11/11) FINISHED
 => [1/6] FROM docker.io/library/python:3.9-slim
 => [2/6] WORKDIR /app
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt
 => [5/6] COPY src/ ./src/
 => [6/6] COPY model/ ./model/
 => exporting to image
Successfully tagged iris-fastapi:v1
```

### 5.3 Verify the Image Was Created

```bash
docker images | grep iris-fastapi
```

**Expected output:**
```
iris-fastapi   v1   <image-id>   2 minutes ago   450MB
```

---

## Step 6: Run the Dockerized Application

### 6.1 Start the Container

```bash
docker run -d -p 8000:8000 --name iris-api iris-fastapi:v1
```

**Flags explained:**
- `-d`: Run in detached mode (background)
- `-p 8000:8000`: Map host port 8000 to container port 8000
- `--name iris-api`: Name the container
- `iris-fastapi:v1`: Use this image

### 6.2 Check Container Status

```bash
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE              COMMAND                  STATUS         PORTS
<id>           iris-fastapi:v1    "uvicorn src.main:a…"    Up 10 seconds  0.0.0.0:8000->8000/tcp
```

### 6.3 View Container Logs

```bash
docker logs iris-api
```

**Expected output:**
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.4 Test the Containerized API

```bash
# Health check
curl http://localhost:8000/

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

---

## Step 7: Container Management

### 7.1 Stop the Container

```bash
docker stop iris-api
```

### 7.2 Restart the Container

```bash
docker start iris-api
```

### 7.3 Remove the Container

```bash
# Stop first if running
docker stop iris-api

# Remove
docker rm iris-api
```

### 7.4 Remove the Image

```bash
docker rmi iris-fastapi:v1
```

---

## Step 8: Advanced Testing

### 8.1 Test Error Handling

Send invalid data:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": "invalid"
  }'
```

## Troubleshooting

### Issue: "Port 8000 is already in use"

**Solution:**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
docker run -d -p 8001:8000 --name iris-api iris-fastapi:v1
```

### Issue: "Model file not found"

**Solution:**
```bash
# Make sure you trained the model first
cd src
python train.py
cd ..
```

### Issue: "Docker build fails"

**Solution:**
```bash
# Clear Docker cache and rebuild
docker builder prune
docker build --no-cache -t iris-fastapi:v1 .
```

### Issue: "Import errors when running locally"

**Solution:**
```bash
# Make sure virtual environment is activated
source fastapi_docker_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Understanding the Results

### Iris Classes

- **0 = Setosa**: Small petals (< 2 cm length)
- **1 = Versicolor**: Medium petals (3-5 cm length)
- **2 = Virginica**: Large petals (> 5 cm length)

### Typical Measurements

| Feature | Setosa | Versicolor | Virginica |
|---------|--------|------------|-----------|
| Sepal Length | 4.5-5.5 | 5.5-6.5 | 6.0-7.5 |
| Sepal Width | 3.0-4.0 | 2.5-3.0 | 2.8-3.5 |
| Petal Length | 1.0-2.0 | 3.5-5.0 | 5.0-6.5 |
| Petal Width | 0.1-0.5 | 1.0-1.5 | 1.5-2.5 |

---

## Clean Up

When you're done with the lab:

```bash
# Deactivate virtual environment
deactivate

# Stop and remove container
docker stop iris-api
docker rm iris-api

# Remove image (optional)
docker rmi iris-fastapi:v1

# Remove virtual environment (optional)
rm -rf fastapi_docker_env
```

---

## Key Takeaways

**FastAPI** provides automatic API documentation and validation  
**Docker** ensures consistent deployment across environments  
**Model serialization** (joblib) allows saving/loading trained models  
**REST APIs** are a standard way to serve ML models  
**Container orchestration** simplifies scaling and management  

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Iris Dataset Info](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

---
**Happy deploying! **