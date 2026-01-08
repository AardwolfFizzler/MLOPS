# Lab 4: MLflow Tutorial - Wine Quality Prediction

## Overview
This lab introduces MLflow, an open-source platform for managing the machine learning lifecycle. You'll train models to predict wine quality, track experiments, compare results, and deploy your best model.

## Learning Objectives
- Track ML experiments with MLflow (parameters, metrics, artifacts)
- Compare different model configurations
- Visualize results in the MLflow UI
- Deploy models for real-time predictions
- Understand regularization in machine learning

## Prerequisites
- Python 3.7+
- Basic understanding of machine learning concepts
- Familiarity with command line/terminal

## Lab Structure
```
lab4-mlflow-1/
├── README.md
├── sklearn_elasticnet_wine/
│   ├── train.py              # Main training script
│   ├── MLproject.yml          # MLflow project configuration
│   ├── python_env.yaml        # Python dependencies
│   └── mlruns/               # MLflow tracking data (created automatically)
└── venv/                     # Virtual environment (already set up)
```

---

## Setup Instructions

### Step 1: Create to the Project Directory

**macOS/Linux:**
```bash
cd ~/MLOPS-Repo/labs/lab4-mlflow-1/sklearn_elasticnet_wine
```

**Windows (Command Prompt):**
```cmd
cd %USERPROFILE%\MLOPS-Repo\labs\lab4-mlflow-1\sklearn_elasticnet_wine
```

**Windows (PowerShell):**
```powershell
cd $HOME\MLOPS-Repo\labs\lab4-mlflow-1\sklearn_elasticnet_wine
```

### Step 2: Activate Virtual Environment

**macOS/Linux:**
```bash
# Go back to lab4-mlflow-1 directory
cd ~/MLOPS-Repo/labs/lab4-mlflow-1
source venv/bin/activate
cd sklearn_elasticnet_wine
```

**Windows (Command Prompt):**
```cmd
cd %USERPROFILE%\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\activate.bat
cd sklearn_elasticnet_wine
```

**Windows (PowerShell):**
```powershell
cd $HOME\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\Activate.ps1
cd sklearn_elasticnet_wine
```

You should see `(venv)` in your terminal prompt when activated.

### Step 3: Verify Installation

**All platforms:**
```bash
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
```

If you get import errors, install the required packages:
```bash
pip install mlflow==2.8.1 scikit-learn pandas numpy matplotlib seaborn
```

---

##  Part 1: Running Your First Experiment

### Understanding the Model
We're using **ElasticNet**, a linear regression model with regularization. It has two key parameters:

- **alpha**: Regularization strength (0.01 to 1.0)
  - Higher values = simpler model (more regularization)
  - Lower values = more complex model (less regularization)

- **l1_ratio**: Mix between L1 and L2 regularization (0.0 to 1.0)
  - 0.0 = Pure Ridge (L2) - keeps all features
  - 1.0 = Pure Lasso (L1) - can eliminate features
  - 0.5 = Equal mix of both

### Experiment 1: Default Parameters

Run the training script with default parameters:

**All platforms:**
```bash
python train.py
```

**Expected Output:**
```
Loading wine quality dataset...
Dataset loaded: 1599 rows, 12 columns

Training set: 1199 samples
Test set: 400 samples

 Training ElasticNet model with alpha=0.5, l1_ratio=0.5

 Model Performance:
  RMSE: 0.7xxx
  MAE:  0.5xxx
  R2:   0.2xxx

 Model logged to MLflow
 Creating feature importance plot...
 Feature importance plot saved and logged
 Predictions plot saved and logged

Run completed! Check MLflow UI to see your results.
```

**What Just Happened?**
- Downloaded the Wine Quality dataset (1,599 wine samples)
- Split data into training (75%) and test (25%) sets
- Trained an ElasticNet model
- Evaluated performance with three metrics:
  - **RMSE** (Root Mean Squared Error): Lower is better
  - **MAE** (Mean Absolute Error): Lower is better
  - **R²** (R-squared): Higher is better (max = 1.0)
- Saved everything to MLflow for tracking

### Experiment 2-6: Try Different Parameters

Run these experiments to see how parameters affect performance:

**Experiment 2: Lower regularization (more complex model)**
```bash
python train.py --alpha 0.1 --l1_ratio 0.5
```

**Experiment 3: Higher regularization (simpler model)**
```bash
python train.py --alpha 1.0 --l1_ratio 0.5
```

**Experiment 4: Pure Lasso (L1 only)**
```bash
python train.py --alpha 0.5 --l1_ratio 1.0
```

**Experiment 5: Pure Ridge (L2 only)**
```bash
python train.py --alpha 0.5 --l1_ratio 0.0
```

**Experiment 6: Very low regularization**
```bash
python train.py --alpha 0.01 --l1_ratio 0.5
```

---

## Part 2: Visualizing Results with MLflow UI

### Step 1: Start the MLflow UI

Make sure you're in the `sklearn_elasticnet_wine` directory:

**All platforms:**
```bash
mlflow ui
```

**Expected Output:**
```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://127.0.0.1:5000
```

**Important:** Leave this terminal window open while using the UI!

### Step 2: Open the UI in Your Browser

Navigate to: **http://localhost:5000**

### Step 3: Explore Your Experiments

#### View All Runs
- Main page shows all your experiment runs
- Each row represents one training run
- Columns show parameters (alpha, l1_ratio) and metrics (RMSE, MAE, R2)

#### Compare Multiple Runs
1. Check the boxes next to 2 or more runs
2. Click the **"Compare"** button
3. See side-by-side comparison:
   - Parameter differences
   - Metric differences
   - Visualizations

#### View Run Details
1. Click on any run's timestamp
2. Explore the tabs:
   - **Parameters**: All hyperparameters used
   - **Metrics**: Performance measurements
   - **Artifacts**: Model files and plots
   - **Tags**: Additional metadata

#### View Artifacts (Plots)
1. Click on a run
2. Scroll to the "Artifacts" section
3. Click to view:
   - `feature_importance.png` - Which wine features matter most
   - `predictions_vs_actual.png` - Model prediction accuracy

---

## Part 3: Understanding Your Results

### Interpreting Metrics

| Metric | What It Means | Typical Range | Goal |
|--------|---------------|---------------|------|
| **RMSE** | Average prediction error (same units as target) | 0.6 - 0.8 | Lower is better |
| **MAE** | Average absolute error | 0.5 - 0.6 | Lower is better |
| **R²** | Percentage of variance explained | 0.2 - 0.4 | Higher is better |

### Parameter Impact

**Alpha (Regularization Strength):**
- **High alpha (0.8-1.0)**: Simple model, may underfit
- **Medium alpha (0.3-0.5)**: Balanced complexity
- **Low alpha (0.01-0.1)**: Complex model, may overfit

**L1 Ratio:**
- **0.0 (Pure Ridge)**: All features retained, weights distributed
- **0.5 (Mixed)**: Balance between feature selection and retention
- **1.0 (Pure Lasso)**: Some features eliminated, sparse model
---

##  Part 4: Finding Your Best Model

### Focused Search

Based on initial results, narrow down to promising regions:

```bash
# Example: If low alpha works well, try variations
python train.py --alpha 0.05 --l1_ratio 0.3
python train.py --alpha 0.08 --l1_ratio 0.4
python train.py --alpha 0.12 --l1_ratio 0.6
```

---

## Part 5: Deploying Your Best Model

### Step 1

1. Open MLflow UI (http://localhost:5000)
2. Click on the best performing run
3. Copy the **Run ID** from the URL

Example URL: `http://localhost:5000/#/experiments/0/runs/a1b2c3d4e5f6g7h8`  
Run ID: `a1b2c3d4e5f6g7h8`

### Step 2: Serve the Model

**macOS/Linux:**
```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 1234
```

**Windows (all):**
```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 1234
```

Replace `<RUN_ID>` with your actual Run ID. Example:
```bash
mlflow models serve -m runs:/a1b2c3d4e5f6g7h8/model -p 1234
```

**Expected Output:**
```
[INFO] Listening at: http://127.0.0.1:1234
```

**Leave this terminal running!**

### Step 3: Make Predictions (New Terminal)

Open a **new terminal/command prompt** and activate venv again:

**macOS/Linux:**
```bash
cd ~/MLOPS-Repo/labs/lab4-mlflow-1
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
cd %USERPROFILE%\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
cd $HOME\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\Activate.ps1
```

Then test the model:

**macOS/Linux:**
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" \
  --data '{
    "columns": [
      "fixed acidity", "volatile acidity", "citric acid",
      "residual sugar", "chlorides", "free sulfur dioxide",
      "total sulfur dioxide", "density", "pH",
      "sulphates", "alcohol"
    ],
    "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
  }' \
  http://127.0.0.1:1234/invocations
```

**Windows (PowerShell):**
```powershell
$body = @{
    columns = @(
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH",
        "sulphates", "alcohol"
    )
    data = @(@(7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4))
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:1234/invocations -Method Post -ContentType "application/json; format=pandas-split" -Body $body
```

**Windows (Command Prompt):** Use PowerShell instead, or install `curl` for Windows.

**Expected Response:**
```json
[5.340615087747574]
```

This predicts a wine quality score of ~5.3 (on a scale of 0-10).

---

## Troubleshooting

### Issue: Virtual Environment Not Activated

**Symptom:** `(venv)` not showing in prompt

**Solution:**

**macOS/Linux:**
```bash
cd ~/MLOPS-Repo/labs/lab4-mlflow-1
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
cd %USERPROFILE%\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
cd $HOME\MLOPS-Repo\labs\lab4-mlflow-1
venv\Scripts\Activate.ps1
```

If PowerShell gives execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Module Not Found Error

**Solution:**
```bash
pip install mlflow scikit-learn pandas numpy matplotlib seaborn
```

### Issue: MLflow UI Port 5000 Already in Use

**macOS/Linux:**
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9

# Or use different port
mlflow ui --port 5001
```

**Windows:**
```cmd
# Find process
netstat -ano | findstr :5000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
mlflow ui --port 5001
```

### Issue: Cannot Download Dataset

**Solution:** Check internet connection. If still failing, download manually:

**macOS/Linux:**
```bash
curl -o winequality-red.csv "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" -OutFile "winequality-red.csv"
```

---

## Key Concepts Learned

**Experiment Tracking**: Logging parameters, metrics, and artifacts  
**Model Comparison**: Using MLflow UI to compare runs  
**Regularization**: Understanding alpha and l1_ratio effects  
**Model Deployment**: Serving models via REST API  

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ElasticNet in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Understanding Regularization](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

---



**Good luck! **