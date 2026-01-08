# Lab 2: Model Training & Versioning with GitHub Actions

## Learning Objectives

- Automate ML model training with GitHub Actions
- Version trained models automatically
- Evaluate model performance in CI/CD pipelines
- Store and track model artifacts

## Prerequisites

- Python 3.8+
- Git installed
- GitHub account
- Completed Lab 1

## Setup Instructions

### 1. Navigate to Lab 2

**Mac:**
```bash
cd ~/path/to/MLOPS-Repo/labs/lab2-model-training-versioning
```

**Windows:**
```bash
cd C:\path\to\MLOPS-Repo\labs\lab2-model-training-versioning
```

### 2. Create Virtual Environment

**Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Required Directories

**Mac/Windows:**
```bash
mkdir -p data models metrics
```

## Project Structure

```
lab2-model-training-versioning/
├── src/
│   ├── train_model.py      # Model training script
│   └── evaluate_model.py   # Model evaluation script
├── data/                    # Training data (generated)
├── models/                  # Trained models (versioned)
├── metrics/                 # Evaluation metrics
├── test/                    # Test files
├── requirements.txt
└── README.md
```

**Note:** The `.github/workflows/` folder is at the repository root, not in this lab directory.

## Running Locally

### Train a Model

**Mac/Windows:**
```bash
# Generate timestamp
timestamp=$(date '+%Y%m%d%H%M%S')

# Train model
python src/train_model.py --timestamp "$timestamp"
```

**Windows (PowerShell):**
```powershell
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
python src/train_model.py --timestamp "$timestamp"
```

### Evaluate Model

```bash
python src/evaluate_model.py --timestamp "$timestamp"
```

### View Results

```bash
# View trained models
ls models/

# View metrics
cat metrics/*_metrics.json
```

## GitHub Actions Workflow

The workflow automatically:
1. Generates a timestamp for versioning
2. Trains a RandomForest model on synthetic data
3. Evaluates the model and calculates F1 Score
4. Saves model as `model_TIMESTAMP_dt_model.joblib`
5. Saves metrics as `TIMESTAMP_metrics.json`
6. Commits both to the repository

**Trigger:** Push to `main` branch with changes in Lab 2 directory

## Understanding the Code

### `train_model.py`
- Generates synthetic classification data using `make_classification()`
- Trains a RandomForestClassifier
- Saves model with timestamp: `model_20241211143022_dt_model.joblib`
- Uses MLflow for experiment tracking

### `evaluate_model.py`
- Loads the trained model using the timestamp
- Generates test data
- Calculates F1 Score
- Saves metrics to JSON file

### Model Versioning
- **Format:** `model_YYYYMMDDHHMMSS_dt_model.joblib`
- **Example:** `model_20241211143022_dt_model.joblib`
- Allows tracking model evolution over time

## Lab Tasks

1. **Setup**: Install dependencies and create directories
2. **Run Locally**: Train and evaluate a model locally
3. **Make Changes**: 
   - Modify `train_model.py` to add more estimators to RandomForest
   - Update evaluation to include accuracy metric
4. **Test Changes**: Run training and evaluation locally
5. **Push to GitHub**: Commit and push your changes
6. **Monitor Workflow**: Check Actions tab to see automated training
7. **View Results**: Check `models/` and `metrics/` directories for new files

## Git Workflow

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Update model training parameters"

# Push to trigger workflow
git push origin main
```

## Troubleshooting

**"No such file or directory: 'data/'":**
- Run: `mkdir -p data models metrics`

**Model file not found:**
- Make sure timestamp matches between training and evaluation
- Check that model was saved in `models/` directory

**Import errors:**
- Activate virtual environment
- Run: `pip install -r requirements.txt`

**GitHub Actions fail:**
- Check that `data/`, `models/`, and `metrics/` directories exist
- Review workflow logs in Actions tab
- Verify all dependencies in `requirements.txt`

## Customization Ideas


### Change Model Type
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
```

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Happy Training!**