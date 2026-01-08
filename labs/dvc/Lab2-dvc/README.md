# Lab 2: DVC Pipelines – Student Guide

## Overview

In this lab, you will build a complete machine learning pipeline using DVC (Data Version Control) . You'll work with a weather prediction dataset to create a reproducible ML workflow that includes data preprocessing, model training, and evaluation.

**Important**: This lab uses DVC v3+ syntax:
* `dvc stage add` - to define pipeline stages
* `dvc repro` - to execute the pipeline
* **NOT** `dvc run` (deprecated in DVC v3)

---

## Learning Objectives

By the end of this lab, you will be able to:
- Set up a DVC project from scratch
- Create a multi-stage ML pipeline with dependencies
- Track data, models, and metrics with DVC
- Reproduce experiments reliably
- Visualize pipeline structure and results

---

## What's in This Repository

```
Lab2-dvc/
├── data/
│   └── .gitkeep              # Placeholder (you'll add weatherAUS.csv)
├── src/
│   ├── preprocess_data.py    # Data cleaning and feature engineering
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation script
│   └── model.py              # Model architecture definition
├── models/
│   └── .gitkeep              # DVC will save trained models here
├── results/
│   └── .gitkeep              # Metrics and plots will be saved here
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Getting Started

### Step 1: Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- Git installed
- Basic knowledge of terminal/command line

**Check your versions:**

```bash
python3 --version    # or python --version on Windows
git --version
pip --version
```

---

### Step 2: Clone the Repository

**On Linux/Mac:**
```bash
git clone <your-repo-url>
cd MLOPS-Repo/labs/Lab2-dvc
```

**On Windows:**
```cmd
git clone <your-repo-url>
cd MLOPS-Repo\labs\Lab2-dvc
```

---

### Step 3: Set Up Virtual Environment

**On Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv)
```

**On Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Your prompt should now show (venv)
```

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If you get an error about execution policy, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify DVC installation
dvc version
```

**Expected output:** `DVC version: 3.x.x`

---

### Step 5: Download the Dataset

1. **Download** `weatherAUS.csv` from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
   - You may need to create a free Kaggle account
   - Download the CSV file

2. **Place the file** in the `data/` folder:
   ```
   Lab2-dvc/data/weatherAUS.csv
   ```

3. **Verify the file exists:**

   **Linux/Mac:**
   ```bash
   ls -lh data/weatherAUS.csv
   ```

   **Windows:**
   ```cmd
   dir data\weatherAUS.csv
   ```

   You should see the file size (around 140MB).

---

## Building Your ML Pipeline

### Part 1: Initialize DVC

DVC needs to be initialized in your project to start tracking data and pipelines.

```bash
# Initialize DVC (use --subdir if Lab2-dvc is a subdirectory)
dvc init --subdir

# Add DVC config files to Git
git add .dvc/.gitignore .dvc/config

# Commit
git commit -m "Initialize DVC for Lab2"
```

**What happened?**
- Created `.dvc/` directory with DVC configuration
- DVC is now ready to track your data and pipelines

---

### Part 2: Track the Dataset with DVC

Instead of tracking large data files with Git, we use DVC:

```bash
# Add the dataset to DVC tracking
dvc add data/weatherAUS.csv

# This creates data/weatherAUS.csv.dvc (a metadata file)
# The actual CSV is now ignored by Git

# Add the metadata file to Git
git add data/weatherAUS.csv.dvc data/.gitignore

# Commit
git commit -m "Track weather dataset with DVC"
```

**What happened?**
- `data/weatherAUS.csv.dvc` contains a hash of your data file
- The actual CSV is added to `.gitignore`
- Git tracks the metadata, DVC manages the actual data

---

### Part 3: Create Pipeline Stage 1 - Preprocessing

Now we'll create our first pipeline stage to preprocess the data.

```bash
dvc stage add -n preprocess \
  -d src/preprocess_data.py \
  -d data/weatherAUS.csv \
  -o data/weatherAUS_processed.csv \
  python3 src/preprocess_data.py data/weatherAUS.csv
```

**Command breakdown:**
- `-n preprocess` - Stage name
- `-d` - Dependencies (inputs that trigger re-runs if changed)
- `-o` - Outputs (files created by this stage)
- Last line - The actual command to run

**What this stage does:**
- Reads `weatherAUS.csv`
- Handles missing values
- Encodes categorical variables
- Saves cleaned data to `weatherAUS_processed.csv`

---

### Part 4: Create Pipeline Stage 2 - Training

Now add the training stage:

```bash
dvc stage add -n train \
  -d src/train.py \
  -d src/model.py \
  -d data/weatherAUS_processed.csv \
  -o models/model.joblib \
  python3 src/train.py data/weatherAUS_processed.csv src/model.py 200
```

**What this stage does:**
- Loads processed data
- Splits data into train/test sets (80/20)
- Trains a Random Forest model with 200 trees
- Saves trained model to `models/model.joblib`

**Note:** The `200` at the end is the number of trees (n_estimators)

---

### Part 5: Create Pipeline Stage 3 - Evaluation

Finally, add the evaluation stage:

```bash
dvc stage add -n evaluate \
  -d src/evaluate.py \
  -d src/model.py \
  -d data/weatherAUS_processed.csv \
  -d models/model.joblib \
  -M results/metrics.json \
  -o results/precision_recall_curve.png \
  -o results/roc_curve.png \
  python3 src/evaluate.py data/weatherAUS_processed.csv src/model.py models/model.joblib
```

**What this stage does:**
- Loads trained model
- Evaluates on test set
- Calculates metrics (accuracy, precision, recall, F1-score)
- Generates visualization plots (ROC curve, Precision-Recall curve)
- Saves metrics to JSON file

---

### Part 6: Run the Pipeline

Now execute the entire pipeline:

```bash
dvc repro
```

**What happens:**
- DVC analyzes dependencies
- Runs each stage in order (preprocess → train → evaluate)
- Caches outputs
- Creates `dvc.lock` file to track exact versions

**This will take a few minutes** - you'll see progress for each stage.

---

### Part 7: Commit Pipeline to Git

```bash
# Add pipeline definition files
git add dvc.yaml dvc.lock

# Commit
git commit -m "Add complete ML pipeline with 3 stages"
```

**What you're tracking:**
- `dvc.yaml` - Pipeline definition (stages, dependencies, commands)
- `dvc.lock` - Exact versions and hashes (like a package-lock.json)

---

## Exploring Your Results

### View Pipeline Structure

```bash
dvc dag
```

**Expected output:**
```
 +-------------------------+  
 | data/weatherAUS.csv.dvc |  
 +-------------------------+  
               *              
               *              
               *              
        +------------+        
        | preprocess |        
        +------------+        
         **        **         
       **            **       
      *                **     
+-------+                *    
| train |              **     
+-------+            **       
         **        **         
           **    **           
             *  *             
         +----------+         
         | evaluate |         
         +----------+
```

This shows how stages depend on each other.

---

### Check Pipeline Status

```bash
dvc status
```

If everything is up to date, you'll see:
```
Data and pipelines are up to date.
```

---

### View Metrics

```bash
dvc metrics show
```

**Example output:**
```
Path                  accuracy    f1       precision    recall        
results/metrics.json  0.84973     0.90747  0.8719       0.94607
```

---

### View Generated Files

**Linux/Mac:**
```bash
ls -lh results/
```

**Windows:**
```cmd
dir results\
```

**You should see:**
- `metrics.json` - Numerical metrics
- `roc_curve.png` - ROC curve visualization
- `precision_recall_curve.png` - Precision-Recall curve

**View the images** to see your model's performance!

---

## Push Your Work to GitHub

Now that you've completed the lab, it's time to push your work to your own repository.

### Step 1: Check Your Git Status

```bash
# See what files have been modified/created
git status
```

You should see:
- Modified: `dvc.yaml`, `dvc.lock`
- Untracked: Generated files in `models/` and `results/` (these are ignored by Git)

---

### Step 2: Review What You're Committing

```bash
# View the differences
git diff dvc.yaml
git diff dvc.lock
```

This shows you the pipeline stages you created.

---

### Step 3: Add and Commit Your Work

```bash
# Add DVC pipeline files
git add dvc.yaml dvc.lock

# Add the dataset metadata file (if not already committed)
git add data/weatherAUS.csv.dvc data/.gitignore

# Commit with a descriptive message
git commit -m "Complete Lab2: Build ML pipeline with DVC"
```

---

### Step 4: Push to Your Remote Repository

**If you haven't set up a remote yet:**

```bash
# Create a new repository on GitHub (via web interface)
# Then link it to your local repository

git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Verify the remote was added
git remote -v

# Push your work
git push -u origin main
```

**If you already have a remote configured:**

```bash
# Simply push your changes
git push origin main
```

**On Windows, you might be prompted for credentials** - use your GitHub username and Personal Access Token (not password).

---

### Step 5: Verify on GitHub

1. Go to your repository on GitHub
2. Navigate to `labs/Lab2-dvc/`
3. You should see:
   - `dvc.yaml` with your pipeline definition
   - `dvc.lock` with execution details
   - `data/weatherAUS.csv.dvc` metadata file
   - Scripts in `src/` directory

**Note:** The actual data files, models, and results are NOT pushed to GitHub. This is by design - DVC manages these large files separately.

---

### Step 6: Understanding What Was NOT Pushed

These files are local only (listed in `.gitignore` and `data/.gitignore`):
- `data/weatherAUS.csv` (140MB - too large for Git)
- `data/weatherAUS_processed.csv` (generated file)
- `models/model.joblib` (model file)
- `results/*.json` and `results/*.png` (generated outputs)

**This is good!** Git should only track:
- Code and scripts
- Pipeline definitions (`dvc.yaml`)
- DVC metadata files (`.dvc` files)
- NOT large data files or generated outputs

---

##  Troubleshooting Guide

### Issue: `ERROR: argument command: invalid choice: 'run'`

**Cause:** You're using DVC v3, but following old tutorials that use `dvc run`

**Solution:** Use `dvc stage add` instead of `dvc run`

---

### Issue: "dvc: command not found"

**Solution:**
```bash
# Make sure virtual environment is activated
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Reinstall DVC if needed
pip install dvc
```

---

### Issue: "FileNotFoundError: weatherAUS.csv"

**Solution:**
```bash
# Check if file exists in correct location
ls data/weatherAUS.csv  # Linux/Mac
dir data\weatherAUS.csv  # Windows

# If missing, re-download from Kaggle
# Make sure it's named exactly: weatherAUS.csv
```

---

### Issue: "ModuleNotFoundError" when running scripts

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Or install specific missing packages
pip install pandas scikit-learn matplotlib joblib
```

---

### Issue: Stage won't re-run even after changes

**Solution:**
```bash
# Check what DVC thinks changed
dvc status

# Force re-run specific stage
dvc repro -f train

# Or force entire pipeline
dvc repro -f
```

---

### Issue: Python script errors

**Solution:**
```bash
# Test the script manually to see detailed errors
python3 src/preprocess_data.py data/weatherAUS.csv

# Check first few lines of data file
head data/weatherAUS.csv  # Linux/Mac
type data\weatherAUS.csv | more  # Windows
```

---

### Issue: Permission denied on Windows

**Solution:**
```powershell
# Run PowerShell as Administrator and set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
---

## What You Learned

### Key Concepts

1. **Pipeline Creation**
   - Multi-stage ML workflows using `dvc stage add`
   - Declaring dependencies between stages
   - Defining inputs and outputs

2. **Dependency Tracking**
   - DVC automatically tracks file changes
   - Only re-runs stages when dependencies change
   - Caches outputs for efficiency

3. **Reproducibility**
   - `dvc repro` reproduces entire workflows
   - `dvc.lock` ensures exact reproducibility
   - Anyone can reproduce your results

4. **Experiment Tracking**
   - Git branches for different experiments
   - `dvc metrics diff` to compare experiments
   - Track what works and what doesn't

5. **Separation of Concerns**
   - Git tracks code and pipeline definitions
   - DVC tracks data, models, and large files
   - Best of both worlds

---

## DVC Commands Reference

```bash
# Initialization
dvc init              # Initialize DVC in project
dvc init --subdir     # Initialize in subdirectory

# Data tracking
dvc add <file>        # Track large file with DVC

# Pipeline creation
dvc stage add -n <name> -d <dep> -o <output> <command>
                      # Define a pipeline stage

# Pipeline execution
dvc repro             # Run pipeline (only changed stages)
dvc repro -f          # Force re-run entire pipeline
dvc repro -f <stage>  # Force re-run specific stage

# Pipeline inspection
dvc dag               # Visualize pipeline as DAG
dvc status            # Check if pipeline is up to date

# Metrics
dvc metrics show      # Display current metrics
dvc metrics diff <branch>
                      # Compare metrics between branches

# Remote storage (if configured)
dvc push              # Upload data to remote
dvc pull              # Download data from remote
```

---

##  Additional Resources

### Official Documentation
- [DVC Documentation](https://dvc.org/doc)
- [DVC Get Started Guide](https://dvc.org/doc/start)
- [DVC Pipelines Guide](https://dvc.org/doc/user-guide/pipelines)

### ML Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest)

### Dataset
- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

### Git Resources
- [Git Branching Guide](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)

---