# Lab 5: ML Workflow Orchestration with Prefect

## Overview
In this lab, you will learn how to orchestrate machine learning workflows using Prefect, a powerful open-source workflow orchestration tool. You'll build a complete ML pipeline with tasks and flows, and deploy it locally with monitoring.

## Learning Objectives
By the end of this lab, you will be able to:
- Understand the core components of Prefect (Tasks, Flows, Deployments, Work Pools)
- Create and orchestrate ML workflows using Prefect decorators
- Deploy and execute workflows locally using Prefect workers
- Monitor workflows using the Prefect UI dashboard

## Prerequisites
- Python 3.8 or higher
- Basic understanding of machine learning workflows
- Windows, macOS, or Linux operating system

## Lab Files
```
lab5-prefect-orchestration/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Main ML workflow with Prefect
├── data/
│   └── train.csv            # Bank churn dataset (download separately)
└── .gitignore               # Git ignore file
```

---

## Part 1: Setup Environment

### Step 1: Create and Activate Virtual Environment

**For Windows (PowerShell or Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

---

### Step 2: Install Dependencies
All required packages are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
prefect version
```
You should see the Prefect version information.

---

## Part 2: Download Dataset

### Step 1: Download Bank Churn Dataset
You need to download the Bank Churn dataset and place it in the `data/` directory.

**Kaggle**
- Go to [Kaggle Bank Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- Download and extract to `data/train.csv`

### Step 2: Verify Data Location

**For Windows:**
```bash
dir data\train.csv
```

**For macOS/Linux:**
```bash
ls data/train.csv
```

---

## Part 3: Understanding the ML Workflow

### Review main.py
Open `main.py` and examine the workflow structure:

**Tasks in the workflow:**
1. `load_data` - Load and preprocess CSV file
2. `preprocessing` - Impute, encode, and scale features
3. `data_split` - Split into train/test sets
4. `train_model` - Feature selection and model training
5. `get_prediction` - Generate predictions
6. `evaluate_model` - Calculate accuracy and F1 score
7. `save_model` - Save model weights using skops

**Flow:**
- `ml_workflow` - Orchestrates all tasks sequentially

### Step 1: Test the Workflow Locally
Run the workflow without Prefect orchestration:
```bash
python main.py
```

**Expected Output:**
```
Accuracy: 0.75 F1: 0.65
```
A file `bank_model.skops` should be created.

---

## Part 4: Deploy Workflow Locally with Prefect

### Step 1: Start Prefect Server
Open a **new terminal window** (Terminal 1) and start the Prefect server:

**For Windows:**
```bash
cd MLOPS-Repo\labs\lab5-prefect-orchestration
venv\Scripts\activate
prefect server start
```

**For macOS/Linux:**
```bash
cd MLOPS-Repo/labs/lab5-prefect-orchestration
source venv/bin/activate
prefect server start
```

**Expected Output:**
```
 ___ ___ ___ ___ ___ ___ _____
| _ \ _ \ __| __| __/ __|_   _|
|  _/   / _|| _|| _| (__  | |
|_| |_|_\___|_| |___\___| |_|

Configure Prefect to communicate with the server with:
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

Check out the dashboard at http://127.0.0.1:4200
```

**Important:** Keep this terminal running! This is your Prefect server.

---

### Step 2: Configure API URL
Open a **second terminal window** (Terminal 2) and tell Prefect where the server is:

**For Windows:**
```bash
cd MLOPS-Repo\labs\lab5-prefect-orchestration
venv\Scripts\activate
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

**For macOS/Linux:**
```bash
cd MLOPS-Repo/labs/lab5-prefect-orchestration
source venv/bin/activate
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

**Expected Output:**
```
Set 'PREFECT_API_URL' to 'http://127.0.0.1:4200/api'
```

**What this does:** Tells all Prefect commands in this terminal to connect to your local server.

---

### Step 3: Create a Work Pool
In the same terminal (Terminal 2), create the work pool:

```bash
prefect work-pool create default-agent-pool --type process
```

**Expected Output:**
```
Created work pool 'default-agent-pool'!
```

**Note:** If you see "Work pool already exists", that's okay! Skip to Step 4.

**What is a work pool?**
- A work pool manages where and how your workflows run
- The `process` type runs workflows as local Python processes

---

### Step 4: Deploy the Workflow
In the same terminal (Terminal 2), create a deployment:

```bash
prefect deploy main.py:ml_workflow -n ml_workflow_bank_churn -p default-agent-pool --tag dev
```

**You'll be asked two questions:**
1. `Would you like your workers to pull your flow code from a remote storage location?`
   - Type: **n** (we'll use local code)
   
2. `Would you like to configure schedules for this deployment?`
   - Type: **n** (we'll run manually)

**Expected Output:**
```
Deployment 'ml-workflow/ml_workflow_bank_churn' successfully created
```

---

### Step 5: Start a Worker
In the same terminal (Terminal 2), start the worker:

```bash
prefect worker start --pool default-agent-pool
```

**Expected Output:**
```
Worker started! Polling for work from 'default-agent-pool'...
```

**Important:** Keep this terminal running! The worker waits for work to execute.

---

### Step 6: Run the Deployment
Open a **third terminal window** (Terminal 3) and execute:

**For Windows:**
```bash
cd MLOPS-Repo\labs\lab5-prefect-orchestration
venv\Scripts\activate
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
prefect deployment run ml-workflow/ml_workflow_bank_churn
```

**For macOS/Linux:**
```bash
cd MLOPS-Repo/labs/lab5-prefect-orchestration
source venv/bin/activate
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
prefect deployment run ml-workflow/ml_workflow_bank_churn
```

**Expected Output:**
```
Creating flow run for deployment 'ml-workflow/ml_workflow_bank_churn'...
Created flow run 'golden-shark-23'
```

---

### Step 7: Monitor Execution
Switch to **Terminal 2** where the worker is running. You should see:
- `Submitting flow run...`
- Task execution logs:
  ```
  load_data | Loading data from data/train.csv
  preprocessing | Preprocessing data...
  data_split | Splitting data into train and test sets
  train_model | Training model...
  evaluate_model | Accuracy: 0.75 F1: 0.65
  save_model | Saving model to bank_model.skops
  ```
- `Finished in state Completed()`

---

### Step 8: View in UI
While all terminals are still running, open your browser and go to:
```
http://127.0.0.1:4200
```

You should see:
- **Flow Runs**: Your completed run with execution details
- **Deployments**: Your `ml_workflow_bank_churn` deployment
- **Work Pools**: Your `default-agent-pool` with status

---

## Part 5: Prefect UI Dashboard

### Explore the UI

#### Flow Runs
1. Click "Flow Runs" in the left sidebar
2. You'll see all executed flows with their status (Completed, Failed, Running)
3. Click on a flow run to see detailed information

#### Flow Run Details
When you click on a specific flow run, you can see:
- **Logs**: Real-time execution logs with timestamps
- **Task execution timeline**: Visual representation of when each task ran
- **Task status**: Which tasks succeeded or failed
- **Parameters**: Any parameters passed to the flow
- **Duration**: How long each task took

#### Deployments
1. Click "Deployments" in the left sidebar
2. View your `ml_workflow_bank_churn` deployment
3. You can run it directly from here using "Quick run" button
4. View deployment settings and schedule (if configured)

#### Work Pools
1. Click "Work Pools" in the left sidebar
2. View your `default-agent-pool` and its status
3. See how many workers are connected
4. View recent flow runs processed by this pool

---

## Summary: What's Running

You now have **3 terminals running**:

**Terminal 1: Prefect Server** 🖥️
```bash
prefect server start
```
- Purpose: Provides the Prefect API and UI
- Keep running while working with Prefect

**Terminal 2: Worker** 👷
```bash
prefect worker start --pool default-agent-pool
```
- Purpose: Executes your workflows
- Keep running to process flow runs

**Terminal 3: Commands** ⌨️
```bash
prefect deployment run ml-workflow/ml_workflow_bank_churn
```
- Purpose: Trigger deployments and run commands
- Use for interactions

---

## Key Concepts Review

### 1. Prefect Tasks
- **Definition**: Discrete units of work (Python functions with `@task` decorator)
- **Features**: Retries, caching, timeout control
- **Use Case**: Reusable operations like data loading, model training
- **Example**: Each step in your ML pipeline is a task

### 2. Prefect Flows
- **Definition**: Containers for workflow logic (`@flow` decorator)
- **Purpose**: Orchestrate task execution and dependencies
- **Benefits**: Observability, logging, error handling
- **Example**: Your `ml_workflow` function that calls all tasks

### 3. Deployments
- **Definition**: Packaged workflows with execution metadata
- **Components**: Name, work pool, tags, scheduling
- **Purpose**: Makes flows executable by workers
- **Example**: Your `ml_workflow_bank_churn` deployment

### 4. Work Pools
- **Definition**: Queue that holds work for workers to execute
- **Purpose**: Manage work distribution to workers
- **Types**: Process (local), Docker, Kubernetes
- **Example**: Your `default-agent-pool`

### 5. Workers
- **Definition**: Processes that execute flow runs from work pools
- **Purpose**: Actually run your code
- **Example**: The worker you started in Terminal 2

---

### Issue 8: Windows PowerShell Execution Policy Error
**Error Message:**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating the virtual environment again:
```bash
venv\Scripts\activate
```

**Alternative:** Use Command Prompt (cmd) instead of PowerShell:
```bash
venv\Scripts\activate.bat
```

---

### Issue 9: Windows Path Issues
**Problem:** Commands work with forward slashes `/` but Windows uses backslashes `\`

**Solution:** 
- Use backslashes for Windows paths: `cd MLOPS-Repo\labs\lab5-prefect-orchestration`
- Or use forward slashes (they usually work): `cd MLOPS-Repo/labs/lab5-prefect-orchestration`
- Python code in `main.py` should use forward slashes or `os.path.join()` for cross-platform compatibility

---

## Platform-Specific Notes

### Windows Users 🪟
- Use `venv\Scripts\activate` to activate virtual environment
- Use `dir` instead of `ls` to list files
- Use backslashes `\` in paths or forward slashes `/` (both usually work)
- If using PowerShell, you may need to adjust execution policy (see Issue 8)
- Use Command Prompt (cmd) if PowerShell gives issues
- Ctrl+C to stop running processes

### macOS/Linux Users 🐧🍎
- Use `source venv/bin/activate` to activate virtual environment
- Use `ls` to list files
- Use forward slashes `/` in paths
- Ctrl+C to stop running processes

---

## Troubleshooting

### Issue 1: "PREFECT_API_URL must be set"
**Cause:** Worker/command started before configuring API URL
**Solution:** 
```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```
Run this in every new terminal before using Prefect commands.

---

### Issue 2: "No module named 'prefect'"
**Solution:**
```bash
pip install prefect
```

---

### Issue 3: "FileNotFoundError: train.csv"
**Solution:**
- Ensure `train.csv` is in the `data/` directory
- Check the file path in `main.py` (default: `data/train.csv`)
- Make sure you're running commands from the lab directory

---

### Issue 4: Worker not picking up work
**Solution:**
- Verify all 3 terminals are running
- Check server is accessible at http://127.0.0.1:4200
- Restart worker with Ctrl+C and start again
- Verify work pool name: `prefect work-pool ls`

---

### Issue 5: Can't access UI at http://127.0.0.1:4200
**Solution:**
- Make sure Terminal 1 (server) is running
- Try http://localhost:4200 instead
- Check no other application is using port 4200
- Look for the "Check out the dashboard at..." message in Terminal 1

---

### Issue 6: "Work pool already exists"
**Solution:**
This is normal! It means the work pool was created before. Just continue to the next step.

---

### Issue 7: Deployment not found
**Solution:**
```bash
# List all deployments
prefect deployment ls

# If missing, recreate it
prefect deploy main.py:ml_workflow -n ml_workflow_bank_churn -p default-agent-pool --tag dev
```

---

## Additional Resources

### Official Documentation
- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect Tutorial](https://docs.prefect.io/tutorial/)
- [Prefect Concepts Guide](https://docs.prefect.io/concepts/)
---

**Estimated Time**: 2 hours

**Difficulty**: Intermediate

**Tags**: #MLOps #Orchestration #Prefect #Workflow #MachineLearning

**Version**: Updated for Prefect 3.x

**Last Updated**: January 2026