# Lab 3: GitHub Actions + AWS S3 - ML Model Storage

## Learning Objectives

- Connect GitHub Actions to AWS S3
- Automate ML model training
- Store trained models in cloud storage
- Use GitHub Secrets for secure credentials

##  Prerequisites

- AWS account (free tier is fine)
- Completed Lab 1 and Lab 2
- Python 3.8+
- Git and GitHub account

## Setup Instructions

### 1. Navigate to Lab 3

**Mac:**
```bash
cd ~/path/to/MLOPS-Repo/labs/lab3-github-actions-aws-beginner-lab
```

**Windows:**
```bash
cd C:\path\to\MLOPS-Repo\labs\lab3-github-actions-aws-beginner-lab
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

## AWS Setup

### Step 1: Create S3 Bucket

1. Go to [S3 Console](https://s3.console.aws.amazon.com/s3/)
2. Click **Create bucket**
3. Enter a **unique bucket name** (e.g., `mlops-models-yourname-2025`)
4. Select your **AWS Region** (e.g., `us-east-1`)
5. Leave other settings as default
6. Click **Create bucket**

##  Configure GitHub Secrets

### Add Secrets to Your Repository

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these four secrets:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `AWS_ACCESS_KEY_ID` | Your access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Your secret access key | `wJalrXUtnFEMI/K7MDENG/bPxRfi...` |
| `AWS_S3_BUCKET` | Your bucket name | `mlops-models-yourname-2024` |
| `AWS_REGION` | Your AWS region | `us-east-1` |

**⚠️ Security Note:** Never commit AWS credentials to your code!

##  Project Structure

```
lab3-github-actions-aws-beginner-lab/
├── train_and_save_model.py    # Training script with S3 upload
├── requirements.txt           # Dependencies
└── README.md
```

**Note:** The workflow file is at `.github/workflows/lab3-train-and-upload.yml` in the repository root.

## Test Locally (Optional)

Set environment variables and run:

**Mac/Linux:**
```bash
export AWS_ACCESS_KEY_ID='your-key'
export AWS_SECRET_ACCESS_KEY='your-secret'
export AWS_S3_BUCKET='your-bucket-name'
export AWS_DEFAULT_REGION='us-east-1'

python train_and_save_model.py
```

**Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID='your-key'
$env:AWS_SECRET_ACCESS_KEY='your-secret'
$env:AWS_S3_BUCKET='your-bucket-name'
$env:AWS_DEFAULT_REGION='us-east-1'

python train_and_save_model.py
```

## Run GitHub Actions Workflow

### Method 1: Push Changes (Automatic)

```bash
git add .
git commit -m "Test Lab 3 workflow"
git push origin main
```

### Method 2: Manual Trigger

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Lab 3 - Train and Upload to S3"**
4. Click **"Run workflow"** button
5. Click **"Run workflow"** to confirm

### Check Results

1. Wait for workflow to complete 
2. Go to your S3 bucket in AWS Console
3. Look in `trained_models/` folder
4. You should see: `model_YYYYMMDDHHMMSS.joblib`

##  Understanding the Code

### `train_and_save_model.py`

The script:
1. **Downloads** Iris dataset from scikit-learn
2. **Splits** data into train/test sets
3. **Trains** a RandomForest classifier
4. **Evaluates** model accuracy
5. **Uploads** model to S3 with timestamp

### S3 Storage Path

Models are saved as:
```
s3://your-bucket-name/trained_models/model_20241211143022.joblib
```

## Troubleshooting

**"Invalid security token" or "Access Denied":**
- Verify AWS credentials in GitHub Secrets
- Check IAM user has S3FullAccess policy
- Ensure bucket name is correct

**"Bucket does not exist":**
- Double-check bucket name in secrets
- Verify bucket region matches `AWS_REGION` secret

**Workflow doesn't trigger:**
- Check file is on `main` branch
- Verify workflow file is in `.github/workflows/`
- Try manual trigger from Actions tab

**Import errors:**
- Activate virtual environment
- Run: `pip install -r requirements.txt`

**Local testing fails:**
- Set all environment variables
- Check AWS CLI configuration: `aws s3 ls`


### Add Model Metadata
```python
import json
metadata = {
    'timestamp': timestamp,
    'accuracy': accuracy,
    'model_type': 'RandomForest',
    'n_estimators': 100
}
# Save metadata.json to S3

```

## Cleanup (After Lab)

To avoid AWS charges:

1. **Delete S3 Objects:**
   - Go to S3 Console
   - Select your bucket
   - Delete all objects in `trained_models/`

2. **Delete S3 Bucket:**
   - Click bucket → Delete
   - Confirm deletion

## 📚 Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [GitHub Actions - AWS](https://github.com/aws-actions/configure-aws-credentials)
- [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)

---

**Happy Cloud Computing! ☁️**