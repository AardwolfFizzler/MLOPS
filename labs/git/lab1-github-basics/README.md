# Lab 1: GitHub Basics - Version Control and Automated Testing

##  Learning Objectives

- Use Git for version control===
- Write and run unit tests (pytest and unittest)
- Set up GitHub Actions for automated testing

##  Prerequisites

- Python 3.8+
- Git installed
- GitHub account

##  Setup Instructions

### 1. Clone the Repository

**Mac:**
```bash
git clone https://github.com/AardwolfFizzler/MLOPS.git
cd MLOPS-Repo/labs/lab1-github-basics
```

**Windows:**
```bash
git clone https://github.com/AardwolfFizzler/MLOPS.git
cd MLOPS-Repo\labs\lab1-github-basics
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

## Project Structure

```
lab1-github-basics/
├── .github/
│   └── workflows/          # GitHub Actions (must be at this location!)
│       ├── pytest.yml
│       └── unittest.yml
├── src/
│   ├── __init__.py
│   └── calculator.py       # Main code
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── requirements.txt
└── README.md
```

** Important:** The `.github/workflows/` folder must be in the `Root` directory for GitHub Actions to work!

##  Running Tests Locally

**Pytest (Mac):**
```bash
pytest test/test_pytest.py -v
```

**Pytest (Windows):**
```bash
pytest test\test_pytest.py -v
```

**Unittest (Mac):**
```bash
python -m unittest test.test_unittest -v
```

**Unittest (Windows):**
```bash
python -m unittest test.test_unittest -v
```

##  GitHub Actions

GitHub Actions will automatically run your tests when you push code. To see them:

1. Push your changes to GitHub
2. Go to your repository on GitHub
3. Click the **"Actions"** tab
4. You'll see your workflows running (pytest and unittest)

**Note:** Workflows trigger on pushes to the `main` branch.

##  Git Workflow

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push origin main

# Create new branch
git checkout -b feature/your-feature
```

##  Troubleshooting


**Virtual environment won't activate (Windows):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Tests fail:**
- Verify Python version: `python --version` (should be 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt`

**GitHub Actions not showing:**
- Make sure `.github/workflows/` exists in your `root` directory
- Verify you pushed to the `main` branch
- Check the Actions tab after pushing

## Resources

- [Git Documentation](https://git-scm.com/doc)
- [Pytest Docs](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Happy Coding!**