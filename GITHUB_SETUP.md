# 🚀 GitHub Deployment Guide

## Prerequisites

You'll need:
1. **GitHub Account** (create at https://github.com if you don't have one)
2. **Git** (install from https://git-scm.com/download/win if not already installed)
3. **GitHub CLI** (optional, install from https://cli.github.com for easy setup)

## Option 1: Using GitHub CLI (Easiest) ⭐

### Step 1: Install GitHub CLI
Download from: https://cli.github.com

Or install via package manager:
```powershell
choco install gh  # If using Chocolatey
```

### Step 2: Authenticate
```powershell
gh auth login
# Follow the prompts - choose "GitHub.com" and "HTTPS"
```

### Step 3: Create & Push Repository
```powershell
cd c:\Users\deepi\Music\claude\claude\claude
git init
git add .
git commit -m "Initial commit: Urban Heat Island ML with Streamlit deployment"
gh repo create urban-heat-island-ml --private --source=. --remote=origin --push
```

---

## Option 2: Using Git + GitHub Web Interface

### Step 1: Initialize Local Repository
```powershell
cd c:\Users\deepi\Music\claude\claude\claude
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git add .
git commit -m "Initial commit: Urban Heat Island ML with Streamlit deployment"
```

### Step 2: Create Repository on GitHub
1. Go to https://github.com/new
2. Enter repository name: `urban-heat-island-ml`
3. Select **Private**
4. Click "Create repository"

### Step 3: Add Remote & Push
Copy the HTTPS URL from GitHub (e.g., `https://github.com/YOUR_USERNAME/urban-heat-island-ml.git`)

```powershell
git remote add origin https://github.com/YOUR_USERNAME/urban-heat-island-ml.git
git branch -M main
git push -u origin main
```

When prompted, enter your GitHub username and a **Personal Access Token** (PAT):
- Go to https://github.com/settings/tokens
- Click "Generate new token"
- Select `repo` scope
- Copy and paste the token in the terminal

---

## Option 3: Using SSH (Most Secure)

### Step 1: Generate SSH Key
```powershell
ssh-keygen -t ed25519 -C "your.email@example.com"
# Press Enter for defaults, no passphrase needed for CI/CD
```

### Step 2: Add SSH Key to GitHub
1. Copy key: `type $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard`
2. Go to https://github.com/settings/keys
3. Click "New SSH key"
4. Paste the key and save

### Step 3: Initialize & Push
```powershell
cd c:\Users\deepi\Music\claude\claude\claude
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git add .
git commit -m "Initial commit: Urban Heat Island ML with Streamlit deployment"
git remote add origin git@github.com:YOUR_USERNAME/urban-heat-island-ml.git
git branch -M main
git push -u origin main
```

---

## What Gets Pushed

### ✅ Included in Repository
- `streamlit_app.py` - Streamlit web application
- `deployment_api.py` - Production inference API
- `deploy_predictions.py` - Batch prediction engine
- `train_and_serialize.py` - Model training script
- `models/` - Trained ML models (.pkl files)
- `results/` - Feature engineering outputs
- `src/` - Original analysis scripts
- `requirements.txt` - Python dependencies
- `DEPLOYMENT_GUIDE.md` - API documentation
- `STREAMLIT_DEPLOYMENT.md` - Streamlit guide
- `STATUS_REPORT.txt` - Project status
- `README.md` - Project overview
- `.gitignore` - Git ignore patterns

### ❌ Excluded from Repository (via .gitignore)
- `data/` - Large satellite data files (optional: add to .gitattributes for LFS)
- `__pycache__/` - Python cache
- `.venv/` - Virtual environment
- `.claude/` - Local Copilot cache
- Large `.tif` files (can use Git LFS if needed)

---

## After Pushing

### Make Repository Private (if not already)
1. Go to https://github.com/YOUR_USERNAME/urban-heat-island-ml
2. Click "Settings"
3. Under "Visibility", select "Private"
4. Click "Change visibility"

### Add Collaborators (Optional)
1. Go to "Settings" → "Collaborators"
2. Click "Add people"
3. Search by username and select access level

### Set Up Deployment (Optional)

**Deploy to Streamlit Cloud:**
1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repo
4. Enter: `streamlit_app.py`
5. Done! Your app is live

**Deploy to Heroku:**
```powershell
heroku login
heroku create your-app-name
git push heroku main
```

**Deploy to AWS/GCP/Azure:**
- Create Docker image from provided Dockerfile
- Push to container registry
- Deploy container

---

## Repository Structure on GitHub

```
urban-heat-island-ml/
├── streamlit_app.py                 (500+ lines, production web app)
├── deployment_api.py                (Production inference API)
├── deploy_predictions.py            (Batch prediction engine)
├── train_and_serialize.py           (Model training)
├── models/                          (Trained ML models)
│   ├── random_forest.pkl
│   ├── voting_ensemble.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
├── results/                         (Feature engineering)
│   ├── features.tif
│   └── feature_names.txt
├── src/                             (Source code)
├── docs/                            (Documentation)
├── requirements.txt                 (Dependencies)
├── STREAMLIT_DEPLOYMENT.md          (Streamlit guide)
├── DEPLOYMENT_GUIDE.md              (API docs)
├── STATUS_REPORT.txt                (Project status)
├── README.md                        (Overview)
└── .gitignore                       (Git ignore rules)
```

---

## Checking Git Installation

If you get "git not found", you need to install Git for Windows:

```powershell
# Check if Git is installed
git --version

# If not found, install via Chocolatey
choco install git

# Or download from https://git-scm.com/download/win
```

---

## Tips

- **Keep models in repo**: .pkl files are text-serialized and GitHub friendly (~15 MB)
- **For large data**: Use Git LFS (Large File Storage) for .tif files
- **Private repo benefits**: Code security, control who sees your work, can still deploy publicly via Streamlit Cloud
- **CI/CD ready**: Add GitHub Actions workflows for automated testing/deployment

---

## Next Steps

1. Choose an authentication method (GitHub CLI recommended)
2. Follow the steps above
3. Verify repo is private on GitHub
4. (Optional) Deploy to Streamlit Cloud for live URL

Let me know if you need help with any step!
