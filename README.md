# 🩺 DiabetesScan — Random Forest Diabetes Risk Predictor

A clinical diabetes risk prediction web application built with **Streamlit**, powered by a **Random Forest classifier** trained live on Streamlit Cloud from your dataset — no pre-saved model files required.

---

## 📸 App Overview

The app accepts 4 patient inputs:

| Input | Type | Range |
|---|---|---|
| HbA1c Level (%) | Continuous slider | 3.5 – 9.0 |
| Age (years) | Integer slider | 1 – 100 |
| BMI | Continuous slider | 10.0 – 60.0 |
| Gender | Radio button | Female / Male |

It outputs:
- **Risk Level** — Low / Moderate / High with color coding
- **Risk Probability** — percentage from the RF model
- **Diabetes Predicted** — Yes / No (threshold: ≥50%)
- **Input vs Clinical Thresholds** — visual bars comparing your values to ADA 2024 cutoffs
- **Feature Importance Chart** — which of the 4 inputs most influenced the model

---

## 📁 Repository Structure

```
your-repo/
├── app.py                    ← Main Streamlit application
├── diabetes_dataset.csv      ← Dataset (100,000 rows × 16 columns) ⭐ required
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Theme and server configuration
├── .gitignore                ← Excludes venv, __pycache__, secrets
└── README.md                 ← This file
```

> ⭐ `diabetes_dataset.csv` **must be committed to the repo root**. Streamlit Cloud has no persistent disk storage — the app reads the CSV directly from the repository at runtime.

---

## 🚀 Deploying to Streamlit Cloud (Step by Step)

### Step 1 — Create a GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name your repo (e.g. `diabetes-scan`)
3. Set it to **Public**
4. **Do NOT** add a README, .gitignore, or license — leave it completely empty
5. Click **Create repository**

### Step 2 — Set up your local repo

Open a terminal and run:

```bash
cd ~/Project/BDA/diabetes-app

git init
git add .
git commit -m "Initial commit — DiabetesScan RF app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/diabetes-scan.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `diabetes-scan` with your actual GitHub username and repo name.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the form:
   - **Repository**: `YOUR_USERNAME/diabetes-scan`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

Your app will be live at:
```
https://YOUR_USERNAME-diabetes-scan-app-XXXX.streamlit.app
```
in approximately 2–4 minutes.

---

## ⚙️ How the App Works (No Pre-saved Models)

On first load, the app automatically runs this full pipeline:

```
diabetes_dataset.csv
        │
        ▼
1. Load & clean         — pd.read_csv() + dropna()
        │
        ▼
2. Feature engineering  — gender → gender_Male (binary: 1=Male, 0=Female)
        │
        ▼
3. Select 4 features    — hbA1c_level, age, bmi, gender_Male
        │
        ▼
4. Train/test split     — 80% train / 20% test, stratified, random_state=42
        │
        ▼
5. SMOTE oversampling   — balances minority class (diabetic ~8.5% → 50/50)
        │
        ▼
6. StandardScaler       — fit on SMOTE train set, applied to test set
        │
        ▼
7. Random Forest        — 200 trees, max_depth=10, random_state=42
        │
        ▼
8. Cache with           — @st.cache_resource (trains once per session)
   st.cache_resource
```

After the first load (~15–20 seconds), all subsequent predictions are **instant** because the trained model is held in memory.

---

## 🌲 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 |
| max_depth | 10 |
| random_state | 42 |
| n_jobs | -1 (all CPU cores) |
| Class balancing | SMOTE (imbalanced-learn) |
| Scaling | StandardScaler |
| Train/test split | 80% / 20%, stratified |
| Features used | hbA1c_level, age, bmi, gender_Male |

### Why only 4 features?

The SHAP beeswarm analysis of the full model (trained on all 74 features) showed that **hbA1c_level** and **blood_glucose_level** dominate all other features by a large margin, with **age** and **bmi** as the next most important. Since `blood_glucose_level` requires a lab test and is closely correlated with HbA1c, the simplified 4-feature model retains the majority of predictive power while using only inputs a patient can quickly provide.

### Why SMOTE?

The original dataset is heavily imbalanced — only ~8.5% of patients have diabetes. Without SMOTE, the Random Forest would be biased toward predicting "No Diabetes" for almost everyone. SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic samples of the minority class in the training set to achieve a 50/50 balance, dramatically improving the model's ability to correctly identify diabetic patients.

> **Important**: SMOTE is applied **only to the training set** to prevent data leakage. The test set is kept in its original imbalanced form for a realistic evaluation.

### Risk thresholds

The app maps the model's output probability to three risk levels:

| Probability | Risk Level | Colour |
|---|---|---|
| < 30% | Low Risk | Green |
| 30% – 59% | Moderate Risk | Amber |
| ≥ 60% | High Risk | Red |

---

## 📊 Dataset

### Required columns

Your `diabetes_dataset.csv` must contain at minimum:

```
age, gender, bmi, hbA1c_level, diabetes
```

The full dataset used in this project also includes:
```
year, race:AfricanAmerican, race:Asian, race:Caucasian, race:Hispanic,
race:Other, hypertension, heart_disease, blood_glucose_level,
location, smoking_history
```

The app only uses the 4 features listed above — extra columns are ignored.

### Dataset statistics (original full dataset)

| Property | Value |
|---|---|
| Total rows | 100,000 |
| Total columns | 16 |
| Missing values | 0 (after dropna) |
| Diabetic patients | ~8,500 (8.5%) |
| Non-diabetic patients | ~91,500 (91.5%) |
| Age range | 0.08 – 100 years |
| HbA1c range | 3.5% – 9.0% |
| BMI range | 10.0 – 60.0 |

---

## 💻 Running Locally

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/diabetes-scan.git
cd diabetes-scan

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📦 Dependencies

```
streamlit>=1.32.0       — Web framework
numpy>=1.24.0           — Numerical computing
pandas>=2.0.0           — Data loading and processing
scikit-learn>=1.4.0     — Random Forest, StandardScaler, train_test_split
imbalanced-learn>=0.12.0 — SMOTE oversampling
matplotlib>=3.7.0       — Feature importance chart
```

All dependencies are pinned with minimum versions for reproducibility. Streamlit Cloud installs them automatically from `requirements.txt`.

---

## 🔬 Clinical Reference (ADA 2024)

### HbA1c thresholds

| Category | HbA1c |
|---|---|
| Normal | Below 5.7% |
| Pre-diabetes | 5.7% – 6.4% |
| Diabetes | ≥ 6.5% |

### BMI categories

| Category | BMI |
|---|---|
| Underweight | Below 18.5 |
| Normal | 18.5 – 24.9 |
| Overweight | 25.0 – 29.9 |
| Obese | ≥ 30.0 |

### Age risk

Risk of Type 2 diabetes increases significantly after age 45 and continues to rise with age.

---

## 🛠️ Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `diabetes_dataset.csv not found` | CSV not in repo root | Copy CSV into `diabetes-app/` and `git add diabetes_dataset.csv` |
| App stuck on spinner | First-time training | Wait 15–30 seconds — this is normal |
| `ModuleNotFoundError` | Dependency missing | Check `requirements.txt` has all 6 packages |
| Push rejected | Remote has commits | Run `git pull origin main --allow-unrelated-histories` first |
| Streamlit Cloud shows old version | Cache not cleared | Go to app settings → Reboot app |

---

## ⚕️ Medical Disclaimer

This application is built for **educational and research purposes only** as part of a Big Data Analytics project. It does not constitute medical advice and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified healthcare provider for any health-related decisions.

---

## 👨‍💻 Project Info

Built as part of a **Big Data Analytics** course project using:
- Apache PySpark for large-scale model training
- scikit-learn for deployment-friendly inference
- Streamlit for rapid web application development
- SMOTE for handling class imbalance
- SHAP for model interpretability

---

*Trained live on Streamlit Cloud · No pre-saved model files · Random Forest · 4 features*