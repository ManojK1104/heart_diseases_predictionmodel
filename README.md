# ❤️ Heart Disease Risk Prediction

A machine learning project that predicts a patient's risk of **heart disease** based on clinical, lifestyle, and wearable-device data. The pipeline covers data ingestion, outlier/statistics analysis, preprocessing, class-imbalance handling, model training, and evaluation — with a CI workflow via GitHub Actions.

---

## 📌 Overview

Early identification of patients at risk of heart disease can enable timely intervention and better health outcomes. This project builds a **Random Forest classifier** to flag patients likely to have heart disease, using a combination of standard clinical indicators (blood pressure, cholesterol, blood sugar) and modern lifestyle/wearable metrics (daily steps, sleep hours, stress score).

---

## 🗂️ Project Structure

```
heart_diseases_predictionmodel/
├── .github/workflows/          # CI pipeline (GitHub Actions)
│   └── python-package.yml
├── data/
│   └── heart_disease_risk_2026.csv   # Patient dataset
├── model/
│   └── model.pkl                # Trained model (generated after running)
├── reseacrh/
│   └── model.ipynb              # Exploratory notebook / experimentation
├── report/
│   └── Reports.pdf              # Project report
├── src/
│   ├── data_ingestion.py        # Loads the dataset
│   ├── data_preprocessing.py    # Cleans, analyzes, encodes, splits & balances data
│   └── model_building.py        # Trains, evaluates & saves the model
├── main.py                      # Entry point that runs the full pipeline
├── requirements.txt             # Project dependencies
└── README.md
```

---

## 📊 Dataset

The project uses a **heart disease risk dataset** (`data/heart_disease_risk_2026.csv`) containing **9,000 patient records** with **26 features**, including:

- **Demographics**: `age`, `sex`
- **Clinical vitals**: `resting_bp_systolic`, `resting_bp_diastolic`, `resting_heart_rate`, `max_heart_rate_achieved`
- **Blood markers**: `cholesterol_total`, `hdl`, `ldl`, `triglycerides`, `fasting_blood_sugar`, `hba1c`
- **Cardiac indicators**: `chest_pain_type`, `exercise_induced_angina`, `st_depression`, `family_history`
- **Lifestyle**: `bmi`, `smoker_status`, `alcohol_units_per_week`, `exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `diet_quality_score`
- **Wearable data**: `wearable_owner`, `daily_steps`
- **Target variable**: `has_heart_disease`

---

## ⚙️ Pipeline

The workflow is broken into three modular stages, orchestrated by `main.py`:

1. **Data Ingestion** (`src/data_ingestion.py`)
   Loads the dataset directly from the repository's raw CSV file into a pandas DataFrame.

2. **Data Preprocessing** (`src/data_preprocessing.py`)
   - Drops duplicate records
   - Separates numerical and categorical columns
   - Drops the non-predictive `patient_id` column
   - Computes descriptive statistics per numerical feature (mean, median, std, variance, skew, kurtosis, IQR, and outlier whisker bounds) to profile the data
   - Label-encodes categorical features
   - Splits data into train/test sets (70/30 split)
   - Handles **class imbalance** using **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set

3. **Model Building & Evaluation** (`src/model_building.py`)
   - Trains a **Random Forest Classifier**
   - Evaluates performance using **accuracy** and a full **classification report** (precision, recall, F1-score)
   - Saves the trained model as a pickle file to `model/model.pkl`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 – 3.11

### Installation

```bash
# Clone the repository
git clone https://github.com/ManojK1104/heart_diseases_predictionmodel.git
cd heart_diseases_predictionmodel

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the full pipeline (ingestion → preprocessing → training → evaluation):

```bash
python main.py
```

This will print the dataset shape, train/test split shapes, model accuracy, and classification report, then save the trained model to `model/model.pkl`.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, FLAML (AutoML) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Environment | Jupyter (ipykernel) |
| CI/CD | GitHub Actions |

---

## 🔄 Continuous Integration

This repo includes a GitHub Actions workflow (`.github/workflows/python-package.yml`) that automatically, on every push/PR to `main`, tests against **Python 3.9, 3.10, and 3.11**:
- Installs dependencies
- Lints the code with `flake8`
- Runs the pipeline via `python main.py`

---

## 📈 Results

The trained Random Forest model's accuracy and detailed classification metrics (precision, recall, F1-score) are printed to the console at the end of each run of `main.py`. A full write-up is available in [`report/Reports.pdf`](report/Reports.pdf). *(Add your latest accuracy number here once you have a final run, e.g. "Accuracy: 0.85")*

---

## 🛣️ Future Improvements

- Hyperparameter tuning (e.g., GridSearchCV / FLAML AutoML)
- Feature importance / SHAP-based explainability for clinical interpretability
- Model comparison (Logistic Regression, XGBoost, Gradient Boosting, etc.)
- Deployment via a simple API (Flask/FastAPI) or Streamlit dashboard

---

## 📄 License

This project is licensed under the **Apache-2.0 License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Manoj K**
GitHub: [@ManojK1104](https://github.com/ManojK1104)
