# Who-Survives-Titanic-ML-Prediction-Model


# 🚢 Titanic Survival Prediction using Random Forest Classifier

A Machine Learning project that predicts whether a Titanic passenger survived or not, based on features like age, gender, ticket class, and fare — using the **Random Forest** algorithm.

---

## 📌 Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project builds a **binary classification model** to predict passenger survival using the Titanic dataset. The model is trained on `train.csv` and generates predictions on `test.csv`.

---

## 📁 Project Structure

```
titanicproject/
│
├── train.csv           # Training dataset
├── test.csv            # Test dataset (no Survived column)
├── modtest.py          # Main script: train, evaluate & predict
├── evaluate.py         # Full model evaluation script
├── model.pkl           # Saved Random Forest model
├── predictions.csv     # Output predictions on test.csv
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

---

## 📊 Dataset

The dataset is the classic [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data) from Kaggle.

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger ID |
| `Survived` | Target — 0 = Not Survived, 1 = Survived |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Siblings/spouses aboard |
| `Parch` | Parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation (C, Q, S) |

---

## ⚙️ How It Works

### 1. Preprocessing
- Dropped irrelevant columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Filled missing values: `Age` → median, `Fare` → median, `Embarked` → mode
- Encoded categorical features: `Sex` (female=0, male=1), `Embarked` (C=0, Q=1, S=2)

### 2. Model
- Algorithm: **Random Forest Classifier**
- Trees: 100
- Max Depth: 6
- Trained on full `train.csv` (891 samples)

### 3. Evaluation
- 5-Fold Stratified Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 4. Output
- Saves trained model as `model.pkl`
- Generates `predictions.csv` with survival predictions for `test.csv`

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Cross-Val Accuracy | ~82% |
| ROC-AUC Score | ~87% |
| Best Fold Accuracy | ~85% |

---

## 🔑 Feature Importances

| Feature | Importance |
|---|---|
| Sex | 44.3% |
| Fare | 17.1% |
| Pclass | 13.7% |
| Age | 13.4% |
| SibSp | 4.9% |
| Parch | 3.5% |
| Embarked | 3.2% |

> **Sex** is the most important feature — reflecting the "women and children first" evacuation policy.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/titanicproject.git
cd titanicproject

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install pandas scikit-learn numpy
```

### Run Training & Prediction

```bash
python modtest.py
```

### Run Evaluation

```bash
python evaluate.py
```

---

## 📦 Output Files

After running `modtest.py`:

- `model.pkl` — Saved trained Random Forest model
- `predictions.csv` — Survival predictions for each passenger in `test.csv`

Sample `predictions.csv`:

| PassengerId | Survived | Survival_Probability |
|---|---|---|
| 892 | 0 | 0.142 |
| 893 | 1 | 0.863 |
| 894 | 0 | 0.241 |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.12 | Programming language |
| pandas | Data manipulation |
| scikit-learn | Machine learning |
| numpy | Numerical computing |
| pickle | Model serialization |
| VS Code | Development environment |

---

## 📚 What I Learned

- Data preprocessing and feature engineering
- Random Forest algorithm and ensemble learning
- Model evaluation using cross-validation
- Saving and loading ML models using pickle
- End-to-end ML pipeline from raw data to predictions

---

## 🙋 Author

**Shekar Golla**
- GitHub: [@YourUsername](https://github.com/YourUsername)

---

