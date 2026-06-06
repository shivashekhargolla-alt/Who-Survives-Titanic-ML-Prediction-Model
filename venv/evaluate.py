"""
Model Evaluation — Titanic Survival Prediction
===============================================
Evaluates the trained model using cross-validation
and training data analysis (no test labels needed)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import pickle
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────
def load_model(path="D:\\meee\\titanicproject\\venv\\model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from → {path}\n")
    return model


# ─────────────────────────────────────────────
# 2. Load & Preprocess Train Data
# ─────────────────────────────────────────────
def load_and_preprocess(filepath="D:\\meee\\titanicproject\\venv\\train.csv"):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    df.drop(columns=[c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns], inplace=True)

    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"]      = df["Sex"].map({"female": 0, "male": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    print(f"Features : {list(X.columns)}")
    print(f"Samples  : {X.shape[0]}\n")
    return X, y


# ─────────────────────────────────────────────
# 3. Cross-Validation Scores
# ─────────────────────────────────────────────
def cross_validation(model, X, y):
    print("=" * 50)
    print("  1. CROSS-VALIDATION (5-Fold)")
    print("=" * 50)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores  = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    prec_scores = cross_val_score(model, X, y, cv=skf, scoring="precision")
    rec_scores  = cross_val_score(model, X, y, cv=skf, scoring="recall")
    f1_scores   = cross_val_score(model, X, y, cv=skf, scoring="f1")
    roc_scores  = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

    print(f"\n{'Metric':<12} {'Fold 1':>8} {'Fold 2':>8} {'Fold 3':>8} {'Fold 4':>8} {'Fold 5':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 76)

    for name, scores in [
        ("Accuracy",  acc_scores),
        ("Precision", prec_scores),
        ("Recall",    rec_scores),
        ("F1 Score",  f1_scores),
        ("ROC-AUC",   roc_scores),
    ]:
        fold_vals = "  ".join([f"{s:.4f}" for s in scores])
        print(f"{name:<12}  {fold_vals}  {scores.mean():.4f}  {scores.std():.4f}")

    print(f"\n>>> Best Accuracy  : {acc_scores.max()*100:.2f}%")
    print(f">>> Mean Accuracy  : {acc_scores.mean()*100:.2f}%")
    print(f">>> Mean ROC-AUC   : {roc_scores.mean():.4f}")


# ─────────────────────────────────────────────
# 4. Training Data Metrics
# ─────────────────────────────────────────────
def training_metrics(model, X, y):
    print("\n" + "=" * 50)
    print("  2. TRAINING DATA METRICS")
    print("=" * 50)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"\nTraining Accuracy : {acc*100:.2f}%")
    print(f"ROC-AUC Score     : {auc:.4f}")

    print("\nClassification Report:")
    print("-" * 45)
    print(classification_report(y, y_pred, target_names=["Not Survived", "Survived"]))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print("-" * 45)
    print(f"                    Pred: Not Survived   Pred: Survived")
    print(f"Actual: Not Survived        {cm[0][0]}                  {cm[0][1]}")
    print(f"Actual: Survived            {cm[1][0]}                  {cm[1][1]}")

    total = cm.sum()
    correct = cm[0][0] + cm[1][1]
    wrong   = cm[0][1] + cm[1][0]
    print(f"\nTotal Samples  : {total}")
    print(f"Correct        : {correct}")
    print(f"Wrong          : {wrong}")


# ─────────────────────────────────────────────
# 5. Feature Importances
# ─────────────────────────────────────────────
def feature_importances(model, X):
    print("\n" + "=" * 50)
    print("  3. FEATURE IMPORTANCES")
    print("=" * 50)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print()
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<12} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────
# 6. Model Info
# ─────────────────────────────────────────────
def model_info(model):
    print("\n" + "=" * 50)
    print("  4. MODEL PARAMETERS")
    print("=" * 50)
    params = model.get_params()
    for key, val in params.items():
        print(f"  {key:<25} : {val}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Model Evaluation — Titanic Survival")
    print("=" * 50 + "\n")

    model    = load_model("D:\\meee\\titanicproject\\venv\\model.pkl")
    X, y     = load_and_preprocess("D:\\meee\\titanicproject\\venv\\train.csv")

    cross_validation(model, X, y)
    training_metrics(model, X, y)
    feature_importances(model, X)
    model_info(model)

    print("\n" + "=" * 50)
    print("  Evaluation Complete.")
    print("=" * 50)