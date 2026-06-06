import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data(filepath=r"D:\meee\titanicproject\venv\train.csv"):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=[c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns], inplace=True)

    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"]      = df["Sex"].map({"female": 0, "male": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return df


# ─────────────────────────────────────────────
# 3. Train & Save Model
# ─────────────────────────────────────────────
def train_and_save(df, model_path="model.pkl"):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    print(f"Model trained on {X.shape[0]} samples.")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(r"D:\meee\titanicproject\venv\train.csv")
    df = preprocess(df)
    train_and_save(df, model_path="model.pkl")
    print("Done.")