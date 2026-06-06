import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


def preprocess_test(df):
    df = df.copy()
    passenger_ids = df["PassengerId"].copy()  # save for output

    df.drop(columns=[c for c in ["PassengerId","Name","Ticket","Cabin"] if c in df.columns], inplace=True)

    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"]      = df["Sex"].map({"female": 0, "male": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return df, passenger_ids


if __name__ == "__main__":
    # Load model
    with open(r"D:\meee\titanicproject\venv\model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")

    # Load & preprocess test data
    test_df = pd.read_csv(r"D:\meee\titanicproject\venv\test.csv")
    test_processed, passenger_ids = preprocess_test(test_df)

    # Predict
    predictions = model.predict(test_processed)
    probabilities = model.predict_proba(test_processed)[:, 1]

    # Save results
    output = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived":    predictions,
        "Survival_Probability": probabilities.round(3)
    })
    output.to_csv(r"D:\meee\titanicproject\venv\test_results.csv", index=False)
    print(f"Predictions saved → test_results.csv")
    print(output.head(10))