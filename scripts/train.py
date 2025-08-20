import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

PREPROCESSED_DATA_PATH = "./data/processed/churn_preprocessed.csv"
MODEL_PATH = "/opt/airflow/models/logistic_model.pkl"

def train_model():
    df = pd.read_csv(PREPROCESSED_DATA_PATH)


    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter = 500)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    return MODEL_PATH, X_test, y_test

if __name__ == '__main__':
    train_model()

