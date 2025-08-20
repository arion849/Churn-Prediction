import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

MODEL_PATH = "./models/logistic_model.pkl"
PREPROCESSED_DATA_PATH = "./data/processed/churn_preprocessed.csv"

def evaluate_model():
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    X = df.drop(columns=["Churn","customerID"])
    y = df["Churn"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

if __name__ == "__main__":
    evaluate_model()
