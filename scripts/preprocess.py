import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

PROCESSED_DATA_PATH = "./data/processed/churn_processed.csv"
PREPROCESSED_DATA_PATH = "./data/processed/churn_preprocessed.csv"


def preprocess_data():
    PREPROCESSED_DATA_PATH = "./data/processed/churn_preprocessed.csv"
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.fillna(0, inplace = True)

    for col in df.select_dtypes(include = "object").columns:
        if col != "customerID":
            df[col] = LabelEncoder().fit_transform(df[col])

    
    df.to_csv(PREPROCESSED_DATA_PATH, index = False)
    print(PREPROCESSED_DATA_PATH)
    return PREPROCESSED_DATA_PATH


if __name__ == '__main__':
    preprocess_data()
