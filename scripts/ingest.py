import pandas as pd 
import os

RAW_DATA_PATH = "./data/raw/churn_data.csv"
PROCESSED_DATA_PATH = "./data/processed/churn_processed.csv"

def ingest_data():
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok = True)
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Ingested {df.shape[0]} rows and {df.shape[1]} columns")
    df.to_csv(PROCESSED_DATA_PATH, index = False)
    return PROCESSED_DATA_PATH

if __name__ == '__main__':
    ingest_data()