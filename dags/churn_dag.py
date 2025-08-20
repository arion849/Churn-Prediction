from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess


def run_ingest():
    subprocess.run(["python", "/opt/airflow/scripts/ingest.py"], check=True)

def run_preprocess():
    subprocess.run(["python", "/opt/airflow/scripts/preprocess.py"], check=True)

def run_train():
    subprocess.run(["python", "/opt/airflow/scripts/train.py"], check=True)

def run_evaluate():
    subprocess.run(["python", "/opt/airflow/scripts/evaluate.py"], check=True)


with DAG(
    dag_id="churn_pipeline",
    start_date=datetime(2025, 8, 20),
    schedule_interval=None,
    catchup=False
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=run_ingest
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocess
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_train
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluate
    )

    ingest_task >> preprocess_task >> train_task >> evaluate_task
