from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append('/opt/airflow/scripts')

import ingest
import preprocess
import train
import evaluate

with DAG(
    dag_id='churn_prediction_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['churn','ml']
) as dag:

    t_ingest = PythonOperator(task_id='ingest', python_callable=ingest.main)
    t_preprocess = PythonOperator(task_id='preprocess', python_callable=preprocess.main)
    t_train = PythonOperator(task_id='train', python_callable=train.main)
    t_evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate.main)

    t_ingest >> t_preprocess >> t_train >> t_evaluate
