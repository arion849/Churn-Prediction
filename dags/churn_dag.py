from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def hello_churn():
    print("Airflow Dag working")



with DAG(
    dag_id = "dummy_churn_dag",
    start_date = datetime(2025, 1, 1),
    schedule_interval = None,
    catchup = False,
) as dag:
    task_print = PythonOperator(
        task_id = "print_hello",
        python_callable = hello_churn
    )

    task_print
