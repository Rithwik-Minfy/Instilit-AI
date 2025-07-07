from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='retrain_pipeline_dag',
    default_args=default_args,
    description='Retrains salary prediction pipeline when drift is detected',
    schedule_interval=None,  # Only trigger manually
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'retrain'],
) as dag:

    # Task: Run the pipeline script
    run_pipeline = BashOperator(
        task_id='run_salary_pipeline',
        bash_command='python /opt/airflow/src/pipeline.py',
    )

    run_pipeline
