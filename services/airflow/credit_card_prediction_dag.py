from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from services.preprocess import preprocess
from services.engineer_features import engineer
from services.split_data import split_data
from services.train_model import train_model


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='credit_card_pipeline',
    default_args=default_args,
    description='Pipeline for Credit Card Approval Prediction',
    schedule_interval='@daily',
    start_date=datetime(2024, 12, 1),
    catchup=False,
    tags=['credit-card', 'ml-pipeline'],
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess,
        provide_context=True,
        dag=dag,
    )

    feature_engineer_task = PythonOperator(
        task_id='feature_engineer',
        python_callable=engineer,
        provide_context=True,
        dag=dag,
    )

    split_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True,
        dag=dag,
    )


    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
        provide_context=True,
    )

    preprocess_task >> feature_engineer_task >> split_task >> train_task

