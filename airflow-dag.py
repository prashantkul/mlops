from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from data_preprocessor import DataPreprocessor
from model_trainer import ModelTraining
from model_trainer import ModelEvaluation
from model_deployer import ModelDeployment


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='credit_card_pipeline',
    default_args=default_args,
    description='Pipeline for Credit Card Approval Prediction',
    schedule_interval='@daily',
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:
    
    # Define the tasks using PythonOperator
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=DataPreprocessor.preprocess_data,
    )
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=ModelTraining.train_model,
    )
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=ModelEvaluation.evaluate_model,
    )
    
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=ModelDeployment.deploy_model,
    )
    
    # Set task dependencies
    preprocess_task >> train_task >> evaluate_task >> deploy_task