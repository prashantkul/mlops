from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    # Load dataset
    data = pd.read_csv('/path/to/credit_card_data.csv')

    # Split into features (X) and outcome (y)
    X = data.drop(columns=['Approval'])
    y = data['Approval']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save datasets
    X_train.to_csv('/path/to/X_train.csv', index=False)
    X_test.to_csv('/path/to/X_test.csv', index=False)
    y_train.to_csv('/path/to/y_train.csv', index=False)
    y_test.to_csv('/path/to/y_test.csv', index=False)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
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
    
    # Define tasks
    split_task = PythonOperator(
        task_id='split_dataset',
        python_callable=split_dataset
    )

    # Define task dependencies
    split_task