
## use click on the command line to create a preprocess.py script
## use the click.option to add arguments to the script

import click
from data_preprocessor import DataPreprocessor

@click.command()
@click.option('--application-data-path', default='data/application_data.csv', help='Path to the application data')
@click.option('--credit-record-path', default='data/credit_record.csv', help='Path to the credit record data')
def preprocess(application_data_path, credit_record_path):
    """Preprocess the data"""
    data_preprocessor = DataPreprocessor()
    data_preprocessor.download_dataset_from()
    data_preprocessor.load_data(application_data_path, credit_record_path)
    data_preprocessor.create_target_variable()
    data_preprocessor.rename_columns()
    # save the preprocessed data to a file to data/preprocessed_data.csv

### create a command to version the data via dvc
