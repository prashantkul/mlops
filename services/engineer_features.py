from feature_engineer import FeatureEngineer

from dvc_operations.download import DVCReader
from config_manager import ConfigManager
import pandas as pd

from feature_store import FeatureUploader

config_manager = ConfigManager()
dvc = DVCReader()

application_data_path = config_manager.get_config('credit_application_dataset')
credit_record_path = config_manager.get_config('credit_record_dataset')

application_data = dvc.read_dataframe(application_data_path)
credit_record = dvc.read_dataframe(credit_record_path)

fe = FeatureEngineer(application_data=application_data, credit_record=credit_record)
data = fe.engineer_features()

filename = "cred_card_featured_engg.csv"
data.to_csv('data/cred_card_featured_engg.csv', index=False)

file_path = 'data/cred_card_featured_engg.csv'  # Path to the CSV with features

uploader = FeatureUploader(file_path)
uploader.upload()





