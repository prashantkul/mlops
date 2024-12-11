from config_manager import ConfigManager
from dvc_operations.download import DVCReader
from feature_engineer import FeatureEngineer
from feature_store import FeatureStore


def engineer():
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

    file_path = 'data/cred_card_featured_engg.csv'

    uploader = FeatureStore(file_path, key="base_feature_set_2")
    uploader.upload()


if __name__ == "__main__":
    engineer()
