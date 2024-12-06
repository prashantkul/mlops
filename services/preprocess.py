from data_preprocessor import DataPreprocessor

"""Preprocess the data"""
data_preprocessor = DataPreprocessor()
data_preprocessor.download_dataset_from()

application_data, credit_record = data_preprocessor.load_data()
application_data.to_csv('data/application_data.csv', index=False)

credit_record = data_preprocessor.rename_columns(credit_record)
credit_record.to_csv('data/credit_record.csv', index=False)

credit_record = data_preprocessor.create_target_variable()
credit_record.to_csv('data/credit_record_preprocessed.csv', index=False)
