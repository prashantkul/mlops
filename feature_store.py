from feast import Entity, Feature, FeatureView, ValueType, FileSource
import pandas as pd
from feast import FeatureStore
from datetime import datetime

from config_manager import ConfigManager


class FeatureUploader:
    def __init__(self, file_path: str):
        """
        Initialize the FeastFeatureUploader.

        Args:
            repo_path (str): Path to the Feast repository.
            file_path (str): Path to the CSV file containing feature data.
        """
        cm = ConfigManager()
        self.config_path = "./feast_repo/feast_config.yaml"  # Optional, path to your Feast config file
        self.repo_path = f"gs://{cm.get_config('gcs_bucket')}/{cm.get_config('gcs_remote_path')}/"  # Path to your Feast repository
        self.file_path = file_path

        self.file_source = FileSource(
            path=self.file_path,  # Path to your CSV file
            event_timestamp_column="event_time",  # Ensure you have a timestamp column
        )

        self.fs = FeatureStore(repo_path=self.repo_path)
        self.entity = Entity(
            name="ID",
            value_type=ValueType.INT64,
            description="UUID"
        )

        self.feature_view = FeatureView(
            name="credit_application_features",
            entities=["ID"],  # This should match the Entity
            ttl=None,
            features=[
                Feature(name="type", dtype=ValueType.INT64),
                Feature(name="Gender", dtype=ValueType.INT64),
                Feature(name="Car", dtype=ValueType.INT64),
                Feature(name="Reality", dtype=ValueType.INT64),
                Feature(name="inc", dtype=ValueType.FLOAT),
                Feature(name="DAYS_BIRTH", dtype=ValueType.INT64),
                Feature(name="DAYS_EMPLOYED", dtype=ValueType.INT64),
                Feature(name="FLAG_MOBIL", dtype=ValueType.INT64),
                Feature(name="wkphone", dtype=ValueType.INT64),
                Feature(name="phone", dtype=ValueType.INT64),
                Feature(name="email", dtype=ValueType.INT64),
                Feature(name="famsize", dtype=ValueType.INT64),
                Feature(name="begin_month", dtype=ValueType.FLOAT),
                Feature(name="dep_value", dtype=ValueType.STRING),
                Feature(name="ChldNo_1", dtype=ValueType.INT64),
                Feature(name="ChldNo_2", dtype=ValueType.INT64),
                Feature(name="ChldNo_3", dtype=ValueType.INT64),
                Feature(name="ChldNo_4", dtype=ValueType.INT64),
                Feature(name="ChldNo_5", dtype=ValueType.INT64),
                Feature(name="ChldNo_7", dtype=ValueType.INT64),
                Feature(name="ChldNo_14", dtype=ValueType.INT64),
                Feature(name="ChldNo_19", dtype=ValueType.INT64),
                Feature(name="gp_inc_high", dtype=ValueType.INT64),
                Feature(name="gp_inc_medium", dtype=ValueType.INT64),
                Feature(name="Age", dtype=ValueType.INT64),
                Feature(name="gp_Age_high", dtype=ValueType.INT64),
                Feature(name="gp_Age_highest", dtype=ValueType.INT64),
                Feature(name="gp_Age_low", dtype=ValueType.INT64),
                Feature(name="gp_Age_lowest", dtype=ValueType.INT64),
                Feature(name="worktm", dtype=ValueType.INT64),
                Feature(name="gp_worktm_high", dtype=ValueType.INT64),
                Feature(name="gp_worktm_highest", dtype=ValueType.INT64),
                Feature(name="gp_worktm_low", dtype=ValueType.INT64),
                Feature(name="gp_worktm_medium", dtype=ValueType.INT64),
                Feature(name="famsizegp_1", dtype=ValueType.INT64),
                Feature(name="famsizegp_3more", dtype=ValueType.INT64),
                Feature(name="inctp_Commercial_associate", dtype=ValueType.INT64),
                Feature(name="inctp_State_servant", dtype=ValueType.INT64),
                Feature(name="occyp_hightecwk", dtype=ValueType.INT64),
                Feature(name="occyp_officewk", dtype=ValueType.INT64),
                Feature(name="houtp_Co_op_apartment", dtype=ValueType.INT64),
                Feature(name="houtp_Municipal_apartment", dtype=ValueType.INT64),
                Feature(name="houtp_Office_apartment", dtype=ValueType.INT64),
                Feature(name="houtp_Rented_apartment", dtype=ValueType.INT64),
                Feature(name="houtp_With_parents", dtype=ValueType.INT64),
                Feature(name="edutp_Higher_education", dtype=ValueType.INT64),
                Feature(name="edutp_Incomplete_higher", dtype=ValueType.INT64),
                Feature(name="edutp_Lower_secondary", dtype=ValueType.INT64),
                Feature(name="famtp_Civil_marriage", dtype=ValueType.INT64),
                Feature(name="famtp_Separated", dtype=ValueType.INT64),
                Feature(name="famtp_Single_not_married", dtype=ValueType.INT64),
                Feature(name="famtp_Widow", dtype=ValueType.INT64),
            ],
            online=True,
            input=self.file_source,
        )

    def load_data(self):
        """
        Load the CSV file into a Pandas DataFrame and ensure event_time column exists.
        """
        self.df = pd.read_csv(self.file_path)

        # Add the event_time column if it doesn't exist
        if 'event_time' not in self.df.columns:
            self.df['event_time'] = datetime.now()

        # Ensure that the event_time column is in datetime format
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])

    def apply_feature_view(self):
        """
        Apply the feature view to the Feast feature store.
        """
        self.fs.apply([self.feature_view])
        print(f"Feature view '{self.feature_view.name}' applied successfully.")

    def upload(self):
        """
        Load the data from the CSV file and ingest it to the Feast feature store.
        """
        self.load_data()
        self.apply_feature_view()
        print(f"Features from {self.file_path} have been uploaded successfully.")

