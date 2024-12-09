import uuid

import pandas as pd
import redis
from datetime import datetime
import json


class FeatureStore:
    def __init__(self, file_path: str, key=uuid.uuid4()):
        """
        Initialize the FeatureUploader.

        Args:
            file_path (str): Path to the CSV file containing feature data.
        """
        # Hard-coded Redis configuration
        self.redis_host = 'localhost'  # Default Redis host
        self.redis_port = 6379  # Default Redis port
        self.redis_db = 0  # Default Redis DB

        # Connect to Redis
        self.redis_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, db=self.redis_db)

        # CSV file path
        self.file_path = file_path
        self.key = key

    def load_data(self):
        """
        Load the CSV file into a Pandas DataFrame and ensure event_time column exists.
        """
        self.df = pd.read_csv(self.file_path)

        if 'event_time' not in self.df.columns:
            self.df['event_time'] = datetime.now()
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])

    def stage_features(self, key):
        """
        Store the features in Redis.
        """
        for _, row in self.df.iterrows():
            feature_data = {
                "ID": row['ID'],
                "event_time": row['event_time'].isoformat(),
                "Gender": row['Gender'],
                "Car": row['Car'],
                "Reality": row['Reality'],
                "inc": row['inc'],
                "DAYS_BIRTH": row['DAYS_BIRTH'],
                "DAYS_EMPLOYED": row['DAYS_EMPLOYED'],
                "FLAG_MOBIL": row['FLAG_MOBIL'],
                "wkphone": row['wkphone'],
                "phone": row['phone'],
                "email": row['email'],
                "famsize": row['famsize'],
                "begin_month": row['begin_month'],
                "dep_value": row['dep_value'],
                "ChldNo_1": row['ChldNo_1'],
                "ChldNo_2": row['ChldNo_2'],
                "ChldNo_3": row['ChldNo_3'],
                "ChldNo_4": row['ChldNo_4'],
                "ChldNo_5": row['ChldNo_5'],
                "ChldNo_7": row['ChldNo_7'],
                "ChldNo_14": row['ChldNo_14'],
                "ChldNo_19": row['ChldNo_19'],
                "gp_inc_high": row['gp_inc_high'],
                "gp_inc_medium": row['gp_inc_medium'],
                "Age": row['Age'],
                "gp_Age_high": row['gp_Age_high'],
                "gp_Age_highest": row['gp_Age_highest'],
                "gp_Age_low": row['gp_Age_low'],
                "gp_Age_lowest": row['gp_Age_lowest'],
                "worktm": row['worktm'],
                "gp_worktm_high": row['gp_worktm_high'],
                "gp_worktm_highest": row['gp_worktm_highest'],
                "gp_worktm_low": row['gp_worktm_low'],
                "gp_worktm_medium": row['gp_worktm_medium'],
                "famsizegp_1": row['famsizegp_1'],
                "famsizegp_3more": row['famsizegp_3more'],
                "inctp_Commercial_associate": row['inctp_Commercial_associate'],
                "inctp_State_servant": row['inctp_State_servant'],
                "occyp_hightecwk": row['occyp_hightecwk'],
                "occyp_officewk": row['occyp_officewk'],
                "houtp_Co_op_apartment": row['houtp_Co_op_apartment'],
                "houtp_Municipal_apartment": row['houtp_Municipal_apartment'],
                "houtp_Office_apartment": row['houtp_Office_apartment'],
                "houtp_Rented_apartment": row['houtp_Rented_apartment'],
                "houtp_With_parents": row['houtp_With_parents'],
                "edutp_Higher_education": row['edutp_Higher_education'],
                "edutp_Incomplete_higher": row['edutp_Incomplete_higher'],
                "edutp_Lower_secondary": row['edutp_Lower_secondary'],
                "famtp_Civil_marriage": row['famtp_Civil_marriage'],
                "famtp_Separated": row['famtp_Separated'],
                "famtp_Single_not_married": row['famtp_Single_not_married'],
                "famtp_Widow": row['famtp_Widow'],
            }

            # Use the ID as the Redis key and store the feature data as a JSON string
            return key, feature_data

    def upload(self):
        """
        Load the data from the CSV file and upload it to Redis.
        """
        self.load_data()
        key, features = self.stage_features(self.key)
        self.redis_client.set(f"features:{key}", json.dumps(features))
        print(f"Features from {self.file_path} have been uploaded to Redis successfully with key '{key}'.")

    def get_features(self, key: str, as_dataframe: bool = False, as_csv: bool = False, csv_path: str = "features.csv"):
        """
        Get the features for a given key from Redis.

        Args:
            key (str): The key to retrieve the features for.
            as_dataframe (bool): If True, return the features as a DataFrame.
            as_csv (bool): If True, return the features as a CSV file.
            csv_path (str): Path where the CSV file will be saved (default: 'features.csv').

        Returns:
            dict | pd.DataFrame | None: The features for the given key as a dict, DataFrame, or CSV file.
        """
        try:
            raw_data = self.redis_client.get(f"features:{key}")

            if raw_data is None:
                print(f"No data found for key '{key}'")
                return None

            features_dict = json.loads(raw_data.decode('utf-8'))

            if as_dataframe:
                df = pd.DataFrame([features_dict])
                return df

            if as_csv:
                df = pd.DataFrame([features_dict])
                df.to_csv(csv_path, index=False)
                print(f"Features saved to CSV at '{csv_path}'")
                return csv_path

            return features_dict

        except Exception as e:
            print(f"Error retrieving features for key '{key}': {e}")
            return None
