import json
import warnings
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Handles feature engineering tasks for credit risk analysis"""

    def __init__(self, application_data, credit_record):
        self.application_data = application_data
        self.credit_record = credit_record
        self.data = self.merge_applications_with_credit_record()

    def merge_applications_with_credit_record(self):
        """Merge applications with credit record"""
        self.rename_columns()
        begin_month = pd.DataFrame(self.credit_record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
        begin_month = begin_month.rename(columns={'MONTHS_BALANCE': 'begin_month'})
        return pd.merge(self.application_data, begin_month, how="left", on="ID")

    def _cache_key(self, feature_name: str, id: str) -> str:
        return f"credit_risk:features:{feature_name}:{id}"

    def store_features(self, id: str, features: Dict[str, Any]):
        """Store features permanently in Redis"""
        for feature_name, value in features.items():
            key = self._cache_key(feature_name, id)
            self.redis_client.set(key, json.dumps(value))

    def engineer_features(self) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        self.data = self.data.copy()
        self.handle_missing_values()
        self.encode_categorical()
        self.create_age_groups()
        self.create_work_time_groups()
        self.create_income_groups()
        self.create_family_size_groups()
        self.one_hot_encode_columns()
        return self.data

    def rename_columns(self) -> pd.DataFrame:
        self.application_data.rename(columns={'CODE_GENDER': 'Gender', 'FLAG_OWN_CAR': 'Car', 'FLAG_OWN_REALTY': 'Reality',
                                 'CNT_CHILDREN': 'ChldNo', 'AMT_INCOME_TOTAL': 'inc',
                                 'NAME_EDUCATION_TYPE': 'edutp', 'NAME_FAMILY_STATUS': 'famtp',
                                 'NAME_HOUSING_TYPE': 'houtp', 'FLAG_EMAIL': 'email',
                                 'NAME_INCOME_TYPE': 'inctp', 'FLAG_WORK_PHONE': 'wkphone',
                                 'FLAG_PHONE': 'phone', 'CNT_FAM_MEMBERS': 'famsize',
                                 'OCCUPATION_TYPE': 'occyp'
                                 }, inplace=True)
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        self.data = self.data.dropna()
        self.data = self.data.mask(self.data == 'NULL').dropna()

    def encode_categorical(self):
        """Encode categorical variables"""
        self.data['Gender'] = self.data['Gender'].replace(['F', 'M'], [0, 1])
        self.data['Reality'] = self.data['Reality'].replace(['N', 'Y'], [0, 1])
        self.data['Car'] = self.data['Car'].replace(['N', 'Y'], [0, 1])
        self.data['wkphone'] = self.data['wkphone'].astype(int)
        self.data['phone'] = self.data['phone'].astype(int)
        self.data['email'] = self.data['email'].astype(int)

    def create_age_groups(self):
        """Create age groups from DAYS_BIRTH"""
        self.data['Age'] = -(self.data['DAYS_BIRTH']) // 365
        self.data = self.create_groups(self.data, 'Age', 5,
                                       ["lowest", "low", "medium", "high", "highest"])

    def create_work_time_groups(self):
        """Create work time groups from DAYS_EMPLOYED"""
        self.data['worktm'] = -(self.data['DAYS_EMPLOYED']) // 365
        self.data.loc[self.data['worktm'] < 0, 'worktm'] = np.nan
        self.data['worktm'].fillna(self.data['worktm'].mean(), inplace=True)
        self.data = self.create_groups(self.data, 'worktm', 5,
                                       ["lowest", "low", "medium", "high", "highest"])

    def create_income_groups(self):
        """Create income groups"""
        self.data['inc'] = self.data['inc'] / 10000
        self.data = self.create_groups(self.data, 'inc', 3,
                                       ["low", "medium", "high"], qcut=True)

    def create_family_size_groups(self):
        """Create family size groups"""
        self.data['famsize'] = self.data['famsize'].astype(int)
        self.data['famsizegp'] = self.data['famsize']
        self.data.loc[self.data['famsizegp'] >= 3, 'famsizegp'] = '3more'

    @staticmethod
    def create_groups(df: pd.DataFrame, col: str, bins: int, labels: List[str], qcut: bool = False) -> pd.DataFrame:
        """Helper method to create groups from continuous variables"""
        if qcut:
            localdf = pd.qcut(df[col], q=bins, labels=labels)
        else:
            localdf = pd.cut(df[col], bins=bins, labels=labels)

        localdf = pd.DataFrame(localdf)
        name = 'gp_' + col
        localdf[name] = localdf[col]
        df = df.join(localdf[name])
        df[name] = df[name].astype(object)
        return df

    def one_hot_encode_columns(self):
        """One-hot encode categorical columns, but keep them as integers"""
        categorical_columns = ['ChldNo', 'inctp', 'occyp', 'houtp', 'edutp', 'famtp', 'famsizegp']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data = self.convert_dummy(self.data, col)

    @staticmethod
    def convert_dummy(df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Convert categorical features to one-hot encoded integers"""
        dummies = pd.get_dummies(df[feature], prefix=feature, dtype=int)
        df = df.drop(columns=[feature])
        df = pd.concat([df, dummies], axis=1)
        return df
