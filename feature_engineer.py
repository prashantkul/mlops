import warnings

import numpy as np
import pandas as pd

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

    def engineer_features(self) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        self.data = self.data.copy()
        self.handle_missing_values()
        self.encode_categorical()
        self.engineer_occyp_column()
        self.create_age_groups()
        self.create_work_time_groups()
        self.create_income_groups()
        self.create_family_size_groups()
        self.one_hot_encode_columns()
        self.create_target_from_credit_record()
        self.clean_column_names()
        return self.data

    def rename_columns(self) -> pd.DataFrame:
        self.application_data.rename(
            columns={'CODE_GENDER': 'Gender', 'FLAG_OWN_CAR': 'Car', 'FLAG_OWN_REALTY': 'Reality',
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

    def engineer_occyp_column(self):
        # Combine similar occupations into broader categories
        self.data.loc[
            self.data['occyp'].isin([
                'Cleaning staff', 'Cooking staff', 'Drivers', 'Laborers',
                'Low-skill Laborers', 'Security staff', 'Waiters/barmen staff'
            ]),
            'occyp'] = 'Laborwk'

        self.data.loc[
            self.data['occyp'].isin([
                'Accountants', 'Core staff', 'HR staff', 'Medicine staff',
                'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries'
            ]),
            'occyp'] = 'officewk'

        self.data.loc[
            self.data['occyp'].isin([
                'Managers', 'High skill tech staff', 'IT staff'
            ]),
            'occyp'] = 'hightecwk'

    def create_age_groups(self):
        self.data['Age'] = -(self.data['DAYS_BIRTH']) // 365
        self.data['gp_Age'] = pd.cut(self.data['Age'],
                                     bins=5,
                                     labels=["lowest", "low", "medium", "high", "highest"])

    def create_work_time_groups(self):
        self.data['worktm'] = -(self.data['DAYS_EMPLOYED']) // 365
        self.data.loc[self.data['worktm'] < 0, 'worktm'] = np.nan
        self.data['worktm'].fillna(self.data['worktm'].mean(), inplace=True)
        self.data['gp_worktm'] = pd.cut(self.data['worktm'],
                                        bins=5,
                                        labels=["lowest", "low", "medium", "high", "highest"])

    def create_income_groups(self):
        self.data['inc'] = self.data['inc'] / 10000
        self.data['gp_inc'] = pd.qcut(self.data['inc'],
                                      q=3,
                                      labels=["low", "medium", "high"])

    def create_family_size_groups(self):
        """Create family size groups"""
        self.data['famsize'] = self.data['famsize'].astype(int)
        self.data['famsizegp'] = self.data['famsize']
        self.data.loc[self.data['famsizegp'] >= 3, 'famsizegp'] = '3more'

    def create_target_from_credit_record(self):
        """Create target column for default prediction from credit record"""
        record = self.credit_record.copy()
        record['dep_value'] = None
        record.loc[record['STATUS'].isin(['2', '3', '4', '5']), 'dep_value'] = 'Yes'

        count_df = record.groupby('ID')['dep_value'].count().reset_index()
        count_df['dep_value'] = np.where(count_df['dep_value'] > 0, 'Yes', 'No')

        self.data = pd.merge(self.data, count_df[['ID', 'dep_value']], on='ID', how='left')
        self.data['target'] = np.where(self.data['dep_value'] == 'Yes', 1, 0)

    def one_hot_encode_columns(self):
        """One-hot encode categorical columns, but keep them as integers"""
        categorical_columns = ['ChldNo', 'inctp', 'occyp', 'houtp', 'edutp', 'famtp', 'famsizegp', 'gp_Age',
                               'gp_worktm', 'gp_inc']
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

    import re

    def clean_column_names(self):
        """Clean column names by replacing spaces with underscores and removing unwanted characters"""
        self.data.columns = [col.replace(" ", "_") for col in self.data.columns]
        self.data.columns = self.data.columns.str.replace(r'[_/\\-]+', '_', regex=True)
        self.data.columns = self.data.columns.str.replace(r'^\_|\_$', '', regex=True)
        self.data.columns = self.data.columns.str.replace(r'_{2,}', '_', regex=True)
