from typing import Tuple

import pandas as pd
import kagglehub

from config_manager import ConfigManager

class DataPreprocessor:
    """Handles loading and basic preprocessing of credit risk data"""
    
    def __init__(self):
        config_manager = ConfigManager()
        self.application_data = None
        self.credit_record = None
        self.kaggle_path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction")
        self.os_sep = config_manager.get_config('os_separator')
        self.credit_application_path = self.kaggle_path + self.os_sep + config_manager.get_config('credit_application_dataset')
        self.credit_record_path = self.kaggle_path + self.os_sep + config_manager.get_config('credit_record_dataset')

        self.application_path = None
    
    def download_dataset_from(self):
        """Download the dataset from Kaggle"""
        # Implementation details depend on the specific Kaggle API and dataset
        # Download latest version

        # read files 
        self.load_data()
        print("Dataset downloaded and loaded successfully.")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the application and credit record data"""

        self.application_data = pd.read_csv(self.credit_application_path, encoding='utf-8')
        self.credit_record = pd.read_csv(self.credit_record_path, encoding='utf-8')
        return self.application_data, self.credit_record
        
    def create_target_variable(self):
        """Create target variable based on credit record status"""
        # Find account open month
        begin_month = pd.DataFrame(self.credit_record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
        begin_month = begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'})
        
        # Add target variable
        self.credit_record['dep_value'] = None
        self.credit_record.loc[self.credit_record['STATUS'] =='2', 'dep_value'] = 'Yes'
        self.credit_record.loc[self.credit_record['STATUS'] =='3', 'dep_value'] = 'Yes'
        self.credit_record.loc[self.credit_record['STATUS'] =='4', 'dep_value'] = 'Yes'
        self.credit_record.loc[self.credit_record['STATUS'] =='5', 'dep_value'] = 'Yes'
        
        cpunt = self.credit_record.groupby('ID').count()
        cpunt['dep_value'][cpunt['dep_value'] > 0] = 'Yes' 
        cpunt['dep_value'][cpunt['dep_value'] == 0] = 'No'
        cpunt = cpunt[['dep_value']]
        
        # Merge with application data
        new_data = pd.merge(self.application_data, begin_month, how='inner', on='ID')
        new_data = pd.merge(new_data, cpunt, how='inner', on='ID')
        new_data['target'] = new_data['dep_value']
        new_data.loc[new_data['target']=='Yes', 'target'] = 1
        new_data.loc[new_data['target']=='No', 'target'] = 0
        
        return new_data
        
    def rename_columns(self, data):
        """Rename columns to more readable format"""
        column_map = {
            'CODE_GENDER': 'Gender',
            'FLAG_OWN_CAR': 'Car',
            'FLAG_OWN_REALTY': 'Reality',
            'CNT_CHILDREN': 'ChldNo',
            'AMT_INCOME_TOTAL': 'inc',
            'NAME_EDUCATION_TYPE': 'edutp',
            'NAME_FAMILY_STATUS': 'famtp',
            'NAME_HOUSING_TYPE': 'houtp',
            'FLAG_EMAIL': 'email',
            'NAME_INCOME_TYPE': 'inctp',
            'FLAG_WORK_PHONE': 'wkphone',
            'FLAG_PHONE': 'phone',
            'CNT_FAM_MEMBERS': 'famsize',
            'OCCUPATION_TYPE': 'occyp'
        }
        return data.rename(columns=column_map)
