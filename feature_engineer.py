import json
import warnings
from typing import List, Dict, Optional, Any
import redis

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Handles feature engineering tasks for credit risk analysis"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.data = None
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    def _cache_key(self, feature_name: str, id: str) -> str:
        return f"credit_risk:features:{feature_name}:{id}"
        
    def store_features(self, id: str, features: Dict[str, Any]):
        """Store features permanently in Redis"""
        for feature_name, value in features.items():
            key = self._cache_key(feature_name, id)
            self.redis_client.set(key, json.dumps(value))
        
    def engineer_features(self, data):
        """Apply all feature engineering steps"""
        self.data = data.copy()
        self.handle_missing_values()
        self.encode_categorical()
        self.create_age_groups()
        self.create_work_time_groups()
        self.create_income_groups()
        self.create_family_size_groups()
        return self.data
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        self.data = self.data.dropna()
        self.data = self.data.mask(self.data == 'NULL').dropna()
        
    def encode_categorical(self):
        """Encode categorical variables"""
        self.data['Gender'] = self.data['Gender'].replace(['F','M'],[0,1])
        self.data['Reality'] = self.data['Reality'].replace(['N','Y'],[0,1])
        self.data['Car'] = self.data['Car'].replace(['N','Y'],[0,1])
        
    def create_age_groups(self):
        """Create age groups from DAYS_BIRTH"""
        self.data['Age'] = -(self.data['DAYS_BIRTH'])//365
        self.data = self.create_groups(self.data, 'Age', 5, 
                                     ["lowest","low","medium","high","highest"])
        
    def create_work_time_groups(self):
        """Create work time groups from DAYS_EMPLOYED"""
        self.data['worktm'] = -(self.data['DAYS_EMPLOYED'])//365
        self.data[self.data['worktm']<0] = np.nan
        self.data['worktm'].fillna(self.data['worktm'].mean(), inplace=True)
        self.data = self.create_groups(self.data, 'worktm', 5,
                                     ["lowest","low","medium","high","highest"])
        
    def create_income_groups(self):
        """Create income groups"""
        self.data['inc'] = self.data['inc']/10000
        self.data = self.create_groups(self.data, 'inc', 3, ["low","medium","high"], qcut=True)
        
    def create_family_size_groups(self):
        """Create family size groups"""
        self.data['famsize'] = self.data['famsize'].astype(int)
        self.data['famsizegp'] = self.data['famsize']
        self.data['famsizegp'] = self.data['famsizegp'].astype(object)
        self.data.loc[self.data['famsizegp']>=3, 'famsizegp'] = '3more'
        
    @staticmethod
    def create_groups(df, col, bins, labels, qcut=False):
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
