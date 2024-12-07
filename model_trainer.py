import json
import warnings
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.automl import H2OAutoML

class ModelTrainer:
    """Handles model training and evaluation for credit risk prediction"""
    
    def __init__(self, data_path, to_split: bool = True):
        self.model = None
        self.data = self.data = pd.read_csv(data_path)
        self.train = None
        self.test = None
        self.drift = None
        self.to_split = to_split
        #self.val = None
        #self.X_test = None
        #self.y_test = None
        
    def prepare_training_data(self, split = 0.1, resample = True):
        """Prepare data for training"""
        
        self.data = pd.read_csv(self.data_path)

        # Verify correct columns:
        features = ["Gender","Reality","ChldNo_1","wkphone","gp_Age_high", "gp_Age_highest", "gp_Age_low", "gp_Age_lowest","gp_worktm_high", "gp_worktm_highest", "gp_worktm_low", 
                "gp_worktm_medium","occyp_hightecwk","occyp_officewk","famsizegp_1", "famsizegp_3more","houtp_Co-op apartment", "houtp_Municipal apartment","houtp_Office apartment",
                "houtp_Rented apartment","houtp_With parents","edutp_Higher education","edutp_Incomplete higher", "edutp_Lower secondary","famtp_Civil marriage","famtp_Separated",
                "famtp_Single / not married","famtp_Widow"]
        if self.data.columns != features + ['target']:
            drop_cols = [col for col in self.data.columns if col not in features + ['target']]
            self.data = self.data.drop(columns=drop_cols)
        
        # Synthetically generate extra data for minority class (this is only for the hypothetical of our project, not for real-world application/production):
        if resample == True:
            smote = SMOTE(sampling_strategy=0.25, random_state=10086)
            X_resampled, y_resampled = smote.fit_resample(self.data.drop(columns=['target']), self.data['target'])
            resampled = pd.DataFrame(X_resampled, columns=features)
            resampled['target'] = y_resampled
            self.data = resampled

        if self.to_split == True:
            self.train, self.test = train_test_split(self.data, test_size=split, random_state=10086)
        else:
            self.train = self.data

        '''
        # Split Data
        self.train, test_val = train_test_split(self.data, test_size=split, random_state=10086)
        y_test_val = test_val['target'].astype('int')
        X_test_val = test_val.drop(columns=['target'])
        
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split( X_test_val, y_test_val, test_size=split, random_state=10086)

        # Apply SMOTE to training data for balance
        if balance == True:
            y_train = self.train['target'].astype('int')
            #X_train = self.train[features]
            X_train = self.train.drop(columns=['target'])
            X_balance, y_balance = SMOTE().fit_resample(X_train, y_train)
            train_balance = pd.DataFrame(X_balance, columns=X.columns)
            train_balance['target'] = y_balance
            self.train = train_balance
        '''
        

    def train_model(self):
        '''
        """Train Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=16
        )
        self.model.fit(self.X_train, self.y_train)
        '''      

        """Train Model Using H2O AutoML"""
        # Initialize H2O:
        #h2o = h2o.init(port = h20_port, ip = h20_ip)
        h2o = h2o.init(port = 54321, ip = "35.184.233.137")

        # Convert training data to H2OFrame:
        h2o_data = h2o.H2OFrame(self.train)
        
        # Split off validation set and designate x and y for H2O:
        train, val = h2o_data.split_frame(ratios = [.777], seed = 47)
        x = train.columns
        y = 'target'
        x.remove(y)

        # Run AutoML for 20 base models
        aml = H2OAutoML(max_models=20, seed=47, balance_classes = True, project_name = "mlops_final")
        aml.train(x=x, y=y, training_frame=train, leaderboard_frame = val)

        self.model = aml.leader

    def save_model_and_test_data(self):
        # Save model to GCS bucket:
        h2o.save_model(self.model, path = 'gs://mlops_final/credit_risk_model')

        # Save test set to GCS bucket:
        self.test.to_csv('gs://mlops_final/credit_risk_test_data.csv', index=False)
