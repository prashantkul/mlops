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
    
    def __init__(self, data: pd.DataFrame):
        self.model = None
        self.data = data
        self.train = None
        self.test = None
        self.drift = None
        #self.val = None
        #self.X_test = None
        #self.y_test = None
        
    def prepare_training_data(self, split = 0.1, resample = True):
        """Prepare data for training"""
        
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

        self.train, self.test = train_test_split(self.data, test_size=split, random_state=10086)
        
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


    def create_feature_drift(self):
        drift_1 = self.test.copy()
        drift_1["ChldNo_1"] = self.test["ChldNo_1"].sample(frac=1).reset_index(drop=True)
        drift_1["gp_Age_low"] = self.test["gp_Age_low"].sample(frac=1).reset_index(drop=True)
        drift_1["gp_worktm_medium"] = self.test["gp_worktm_medium"].sample(frac=1).reset_index(drop=True)
        self.drift = drift_1

    def evaluate_model(self, data = None):
        """Evaluate model performance"""
        if data is None:
            data = self.test
            set_name = 'Test'

        elif data = 'drift':
            data = self.drift
            set_name = 'Drifted'
        
        # Split data into features and target
        X = data.drop(columns=['target'])
        y = data['target']

        y_predict = self.model.predict(X)
        accuracy = accuracy_score(y, y_predict)
        conf_matrix = confusion_matrix(y, y_predict)
        
        print(f'Accuracy Score is {accuracy:.5}')
        print(pd.DataFrame(conf_matrix))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix/np.sum(conf_matrix), 
                   annot=True, fmt='.2%',
                   cmap='Blues')
        plt.title(f'Normalized Confusion Matrix: {set_name} Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        ### May need to add logic here to save and export evaluation results, unsure how this will interract with AirFlow and Model Monitoring ###