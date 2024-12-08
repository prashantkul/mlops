import json
import warnings
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
from config_manager import ConfigManager

import h2o
from h2o.automl import H2OAutoML

from gcloud_handler import GCSHandler

from dvc_operations.download import DVCReader

class ModelTrainer:
    """Handles model training and evaluation for credit risk prediction"""
    
    def __init__(self, file_path = "cred_card_featured_engg_train.csv", to_split: bool = True):
        config = ConfigManager()
        reader = DVCReader()
        self.features = config.get_config('features')  # Feature column names from config
        self.file_path = file_path
        self.data = reader.read_dataframe(self.file_path)
        self.h2o_port = config.get_config('h2o_port')   # h2o port from config
        self.h2o_ip = config.get_config('h2o_ip')   # h2o ip from config
        
    def train_model(self):
        """Train Model Using H2O AutoML"""

        # Verify correct columns:
        column_list = self.features
        column_list.append('target')

        if column_list != self.data.columns.tolist():
            drop_cols = [col for col in self.data.columns if col not in self.features + ['target']]
            self.data = self.data.drop(columns=drop_cols)
        
        # Initialize H2O:
        #h2o = h2o.init(port = 54321, ip = "35.184.233.137")
        h2o = h2o.init(port = self.h2o_port, ip = self.h2o_ip)

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
        h2o.save_model(self.model, path='models/credit_risk_model')

        # Upload model to GCP:
        gcs = GCSHandler()
        gcs.handle_file("upload", "models/credit_risk_model", "models/credit_risk_model")

        '''
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


