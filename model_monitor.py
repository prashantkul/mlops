import json
import warnings
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np



class ModelMonitor:
    """Monitor model performance and detect drift"""
    def __init__(self, model_path = "", test):
        #self.model = #load model from model_path
        self.test = test
        self.drift = []

    def create_feature_drift(self):
        drift_1 = self.test.copy()
        drift_1["ChldNo_1"] = self.test["ChldNo_1"].sample(frac=1).reset_index(drop=True)
        drift_1["gp_Age_low"] = self.test["gp_Age_low"].sample(frac=1).reset_index(drop=True)
        drift_1["gp_worktm_medium"] = self.test["gp_worktm_medium"].sample(frac=1).reset_index(drop=True)
        self.drift.append(drift_1)

        drift_2 = self.test.copy()
        drift_2["occyp_hightecwk"] = self.test["occyp_officewk"].sample(frac=1).reset_index(drop=True)
        drift_2["occyp_officewk"] = self.test["occyp_hightecwk"].sample(frac=1).reset_index(drop=True)
        drift_2["famsizegp_1"] = self.test["famsizegp_3more"].sample(frac=1).reset_index(drop=True)
        drift_2["famsizegp_3more"] = self.test["famsizegp_1"].sample(frac=1).reset_index(drop=True)
        self.drift.append(drift_2)

    def evaluate_model(self, data = 'test'):
        """Evaluate model performance"""
        if data == 'test':
            data = self.test
            set_name = 'Test'

        elif data == 'drift':
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
    
    def detect_drift(self):
        """Detect drift in data"""
        for i, drift_data in enumerate(self.drift):
            self.evaluate_model(data = drift_data)
            print(f'Drift {i+1} detected')