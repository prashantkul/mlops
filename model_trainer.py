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

class ModelTrainer:
    """Handles model training and evaluation for credit risk prediction"""
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_training_data(self, data, feature_columns):
        """Prepare data for training"""
        Y = data['target'].astype('int')
        X = data[feature_columns]
        
        # Apply SMOTE for balance
        X_balance, Y_balance = SMOTE().fit_resample(X, Y)
        X_balance = pd.DataFrame(X_balance, columns=X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_balance, Y_balance,
            stratify=Y_balance,
            test_size=0.3,
            random_state=10086
        )
        
    def train_model(self):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=16
        )
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """Evaluate model performance"""
        y_predict = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_predict)
        conf_matrix = confusion_matrix(self.y_test, y_predict)
        
        print(f'Accuracy Score is {accuracy:.5}')
        print(pd.DataFrame(conf_matrix))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix/np.sum(conf_matrix), 
                   annot=True, fmt='.2%',
                   cmap='Blues')
        plt.title('Normalized Confusion Matrix: Random Forests')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()