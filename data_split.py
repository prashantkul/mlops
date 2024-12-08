import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from dvc_operations.download import DVCReader


class DataSplitter:
    """Handles splitting of data into train and test sets"""
    """Includes optional resampling of minority class"""
    
    def __init__(self, data_path = "cred_card_featured_engg.csv", resample: bool = True):
        reader = DVCReader()
        self.data = reader.read_dataframe(self.data_path)
        self.train = None
        self.test = None
        self.resample = resample
        self.drift_random = None
        self.drift_gaus = None
        self.drift_lap = None

    def split_data(self, resample_ratio = .25, split: float = 0.2):
        # Note: Synthetic data generation at this stage was only for the hypothetical of our project, not for real-world
        # application/production. This dataset was relatively small, and the train/validate/test split left only 5 records of the
        # minority class in the test (production) dataset, which made for an unrealistic/anticlimactic demo of the model pipeline.

        if self.resample == True:
            smote = SMOTE(sampling_strategy=resample_ratio, random_state=47)
            X_resampled, y_resampled = smote.fit_resample(self.data.drop(columns=['target']), self.data['target'])
            resampled = pd.DataFrame(X_resampled, columns=X_resampled.columns)
            resampled['target'] = y_resampled
            self.data = resampled

        self.train, self.test = train_test_split(self.data, test_size=split, random_state=47, stratify=self.data['target'])
        self.train.to_csv("data\cred_card_featured_eng_train.csv", index=False)
        self.test.to_csv("data\cred_card_featured_eng_test_ref.csv.csv", index=False)
    
    def drift_data(self):
        """Simulate data drift by altering feature columns"""
        # Randomly alter to income (continuous variable) for 50% of observations and randomly assign gender (categorical variable) to heavily skew male
        self.drift_random = self.test.copy()
        multiplier = np.random.uniform(0.75, 1.5, len(self.drift_random))
        mask = np.random.choice([1, 0], size=len(self.drift_random), p=[0.5, 0.5])
        multiplier = np.where(mask == 1, multiplier, 1)
        self.drift_random['inc'] = self.drift_random['inc'] * multiplier
        self.drift_random['Gender'] = np.random.choice([0, 1], size=len(df), p=[0.25, 0.75])

        # Calculate a moderate level of differential privacy for income:
        sensitivity = self.test['income'].max() - self.test['income'].min()
        epsilon = 0.5
        scale = sensitivity/epsilon

        # Add Gaussian noise to income (continuous variable)
        self.drift_gaus = self.test.copy()
        gauss_noise = np.random.normal(0, scale, len(self.drift_gaus))
        self.drift_gaus['inc'] = self.drift_gaus['inc'] + gauss_noise

        # Add Laplacian noise to income (continuous variable)
        self.drift_lap = self.test.copy()
        lap_noise = np.random.laplace(0, scale, len(self.drift_lap))
        self.drift_lap['inc'] = self.drift_lap['inc'] + lap_noise

        self.drift_random.to_csv("data\credit_record_test_drift_random.csv", index=False)
        self.drift_gaus.to_csv("data\credit_record_test_drift_gaus.csv", index=False)
        self.drift_lap.to_csv("data\credit_record_test_drift_lap.csv", index=False)