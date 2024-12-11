from config_manager import ConfigManager
from dvc_operations.download import DVCReader

import h2o
from h2o.automl import H2OAutoML

from gcs_handler import GCSHandler


class ModelTrainer:
    """Handles model training and evaluation for credit risk prediction"""
    
    def __init__(self, file_path = "cred_card_featured_eng_train.csv"):
        config = ConfigManager()
        reader = DVCReader()
        self.features = config.get_config('features')  # Feature column names from config
        self.data = reader.read_dataframe(file_path)
        self.h2o_port = config.get_config('h2o_port')   # h2o port from config
        self.h2o_ip = config.get_config('h2o_ip')   # h2o ip from config
        
    def train_model(self):
        """Train Model Using H2O AutoML"""

        # Verify correct columns:
        column_list = self.features
        column_list.append('target')

        if column_list != self.data.columns.tolist():
            drop_cols = [col for col in self.data.columns if col not in column_list]
            self.data = self.data.drop(columns=drop_cols)
        
        # Initialize H2O:
        #h2o.init(port = 54321, ip = "35.184.233.137")
        h2o.init(port = self.h2o_port, ip = self.h2o_ip)

        # Convert training data to H2OFrame:
        h2o_data = h2o.H2OFrame(self.data)
        h2o_data['target'] = h2o_data['target'].asfactor()
        
        # Split off validation set and designate x and y for H2O:
        train, val = h2o_data.split_frame(ratios = [.777], seed = 47)
        x = train.columns
        y = 'target'
        x.remove(y)

        # Run AutoML for 20 base models
        aml = H2OAutoML(max_models=20, seed=47, balance_classes = True, project_name = "mlops_final_project")
        aml.train(x=x, y=y, training_frame=train, leaderboard_frame = val)


        # Find the highest-ranking non-ensemble model
        single_leader = None
        for row in lb.as_data_frame().itertuples():
            if "StackedEnsemble" not in row.model_id:
                single_leader = row.model_id
                break

        if single_leader:
            print(f"Highest-ranking non-stacked model: {single_leader}")
            single_leader_model = h2o.get_model(single_leader)
        else:
            print("No single model found in the leaderboard.")

        self.model = single_leader_model
        h2o.save_model(self.model, path='models/credit_risk_model')

        # Upload model to GCP:
        gcs = GCSHandler()
        gcs.handle_file("upload", "models/credit_risk_model", "models/credit_risk_model")

