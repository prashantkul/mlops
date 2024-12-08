import os
import yaml
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.auth.credentials import AnonymousCredentials

from model_deployer import ModelDeployment
from config_manager import ConfigManager

from dotenv import load_dotenv

class Evaluator:
    DEFAULT_LOCATION = "us-central1"
    DEFAULT_API_ENDPOINT = "us-central1-aiplatform.googleapis.com"
    
    def __init__(self):
        """Initializer for Evaluator class."""
        self.config_manager = ConfigManager()
        self.storage_client = storage.Client.create_anonymous_client()
        self.bucket = self.storage_client.bucket(bucket_name=self.config_manager.get_config("gcs_bucket"))
        self.project = self.config_manager.get_config("gcs_client_id")
        
    def predict_custom_trained_model_sample(
        self,
        project: str,
        endpoint_id: str,
        instances: List[List[Union[int, float, bool]]],  # List of lists
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    ) -> Dict:
        """Sends a prediction request to a custom-trained model deployed on Vertex AI."""
        try:
            print(f"📡 Starting prediction for endpoint: {endpoint_id} in project: {project}")
            
            client_options = {"api_endpoint": api_endpoint}
            
            # Get credentials
            deployer = ModelDeployment(
                project=self.config_manager.get_config("gcs_client_id"), 
                location=self.config_manager.get_config("location"), 
                environment="local"
            )
            credentials = deployer.create_credentials_from_service_account(key_path=os.getenv("GOOGLE_SA_KEY_FILE"))
            prediction_client = PredictionServiceClient(client_options=client_options, credentials=credentials)
            
            endpoint = prediction_client.endpoint_path(project, location, endpoint_id)
            
            print(f"📤 Sending the following instances directly (list of lists): {instances}")
            
            # Pass the list of lists directly to the SDK (DO NOT wrap in "instances")
            response = prediction_client.predict(
                endpoint=endpoint, 
                instances=instances,  # Send raw list, not { "instances": [...] }
                parameters=None  # No protobuf logic required
            )

            predictions = [json_format.MessageToDict(pred) for pred in response.predictions]
            
            return {
                "deployed_model_id": response.deployed_model_id,
                "predictions": predictions
            }
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}

    def create_instances_from_dataframe(
            self, 
            df: pd.DataFrame
        ) -> List[List[Union[int, float, bool]]]:
            """Converts a DataFrame into a list of instances for prediction."""
            successful_case = {
                            'Gender': 0,                    
                            'Reality': 0,                   
                            'ChldNo_1': True,              
                            'wkphone': 1,                  
                            'gp_Age_high': True,           
                            'gp_Age_highest': False,
                            'gp_Age_low': False,
                            'gp_Age_lowest': True,         
                            'gp_worktm_high': True,        
                            'gp_worktm_highest': True,     
                            'gp_worktm_low': False,
                            'gp_worktm_medium': False,
                            'occyp_hightecwk': True,       
                            'occyp_officewk': True,        
                            'famsizegp_1': False,
                            'famsizegp_3more': False,
                            'houtp_Co-op apartment': True,  
                            'houtp_Municipal apartment': True,  
                            'houtp_Office apartment': False,
                            'houtp_Rented apartment': False,
                            'houtp_With parents': False,
                            'edutp_Higher education': True, 
                            'edutp_Incomplete higher': True, 
                            'edutp_Lower secondary': False,
                            'famtp_Civil marriage': False,
                            'famtp_Separated': True,       
                            'famtp_Single_not_married': False,
                            'famtp_Widow': False
                            }
            
            print("Creating instances from DataFrame...")
            try:
                # Step 1: Ensure required features are from successful_case
                required_columns = list(successful_case.keys())
                
                # Step 2: Add missing columns with values from successful_case
                for col, default_value in successful_case.items():
                    if col not in df.columns:
                        print(f"⚠️ Missing column: {col} - Adding default value: {default_value}")
                        df[col] = default_value
                
                # Step 3: Ensure the order of the columns matches the successful_case order
                df_filtered = df[required_columns]
                
                # Step 4: Convert each row into a list of values (not a dictionary)
                instances = df_filtered.values.tolist()
                
                print(f"✅ Created {len(instances)} instances from DataFrame.")
                return instances
                
            except Exception as e:
                print(f"Error creating instances from DataFrame: {e}")
                return []

    def evaluate(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Dict:
        """
        **Wrapper Method** that performs the entire evaluation process:
        - Reads the schema from GCS.
        - Converts the DataFrame into instances using the schema.
        - Sends the prediction request using the instances.
        - Calculates evaluation metrics.
        """
        try:
            print("Starting evaluation process...")
            
            y_true = df[target_column].tolist()
            instances = self.create_instances_from_dataframe(df)
            if not instances:
                raise ValueError("No instances created from DataFrame.")
            
            print(f"Instances created: {instances}")
            
            response = self.predict(instances)
            predictions = [pred['predicted_class'] for pred in response.get("predictions", [])]
            
            df['prediction'] = predictions
            
            metrics = self.calculate_evaluation_metrics(y_true, predictions)
            
            return {
                "predictions": predictions,
                "metrics": metrics
            }
        
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"error": str(e)}

    def read_schema_from_gcs(self, csv_columns) -> List[str]:
        """Reads the schema.yaml from a GCS path and extracts the required columns."""
        gcs_path = self.config_manager.get_config("schema_gcs_path")
        try:
            if not gcs_path.startswith("gs://"):
                raise ValueError(f"Invalid GCS URL: {gcs_path}")
            
            bucket_name, object_path = gcs_path.replace("gs://", "").split("/", 1)
            blob = self.storage_client.bucket(bucket_name).blob(object_path)
            schema_content = blob.download_as_text()
            schema = yaml.safe_load(schema_content)
            
            all_columns = list(schema['components']['schemas']['instance']['properties'].keys())
            required_columns = schema['components']['schemas']['instance'].get('required', [])
            
            excluded_columns = {'target', 'classes'}
            available_columns = [col for col in all_columns if col not in excluded_columns]

            valid_csv_columns = [col for col in csv_columns if col in available_columns]
            
            # Log missing required columns but do not fail
            missing_required_columns = [col for col in required_columns if col not in csv_columns]
            if missing_required_columns:
                print(f"⚠️ Warning: Missing required columns in the CSV: {missing_required_columns}")
            
            print(f"✅ Valid CSV columns: {valid_csv_columns}")
            
            return valid_csv_columns
            
        except Exception as e:
            print(f"Schema read error from GCS: {e}")
            return []

    def calculate_evaluation_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Calculates classification metrics: Accuracy, Precision, Recall, F1 Score."""
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=1),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=1),
                "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=1)
            }
            print(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            return {"error": str(e)}
    
    def predict(
    self,
    instances: Union[Dict, List[Dict]]
    ) -> Dict:
        """Wrapper function to predict using values from the `ConfigManager`."""
        try:
            vertex_config = self.config_manager.get_config("vertex_api_endpoint_details", {})
            
            project = vertex_config.get("project")
            endpoint_id = vertex_config.get("endpoint_id")
            location = vertex_config.get("location", self.DEFAULT_LOCATION)

            if not project or not endpoint_id:
                raise ValueError("Missing 'project' or 'endpoint_id' in 'vertex_api_endpoint_details'.")

            return self.predict_custom_trained_model_sample(
                project=project, 
                endpoint_id=endpoint_id, 
                instances=instances, 
                location=location
            )
        
        except Exception as e:
            print(f"Prediction error from config: {e}")
            return {"error": str(e)}

import pandas as pd

# Path to the CSV file
csv_path = 'successful_case.csv'

# Load the CSV
df = pd.read_csv(csv_path)
evaluator = Evaluator()

# Create instances from the DataFrame
instances = evaluator.create_instances_from_dataframe(df)

# Call the predict method with the instance data
response = evaluator.predict(instances)

# Print the prediction response
print(response)