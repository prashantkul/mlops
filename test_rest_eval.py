import requests
import json
import subprocess
import pandas as pd
import os

import google.auth.transport.requests


from config_manager import ConfigManager
from model_deployer import ModelDeployment

class RestEvaluator:
    def __init__(self, project_id: str, location_id: str, endpoint_id: str):
        """Initializes the Evaluator with project, location, and endpoint information."""
        self.project_id = project_id
        self.location_id = location_id
        self.endpoint_id = endpoint_id
        self.config_manager = ConfigManager()
        

    def get_access_token(self) -> str:
        """Generates an access token using the gcloud command."""
        try:
            # Get credentials
            deployer = ModelDeployment(
                project=self.config_manager.get_config("gcs_client_id"), 
                location=self.config_manager.get_config("location"), 
                environment="local"
            )
            # Common Google Cloud API scopes
            SCOPES = [
                'https://www.googleapis.com/auth/cloud-platform'
            ]
            
            # Load credentials from service account key
            credentials = deployer.create_credentials_from_service_account(key_path=os.getenv("GOOGLE_SA_KEY_FILE"))
            
            if credentials:
                # Get the access token
                request = google.auth.transport.requests.Request()
                credentials.refresh(request)
                access_token = credentials.token
                print("✅ Access token generated successfully.")
            
            return access_token
        except Exception as e:
            print(f"❌ Error generating access token: {e}")
            return None

    def predict(self, instances: list) -> dict:
        """Sends the list of lists to Vertex AI for prediction."""
        try:
            print(f"📡 Starting prediction for endpoint: {self.endpoint_id} in project: {self.project_id}")
            
            url = f"https://{self.location_id}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location_id}/endpoints/{self.endpoint_id}:predict"
            
            payload = {
                "instances": instances  # List of lists, not key-value pairs
            }

            # Get Bearer Token
            token = self.get_access_token()
            if not token:
                raise ValueError("Access token not found. Make sure you are authenticated using gcloud.")
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8"
            }

            print(f"📤 Sending the following instances: {instances}")
            
            response = requests.post(url, headers=headers, json=payload)
            
            print(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                print(f"❌ Prediction failed: {response.json()}")

            return response.json()
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {"error": str(e)}

    def create_instances_from_dataframe(self, df: pd.DataFrame) -> list:
        """Converts a DataFrame into a list of lists for Vertex AI REST API."""
        try:
            print("🔄 Creating instances from DataFrame...")
            
            # Convert DataFrame to list of lists
            instances = df.values.tolist()
            
            print(f"✅ Created {len(instances)} instances from DataFrame.")
            return instances
        
        except Exception as e:
            print(f"❌ Error creating instances from DataFrame: {e}")
            return []

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Wrapper method to create instances from DataFrame and send them to Vertex AI for predictions."""
        try:
            print("🚀 Starting evaluation process...")
            
            instances = self.create_instances_from_dataframe(df)
            if not instances:
                raise ValueError("No instances created from DataFrame.")
            
            print(f"📤 Instances created for prediction: {instances}")
            
            response = self.predict(instances)
            
            print(f"📊 Prediction response: {response}")
            
            return response
        
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {"error": str(e)}


# Usage Example
if __name__ == "__main__":
    # Sample DataFrame (Only Values, No Key-Value Pairs)
    df = pd.DataFrame([
        [0, 0, True, 1, True, False, False, True, True, True, False, False, True, True, False, False, True, True, False, False, False, True, True, False, False, True, False, False]
    ])

    # Replace these with your actual GCP details
    PROJECT_ID = "271854447431"
    LOCATION_ID = "us-central1"
    ENDPOINT_ID = "8280511129222905856"

    evaluator = RestEvaluator(project_id=PROJECT_ID, location_id=LOCATION_ID, endpoint_id=ENDPOINT_ID)
    
    # Call evaluate method
    response = evaluator.evaluate(df)
    print("✅ Final Response:", response)