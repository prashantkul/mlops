from typing import Dict, Optional, Sequence
from dotenv import load_dotenv
import os
from google.auth import credentials
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.cloud.aiplatform import explain
from google.cloud.aiplatform.explain import ExplanationMetadata, ExplanationParameters
import pprint

class ModelDeployment:
    """ Import model into Vertex Model Registry and deploy to an endpoint """

    def __init__(self, project, location, environment):
        """ Initialize the Vertex AI Platform client """
        
        if environment == "local":
            load_dotenv()
            key_path = os.getenv("GOOGLE_SA_KEY_FILE")
            if not key_path:
                raise ValueError("GOOGLE_SA_KEY_FILE environment variable is not set.")
            credentials = self.create_credentials_from_service_account(key_path=key_path )
            aiplatform.init(project=project, location=location, credentials=credentials)
            print("Initialized Vertex AI Platform client.")
        else:
            aiplatform.init(project=project, location=location)
    
    
    def create_credentials_from_service_account(self, key_path: str):
        """
        Creates google.auth.credentials.Credentials from a service account key file.

        Args:
            key_path (str): Path to the service account JSON key file.
            
        Returns:
            google.auth.credentials.Credentials: The credentials object.
        """
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Service account key file not found: {key_path}")
        
        try:
            credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
            print("Successfully created credentials.")
            return credentials
        except Exception as e:
            print(f"Error creating credentials: {e}")
            return None
    
   
    def deploy_model(self, model_name):
        """ Deploy the model to an endpoint """
        # # Import the model into Vertex Model Registry
        #model = self.upload_model_sample(model_name)
        
        model = aiplatform.Model(model_name='projects/271854447431/locations/us-central1/models/4725565736251031552')
        print("Query endpoints")
        # get an existing endpoint
        endpoints = aiplatform.Endpoint.list()
        print("Endpoints: ", len(endpoints))
        for endpoint in endpoints:
            print(endpoint.display_name)
            print(endpoint.name)
            print(endpoint.resource_name)
            print(endpoint.traffic_split)
            print(endpoint.list_models())
            print(endpoint.network)
            print(endpoint.create_time)
        
        
        # # Deploy the model to an endpoint
        print(f"Deploying model {model.resource_name} to an existing endpoint")
        model.deploy(
        endpoint=endpoints[0],
        deployed_model_display_name=model.resource_name,
        traffic_percentage=100,
        sync=True,
        )

        model.wait()
        print("Model is deployed to the endpoint.")
        print(model.display_name)
        print(model.resource_name)
        return model, endpoints[0]
    
    def upload_model_sample(self, 
        display_name: str,
        sync: bool = True,
    ):
        """ Upload a model to Vertex Model Registry """
        
        # Get an existing model version from the Model Registry
        model = aiplatform.Model.list(filter=f"display_name={display_name}")
        # model display name
        
        if len(model) > 1:
            raise ValueError(f"Multiple models with the same display name {display_name} found in the Model Registry.")
          
        if not model:
            raise ValueError(f"Model {display_name} doesn't exists in the Model Registry.")
        else:
            version = model[0].version_id
            print(f"Model {model[0].display_name} version {version} found in the Model Registry. Will upload a new version.")
            new_model_name = model[0].display_name

        # Define the model parameters
        serving_container_image_uri = "us-docker.pkg.dev/cloud-aiplatform/prediction/sklearn-cpu.1-3:latest"
        artifact_uri: Optional[str] = "gs://mlops-final-project-232/h2o_output"
        serving_container_predict_route: Optional[str] = None
        serving_container_health_route: Optional[str] = None
        description: Optional[str] = "Credit card prediction model"
        instance_schema_uri: Optional[str] = "gs://mlops-final-project-232/schemata/schema.yaml"
        parameters_schema_uri: Optional[str] = "gs://mlops-final-project-232/schemata/parameters.yaml"
        prediction_schema_uri: Optional[str] = "gs://mlops-final-project-232/schemata/predictions.yaml"
        explanation_metadata: Optional[ExplanationMetadata] = None
        explanation_parameters: Optional[ExplanationParameters] = None
        
             
        model = aiplatform.Model.upload(
            display_name= display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            instance_schema_uri=instance_schema_uri,
            parameters_schema_uri=parameters_schema_uri,
            prediction_schema_uri=prediction_schema_uri,
            description=description,
            explanation_metadata=explanation_metadata,
            explanation_parameters=explanation_parameters,
            parent_model=model[0].resource_name,
            sync=sync,
        )

        model.wait()

        print(model.display_name)
        print(model.resource_name)
        return model
    
# # define usage of the class
# environment="local"
# project = "pk-arg-prj4-datasci"
# location = "us-central1"
# model_name = "credit_card_prediction_model"
# model = ModelDeployment(project, location, environment)
# #model.upload_model_sample(display_name=model_name)
# model.deploy_model(model_name)

        
