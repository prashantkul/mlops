import os
import zipfile
import shutil
import subprocess
import logging
import time
import json

class CloudRunDeployer:
    def __init__(self, model_zip_path: str, base_path: str = '.'):
        """
        Initialize the CloudRunDeployer.

        Args:
            model_zip_path (str): Path to the H2O model .zip file.
            base_path (str): Base directory where cloud_run folder is located.
        """
        self.model_zip_path = model_zip_path
        self.base_path = base_path
        self.cloud_run_path = os.path.join(base_path, 'cloud_run')
        self.lib_path = os.path.join(self.cloud_run_path, 'lib')
        self.model_path = os.path.join(self.cloud_run_path, 'model')
        self.cloudbuild_yaml_path = os.path.join(base_path, 'cloudbuild.yaml')

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create required directories if they do not exist."""
        os.makedirs(self.lib_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def _unzip_model(self):
        """Unzip the H2O model and copy files to the appropriate directories."""
        try:
            self.logger.info(f"Unzipping model from {self.model_zip_path}...")

            with zipfile.ZipFile(self.model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.model_path)

            self.logger.info("Model unzipped successfully.")

            # Move all .jar files to lib/ and other files to model/
            for root, _, files in os.walk(self.model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.jar'):
                        shutil.move(file_path, os.path.join(self.lib_path, file))
                        self.logger.info(f"Moved {file} to lib directory.")
                    elif file_path != self.model_path:  # Don't move if it's already in model/
                        shutil.move(file_path, self.model_path)
                        self.logger.info(f"Moved {file} to model directory.")
                        
        except Exception as e:
            self.logger.error(f"Failed to unzip the model: {e}")
            raise

    def _copy_cloudbuild_yaml(self):
        """Check if the cloudbuild.yaml file exists in the base directory."""
        if not os.path.isfile(self.cloudbuild_yaml_path):
            raise FileNotFoundError(f"{self.cloudbuild_yaml_path} not found in the base directory.")
        self.logger.info(f"cloudbuild.yaml is present at {self.cloudbuild_yaml_path}.")

    def _trigger_cloud_run_build(self):
        """Trigger the GCP Cloud Run build using gcloud CLI."""
        try:
            project_id = subprocess.check_output(
                ['gcloud', 'config', 'get-value', 'project'], 
                encoding='utf-8'
            ).strip()

            self.logger.info(f"Starting Cloud Build for project: {project_id}")

            build_command = [
                'gcloud', 'builds', 'submit',
                '--config', self.cloudbuild_yaml_path,
                '--format', 'json'
            ]

            result = subprocess.check_output(build_command, encoding='utf-8')
            build_info = json.loads(result)
            build_id = build_info.get('id')

            self.logger.info(f"Cloud Run build successfully triggered. Build ID: {build_id}")
            return build_id
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Cloud Build failed: {e}")
            raise

    def _poll_build_status(self, build_id: str, poll_interval: int = 10):
        """
        Poll the status of the Cloud Build.

        Args:
            build_id (str): The build ID of the Cloud Build.
            poll_interval (int): The interval (in seconds) to check the build status.
        """
        try:
            self.logger.info(f"Polling build status for Build ID: {build_id}...")
            
            while True:
                command = [
                    'gcloud', 'builds', 'describe', build_id,
                    '--format', 'json'
                ]

                result = subprocess.check_output(command, encoding='utf-8')
                build_info = json.loads(result)
                status = build_info.get('status')

                self.logger.info(f"Current status: {status}")

                if status in ['SUCCESS', 'FAILURE', 'CANCELLED']:
                    self.logger.info(f"Build completed with status: {status}")
                    break

                time.sleep(poll_interval)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to poll build status: {e}")
            raise

    def deploy(self):
        """Run the full deployment process."""
        try:
            self._create_directories()
            self._unzip_model()
            self._copy_cloudbuild_yaml()
            build_id = self._trigger_cloud_run_build()
            self._poll_build_status(build_id)
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise

if __name__ == '__main__':
    # Path to the H2O model zip file
    model_zip_name = '*.zip'  # update needed
    model_file_location = "/home/admin/h2o/h2o-3.46.0.6/credit_risk_model/"
    model_zip_path = os.path.join(model_file_location, model_zip_name)
    deployer = CloudRunDeployer(model_zip_path=model_zip_path)
    deployer.deploy()