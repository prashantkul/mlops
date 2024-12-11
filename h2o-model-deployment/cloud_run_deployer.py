import os
import zipfile
import shutil
import subprocess
import time
import json
import glob

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
        print(f"Base path: {self.base_path}")
        self.cloud_run_path = os.path.join(base_path, 'cloud_run')
        self.lib_path = os.path.join(self.cloud_run_path, 'lib')
        self.model_path = os.path.join(self.cloud_run_path, 'model')
        self.cloudbuild_yaml_path = os.path.join(self.cloud_run_path, 'cloudbuild.yaml')

    def _clear_directories(self):
        """Clear existing model directory (optional)"""
        if os.path.exists(self.model_path):
            print(f"Clearing existing model directory: {self.model_path}")
            shutil.rmtree(self.model_path)  # Remove the existing model folder
    
    def _unzip_model(self):
        """Unzip the H2O model and preserve paths"""
        try:
            print(f"Unzipping model from {self.model_zip_path} into {self.model_path}...")

            # Extract the zip into the model path
            with zipfile.ZipFile(self.model_zip_path, 'r') as zip_ref:
                print(f"Files in the ZIP archive: {zip_ref.namelist()}")
                zip_ref.extractall(self.model_path)  # Extract while preserving paths

            print(f"Model successfully extracted to {self.model_path}.")

            # Ensure d0* files are in the domains directory
            domains_path = os.path.join(self.model_path, 'domains')
            os.makedirs(domains_path, exist_ok=True)  # Create domains folder if it doesn't exist

            for file in os.listdir(self.model_path):
                if file.startswith('d0') and file.endswith('.txt'):
                    source_path = os.path.join(self.model_path, file)
                    dest_path = os.path.join(domains_path, file)
                    print(f"Moving {file} to {domains_path}")
                    shutil.move(source_path, dest_path)

        except Exception as e:
            print(f"Failed to unzip the model: {e}")
            raise

    def _copy_cloudbuild_yaml(self):
        """Check if the cloudbuild.yaml file exists in the base directory."""
        if not os.path.isfile(self.cloudbuild_yaml_path):
            raise FileNotFoundError(f"{self.cloudbuild_yaml_path} not found in the base directory.")
        print(f"cloudbuild.yaml is present at {self.cloudbuild_yaml_path}.")

    def _trigger_cloud_run_build(self):
        """Trigger the GCP Cloud Run build using gcloud CLI."""
        try:
            project_id = subprocess.check_output(
                ['gcloud', 'config', 'get-value', 'project'], 
                encoding='utf-8'
            ).strip()

            print(f"Starting Cloud Build for project: {project_id}")

            build_command = [
                'gcloud', 'builds', 'submit',
                '--config', self.cloudbuild_yaml_path,
                '--format', 'json'
            ]

            result = subprocess.check_output(build_command, encoding='utf-8')
            build_info = json.loads(result)
            build_id = build_info.get('id')

            print(f"Cloud Run build successfully triggered. Build ID: {build_id}")
            return build_id
        except subprocess.CalledProcessError as e:
            print(f"Cloud Build failed: {e}")
            raise

    def _poll_build_status(self, build_id: str, poll_interval: int = 10):
        """
        Poll the status of the Cloud Build.

        Args:
            build_id (str): The build ID of the Cloud Build.
            poll_interval (int): The interval (in seconds) to check the build status.
        """
        try:
            print(f"Polling build status for Build ID: {build_id}...")
            
            while True:
                command = [
                    'gcloud', 'builds', 'describe', build_id,
                    '--format', 'json'
                ]

                result = subprocess.check_output(command, encoding='utf-8')
                build_info = json.loads(result)
                status = build_info.get('status')

                print(f"Current status: {status}")

                if status in ['SUCCESS', 'FAILURE', 'CANCELLED']:
                    print(f"Build completed with status: {status}")
                    break

                time.sleep(poll_interval)

        except subprocess.CalledProcessError as e:
            print(f"Failed to poll build status: {e}")
            raise

    def deploy(self):
        """Run the full deployment process."""
        try:
            self._clear_directories()
            self._unzip_model()
            self._copy_cloudbuild_yaml()
            build_id = self._trigger_cloud_run_build()
            self._poll_build_status(build_id)
        except Exception as e:
            print(f"Deployment failed: {e}")
            raise

if __name__ == '__main__':
    # Path to the directory containing the H2O model zip files
    model_file_location = "/home/admin/h2o/h2o-3.46.0.6/credit_risk_model/"
    
    # Find all files matching the pattern 'X*.zip'
    zip_files = glob.glob(os.path.join(model_file_location, 'X*.zip'))
    
    if not zip_files:
        raise FileNotFoundError(f"No files matching 'X*.zip' found in {model_file_location}")
    
    # Get the latest file by modification time
    latest_file = max(zip_files, key=os.path.getmtime)
    print(f"Latest file found: {latest_file}")
    
    # Pass the full path of the latest file
    model_zip_path = latest_file
    print(f"Model zip path: {model_zip_path}")
    
    # Deploy the model
    deployer = CloudRunDeployer(model_zip_path=model_zip_path)
    deployer.deploy()