import os

from dvc.api import DVCFileSystem
from dvc.repo import Repo
from dvc.api import open as dvc_open
import pandas as pd


from config_manager import ConfigManager


class DVCReader:


    def __init__(self, data_path='data'):
        """
        Initialize the DVCReader with the local and remote GCS file system.

        :param data_path: Path to the local DVC repository.
        """
        config_manager = ConfigManager()
        self.dvc_repo = data_path
        self.gcs_bucket = config_manager.get_config('gcs_bucket')  # Get GCS bucket name from config
        self.gcs_remote_path = config_manager.get_config('gcs_remote_path')  # GCS remote path
        self.url = f"gs://{self.gcs_bucket}/{self.gcs_remote_path}"  # Full GCS path
        self.remote_fs = DVCFileSystem(self.url)  # Remote GCS file system
        self.repo = Repo(self.dvc_repo)  # DVC Repo object

    def read_dataframe(self, filename, revision=None):
        """
        Reads a file from the DVC repo and returns it as a pandas DataFrame.

        :param file_path: Path to the file in the DVC repo.
        :param revision: Git/DVC revision (e.g., commit hash, branch name, or tag).
        :return: pandas DataFrame containing the file contents.
        """
        try:
            file_path = f"{self.dvc_repo}/{filename}"
            self.repo.pull(f"{self.dvc_repo}/{filename}")

            df = pd.read_csv(file_path)
            print(f"File '{file_path}' successfully read as DataFrame.")

            return df
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            return None

    def read_csv(self, dvc_file_path, local_file_path=None, revision=None):
        """

        Reads a file from the DVC repo and saves it locally as a CSV in the 'data' directory.

        :param dvc_file_path: Path to the file in the DVC repo.
        :param local_file_path: local file path where the file will be saved.
        :param revision: Git/DVC revision (e.g., commit hash, branch name, or tag).
        :return: None
        """
        try:
            df = self.read_dataframe(dvc_file_path, revision)

            df.to_csv(local_file_path, index=False)
            print(f"File '{local_file_path}' successfully saved as CSV to '{os.path.basename(local_file_path or dvc_file_path)}'.")

        except Exception as e:
            print(f"Error processing file '{dvc_file_path}': {e}")