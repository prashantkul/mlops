from download.api import DVCFileSystem


from dvc.api import open as dvc_open
from dvc.repo import Repo

class DVCReader:

    def __init__(self, data_path):
        self.dvc_repo = data_path
        self.url = "https://github.com/prashantkul/mlops.git"
        self.local_fs = DVCFileSystem(self.dvc_repo)
        self.remote_fs = DVCFileSystem(self.url)
        self.repo = Repo(self.dvc_repo)

    def read_file(self, file_path, revision=None):
        """
        Reads a file from the DVC repo.

        :param file_path: Path to the file in the DVC repo.
        :param revision: Git/DVC revision (e.g., commit hash, branch name, or tag).
        :return: Content of the file as a string.
        """
        try:
            with dvc_open(file_path, repo=self.dvc_repo, rev=revision) as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def get_diffs(self):
        """
        Get the differences between the local and remote DVC repositories.

        :return: List of changes.
        """
        try:
            diff = self.repo.diff()
            return diff
        except Exception as e:
            print(f"Error getting diffs: {e}")
            return None

    def download_file(self, file_path):
        """
        Downloads a file from the DVC remote repository to the local file system.

        :param file_path: Path to the file in the DVC repo.
        :return: None
        """
        try:
            self.repo.pull(file_path)
            print(f"File '{file_path}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading file: {e}")
