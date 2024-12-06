from dvc.api import DVCFileSystem


class DVCReader:

    def __init__(self, data_path):
        self.dvc_repo = "/data"
        self.url = "https://github.com/prashantkul/mlops.git"

        self.local_fs = DVCFileSystem(self.dvc_repo)
        self.remote_fs = DVCFileSystem(self.url)
