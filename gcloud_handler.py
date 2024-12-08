import os
from google.cloud import storage

class GCSHandler:
    def __init__(self, bucket_name = "mlops-final-project-232"):
        self.bucket_name = bucket_name
        # Initialize the GCS client and bucket
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
 
    def handle_file(self, operation, local_path, gcs_path):
        """
        Handles uploading or downloading a file to/from GCS.
 
        Args:
            operation (str): "upload" or "download".
            local_path (str): Local file path for uploading or saving during download.
            gcs_path (str): Path in GCS bucket.
 
        Returns:
            str: Path of the uploaded/downloaded file or public URL (for upload).
        """
        try:
            if operation == "upload":
                # Upload file to GCS
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                print(f"File {local_path} uploaded to {gcs_path} in bucket {self.bucket_name}.")
                
                # Optionally make public
                blob.make_public()
                print(f"Public URL: {blob.public_url}")
                return blob.public_url
 
            elif operation == "download":
                # Download file from GCS
                blob = self.bucket.blob(gcs_path)
                if not blob.exists():
                    raise FileNotFoundError(f"File {gcs_path} does not exist in bucket {self.bucket_name}.")
                
                blob.download_to_filename(local_path)
                print(f"File {gcs_path} downloaded to {local_path}.")
                return local_path
 
            else:
                raise ValueError("Invalid operation. Use 'upload' or 'download'.")
        except Exception as e:
            print(f"Error during {operation}: {e}")
            return None