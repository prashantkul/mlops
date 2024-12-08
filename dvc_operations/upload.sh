#!/bin/bash

# Define variables
DVC_REPO_PATH=""  # Set the path to your DVC repository
GCS_BUCKET="mlops-final-project-232"  # Your GCS bucket name
GCS_REMOTE_PATH="dataset"  # Path in the GCS bucket to store DVC files
GCS_CLIENT_ID="pk-arg-prj4-datasci"
FILE_TO_UPLOAD=$1  # File path to upload (passed as an argument)
COMMIT_MESSAGE="Adding file to DVC repository"

# Check if the file to upload is provided
if [ -z "$FILE_TO_UPLOAD" ]; then
    echo "Usage: $0 <file_to_upload>"
    exit 1
fi

# Ensure the DVC repository path exists
if [ ! -d "$DVC_REPO_PATH" ]; then
    echo "Creating DVC repository path '$DVC_REPO_PATH'..."
    mkdir -p "$DVC_REPO_PATH"
fi

# Navigate to the DVC repository
cd "$DVC_REPO_PATH" || exit 1

# Debug: Show current directory and file path
echo "Current directory: $(pwd)"
echo "File to upload: $FILE_TO_UPLOAD"

# Ensure the file exists
if [ ! -f "$FILE_TO_UPLOAD" ]; then
    echo "Error: File '$FILE_TO_UPLOAD' does not exist."
    exit 1
fi

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    echo "DVC initialized."
fi

# Set up DVC remote for GCS bucket if not already configured
if ! dvc remote list | grep -q "gcs_remote"; then
    echo "Setting up DVC remote for GCS bucket..."
    dvc remote add -d gcs_remote gs://$GCS_BUCKET/$GCS_REMOTE_PATH
    dvc remote modify gcs_remote --local gdrive_client_id $GCS_CLIENT_ID
    dvc remote modify --local gcs_remote gdrive_use_service_account true
    echo "DVC remote 'gcs_remote' configured for GCS bucket."
fi

# Add the file to DVC tracking
echo "Adding '$FILE_TO_UPLOAD' to DVC tracking..."
dvc add "$FILE_TO_UPLOAD"
git add "$FILE_TO_UPLOAD.dvc"
if [ $? -ne 0 ]; then
    echo "Failed to add file to DVC tracking."
    exit 1
fi

# Stage changes for Git commit (add .dvc file and .gitignore changes)
echo "Staging changes for Git commit..."
git add "$FILE_TO_UPLOAD" .gitignore
git add data/.gitignore "$FILE_TO_UPLOAD" dvc
git add config_manager.py data_preprocessor.py dvc/download.py dvc/upload.sh requirements.txt services/preprocess.py

# Commit changes to Git
echo "Committing changes to Git..."
git commit -m "$COMMIT_MESSAGE" || echo "No changes to commit."

# Push the file to DVC remote
echo "Pushing file to DVC remote..."
dvc push
if [ $? -ne 0 ]; then
    echo "Failed to push file to DVC remote."
    exit 1
fi

# Push Git changes to the remote repository
echo "Pushing Git changes to the remote repository..."
git push origin "$(git branch --show-current)"
if [ $? -ne 0 ]; then
    echo "Failed to push Git changes. Please check your Git setup."
    exit 1
fi

echo "File '$FILE_TO_UPLOAD' successfully uploaded to DVC and Git."
