#!/bin/bash

# Define variables
DVC_REPO_PATH=""  # Set the path to your DVC repository
DVC_REMOTE_URL="https://github.com/prashantkul/mlops.git"
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

# Ensure the DVC remote is set
if ! dvc remote list | grep -q "origin"; then
    echo "Setting up DVC remote 'origin'..."
    dvc remote add -d origin "$DVC_REMOTE_URL"
    echo "DVC remote 'origin' configured."
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
