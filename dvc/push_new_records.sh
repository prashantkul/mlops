#!/bin/bash

# Define variables
DVC_REPO_PATH="data"  # Set the path to your DVC repository
DVC_REMOTE_URL="https://github.com/prashantkul/mlops.git"
COMMIT_MESSAGE="Adding new DVC records to Git"

# Ensure the DVC repository path exists
if [ ! -d "$DVC_REPO_PATH" ]; then
    echo "Error: DVC repository path '$DVC_REPO_PATH' does not exist."
    exit 1
fi

# Navigate to the DVC repository
cd "$DVC_REPO_PATH" || exit 1

# Debug: Show current directory
echo "Current directory: $(pwd)"

# Find all `.dvc` files that are not staged in Git
UNSTAGED_DVC_FILES=$(git ls-files --others --exclude-standard | grep ".dvc")

if [ -z "$UNSTAGED_DVC_FILES" ]; then
    echo "No new DVC records to add to version control."
    exit 0
fi

# Stage the unstaged DVC files
echo "Staging new DVC files..."
git add $UNSTAGED_DVC_FILES
if [ $? -ne 0 ]; then
    echo "Failed to stage DVC files."
    exit 1
fi

# Commit changes to Git
echo "Committing new DVC files to Git..."
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo "Failed to commit new DVC records."
    exit 1
fi

# Push the changes to the Git remote repository
echo "Pushing changes to the Git remote repository..."
git push origin "$(git branch --show-current)"
if [ $? -ne 0 ]; then
    echo "Failed to push Git changes. Please check your Git setup."
    exit 1
fi

# Push the files to the DVC remote
echo "Pushing new DVC records to the DVC remote..."
dvc push
if [ $? -ne 0 ]; then
    echo "Failed to push DVC records to the remote."
    exit 1
fi

echo "New DVC records successfully pushed to both Git and DVC."
