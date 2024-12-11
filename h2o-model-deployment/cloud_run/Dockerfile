# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Java and required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p model lib

# Copy model directory and its contents
COPY model ./model/

# Copy Java library
COPY lib/h2o-genmodel.jar ./lib/

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]
