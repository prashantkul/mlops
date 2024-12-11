# mlops
Final Project for MSADS 32021

## Objective 
Build, train, deploy, and monitor a model to predict credit card approval

## Dataset
credit card dataset can be found on [Kaggle]([url](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction))

## Architecture
<img width="722" alt="Screen Shot 2024-12-11 at 15 24 07" src="https://github.com/user-attachments/assets/97670efe-6096-4552-8c4f-992aba645384" />

## Data Versioning
Data Versioning is done with DVC, the data directory is at data/, actual files are stored in a google cloud services bucket. 

### uploading
```bash
./dvc_operations/upload.sh <file to upload>

```

### downloading

```python
from dvc_operations.download import DVCReader

dvc = DVCReader()
dvc.read_dataframe('filepath_in_bucket') # read to a pandas dataframe
dvc.read_csv('filepath_in_bucket') # read to a csv file

```

## Feature Store
upload and download files to/from redis server hosted on google cloud

### uploading
```python

  from feature_store import FeatureStore

  fs = FeatureStore('filepath')
  fs.upload()
```

### downloading
```python

  from feature_store import FeatureStore
  fs = FeatureStore()
  fs.get_features(key=feature_store_key, as_dataframe=True, as_csv=False)
```
## Model Training Orchestration

kicking of model training in apache airflow

### pipeline
<img width="771" alt="Screen Shot 2024-12-10 at 20 28 40" src="https://github.com/user-attachments/assets/5ec364d6-7045-4352-9b13-d5f539ddb0f4" />

```bash
  gcloud composer environments run airflow \
    --location <location> \
    dags trigger -- credit_card_prediction_dag
```

## Model Deployment
```python
  from model_deployer import ModelDeployment

  model = ModelDeployment(project, location, environment)
  model.upload_model_sample(display_name=model_name)
  model.deploy_model(model_name)
```
## AutoML
We performed model monitoring with H20

```
  TK
```

## Model Monitoring
Arize is our model monitoring platform

```
  TK
```
