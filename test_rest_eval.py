
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import sys

from arize.api import Client
from arize.utils.types import ModelTypes, Environments

EXPECTED_COLUMNS =  [
    'ChldNo_1',
    'gp_Age_high',
    'gp_Age_highest',
    'gp_Age_low',
    'gp_Age_lowest',
    'gp_worktm_high',
    'gp_worktm_highest',
    'gp_worktm_low',
    'gp_worktm_medium',
    'occyp_hightecwk',
    'occyp_officewk',
    'famsizegp_1',
    'famsizegp_3more',
    'houtp_Co_op_apartment',
    'houtp_Municipal_apartment',
    'houtp_Office_apartment',
    'houtp_Rented_apartment',
    'houtp_With_parents',
    'edutp_Higher_education',
    'edutp_Incomplete_higher',
    'edutp_Lower_secondary',
    'famtp_Civil_marriage',
    'famtp_Separated',
    'famtp_Single_not_married',
    'famtp_Widow',
    'Gender',
    'inc',
    'Reality',
    'wkphone'
]

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class PredictionClient:
    def __init__(self, endpoint_url, output_dir="results", arize_api_key=None, arize_space_key=None):
        self.endpoint_url = endpoint_url
        self.output_dir = output_dir
        self.results = []
        self.stats = {
            'total_rows_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'successful_arize_logs': 0,
            'failed_arize_logs': 0,
            'rows_with_target': 0,
            'rows_without_target': 0
        }
        self.failed_rows = []
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Arize client
        self.arize_client = None
        arize_api_key="1885b2e264de6bfa175"
        arize_space_id="U3BhY2U6MTMxNTY6aXIrYg=="
        if arize_api_key and arize_space_id:
            self.arize_client = Client(
                api_key=arize_api_key,
                space_id=arize_space_id
            )
            print("Arize AI client initialized")

    def validate_csv(self, df):
        """Validate CSV structure with error tracking"""
        try:
            missing_cols = [col for col in EXPECTED_COLUMNS[:-1] if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {', '.join(missing_cols)}"
                self.failed_rows.append({
                    'error_type': 'csv_validation',
                    'error_message': error_msg,
                    'missing_columns': missing_cols
                })
                print("\nERROR: " + error_msg)
                print("\nFound columns:")
                print("\n".join(df.columns))
                raise ValueError(f"Missing {len(missing_cols)} required columns")
            return True
        except Exception as e:
            self.failed_rows.append({
                'error_type': 'csv_validation',
                'error_message': str(e)
            })
            raise

    def validate_prediction_response(self, response, row_index):
        """Validate prediction response structure"""
        try:
            if 'predictions' not in response:
                raise ValueError("Missing 'predictions' key in response")
            
            pred = response['predictions']
            if 'predicted_class' not in pred:
                raise ValueError("Missing 'predicted_class' in predictions")
            if 'probabilities' not in pred:
                raise ValueError("Missing 'probabilities' in predictions")
            if '1' not in pred['probabilities']:
                raise ValueError("Missing probability for class '1'")
            
            # Validate data types
            if not isinstance(pred['predicted_class'], (int, str)):
                raise ValueError(f"Invalid predicted_class type: {type(pred['predicted_class'])}")
            if not isinstance(pred['probabilities']['1'], (float, int)):
                raise ValueError(f"Invalid probability type: {type(pred['probabilities']['1'])}")
            
            return True
        except ValueError as e:
            self.failed_rows.append({
                'row_index': row_index,
                'error_type': 'response_validation',
                'error_message': str(e),
                'response': response
            })
            return False

    def prepare_request_data(self, row, row_index):
        """Convert a pandas row to request format with error tracking"""
        request_dict = {}
        
        try:
            for col in EXPECTED_COLUMNS[:-1]:  # Exclude target
                value = row[col]
                if pd.isna(value):
                    request_dict[col] = None
                elif col in ["Gender", "inc", "Reality", "wkphone"]:
                    try:
                        request_dict[col] = float(value)
                    except ValueError:
                        error_msg = f"Invalid numeric value in column {col}: '{value}'"
                        self.failed_rows.append({
                            'row_index': row_index,
                            'error_type': 'data_conversion',
                            'error_message': error_msg,
                            'column': col,
                            'value': value
                        })
                        raise ValueError(error_msg)
                else:
                    # Handle boolean conversion properly
                    if isinstance(value, bool) or isinstance(value, np.bool_):
                        request_dict[col] = bool(value)
                    elif isinstance(value, (int, float)):
                        request_dict[col] = bool(value)
                    elif isinstance(value, str):
                        request_dict[col] = value.lower() == 'true'
                    else:
                        request_dict[col] = bool(value)
            
            return {"instances": [request_dict]}
        except Exception as e:
            self.failed_rows.append({
                'row_index': row_index,
                'error_type': 'request_preparation',
                'error_message': str(e),
                'row_data': row.to_dict()
            })
            raise

    def make_prediction(self, request_data, row_index):
        """Make a single prediction request with error tracking"""
        self.stats['total_rows_processed'] += 1
        try:
            response = requests.post(
                self.endpoint_url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            response_data = response.json()
            
            if self.validate_prediction_response(response_data, row_index):
                self.stats['successful_predictions'] += 1
                return response_data
            else:
                self.stats['failed_predictions'] += 1
                raise PredictionError(f"Invalid response structure for row {row_index}")
                
        except requests.exceptions.RequestException as e:
            self.stats['failed_predictions'] += 1
            self.failed_rows.append({
                'row_index': row_index,
                'error_type': 'http_error',
                'error_message': str(e),
                'request_data': request_data
            })
            raise PredictionError(f"Prediction failed for row {row_index}")

    def log_to_arize(self, row_data, prediction, actual=None, prediction_id=None):
        """Log a single prediction to Arize with error tracking"""
        if not self.arize_client:
            return False
            
        try:
            features = {k: str(v) if not isinstance(v, (int, float, bool)) else v 
                       for k, v in row_data.items() if k != 'target'}

            pred_prob = float(prediction['probabilities']['1'])
            pred_label = int(prediction['predicted_class'])

            response = self.arize_client.log(
                model_id="h2o_mojo_prediction_model",
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                model_version="v1",
                prediction_id=prediction_id or str(int(time.time())),
                features=features,
                prediction_label=(str(pred_label), pred_prob),
                actual_label=actual if actual is not None else None,
                environment=Environments.PRODUCTION,
                prediction_timestamp=int(time.time())
            )
            
            res = response.result()
            if res.status_code == 200:
                self.stats['successful_arize_logs'] += 1
                return True
            else:
                self.stats['failed_arize_logs'] += 1
                self.failed_rows.append({
                    'error_type': 'arize_log_error',
                    'error_message': f'Log failed with response code {res.status_code}',
                    'response': res.text
                })
                raise PredictionError("Arize log failed", res.text)
                #return False
            
                
        except Exception as e:
            self.stats['failed_arize_logs'] += 1
            self.failed_rows.append({
                'error_type': 'arize_log_error',
                'error_message': str(e),
                'row_data': row_data
            })
            raise PredictionError("Arize log failed", e)
            # return False

    def process_file(self, csv_file, delay_ms=0, progress_interval=10):
        """Process CSV file and make predictions with comprehensive error tracking"""
        try:
            # Reset statistics for new processing run
            self.stats = {
                'total_rows_processed': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'successful_arize_logs': 0,
                'failed_arize_logs': 0,
                'rows_with_target': 0,
                'rows_without_target': 0
            }
            self.failed_rows = []
            
            print(f"\nReading file: {csv_file}")
            df = pd.read_csv(csv_file)
            total_rows = len(df)
            print(f"Loaded {total_rows} records")
            
            self.validate_csv(df)
            print("CSV validation successful")
            
            has_target = 'target' in df.columns
            if has_target:
                print("Target column found - will calculate evaluation metrics")
            
            actual_values = []
            predicted_values = []
            predicted_probs = []
            batch_predictions = []
            
            # Process first 10 rows
            preview_size = min(10, total_rows)
            print(f"\nProcessing preview batch of {preview_size} rows...")
            
            for index in range(preview_size):
                try:
                    row_data = df.iloc[index].to_dict()
                    request_data = self.prepare_request_data(df.iloc[index], index)
                    
                    prediction = self.make_prediction(request_data, index)
                    if prediction:
                        batch_predictions.append(prediction['predictions'])
                        
                        if has_target:
                            self.stats['rows_with_target'] += 1
                            actual = int(row_data['target'])
                            actual_values.append(actual)
                            predicted_values.append(int(prediction['predictions']['predicted_class']))
                            predicted_probs.append(float(prediction['predictions']['probabilities']['1']))
                        else:
                            self.stats['rows_without_target'] += 1
                        
                        actual = int(row_data['target']) if has_target else None
                        self.log_to_arize(
                            row_data=row_data,
                            prediction=prediction['predictions'],
                            actual=actual,
                            prediction_id=f"pred_{index}_{int(time.time())}"
                        )
                    
                except Exception as e:
                    print(f"\nError processing preview row {index}: {str(e)}")
                    continue

            # Show preview results and get confirmation
            self.show_prediction_results(0, preview_size-1, df, batch_predictions)
            self.print_statistics()
            
            if total_rows > preview_size:
                if not self.get_user_confirmation():
                    print("\nProcessing stopped by user after preview")
                    return
                
                # Process remaining rows without batch checks
                for index in range(preview_size, total_rows):
                    try:
                        row_data = df.iloc[index].to_dict()
                        request_data = self.prepare_request_data(df.iloc[index], index)
                        
                        prediction = self.make_prediction(request_data, index)
                        if prediction:
                            if has_target:
                                self.stats['rows_with_target'] += 1
                                actual = int(row_data['target'])
                                actual_values.append(actual)
                                predicted_values.append(int(prediction['predictions']['predicted_class']))
                                predicted_probs.append(float(prediction['predictions']['probabilities']['1']))
                            else:
                                self.stats['rows_without_target'] += 1
                            
                            actual = int(row_data['target']) if has_target else None
                            self.log_to_arize(
                                row_data=row_data,
                                prediction=prediction['predictions'],
                                actual=actual,
                                prediction_id=f"pred_{index}_{int(time.time())}"
                            )
                        
                        if (index + 1) % progress_interval == 0:
                            print(f"Processed {index + 1} of {total_rows} records")
                        
                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000)
                            
                    except Exception as e:
                        print(f"\nError processing row {index}: {str(e)}")
                        continue
            
            # Calculate final metrics if we have target values
            if has_target and actual_values:
                self.calculate_and_print_metrics(actual_values, predicted_values, predicted_probs)
            
            # Print final statistics
            print("\nProcessing completed")
            self.print_statistics()
            
        except Exception as e:
            print("\nERROR: Processing stopped due to error")
            print(str(e))
            self.print_statistics()
            raise

    def show_prediction_results(self, start_idx, end_idx, df, predictions):
        """Show prediction results for a range of rows"""
        print(f"\nResults for rows {start_idx+1} to {end_idx+1}:")
        print("-" * 80)
        for i, idx in enumerate(range(start_idx, end_idx + 1)):
            if i < len(predictions):  # Check if prediction exists
                pred = predictions[i]
                row_data = df.iloc[idx]
                print(f"\nRow {idx+1}:")
                print(f"Input: {json.dumps(row_data.to_dict(), indent=2)}")
                print(f"Prediction: {json.dumps(pred, indent=2)}")
        print("-" * 80)

    def get_user_confirmation(self):
        """Get user confirmation to continue"""
        while True:
            response = input("\nWould you like to continue? (yes/no): ").lower()
            if response in ['yes', 'y']:
                return True
            if response in ['no', 'n']:
                return False
            print("Please enter 'yes' or 'no'")

    def calculate_metrics(self, actual, predicted, predicted_probs):
        """Calculate evaluation metrics with error handling"""
        try:
            metrics = {}
            metrics['accuracy'] = accuracy_score(actual, predicted)
            precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='binary')
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            metrics['roc_auc'] = roc_auc_score(actual, predicted_probs)
            
            tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
            metrics['confusion_matrix'] = {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
            return metrics
        except Exception as e:
            print(f"\nError calculating metrics: {str(e)}")
            self.failed_rows.append({
                'error_type': 'metrics_calculation',
                'error_message': str(e)
            })
            return None

    def calculate_and_print_metrics(self, actual_values, predicted_values, predicted_probs):
        """Calculate and print evaluation metrics"""
        metrics = self.calculate_metrics(actual_values, predicted_values, predicted_probs)
        if metrics:
            print("\nEvaluation Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
            print("\nConfusion Matrix:")
            print(f"True Negatives: {metrics['confusion_matrix']['true_negative']}")
            print(f"False Positives: {metrics['confusion_matrix']['false_positive']}")
            print(f"False Negatives: {metrics['confusion_matrix']['false_negative']}")
            print(f"True Positives: {metrics['confusion_matrix']['true_positive']}")

    def print_statistics(self):
        """Print processing statistics"""
        print("\nProcessing Statistics:")
        print(f"Total rows processed: {self.stats['total_rows_processed']}")
        print(f"Successful predictions: {self.stats['successful_predictions']}")
        print(f"Failed predictions: {self.stats['failed_predictions']}")
        print(f"Successful Arize logs: {self.stats['successful_arize_logs']}")
        print(f"Failed Arize logs: {self.stats['failed_arize_logs']}")
        print(f"Rows with target: {self.stats['rows_with_target']}")
        print(f"Rows without target: {self.stats['rows_without_target']}")
        
        if self.failed_rows:
            print("\nFailed Rows Summary:")
            error_types = {}
            for row in self.failed_rows:
                error_type = row['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print("\nErrors by type:")
            for error_type, count in error_types.items():
                print(f"{error_type}: {count}")

def main():
    # Configuration
    ENDPOINT_URL = "https://h2o-mojo-service-271854447431.us-central1.run.app/predict"
    #ENDPOINT_URL = "http://35.184.233.137:38080/predict"
    CSV_FILE = "/Users/prashantkulkarni/Documents/ml-ops/project/dataset/dataset-v2/cred_card_featured_eng_test_ref.csv"  # Replace with your CSV file path
    DELAY_MS = 100  # 100ms delay between requests
    
    try:
        client = PredictionClient(
            endpoint_url=ENDPOINT_URL,
            arize_api_key="your_api_key",  
            arize_space_key="your_space_key"  
        )
        client.process_file(
            csv_file=CSV_FILE,
            delay_ms=DELAY_MS
        )
    except Exception as e:
        print("\nExecution stopped due to error")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()    