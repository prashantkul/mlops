from flask import Flask, request, jsonify
import os
import jpype
import jpype.imports
import json
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model wrapper
model_wrapper = None

# Global feature lists
NUMERIC_FEATURES = ["Gender", "inc", "Reality", "wkphone"]
BINARY_FEATURES = [
    "ChldNo_1", "gp_Age_high", "gp_Age_highest", "gp_Age_low", "gp_Age_lowest",
    "gp_worktm_high", "gp_worktm_highest", "gp_worktm_low", "gp_worktm_medium",
    "occyp_hightecwk", "occyp_officewk", "famsizegp_1", "famsizegp_3more",
    "houtp_Co_op_apartment", "houtp_Municipal_apartment", "houtp_Office_apartment",
    "houtp_Rented_apartment", "houtp_With_parents", "edutp_Higher_education",
    "edutp_Incomplete_higher", "edutp_Lower_secondary", "famtp_Civil_marriage",
    "famtp_Separated", "famtp_Single_not_married", "famtp_Widow"
]

def initialize_jvm():
    """Initialize JVM if not already started"""
    if not jpype.isJVMStarted():
        jar_path = os.path.abspath("lib/h2o-genmodel.jar")
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_path}")
        from hex.genmodel.easy import EasyPredictModelWrapper
        from hex.genmodel import MojoModel
        
        global model_wrapper
        model_path = os.path.abspath("model")
        model = MojoModel.load(model_path)
        model_wrapper = EasyPredictModelWrapper(model)
        logger.info(json.dumps({"message": "Model loaded successfully"}))

def create_row_data(input_data: Dict) -> 'hex.genmodel.easy.RowData':
    """Convert input data to RowData format matching MOJO's encoding"""
    from hex.genmodel.easy import RowData
    row = RowData()
    
    # Create a dictionary to track all feature values for logging
    feature_values = {}
    
    # Handle numeric features
    for feature in NUMERIC_FEATURES:
        if feature in input_data:
            value = float(input_data[feature])
            row.put(feature, value)
            feature_values[feature] = value
    
    # Handle binary features with three-state encoding
    for feature in BINARY_FEATURES:
        if feature in input_data:
            value = input_data[feature]
            row.put(f"{feature}.True", 1.0 if value else 0.0)
            row.put(f"{feature}.False", 0.0 if value else 1.0)
            row.put(f"{feature}.missing(NA)", 0.0)
            feature_values[feature] = value
            
    logger.info(json.dumps({
        "message": "Feature values for prediction",
        "features": feature_values
    }))
    
    return row

def convert_prediction_to_dict(prediction):
    """Convert Java prediction object to Python dictionary"""
    # Get probabilities for each class
    prob_0 = float(prediction.classProbabilities[0])
    prob_1 = float(prediction.classProbabilities[1])
    
    # Determine predicted class based on higher probability
    predicted_class = "1" if prob_1 > prob_0 else "0"
    
    result = {
        'predicted_class': predicted_class,
        'probabilities': {
            '0': prob_0,
            '1': prob_1
        }
    }
    logger.info(json.dumps({
        "message": "Prediction result",
        "prediction": result
    }))
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model_wrapper is not None
    }
    logger.info(json.dumps({
        "message": "Health check",
        "status": status
    }))
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for H2O MOJO"""
    try:
        logger.info(json.dumps({"message": "Starting new prediction"}))
        data = request.get_json()
        
        # Log the raw input data
        logger.info(json.dumps({
            "message": "Received prediction request",
            "input_data": data
        }))
        
        if not data:
            error_msg = "No input data provided"
            logger.error(json.dumps({
                "message": error_msg,
                "error": error_msg
            }))
            return jsonify({'error': error_msg}), 400
        
        instances = data.get('instances', [])
        if not isinstance(instances, list):
            instances = [instances]
        
        logger.info(json.dumps({
            "message": "Processing instances",
            "instance_count": len(instances)
        }))
        
        results = []
        for idx, instance in enumerate(instances):
            logger.info(json.dumps({
                "message": f"Processing instance {idx + 1}",
                "instance_data": instance
            }))
            
            row = create_row_data(instance)
            prediction = model_wrapper.predictBinomial(row)
            results.append(convert_prediction_to_dict(prediction))
        
        response = {
            'predictions': results[0] if len(results) == 1 else results
        }
        
        logger.info(json.dumps({
            "message": "Prediction complete",
            "response": response
        }))
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = {
            "message": "Error in prediction",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(json.dumps(error_msg))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_jvm()
    app.run(host='0.0.0.0', port=8080)