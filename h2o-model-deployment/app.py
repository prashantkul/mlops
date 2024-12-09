# app.py
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
        logger.info("Model loaded successfully")

def create_row_data(input_data: Dict) -> 'hex.genmodel.easy.RowData':
    """Convert input data to RowData format"""
    from hex.genmodel.easy import RowData
    row = RowData()
    
    # Handle binary features with True/False/Missing states
    binary_features = [
        "ChldNo_1", "gp_Age_high", "gp_Age_highest", "gp_Age_low", "gp_Age_lowest",
        "gp_worktm_high", "gp_worktm_highest", "gp_worktm_low", "gp_worktm_medium",
        "occyp_hightecwk", "occyp_officewk", "famsizegp_1", "famsizegp_3more",
        "houtp_Co-opapartment", "houtp_Municipalapartment", "houtp_Officeapartment",
        "houtp_Rentedapartment", "houtp_Withparents", "edutp_Highereducation",
        "edutp_Incompletehigher", "edutp_Lowersecondary", "famtp_Civilmarriage",
        "famtp_Separated", "famtp_Single/notmarried", "famtp_Widow"
    ]
    
    # Set defaults for all binary features
    for feature in binary_features:
        row.put(f"{feature}.True", 0.0)
        row.put(f"{feature}.False", 1.0)
        row.put(f"{feature}.missing(NA)", 0.0)
    
    # Update binary features based on input
    for feature in binary_features:
        if feature in input_data:
            value = input_data[feature]
            if value:
                row.put(f"{feature}.True", 1.0)
                row.put(f"{feature}.False", 0.0)
            else:
                row.put(f"{feature}.True", 0.0)
                row.put(f"{feature}.False", 1.0)
    
    # Handle numeric features
    numeric_features = ["Gender", "inc", "Reality", "wkphone"]
    for feature in numeric_features:
        if feature in input_data:
            row.put(feature, float(input_data[feature]))
    
    return row

def convert_prediction_to_dict(prediction):
    """Convert Java prediction object to Python dictionary"""
    return {
        'predicted_class': str(prediction.label),  # Convert Java String to Python string
        'probabilities': {
            '0': float(prediction.classProbabilities[0]),  # Convert Java double to Python float
            '1': float(prediction.classProbabilities[1])
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_wrapper is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Handle both single instance and batch predictions
        instances = data.get('instances', [])
        if not isinstance(instances, list):
            instances = [instances]
        
        results = []
        for instance in instances:
            row = create_row_data(instance)
            prediction = model_wrapper.predictBinomial(row)
            results.append(convert_prediction_to_dict(prediction))
        
        return jsonify({
            'predictions': results[0] if len(results) == 1 else results
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_jvm()
    app.run(host='0.0.0.0', port=8080)