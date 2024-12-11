# test_mojo.py
import os
import jpype
import jpype.imports
import json

# Initialize JVM and point to genmodel.jar
jar_path = os.path.abspath("lib/h2o-genmodel.jar")
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_path}")

# Import required Java classes
from hex.genmodel.easy import EasyPredictModelWrapper, RowData
from hex.genmodel import MojoModel

def create_sample_row():
    # Create a sample row with all features
    row = RowData()
    
    # Binary columns (True/False/Missing)
    binary_features = [
        "ChldNo_1", "gp_Age_high", "gp_Age_highest", "gp_Age_low", "gp_Age_lowest",
        "gp_worktm_high", "gp_worktm_highest", "gp_worktm_low", "gp_worktm_medium",
        "occyp_hightecwk", "occyp_officewk", "famsizegp_1", "famsizegp_3more",
        "houtp_Co-opapartment", "houtp_Municipalapartment", "houtp_Officeapartment",
        "houtp_Rentedapartment", "houtp_Withparents", "edutp_Highereducation",
        "edutp_Incompletehigher", "edutp_Lowersecondary", "famtp_Civilmarriage",
        "famtp_Separated", "famtp_Single/notmarried", "famtp_Widow"
    ]
    
    # Set all binary features to False by default
    for feature in binary_features:
        row.put(f"{feature}.True", 0.0)
        row.put(f"{feature}.False", 1.0)
        row.put(f"{feature}.missing(NA)", 0.0)
    
    # Set specific features to True
    true_features = [
        "ChldNo_1", "gp_Age_low", "gp_worktm_low", "occyp_officewk",
        "famsizegp_1", "houtp_Municipalapartment", "edutp_Highereducation",
        "famtp_Single/notmarried"
    ]
    
    for feature in true_features:
        row.put(f"{feature}.True", 1.0)
        row.put(f"{feature}.False", 0.0)
    
    # Numeric features (using float)
    row.put("Gender", 1.0)
    row.put("inc", 50000.0)
    row.put("Reality", 1.0)
    row.put("wkphone", 1.0)
    
    return row

def test_model():
    try:
        # Load model
        model_path = os.path.abspath("model")
        print(f"Loading model from: {model_path}")
        
        model = MojoModel.load(model_path)
        wrapper = EasyPredictModelWrapper(model)
        
        print("Model loaded successfully!")
        print(f"Model category: {wrapper.getModelCategory()}")
        
        # Create sample data and make prediction
        row = create_sample_row()
        prediction = wrapper.predictBinomial(row)
        
        print("\nPrediction results:")
        print(f"Predicted class: {prediction.label}")
        print(f"Class probabilities: {json.dumps(dict(zip(['0', '1'], prediction.classProbabilities)), indent=2)}")
            
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model()