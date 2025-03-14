import pytest
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import shap
import os

# Use environment variables for model paths
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(__file__).parent.parent / 'models' / 'rf_model_compressed.joblib'))
EXPLAINER_PATH = os.getenv('EXPLAINER_PATH', str(Path(__file__).parent.parent / 'models' / 'shap_explainer_new.joblib'))

@pytest.fixture(scope="module")
def model_and_explainer():
    """Load model and explainer for testing"""
    try:
        model = joblib.load(MODEL_PATH)
        # Create a new SHAP explainer instead of loading
        X_sample = pd.DataFrame([[0] * len(model.feature_names_in_)], columns=model.feature_names_in_)
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        pytest.skip(f"Could not load model or create explainer: {str(e)}")

@pytest.fixture
def sample_input_data():
    """Create sample input data for testing"""
    return {
        'Recipientgender': 1,
        'Stemcellsource': 0,
        'Donorage': 30,
        'Donorage35': 0,
        'IIIV': 0,
        'Gendermatch': 1,
        'DonorABO': 0,
        'RecipientABO': 0,
        'RecipientRh': 1,
        'ABOmatch': 1,
        'CMVstatus': 0,
        'DonorCMV': 0,
        'RecipientCMV': 0,
        'Disease': 0,
        'Riskgroup': 0,
        'Txpostrelapse': 0,
        'Diseasegroup': 0,
        'HLAmatch': 5,
        'HLAmismatch': 5,
        'Antigen': 0,
        'Alel': 5,
        'HLAgrI': 0,
        'Recipientage': 10,
        'Recipientage10': 1,
        'Recipientageint': 1,
        'Relapse': 0,
        'aGvHDIIIIV': 0,
        'extcGvHD': 0,
        'CD34kgx10d6': 10,
        'CD3dCD34': 0,
        'CD3dkgx10d8': 6,
        'Rbodymass': 20,
        'ANCrecovery': 0,
        'PLTrecovery': 0,
        'time_to_aGvHD_III_IV': 0,
        'survival_time': 0
    }

def test_predict_success_rate(model_and_explainer, sample_input_data):
    """Test prediction functionality"""
    model, _ = model_and_explainer
    
    # Convert input data to DataFrame
    df = pd.DataFrame([sample_input_data])
    
    # Make prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # Verify prediction format
    assert isinstance(prediction, np.ndarray), "Prediction should be numpy array"
    assert prediction.shape == (1,), "Should have one prediction"
    assert prediction[0] in [0, 1], "Prediction should be binary"
    assert prediction_proba.shape == (1, 2), "Should have probabilities for both classes"
    assert np.isclose(np.sum(prediction_proba[0]), 1), "Probabilities should sum to 1"

def test_generate_shap_explanation(model_and_explainer, sample_input_data):
    """Test SHAP explanation generation"""
    _, explainer = model_and_explainer
    
    # Convert input data to DataFrame
    df = pd.DataFrame([sample_input_data])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(df, check_additivity=False)
    
    # Verify SHAP values format
    assert isinstance(shap_values, list), "SHAP values should be a list for binary classification"
    assert len(shap_values) == 2, "Should have values for both classes"
    assert isinstance(shap_values[0], np.ndarray), "Each class's values should be numpy array"
    assert shap_values[0].shape[0] == 1, "Should have one sample"
    assert shap_values[0].shape[1] == len(df.columns), "Should have values for all features"

def test_input_data_preprocessing(sample_input_data):
    """Test input data preprocessing"""
    df = pd.DataFrame([sample_input_data])
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) == len(df.columns), "All columns should be numeric"
    
    # Check value ranges
    assert df['Recipientage'].between(0, 18).all(), "Age should be in pediatric range"
    assert df['HLAmatch'].between(0, 10).all(), "HLA match should be between 0 and 10"
    assert df['Donorage'].between(0, 100).all(), "Donor age should be valid"

def test_feature_importance_plot(model_and_explainer, sample_input_data):
    """Test feature importance plot generation"""
    _, explainer = model_and_explainer
    
    # Calculate SHAP values
    df = pd.DataFrame([sample_input_data])
    shap_values = explainer.shap_values(df, check_additivity=False)
    
    # Create plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get importance values for positive class (index 1)
    importance_values = np.abs(shap_values[1][0])  # Get absolute values for first sample
    feature_names = df.columns
    
    # Sort by absolute importance
    sorted_idx = np.argsort(importance_values)
    feature_names = np.array(feature_names)[sorted_idx]
    importance_values = importance_values[sorted_idx]
    
    # Create bar plot
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    
    # Verify plot elements
    assert len(ax.patches) == len(feature_names), "Should have one bar per feature"
    plt.close()

def test_error_handling(model_and_explainer, sample_input_data):
    """Test error handling for invalid inputs"""
    model, _ = model_and_explainer
    
    # Test with missing feature
    invalid_data = sample_input_data.copy()
    del invalid_data['Recipientage']
    df_invalid = pd.DataFrame([invalid_data])
    
    with pytest.raises(KeyError):
        df_invalid[model.feature_names_in_]