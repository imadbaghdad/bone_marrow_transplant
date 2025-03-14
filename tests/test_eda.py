import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_missing_values():
    """Test missing values identification and handling"""
    # Create sample data with known missing values
    data = {
        'Recipientage': [10, 15, np.nan, 8],
        'Donorage': [30, np.nan, 25, 35],
        'HLAmatch': [8, 7, np.nan, 6],
        'survival_time': [100, 200, 150, np.nan]
    }
    df = pd.DataFrame(data)
    
    # Test missing value identification
    assert df.isnull().sum().sum() == 4, "Should identify 4 missing values"
    
    # Test missing value handling
    df_filled = df.fillna({
        'Recipientage': df['Recipientage'].mean(),
        'Donorage': df['Donorage'].mean(),
        'HLAmatch': df['HLAmatch'].median(),
        'survival_time': df['survival_time'].median()
    })
    
    assert df_filled.isnull().sum().sum() == 0, "All missing values should be filled"

def test_data_types():
    """Test data type consistency"""
    data = {
        'Recipientage': [10, 15, 8],
        'Recipientgender': [1, 0, 1],
        'survival_time': [100, 200, 150]
    }
    df = pd.DataFrame(data)
    
    assert df['Recipientage'].dtype in ['int64', 'float64'], "Age should be numeric"
    assert df['Recipientgender'].dtype in ['int64', 'bool'], "Gender should be binary"
    assert df['survival_time'].dtype in ['int64', 'float64'], "Survival time should be numeric"

def test_value_ranges():
    """Test value ranges are within expected bounds"""
    data = {
        'Recipientage': [10, 15, 8],
        'HLAmatch': [8, 7, 6],
        'Donorage': [30, 25, 35]
    }
    df = pd.DataFrame(data)
    
    assert df['Recipientage'].between(0, 18).all(), "Patient age should be between 0 and 18"
    assert df['HLAmatch'].between(0, 10).all(), "HLA match should be between 0 and 10"
    assert df['Donorage'].between(0, 100).all(), "Donor age should be between 0 and 100"

def test_categorical_values():
    """Test categorical values are within expected categories"""
    data = {
        'Stemcellsource': [0, 1, 2],
        'Disease': [0, 1, 2],
        'Riskgroup': [0, 1, 2]
    }
    df = pd.DataFrame(data)
    
    assert df['Stemcellsource'].isin([0, 1, 2]).all(), "Invalid stem cell source category"
    assert df['Disease'].isin([0, 1, 2]).all(), "Invalid disease category"
    assert df['Riskgroup'].isin([0, 1, 2]).all(), "Invalid risk group category"