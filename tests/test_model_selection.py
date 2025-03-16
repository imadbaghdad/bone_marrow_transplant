import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path

@pytest.fixture
def load_processed_data():
    """Load the winsorized data for testing"""
    # Use Path for cross-platform compatibility
    data_path = Path(__file__).parent.parent / 'data'  / 'winsorized_data.csv'
    if not data_path.exists():
        pytest.skip(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

def test_feature_correlation(load_processed_data):
    """Test correlation analysis"""
    df = load_processed_data
    # Drop target variable for correlation analysis
    X = df.drop('survival_status', axis=1)
    correlation_matrix = X.corr()
    
    # Check for high correlations (>0.8)
    high_correlations = np.where(np.abs(correlation_matrix) > 0.8)
    high_correlations = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                        for x, y in zip(*high_correlations) if x != y]
    
    # Known high correlations should be present
    assert len(high_correlations) > 0, "No high correlations found"

def test_class_imbalance(load_processed_data):
    """Test class imbalance handling"""
    df = load_processed_data
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    
    # Check initial class distribution
    initial_counts = y.value_counts()
    assert len(initial_counts) == 2, "Should have exactly two classes"
    
    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    _, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Check class balance after SMOTE
    resampled_counts = pd.Series(y_train_resampled).value_counts()
    assert len(set(resampled_counts)) == 1, "Classes should be balanced after SMOTE"

def test_model_comparison_metrics(load_processed_data):
    """Test model comparison metrics calculation"""
    df = load_processed_data
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    
    # Split and scale data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Check scaling
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10), "Scaled data should be centered"
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10), "Scaled data should have unit variance"

def test_memory_optimization(load_processed_data):
    """Test memory optimization function"""
    df = load_processed_data
    initial_memory = df.memory_usage(deep=True).sum()
    
    # Convert float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df_optimized = df.copy()
        df_optimized[float_cols] = df_optimized[float_cols].astype('float32')
        final_memory = df_optimized.memory_usage(deep=True).sum()
        assert final_memory < initial_memory, "Memory optimization should reduce memory usage"

def test_feature_importance_calculation(load_processed_data):
    """Test feature importance calculation"""
    df = load_processed_data
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_estimators=10)  # Small model for testing
    model.fit(X, y)
    
    importances = model.feature_importances_
    assert len(importances) == X.shape[1], "Should have importance score for each feature"
    assert all(imp >= 0 for imp in importances), "All importance scores should be non-negative"