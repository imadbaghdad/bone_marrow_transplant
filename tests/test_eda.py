import pandas as pd
import numpy as np
from scipy.io import arff
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the dataset
def load_dataset():
    data, meta = arff.loadarff("../data/bone-marrow.arff")
    df = pd.DataFrame(data)
    return df

def test_load_dataset():
    df = load_dataset()
    assert not df.empty, "Dataset is empty"
    assert df.shape[0] > 0, "Dataset has no rows"
    assert df.shape[1] > 0, "Dataset has no columns"

def test_handle_missing_values():
    df = load_dataset()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    assert df.isnull().sum().sum() == 0, "There are still missing values in the dataset"

def test_winsorization():
    df = load_dataset()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_winsorized = df.copy()
    for col in numeric_cols:
        df_winsorized[col] = winsorize(df[col], limits=[0.05, 0.05])
    assert df_winsorized.isnull().sum().sum() == 0, "There are missing values after Winsorization"

def test_smote():
    df = load_dataset()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df_winsorized = df.copy()
    for col in numeric_cols:
        df_winsorized[col] = winsorize(df[col], limits=[0.05, 0.05])
    X = df_winsorized.drop('survival_status', axis=1)
    y = df_winsorized['survival_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train[numeric_cols], y_train)
    assert len(X_train_resampled) == len(y_train_resampled), "SMOTE did not resample correctly"
    assert y_train_resampled.value_counts().min() == y_train_resampled.value_counts().max(), "SMOTE did not balance the classes"

def test_correlation_removal():
    df = load_dataset()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df_winsorized = df.copy()
    for col in numeric_cols:
        df_winsorized[col] = winsorize(df[col], limits=[0.05, 0.05])
    X = df_winsorized.drop('survival_status', axis=1)
    y = df_winsorized['survival_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train[numeric_cols], y_train)
    correlation_matrix = pd.DataFrame(X_train_resampled, columns=numeric_cols).corr()
    threshold = 0.8
    highly_correlated_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and abs(correlation_matrix.loc[i, j]) > threshold]
    features_to_remove = set()
    for i, j in highly_correlated_pairs:
        if i not in features_to_remove and j not in features_to_remove:
            features_to_remove.add(j)
    X_train_resampled_reduced = pd.DataFrame(X_train_resampled, columns=numeric_cols).drop(columns=list(features_to_remove))
    assert len(X_train_resampled_reduced.columns) < len(numeric_cols), "No features were removed for correlation"

if __name__ == "__main__":
    import pytest
    pytest.main()