import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import shap

# Load the preprocessed data
def load_preprocessed_data():
    return pd.read_csv('../data/preprocessed_data.csv')

def test_load_preprocessed_data():
    df = load_preprocessed_data()
    assert not df.empty, "Preprocessed data is empty"
    assert df.shape[0] > 0, "Preprocessed data has no rows"
    assert df.shape[1] > 0, "Preprocessed data has no columns"

def test_split_data():
    df = load_preprocessed_data()
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    assert X_train.shape[0] > 0, "Training data has no rows"
    assert X_test.shape[0] > 0, "Testing data has no rows"
    assert y_train.shape[0] > 0, "Training labels have no rows"
    assert y_test.shape[0] > 0, "Testing labels have no rows"

def test_train_models():
    df = load_preprocessed_data()
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        assert roc_auc_score(y_test, y_pred_proba) > 0.5, f"{model_name} ROC-AUC score is too low"
        assert accuracy_score(y_test, y_pred) > 0.5, f"{model_name} accuracy score is too low"
        assert precision_score(y_test, y_pred) > 0.5, f"{model_name} precision score is too low"
        assert recall_score(y_test, y_pred) > 0.5, f"{model_name} recall score is too low"
        assert f1_score(y_test, y_pred) > 0.5, f"{model_name} F1 score is too low"

def test_shap_explainability():
    df = load_preprocessed_data()
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    xgboost_model = XGBClassifier(random_state=42)
    xgboost_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X_test)
    assert shap_values is not None, "SHAP values are None"
    assert len(shap_values) == len(X_test), "SHAP values length does not match test data length"

if __name__ == "__main__":
    import pytest
    pytest.main()