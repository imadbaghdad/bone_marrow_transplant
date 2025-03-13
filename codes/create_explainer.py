import pandas as pd
import numpy as np
import joblib
import shap

# Load the model and training data
model = joblib.load(r'C:\Users\imadb\Documents\GitHub\bone_marrow_transplant\models\rf_model_compressed.joblib')
df = pd.read_csv(r'C:\Users\imadb\Documents\GitHub\bone_marrow_transplant\data\winsorized_dataset_optimized.csv')

# Remove target variables and non-feature columns
target_columns = ['survival_status', 'survival_time']
df_features = df.drop(columns=target_columns)

# Convert all columns to numeric
for column in df_features.columns:
    if df_features[column].dtype == 'object':
        df_features[column] = pd.to_numeric(df_features[column], errors='coerce')

# Fill any NaN values with column means
df_features = df_features.fillna(df_features.mean())

print("Dataset shape:", df_features.shape)
print("\nFeature names:")
print(df_features.columns.tolist())
print("\nData types after conversion:")
print(df_features.dtypes)

# Create a background dataset for SHAP (use a subset of training data)
background_data = df_features.sample(n=100, random_state=42)

# Create the SHAP explainer
try:
    explainer = shap.TreeExplainer(model, background_data)
    
    # Save the explainer
    joblib.dump(explainer, r'C:\Users\imadb\Documents\GitHub\bone_marrow_transplant\models\shap_explainer_new.joblib')
    print("\nSHAP explainer created and saved successfully!")
except Exception as e:
    print("\nError creating SHAP explainer:", str(e))
    print("\nModel feature names:", model.feature_names_in_)
    print("\nBackground data columns:", background_data.columns.tolist())