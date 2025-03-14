import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and SHAP explainer
try:
    model = joblib.load(r'C:\Users\imadb\Documents\GitHub\bone_marrow_transplant\models\rf_model_compressed.joblib')
    explainer = joblib.load(r'C:\Users\imadb\Documents\GitHub\bone_marrow_transplant\models\shap_explainer_new.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please ensure the model files are in the correct location.")
    st.stop()

def predict_success_rate(data):
    """
    Predict the success rate of bone marrow transplant
    Args:
        data (dict): Dictionary containing patient data
    Returns:
        tuple: (prediction, prediction_probability)
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    return prediction, prediction_proba

def generate_shap_explanation(data):
    """
    Generate SHAP explanation using summary plot
    Args:
        data (dict): Dictionary containing patient data
    Returns:
        tuple: (success, plot or error message)
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure correct feature order
        df = df[model.feature_names_in_]
        
        # Calculate SHAP values with additivity check disabled
        shap_values = explainer.shap_values(df, check_additivity=False)
        
        # Create visualization
        plt.clf()  # Clear any existing plots
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get importance values for positive class (index 1)
        # SHAP values shape is (samples, features, classes)
        importance_values = shap_values[0, :, 1]
        feature_names = df.columns
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importance_values))
        feature_names = np.array(feature_names)[sorted_idx]
        importance_values = importance_values[sorted_idx]
        
        # Create bar plot
        y_pos = np.arange(len(feature_names))
        colors = ['#ff4b4b' if v < 0 else '#2e8b57' for v in importance_values]
        ax.barh(y_pos, importance_values, color=colors)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP Impact')
        ax.set_title('Feature Importance Analysis')
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return True, fig
        
    except Exception as e:
        return False, f"Error generating explanation: {str(e)} ({type(e).__name__})"

# Page configuration
st.set_page_config(
    page_title="Bone Marrow Transplant Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for patient information
st.sidebar.title("Patient Information")
st.sidebar.markdown("---")

# Main content
st.title("🏥 Bone Marrow Transplant Success Prediction")
st.markdown("""
    This tool helps predict the success rate of bone marrow transplants based on patient data.
    Please fill in the required information in the sidebar.
""")

# Patient information input fields in sidebar
with st.sidebar:
    st.subheader("Patient Demographics")
    age = st.number_input("Age (years)", 
                         min_value=0, 
                         max_value=18, 
                         value=5, 
                         help="Pediatric patients up to 18 years",
                         key="age_input")
    gender = st.selectbox("Gender", 
                         ["Male", "Female"], 
                         key="gender_select")
    weight = st.number_input("Weight (kg)", 
                           min_value=1, 
                           max_value=100, 
                           value=20, 
                           help="Typical pediatric weight range",
                           key="weight_input")
    
    st.subheader("Disease Information")
    disease_status = st.selectbox(
        "Disease Status",
        ["Early", "Intermediate", "Advanced"],
        help="Current stage of the disease",
        key="disease_status_select"
    )
    
    st.subheader("Donor Information")
    donor_age = st.number_input("Donor Age", 
                               min_value=0, 
                               max_value=100, 
                               value=30,
                               key="donor_age_input")
    donor_relation = st.selectbox(
        "Donor Relation",
        ["Sibling", "Parent", "Child", "Other Related", "Unrelated"],
        key="donor_relation_select"
    )

    st.subheader("Blood Type Information")
    donor_blood = st.selectbox("Donor Blood Type", 
                              ["A", "B", "AB", "O"],
                              key="donor_blood_select")
    donor_rh = st.selectbox("Donor Rh Factor", 
                           ["Positive", "Negative"],
                           key="donor_rh_select")
    recipient_blood = st.selectbox("Recipient Blood Type", 
                                  ["A", "B", "AB", "O"],
                                  key="recipient_blood_select")
    recipient_rh = st.selectbox("Recipient Rh Factor", 
                               ["Positive", "Negative"],
                               key="recipient_rh_select")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Clinical Parameters")
    
    # Create two columns for better layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        hla_match = st.slider("HLA Match", 
                             0, 10, 5,
                             key="hla_match_slider")
        cmv_status = st.selectbox("CMV Status", 
                                 ["Positive", "Negative"],
                                 key="cmv_status_select")
        
    with right_col:
        stem_cell_source = st.selectbox(
            "Stem Cell Source",
            ["Bone Marrow", "Peripheral Blood", "Cord Blood"],
            key="stem_cell_source_select"
        )
        conditioning = st.selectbox(
            "Conditioning Regimen",
            ["Myeloablative", "Reduced Intensity", "Non-myeloablative"],
            key="conditioning_select"
        )

# Create a dictionary with all input values using the correct feature names
input_data = {
    'Recipientgender': 1 if gender == "Male" else 0,
    'Stemcellsource': {'Bone Marrow': 0, 'Peripheral Blood': 1, 'Cord Blood': 2}[stem_cell_source],
    'Donorage': donor_age,
    'Donorage35': 1 if donor_age > 35 else 0,
    'IIIV': 0,  # This might need additional input field
    'Gendermatch': 1,  # This might need additional input for donor gender
    'DonorABO': 0,  # Add blood type selection
    'RecipientABO': 0,  # Add blood type selection
    'RecipientRh': 1,  # Add Rh factor selection
    'ABOmatch': 1,  # Calculate based on donor and recipient blood types
    'CMVstatus': 1 if cmv_status == "Positive" else 0,
    'DonorCMV': 0,  # Add donor CMV status
    'RecipientCMV': 1 if cmv_status == "Positive" else 0,
    'Disease': 0,  # Add disease type selection
    'Riskgroup': {'Early': 0, 'Intermediate': 1, 'Advanced': 2}[disease_status],
    'Txpostrelapse': 0,  # Add checkbox for post-relapse transplant
    'Diseasegroup': {'Early': 0, 'Intermediate': 1, 'Advanced': 2}[disease_status],
    'HLAmatch': hla_match,
    'HLAmismatch': 10 - hla_match,
    'Antigen': 0,  # Add antigen matching information
    'Alel': hla_match,  # This might need different calculation
    'HLAgrI': 0,  # Add HLA group I information
    'Recipientage': age,
    'Recipientage10': age // 10,  # Age in decades
    'Recipientageint': 1 if 18 <= age <= 60 else 0,  # Age interval indicator
    'Relapse': 0,  # Add relapse information
    'aGvHDIIIIV': 0,  # Acute GvHD information
    'extcGvHD': 0,  # Extended chronic GvHD information
    'CD34kgx10d6': weight * 0.5,  # This needs actual CD34+ cell count
    'CD3dCD34': 0,  # This needs actual CD3/CD34 ratio
    'CD3dkgx10d8': weight * 0.3,  # This needs actual CD3+ cell count
    'Rbodymass': weight,
    'ANCrecovery': 0,  # This might need to be removed if it's an outcome
    'PLTrecovery': 0,  # This might need to be removed if it's an outcome
    'time_to_aGvHD_III_IV': 0,  # This might need to be removed if it's an outcome
    'survival_time': 0,  # This should be removed as it's the target outcome

}

# Update the prediction section
if st.button('Generate Prediction'):
    try:
        # Make prediction
        prediction, prediction_proba = predict_success_rate(input_data)
        
        # Display results
        with col2:
            st.markdown("### Prediction Results")
            
            # Display prediction
            outcome = "Success" if prediction[0] == 1 else "Failure"
            success_rate = prediction_proba[0][1]  # Probability of success
            
            st.metric("Predicted Outcome", outcome)
            st.metric("Success Probability", f"{success_rate:.1%}")
            
            # Generate and display SHAP explanation
            with st.expander("See Feature Importance"):
                success, result = generate_shap_explanation(input_data)
                if success:
                    st.pyplot(result)
                else:
                    st.error(result)
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <small>This tool is for research purposes only. Always consult with medical professionals for clinical decisions.</small>
    </div>
""", unsafe_allow_html=True)