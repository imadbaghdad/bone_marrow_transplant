import streamlit as st
import joblib
import numpy as np

# Charger le modèle
MODEL_PATH = r"C:\Users\WINDOWS 11\Documents\GitHub\bone_marrow_transplant\notebooks\app\model.joblib"

@st.cache_data
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# Interface utilisateur
st.title("Prédiction pour la transplantation de moelle osseuse")
st.write("Entrez les caractéristiques du patient pour obtenir une prédiction.")

# Exemple de champs pour l'utilisateur
age = st.number_input("Âge du patient", min_value=0, max_value=120, value=30)
gender = st.selectbox("Genre", ["Homme", "Femme"])
white_blood_cell_count = st.number_input("Nombre de globules blancs (WBC)", min_value=0.0, value=5.0)

# Conversion des entrées utilisateur en données exploitables par le modèle
gender_encoded = 1 if gender == "Homme" else 0  # Encodage binaire du genre

# Bouton pour prédire
if st.button("Prédire"):
    input_data = np.array([[age, gender_encoded, white_blood_cell_count]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Le patient est éligible pour la transplantation.")
    else:
        st.error("Le patient n'est pas éligible pour la transplantation.")
