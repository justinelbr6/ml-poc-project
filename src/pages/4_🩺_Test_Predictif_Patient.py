import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import sys

# Pour pouvoir utiliser le pipeline de données
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data import create_features, transform_data, clean_data, load_data

st.set_page_config(page_title="Test Prédictif Patient", page_icon="🩺", layout="wide")

# CSS spécifique pour cette page
st.markdown("""
<style>
    .risk-high { background-color: #ff4d4d; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;}
    .risk-medium { background-color: #ffb84d; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;}
    .risk-low { background-color: #4CAF50; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #ff6b6b;'>🩺 Test Prédictif du Risque Cardiaque</h1>", unsafe_allow_html=True)
st.write("Entrez vos paramètres médicaux (basés sur le dataset Cleveland) pour évaluer votre risque cardiovasculaire.")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "gradient_boosting.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- FORMULAIRE DE SAISIE ---
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Démographie & Symptômes")
        age = st.number_input("Âge", min_value=20, max_value=100, value=50)
        sex_str = st.selectbox("Sexe", ["Male (1)", "Female (0)"])
        sex = 1 if "Male" in sex_str else 0
        
        cp_str = st.selectbox("Type de Douleur Thoracique (cp)", [
            "0: Asymptomatique", 
            "1: Angine typique", 
            "2: Angine atypique", 
            "3: Douleur non angineuse"
        ])
        cp = int(cp_str.split(":")[0])

    with col2:
        st.subheader("Bilan Clinique")
        trestbps = st.number_input("Tension Systolique au repos (trestbps)", min_value=90.0, max_value=200.0, value=120.0)
        chol = st.number_input("Cholestérol (chol) mg/dL", min_value=100.0, max_value=600.0, value=200.0)
        
        fbs_str = st.selectbox("Glycémie à jeun > 120 mg/dL ? (fbs)", ["Non (0)", "Oui (1)"])
        fbs = 1 if "Oui" in fbs_str else 0
        
        restecg_str = st.selectbox("ECG au repos (restecg)", [
            "0: Normal",
            "1: Anomalie onde ST-T",
            "2: Hypertrophie ventriculaire gauche"
        ])
        restecg = int(restecg_str.split(":")[0])

    with col3:
        st.subheader("Test d'effort (Ergométrie)")
        thalach = st.number_input("Fréquence cardiaque maximale (thalach)", min_value=60, max_value=220, value=150)
        
        exang_str = st.selectbox("Angine induite par l'effort ? (exang)", ["Non (0)", "Oui (1)"])
        exang = 1 if "Oui" in exang_str else 0
        
        oldpeak = st.number_input("Dépression ST induite (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
        
        slope_str = st.selectbox("Pente du segment ST (slope)", ["0: Ascendante", "1: Plate", "2: Descendante"])
        slope = int(slope_str.split(":")[0])
        
        ca = st.slider("Vaisseaux majeurs colorés (ca)", 0, 3, 0)
        
        thal_str = st.selectbox("Thalassémie (thal)", ["1: Normal", "2: Défaut fixe", "3: Défaut réversible"])
        thal = int(thal_str.split(":")[0])

    submitted = st.form_submit_button("Évaluer mon Risque 🚀")

# --- TRAITEMENT ET PRÉDICTION ---
if submitted:
    if model is None:
        st.error("Le modèle n'a pas pu être chargé. Assurez-vous d'avoir exécuté l'entraînement sur les nouvelles données.")
    else:
        input_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])
        
        with st.spinner("Analyse clinique en cours..."):
            df_clean = clean_data(input_data)
            df_feat = create_features(df_clean)
            
            try:
                # Astuce POC : On concatène pour gérer le One-Hot et le scaling correctement
                df_original = load_data(os.path.join(os.path.dirname(__file__), "..", "..", "data", "Heart_disease_cleveland_new.csv"))
                df_original = clean_data(df_original)
                df_original = create_features(df_original)
                
                df_combined = pd.concat([df_feat, df_original]).reset_index(drop=True)
                df_combined_trans = transform_data(df_combined)
                
                # Isoler le patient
                patient_features = df_combined_trans.drop(columns=['target'], errors='ignore').iloc[[0]]
                
                # Réaligner
                if hasattr(model, 'feature_names_in_'):
                    patient_features = patient_features[model.feature_names_in_]
                
                prob = model.predict_proba(patient_features)[0][1]
                risk_percentage = round(prob * 100, 1)
                
                st.markdown("---")
                st.header("Résultat de l'analyse")
                
                if risk_percentage >= 50:
                    st.markdown(f"<div class='risk-high'>Risque Élevé : {risk_percentage}%</div>", unsafe_allow_html=True)
                    st.write("Le modèle détecte une probabilité importante de maladie cardiaque basée sur la base de données clinique de Cleveland. Consultez un cardiologue.")
                elif risk_percentage >= 20:
                    st.markdown(f"<div class='risk-medium'>Risque Modéré : {risk_percentage}%</div>", unsafe_allow_html=True)
                    st.write("Le modèle détecte des facteurs de risque à surveiller.")
                else:
                    st.markdown(f"<div class='risk-low'>Risque Faible : {risk_percentage}%</div>", unsafe_allow_html=True)
                    st.write("Félicitations, votre profil de santé actuel suggère un risque cardiovasculaire faible.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")
