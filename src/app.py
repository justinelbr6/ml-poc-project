import streamlit as st
from PIL import Image

def build_app():
    st.set_page_config(
        page_title="CardioCare AI - Prédiction du Risque Cardiaque",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for aesthetic pastel red design
    st.markdown("""
    <style>
        .main-title {
            color: #d94545;
            font-size: 3rem !important;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0px;
        }
        .sub-title {
            color: #ff6b6b;
            font-size: 1.5rem;
            text-align: center;
            font-weight: 300;
            margin-bottom: 40px;
        }
        .stButton>button {
            border-radius: 20px;
            background-color: #ff6b6b;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #d94545;
            transform: scale(1.05);
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 5px solid #ff6b6b;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>❤️ CardioCare AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Votre assistant préventif de santé cardiovasculaire</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>👋 Bienvenue sur notre Proof of Concept !</h4>
            <p>Ce projet de Machine Learning vise à prédire le risque de maladie cardiaque à partir d'informations cliniques et comportementales.</p>
            <p>Dans la barre latérale, vous pourrez explorer :</p>
            <ul>
                <li>📊 <b>Exploration des données</b> : Comprendre le profil de nos 10 000 patients.</li>
                <li>⚙️ <b>Feature Engineering</b> : Comment nous avons créé de nouvelles variables médicales pertinentes.</li>
                <li>🤖 <b>Performances des modèles</b> : Pourquoi nous avons choisi le Gradient Boosting.</li>
                <li>🩺 <b>Test Prédictif Patient</b> : Un espace interactif pour tester l'algorithme sur vous-même !</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><hr><center><small>Projet de Machine Learning - Démo Proof of Concept</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    build_app()
