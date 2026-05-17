import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Exploration des Données", page_icon="📊", layout="wide")

st.markdown("<h1 style='color: #ff6b6b;'>📊 Exploration des données (EDA)</h1>", unsafe_allow_html=True)

st.write("Avant de construire nos algorithmes d'intelligence artificielle, l'étape la plus cruciale est l'Exploration des Données (EDA). Nous travaillons sur la cohorte clinique de Cleveland (303 patients), considérée comme une référence fiable en cardiologie. Notre objectif ici est de justifier *pourquoi* des variables médicales complexes nécessitent des modèles de Machine Learning avancés plutôt que de simples règles empiriques.")

plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots")

st.markdown("### 🔍 1. Équilibre des classes et pertinence statistique")
col1, col2 = st.columns([1, 1.5])
with col1:
    try:
        st.image(os.path.join(plots_dir, "01_target_distribution.png"), use_column_width=True)
    except:
        pass
with col2:
    st.markdown("""
    **Justification du choix d'évaluation :**
    Contrairement à de nombreux datasets médicaux très déséquilibrés (où 99% des patients sont sains), le dataset de Cleveland est **parfaitement équilibré** (54% sains, 46% malades). 
    
    *Pourquoi c'est important ?* Cela nous garantit que si notre algorithme se trompe, ce n'est pas parce qu'il a "appris" à toujours prédire la majorité. De plus, cela valide l'utilisation de métriques classiques comme l'Accuracy (Précision globale) en plus du Recall (Rappel, notre priorité médicale pour ne rater aucun patient malade).
    """)

st.markdown("---")

st.markdown("### 🧩 2. Les Limites de l'Analyse Univariée (Ex: Cholestérol)")
col3, col4 = st.columns([1.5, 1])
with col3:
    st.markdown("""
    **Justification du besoin d'IA :**
    Souvent, on imagine qu'un seul facteur de risque suffit à poser un diagnostic (ex: "Un cholestérol très élevé implique une maladie cardiaque"). Le graphique ci-contre prouve le contraire : les distributions de cholestérol entre les patients sains et malades **se chevauchent énormément**. 
    
    Il est impossible de tracer une ligne de séparation stricte sur la seule base du cholestérol. C'est précisément pour cela que nous avons besoin d'un **modèle multivarié** (qui croise des dizaines de variables simultanément).
    """)
with col4:
    try:
        st.image(os.path.join(plots_dir, "02_cholesterol_distribution.png"), use_column_width=True)
    except:
        pass

st.markdown("---")

st.markdown("### 🔗 3. Matrice de Corrélation : L'interaction des variables")
st.write("La matrice de corrélation globale nous permet de quantifier la relation mathématique entre chaque paire de variables. Les cases rouges montrent une corrélation positive (les deux variables augmentent ensemble), tandis que les bleues montrent une corrélation négative.")

col5, col6 = st.columns([1.5, 1])
with col5:
    try:
        st.image(os.path.join(plots_dir, "03_global_correlation.png"), use_column_width=True)
    except:
        st.warning("Image matrice non trouvée.")
with col6:
    st.markdown("""
    **Analyse Clinique et Stratégie :**
    - **`cp` (Type de douleur thoracique), `thalach` (Fréquence cardiaque max) et `slope`** sont fortement corrélés positivement avec la variable cible (`target`). Ce sont nos meilleurs prédicteurs cliniques simples.
    - **`exang` (Angine à l'effort), `oldpeak` et `ca` (Nombre de vaisseaux)** sont fortement corrélés négativement.
    - *Pourquoi ce visuel est fondamental ?* Il montre qu'**aucune variable n'atteint une corrélation de 1 ou -1 avec la cible**. La maladie cardiaque est donc bien le fruit d'une synergie complexe de plusieurs facteurs physiologiques modérés. Cela justifie totalement notre approche par *Ensemble Learning* (Gradient Boosting / Random Forest) qui excelle dans l'agrégation de signaux faibles et non-linéaires.
    """)

st.info("💡 **Conclusion de l'EDA :** Les données sont saines, équilibrées et pleines d'intéractions complexes. L'absence de corrélation parfaite univariée nous pousse vers l'étape suivante de **Feature Engineering**, afin d'aider les algorithmes à extraire encore plus de sens médical.")
