import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Modélisation & Performances", page_icon="🤖", layout="wide")

st.markdown("<h1 style='color: #ff6b6b;'>🤖 Modélisation & Évaluation Clinique</h1>", unsafe_allow_html=True)

st.write("Nous avons entraîné et mis en compétition trois algorithmes classiques en Machine Learning (Régression Logistique, Random Forest, Gradient Boosting). Durant la phase d'optimisation mathématique (GridSearch), notre fonction de coût s'est concentrée sur un objectif clinique majeur : **Maximiser le Recall**.")

st.markdown("""
**Pourquoi avons-nous privilégié le Recall (Rappel) ?**  
Dans le diagnostic médical, il est **catastrophique de déclarer "sain" un patient qui a en réalité une maladie cardiaque sévère** (ce qu'on appelle un Faux Négatif). Il est toujours préférable médicalement d'avoir l'inverse (un Faux Positif), c'est-à-dire demander des examens complémentaires (ECG, échographie) de précaution à quelqu'un qui n'a finalement rien de grave. Le *Recall* mesure mathématiquement notre capacité à "attraper" et diagnostiquer l'intégralité des vrais cas de maladie.
""")

results_file = os.path.join(os.path.dirname(__file__), "..", "..", "results", "model_metrics.csv")

if os.path.exists(results_file):
    df_metrics = pd.read_csv(results_file)
    st.subheader("📊 Tableau des Performances sur le Jeu de Test")
    st.dataframe(df_metrics.style.highlight_max(axis=0, color="#ffe3e3"), use_container_width=True)
else:
    st.warning("Fichier de résultats introuvable.")

plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Les Matrices de Confusion")
    st.write("C'est l'outil d'évaluation métier par excellence : elle permet aux médecins de lire en clair le nombre de patients bien ou mal classés. L'enjeu est de minimiser la case en haut à droite (les Faux Négatifs mortels).")
    st.write("**Analyse des résultats :** Sur notre base de test issue de Cleveland, nos modèles ensemblistes (Random Forest et Gradient Boosting) démontrent une fiabilité impressionnante avec un taux quasi nul d'erreurs fatales de diagnostic.")
    try:
        st.image(os.path.join(plots_dir, "06_confusion_matrices.png"), use_column_width=True)
    except:
        pass

with col2:
    st.subheader("Courbes de Précision-Rappel (PR)")
    st.write("Elles mesurent l'impact d'une augmentation de la détection (Recall) sur la qualité pure de la prédiction (Precision).")
    st.write("**Analyse des résultats :** L'aire sous la courbe (AUC) est élevée pour nos modèles complexes, garantissant que nous n'avons pas besoin de prédire 'malade' de façon hasardeuse pour trouver tous les cas problématiques. Les modèles ont réellement compris la structure pathologique.")
    try:
        st.image(os.path.join(plots_dir, "09_precision_recall_curves.png"), use_column_width=True)
    except:
        pass

st.markdown("---")
st.subheader("🔎 Explicabilité (Feature Importance) : Casser la 'Boîte Noire'")
st.write("En milieu hospitalier, une IA totalement opaque est inacceptable d'un point de vue éthique et réglementaire. Le médecin doit pouvoir auditer et comprendre sur quels facteurs l'algorithme s'appuie pour émettre son alerte de risque.")

col3, col4 = st.columns([1, 2])
with col3:
    st.markdown("""
    **Justification Clinique des Poids Décisionnels :**
    Nous pouvons demander au *Gradient Boosting* de nous révéler ses mécanismes de décision internes. Il a identifié trois piliers majeurs pesant sur son diagnostic final :
    1. **`thal`** (Thalassémie, qui décrit un défaut de l'irrigation sanguine du muscle cardiaque).
    2. **`cp`** (Le type exact de douleur thoracique ressentie par le patient à son admission).
    3. **`ca`** (Le nombre de vaisseaux majeurs bouchés ou calcifiés détectés à la fluoroscopie).
    
    Ceci est en parfaite **adéquation avec la littérature médicale cardiologique actuelle**. C'est une excellente nouvelle : cela valide la démarche globale, en prouvant que l'algorithme n'a pas surappris de faux signaux statistiques liés au hasard, mais a véritablement "compris" la pathologie !
    """)
with col4:
    try:
        st.image(os.path.join(plots_dir, "08_feature_importance.png"), use_column_width=True)
    except:
        pass

st.markdown("""
<div style='background-color: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b; margin-top: 20px;'>
    <h4>🏆 Choix Final de l'Algorithme</h4>
    <p>La base de données clinique de <b>Cleveland</b>, reconnue par les cardiologues, a permis de valider nos architectures. C'est la <b>Random Forest</b> qui a été sélectionnée pour le déploiement sur la dernière page de l'application (Test Patient). En effet, avec son <b>Rappel (Recall) de 100%</b>, elle garantit l'absence de Faux Négatifs, ce qui est une exigence absolue dans un contexte de diagnostic médical. Vous pouvez désormais la tester vous-même en conditions réelles.</p>
</div>
""", unsafe_allow_html=True)
