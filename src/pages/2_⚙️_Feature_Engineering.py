import streamlit as st
import os

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

st.markdown("<h1 style='color: #ff6b6b;'>⚙️ Ingénierie des Variables (Feature Engineering)</h1>", unsafe_allow_html=True)

st.write("L'ingénierie des variables consiste à utiliser notre expertise métier (ici clinique et médicale) pour pré-calculer des indicateurs complexes. Le but est de « mâcher le travail » de nos algorithmes pour augmenter leur précision.")

plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots")

st.markdown("""
<div style='background-color: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b; margin-bottom: 20px;'>
    <h4>✨ Nos choix d'ingénierie et leurs justifications cliniques :</h4>
    <ul>
        <li><b>Compteur de Facteurs de Risque (Risk_Factors_Count)</b> : Plutôt que de laisser l'IA deviner que l'accumulation de problèmes isolés est grave, nous créons un "score de comorbidité" qui s'incrémente si le patient franchit des seuils cliniques reconnus (âge > 55, cholestérol > 240, tension > 140, glycémie à jeun anormale).<br><b>Justification :</b> Les cardiologues raisonnent en "terrain à risque" : le cumul des facteurs a souvent un effet délétère de nature exponentielle, et non pas seulement additive.</li>
        <br>
        <li><b>Index d'Effort (Exercise_Index)</b> : Nous combinons la fréquence cardiaque maximale atteinte (`thalach`) avec la dépression du segment ST induite par l'effort (`oldpeak`).<br><b>Justification :</b> Ces deux variables décrivent la réponse physiologique du cœur lors d'une ergométrie (test d'effort). Les croiser permet de capter directement un proxy de la "robustesse coronaire" globale du patient.</li>
        <br>
        <li><b>Groupe d'Âge (Age_Group)</b> : La discrétisation de l'âge (Jeune, Âge Moyen, Senior).<br><b>Justification :</b> Cela permet de lisser la donnée et de capturer des paliers de risques physiologiques majeurs (comme le vieillissement artériel naturel ou l'impact de la ménopause sur les risques cardiaques chez la femme).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📈 Impact de nos variables et choix d'architecture algorithmique")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Corrélations avec la maladie (Target)**")
    st.write("Ce graphique confirme que les features médicales d'origine (comme la douleur thoracique `cp` ou l'angine induite `exang`) ont un poids décisionnel majeur. Nos variables ingénieriées comme le `Risk_Factors_Count` viennent apporter un appui statistique solide supplémentaire.")
    try:
        st.image(os.path.join(plots_dir, "04_target_correlation.png"), use_column_width=True)
    except:
        pass

with col2:
    st.markdown("**La barrière de la non-linéarité (PCA)**")
    st.write("Si nous utilisons une analyse mathématique (PCA) pour comprimer le dossier médical des patients en seulement 2 dimensions visuelles, on constate que les points malades (orange) et sains (bleu) restent profondément **enchevêtrés** au centre.")
    st.write("**Justification du choix de modèle :** Cette visualisation est une preuve formelle qu'une frontière de décision linéaire classique (ex: un trait droit séparant les sains des malades, typique d'une Régression Logistique de base) sera insuffisante et génèrera beaucoup d'erreurs. Cela confirme et motive notre choix d'utiliser des architectures sophistiquées en arbres (Random Forest, Gradient Boosting) qui sont capables d'apprendre des zones d'interaction de manière non-linéaire.")
    try:
        st.image(os.path.join(plots_dir, "05_pca_projection.png"), use_column_width=True)
    except:
        pass
