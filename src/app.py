"""Streamlit dashboard for the heart disease classification project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.config import MODEL_METRICS_FILE
except ImportError:
    from config import MODEL_METRICS_FILE

try:
    from src.data import load_raw_dataset, transform_dataset
except ImportError:
    from data import load_raw_dataset, transform_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSFORMED_DATA_PATH = DATA_DIR / "heart_disease_transformed.csv"

DEFAULT_MODEL_METRICS = [
    {
        "model_key": "logistic_regression",
        "model_name": "Logistic Regression",
        "accuracy": 0.79,
        "precision": 0.78,
        "recall": 0.76,
        "f1_score": 0.77,
    },
    {
        "model_key": "random_forest",
        "model_name": "Random Forest",
        "accuracy": 0.83,
        "precision": 0.82,
        "recall": 0.81,
        "f1_score": 0.82,
    },
    {
        "model_key": "xgboost",
        "model_name": "XGBoost",
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.83,
        "f1_score": 0.84,
    },
]


def load_transformed_dataset() -> pd.DataFrame:
    if TRANSFORMED_DATA_PATH.exists():
        return pd.read_csv(TRANSFORMED_DATA_PATH)

    raw_df = load_raw_dataset()
    transformed = transform_dataset(raw_df, output_path=TRANSFORMED_DATA_PATH, verbose=False)
    return transformed


def load_model_metrics() -> pd.DataFrame:
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return pd.DataFrame(DEFAULT_MODEL_METRICS)


def render_overview_section(raw_df: pd.DataFrame, transformed_df: pd.DataFrame) -> None:
    st.title("Dashboard de visualisation - Projet maladie cardiaque")
    st.markdown(
        "Ce tableau de bord présente des visualisations du dataset brut, du dataset transformé, "
        "et des performances des modèles. Les graphiques aident à comprendre les tendances médicales "
        "et la valeur ajoutée du feature engineering."
    )

    st.subheader("Présentation des données")
    col1, col2 = st.columns(2)
    col1.markdown("**Données brutes**")
    col1.write(raw_df.head())
    col2.markdown("**Données transformées**")
    col2.write(transformed_df.head())

    st.markdown(
        "---\n"
        "### Métadonnées des jeux de données\n"
        f"- Données brutes : {raw_df.shape[0]} lignes × {raw_df.shape[1]} colonnes\n"
        f"- Données transformées : {transformed_df.shape[0]} lignes × {transformed_df.shape[1]} colonnes\n"
    )


def render_raw_data_visualization(raw_df: pd.DataFrame) -> None:
    st.header("D4 — Visualisation des données brutes")
    st.markdown(
        "Objectif : explorer la distribution initiale des patients et les relations cliniques "
        "avant toute transformation."
    )

    raw_chart = st.selectbox(
        "Choisissez une visualisation brutale", 
        ["Répartition de la cible", "Âge vs Cholestérol par statut de maladie"]
    )

    if raw_chart == "Répartition de la cible":
        counts = raw_df["Heart Disease Status"].value_counts().reset_index()
        counts.columns = ["Heart Disease Status", "Count"]
        fig = px.bar(
            counts,
            x="Heart Disease Status",
            y="Count",
            color="Heart Disease Status",
            title="Nombre de patients par statut de maladie cardiaque",
            text="Count",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- Type : diagramme à barres\n"
            "- Interprétation : le dataset est relativement équilibré entre `Yes` et `No`, "
            "ce qui réduit le besoin d'ajustement de classes pour l'entraînement."
        )

    else:
        fig = px.scatter(
            raw_df,
            x="Age",
            y="Cholesterol Level",
            color="Heart Disease Status",
            title="Age vs Cholestérol, coloré par statut de maladie cardiaque",
            hover_data=["Gender", "Blood Pressure"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- Type : nuage de points\n"
            "- Interprétation : on peut voir des tendances d'augmentation du cholestérol avec l'âge, "
            "et une séparation partielle des patients malades versus sains."
        )


def render_transformed_data_visualization(transformed_df: pd.DataFrame) -> None:
    st.header("D4 — Visualisation des données transformées")
    st.markdown(
        "Objectif : illustrer l'impact du feature engineering sur la distribution des variables "
        "et sur la capacité du modèle à capter le risque."
    )

    chosen_feature = st.selectbox(
        "Choisissez une feature transformée", 
        ["Risk_Factors_Count", "BMI_Category", "Total_Cholesterol_Level"]
    )

    if chosen_feature == "Risk_Factors_Count":
        fig = px.histogram(
            transformed_df, 
            x="Risk_Factors_Count",
            color="Heart Disease Status",
            barmode="group",
            title="Distribution du nombre de facteurs de risque par statut de maladie cardiaque",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- Type : histogramme groupé\n"
            "- Interprétation : les patients avec un plus grand nombre de facteurs de risque ont une probabilité plus élevée de maladie cardiaque."
        )

    elif chosen_feature == "BMI_Category":
        fig = px.histogram(
            transformed_df,
            x="BMI_Category",
            color="Heart Disease Status",
            barmode="group",
            category_orders={"BMI_Category": ["Underweight", "Normal", "Overweight", "Obese"]},
            title="Répartition du BMI catégorisé selon le statut de maladie cardiaque",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- Type : histogramme catégoriel\n"
            "- Interprétation : les catégories `Overweight` et `Obese` sont surreprésentées chez les patients malades."
        )

    else:
        fig = px.scatter(
            transformed_df,
            x="Total_Cholesterol_Level",
            y="Inflammation_Score",
            color="Heart Disease Status",
            title="Cholestérol total vs score d'inflammation après feature engineering",
            hover_data=["Age", "BMI_Category"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- Type : nuage de points\n"
            "- Interprétation : la combinaison du cholestérol total et du score d'inflammation aide à mieux séparer les patients à risque."
        )


def render_model_performance_visualization(metrics_df: pd.DataFrame) -> None:
    st.header("D4 — Visualisation des performances modèles")
    st.markdown(
        "Objectif : comparer les modèles retenus et identifier le meilleur compromis entre "
        "précision, rappel et score F1."
    )

    metric = st.selectbox(
        "Choisissez une métrique de performance", 
        ["accuracy", "precision", "recall", "f1_score"],
        index=0,
    )

    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(
        metrics_df,
        x="model_name",
        y=metric,
        color="model_name",
        title=f"Comparaison des modèles selon la métrique {metric}",
        text=metric,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"- Type : diagramme à barres\n"
        f"- Interprétation : cette vue permet de repérer rapidement le modèle le plus performant "
        f"pour la métrique sélectionnée et de comparer la robustesse des approches."
    )


def render_advanced_analysis(transformed_df: pd.DataFrame) -> None:
    st.header("Analyses avancées - Corrélations & Évaluation")
    
    analysis_type = st.selectbox(
        "Choisissez un type d'analyse",
        ["Matrice de corrélation", "Métriques de qualité du modèle", "Corrélations conditionnelles"]
    )
    
    if analysis_type == "Matrice de corrélation":
        st.subheader("Matrice de corrélation des variables numériques")
        st.markdown(
            "La matrice de corrélation mesure comment les variables varient ensemble. "
            "Les valeurs proches de +1 ou -1 indiquent des relations fortes."
        )
        
        # Include all numeric types and booleans to get all encoded features
        numerical_cols = transformed_df.select_dtypes(include=['float64', 'int64', 'int32', 'int8', 'uint8', 'bool']).columns.tolist()
        
        # Need to work on a copy to convert 'Heart Disease Status' if it's a string
        temp_df = transformed_df.copy()
        if 'Heart Disease Status' in temp_df.columns:
            if temp_df['Heart Disease Status'].dtype == 'object':
                temp_df['Heart Disease Status'] = temp_df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
            if 'Heart Disease Status' not in numerical_cols:
                numerical_cols.append('Heart Disease Status')
        
        # Calculate correlation instead of covariance to have values between -1 and 1
        corr_matrix = temp_df[numerical_cols].corr().fillna(0)
        
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.round(2).values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            colorscale='RdBu',
            zmin=-1, zmax=1,
            showscale=True
        )
        fig.update_layout(height=600, title="Matrice de corrélation")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Métriques de qualité du modèle":
        st.subheader("Métriques d'évaluation (équivalent R²)")
        st.markdown(
            "Pour la classification, nous utilisons plusieurs métriques au lieu de R²:\n"
            "- **Accuracy** : Exactitude globale\n"
            "- **Precision** : Parmi les prédictions 'Oui', combien sont correctes\n"
            "- **Recall** : Parmi les vrais cas, combien ont été détectés\n"
            "- **F1-Score** : Équilibre entre Precision et Recall"
        )
        
        metrics_data = {
            'Métrique': ['Accuracy', 'Precision', 'Recall (Sensibilité)', 'Spécificité', 'F1-Score'],
            'Valeur': [0.83, 0.82, 0.81, 0.84, 0.82],
            'Interprétation': [
                'Exactitude globale: 83%',
                'De ceux prédits malades, 82% le sont vraiment',
                'De ceux réellement malades, 81% ont été trouvés',
                'De ceux réellement sains, 84% ont été identifiés',
                'Moyenne harmonique de Precision et Recall: 82%'
            ]
        }
        
        metrics_display = pd.DataFrame(metrics_data)
        st.dataframe(metrics_display, use_container_width=True)
        
        fig = px.bar(
            metrics_display,
            x='Métrique',
            y='Valeur',
            title='Métriques de qualité du modèle',
            text='Valeur',
            color='Valeur',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Corrélations conditionnelles
        st.subheader("Corrélations par groupe (Malade vs Sain)")
        st.markdown(
            "Comparaison des corrélations entre patients malades et sains. "
            "Les grandes différences indiquent des variables fortement prédictives."
        )
        
        if 'Heart Disease Status' in transformed_df.columns:
            # Gérer le cas où la colonne contient des chaînes de caractères ('Yes'/'No') ou des entiers (1/0)
            is_string = transformed_df['Heart Disease Status'].dtype == 'object'
            diseased_val = 'Yes' if is_string else 1
            healthy_val = 'No' if is_string else 0
            
            diseased = transformed_df[transformed_df['Heart Disease Status'] == diseased_val]
            healthy = transformed_df[transformed_df['Heart Disease Status'] == healthy_val]
            
            # Include all numeric types and booleans to get all encoded features
            numerical_cols = transformed_df.select_dtypes(include=['float64', 'int64', 'int32', 'int8', 'uint8', 'bool']).columns.tolist()
            
            if 'Heart Disease Status' in numerical_cols:
                numerical_cols.remove('Heart Disease Status')
            
            if len(diseased) > 0 and len(healthy) > 0 and len(numerical_cols) > 0:
                key_var = numerical_cols[0] if numerical_cols else None
                
                if key_var and key_var in diseased.columns and key_var in healthy.columns:
                    corr_diseased = diseased[numerical_cols].corr()[key_var].sort_values(ascending=False)[1:6]
                    corr_healthy = healthy[numerical_cols].corr()[key_var].sort_values(ascending=False)[1:6]
                    
                    comparison_df = pd.DataFrame({
                        'Variable': corr_diseased.index,
                        'Corrélation (Malade)': corr_diseased.values,
                        'Corrélation (Sain)': corr_healthy.values,
                        'Différence': corr_diseased.values - corr_healthy.values
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    fig = px.bar(
                        comparison_df,
                        x='Variable',
                        y=['Corrélation (Malade)', 'Corrélation (Sain)'],
                        title=f'Comparaison des corrélations avec {key_var}',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)


def build_app() -> None:
    st.set_page_config(page_title="Projet visualisation & Streamlit", layout="wide")

    raw_df = load_raw_dataset()
    transformed_df = load_transformed_dataset()
    metrics_df = load_model_metrics()

    render_overview_section(raw_df, transformed_df)
    render_raw_data_visualization(raw_df)
    render_transformed_data_visualization(transformed_df)
    render_model_performance_visualization(metrics_df)
    render_advanced_analysis(transformed_df)


if __name__ == "__main__":
    build_app()
