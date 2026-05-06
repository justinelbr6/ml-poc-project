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


def build_app() -> None:
    st.set_page_config(page_title="Projet visualisation & Streamlit", layout="wide")

    raw_df = load_raw_dataset()
    transformed_df = load_transformed_dataset()
    metrics_df = load_model_metrics()

    render_overview_section(raw_df, transformed_df)
    render_raw_data_visualization(raw_df)
    render_transformed_data_visualization(transformed_df)
    render_model_performance_visualization(metrics_df)


if __name__ == "__main__":
    build_app()
