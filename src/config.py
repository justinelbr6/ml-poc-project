import os
from pathlib import Path

# Chemins de base du projet (compatibilité avec scripts/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = SRC_DIR / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_PORT = 8501

# Nos chemins spécifiques pour la donnée
RAW_DATA_PATH = DATA_DIR / "heart_disease.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_disease_transformed.csv"

# Paramètres de Split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Registre des modèles entraînés requis par main.py
MODELS = {
    "logistic_regression": {
        "name": "Régression Logistique",
        "description": "Baseline linéaire (class_weight='balanced')",
        "path": MODELS_DIR / "logistic_regression.pkl",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble d'arbres capturant les interactions",
        "path": MODELS_DIR / "random_forest.pkl",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Modèle très performant sur ce type de données",
        "path": MODELS_DIR / "gradient_boosting.pkl",
    }
}

# Dictionnaire de configuration des hyperparamètres pour GridSearchCV (utilisé par scripts/train_and_save_models.py)
MODEL_PARAMS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", None]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}
