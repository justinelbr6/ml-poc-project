from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Students must replace this example with their trained models.
# Each entry must point to a serialized model saved as `.joblib`, `.pkl`, or `.pickle`.
MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Linear model for binary classification. Baseline interpretable model.",
        "path": MODELS_DIR / "logistic_regression.pkl",
        "type": "linear",
        "advantages": ["Highly interpretable", "Fast training", "Good baseline"],
        "limitations": ["Assumes linear relationships", "Sensitive to outliers"],
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of decision trees. Robust to overfitting and handles mixed data types.",
        "path": MODELS_DIR / "random_forest.pkl",
        "type": "tree_ensemble",
        "advantages": ["Handles mixed data types", "Robust to outliers", "Feature importance"],
        "limitations": ["Less interpretable", "Can be slow on large datasets"],
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosting framework. State-of-the-art performance for tabular data.",
        "path": MODELS_DIR / "xgboost.pkl",
        "type": "boosting",
        "advantages": ["High performance", "Handles missing values", "Feature selection"],
        "limitations": ["Computationally intensive", "Hyperparameter tuning required"],
    },
}
