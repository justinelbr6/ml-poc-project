import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Calcule et retourne les métriques de performance du modèle.
    Si y_prob est fourni, calcule également l'AUC-ROC.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        # Assure-toi que y_prob contient les probabilités de la classe positive (classe 1)
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob)
        
    return metrics

def print_evaluation(model_name: str, metrics: dict, conf_matrix: list = None):
    """
    Affiche joliment les résultats de l'évaluation.
    """
    print(f"--- Évaluation du modèle : {model_name} ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        
    if conf_matrix is not None:
        print("Matrice de confusion :")
        print(conf_matrix)
    print("-" * 40)

def compute_metrics(y_true: any, y_pred: any) -> dict[str, float]:
    """
    Interface requise par le template (scripts/main.py).
    Doit renvoyer un dictionnaire de floats.
    """
    metrics_dict = evaluate_model(y_true, y_pred, y_prob=None)
    
    # Assurer que toutes les valeurs sont bien des floats natifs (au cas où ce soient des np.float64)
    clean_metrics = {}
    for k, v in metrics_dict.items():
        clean_metrics[k] = float(v)
        
    return clean_metrics
