"""Metrics for Heart Disease Classification Project.

This module provides evaluation metrics specifically designed for binary classification
of heart disease prediction, where both false negatives (missing disease) and false
positives (unnecessary medical procedures) have significant consequences.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive metrics for heart disease classification.

    Args:
        y_true: True labels (0/1 or 'No'/'Yes')
        y_pred: Predicted labels (0/1)
        y_pred_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary with all evaluation metrics

    Key Metrics for Medical Classification:
    - Accuracy: Overall correctness
    - Precision: True Positives / (True Positives + False Positives)
      -> Avoids unnecessary medical procedures
    - Recall (Sensitivity): True Positives / (True Positives + False Negatives)
      -> Avoids missing actual heart disease cases (CRITICAL)
    - Specificity: True Negatives / (True Negatives + False Positives)
      -> Avoids false alarms
    - F1-Score: Harmonic mean of Precision and Recall
    - AUC-ROC: Area Under ROC Curve (discriminative ability)
    """

    # Convert string labels to numeric if needed
    if isinstance(y_true[0], str):
        y_true_numeric = np.where(y_true == 'Yes', 1, 0)
    else:
        y_true_numeric = y_true

    # Basic classification metrics
    metrics = {
        'accuracy': accuracy_score(y_true_numeric, y_pred),
        'precision': precision_score(y_true_numeric, y_pred, zero_division=0),
        'recall': recall_score(y_true_numeric, y_pred, zero_division=0),
        'f1_score': f1_score(y_true_numeric, y_pred, zero_division=0),
    }

    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true_numeric, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC-ROC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true_numeric, y_pred_proba)
        except ValueError:
            # Handle cases where only one class is present in y_true
            metrics['auc_roc'] = 0.5  # Random classifier baseline

    # Additional medical relevance metrics
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False alarm rate

    return metrics


def get_confusion_matrix_details(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Get detailed confusion matrix breakdown with medical interpretation.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion matrix components and medical interpretation
    """
    if isinstance(y_true[0], str):
        y_true_numeric = np.where(y_true == 'Yes', 1, 0)
    else:
        y_true_numeric = y_true

    tn, fp, fn, tp = confusion_matrix(y_true_numeric, y_pred).ravel()

    return {
        'true_negatives': int(tn),      # Correctly identified healthy patients
        'false_positives': int(fp),     # Healthy patients incorrectly flagged as diseased
        'false_negatives': int(fn),     # Diseased patients missed by the model (CRITICAL)
        'true_positives': int(tp),      # Diseased patients correctly identified
        'total_predictions': int(tn + fp + fn + tp),
        'medical_interpretation': {
            'sensitivity': f"{tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "N/A",  # True positive rate
            'specificity': f"{tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "N/A",  # True negative rate
            'ppv': f"{tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "N/A",          # Positive predictive value
            'npv': f"{tn/(tn+fn)*100:.1f}%" if (tn+fn) > 0 else "N/A",          # Negative predictive value
        }
    }


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Formatted classification report string
    """
    if isinstance(y_true[0], str):
        y_true_numeric = np.where(y_true == 'Yes', 1, 0)
        target_names = ['No Heart Disease', 'Heart Disease']
    else:
        y_true_numeric = y_true
        target_names = ['Class 0', 'Class 1']

    return classification_report(y_true_numeric, y_pred, target_names=target_names)


def compare_models_metrics(models_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Compare multiple models based on their metrics.

    Args:
        models_metrics: Dictionary with model names as keys and metrics dicts as values

    Returns:
        Dictionary with comparison results and rankings
    """
    if not models_metrics:
        return {}

    # Define metric priorities for medical classification
    # Higher weight on recall (sensitivity) for medical safety
    metric_weights = {
        'accuracy': 0.15,
        'precision': 0.20,
        'recall': 0.35,      # Highest weight - critical for medical diagnosis
        'f1_score': 0.20,
        'specificity': 0.10,
    }

    comparison = {}
    rankings = {}

    # Calculate composite scores
    for model_name, metrics in models_metrics.items():
        composite_score = sum(
            metrics.get(metric, 0) * weight
            for metric, weight in metric_weights.items()
        )
        comparison[model_name] = {
            'metrics': metrics,
            'composite_score': composite_score
        }

    # Rank models by composite score
    sorted_models = sorted(
        comparison.items(),
        key=lambda x: x[1]['composite_score'],
        reverse=True
    )

    rankings = {
        'best_model': sorted_models[0][0],
        'worst_model': sorted_models[-1][0],
        'ranking': [model[0] for model in sorted_models],
        'score_differences': [
            (sorted_models[i][0], sorted_models[i][1]['composite_score'] - sorted_models[i+1][1]['composite_score'])
            for i in range(len(sorted_models) - 1)
        ]
    }

    return {
        'comparison': comparison,
        'rankings': rankings,
        'metric_weights': metric_weights
    }


# ============================================================================
# MODEL SELECTION CRITERIA
# ============================================================================

PROBLEM_DEFINITION = {
    'type': 'Binary Classification',
    'target': 'Heart Disease Status (Yes/No)',
    'business_context': 'Medical diagnosis - high cost of false negatives (missed disease)',
    'primary_metric': 'Recall (Sensitivity) - minimize missed heart disease cases',
    'secondary_metrics': ['F1-Score', 'AUC-ROC', 'Precision'],
    'evaluation_protocol': {
        'cross_validation': 'Stratified 5-fold CV (maintains class balance)',
        'test_set': '20% holdout with stratification',
        'threshold_optimization': 'Consider probability thresholds for medical decision-making',
        'baseline_comparison': 'Compare against naive classifier (predict majority class)'
    }
}

SELECTED_MODELS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'type': 'Linear Model',
        'advantages': [
            'Highly interpretable - coefficients show feature importance',
            'Fast training and prediction',
            'Provides probability estimates',
            'Good baseline for comparison',
            'Works well with scaled features'
        ],
        'limitations': [
            'Assumes linear relationships between features and target',
            'Sensitive to outliers',
            'May underfit complex patterns',
            'Requires feature scaling'
        ],
        'medical_fit': [
            'Interpretability crucial for medical stakeholders',
            'Probability outputs useful for risk stratification',
            'Fast inference suitable for clinical deployment'
        ]
    },

    'random_forest': {
        'name': 'Random Forest',
        'type': 'Tree Ensemble',
        'advantages': [
            'Handles mixed data types (numerical + categorical)',
            'Robust to outliers and missing values',
            'Provides feature importance rankings',
            'Non-parametric - no assumptions about data distribution',
            'Good at capturing non-linear relationships'
        ],
        'limitations': [
            'Less interpretable than linear models',
            'Can be slow to train on large datasets',
            'May overfit if not properly tuned',
            'Requires more memory than linear models'
        ],
        'medical_fit': [
            'Handles complex medical feature interactions',
            'Robust to outliers common in medical data',
            'Feature importance helps identify key risk factors',
            'Ensemble nature provides reliable predictions'
        ]
    },

    'xgboost': {
        'name': 'XGBoost (Extreme Gradient Boosting)',
        'type': 'Boosting Ensemble',
        'advantages': [
            'State-of-the-art performance on tabular data',
            'Handles missing values internally',
            'Built-in feature selection and regularization',
            'Excellent handling of imbalanced datasets',
            'Provides feature importance and SHAP values'
        ],
        'limitations': [
            'Computationally intensive',
            'Requires extensive hyperparameter tuning',
            'Less interpretable than simpler models',
            'Can overfit if not regularized properly'
        ],
        'medical_fit': [
            'High performance critical for medical accuracy',
            'Handles imbalanced medical datasets well',
            'Feature importance helps clinical interpretation',
            'Regularization prevents overfitting on medical data'
        ]
    }
}

MODEL_COMPARISON_PROTOCOL = {
    'evaluation_method': 'Stratified 5-fold cross-validation',
    'primary_metric': 'F1-Score (balance precision/recall)',
    'secondary_metrics': ['Recall', 'Precision', 'AUC-ROC', 'Accuracy'],
    'statistical_test': 'Paired t-test for significance testing',
    'baseline': 'Logistic Regression (interpretable baseline)',
    'hyperparameter_tuning': 'Grid search with 5-fold CV',
    'final_evaluation': 'Test set performance with confidence intervals',
    'medical_validation': 'Clinical expert review of predictions'
}
