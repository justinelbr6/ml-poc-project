# Model Storage Directory

This directory contains trained machine learning models for the heart disease prediction project.

## Expected Models

After training, the following model files should be present:

- `logistic_regression.pkl` - Baseline linear model
- `random_forest.pkl` - Ensemble tree model
- `xgboost.pkl` - Gradient boosting model

## Model Training

Models are trained using the transformed dataset located in `../data/heart_disease_transformed.csv`.

Training scripts should be placed in `../scripts/` and use functions from `../src/data.py` for data loading and preprocessing.

## Model Evaluation

All models are evaluated using metrics defined in `../src/metrics.py`, with emphasis on:
- Recall (sensitivity) for medical safety
- F1-Score for balanced performance
- AUC-ROC for discriminative ability