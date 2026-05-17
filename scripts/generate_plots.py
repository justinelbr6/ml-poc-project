import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_and_preprocess_data, load_dataset_split

os.makedirs('plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load Data
df = pd.read_csv('data/Heart_disease_cleveland_new.csv')

# Plot 1: Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='pastel')
plt.title('Target Distribution (0: No Disease, 1: Disease)')
plt.savefig('plots/01_target_distribution.png')
plt.close()

# Plot 2: Cholesterol Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='chol', hue='target', kde=True, palette='pastel', element="step")
plt.title('Cholesterol Distribution by Target')
plt.savefig('plots/02_cholesterol_distribution.png')
plt.close()

# Plot 3: Global Correlation
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Global Feature Correlation')
plt.savefig('plots/03_global_correlation.png')
plt.close()

# Plot 4: Target Correlation
plt.figure(figsize=(8, 6))
target_corr = corr['target'].drop('target').sort_values()
target_corr.plot(kind='barh', color=sns.color_palette("pastel")[0])
plt.title('Correlation with Target')
plt.savefig('plots/04_target_correlation.png')
plt.close()

# Prepare transformed data
X, y = load_and_preprocess_data("data/Heart_disease_cleveland_new.csv")

# Plot 5: PCA Projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='pastel', alpha=0.7)
plt.title('PCA Projection (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('plots/05_pca_projection.png')
plt.close()

# Model Evaluations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
models = {
    'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
    'Random Forest': joblib.load('models/random_forest.pkl'),
    'Gradient Boosting': joblib.load('models/gradient_boosting.pkl')
}

X_train, X_test, y_train, y_test = load_dataset_split()

# Plot 6: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name}')
plt.suptitle('Confusion Matrices')
plt.savefig('plots/06_confusion_matrices.png')
plt.close()

# Plot 7: ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/07_roc_curves.png')
plt.close()

# Plot 8: Feature Importance
model_gb = models['Gradient Boosting']
if hasattr(model_gb, "feature_importances_"):
    importances = model_gb.feature_importances_
    feat_names = X.columns
    indices = np.argsort(importances)[::-1][:15] # top 15
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align='center', color=sns.color_palette("pastel")[1])
    plt.yticks(range(len(indices)), feat_names[indices][::-1])
    plt.title('Gradient Boosting - Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('plots/08_feature_importance.png')
    plt.close()

# Plot 9: Precision-Recall Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, lw=2, label=f'{name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.savefig('plots/09_precision_recall_curves.png')
plt.close()

print("All plots generated successfully!")
