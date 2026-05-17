# Assignment 2 - Data Preparation and Transformation

## Objectif
Ce document décrit la préparation des données pour le projet de prédiction des maladies cardiaques (Dataset Clinique de Cleveland).

## Data Cleaning

Le dataset de Cleveland utilisé a déjà été pré-nettoyé en amont et s'avère de très haute qualité (pas de valeurs aberrantes outrancières ou de `NaN` à la volée). Toutefois, par précaution algorithmique et sécurité de mise en production :
- Un module d'imputation par la médiane est intégré pour les variables numériques continues en cas de données manquantes lors de tests de futurs patients.

## Feature Engineering

### Nouvelles features créées (Variables Cliniques Dérivées)

1. **`Risk_Factors_Count`** : Compteur de facteurs de risque avérés (âge > 55, cholestérol > 240, tension > 140, fbs > 0).
  - **Impact :** Mesure consolidée du "terrain à risque" du patient. Les cardiologues résonnent souvent en accumulation de comorbidités à risque exponentiel.
2. **`Exercise_Index`** : Rapport calculé via `thalach` - (`oldpeak` * 10).
  - **Impact :** Évalue la "robustesse cardiaque" globale lors du test d'effort, en mixant le rythme maximum et l'anomalie électrique de récupération.
3. **`Age_Group`** : Discrétisation de l'âge (Jeune, Âge Moyen, Senior).
  - **Impact :** Lisse les effets biologiques liés au vieillissement naturel et à certains seuils hormonaux.

## Transformations appliquées

### Scaling (Normalisation)
- L'algorithme `StandardScaler` (z-score) est appliqué aux variables numériques continues (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `Risk_Factors_Count`, `Exercise_Index`).
- **Justification :** Obligatoire pour garantir la convergence mathématique d'algorithmes basés sur la distance et pour harmoniser les poids dans la Régression Logistique.

### Encoding (Variables catégoriques)
- L'algorithme `One-Hot Encoding` (pd.get_dummies) est appliqué aux variables catégorielles (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, `Age_Group`).
- **Justification cruciale :** Bien que les données soient numérisées à l'origine (0, 1, 2, 3), certaines variables cliniques comme `cp` (type de douleur) ou `thal` n'ont *aucun ordre mathématique de grandeur* (la douleur de type 3 n'est pas "trois fois pire" que la douleur de type 1, elle est juste de nature différente). Le One-Hot Encoding est donc l'unique solution mathématiquement valide pour ne pas tromper l'IA.

### Organisation du code (`src/data.py`)
Le pipeline Python industriel orchestre le flux de données de bout en bout et de manière reproductible via les fonctions chaînées : `load_data()`, `clean_data()`, `create_features()`, et `transform_data()`.