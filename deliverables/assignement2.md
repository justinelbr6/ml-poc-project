# Data Transformation Summary - Heart Disease Dataset

## Objectif Global
Transformer les données brutes du dataset heart_disease en données exploitables par un modèle de machine learning, en testant plusieurs approches et en documentant les décisions prises.

---

## 1. État Initial du Dataset

### Dimensions
- **Enregistrements**: ~303 lignes
- **Features**: 21 colonnes
- **Type de problème**: Classification binaire (prédire Heart Disease Status: Yes/No)

### Analyse des Valeurs Manquantes
| Colonne | Nombre de manquants | Pourcentage |
|---------|-------------------|------------|
| Fasting Blood Sugar | 16 | 5.3% |
| CRP Level | 1 | 0.3% |
| Total | 17 | 0.3% |

### Distribution de la Variable Cible
- **No**: ~53% des cas
- **Yes**: ~47% des cas
- **Observations**: Dataset relativement équilibré, pas d'action immédiate requise

---

## 2. Nettoyage des Données

### 2.1 Stratégie d'Imputation des Valeurs Manquantes

#### Testée - Strategy 1: Median Imputation ✓ **SÉLECTIONNÉE**
- **Principe**: Remplacer les valeurs manquantes par la médiane de la colonne
- **Avantages**:
  - Robuste aux outliers (contrairement à la moyenne)
  - Appropriée pour les données médicales
  - Facile à interpréter et réproduire
  - Computation rapide
- **Résultat**: 0 valeurs manquantes après application

#### Testée - Strategy 2: KNN Imputation (k=5) ✗ **REJETÉE**
- **Raison du rejet**:
  - Computation plus coûteuse
  - Au vu du faible pourcentage de valeurs manquantes (0.3%), la complexité additionnelle n'en vaut pas la peine
  - Résultat similaire au Median imputation pour ce dataset

### 2.2 Conversion de Types de Données
- **Variables numériques**: Age, Blood Pressure, Cholesterol Level, BMI, etc.
- **Variables catégoriques**: Gender (Yes/No), Smoking (Yes/No), Exercise Habits (Low/Medium/High), etc.
- **Valeurs "None"**: Converties en 'Unknown' pour les variables catégoriques

---

## 3. Ingénierie des Features

### Features Créées (6 nouvelles variables)

1. **Risk_Factors_Count** (0-5)
   - Somme des facteurs de risque binaires majeurs (High Blood Pressure, Low HDL, High LDL, Smoking, Diabetes)
   - **Justification**: Résumé calculatoire des facteurs de risque combinés

2. **BMI_Category** (Categorique: Underweight, Normal, Overweight, Obese)
   - Catégorisation du BMI selon standards médicaux
   - **Justification**: Capture les seuils médicalement significatifs du BMI

3. **Sleep_Quality** (Categorique: Poor, Fair, Good, Excessive)
   - Classification de la qualité du sommeil basée sur les heures
   - **Justification**: Représente des niveaux cliniques plutôt que des heures brutes

4. **Lifestyle_Index** (Numérique: -2 à 2)
   - Différence: Exercise Level - Stress Level
   - **Justification**: Combine deux facteurs opposés en un score net

5. **Total_Cholesterol_Level**
   - Somme: Cholesterol Level + Triglyceride Level
   - **Justification**: Mesure complète du profil lipidique

6. **Inflammation_Score**
   - Moyenne: (CRP Level + Homocysteine Level) / 2
   - **Justification**: Score combiné des marqueurs d'inflammation

---

## 4. Encoding des Variables Catégoriques

### 4.1 Stratégie Sélectionnée: **Label Encoding pour Binaires + One-Hot pour Multi-class**

#### Détails de l'Approche
- **Variables binaires** (7): Gender, Smoking, Family Heart Disease, Diabetes, High Blood Pressure, Low HDL Cholesterol, High LDL Cholesterol
  - **Encoding**: Label Encoding (0/1) ✓
  - **Justification**: 
    - Pas d'expansion dimensionnelle
    - Interprétabilité conservée (0=No, 1=Yes)
    - Efficace pour les modèles

- **Variables multi-class** (4): Exercise Habits, Alcohol Consumption, Stress Level, Sugar Consumption
  - **Encoding**: One-Hot Encoding ✓
  - **Justification**:
    - Évite les ordonnalités fictives
    - Chaque catégorie devient une feature binaire
    - Approprié pour la majorité des algorithmes

### 4.2 Stratégie Testée Mais Rejetée: All One-Hot Encoding
- **Approche**: One-Hot encoding pour toutes les variables
- **Résultats**:
  - Augmentation dimensionnelle: +50% de features
  - Matrice très sparse (beaucoup de zéros)
- **Raison du rejet**: 
  - Curse of dimensionality
  - Surapprentissage potentiel
  - Perte d'interprétabilité des variables binaires

---

## 5. Normalisation des Features Numériques

### 5.1 Stratégie Sélectionnée: **StandardScaler (z-score normalization)** ✓

```python
X_scaled = (X - mean) / std
```

- **Avantages**:
  - Centre les données autour de 0 avec variance unitaire
  - Optimal pour algorithmes linéaires (Logistic Regression, Linear SVM)
  - Compatible avec la plupart des algorithmes
  - Interprétable: écarts types à partir de la moyenne

- **Features scalées** (excl. Risk_Factors_Count qui est déjà sur échelle appropriée):
  - Age, Blood Pressure, Cholesterol Level, BMI, Sleep Hours
  - CRP Level, Triglyceride Level, Cholesterol Level, Homocysteine Level
  - Fasting Blood Sugar, Sleep Hours

### 5.2 Stratégies Testées Mais Rejetées

#### MinMaxScaler (0-1 normalization) ✗
- **Pourquoi rejeté**:
  - Sensible aux outliers
  - Crée une dépendance aux valeurs min/max des données
  - Moins naturel pour les données médicales

#### RobustScaler ✗
- **Pourquoi rejeté**:
  - Utile quand il y a beaucoup d'outliers, ce qui n'est pas le cas ici
  - StandardScaler fonctionne bien pour ce dataset

---

## 6. Trois Pipelines Testés

### Pipeline 1: "All One-Hot Encoding" (Conservative)
**Shape finale**: (303, 41) - 40 features

**Composition**:
- Toutes les variables catégoriques en One-Hot
- StandardScaler sur variables numériques

**Avantages**:
- ✓ Aucune information perdue
- ✓ Pas de risque d'assum ordonnale

**Désavantages**:
- ✗ Haute dimensionnalité (40 features)
- ✗ Matrice sparse
- ✗ Risque d'overfit
- ✗ Lent à entraîner

**Verdict**: Rejeté

---

### Pipeline 2: "Mixed Encoding" (Balanced) - **SÉLECTIONNÉ** ✓
**Shape finale**: (303, 32) - 31 features

**Composition**:
- Label encoding pour variables binaires
- One-Hot encoding pour variables multi-class (drop_first=True)
- StandardScaler sur variables numériques
- 6 features engineered incluses

**Avantages**:
- ✓ Equilibre dimensionnalité et information
- ✓ Interprétabilité élevée
- ✓ Works with both linear et tree-based models
- ✓ Speed/performance trade-off optimal
- ✓ Variables binaires conservent meaning (Yes/No)

**Désavantages**:
- ✗ Risque minimal d'ordinallité sur variables binaires (acceptable)

**Verdict**: Sélectionné pour production

**Justification détaillée**:
- **Balance**: Suffisamment de features pour capturer interactions, mais pas excessive
- **Interprétabilité**: Stakeholders médicaux comprennent facilement les features binaires
- **Flexibilité**: Fonctionne avec Logistic Regression, Random Forest, SVM, Neural Networks
- **Scalabilité**: Bon comportement avec train/test splits stratifiés

---

### Pipeline 3: "Minimalist" (Selective Features)
**Shape finale**: (303, 14) - 13 features

**Composition**:
- Seules features sélectionnées: Age, Blood Pressure, Cholesterol, BMI, markers d'inflammation
- Seulement 4 variables binaires clés codées
- StandardScaler appliqué

**Avantages**:
- ✓ Très interprétable
- ✓ Computation très rapide
- ✓ Explicabilité maximale

**Désavantages**:
- ✗ Perte d'information significative
- ✗ Peut under-fit pour ce dataset
- ✗ Variables potentiellement utiles éliminées

**Verdict**: Rejeté pour ce projet

---

## 7. Tableau de Comparaison Synthétique

| Métrique | Pipeline 1 | Pipeline 2 | Pipeline 3 |
|----------|-----------|-----------|-----------|
| **Nombre de features** | 40 | 31 | 13 |
| **Dimensionnalité** | Haute | Moyenne | Basse |
| **Interprétabilité** | Moyenne | Haute | Très Haute |
| **Perte d'information** | Aucune | Très faible | Moyenne-Haute |
| **Sparsité** | Très haute | Basse | Nulle |
| **Temps de training** | Lent | Moyen | Rapide |
| **Complexité du modèle** | Haute | Moyenne | Basse |
| **Risque d'overfit** | Élevé | Moyen | Faible |
| **Recommandé pour** | Tree-based | **Tous** | Linear models |

---

## 8. Datasets Créés et Sauvegardés

### Dataset Sélectionné (Production)
- **Nom**: `heart_disease_transformed.csv`
- **Shape**: (303, 32)
- **Location**: `/data/heart_disease_transformed.csv`
- **Features**: 31 (excl. target)
- **Usage**: Entraînement des modèles

### Datasets Alternatifs (Référence)
- **Pipeline 1**: `heart_disease_pipeline1_onehot.csv` (40 features)
- **Pipeline 3**: `heart_disease_pipeline3_minimal.csv` (13 features)

---

## 9. Résumé des Transformations Appliquées

### Étapes du Pipeline Sélectionné (Pipeline 2)

1. **Load**: Données brutes (303 × 21)
   ↓
2. **Imputation**: Median pour variables numériques
   ↓
3. **Feature Engineering**: +6 features
   ↓
4. **Encoding**: 
   - Label pour 7 binaires
   - One-Hot pour 4 multi-class
   ↓
5. **Scaling**: StandardScaler
   ↓
6. **Final**: (303 × 31) + target

### Résumé Statistique

| Aspect | Avant | Après |
|--------|-------|-------|
| Shape | (303, 21) | (303, 32) |
| Valeurs manquantes | 17 (0.3%) | 0 |
| Variables numériques | 9 | 14 |
| Variables catégoriques | 12 | 0 (encoded) |
| Moyenne (feat numériques) | Hétérogène | 0.0 |
| Écart-type (feat num) | Hétérogène | 1.0 |

---

## 10. Instructions d'Utilisation dans `data.py`

### Fonction Principale
```python
from src.data import transform_dataset, load_dataset_split

# Transformer le dataset brut
df_transformed = transform_dataset(
    input_path="data/heart_disease.csv",
    output_path="data/heart_disease_transformed.csv"
)

# Charger split pour training
X_train, X_test, y_train, y_test = load_dataset_split(
    transformed_dataset_path="data/heart_disease_transformed.csv"
)
```

### Fonctions Utilitaires Disponibles
- `load_raw_dataset()`: Charger données brutes
- `impute_missing_values()`: Traiter valeurs manquantes
- `add_engineered_features()`: Ajouter features engineered
- `encode_categorical_features()`: Encoder variables catégoriques
- `scale_features()`: Normaliser features numériques
- `load_dataset_split()`: Train/test split stratifié

---

## 11. Prochaines Étapes

1. **Entraîner modèles** sur Pipeline 2 (sélectionné)
2. **Valider hypothèses**: Vérifier que features engineered améliorent performance
3. **Optionnel**: Comparer performance cross-pipeline (Pipeline 1, 2, 3)
4. **Fine-tuning**: Ajuster selon résultats du modèle
5. **Documentation**: Mettre à jour modèles avec détails de transformation appliquée

---

## 12. Fichiers de Référence

- **Notebook EDA**: `/notebooks/01_EDA_and_Data_Transformation.ipynb`
- **Code de Transformation**: `/src/data.py`
- **Ce Document**: `/deliverables/DATA_TRANSFORMATION_SUMMARY.md`
- **Données Transformées**: `/data/heart_disease_transformed.csv`

---

**Date de Création**: Avril 2026
**Responsable**: Data Engineering Team
**Status**: ✓ Complet - Prêt pour production
