# Assignment 2 - Data Preparation and Transformation

## 1. Objectif du document
Ce document décrit la préparation des données pour le projet de classification de maladie cardiaque. Il se concentre sur le nettoyage, la transformation et l'ingénierie des features, sans répéter les choix de modèles documentés dans `assignment3.md`.

## 2. Données et contexte
- **Source** : `data/heart_disease.csv`
- **Format** : CSV
- **Dimensions** : 303 lignes × 21 colonnes
- **Cible** : `Heart Disease Status` (binaire)
- **Jeu transformé final** : `data/heart_disease_transformed.csv`

## 3. Nettoyage des données
### 3.1 Analyse initiale
- Vérification des types, distribution et valeurs manquantes.
- Dataset globalement équilibré : ~53 % `No`, ~47 % `Yes`.

### 3.2 Traitement des valeurs manquantes
- Stratégie retenue : imputation médiane sur les colonnes numériques.
- Justification : robustesse aux outliers, simplicité et rapidité.
- Colonnes concernées : `Fasting Blood Sugar`, `CRP Level`, et quelques valeurs isolées.
- Résultat : plus aucune valeur manquante dans le jeu final.

### 3.3 Conversion de types
- Variables numériques conservées comme `int`/`float`.
- Variables catégoriques maintenues en type texte avant encodage.
- Valeurs manquantes catégoriques normalisées en `Unknown` pour les transformations ultérieures.

## 4. Transformations appliquées
### 4.1 Encodage des variables catégoriques
- Variables binaires : `Label Encoding` (`0/1`) pour `Gender`, `Smoking`, `Family Heart Disease`, `Diabetes`, `High Blood Pressure`, `Low HDL Cholesterol`, `High LDL Cholesterol`.
- Variables multi-classes : `One-Hot Encoding` pour `Exercise Habits`, `Alcohol Consumption`, `Stress Level`, `Sugar Consumption`.
- Justification : mixte efficace pour limiter la dimensionnalité tout en évitant les ordonnalités fictives.

### 4.2 Normalisation
- Méthode retenue : `StandardScaler` (z-score) sur les variables numériques.
- Raisons : centre les variables, homogénéise les échelles, compatible avec les modèles linéaires et la plupart des algorithmes.
- Alternatives testées mais rejetées : `MinMaxScaler` (sensibilité aux outliers), `RobustScaler` (distribution pas assez extrême).

### 4.3 Pipeline de transformation final
- Nettoyage des valeurs manquantes
- Génération des nouvelles features
- Encodage mixte des variables catégoriques
- Normalisation des variables numériques
- Export du dataset final

## 5. Ingénierie des features
### 5.1 Nouvelles features créées
1. `Risk_Factors_Count`
   - Somme des indicateurs binaires de risque : `High Blood Pressure`, `Low HDL`, `High LDL`, `Smoking`, `Diabetes`.
   - Impact : mesure consolidée du profil de risque.

2. `BMI_Category`
   - Catégorisation du BMI en `Underweight`, `Normal`, `Overweight`, `Obese`.
   - Impact : capture les seuils médicaux significatifs.

3. `Sleep_Quality`
   - Catégorisation du sommeil en `Poor`, `Fair`, `Good`, `Excessive`.
   - Impact : transforme la variable continue en signal comportemental utile.

4. `Lifestyle_Index`
   - Score calculé comme `Exercise Level - Stress Level`.
   - Impact : synthèse du style de vie.

5. `Total_Cholesterol_Level`
   - Somme de `Cholesterol Level` et `Triglyceride Level`.
   - Impact : profil lipidique plus riche.

6. `Inflammation_Score`
   - Moyenne de `CRP Level` et `Homocysteine Level`.
   - Impact : score combiné des marqueurs inflammatoires.

### 5.2 Raisons des choix
- Les nouvelles features ajoutent du signal médical pertinent.
- Elles réduisent la nécessité pour le modèle de recomposer ces relations à partir des variables brutes.
- Elles équilibrent interprétabilité et puissance prédictive.

## 6. Alternatives évaluées et rejetées
### 6.1 Imputation
- `KNN Imputation` : rejeté pour sa complexité et son coût de calcul, sans gain significatif.

### 6.2 Encodage
- `All One-Hot Encoding` : rejeté en raison de la hausse de dimensionnalité et du surapprentissage potentiel.

### 6.3 Scaling
- `MinMaxScaler` : rejeté car trop sensible aux outliers.
- `RobustScaler` : rejeté car non nécessaire pour cette distribution de données.

### 6.4 Features minimales
- Pipeline trop réduit : rejeté car perte d’information potentielle et risque de sous-apprentissage.

## 7. Impact attendu sur les modèles
- Meilleure qualité de données : moins de bruit, cohérence renforcée.
- Convergence plus stable : variables normalisées et bien calibrées.
- Signal plus riche : features médicales synthétiques renforcent la capacité prédictive.
- Interprétabilité maintenue : choix d’encodage clair pour les variables binaires.
- Dimensionnalité contrôlée : suffisamment de variables pour capter des interactions sans créer un modèle trop large.

## 8. Documents et ressources associés
- Notebook principal : `notebooks/01_EDA_and_Data_Transformation.ipynb`
- Jeu transformé utilisé en production : `data/heart_disease_transformed.csv`
- Fonction de transformation : `src/data.py`

## 9. Remarque
Ce document se concentre sur le travail de préparation des données. Les décisions de sélection et d’évaluation des modèles sont traitées séparément dans `assignment3.md`.
