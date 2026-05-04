# Assignment 3 - Machine Learning Problem Definition and Model Selection

## 1. Définition du problème ML

### Type de problème
- **Classification binaire supervisée**
- **Variable cible :** `Heart Disease Status` (Yes / No)
- **Objectif du modèle :** prédire la présence ou l'absence d'une maladie cardiaque à partir de variables cliniques, biométriques et comportementales.

### Contexte du problème
- Le problème vise à soutenir une décision médicale préventive.
- Les faux négatifs (patients malades non détectés) sont très coûteux.
- Les faux positifs génèrent des examens supplémentaires mais restent moins critiques que les faux négatifs.

## 2. Définition de la métrique d’évaluation

### Métrique prioritaire
- **Recall (sensibilité)**
- **Justification :** Dans un contexte médical, il est prioritaire de détecter le plus grand nombre possible de cas positifs réels pour réduire le risque de patients malades non traités.

### Métriques secondaires
- **F1-Score** : équilibre entre précision et rappel.
- **AUC-ROC** : capacité du modèle à discriminer les classes.
- **Precision** : limite les faux positifs inutiles.
- **Specificity** : évalue la capacité à reconnaître les patients sains.

### Mesures complémentaires
- **False Negative Rate** : indicateur critique du risque de manquer des patients malades.
- **False Positive Rate** : utile pour mesurer le coût des erreurs.

## 3. Protocole d’évaluation

### Split et validation
- **Train/Test split** : 80 % entraînement / 20 % test
- **Stratification** : effectuée sur `Heart Disease Status` pour conserver la proportion de classes dans chaque sous-ensemble.
- **Validation croisée** : Stratified 5-fold cross-validation pour stabiliser l’évaluation et éviter le surapprentissage.

### Tuning et sélection
- **Hyperparameter tuning** : grid search ou recherche par grille avec validation croisée
- **Baselines** : régression logistique comme référence interprétable
- **Final evaluation** : performance sur jeu de test indépendant avec intervalles de confiance si possible
- **Comparaison** : comparer les modèles sur les mêmes métriques et la même procédure d’évaluation

### Protocole recommandé
1. Prétraiter les données sur l’ensemble d’entraînement uniquement.
2. Faire la validation croisée stratifiée sur l’entraînement.
3. Sélectionner les meilleurs hyperparamètres.
4. Évaluer le modèle final sur le jeu de test.
5. Comparer les modèles grâce à un score composite pondéré (recall prioritaire).

## 4. Présentation des trois modèles sélectionnés

### 4.1 Logistic Regression

#### Hypothèses principales
- La relation entre les variables et la probabilité de maladie est logistique et approximativement linéaire dans l’espace transformé.
- Les variables pertinentes sont correctement sélectionnées ou prétraitées.

#### Avantages attendus
- Très interprétable : coefficients faciles à comprendre.
- Rapide à entraîner et à déployer.
- Fournit des probabilités de risque utiles en contexte médical.
- Bon baseline pour comparaison.

#### Limites attendues
- Peut sous-estimer des relations non linéaires.
- Sensible aux outliers.
- Nécessite un bon prétraitement des variables numériques.

#### Adéquation avec le problème et la métrique
- **Recall** : bon pour identifier une baseline fiable.
- **Interprétabilité** : excellent pour justification clinique.
- **Usage** : utile comme modèle de référence et pour expliquer les prédictions.

### 4.2 Random Forest

#### Hypothèses principales
- Il existe des relations non linéaires et des interactions entre les variables.
- Les erreurs de variance peuvent être réduites par un ensemble d'arbres.

#### Avantages attendus
- Robuste aux outliers et aux variables de types différents.
- Capture les interactions et les non-linéarités.
- Fournit des importances de features.
- Relativement résistant au surapprentissage comparé à un arbre unique.

#### Limites attendues
- Moins interprétable que la régression logistique.
- Plus gourmand en mémoire et en temps d’entraînement.
- Peut devenir plus lent avec un grand nombre d'arbres ou de features.

#### Adéquation avec le problème et la métrique
- **Recall** : généralement bon pour identifier les positives réelles grâce aux arbres.
- **F1-Score** : favorable dans les datasets médicaux avec relations complexes.
- **Adaptabilité** : adapté aux données tabulaires et mixtes du dataset.

### 4.3 XGBoost

#### Hypothèses principales
- Le boosting graduel améliore la performance en corrigeant les erreurs successives.
- Les patterns cachés peuvent être capturés par une combinaison d’arbres peu profonds.

#### Avantages attendus
- Excellente performance sur données tabulaires.
- Gestion native de certaines valeurs manquantes.
- Regularisation intégrée pour réduire l’overfitting.
- Bon équilibre entre précision et rappel.

#### Limites attendues
- Hyperparamétrage plus complexe.
- Entraînement plus lent et plus coûteux en calcul.
- Moins facile à interpréter que les modèles simples.

#### Adéquation avec le problème et la métrique
- **Recall** : peut être optimisé avec un bon tuning et une cible de sensibilité élevée.
- **AUC-ROC** : modèle souvent performant pour discriminer la maladie.
- **Robustesse** : bon choix pour capturer les signaux complexes médicamenteux.

## 5. Justification du choix des trois modèles

### Raisons générales
- **Diversité algorithmique** : couverture d’un modèle linéaire, d’un modèle basé sur des arbres et d’un modèle boosté.
- **Interprétabilité vs performance** : la régression logistique apporte l’intelligibilité, les autres apportent la puissance.
- **Robustesse et flexibilité** : Random Forest et XGBoost gèrent bien les non-linéarités, les outliers et les interactions complexes.
- **Adéquation médicale** : combinaison prudente entre un modèle explicable et des modèles performants.

### Choix des trois modèles exacts
- **Juste trois modèles** pour rester focalisé, facile à comparer et à expliquer.
- **Logistic Regression** pour le seuil basique et l’interprétabilité.
- **Random Forest** pour la robustesse des ensembles et l’analyse de l’importance des features.
- **XGBoost** pour la meilleure performance attendue sur données tabulaires et la capacité de calibration.

### Conclusion
Ces trois modèles offrent un compromis pertinent pour ce projet :
- un modèle simple et interprétable,
- un modèle robuste et polyvalent,
- un modèle de haute performance optimisé pour les données tabulaires.

## 6. Fichiers .py pertinents

- `src/metrics.py` : définition des métriques d’évaluation et du protocole de comparaison.
- `src/config.py` : liste des trois modèles sélectionnés et chemins de sauvegarde.
- `src/data.py` : pipeline de chargement et transformation des données utilisé pour l’entraînement.

## 7. Notebooks utilisés

### Notebook principal
- `notebooks/01_EDA_and_Data_Transformation.ipynb`

### Utilisation
- Ouvrir le notebook avec Jupyter ou VS Code.
- Exécuter cellule par cellule pour reproduire l’EDA, la préparation des données et la sauvegarde des jeux transformés.
- Le notebook génère les fichiers transformés dans `data/`, dont `heart_disease_transformed.csv`.

### Reproduire les expériences
1. Activer l’environnement Python avec les dépendances listées dans `requirements.txt`.
2. Lancer `jupyter notebook` ou ouvrir le notebook dans VS Code.
3. Exécuter toutes les cellules.
4. Vérifier la présence du dataset transformé dans `data/`.

## 8. Complément

Ce document se concentre uniquement sur les trois modèles exacts demandés et sur la méthodologie de comparaison. Les autres assets du projet (metrics, config, notebooks, transformation) sont alignés sur cette sélection et permettent de reproduire l’analyse et le benchmark.