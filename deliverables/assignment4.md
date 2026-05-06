# Assignment 4 - Visualisations et Application Streamlit

## 1. Objectif du livrable
Ce document décrit les visualisations implémentées et l'application Streamlit construite pour présenter le projet de classification de maladie cardiaque. Il explique les choix graphiques, l'interprétation des résultats et comment exécuter l'application.

## 2. Application Streamlit
### 2.1 Localisation
- Interface principale : `src/app.py`
- Wrapper top-level : `app.py`
- Point de lancement recommandé : `python scripts/main.py` ou `streamlit run app.py`

### 2.2 Fonctionnalités
- Présentation du projet et du dataset
- Chargement des données brutes depuis `data/heart_disease.csv`
- Chargement ou génération automatique des données transformées dans `data/heart_disease_transformed.csv`
- Affichage du jeu brut et du jeu transformé en aperçu tableau
- Trois sections de visualisation interactives
- Sélections dynamiques pour choisir la vue de données et la métrique de performance

### 2.3 Interaction utilisateur
- Sélecteur de visualisation pour les données brutes
- Sélecteur de feature transformée
- Sélecteur de métrique de performance du modèle
- Interface responsive et adaptée à un affichage large

## 3. Visualisations réalisées

### 3.1 Visualisation des données brutes
- **Graphique** : barres de répartition de la variable cible `Heart Disease Status`
- **Type** : diagramme à barres
- **Objectif** : vérifier l'équilibre de la classe cible avant transformation ou entraînement
- **Interprétation** : le dataset brut est relativement équilibré entre les patients malades et non malades, ce qui réduit le besoin d'une correction spécifique de classe.
- **Pertinence** : essentiel pour décider si un échantillonnage ou une pondération de classes est nécessaire.

### 3.2 Visualisation des données brutes additionnelle
- **Graphique** : nuage de points `Age` vs `Cholesterol Level` coloré par statut de maladie cardiaque
- **Type** : scatter plot
- **Objectif** : observer la relation entre l'âge, le cholestérol et la présence de maladie cardiaque
- **Interprétation** : les patients plus âgés affichent une tendance à des valeurs de cholestérol plus élevées, et la séparation des classes commence à être visible.
- **Pertinence** : donne une première idée des relations cliniques non linéaires présentes dans le jeu brut.

### 3.3 Visualisation des données transformées
- **Graphique** : histogramme groupé de `Risk_Factors_Count` par statut de maladie cardiaque
- **Type** : histogramme
- **Objectif** : montrer l'impact du feature engineering sur l'identification du risque combiné
- **Interprétation** : un nombre élevé de facteurs de risque est corrélé à une probabilité plus forte de maladie cardiaque.
- **Pertinence** : valide le choix d'une feature synthétique qui agrège plusieurs marqueurs cliniques.

### 3.4 Visualisation des données transformées additionnelle
- **Graphique** : histogramme catégoriel de `BMI_Category` par statut de maladie cardiaque
- **Type** : histogramme catégoriel
- **Objectif** : vérifier si les catégories de BMI créées capturent bien le risque
- **Interprétation** : les catégories `Overweight` et `Obese` sont plus fréquentes chez les patients malades.
- **Pertinence** : utile pour relier les signatures médicales à des seuils cliniques.

### 3.5 Visualisation des performances modèles
- **Graphique** : diagramme à barres des scores de performance selon le modèle sélectionné
- **Type** : bar chart
- **Objectif** : comparer les modèles sur des métriques communes telles que `accuracy`, `precision`, `recall`, et `f1_score`
- **Interprétation** : le modèle le plus performant peut être identifié rapidement selon la métrique choisie.
- **Pertinence** : critique pour sélectionner le modèle le plus robuste et cohérent avec l'objectif métier.

### 3.6 Matrice de covariance
- **Graphique** : heatmap de la matrice de covariance des variables numériques
- **Type** : heatmap avec annotation
- **Objectif** : visualiser comment les variables varient ensemble
- **Interprétation** : 
  - Valeurs proches de +1 : relation fortement positive
  - Valeurs proches de -1 : relation fortement négative
  - Valeurs proches de 0 : relation faible ou inexistante
- **Pertinence** : aide à identifier la multicolinéarité et les variables redondantes.

### 3.7 Métriques de qualité du modèle (Équivalent R²)
- **Objectif** : évaluer la qualité des prédictions
- **Métriques** :
  - **Accuracy** : proportion totale de prédictions correctes
  - **Precision** : parmi les prédictions positives, combien sont vraiment positives
  - **Recall (Sensibilité)** : parmi les vrais cas positifs, combien ont été détectés
  - **Spécificité** : parmi les vrais cas négatifs, combien ont été correctement identifiés  
  - **F1-Score** : moyenne harmonique de Precision et Recall
- **Note sur R²** : Pour la classification binaire (comme notre problème de maladie cardiaque), on n'utilise pas R² mais plutôt ces métriques qui sont plus appropriées.
- **Pertinence** : indispensable pour évaluer si le modèle prédit bien, surtout avec données déséquilibrées.

### 3.8 Corrélations conditionnelles (Malade vs Sain)
- **Graphique** : comparaison des corrélations entre variables pour les deux groupes
- **Type** : bar chart groupé
- **Objectif** : identifier les variables qui ont des comportements différents entre patients malades et sains
- **Interprétation** : les grandes différences de corrélation indiquent des variables fortement prédictives
- **Pertinence** : aide à comprendre quelles variables discriminent le mieux les deux groupes.

## 4. Justification des visualisations
- Les diagrammes à barres sont adaptés aux comparaisons de classes et aux métriques de modèle.
- Les nuages de points permettent de détecter des tendances et des séparations potentielles entre classes.
- Les histogrammes des features transformées montrent l'apport du feature engineering à la création d'un signal plus discriminant.
- **La matrice de covariance** révèle la structure de dépendance entre variables et aide à identifier la multicolinéarité.
- **Les métriques de qualité du modèle** (Accuracy, Precision, Recall, F1) remplacent R² pour la classification et fournissent une évaluation multidimensionnelle.
- **Les corrélations conditionnelles** montrent comment les relations entre variables diffèrent selon l'état de santé, révélant les vraies variables prédictives.
- L'ensemble des sept types de visualisation couvre :
  1. L'analyse du dataset brut
  2. L'impact du preprocessing
  3. La structure de covariance
  4. L'évaluation des modèles
  5. Les relations conditionnelles complexes

## 5. Exécution de l'application
### Méthode recommandée
1. Installer les dépendances si nécessaire : `pip install -r requirements.txt`
2. Lancer l'application avec :
   - `python scripts/main.py`
   - ou `streamlit run app.py`

### Description
- `python scripts/main.py` valide le point d'entrée et lance Streamlit sur `src/app.py`
- `streamlit run app.py` exécute le wrapper top-level qui appelle `src.app.build_app()`

## 6. Scripts et notebook associés
- **Script principal** : `src/app.py` contient l'application interactive et les visualisations (basiques + avancées)
- **Wrapper de lancement** : `app.py` au niveau racine pour une exécution directe
- **Notebook d'EDA** : `notebooks/01_EDA_and_Data_Transformation.ipynb` 
  - Sections 1-4 : Chargement, nettoyage, types, EDA basique
  - Section 4.1 : Matrice de covariance
  - Section 4.2 : Confusion matrix et métriques (Accuracy, Precision, Recall, F1)
  - Section 4.3 : Relations approfondies et corrélations conditionnelles
  - Sections 5-10 : Scaling, encoding, engineering, pipelines, sauvegarde

## 7. Données utilisées
- **Données brutes** : `data/heart_disease.csv`
- **Données transformées** : `data/heart_disease_transformed.csv`

## 8. Compléments
- Si `results/model_metrics.csv` existe, l'application affiche les performances réelles des modèles.
- Si le fichier n'est pas présent, l'application charge des métriques mock pour permettre la démonstration.

## 9. Concernant le R² et l'évaluation des modèles

### Pourquoi pas de R² pour la classification ?
Le R² (coefficient de détermination) est une métrique **appropriée pour la régression** (prédiction de valeurs continues). Pour la **classification** (comme le diagnostic de maladie cardiaque), on utilise d'autres métriques :

### Métriques utilisées à la place du R²
1. **Accuracy** : (TP + TN) / Total
   - Exactitude globale du modèle
   - ⚠️ Peut être trompeuse avec données déséquilibrées

2. **Precision** : TP / (TP + FP)
   - Parmi les patients prédits positifs, combien le sont vraiment
   - Important pour éviter les fausses alarmes

3. **Recall (Sensibilité)** : TP / (TP + FN)
   - Parmi les vrais malades, combien ont été détectés
   - Crucial en médecine : on veut minimiser les cas manqués

4. **F1-Score** : 2 × (Precision × Recall) / (Precision + Recall)
   - Équilibre entre Precision et Recall
   - Méthode recommandée pour données déséquilibrées

### Interprétation avec la confusion matrix
```
                 Prédiction
                 Oui    Non
Réalité Oui     TP     FN  ← Cas manqués
        Non     FP     TN  ← Fausses alarmes
```

**Pour notre projet** : 
- **FN (Faux Négatifs)** = malades pas détectés ❌ TRÈS GRAVE en médecine
- **FP (Faux Positifs)** = gens sains traités inutilement ⚠️ moins grave
- **TP (Vrais Positifs)** = malades correctement diagnostiqués ✓
- **TN (Vrais Négatifs)** = sains correctement identifiés ✓
