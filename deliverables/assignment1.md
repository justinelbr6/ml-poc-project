# Projet : Analyse et Prédiction du Risque Cardiovasculaire

## 1. Description du projet
Ce projet vise à étudier les facteurs de risque liés aux maladies cardiaques (notamment le cholestérol, l'âge, l'IMC et le mode de vie) afin de construire un modèle capable d'estimer la probabilité qu'un individu développe une pathologie cardiovasculaire ou subisse une attaque. L'idée est d'identifier les profils à risque pour permettre une intervention médicale préventive.

## 2. Définition du problème
Il s'agit d'un problème de **classification binaire**.
* **Variable cible :** `Heart Disease Status` (Yes / No).
* **Objectif :** Prédire si un patient est atteint d'une maladie cardiaque en fonction de ses caractéristiques cliniques et comportementales.

## 3. Description du dataset choisi
Le jeu de données utilisé est `heart_disease.csv`. Il contient **10 000 entrées** et **21 colonnes**, mélangeant des données biométriques et comportementales.
* **Source :** Les données proviennent de la plateforme **Kaggle**.

## 4. Description des features disponibles
Le dataset se compose des variables suivantes :
* **Données démographiques :** `Age`, `Gender`.
* **Indicateurs cliniques :** `Blood Pressure`, `Cholesterol Level`, `BMI` (IMC), `Triglyceride Level`, `Fasting Blood Sugar`, `CRP Level`, `Homocysteine Level`.
* **Indicateurs binaires de santé :** `Diabetes`, `High Blood Pressure`, `Low HDL Cholesterol`, `High LDL Cholesterol`.
* **Habitudes de vie :** `Exercise Habits`, `Smoking`, `Alcohol Consumption`, `Stress Level`, `Sleep Hours`, `Sugar Consumption`.
* **Antécédents :** `Family Heart Disease`.

## 5. Premières analyses exploratoires (EDA)
*Un notebook détaillé est disponible dans le dossier `notebooks/`.*
**Observations clés :**
* **Valeurs manquantes :** Présence de données nulles dans `Alcohol Consumption` et `Fasting Blood Sugar`.
* **Distribution :** Analyse de la répartition du cholestérol total par rapport à la variable cible.
* **Corrélations :** Identification des variables les plus corrélées à l'état de santé cardiaque.

## 6. Objectif business
Fournir un outil d'aide à la décision pour les professionnels de santé. En identifiant précocement les patients à haut risque, les cliniciens peuvent prescrire des changements de mode de vie ou des traitements préventifs, réduisant ainsi les complications graves et les coûts de prise en charge.

## 7. Contexte Machine Learning
* **Type d'apprentissage :** Supervisé.
* **Algorithmes envisagés :** Régression Logistique, Random Forest, XGBoost.
* **Prétraitement :** Nettoyage des valeurs manquantes, normalisation des variables numériques et encodage des variables catégorielles.

## 8. Métrique ou fonction de coût envisagée
* **Recall (Sensibilité) :** Prioritaire pour minimiser les faux négatifs (patients malades non détectés).
* **F1-Score :** Pour équilibrer précision et rappel.
* **AUC-ROC :** Pour mesurer la performance globale de séparation des classes.

## 9. Hypothèses, risques et limites identifiées
* **Généralisation :** Les données de Kaggle peuvent présenter des biais de sélection.
* **Causalité :** Le modèle établit des corrélations mais ne remplace pas un diagnostic médical complet.
* **Qualité :** Les données déclaratives sur le mode de vie peuvent manquer de précision.

## 10. Données / Notebooks
* **Obtention :** Téléchargé depuis Kaggle.
* **Localisation :**
    * Dataset : `deliverables/heart_disease.csv`
    * Notebook : `notebooks/assignment1_eda.ipynb`
* **Exécution :** Utiliser un environnement Python avec `pandas`, `seaborn` et `sklearn`. Charger le fichier CSV via le chemin relatif précisé dans le notebook.
