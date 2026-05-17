# Assignment 5 : Interface Utilisateur (Streamlit)

Ce document décrit l'application web développée pour la restitution du projet de prédiction cardiovasculaire. L'application est basée sur le framework **Streamlit**.

## 1. Description de l'application Streamlit
**CardioCare AI** est une application web interactive servant de Proof of Concept (POC) pour notre pipeline de Machine Learning. Elle permet de vulgariser notre démarche scientifique auprès d'un jury non-expert tout en fournissant un outil clinique de pointe basé sur la base de données de référence de Cleveland (303 patients). L'interface arbore un design clair, organisé et médicalisé.

## 2. Objectif de l'interface
L'objectif de l'application est multiple et suit le cycle de vie de la donnée :
- **Exploration (Pédagogie) :** Rendre les données cliniques compréhensibles et justifier la nécessité de recourir à l'IA (complexité et non-linéarité).
- **Démonstration & Explicabilité :** Montrer de façon transparente comment les algorithmes évaluent les patients (Feature Importance, Matrices de confusion). L'objectif est de casser la "Boîte Noire".
- **Prédiction Médicale (Live) :** Un cas d'usage interactif simulant un outil d'aide au diagnostic pour un médecin, qui entre les données d'un patient et obtient une estimation instantanée de son risque.

## 3. Structure de l'application
Le code source est modulaire et divisé dans le répertoire `src/`. La navigation s'effectue via un menu latéral multipages :
- **Accueil (`src/app.py`)** : Page d'atterrissage introduisant le POC et le fonctionnement du menu.
- **Page 1 (`1_📊_Exploration_des_donnees.py`)** : Analyse Exploratoire des Données (EDA), distribution des classes (malades vs sains), et matrice de corrélation globale.
- **Page 2 (`2_⚙️_Feature_Engineering.py`)** : Explication de la création de nouvelles variables cliniques (ex: *Risk_Factors_Count*, *Exercise_Index*) et visualisation de l'incapacité d'une séparation linéaire simple (graphique PCA).
- **Page 3 (`3_🤖_Modelisation_et_Performances.py`)** : Comparaison des algorithmes (Régression Logistique, Random Forest, Gradient Boosting). Justification du choix d'optimiser le **Recall** (Rappel) et analyse de l'importance des variables.
- **Page 4 (`4_🩺_Test_Predictif_Patient.py`)** : Le simulateur interactif final embarquant le modèle Gradient Boosting en production.

## 4. Description des inputs utilisateurs
Dans la page **"Test Prédictif Patient"**, l'utilisateur saisit 13 variables cliniques réelles correspondant au standard de la base de Cleveland :
*   **Démographie & Symptômes :** Âge, Sexe, Type de douleur thoracique (4 catégories).
*   **Bilan Clinique au repos :** Tension Systolique (trestbps), Cholestérol (chol), Présence d'une glycémie à jeun anormale (fbs), Résultat de l'ECG au repos.
*   **Test d'effort (Ergométrie) :** Fréquence cardiaque maximale atteinte (thalach), Angine induite par l'effort (exang), Dépression ST induite (oldpeak), Pente du segment ST (slope), Vaisseaux majeurs colorés à la fluoroscopie (ca), Statut de Thalassémie (thal).

## 5. Description des outputs affichés
Une fois le bouton "Évaluer mon Risque" pressé, le pipeline complet (imputation, création des features, encodage One-Hot, standardisation StandardScaler) est appliqué à la volée sur la donnée du patient. 
Le modèle retourne et affiche :
*   **Une jauge/score de probabilité** calculée par l'algorithme (ex: `99.5%`).
*   **Une alerte visuelle colorée** (Vert = Risque Faible, Orange = Risque Modéré, Rouge = Risque Élevé) utilisant du code HTML/CSS embarqué.
*   **Une recommandation médicale ciblée** invitant par exemple le patient à consulter un spécialiste si le seuil critique est dépassé.

## 6. Code / Exécution : Comment lancer l'application
Pour lancer l'application depuis votre environnement de développement :

1. Placez-vous à la racine du projet (`demo-ml-poc-project`).
2. Assurez-vous que l'environnement virtuel contenant les dépendances (Streamlit, Pandas, Scikit-Learn...) est bien activé.
3. Exécutez le script principal d'orchestration qui s'assure que les modèles sont prêts et lance l'interface :
   ```bash
   python scripts/main.py
   ```
4. L'application Streamlit compilera et ouvrira automatiquement une fenêtre dans votre navigateur web par défaut à l'adresse : `http://localhost:8501`.
