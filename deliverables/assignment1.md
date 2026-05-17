# Sujet : Analyse et Prédiction du Risque Cardiovasculaire (Base Cleveland)

Fournir un outil d'aide à la décision pour les professionnels de santé. En identifiant précocement les patients à haut risque, les cliniciens peuvent prescrire des examens préventifs, réduisant ainsi les complications graves.

## 1. Problématique

**Description du projet :**
Ce projet vise à étudier les facteurs de risque liés aux maladies cardiaques à partir d'une véritable base de données clinique (Cleveland) afin de construire un modèle capable d'estimer la probabilité qu'un patient développe une pathologie cardiovasculaire.

## 2. Formulation ML

**Type de problème :** Classification binaire (Apprentissage supervisé)
**Variable cible :** `target` (0 = Patient sain, 1 = Patient malade).
**Objectif :** Prédire si un patient est atteint d'une maladie cardiaque.

**Contexte Machine Learning :**

- **Algorithmes envisagés :** Régression Logistique, Random Forest, Gradient Boosting.
- **Prétraitement :** Normalisation des variables continues, encodage One-Hot des variables catégorielles médicales.

**Objectif d’évaluation :**

- **Recall (Sensibilité) :** Prioritaire pour minimiser les faux négatifs (diagnostics mortels manqués).
- **Accuracy / F1-Score :** Pour évaluer la discrimination globale, puisque le dataset a l'avantage d'être équilibré.

## 3. Choix des données

**Nom du dataset :** `Heart_disease_cleveland_new.csv`
**Source :** Base de données clinique de la fondation Cleveland.
**Type :** Tabulaire (303 entrées et 14 colonnes). Le dataset est bien équilibré (54% de classe 0, 46% de classe 1).

**Description des variables principales :**
- **Démographie & Symptômes :** `age`, `sex`, `cp` (Type de douleur thoracique).
- **Indicateurs cliniques au repos :** `trestbps` (Tension), `chol` (Cholestérol), `fbs` (Glycémie à jeun), `restecg` (ECG).
- **Variables d'effort (Ergométrie) :** `thalach` (Fréq max), `exang` (Angine induite), `oldpeak` (Dépression ST), `slope` (Pente ST).
- **Pathologies lourdes :** `ca` (Vaisseaux bouchés/colorés), `thal` (Thalassémie).

**Hypothèses et limites :**
- **Causalité :** Le modèle capte des intéractions statistiques multivariées mais ne remplace pas l'avis formel d'un cardiologue.
- **Taille de l'échantillon :** 303 patients est suffisant pour des modèles robustes, mais un dataset plus large serait utile pour un déploiement mondial.

## 4. Première collecte et EDA

**Dataset localisé dans :** `data/Heart_disease_cleveland_new.csv`

**Observations clés (issues de l'EDA) :**
- L'absence de corrélation parfaite linéaire entre n'importe quelle variable isolée et la maladie prouve la nature extrêmement complexe de la pathologie.
- L'équilibre des classes est excellent, aucune technique de rééchantillonnage artificielle (type SMOTE) n'est requise.
