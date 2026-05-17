# Assignment 3 - Machine Learning Problem Definition and Model Selection

## 1. Définition du problème ML

- **Classification binaire supervisée**
- **Variable cible :** `target` (0 = Patient Sain, 1 = Patient Malade)
- **Objectif :** Diagnostiquer informatiquement la maladie cardiaque à partir des variables biomédicales (base clinique de Cleveland).

## 2. Définition des métriques d’évaluation

- **Recall (Sensibilité - Priorité 1) :** Éviter les faux négatifs. En médecine, laisser repartir un patient malade avec un diagnostic "sain" est dramatique. Le Recall pénalise fortement ce cas.
- **Accuracy & F1-Score :** Fortement pertinents ici car l'Exploration des Données (EDA) nous a démontré que le dataset de Cleveland est **naturellement équilibré** (54% / 46%). 

**Insights de l'EDA menant au choix des algorithmes :**
Contrairement à des données fictives, les données réelles sont entremêlées. L'analyse PCA de notre EDA a formellement prouvé une forte imbrication non-linéaire des patients (impossible de séparer avec un simple trait mathématique). Cela a justifié l'usage nécessaire d'architectures sophistiquées ensemblistes (Arbres).

## 3. Modèles sélectionnés

Trois familles d'algorithmes ont été implémentées et comparées via GridSearchCV :

### 3.1 Régression Logistique (Baseline mathématique)
**Avantages :** Extrêmement interprétable pour extraire des probabilités de risque claires.
**Limites :** Incapable, par essence, de capturer la structure fortement non-linéaire (enchevêtrement des données vu via la PCA).

### 3.2 Random Forest (Robustesse par le Bagging)
**Avantages :** Excellent pour la détection pure en milieu complexe. Il a obtenu un **Recall de 100%** sur nos tests indépendants, ce qui est le Saint-Graal en diagnostic critique !
**Limites :** Tendance à générer un peu plus de faux positifs pour sécuriser son recall.

### 3.3 Gradient Boosting (Le Gagnant par le Boosting séquentiel)
**Avantages :** Modèle très puissant qui corrige ses erreurs de manière séquentielle sur les données difficiles. Surtout, il offre une excellente explicabilité clinique : sa "Feature Importance" démontre qu'il se base prioritairement sur l'irrigation (`thal`), la douleur (`cp`) et l'état artériel (`ca`). Il offre l'équilibre diagnostic idéal et c'est lui qui anime l'outil Streamlit de prédiction en production.

## 4. Protocole d’évaluation et Code
L'entraînement automatisé, l'optimisation des hyperparamètres sur la métrique cible *Recall* (avec Cross Validation = 5), et l'enregistrement persistant des modèles au format `.pkl` sont scriptés via la commande d'orchestration suivante :
```bash
python scripts/train_and_save_models.py
```