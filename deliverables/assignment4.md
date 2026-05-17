# Assignment 4 - Analyse Visuelle et Présentation des Résultats

L'analyse visuelle a été entièrement automatisée et unifiée via le script `scripts/generate_plots.py` qui produit 9 graphiques de référence absolue dans le répertoire central `plots/`. Ces graphiques nourrissent ensuite le dashboard Streamlit de restitution.

## 1. L'Équilibre Clinique (Data Raw)
**Fichier généré :** `plots/01_target_distribution.png`
- **Analyse :** Contrairement aux biais médicaux habituels sur ce type de projet (données souvent à 90% saines), la base de Cleveland est remarquablement équilibrée (54% non atteints, 46% atteints). Cela sécurise les entraînements en évitant à l'algorithme d'apprendre la facilité de "la voie de la majorité".

## 2. Limite Univariée et Barrière Géométrique (Feature Engineering)
**Fichiers générés :** `plots/02_cholesterol_distribution.png` et `plots/05_pca_projection.png`
- **Le Mythe du Cholestérol :** Les distributions de cholestérol des patients sains et malades se chevauchent presque parfaitement, prouvant mathématiquement qu'il est impossible d'établir un diagnostic sur un seul biomarqueur.
- **La Barrière Linéaire (PCA) :** Même en croisant la totalité des variables, le nuage de points PCA montre que les groupes "sains" et "malades" restent lourdement entremêlés au centre. Cela a dicté notre obligation de rejeter la Régression Linéaire simple au profit de modèles complexes (Gradient Boosting).

## 3. Matrice de Corrélation
**Fichier généré :** `plots/03_global_correlation.png`
- **Analyse :** Elle confirme cliniquement que la pathologie est fortement multifactorielle. Aucune variable ne dépasse `+/- 0.55` de corrélation stricte avec la maladie. Nos variables dérivées par feature engineering (ex: `Risk_Factors_Count`) se révèlent statistiquement pertinentes.

## 4. Évaluation & Levée de la "Boîte Noire"
**Fichiers générés :** `plots/06_confusion_matrices.png` et `plots/08_feature_importance.png`
- **Performance (Matrices de confusion) :** Les résultats sont de très haute qualité. La Random Forest atteint `0 Faux Négatifs`. Les architectures ensemblistes démontrent que l'interaction des variables médicales est leur point fort.
- **Explicabilité (Boîte Noire) :** Le barplot des "Feature Importances" du Gradient Boosting rassure les praticiens : l'algorithme a pris ses décisions finales en évaluant massivement la thalassémie (`thal`), le type de douleur thoracique (`cp`) et l'état des artères (`ca`). C'est l'exact reflet du consensus médical en cardiologie, validant que l'IA a compris les fondements du risque cardiaque.

L'ensemble de ce storytelling visuel est à retrouver de manière interactive dans l'application web Streamlit (rendu final).