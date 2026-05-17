import os
import sys
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ajout du dossier parent au path pour importer les modules de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_and_preprocess_data
from src.config import MODEL_PARAMS, TEST_SIZE, RANDOM_STATE

def main():
    print("Chargement et préparation des données...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Heart_disease_cleveland_new.csv")
    X, y = load_and_preprocess_data(data_path)
    
    if y is None:
        print("Erreur: Colonne cible introuvable.")
        return

    # Split stratifié
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    # Les noms dans MODEL_PARAMS ne sont pas exactement les mêmes (ex: LogisticRegression vs logistic_regression)
    # Créons une map
    param_map = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForest",
        "gradient_boosting": "GradientBoosting"
    }

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    for file_name, model in models.items():
        param_key = param_map[file_name]
        print(f"\nEntraînement et optimisation de {param_key}...")
        
        grid = GridSearchCV(
            estimator=model, 
            param_grid=MODEL_PARAMS[param_key], 
            cv=5, 
            scoring='recall',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        model_path = os.path.join(models_dir, f"{file_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Modèle sauvegardé avec succès dans : {model_path}")

    print("\nTous les modèles ont été sauvegardés dans le dossier 'models/' !")

if __name__ == "__main__":
    main()
