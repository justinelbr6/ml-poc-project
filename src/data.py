import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any
import os
import sys

# Pour pouvoir importer depuis config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from config import TEST_SIZE, RANDOM_STATE
except ImportError:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

def load_data(file_path: str = "data/Heart_disease_cleveland_new.csv") -> pd.DataFrame:
    """
    Charge les données brutes depuis le fichier CSV.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes s'il y en a. 
    Dans le dataset Cleveland_new, les données sont déjà propres (pas de valeurs nulles).
    """
    df_clean = df.copy()
    
    # Remplacement de potentiels NaN par la médiane pour les numériques
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère de nouvelles features médicales pertinentes pour le dataset Cleveland.
    """
    df_feat = df.copy()
    
    # 1. Risk Factors Count
    # Facteurs de risque: age > 55, chol > 240, trestbps > 140, fbs == 1
    risk_factors = np.zeros(len(df_feat))
    if 'age' in df_feat.columns: risk_factors += (df_feat['age'] > 55).astype(int)
    if 'chol' in df_feat.columns: risk_factors += (df_feat['chol'] > 240).astype(int)
    if 'trestbps' in df_feat.columns: risk_factors += (df_feat['trestbps'] > 140).astype(int)
    if 'fbs' in df_feat.columns: risk_factors += (df_feat['fbs'] == 1).astype(int)
    df_feat['Risk_Factors_Count'] = risk_factors

    # 2. Exercise Index
    # Capacité à l'effort : (Fréquence cardiaque max) - (oldpeak * 10)
    if 'thalach' in df_feat.columns and 'oldpeak' in df_feat.columns:
        df_feat['Exercise_Index'] = df_feat['thalach'] - (df_feat['oldpeak'] * 10)

    # 3. Age Group
    if 'age' in df_feat.columns:
        def categorize_age(age):
            if age < 40: return 'Young'
            elif age < 60: return 'Middle-Aged'
            else: return 'Senior'
        df_feat['Age_Group'] = df_feat['age'].apply(categorize_age)

    return df_feat

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'encodage et le scaling.
    """
    df_trans = df.copy()
    
    # Les variables catégorielles du dataset Cleveland
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'Age_Group']
    cat_cols = [c for c in cat_cols if c in df_trans.columns]
    
    # Les variables continues
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'Risk_Factors_Count', 'Exercise_Index']
    cont_cols = [c for c in cont_cols if c in df_trans.columns]

    # Convertir catégorielles en string pour One-Hot
    for c in cat_cols:
        df_trans[c] = df_trans[c].astype(str)

    # Séparation cible
    target = None
    if 'target' in df_trans.columns:
        target = df_trans['target']
        df_trans = df_trans.drop('target', axis=1)

    # One-Hot Encoding pour catégorielles
    if len(cat_cols) > 0:
        df_trans = pd.get_dummies(df_trans, columns=cat_cols, drop_first=True)

    # Scaling des variables numériques continues
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        df_trans[cont_cols] = scaler.fit_transform(df_trans[cont_cols])

    # Remettre la cible si présente
    if target is not None:
        df_trans['target'] = target

    return df_trans

def load_and_preprocess_data(file_path: str = "data/Heart_disease_cleveland_new.csv"):
    """
    Orchestrateur global. Exécute tout le pipeline de la donnée brute à la donnée prête pour les modèles.
    Retourne X (features) et y (target).
    """
    print("Chargement des données...")
    df = load_data(file_path)
    
    print("Nettoyage des valeurs manquantes...")
    df_clean = clean_data(df)
    
    print("Ingénierie des features (Feature Engineering)...")
    df_feat = create_features(df_clean)
    
    print("Encodage et Normalisation...")
    df_final = transform_data(df_feat)
    
    # Séparation X, y
    if 'target' in df_final.columns:
        y = df_final['target']
        X = df_final.drop('target', axis=1)
    else:
        print("Avertissement : Colonne cible 'target' introuvable.")
        y = None
        X = df_final
        
    print("Pipeline de données terminé avec succès !")
    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data("../data/Heart_disease_cleveland_new.csv")
    if X is not None:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """
    Interface requise par le template (scripts/main.py).
    Retourne X_train, X_test, y_train, y_test.
    """
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "Heart_disease_cleveland_new.csv")
    
    X, y = load_and_preprocess_data(data_path)
    
    if y is None:
        raise ValueError("Erreur lors du chargement: la variable cible n'a pas été trouvée.")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return (X_train, X_test, y_train, y_test)
