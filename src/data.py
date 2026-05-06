import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path

# Define data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def load_raw_dataset(filename="heart_disease.csv"):
    """
    Charge le dataset brut.
    
    Args:
        filename (str): Nom du fichier du dataset
        
    Returns:
        pd.DataFrame: Dataset brut
    """
    dataset_path = DATA_DIR / filename
    return pd.read_csv(dataset_path)


def impute_missing_values(df, strategy='median'):
    """
    Traite les valeurs manquantes par imputation médiane pour les variables numériques.
    
    Args:
        df (pd.DataFrame): Dataset avec valeurs manquantes
        strategy (str): Stratégie d'imputation ('median', 'mean')
        
    Returns:
        pd.DataFrame: Dataset avec valeurs imputées
        
    Justification:
    - Median imputation est robuste aux outliers
    - Appropriée pour les données médicales
    - Préservé en tant que stratégie sélectionnée vs KNN imputation
    """
    df_imputed = df.copy()
    
    # Identifier colonnes numériques
    numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
    
    # Imputer valeurs manquantes
    imputer = SimpleImputer(strategy=strategy)
    df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    
    # Remplir valeurs manquantes catégoriques
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_imputed[col] = df_imputed[col].fillna('Unknown')
    
    return df_imputed


def add_engineered_features(df):
    """
    Crée des variables pertinentes basées sur le domaine médical.
    
    Args:
        df (pd.DataFrame): Dataset avec données nettoyées
        
    Returns:
        pd.DataFrame: Dataset avec nouvelles features
        
    Features créées:
    1. Risk_Factors_Count: Somme des facteurs de risque majeurs
    2. BMI_Category: Catégorisation du BMI
    3. Sleep_Quality: Classification de la qualité du sommeil
    4. Lifestyle_Index: Équilibre entre exercice et stress
    5. Total_Cholesterol_Level: Cholestérol total (cholestérol + triglycérides)
    6. Inflammation_Score: Score d'inflammation moyen
    """
    df_eng = df.copy()
    
    # 1. Risk Factors Count
    df_eng['Risk_Factors_Count'] = (
        (df_eng['High Blood Pressure'] == 'Yes').astype(int) +
        (df_eng['Low HDL Cholesterol'] == 'Yes').astype(int) +
        (df_eng['High LDL Cholesterol'] == 'Yes').astype(int) +
        (df_eng['Smoking'] == 'Yes').astype(int) +
        (df_eng['Diabetes'] == 'Yes').astype(int)
    )
    
    # 2. BMI Category
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df_eng['BMI_Category'] = df_eng['BMI'].apply(categorize_bmi)
    
    # 3. Sleep Quality
    def categorize_sleep(hours):
        if hours < 6:
            return 'Poor'
        elif hours < 7:
            return 'Fair'
        elif hours <= 9:
            return 'Good'
        else:
            return 'Excessive'
    
    df_eng['Sleep_Quality'] = df_eng['Sleep Hours'].apply(categorize_sleep)
    
    # 4. Lifestyle Index (Exercise - Stress)
    stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
    exercise_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df_eng['Stress_Level_Num'] = df_eng['Stress Level'].map(stress_map)
    df_eng['Exercise_Level_Num'] = df_eng['Exercise Habits'].map(exercise_map)
    df_eng['Lifestyle_Index'] = df_eng['Exercise_Level_Num'] - df_eng['Stress_Level_Num']
    
    # 5. Total Cholesterol Level
    df_eng['Total_Cholesterol_Level'] = df_eng['Cholesterol Level'] + df_eng['Triglyceride Level']
    
    # 6. Inflammation Score
    df_eng['Inflammation_Score'] = (df_eng['CRP Level'] + df_eng['Homocysteine Level']) / 2
    
    return df_eng


def encode_categorical_features(df):
    """
    Encode les variables catégoriques (Label encoding pour binaires, One-Hot pour multi-class).
    
    Args:
        df (pd.DataFrame): Dataset avec variables catégoriques
        
    Returns:
        tuple: (DataFrame encodé, dict avec encoders pour inverse transform)
        
    Justification de l'approche sélectionnée:
    - Label encoding pour variables binaires (interprétabilité, pas d'expansion dimensionnelle)
    - One-Hot encoding pour variables multi-class (pas d'hypothèse d'ordre)
    """
    df_encoded = df.copy()
    
    # Variables binaires pour Label Encoding
    binary_cats = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes', 
                   'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']
    
    # Variables multi-class pour One-Hot Encoding
    multi_cats = ['Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sugar Consumption']
    
    label_encoders = {}
    
    # Label Encoding for binary
    for col in binary_cats:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    # One-Hot Encoding for multi-class
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cats, drop_first=True)
    
    return df_encoded, label_encoders


def scale_features(df, cols_to_scale=None):
    """
    Normalise les features numériques avec StandardScaler.
    
    Args:
        df (pd.DataFrame): Dataset
        cols_to_scale (list): Colonnes à normaliser (tous numériques si None)
        
    Returns:
        tuple: (DataFrame normalisé, scaler)
        
    Justification:
    - StandardScaler (z-score normalization) sélectionné
    - Appropriate pour algorithmes linéaires et SVM
    - Centre les données autour de 0 avec variance unitaire
    """
    df_scaled = df.copy()
    
    # Determine columns to scale
    if cols_to_scale is None:
        cols_to_scale = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude Risk_Factors_Count (déjà sur une échelle raisonnable)
        if 'Risk_Factors_Count' in cols_to_scale:
            cols_to_scale.remove('Risk_Factors_Count')
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    return df_scaled, scaler


def transform_dataset(input_path=None, output_path=None, verbose=True):
    """
    Pipeline complet de transformation: chargement -> imputation -> engineering -> encoding -> scaling.
    
    Args:
        input_path (str): Chemin du dataset brut
        output_path (str): Chemin de sauvegarde du dataset transformé
        verbose (bool): Affiche les détails de la transformation
        
    Returns:
        pd.DataFrame: Dataset transformé
        
    Étapes:
    1. Load raw data
    2. Impute missing values (Median strategy)
    3. Add engineered features
    4. Encode categorical variables
    5. Scale numerical features
    """
    # Load dataset
    if input_path is None:
        input_path = DATA_DIR / "heart_disease.csv"
    
    df = pd.read_csv(input_path) if isinstance(input_path, (str, Path)) else input_path
    
    if verbose:
        print(f"✓ Raw dataset loaded: {df.shape}")
    
    # Impute missing values
    df = impute_missing_values(df)
    if verbose:
        print(f"✓ Missing values imputed: {df.isnull().sum().sum()} null values remaining")
    
    # Add engineered features
    df = add_engineered_features(df)
    if verbose:
        print(f"✓ Feature engineering completed: {df.shape[1]} features total")
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    if verbose:
        print(f"✓ Categorical encoding completed: {df.shape[1]} features after encoding")
    
    # Scale features
    df, scaler = scale_features(df)
    if verbose:
        print(f"✓ Feature scaling completed")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"✓ Transformed dataset saved to {output_path}")
    
    return df


def load_dataset_split(transformed_dataset_path=None):
    """
    Charge le dataset transformé et le sépare en train/test split avec stratification.
    
    Args:
        transformed_dataset_path (str): Chemin du dataset transformé
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Notes:
    - Utilise stratified split pour maintenir balance des classes
    - Essential pour datasets médicaux déséquilibrés
    """
    if transformed_dataset_path is None:
        transformed_dataset_path = DATA_DIR / "heart_disease_transformed.csv"

    if not Path(transformed_dataset_path).exists():
        raw_path = DATA_DIR / "heart_disease.csv"
        df = pd.read_csv(raw_path)
        df = transform_dataset(df, output_path=transformed_dataset_path, verbose=False)
    else:
        df = pd.read_csv(transformed_dataset_path)

    # Separate features and target
    X = df.drop(columns=["Heart Disease Status"])
    y = df["Heart Disease Status"]

    # Apply stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================================
# PIPELINE ALTERNATIVES (for comparison)
# ============================================================================

def transform_dataset_pipeline1(df):
    """Pipeline 1: All One-Hot Encoding (Tested but not selected)"""
    df = impute_missing_values(df)
    df = add_engineered_features(df)
    df = pd.get_dummies(df, drop_first=True)
    df, _ = scale_features(df)
    return df


def transform_dataset_pipeline3(df):
    """Pipeline 3: Minimal features with engineered features only (Tested but not selected)"""
    df = impute_missing_values(df)
    df = add_engineered_features(df)
    # Keep only essential features
    keep_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                 'Sleep Hours', 'CRP Level', 'Homocysteine Level', 
                 'Triglyceride Level', 'Fasting Blood Sugar',
                 'Risk_Factors_Count', 'Lifestyle_Index', 'Total_Cholesterol_Level', 
                 'Inflammation_Score', 'Heart Disease Status']
    df = df[[col for col in keep_cols if col in df.columns]]
    df, _ = scale_features(df)
    return df