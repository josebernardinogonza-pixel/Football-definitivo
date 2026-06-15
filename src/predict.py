import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path('models/model.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def prepare_features(df, model_features):
    # Validar que el dataframe no esté vacío y tenga las columnas requeridas
    if df.empty:
        return pd.DataFrame(columns=model_features)
    
    if 'home_team' not in df.columns or 'away_team' not in df.columns:
        raise ValueError(f"DataFrame debe contener 'home_team' y 'away_team'. Columnas disponibles: {df.columns.tolist()}")
    
    X = pd.get_dummies(df[['home_team', 'away_team']])
    
    # Agregar columnas de odds si existen
    if 'home_odds' in df.columns:
        X['home_odds'] = df['home_odds']
    if 'draw_odds' in df.columns:
        X['draw_odds'] = df['draw_odds']
    if 'away_odds' in df.columns:
        X['away_odds'] = df['away_odds']

    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]
    return X

def predict(model, df):
    if df.empty:
        print("Advertencia: DataFrame vacío, no hay datos para predecir")
        return df
    
    X = prepare_features(df, model.feature_names_in_)
    df['predicted_proba'] = model.predict_proba(X)[:, 1]
    df['prediction'] = model.predict(X)
    return df
