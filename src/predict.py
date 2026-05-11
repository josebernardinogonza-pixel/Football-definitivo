import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path('models/model.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def prepare_features(df, model_features):
    X = pd.get_dummies(df[['home_team', 'away_team']])
    X['home_odds'] = df['home_odds']
    X['draw_odds'] = df['draw_odds']
    X['away_odds'] = df['away_odds']

    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]
    return X

def predict(model, df):
    X = prepare_features(df, model.feature_names_in_)
    df['predicted_proba'] = model.predict_proba(X)[:, 1]
    df['prediction'] = model.predict(X)
    return df
