from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pathlib import Path

MODEL_PATH = Path('models/model.pkl')

def train(df):
    X = pd.get_dummies(df[['home_team', 'away_team']])
    X['home_odds'] = df['home_odds']
    X['draw_odds'] = df['draw_odds']
    X['away_odds'] = df['away_odds']

    y = df['result_home_win']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    print(f"Validación accuracy: {val_score:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model
