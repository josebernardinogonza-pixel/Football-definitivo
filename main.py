from src.fetch_data import fetch_fixtures, fetch_odds
from src.preprocess import clean_and_merge
from src.train import train
from src.predict import load_model, predict
from src.telegram_bot import send_message
import os

def main():
    print("Descargando fixtures y odds reales...")
    fixtures = fetch_fixtures()
    odds = fetch_odds()

    print("Procesando datos reales...")
    df = clean_and_merge(fixtures, odds)

    if not os.path.exists('models/model.pkl'):
        print("Entrenando modelo con datos reales actuales...")
        model = train(df)
    else:
        print("Cargando modelo entrenado...")
        model = load_model()

    print("Generando predicciones...")
    pred_df = predict(model, df.head(10))

    message = "*Predicciones Futuras*\n"
    for _, row in pred_df.iterrows():
        message += f"{row['home_team']} vs {row['away_team']}: "
        message += f"{'Victoria Local' if row['prediction'] == 1 else 'No Victoria Local'} "
        message += f"(Probabilidad: {row['predicted_proba']:.2%})\n"

    print("Enviando predicciones a Telegram...")
    send_message(message)
    print("Predicciones enviadas con éxito.")

if __name__ == "__main__":
    main()
