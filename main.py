import os
import pandas as pd
from src.fetch_data import fetch_fixtures, fetch_odds
from src.preprocess import clean_and_merge
from src.train import train, update_training_data
from src.predict import load_model, predict
from src.telegram_bot import send_message

def main():
    print("Descargando fixtures y odds...")
    fixtures = fetch_fixtures()
    odds = fetch_odds()

    print("Procesando y limpiando datos...")
    df_new = clean_and_merge(fixtures, odds)

    # Directorios para guardar datos
    os.makedirs('data/processed', exist_ok=True)
    training_data_path = 'data/processed/training_data.csv'

    # Acumular datos para entrenamiento histórico
    if os.path.exists(training_data_path):
        df_train = pd.read_csv(training_data_path)
        df_train = pd.concat([df_train, df_new], ignore_index=True).drop_duplicates(subset=['home_team', 'away_team', 'home_goals', 'away_goals'])
    else:
        df_train = df_new

    # Guardar datos combinados para entrenamiento
    df_train.to_csv(training_data_path, index=False)

    # Entrenar o cargar modelo
    model_path_exists = os.path.exists('models/model.pkl')
    if not model_path_exists or df_new.shape[0] > 50:
        print("Entrenando modelo con datos acumulados...")
        model = train(df_train)
    else:
        print("Cargando modelo existente...")
        model = load_model()

    print("Generando predicciones...")
    pred_df = predict(model, df_new)

    # Guardar predicciones
    pred_path = 'data/processed/predictions.csv'
    pred_df.to_csv(pred_path, index=False)

    # Preparar mensaje para Telegram
    msg = "*Predicciones deportivas actuales*\n"
    for _, row in pred_df.head(10).iterrows():
        equipo = row['home_team']
        visitante = row['away_team']
        prob = row['predicted_proba']
        pred = "Victoria local" if row['prediction'] == 1 else "No victoria local"
        msg += f"{equipo} vs {visitante}: {pred} ({prob:.2%})\n"

    print("Enviando predicciones por Telegram...")
    send_message(msg)
    print("Proceso finalizado.")

if __name__ == "__main__":
    main()
