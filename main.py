import argparse
import numpy as np

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def run_pipeline():
    print("=" * 50)
    print("   PIPELINE ML — CUSTOMER CHURN PREDICTION")
    print("=" * 50)

    print("\nÉtape 1 : Préparation des données...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)

    print("\nÉtape 2 : Entraînement du modèle...")
    model = train_model(X_train, y_train)

    print("\nÉtape 3 : Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)

    print("\nÉtape 4 : Sauvegarde du modèle...")
    save_model(model, MODEL_PATH)

    print("\nPipeline terminé !")


def run_step(step: str):
    if step == "prepare":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        print("OK")

    elif step == "train":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH)
        print("OK")

    elif step == "evaluate":
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    elif step == "predict":
        model = load_model(MODEL_PATH)

        new_client = np.array([[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]])
        prediction = model.predict(new_client)
        proba = model.predict_proba(new_client)[0]

        print(new_client.tolist()[0])
        print(prediction[0])
        print(proba[1])

    else:
        print("Étape inconnue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default=None)

    args = parser.parse_args()

    if args.step is None:
        run_pipeline()
    else:
        run_step(args.step)
