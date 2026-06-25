"""
app.py
API REST pour la prédiction du churn client via FastAPI.

Routes :
    GET  /          → message de bienvenue
    POST /predict   → prédiction churn pour un client
    POST /retrain   → réentraîner le modèle (excellence)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from model_pipeline import prepare_data, train_model, save_model

# ─────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API REST pour prédire si un client va quitter la banque.",
    version="1.0.0",
)

MODEL_PATH = "classifier.joblib"
DATA_PATH  = "Churn_Modelling.csv"

# Chargement du modèle au démarrage
try:
    model = joblib.load(MODEL_PATH)
    print(f"[app] Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[app] Erreur chargement modèle : {e}")


# ─────────────────────────────────────────────
# SCHÉMA D'ENTRÉE — données d'un client
# ─────────────────────────────────────────────

class ClientData(BaseModel):
    CreditScore: float
    Gender: int            # 0 = Female, 1 = Male
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int         # 0 ou 1
    IsActiveMember: int    # 0 ou 1
    EstimatedSalary: float

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Gender": 1,
                "Age": 35,
                "Tenure": 5,
                "Balance": 75000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
            }
        }


# ─────────────────────────────────────────────
# SCHÉMA D'ENTRÉE — paramètres retrain
# ─────────────────────────────────────────────

class RetrainParams(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

    class Config:
        json_schema_extra = {
            "example": {
                "n_estimators": 200,
                "random_state": 0,
            }
        }


# ─────────────────────────────────────────────
# ROUTE 1 — Accueil
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API Customer Churn Prediction !",
        "routes": {
            "predict": "POST /predict",
            "retrain": "POST /retrain",
            "docs":    "GET  /docs",
        },
    }


# ─────────────────────────────────────────────
# ROUTE 2 — Prédiction
# ─────────────────────────────────────────────

@app.post("/predict")
def predict(client: ClientData):
    """
    Prédit si un client va quitter la banque.

    - **0** → Le client reste ✅
    - **1** → Le client part ⚠️
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modèle non chargé. Lancez d'abord make train.",
        )

    try:
        features = np.array([[
            client.CreditScore,
            client.Gender,
            client.Age,
            client.Tenure,
            client.Balance,
            client.NumOfProducts,
            client.HasCrCard,
            client.IsActiveMember,
            client.EstimatedSalary,
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return {
            "prediction": int(prediction),
            "result": "⚠️ Churn : le client va quitter" if prediction == 1 else "✅ Le client reste",
            "probabilite_churn": round(float(probability[1]) * 100, 2),
            "probabilite_reste": round(float(probability[0]) * 100, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────
# ROUTE 3 — Retrain (Excellence)
# ─────────────────────────────────────────────

@app.post("/retrain")
def retrain(params: RetrainParams):
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres.

    - **n_estimators** : nombre d'arbres (défaut 100)
    - **random_state** : graine aléatoire (défaut 42)
    """
    global model

    try:
        X_train, X_test, y_train, y_test, scaler = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train, params.n_estimators, params.random_state)
        save_model(model, MODEL_PATH)

        return {
            "message": "✅ Modèle réentraîné et sauvegardé avec succès !",
            "parametres": {
                "n_estimators": params.n_estimators,
                "random_state": params.random_state,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
