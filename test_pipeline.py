import os
import pytest
import numpy as np
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

DATA_PATH  = "Churn_Modelling.csv"
MODEL_PATH = "test_classifier.joblib"


# ─────────────────────────────────────────────
# TEST 1 — prepare_data
# ─────────────────────────────────────────────

def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    assert X_train.shape[0] > 0, "X_train vide"
    assert X_test.shape[0]  > 0, "X_test vide"
    assert len(y_train)     > 0, "y_train vide"
    assert len(y_test)      > 0, "y_test vide"
    print("✅ prepare_data OK")


# ─────────────────────────────────────────────
# TEST 2 — train_model
# ─────────────────────────────────────────────

def test_train_model():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    model = train_model(X_train, y_train)
    assert model is not None, "Le modèle est None"
    print("✅ train_model OK")


# ─────────────────────────────────────────────
# TEST 3 — evaluate_model
# ─────────────────────────────────────────────

def test_evaluate_model():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    model    = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert 0.0 <= accuracy <= 1.0, "Accuracy hors limites"
    print("✅ evaluate_model OK")


# ─────────────────────────────────────────────
# TEST 4 — save_model / load_model
# ─────────────────────────────────────────────

def test_save_and_load_model():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    model = train_model(X_train, y_train)
    save_model(model, MODEL_PATH)
    assert os.path.exists(MODEL_PATH), "Fichier joblib non créé"
    loaded_model = load_model(MODEL_PATH)
    assert loaded_model is not None, "Modèle chargé est None"
    # Nettoyage
    os.remove(MODEL_PATH)
    print("✅ save_model / load_model OK")
