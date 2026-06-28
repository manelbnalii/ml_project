# modification test CI/CD
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 1):
    df = pd.read_csv(filepath)
    print(
        f"[prepare_data] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes"
    )

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_size = X_train_scaled.shape[0]
    test_size = X_test_scaled.shape[0]
    print(f"[prepare_data] Train : {train_size} | Test : {test_size}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîner le modèle avec suivi MLflow
    """
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("customer_churn")

    with mlflow.start_run():
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Log des hyperparamètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Log des métriques
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", accuracy)

        # Log du modèle
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"[train_model] Modèle entraîné avec {n_estimators} arbres.")
        print(f"[train_model] Train accuracy : {accuracy * 100:.2f}%")

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {accuracy * 100:.2f}%")
    print("\n[evaluate_model] Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=["Not Exited", "Exited"]))

    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Not Exited", "Exited"]
    )

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    print("[evaluate_model] Matrice sauvegardée → confusion_matrix.png")

    return accuracy


def save_model(model, filepath: str = "classifier.joblib"):
    joblib.dump(model, filepath)
    print(f"[save_model] Modèle sauvegardé → {filepath}")


def load_model(filepath: str = "classifier.joblib"):
    model = joblib.load(filepath)
    print(f"[load_model] Modèle chargé depuis → {filepath}")
    return model
