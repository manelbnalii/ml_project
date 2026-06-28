# Image de base Python
FROM python:3.12-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet
COPY requirements.txt .
COPY model_pipeline.py .
COPY main.py .
COPY app.py .
COPY classifier.joblib .
COPY Churn_Modelling.csv .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
