# ─────────────────────────────
# Makefile — Customer Churn Prediction
# ─────────────────────────────

PYTHON     = python3
PIP        = pip
MAIN       = main.py
SRC        = model_pipeline.py main.py
MODEL_PATH = classifier.joblib
DATA_PATH  = Churn_Modelling.csv

# ─────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────

all: install format lint security test prepare train evaluate
	@echo "✅ PIPELINE COMPLET TERMINÉ"

# ─────────────────────────────
# INSTALLATION
# ─────────────────────────────

install:
	@echo "📦 Installation dépendances..."
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 bandit pytest
	@echo "OK"

# ─────────────────────────────
# FORMATAGE
# ─────────────────────────────

format:
	@echo "🎨 Formatage..."
	black $(SRC)
	@echo "OK"

# ─────────────────────────────
# LINT
# ─────────────────────────────

lint:
	@echo "🔍 Lint..."
	flake8 $(SRC) --max-line-length=88 --ignore=E203,W503
	@echo "OK"

# ─────────────────────────────
# SÉCURITÉ
# ─────────────────────────────

security:
	@echo "🔒 Security check..."
	bandit model_pipeline.py main.py -ll
	@echo "OK"

# ─────────────────────────────
# DONNÉES
# ─────────────────────────────

prepare:
	$(PYTHON) $(MAIN) --step prepare

# ─────────────────────────────
# TRAIN
# ─────────────────────────────

train:
	$(PYTHON) $(MAIN) --step train

# ─────────────────────────────
# EVALUATION
# ─────────────────────────────

evaluate:
	$(PYTHON) $(MAIN) --step evaluate

# ─────────────────────────────
# TESTS
# ─────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ─────────────────────────────
# NETTOYAGE
# ─────────────────────────────

clean:
	@echo "🧹 Clean..."
	rm -f $(MODEL_PATH)
	rm -f confusion_matrix.png
	rm -rf __pycache__
	rm -rf .pytest_cache
	@echo "OK"

# ─────────────────────────────
# API FASTAPI (IMPORTANT POUR PROCHAIN ATELIER)
# ─────────────────────────────

api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# ─────────────────────────────
# MLFLOW UI
# ─────────────────────────────
mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "✅ MLflow disponible sur http://127.0.0.1:5000"


# ─────────────────────────────
# DOCKER
# ─────────────────────────────

IMAGE_NAME = manel-bnnali-classe-mlops
DOCKER_USER = manelbenali

docker-build:
	@echo "🐳 Construction de l'image Docker..."
	docker build -t $(IMAGE_NAME) .
	@echo "✅ Image construite."

docker-run:
	@echo "🚀 Lancement du conteneur..."
	docker run -d -p 8000:8000 --name churn-api $(IMAGE_NAME)
	@echo "✅ API disponible sur http://localhost:8000"

docker-push:
	@echo "📤 Push de l'image sur Docker Hub..."
	docker tag $(IMAGE_NAME):latest $(DOCKER_USER)/$(IMAGE_NAME):latest
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest
	@echo "✅ Image pushée sur Docker Hub."

docker-stop:
	@echo "🛑 Arrêt du conteneur..."
	docker stop churn-api
	docker rm churn-api
	@echo "✅ Conteneur arrêté."
# ─────────────────────────────
# HELP
# ─────────────────────────────

help:
	@echo "make install"
	@echo "make format"
	@echo "make lint"
	@echo "make security"
	@echo "make prepare"
	@echo "make train"
	@echo "make evaluate"
	@echo "make test"
	@echo "make api"
	@echo "make clean"
	@echo "make all"

.PHONY: all install format lint security prepare train evaluate test clean api help
