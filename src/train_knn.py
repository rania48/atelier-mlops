import os
import yaml
import json
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

with open("params.yaml", "r") as file:
    all_params = yaml.safe_load(file)
split = all_params["train"]
knn_params = all_params["knn"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split["test_size"],
    random_state=split["random_state"], stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

with mlflow.start_run(run_name="KNN_Iris"):
    model = KNeighborsClassifier(n_neighbors=knn_params["n_neighbors"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted")
    }

    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("n_neighbors", knn_params["n_neighbors"])
    mlflow.log_param("test_size", split["test_size"])
    mlflow.log_param("random_state", split["random_state"])
    mlflow.log_metrics(metrics)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/knn.pkl")
    mlflow.log_artifact("models/knn.pkl", artifact_path="model")

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_knn.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metriques KNN :", metrics)
    print("Modele KNN entraine et enregistre.")