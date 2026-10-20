import os
import time
import json
import yaml
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

def train():
    start_time = time.time()
    
    with open("mlops.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_col = config.get("target_column", "species")
    num_features = config.get("numerical_features", [])
    data_path = config.get("dataset_path", "data/iris.csv")

    df = pd.read_csv(data_path)
    X = df[num_features]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    loss = float(log_loss(y_test, y_proba))
    duration = round(time.time() - start_time, 2)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": num_features}, f)

    metrics_data = {
        "task_type": "classification",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "loss": round(loss, 4),
        "training_time_sec": duration,
        "commit_sha": os.getenv("COMMIT_SHA", "demo_commit_v1"),
        "model_artifact_path": "artifacts/model.pkl"
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("Demo training completed successfully. Metrics:", metrics_data)

if __name__ == "__main__":
    train()
