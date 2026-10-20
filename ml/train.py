import sys
import os

# Make sure the project root is on the path regardless of execution directory
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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


def _load_dataset(config: dict) -> pd.DataFrame:
    """
    Loads the dataset declared in mlops.yaml.

    Supports:
      - CSV / any tabular file  (default)
      - Single PCAP file        (dataset_path ends with .pcap or .pcapng)
      - Directory of PCAPs      (dataset_type: pcap_dir  OR  auto-detected)

    For PCAP inputs the function:
      1. Extracts 24 tabular network-flow features per 5-tuple flow using scapy.
      2. Optionally applies a pcap_label_map to assign a label column.
      3. Returns a standard pandas DataFrame — the rest of the pipeline is unchanged.
    """
    data_path = config.get("dataset_path", "data/iris.csv")
    dataset_type = config.get("dataset_type", "csv").lower()
    label_map = config.get("pcap_label_map", {})
    target_col = config.get("target_column", "label")

    # Resolve path relative to repo root
    if not os.path.exists(data_path):
        for prefix in ("demo_repo", "."):
            candidate = os.path.join(prefix, data_path)
            if os.path.exists(candidate):
                data_path = candidate
                break

    # Auto-detect PCAP by extension even when dataset_type not set
    p = Path(data_path)
    if p.suffix.lower() in (".pcap", ".pcapng"):
        dataset_type = "pcap"
    elif p.is_dir() and dataset_type not in ("pcap_dir",):
        # Check if the directory contains PCAP files
        if list(p.glob("**/*.pcap")) or list(p.glob("**/*.pcapng")):
            dataset_type = "pcap_dir"

    if dataset_type in ("pcap", "pcap_dir"):
        try:
            # Lazy import so the module works without scapy when using CSV
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            from ml.pcap_extractor import load_pcap_as_dataframe, load_pcap_directory
        except ImportError:
            raise ImportError(
                "scapy is required for PCAP datasets. "
                "Run: pip install scapy"
            )

        if dataset_type == "pcap_dir":
            df = load_pcap_directory(
                data_path,
                label_map={k: v for k, v in label_map.items()} if label_map else None,
                label_column=target_col,
            )
        else:
            # Single PCAP file: label_map key "" means "all rows get this label"
            lval = label_map.get("all") or label_map.get("") or None
            df = load_pcap_as_dataframe(
                data_path,
                label_column=target_col if lval is not None else None,
                label_value=lval,
            )

        print(f"[AutoMLOps] PCAP extraction: {len(df)} flows, columns: {list(df.columns)}")
        return df

    # Default: CSV (or any pandas-readable tabular file)
    ext = p.suffix.lower()
    if ext in (".parquet",):
        return pd.read_parquet(data_path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(data_path, lines=(ext == ".jsonl"))
    return pd.read_csv(data_path)


def train():
    start_time = time.time()

    yaml_path = "mlops.yaml" if os.path.exists("mlops.yaml") else "demo_repo/mlops.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_col = config.get("target_column", "species")
    num_features = config.get("numerical_features", [])

    # ── Load dataset (CSV or PCAP) ────────────────────────────────────────────
    df = _load_dataset(config)

    # If no numerical_features declared in mlops.yaml, infer from PCAP extractor
    if not num_features:
        from ml.pcap_extractor import PCAP_FEATURE_COLUMNS
        num_features = [c for c in PCAP_FEATURE_COLUMNS if c in df.columns]
        print(f"[AutoMLOps] Auto-detected numerical_features: {num_features}")

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
