import os
import pickle
import pandas as pd
from typing import Dict, Any

MODEL_BUNDLE = None

def load_bundle():
    global MODEL_BUNDLE
    path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                MODEL_BUNDLE = pickle.load(f)
        except Exception:
            pass

load_bundle()

def predict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        load_bundle()

    if MODEL_BUNDLE is None:
        return {"prediction": 0, "confidence": 0.5, "warning": "Demo model not loaded"}

    model = MODEL_BUNDLE["model"]
    scaler = MODEL_BUNDLE["scaler"]
    features = MODEL_BUNDLE["features"]

    df = pd.DataFrame([input_dict])
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    conf = round(float(max(proba)), 4)

    return {
        "prediction": int(pred),
        "confidence": conf
    }
