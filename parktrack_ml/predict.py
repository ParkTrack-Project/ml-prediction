import numpy as np
from datetime import datetime
from typing import Dict

from .config import CLASS_CENTER_RATES
from .model import CustomLogisticRegression, CustomScaler
from .features import build_prediction_vector

CLASS_NAMES = {0: 'Low', 1: 'Medium', 2: 'High'}


def load_model_and_scaler(model_file: str, scaler_file: str):
    model = CustomLogisticRegression()
    model.load_weights(model_file)
    scaler = CustomScaler()
    scaler.load(scaler_file)
    return model, scaler


def predict_from_vector(
    feature_vector: np.ndarray,
    model:    CustomLogisticRegression,
    scaler:   CustomScaler,
    capacity: int,
) -> dict:
    X_scaled      = scaler.transform(feature_vector.reshape(1, -1))
    class_code    = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    expected_rate = sum(float(probabilities[c]) * CLASS_CENTER_RATES[c] for c in range(3))
    predicted_occupied    = max(0, min(capacity, round(expected_rate * capacity)))
    probability_free_space = 1.0 - float(probabilities[2])

    return {
        'class':                  CLASS_NAMES[class_code],
        'class_code':             class_code,
        'confidence':             float(probabilities[class_code]),
        'all_probabilities':      {
            'Low':    float(probabilities[0]),
            'Medium': float(probabilities[1]),
            'High':   float(probabilities[2]),
        },
        'predicted_occupied':     predicted_occupied,
        'probability_free_space': probability_free_space,
        'capacity':               capacity,
    }


def predict_for_zone(
    zone_id:       int,
    predicted_for: datetime,
    model:         CustomLogisticRegression,
    scaler:        CustomScaler,
    zone_meta:     Dict,
) -> dict:
    from .data_loader import load_recent_observations, aggregate_hourly
    from .weather import get_at as get_weather

    raw = load_recent_observations(zone_id, predicted_for, hours=25)
    recent_hourly = aggregate_hourly(raw) if not raw.empty else raw

    wx = get_weather(zone_id, predicted_for)
    features = build_prediction_vector(zone_id, predicted_for, recent_hourly, zone_meta, weather=wx)

    return predict_from_vector(features, model, scaler, int(zone_meta.get('capacity', 10)))
