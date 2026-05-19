"""
from parktrack_ml import predict

result = predict(zone_id=3, predicted_for=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
print(result.occupancy_class)         # "Medium"
print(result.predicted_occupied)      # 5
print(result.probability_free_space)  # 0.73
"""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from .config import MODEL_WEIGHTS_FILE, SCALER_FILE, ZONE_META_FILE, FEATURE_NAMES
from .model import CustomLogisticRegression, CustomScaler
from .predict import predict_for_zone


@dataclass
class PredictOutput:
    occupancy_class:       Literal["Low", "Medium", "High"]
    class_code:            Literal[0, 1, 2]
    confidence:            float
    prob_low:              float
    prob_medium:           float
    prob_high:             float
    predicted_occupied:    int
    probability_free_space: float
    capacity:              int


_model:     Optional[CustomLogisticRegression] = None
_scaler:    Optional[CustomScaler]             = None
_zone_meta: Optional[dict]                     = None


def _load_once():
    global _model, _scaler, _zone_meta
    if _model is not None:
        return

    if set(FEATURE_NAMES) != set(_load_feature_names_from_file()):
        raise RuntimeError("Feature mismatch — retrain the model first (python train.py)")

    _model = CustomLogisticRegression()
    _model.load_weights(MODEL_WEIGHTS_FILE)

    _scaler = CustomScaler()
    _scaler.load(SCALER_FILE)

    with open(ZONE_META_FILE) as f:
        _zone_meta = {int(k): v for k, v in json.load(f).items()}


def _load_feature_names_from_file():
    try:
        with open(MODEL_WEIGHTS_FILE) as f:
            return json.load(f).get('feature_names', FEATURE_NAMES)
    except FileNotFoundError:
        return FEATURE_NAMES


def predict(zone_id: int, predicted_for: datetime) -> PredictOutput:
    """Predict parking occupancy for zone_id at predicted_for (UTC)."""
    _load_once()

    meta = _zone_meta.get(zone_id, {'capacity': 10, 'zone_type_standard': 1})
    raw  = predict_for_zone(zone_id, predicted_for, _model, _scaler, meta)

    return PredictOutput(
        occupancy_class=raw['class'],
        class_code=raw['class_code'],
        confidence=raw['confidence'],
        prob_low=raw['all_probabilities']['Low'],
        prob_medium=raw['all_probabilities']['Medium'],
        prob_high=raw['all_probabilities']['High'],
        predicted_occupied=raw['predicted_occupied'],
        probability_free_space=raw['probability_free_space'],
        capacity=raw['capacity'],
    )


def reload():
    """Reload model artifacts from disk (call after retraining)."""
    global _model, _scaler, _zone_meta
    _model = _scaler = _zone_meta = None
    _load_once()
