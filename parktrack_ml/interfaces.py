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

from .config import MODEL_FILE, ZONE_META_FILE, FEATURE_NAMES
from .model import LGBMWrapper
from .predict import predict_for_zone


@dataclass
class PredictOutput:
    occupancy_class:        Literal["Low", "Medium", "High"]
    class_code:             Literal[0, 1, 2]
    confidence:             float
    prob_low:               float
    prob_medium:            float
    prob_high:              float
    predicted_occupied:     int
    probability_free_space: float
    capacity:               int


_model:     Optional[LGBMWrapper] = None
_zone_meta: Optional[dict]        = None


def _load_once() -> None:
    global _model, _zone_meta
    if _model is not None:
        return

    _model = LGBMWrapper.load(MODEL_FILE)

    saved_features = _model.feature_names or []
    if saved_features and set(saved_features) != set(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature mismatch — model has {len(saved_features)} features, "
            f"config has {len(FEATURE_NAMES)}. Retrain: python -m parktrack_ml.train"
        )

    with open(ZONE_META_FILE) as f:
        _zone_meta = {int(k): v for k, v in json.load(f).items()}


def predict(zone_id: int, predicted_for: datetime, recent_hourly=None) -> PredictOutput:
    """Predict parking occupancy for zone_id at predicted_for (UTC)."""
    _load_once()

    meta = _zone_meta.get(zone_id, {'capacity': 10, 'zone_type_standard': 1})
    raw  = predict_for_zone(zone_id, predicted_for, _model, meta, recent_hourly=recent_hourly)

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


def reload() -> None:
    """Reload model artifacts from disk (call after retraining)."""
    global _model, _zone_meta
    _model = _zone_meta = None
    _load_once()
