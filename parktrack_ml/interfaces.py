"""
from parktrack_ml import predict

result = predict(zone_id=3, predicted_for=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
print(result.occupancy_class)         # "Medium"
print(result.predicted_occupied)      # 5
print(result.probability_free_space)  # 0.73
"""
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Literal, Optional, Tuple

from .config import MODEL_V4_FILE
from .model_h import HorizonModels
from .predict import predict_for_zone, predict_many_for_zone


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


_snapshot: Optional[Tuple[HorizonModels, dict]] = None
_lock = RLock()


def _read_snapshot() -> Tuple[HorizonModels, dict]:
    models, meta = HorizonModels.load(MODEL_V4_FILE)
    prepared_meta = {
        "zone_meta": {int(k): v for k, v in meta["zone_meta"].items()},
        "profiles":  {int(k): v for k, v in meta["profiles"].items()},
        "data_until": meta.get("data_until"),
        "trained_at": meta.get("trained_at"),
    }
    return models, prepared_meta


def _load_once() -> Tuple[HorizonModels, dict]:
    global _snapshot
    with _lock:
        if _snapshot is None:
            _snapshot = _read_snapshot()
        return _snapshot


def predict(zone_id: int, predicted_for: datetime, recent_hourly=None,
            as_of: Optional[datetime] = None) -> PredictOutput:
    """Predict parking occupancy for zone_id at predicted_for (UTC)."""
    models, meta = _load_once()
    _validate_as_of(meta, as_of)
    zone_meta = meta["zone_meta"].get(zone_id, {'zone_type_standard': 1})
    raw = predict_for_zone(zone_id, predicted_for, models, zone_meta, meta["profiles"],
                           recent_hourly=recent_hourly, as_of=as_of)

    return _output(raw)


def _output(raw: dict) -> PredictOutput:
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


def _validate_as_of(meta, as_of):
    if as_of is None:
        return
    from .features_h import utc_timestamp
    for field in ("data_until", "trained_at"):
        if meta.get(field) and utc_timestamp(meta[field]) > utc_timestamp(as_of):
            raise ValueError(f"Model {field} is later than the requested as_of")


def predict_many(zone_id: int, predicted_for, recent_hourly=None, as_of=None) -> list[PredictOutput]:
    models, meta = _load_once()
    _validate_as_of(meta, as_of)
    zone_meta = meta["zone_meta"].get(zone_id, {'zone_type_standard': 1})
    rows = predict_many_for_zone(zone_id, predicted_for, models, zone_meta, meta["profiles"],
                                 recent_hourly=recent_hourly, as_of=as_of)
    return [_output(row) for row in rows]


def reload() -> None:
    """Reload model artifacts from disk (call after retraining)."""
    global _snapshot
    with _lock:
        snapshot = _read_snapshot()
        _snapshot = snapshot
