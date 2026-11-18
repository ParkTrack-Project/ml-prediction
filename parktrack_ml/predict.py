"""Batched horizon inference with an explicit information cutoff."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import LOOKBACK_HOURS
from .features_h import FEATURES_H, build_inference_frame, label_rate, utc_timestamp
from .model_h import HorizonModels

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def _result(rate, probs, free_probability, capacity, horizon) -> dict:
    if not np.isfinite(free_probability):
        raise ValueError("No trained availability head; retrain the current model")
    occupied = max(0, min(capacity, int(round(float(rate) * capacity))))
    class_code = int(label_rate([occupied / capacity])[0])
    return {
        "class": CLASS_NAMES[class_code],
        "class_code": class_code,
        "confidence": float(probs[class_code]),
        "all_probabilities": {CLASS_NAMES[k]: float(probs[k]) for k in range(3)},
        "predicted_occupied": occupied,
        "probability_free_space": float(free_probability),
        "capacity": capacity,
        "horizon_model": int(horizon),
        "rate_hat": float(rate),
        "free_target": "hourly_mean_below_rounding_threshold",
    }


def predict_many_for_zone(
    zone_id: int, predicted_for, models: HorizonModels, zone_meta: Dict, profiles: Dict,
    recent_hourly: Optional[pd.DataFrame] = None, as_of=None,
) -> list[dict]:
    targets = [utc_timestamp(t) for t in predicted_for]
    if not targets:
        return []
    cutoff = utc_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if min(targets) < cutoff.floor("h"):
        raise ValueError("Historical predictions require a matching explicit as_of")
    if recent_hourly is None:
        from .data_loader import aggregate_hourly, load_recent_observations
        raw = load_recent_observations(zone_id, cutoff.to_pydatetime(), hours=LOOKBACK_HOURS, as_of=cutoff)
        recent_hourly = aggregate_hourly(raw)

    anchors = sorted({t.floor("h") for t in targets} | {
        t.ceil("h") for t in targets if t != t.floor("h")
    })
    buckets, frame = build_inference_frame(zone_id, anchors, recent_hourly, zone_meta, profiles, cutoff)
    X = frame[FEATURES_H].to_numpy(dtype=float)
    rates = np.empty(len(anchors))
    probabilities = np.empty((len(anchors), 3))
    free = np.empty(len(anchors))
    for H in sorted(set(buckets)):
        mask = buckets == H
        probabilities[mask], rates[mask] = models.predict(int(H), X[mask])
        free[mask] = models.predict_free(int(H), X[mask])
    positions = {t: i for i, t in enumerate(anchors)}
    capacity = int(frame.capacity.iloc[0])
    if capacity <= 0:
        raise ValueError("Zone capacity must be positive")
    result = []
    for t in targets:
        left = positions[t.floor("h")]
        weight = float((t - t.floor("h")) / pd.Timedelta(hours=1))
        right = positions[t.ceil("h")] if weight else left
        rate = (1 - weight) * rates[left] + weight * rates[right]
        probs = (1 - weight) * probabilities[left] + weight * probabilities[right]
        probability = (1 - weight) * free[left] + weight * free[right]
        result.append(_result(rate, probs, probability, capacity, buckets[left]))
    return result


def predict_for_zone(
    zone_id: int, predicted_for: datetime, models: HorizonModels, zone_meta: Dict,
    profiles: Dict, recent_hourly: Optional[pd.DataFrame] = None, as_of=None,
) -> dict:
    return predict_many_for_zone(zone_id, [predicted_for], models, zone_meta, profiles, recent_hourly, as_of)[0]
