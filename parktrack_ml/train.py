"""
Training pipeline — logistic regression on data from the ParkTrack API.

Usage:
    python -m parktrack_ml.train
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta, timezone

from .config import (
    MODEL_PATH, MODEL_WEIGHTS_FILE, SCALER_FILE, ZONE_META_FILE,
    FEATURE_NAMES, MODEL_PARAMS, TRAIN_DAYS_BACK,
)
from .data_loader import load_observations, load_zone_meta, aggregate_hourly
from .features import build_training_dataset
from .model import CustomLogisticRegression, CustomScaler

logger = logging.getLogger(__name__)
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def train(save: bool = True):
    os.makedirs(MODEL_PATH, exist_ok=True)

    zone_meta_df = load_zone_meta()
    logger.info("Zones: %s", zone_meta_df["zone_id"].tolist())

    to_dt   = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=TRAIN_DAYS_BACK)

    raw_df = load_observations(from_dt=from_dt, to_dt=to_dt)
    logger.info("Observations: %d rows", len(raw_df))

    hourly_df = aggregate_hourly(raw_df)
    logger.info("Hourly: %d records", len(hourly_df))

    try:
        from .weather import load_for_training
        weather_df = load_for_training()
        logger.info("Weather: %d records", len(weather_df))
    except Exception as exc:
        logger.warning("Weather unavailable (%s), using seasonal fallback", exc)
        weather_df = None

    dataset = build_training_dataset(hourly_df, zone_meta_df, weather_df)
    n = len(dataset)
    logger.info("Dataset: %d samples", n)
    for cls, name in CLASS_NAMES.items():
        cnt = (dataset["label"] == cls).sum()
        logger.info("  %s: %d (%.1f%%)", name, cnt, cnt / n * 100)

    X = dataset[FEATURE_NAMES].values.astype(float)
    y = dataset["label"].values.astype(int)

    scaler = CustomScaler()
    X_scaled = scaler.fit_transform(X)

    model = CustomLogisticRegression(**MODEL_PARAMS)
    model.feature_names = FEATURE_NAMES
    model.fit(X_scaled, y)

    acc = float((model.predict(X_scaled) == y).mean())
    logger.info("Train accuracy: %.3f", acc)

    if save:
        scaler.save(SCALER_FILE)
        model.save_weights(MODEL_WEIGHTS_FILE)

        zone_meta_dict = {
            int(k): {kk: int(vv) for kk, vv in v.items()}
            for k, v in (
                zone_meta_df
                .set_index("zone_id")[["capacity", "zone_type_standard"]]
                .to_dict("index")
            ).items()
        }
        with open(ZONE_META_FILE, "w") as f:
            json.dump(zone_meta_dict, f, indent=2)

        logger.info("Saved to %s/", MODEL_PATH)

    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    train()
