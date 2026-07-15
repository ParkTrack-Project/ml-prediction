"""
Training pipeline — LightGBM on data from the ParkTrack API.

Usage:
    python -m parktrack_ml.train
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import (
    MODEL_PATH, MODEL_FILE, ZONE_META_FILE,
    FEATURE_NAMES, CATEGORICAL_FEATURES,
    LGBM_PARAMS, TRAIN_DAYS_BACK,
)
from .data_loader import load_observations, load_zone_meta, aggregate_hourly
from .features import build_training_dataset
from .model import LGBMWrapper

logger = logging.getLogger(__name__)
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = float((y_pred == y_true).mean())
    logger.info("  Accuracy: %.3f", acc)
    for cls, name in CLASS_NAMES.items():
        mask = y_true == cls
        if not mask.any():
            continue
        tp = int(((y_pred == cls) & mask).sum())
        fp = int(((y_pred == cls) & ~mask).sum())
        fn = int(((y_pred != cls) & mask).sum())
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        logger.info(
            "  %-8s  precision=%.2f  recall=%.2f  f1=%.2f  support=%d",
            name, p, r, f1, mask.sum(),
        )


def train(save: bool = True) -> LGBMWrapper:
    os.makedirs(MODEL_PATH, exist_ok=True)

    zone_meta_df = load_zone_meta()
    logger.info("Zones: %s", zone_meta_df["zone_id"].tolist())

    to_dt   = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=TRAIN_DAYS_BACK)

    zone_ids = zone_meta_df["zone_id"].tolist()
    raw_df = load_observations(zone_ids=zone_ids, from_dt=from_dt, to_dt=to_dt)
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

    if hourly_df.empty:
        logger.warning(
            "No occupancy data in the last %d days — skipping training. "
            "Try increasing TRAIN_DAYS_BACK or wait for data to accumulate.",
            TRAIN_DAYS_BACK,
        )
        if os.path.exists(MODEL_FILE):
            logger.info("Keeping existing model at %s", MODEL_FILE)
            return LGBMWrapper.load(MODEL_FILE)
        raise RuntimeError("No training data and no existing model to fall back to.")

    dataset = build_training_dataset(hourly_df, zone_meta_df, weather_df)
    n = len(dataset)
    if n < 100:
        logger.warning("Only %d training samples — model quality will be low.", n)
    logger.info("Dataset: %d samples, %d features", n, len(FEATURE_NAMES))
    for cls, name in CLASS_NAMES.items():
        cnt = int((dataset["label"] == cls).sum())
        logger.info("  %s: %d (%.1f%%)", name, cnt, cnt / n * 100)

    # Temporal train/val split — last 20% as validation (preserves time order)
    dataset = dataset.sort_values("hour").reset_index(drop=True) if "hour" in dataset.columns else dataset
    split = int(n * 0.8)
    X = dataset[FEATURE_NAMES].values.astype(float)
    y = dataset["label"].values.astype(int)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Find categorical feature indices
    cat_indices = [FEATURE_NAMES.index(f) for f in CATEGORICAL_FEATURES if f in FEATURE_NAMES]

    model = LGBMWrapper(params=dict(LGBM_PARAMS))
    model.feature_names = FEATURE_NAMES
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        categorical_feature=cat_indices if cat_indices else None,
    )
    logger.info("Best iteration: %d", model._booster.best_iteration)

    logger.info("--- Train set ---")
    _metrics(y_train, model.predict(X_train))
    logger.info("--- Val set ---")
    _metrics(y_val, model.predict(X_val))

    importance = model.feature_importance()
    top10 = list(importance.items())[:10]
    logger.info("Top-10 features by gain: %s", top10)

    if save:
        model.save(MODEL_FILE)

        # Per-zone per-hour historical averages — used as smarter fallback at inference
        # when no recent occupancy data is available (e.g. forecasting 24h+ ahead).
        hourly_avgs_by_zone: dict[int, dict[str, float]] = {}
        if not hourly_df.empty:
            hdf = hourly_df.copy()
            hdf['_hod'] = hdf['hour'].dt.hour
            for (zid, hod), grp in hdf.groupby(['zone_id', '_hod']):
                zid_int = int(zid)
                if zid_int not in hourly_avgs_by_zone:
                    hourly_avgs_by_zone[zid_int] = {}
                hourly_avgs_by_zone[zid_int][str(int(hod))] = round(float(grp['occupancy_rate'].mean()), 4)

        zone_meta_dict: dict = {}
        for zid, row in (
            zone_meta_df.set_index("zone_id")[["capacity", "zone_type_standard"]].iterrows()
        ):
            zid_int = int(zid)
            zone_meta_dict[zid_int] = {
                "capacity":           int(row["capacity"]),
                "zone_type_standard": int(row["zone_type_standard"]),
                "hourly_avgs":        hourly_avgs_by_zone.get(zid_int, {}),
            }

        with open(ZONE_META_FILE, "w") as f:
            json.dump(zone_meta_dict, f, indent=2)

        logger.info("Saved to %s", MODEL_FILE)

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    train()
