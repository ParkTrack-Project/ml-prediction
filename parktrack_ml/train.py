"""
Training pipeline (v4.1) - class, rate and hourly availability proxy per horizon.

Usage:
    python -m parktrack_ml.train                       # data from the ParkTrack API
    python -m parktrack_ml.train --hourly-csv file.gz  # offline, prepared hourly dataset
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .backtest import load_hourly_csv, unreliable_zones, zone_meta_from_hourly
from .config import (
    HORIZONS, ML_MODEL_VERSION, MODEL_PATH, MODEL_V4_FILE, TRAIN_DAYS_BACK, VAL_DAYS,
    LGBM_V4_PARAMS, LOCAL_TIMEZONE, PROFILE_DAYS, FFILL_LIMIT, THRESHOLD_LOW, THRESHOLD_MEDIUM,
)
from .data_loader import aggregate_hourly, load_observations, load_zone_meta
from .features_h import FEATURES_H, add_profile_features, build_horizon_dataset, compute_profiles, normalise_hourly, label_rate
from .model_h import HorizonModels

logger = logging.getLogger(__name__)


def fit_all(hourly: pd.DataFrame, zone_meta_df: pd.DataFrame, save: bool = True) -> HorizonModels:
    hourly = normalise_hourly(hourly)
    if hourly.empty:
        raise RuntimeError("No valid hourly training data")
    span = hourly["hour"].max() - hourly["hour"].min()
    cutoff = (hourly["hour"].max() - min(pd.Timedelta(days=VAL_DAYS), span * 0.2)).floor("h")
    fit_source = hourly[hourly.hour < cutoff]
    bad = unreliable_zones(fit_source)
    if bad:
        logger.info("Zones with unreliable CV (low detection confidence), excluded from training: %s", bad)
    data = hourly[~hourly["zone_id"].isin(bad)].copy()
    data["hour"] = pd.to_datetime(data["hour"], utc=True)
    if len(data) < 500:
        logger.warning("Only %d hourly rows - model quality will be low.", len(data))

    fit_source = data[data.hour < cutoff]
    if fit_source.empty:
        raise RuntimeError("No fit rows before validation cutoff")
    fit_meta = zone_meta_from_hourly(fit_source)
    if "zone_type_standard" not in fit_source and "zone_type_standard" in zone_meta_df:
        types = zone_meta_df.drop_duplicates("zone_id").set_index("zone_id").zone_type_standard
        fit_meta["zone_type_standard"] = fit_meta.zone_id.map(types).fillna(1).astype(int)
    zone_meta_df = fit_meta
    models = HorizonModels()
    val_metrics = {}

    for H in HORIZONS:
        d = build_horizon_dataset(data, zone_meta_df, H)

        d = d[(d.hour < cutoff) | (d.origin >= cutoff)].reset_index(drop=True)
        dv = add_profile_features(d, (d["hour"] < cutoff).values, source=fit_source)
        tr, va = dv[dv["hour"] < cutoff], dv[dv["hour"] >= cutoff]
        if len(va) < 200:
            raise RuntimeError(f"Not enough validation rows for H={H}: {len(va)}")
        n_cls, n_reg = models.fit_one(H, tr, va)

        probs, rate = models.predict(H, va[FEATURES_H].values)
        val_metrics[str(H)] = {
            "acc_classifier": round(float((probs.argmax(1) == va["label"].values).mean()), 4),
            "acc": round(float((label_rate((rate * va.capacity).round().to_numpy() / va.capacity.to_numpy()) == va["label"].values).mean()), 4),
            "mae_rate": round(float(abs(rate - va["occupancy_rate"].values).mean()), 4),
            "n_val": int(len(va)), "temperature": round(models.temperature[H], 2),
            "n_zones": int(va.zone_id.nunique()), "rounds_cls": int(n_cls), "rounds_reg": int(n_reg),
        }
        logger.info("H=%-3d val acc=%.3f mae=%.3f (rounds cls=%d reg=%d, T=%.2f)",
                    H, val_metrics[str(H)]["acc"], val_metrics[str(H)]["mae_rate"], n_cls, n_reg, models.temperature[H])

        if va.zone_id.nunique() < 5:
            logger.warning("H=%d validation covers only %d zones", H, va.zone_id.nunique())

    if save:
        zone_meta = {
            int(r.zone_id): {"capacity": int(r.capacity), "zone_type_standard": int(r.zone_type_standard)}
            for r in zone_meta_df.itertuples()
        }
        os.makedirs(MODEL_PATH, exist_ok=True)
        models.save(MODEL_V4_FILE, {
            "version": ML_MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_until": data["hour"].max().isoformat(),
            "fit_until": fit_source.hour.max().isoformat(),
            "validation_start": cutoff.isoformat(),
            "selection_metrics_independent_test": False,
            "feature_schema": "4.1-origin-local-past-only",
            "free_target": "hourly_mean_below_rounding_threshold",
            "training_config": {
                "horizons": HORIZONS, "parameters": LGBM_V4_PARAMS,
                "timezone": LOCAL_TIMEZONE, "profile_days": PROFILE_DAYS,
                "ffill_limit": FFILL_LIMIT, "thresholds": [THRESHOLD_LOW, THRESHOLD_MEDIUM],
                "sample_weight": "clip(sqrt(n_obs),1,8)",
                "parameters_by_head": {
                    str(H): {name: getattr(models, name)[H].params for name in ("cls", "reg", "free")}
                    for H in HORIZONS
                },
                "iterations_by_head": {
                    str(H): {name: int(getattr(models, name)[H].best_iteration) for name in ("cls", "reg", "free")}
                    for H in HORIZONS
                },
            },
            "python": sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name)
                         for name in ("lightgbm", "pandas", "numpy", "requests")},
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in Path(__file__).parent.glob("*.py")},
            "hourly_fingerprint_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(hourly, index=False).values.tobytes()).hexdigest(),
            "val": val_metrics,
            "zone_meta": zone_meta,
            "profiles": compute_profiles(fit_source),
            "unreliable_zones": bad,
        })
        logger.info("Saved to %s", MODEL_V4_FILE)
    return models


def train(save: bool = True, hourly_csv: Optional[str] = None) -> HorizonModels:
    if hourly_csv:
        hourly = load_hourly_csv(hourly_csv)
        zone_meta_df = zone_meta_from_hourly(hourly.iloc[:0])
    else:
        zone_meta_df = load_zone_meta()
        logger.info("Zones: %d", len(zone_meta_df))
        to_dt   = datetime.now(tz=timezone.utc)
        from_dt = to_dt - timedelta(days=TRAIN_DAYS_BACK)
        raw = load_observations(zone_ids=zone_meta_df["zone_id"].tolist(), from_dt=from_dt, to_dt=to_dt)
        logger.info("Observations: %d rows", len(raw))
        hourly = aggregate_hourly(raw)

    if hourly.empty:
        if os.path.exists(MODEL_V4_FILE):
            logger.warning("No occupancy data in the last %d days - keeping existing model.", TRAIN_DAYS_BACK)
            return HorizonModels.load(MODEL_V4_FILE)[0]
        raise RuntimeError("No training data and no existing model to fall back to.")

    logger.info("Hourly rows: %d, zones: %d", len(hourly), hourly["zone_id"].nunique())
    return fit_all(hourly, zone_meta_df, save=save)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-csv", default=None)
    train(hourly_csv=ap.parse_args().hourly_csv)
