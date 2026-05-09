#!/usr/bin/env python3
"""
Fetch historical occupancy data from ParkTrack API, train a LightGBM model,
and save it to MODEL_PATH.

Usage:
    python -m parktrack_ml.train

Required env vars (see .env.example):
    API_URL, API_TOKEN, MODEL_PATH, TRAIN_DAYS_BACK
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from parktrack_ml.api_client import ParkTrackClient
from parktrack_ml.features import FEATURE_COLS, build_features

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS = [int(h) for h in os.getenv("FORECAST_HORIZONS", "15,30,60").split(",")]
TRAIN_DAYS_BACK = int(os.getenv("TRAIN_DAYS_BACK", "90"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/forecast_model.pkl"))
API_URL = os.environ["API_URL"]
API_TOKEN = os.environ["API_TOKEN"]


def fetch_training_data(client: ParkTrackClient) -> pd.DataFrame:
    to_dt = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=TRAIN_DAYS_BACK)

    log.info("Fetching occupancy from %s to %s", from_dt.date(), to_dt.date())
    records = client.get_occupancy(from_dt=from_dt, to_dt=to_dt, view="points")
    log.info("Fetched %d occupancy records", len(records))

    if not records:
        raise RuntimeError("No occupancy data returned from API.")

    df = pd.DataFrame(records)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df = df[["zone_id", "observed_at", "occupied", "capacity"]].dropna()
    df["capacity"] = df["capacity"].astype(int)
    df["occupied"] = df["occupied"].astype(int)
    return df.sort_values(["zone_id", "observed_at"]).reset_index(drop=True)


def build_training_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """For each observation, build a target: occupied N minutes later."""
    frames = []
    for horizon in HORIZONS:
        feat_df = build_features(df, horizon_minutes=horizon)
        feat_df["target_occupied"] = (
            df.groupby("zone_id")["occupied"]
            .shift(-round(horizon))  # approximate: assumes ~1 obs/min density
            .values
        )
        feat_df["capacity"] = df["capacity"].values
        feat_df = feat_df.dropna(subset=["target_occupied"])
        frames.append(feat_df)

    return pd.concat(frames, ignore_index=True)


def train(pairs: pd.DataFrame) -> lgb.LGBMRegressor:
    X = pairs[FEATURE_COLS]
    y = pairs["target_occupied"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    log.info("Validation MAE: %.3f occupied spots", mae)
    return model


def save_model(model: lgb.LGBMRegressor, pairs: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    capacity_map = (
        pairs.groupby("zone_id")["capacity"].median().astype(int).to_dict()
    )

    artifact = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "horizons": HORIZONS,
        "capacity_map": capacity_map,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    meta_path = MODEL_PATH.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "trained_at": artifact["trained_at"],
                "horizons": HORIZONS,
                "train_days_back": TRAIN_DAYS_BACK,
                "n_estimators": model.n_estimators_,
                "n_zones": len(capacity_map),
            },
            f,
            indent=2,
        )

    log.info("Model saved to %s", MODEL_PATH)
    log.info("Metadata saved to %s", meta_path)


def main() -> None:
    client = ParkTrackClient(API_URL, API_TOKEN)

    log.info("Step 1/3 — fetching training data")
    df = fetch_training_data(client)

    log.info("Step 2/3 — building training pairs (%d horizons: %s)", len(HORIZONS), HORIZONS)
    pairs = build_training_pairs(df)
    log.info("  Total training samples: %d", len(pairs))

    log.info("Step 3/3 — training model")
    model = train(pairs)

    save_model(model, pairs)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
