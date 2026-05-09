#!/usr/bin/env python3
"""
Load a trained model, generate occupancy forecasts for configured horizons,
and POST them to POST /forecasts/new.

Usage:
    python -m parktrack_ml.predict

Required env vars (see .env.example):
    API_URL, API_TOKEN, MODEL_PATH, FORECAST_HORIZONS
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from parktrack_ml.api_client import ParkTrackClient
from parktrack_ml.features import build_predict_features

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/forecast_model.pkl"))
API_URL = os.environ["API_URL"]
API_TOKEN = os.environ["API_TOKEN"]
FORECAST_HORIZONS = [
    int(h) for h in os.getenv("FORECAST_HORIZONS", "15,30,60").split(",")
]


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _confidence(predicted_occupied: int, capacity: int) -> float:
    """Simple heuristic: lower confidence when predicted_occupied ≈ capacity/2."""
    if capacity == 0:
        return 0.5
    ratio = predicted_occupied / capacity
    return float(round(0.5 + 0.5 * abs(ratio - 0.5) * 2, 4))


def _prob_free(predicted_occupied: int, capacity: int) -> float:
    if capacity == 0:
        return 0.0
    free = capacity - predicted_occupied
    return float(round(max(0.0, min(1.0, free / capacity)), 4))


def run_predictions(
    client: ParkTrackClient,
    artifact: dict,
    generated_at: datetime,
) -> tuple[int, int]:
    model = artifact["model"]
    capacity_map: dict[int, int] = artifact["capacity_map"]
    model_version: str = artifact.get("trained_at", "unknown")[:10]

    zone_ids = list(capacity_map.keys())
    if not zone_ids:
        log.warning("No zones in model capacity map.")
        return 0, 0

    inserted = 0
    skipped = 0

    for horizon in FORECAST_HORIZONS:
        predicted_for = generated_at + timedelta(minutes=horizon)
        predict_ts = pd.Timestamp(predicted_for)

        feat_df = build_predict_features(zone_ids, predict_ts, horizon)
        raw_preds = model.predict(feat_df)

        for zone_id, raw_pred in zip(zone_ids, raw_preds):
            cap = capacity_map[zone_id]
            pred_occ = int(np.clip(round(raw_pred), 0, cap))
            prob_free = _prob_free(pred_occ, cap)
            conf = _confidence(pred_occ, cap)

            fid = client.post_forecast(
                zone_id=zone_id,
                generated_at=generated_at,
                predicted_for=predicted_for,
                predicted_occupied=pred_occ,
                probability_free_space=prob_free,
                confidence=conf,
                capacity=cap,
                model_version=f"parktrack-ml-{model_version}",
                metadata={"horizon_minutes": horizon},
            )

            if fid is None:
                skipped += 1
            else:
                inserted += 1

        log.info(
            "  Horizon +%d min: posted forecasts for %d zones (predicted_for=%s)",
            horizon,
            len(zone_ids),
            predicted_for.strftime("%H:%M UTC"),
        )

    return inserted, skipped


def main() -> None:
    log.info("Loading model from %s", MODEL_PATH)
    artifact = load_artifact()
    log.info(
        "Model trained at %s, horizons=%s",
        artifact.get("trained_at", "?"),
        artifact.get("horizons", []),
    )

    client = ParkTrackClient(API_URL, API_TOKEN)
    generated_at = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

    log.info("Generating forecasts at %s for horizons %s", generated_at, FORECAST_HORIZONS)
    inserted, skipped = run_predictions(client, artifact, generated_at)

    log.info(
        "Done: %d forecasts posted, %d skipped (duplicates).",
        inserted,
        skipped,
    )


if __name__ == "__main__":
    main()
