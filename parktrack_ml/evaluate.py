"""
Backtest validator — walk-forward evaluation of the saved model.

For each historical hourly bucket, the model predicts using only data
strictly before that timestamp (no lookahead), matching real inference.
Compares predicted class and predicted_occupied against actual values.

Usage:
    python -m parktrack_ml.evaluate
    python -m parktrack_ml.evaluate --days 60
    python -m parktrack_ml.evaluate --days 30 --zone-id 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    MODEL_FILE, ZONE_META_FILE,
    FEATURE_NAMES, CLASS_CENTER_RATES,
    THRESHOLD_LOW, THRESHOLD_MEDIUM, TRAIN_DAYS_BACK,
)
from .data_loader import load_observations, load_zone_meta, aggregate_hourly
from .features import build_prediction_vector, label_occupancy
from .model import LGBMWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluate")

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _predict_row(
    zone_id: int,
    hour: pd.Timestamp,
    history: pd.DataFrame,
    zone_meta: dict,
    model: LGBMWrapper,
    capacity: int,
) -> dict:
    """Predict for a single (zone, hour) using history strictly before that hour."""
    past = history[history["hour"] < hour]
    vec  = build_prediction_vector(zone_id, hour.to_pydatetime(), past, zone_meta)

    X            = vec.reshape(1, -1)
    class_code   = int(model.predict(X)[0])
    probs        = model.predict_proba(X)[0]
    expected_rate = sum(float(probs[c]) * CLASS_CENTER_RATES[c] for c in range(3))
    predicted_occ = max(0, min(capacity, round(expected_rate * capacity)))

    return {
        "class_code":        class_code,
        "confidence":        float(probs[class_code]),
        "predicted_occupied": predicted_occ,
    }


def evaluate(
    days: int = 30,
    zone_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Run walk-forward backtest.
    Returns a DataFrame with one row per (zone, hour) containing
    actual and predicted values.
    """
    log.info("Loading model from %s", MODEL_FILE)
    model = LGBMWrapper.load(MODEL_FILE)

    with open(ZONE_META_FILE) as f:
        zone_meta_all: dict[int, dict] = {int(k): v for k, v in json.load(f).items()}

    to_dt   = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=days)

    if zone_ids is None:
        zone_ids = sorted(zone_meta_all.keys())
        if not zone_ids:
            zone_meta_df = load_zone_meta()
            zone_ids = zone_meta_df["zone_id"].tolist()

    log.info("Fetching %d days of occupancy for zones %s ...", days, zone_ids)
    raw    = load_observations(zone_ids=zone_ids, from_dt=from_dt, to_dt=to_dt)
    hourly = aggregate_hourly(raw)

    if hourly.empty:
        log.error("No occupancy data for the requested period. Try --days with a larger value.")
        return pd.DataFrame()

    log.info("Evaluating %d hourly records across %d zones...",
             len(hourly), hourly["zone_id"].nunique())

    results = []
    for zone_id, grp in hourly.groupby("zone_id"):
        zone_id = int(zone_id)
        meta    = zone_meta_all.get(zone_id, {"capacity": 10, "zone_type_standard": 1})
        capacity = int(meta.get("capacity", 10))
        zone_history = grp.sort_values("hour").reset_index(drop=True)

        for _, row in zone_history.iterrows():
            hour         = row["hour"]
            actual_rate  = float(row["occupancy_rate"])
            actual_occ   = int(round(actual_rate * capacity))
            actual_label = label_occupancy(actual_rate)

            pred = _predict_row(zone_id, hour, zone_history, meta, model, capacity)

            results.append({
                "zone_id":           zone_id,
                "hour":              hour,
                "actual_class":      actual_label,
                "predicted_class":   pred["class_code"],
                "correct":           int(pred["class_code"] == actual_label),
                "actual_occupied":   actual_occ,
                "predicted_occupied": pred["predicted_occupied"],
                "abs_error":         abs(pred["predicted_occupied"] - actual_occ),
                "confidence":        pred["confidence"],
                "actual_rate":       actual_rate,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def print_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data to report.")
        return

    n     = len(df)
    acc   = df["correct"].mean()
    mae   = df["abs_error"].mean()

    print()
    print("=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)
    print(f"  Samples   : {n:,}")
    print(f"  Accuracy  : {acc:.1%}  {_bar(acc)}")
    print(f"  MAE (spots): {mae:.2f}")
    print()

    # Per-class metrics
    print("  ── Per-class ─────────────────────────────────────")
    print(f"  {'Class':<8}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
    for cls, name in CLASS_NAMES.items():
        y_true = (df["actual_class"] == cls)
        y_pred = (df["predicted_class"] == cls)
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"  {name:<8}  {p:>6.1%}  {r:>6.1%}  {f1:>6.1%}  {y_true.sum():>7,}")
    print()

    # Per-zone
    print("  ── Per-zone ──────────────────────────────────────")
    print(f"  {'Zone':>5}  {'Accuracy':>9}  {'MAE':>6}  {'Samples':>7}")
    for zid, g in df.groupby("zone_id"):
        z_acc = g["correct"].mean()
        z_mae = g["abs_error"].mean()
        print(f"  {zid:>5}  {z_acc:>9.1%}  {z_mae:>6.2f}  {len(g):>7,}")
    print()

    # Per-hour-of-day accuracy (key check for static-forecast bug)
    print("  ── Accuracy by hour of day ───────────────────────")
    df["hod"] = pd.to_datetime(df["hour"], utc=True).dt.hour
    hod = df.groupby("hod")["correct"].mean()
    for h, a in hod.items():
        bar = _bar(a, 15)
        print(f"  {h:02d}:00  {a:5.1%}  {bar}")
    print()

    # Confusion matrix
    print("  ── Confusion matrix (actual → predicted) ─────────")
    labels = [0, 1, 2]
    header = "         " + "".join(f"  {CLASS_NAMES[p]:>8}" for p in labels)
    print(header)
    for a in labels:
        row_mask = df["actual_class"] == a
        row = "  ".join(
            f"{int((df.loc[row_mask, 'predicted_class'] == p).sum()):>8}"
            for p in labels
        )
        print(f"  {CLASS_NAMES[a]:<7}  {row}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML occupancy model")
    parser.add_argument("--days",    type=int, default=30,
                        help="Days of history to evaluate (default: 30)")
    parser.add_argument("--zone-id", type=int, default=None,
                        help="Evaluate a single zone only")
    args = parser.parse_args()

    zone_ids = [args.zone_id] if args.zone_id else None
    df = evaluate(days=args.days, zone_ids=zone_ids)
    print_report(df)


if __name__ == "__main__":
    main()
