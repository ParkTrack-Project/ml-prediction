"""Feature engineering for occupancy forecasting."""

from __future__ import annotations

import pandas as pd


FEATURE_COLS = [
    "zone_id",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "month",
    "horizon_minutes",
]


def build_features(df: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Add time-based features to an occupancy DataFrame.

    Expected input columns: zone_id, observed_at (datetime, UTC), occupied, capacity.
    Returns a copy with feature columns appended.
    """
    df = df.copy()
    dt = pd.to_datetime(df["observed_at"], utc=True)

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["horizon_minutes"] = horizon_minutes

    return df


def build_predict_features(
    zone_ids: list[int],
    predict_at: pd.Timestamp,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build a feature row per zone for a single future timestamp."""
    rows = []
    for zid in zone_ids:
        rows.append(
            {
                "zone_id": zid,
                "hour": predict_at.hour,
                "minute": predict_at.minute,
                "day_of_week": predict_at.dayofweek,
                "is_weekend": int(predict_at.dayofweek >= 5),
                "month": predict_at.month,
                "horizon_minutes": horizon_minutes,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLS)
