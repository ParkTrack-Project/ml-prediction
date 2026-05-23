"""Data loading via ParkTrack API."""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from .config import API_URL, API_TOKEN
from .api_client import ParkTrackClient


def _client() -> ParkTrackClient:
    return ParkTrackClient(API_URL, API_TOKEN)


def load_zone_meta() -> pd.DataFrame:
    """Return DataFrame: zone_id, capacity, zone_type_standard (0/1). Only active zones."""
    zones = _client().get_zones()
    rows = []
    for z in zones:
        zid      = z.get("parking_zone_id") or z.get("id") or z.get("zone_id")
        cap      = z.get("capacity", 10)
        ztype    = z.get("zone_type", "standard")
        is_active = z.get("is_active", True)
        if not is_active or not zid:
            continue
        rows.append({
            "zone_id":            int(zid),
            "capacity":           int(cap),
            "zone_type_standard": int(ztype == "standard"),
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["zone_id", "capacity", "zone_type_standard"]
    )
    return df.sort_values("zone_id").reset_index(drop=True)


def _parse_occupancy(records: list) -> pd.DataFrame:
    empty_cols = ["zone_id", "observed_at", "occupied", "capacity", "occupancy_rate"]
    if not records:
        return pd.DataFrame(columns=empty_cols)
    df = pd.DataFrame(records)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["occupied"]    = df["occupied"].astype(int)
    df["capacity"]    = df["capacity"].astype(int)
    df["occupancy_rate"] = df["occupied"] / df["capacity"].clip(lower=1)
    return df.sort_values(["zone_id", "observed_at"]).reset_index(drop=True)


def load_observations(
    zone_ids: Optional[List[int]] = None,
    from_dt:  Optional[datetime]  = None,
    to_dt:    Optional[datetime]  = None,
) -> pd.DataFrame:
    """Load raw occupancy observations. Returns zone_id, observed_at, occupied, capacity, occupancy_rate."""
    client = _client()
    if zone_ids:
        frames = [
            _parse_occupancy(client.get_occupancy(zone_id=zid, from_dt=from_dt, to_dt=to_dt))
            for zid in zone_ids
        ]
        return pd.concat(frames, ignore_index=True) if frames else _parse_occupancy([])
    return _parse_occupancy(client.get_occupancy(from_dt=from_dt, to_dt=to_dt))


def load_recent_observations(zone_id: int, before_dt: datetime, hours: int = 25) -> pd.DataFrame:
    """Load the most recent `hours` of observations for a zone. Used at inference time."""
    from_dt = before_dt - timedelta(hours=hours)
    return load_observations(zone_ids=[zone_id], from_dt=from_dt, to_dt=before_dt)


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw observations to hourly buckets per zone."""
    if df.empty:
        return df
    df = df.copy()
    df["hour"] = df["observed_at"].dt.floor("h")
    hourly = (
        df.groupby(["zone_id", "hour"])
        .agg(occupancy_rate=("occupancy_rate", "mean"), capacity=("capacity", "max"))
        .reset_index()
    )
    return hourly.sort_values(["zone_id", "hour"]).reset_index(drop=True)
