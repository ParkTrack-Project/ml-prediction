"""Data loading via ParkTrack API."""

from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .config import API_URL, API_TOKEN
from .api_client import ParkTrackClient

log = logging.getLogger(__name__)

_CHUNK_DAYS = 7


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
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
    df["zone_id"]     = pd.to_numeric(df["zone_id"],  errors="coerce")
    df["occupied"]    = pd.to_numeric(df["occupied"],  errors="coerce").fillna(0)
    df["capacity"]    = pd.to_numeric(df["capacity"],  errors="coerce").fillna(1)
    df = df.dropna(subset=["zone_id", "observed_at"])
    df["zone_id"]  = df["zone_id"].astype(int)
    df["occupied"] = df["occupied"].astype(int)
    df["capacity"] = df["capacity"].clip(lower=1).astype(int)
    df["occupancy_rate"] = df["occupied"] / df["capacity"]
    return df[empty_cols].sort_values(["zone_id", "observed_at"]).reset_index(drop=True)


def _date_chunks(from_dt: datetime, to_dt: datetime, days: int = _CHUNK_DAYS):
    """Yield (chunk_from, chunk_to) pairs covering [from_dt, to_dt] with step=days."""
    cur = from_dt
    while cur < to_dt:
        nxt = min(cur + timedelta(days=days), to_dt)
        yield cur, nxt
        cur = nxt


def _fetch_chunked(client: ParkTrackClient, zone_id: int | None,
                   from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    frames = []
    for chunk_from, chunk_to in _date_chunks(from_dt, to_dt):
        log.debug("occupancy chunk zone=%s %s – %s", zone_id, chunk_from, chunk_to)
        records = client.get_occupancy(zone_id=zone_id, from_dt=chunk_from, to_dt=chunk_to)
        df = _parse_occupancy(records)
        if not df.empty:
            frames.append(df)
    if not frames:
        return _parse_occupancy([])
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=["zone_id", "observed_at"]).reset_index(drop=True)


def load_observations(
    zone_ids: Optional[List[int]] = None,
    from_dt:  Optional[datetime]  = None,
    to_dt:    Optional[datetime]  = None,
) -> pd.DataFrame:
    """Load raw occupancy observations. Returns zone_id, observed_at, occupied, capacity, occupancy_rate."""
    client = _client()

    if from_dt is None or to_dt is None:
        # Short window — single request is fine
        if zone_ids:
            frames = [
                _parse_occupancy(client.get_occupancy(zone_id=zid, from_dt=from_dt, to_dt=to_dt))
                for zid in zone_ids
            ]
            return pd.concat(frames, ignore_index=True) if frames else _parse_occupancy([])
        return _parse_occupancy(client.get_occupancy(from_dt=from_dt, to_dt=to_dt))

    if zone_ids:
        frames = [_fetch_chunked(client, zid, from_dt, to_dt) for zid in zone_ids]
        return pd.concat(frames, ignore_index=True) if frames else _parse_occupancy([])
    return _fetch_chunked(client, None, from_dt, to_dt)


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
