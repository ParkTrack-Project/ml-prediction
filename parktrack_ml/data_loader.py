"""Data loading via ParkTrack API."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .config import API_URL, API_TOKEN
from .api_client import ParkTrackClient

log = logging.getLogger(__name__)

_CHUNK_DAYS = 4


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
    empty_cols = ["zone_id", "observed_at", "occupied", "capacity", "occupancy_rate", "confidence"]
    if not records:
        return pd.DataFrame(columns=empty_cols)
    df = pd.DataFrame(records)
    df["observed_at"] = pd.to_datetime(
        df.reindex(columns=["observed_at"])["observed_at"],
        utc=True, errors="coerce", format="mixed",
    )
    for col in ("zone_id", "occupied", "capacity"):
        values = df.reindex(columns=[col])[col]
        values = values.mask(values.map(lambda value: isinstance(value, (bool, np.bool_))))
        df[col] = pd.to_numeric(values, errors="coerce")
    valid = df["observed_at"].notna()
    for col in ("zone_id", "occupied", "capacity"):
        valid &= np.isfinite(df[col]) & df[col].mod(1).eq(0) & df[col].lt(float(2**63))
    valid &= df["capacity"].gt(0) & df["occupied"].ge(0) & df["zone_id"].ge(0)
    if (~valid).any():
        log.warning("Dropped %d invalid occupancy records", int((~valid).sum()))
    df = df.loc[valid].copy()
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    if "confidence" not in df.columns:
        df["confidence"] = float("nan")
    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    df["confidence"] = confidence.where(np.isfinite(confidence) & confidence.between(0, 1))
    for col in ("zone_id", "occupied", "capacity"):
        df[col] = df[col].astype("int64")

    keys = ["zone_id", "observed_at"]
    conflicts = df.groupby(keys)["occupied"].nunique().gt(1).sum()
    if conflicts:
        log.warning("Conflicting occupied counts in %d duplicate observations", int(conflicts))
    df["_ingested_at"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in ("ingested_at", "received_at", "created_at"):
        if col in df.columns:
            stamp = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
            df["_ingested_at"] = df["_ingested_at"].fillna(stamp)
    df = df.sort_values("_ingested_at", ascending=False, kind="stable", na_position="last")
    chosen_occupied = df.groupby(keys, sort=False)["occupied"].transform("first")
    df = df.loc[df["occupied"].eq(chosen_occupied)].copy()
    df["_valid_confidence"] = df["confidence"].notna()
    df = df.sort_values("_valid_confidence", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=keys)

    over_capacity = df["occupied"].gt(df["capacity"])
    if over_capacity.any():
        log.warning("Clipped %d occupancy records exceeding capacity", int(over_capacity.sum()))
    df["occupied"] = df["occupied"].clip(upper=df["capacity"])
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
    records = []
    for chunk_from, chunk_to in _date_chunks(from_dt, to_dt):
        log.debug("occupancy chunk zone=%s %s – %s", zone_id, chunk_from, chunk_to)
        records.extend(client.get_occupancy(zone_id=zone_id, from_dt=chunk_from, to_dt=chunk_to))
    return _parse_occupancy(records)


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
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else _parse_occupancy([])
    return _fetch_chunked(client, None, from_dt, to_dt)


def load_recent_observations(
    zone_id: int,
    before_dt: datetime,
    hours: int = 25,
    *,
    as_of: datetime | None = None,
    completed_hour: bool = True,
) -> pd.DataFrame:
    """Load observations before min(before_dt, now, as_of) in UTC.

    Inference uses completed hours by default. Set completed_hour=False for an
    exact historical cutoff. as_of can further restrict the available history.
    """
    cutoff = min(pd.to_datetime(before_dt, utc=True), pd.Timestamp(datetime.now(timezone.utc)))
    if as_of is not None:
        cutoff = min(cutoff, pd.to_datetime(as_of, utc=True))
    if completed_hour:
        cutoff = cutoff.floor("h")
    to_dt = cutoff.to_pydatetime()
    from_dt = to_dt - timedelta(hours=hours)
    raw = load_observations(zone_ids=[zone_id], from_dt=from_dt, to_dt=to_dt)
    observed_at = pd.to_datetime(raw["observed_at"], utc=True, errors="coerce", format="mixed")
    return raw.loc[observed_at < cutoff].reset_index(drop=True)


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw observations to hourly buckets per zone.

    Returns zone_id, hour, occupancy_rate (plain mean), capacity, n_obs, confidence
    and observed free_fraction (the fraction of observations below capacity).
    The detector reports confidence = 0 for an empty zone (no cars to score), so confidence is
    averaged over observations with cars only (NaN for hours without any) and is not used as a weight.
    """
    cols = ["zone_id", "hour", "occupancy_rate", "capacity", "n_obs", "confidence", "free_fraction"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["observed_at"])
    if "confidence" not in df.columns:
        df["confidence"] = float("nan")
    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    df["confidence"] = confidence.where(np.isfinite(confidence) & confidence.between(0, 1))
    df["hour"] = df["observed_at"].dt.floor("h")
    df["_conf_cars"] = df["confidence"].where(df["occupied"] > 0)
    df["_free"] = df["occupied"] < df["capacity"]
    g = df.groupby(["zone_id", "hour"])
    hourly = pd.DataFrame({
        "occupancy_rate": g["occupancy_rate"].mean(),
        "capacity":       g["capacity"].max(),
        "n_obs":          g["occupancy_rate"].size(),
        "confidence":     g["_conf_cars"].mean(),
        "free_fraction":  g["_free"].mean(),
    }).reset_index()
    return hourly[cols].sort_values(["zone_id", "hour"]).reset_index(drop=True)
