"""Origin-aware features shared by offline training and inference."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .config import (
    FFILL_LIMIT, HORIZONS, LOCAL_TIMEZONE, PROFILE_BLOCK_DAYS, PROFILE_DAYS,
    THRESHOLD_LOW, THRESHOLD_MEDIUM,
)
from .features import _is_holiday

FEATURES_H = [
    "last", "last1", "last2", "last3", "ma4",
    "lag_day", "lag_day2", "lag_day3", "lag_week",
    "hod", "dow", "is_weekend", "hour_sin", "hour_cos", "is_holiday",
    "prof", "prof_hod", "zone_mean", "prof_14d_hod", "prof_last_hod",
    "diff_1h", "diff_3h", "anomaly_vs_prof", "rolling_mean_24h", "rolling_max_24h",
    "lag_coverage", "exact_gap", "zone_id", "capacity", "zone_type_standard",
]
_H = pd.Timedelta(hours=1)
_PROFILE_COLUMNS = ["prof", "prof_hod", "zone_mean", "prof_14d_hod", "prof_last_hod"]


def utc_timestamp(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def bucket_for_gap(gap_hours: float) -> int:
    if not np.isfinite(gap_hours) or gap_hours < 1 or gap_hours > max(HORIZONS):
        raise ValueError(f"Unsupported history gap: {gap_hours}")
    return next(h for h in HORIZONS if gap_hours <= h)


def label_rate(rate) -> np.ndarray:
    values = np.asarray(rate, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Occupancy targets must be finite rates in [0, 1]")
    return np.digitize(values, [THRESHOLD_LOW, THRESHOLD_MEDIUM])


def _lag_offsets(gap: float) -> Dict[str, float]:
    day = 24 * math.ceil(gap / 24)
    return {
        "last": gap, "last1": gap + 1, "last2": gap + 2, "last3": gap + 3,
        "lag_day": day, "lag_day2": day + 24, "lag_day3": day + 48, "lag_week": 168,
    }


def _calendar(hours: pd.Series) -> Dict[str, np.ndarray]:
    local = pd.to_datetime(hours, utc=True).dt.tz_convert(LOCAL_TIMEZONE)
    hod, dow = local.dt.hour.to_numpy(), local.dt.dayofweek.to_numpy()
    return {
        "hod": hod, "dow": dow, "is_weekend": (dow >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * hod / 24), "hour_cos": np.cos(2 * np.pi * hod / 24),
        "is_holiday": np.array([_is_holiday(m, d) for m, d in zip(local.dt.month, local.dt.day)]),
    }


def normalise_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    d = hourly.copy()
    d["hour"] = pd.to_datetime(d["hour"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    for col in ("zone_id", "occupancy_rate", "capacity"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["hour", "zone_id", "occupancy_rate", "capacity"])
    valid = np.isfinite(d[["zone_id", "occupancy_rate", "capacity"]]).all(axis=1)
    valid &= d.occupancy_rate.between(0, 1) & (d.capacity > 0)
    d = d.loc[valid].copy()
    if (d.hour != d.hour.dt.floor("h")).any():
        raise ValueError("Hourly timestamps must be aligned to UTC hours")
    if d.duplicated(["zone_id", "hour"]).any():
        raise ValueError("Duplicate hourly (zone_id, hour) keys")
    d["zone_id"] = d.zone_id.astype(int)
    if "n_obs" not in d:
        d["n_obs"] = 1
    d["n_obs"] = pd.to_numeric(d.n_obs, errors="coerce").fillna(1).clip(lower=1)
    if "confidence" not in d:
        d["confidence"] = np.nan
    return d.sort_values(["zone_id", "hour"]).reset_index(drop=True)


def compute_profiles(df: pd.DataFrame, as_of=None) -> Dict[int, dict]:
    if df.empty:
        return {}
    end = utc_timestamp(as_of) if as_of is not None else pd.to_datetime(df.hour, utc=True).max() + _H
    d = df.copy()
    d["hour"] = pd.to_datetime(d.hour, utc=True)
    d = d[(d.hour < end) & (d.hour >= end - pd.Timedelta(days=PROFILE_DAYS))].copy()
    local = d.hour.dt.tz_convert(LOCAL_TIMEZONE)
    d["hod"], d["dow"] = local.dt.hour, local.dt.dayofweek
    out = {}
    for zid, g in d.groupby("zone_id"):
        short = g[g.hour >= end - pd.Timedelta(days=14)]
        out[int(zid)] = {
            "dow_hod": {f"{a}_{b}": float(v) for (a, b), v in g.groupby(["dow", "hod"]).occupancy_rate.mean().items()},
            "hod": {str(int(k)): float(v) for k, v in g.groupby("hod").occupancy_rate.mean().items()},
            "hod_14d": {str(int(k)): float(v) for k, v in short.groupby("hod").occupancy_rate.mean().items()},
            "mean": float(g.occupancy_rate.mean()),
        }
    return out


def _attach_profiles(d: pd.DataFrame, mask: np.ndarray, profiles: dict) -> None:
    cur = d.loc[mask]
    if cur.empty:
        return
    target_local = cur.hour.dt.tz_convert(LOCAL_TIMEZONE)
    origin_local = cur.origin.dt.tz_convert(LOCAL_TIMEZONE)
    for col in _PROFILE_COLUMNS:
        vals = []
        for zid, day, hour, old_hour in zip(cur.zone_id, target_local.dt.dayofweek, target_local.dt.hour, origin_local.dt.hour):
            p = profiles.get(int(zid), {})
            if col == "prof":
                value = p.get("dow_hod", {}).get(f"{day}_{hour}", np.nan)
            elif col == "prof_hod":
                value = p.get("hod", {}).get(str(hour), np.nan)
            elif col == "prof_last_hod":
                value = p.get("hod", {}).get(str(old_hour), np.nan)
            elif col == "prof_14d_hod":
                value = p.get("hod_14d", {}).get(str(hour), np.nan)
            else:
                value = p.get("mean", np.nan)
            vals.append(value)
        d.loc[mask, col] = vals


def add_profile_features(d: pd.DataFrame, train_mask: np.ndarray, source: pd.DataFrame | None = None) -> pd.DataFrame:
    d = d.copy()
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(train_mask) != len(d):
        raise ValueError("Profile mask length mismatch")
    if source is None:
        source = d.loc[train_mask, ["zone_id", "hour", "occupancy_rate"]].drop_duplicates(["zone_id", "hour"])
    else:
        source = source.copy()
    source["hour"] = pd.to_datetime(source.hour, utc=True)
    if "origin" not in d:
        raise ValueError("Profile features require an explicit forecast origin")
    for col in _PROFILE_COLUMNS:
        d[col] = np.nan
    if source.empty:
        d["anomaly_vs_prof"] = np.nan
        return d
    blocks = d.origin.dt.floor(f"{PROFILE_BLOCK_DAYS}D")
    for block in sorted(blocks[train_mask].unique()):
        mask = train_mask & (blocks == block).to_numpy()
        _attach_profiles(d, mask, compute_profiles(source[source.hour < block], as_of=block))
    frozen_end = source.hour.max() + _H
    rest = ~train_mask
    if rest.any() and (d.loc[rest, "origin"] < frozen_end - _H).any():
        raise ValueError("Frozen profiles include observations unavailable at forecast origin")
    _attach_profiles(d, rest, compute_profiles(source, as_of=frozen_end))
    d["anomaly_vs_prof"] = d["last"] - d.prof_last_hod
    return d


def training_gaps(H: int) -> list[int]:
    lower = max([h for h in HORIZONS if h < H], default=0) + 1
    return sorted({lower, (lower + H) // 2, H})


def _history_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    grids = []
    for zid, g in hourly.groupby("zone_id", sort=False):
        index = pd.date_range(g.hour.min(), g.hour.max(), freq="h")
        s = g.set_index("hour").occupancy_rate.reindex(index)
        filled = s.ffill(limit=FFILL_LIMIT)
        grids.append(pd.DataFrame({
            "zone_id": zid, "hour": index, "value": filled.to_numpy(),
            "rolling_mean_24h": filled.rolling(24, min_periods=1).mean().to_numpy(),
            "rolling_max_24h": filled.rolling(24, min_periods=1).max().to_numpy(),
        }))
    if not grids:
        return pd.DataFrame(columns=["value", "rolling_mean_24h", "rolling_max_24h"])
    return pd.concat(grids, ignore_index=True).set_index(["zone_id", "hour"])


def _with_history_features(d: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    gaps = (d.hour - d.origin) / _H
    day = np.ceil(gaps / 24).astype(int) * 24
    offsets = {
        "last": gaps, "last1": gaps + 1, "last2": gaps + 2, "last3": gaps + 3,
        "lag_day": day, "lag_day2": day + 24, "lag_day3": day + 48,
        "lag_week": np.full(len(d), 168),
    }
    for name, offset in offsets.items():
        times = d.hour - pd.to_timedelta(offset, unit="h")
        idx = pd.MultiIndex.from_arrays([d.zone_id, times])
        d[name] = grid.value.reindex(idx).to_numpy()
    d["ma4"] = d[["last", "last1", "last2", "last3"]].mean(axis=1)
    d["diff_1h"], d["diff_3h"] = d["last"] - d.last1, d["last"] - d.last3
    d["lag_coverage"] = d[list(offsets)].notna().mean(axis=1)
    origin_idx = pd.MultiIndex.from_arrays([d.zone_id, d.origin])
    for col in ("rolling_mean_24h", "rolling_max_24h"):
        d[col] = grid[col].reindex(origin_idx).to_numpy()
    d["exact_gap"] = gaps.to_numpy()
    for key, value in _calendar(d.hour).items():
        d[key] = value
    return d


def build_horizon_dataset(hourly: pd.DataFrame, zone_meta_df: pd.DataFrame, H: int) -> pd.DataFrame:
    data = normalise_hourly(hourly)
    grid = _history_grid(data)
    samples = []
    for gap in training_gaps(H):
        for _, g in data.groupby("zone_id", sort=False):
            times = g.hour.astype("int64").to_numpy()
            pos = times.searchsorted(times - gap * _H.value, side="right") - 1
            mask = pos >= 0
            cur = g.loc[mask].copy()
            cur["origin"] = pd.to_datetime(times[pos[mask]], utc=True)
            actual_gap = (cur.hour - cur.origin) / _H
            lower = max([h for h in HORIZONS if h < H], default=0)
            cur = cur[(actual_gap > lower) & (actual_gap <= H)]
            samples.append(cur)
    if not samples:
        raise ValueError(f"No eligible history for H={H}")
    d = pd.concat(samples, ignore_index=True).drop_duplicates(["zone_id", "hour", "origin"])
    d = _with_history_features(d, grid)
    meta = zone_meta_df[["zone_id", "zone_type_standard"]].rename(columns={"zone_type_standard": "meta_type"})
    if meta.zone_id.duplicated().any():
        raise ValueError("Duplicate zone metadata")
    d = d.drop(columns="zone_type_standard", errors="ignore").merge(meta, on="zone_id", how="left", validate="many_to_one")
    origin_idx = pd.MultiIndex.from_arrays([d.zone_id, d.origin])
    known_capacity = data.set_index(["zone_id", "hour"]).capacity.reindex(origin_idx).to_numpy()
    d["capacity"] = known_capacity
    known_type = (data.set_index(["zone_id", "hour"]).zone_type_standard.reindex(origin_idx).to_numpy()
                  if "zone_type_standard" in data else np.full(len(d), np.nan))
    d["zone_type_standard"] = d.meta_type.fillna(pd.Series(known_type, index=d.index)).fillna(1)
    d = d.drop(columns="meta_type")
    d["label"] = label_rate(d.occupancy_rate)
    return d.reset_index(drop=True)


def build_inference_vector(
    zone_id: int, predicted_for: datetime, recent_hourly: pd.DataFrame,
    zone_meta: dict, profiles: Dict[int, dict], as_of=None,
) -> tuple[int, np.ndarray]:
    buckets, rows = build_inference_frame(zone_id, [predicted_for], recent_hourly, zone_meta, profiles, as_of)
    return int(buckets[0]), rows[FEATURES_H].to_numpy(dtype=float)[0]


def build_inference_frame(zone_id, predicted_for, recent_hourly, zone_meta, profiles, as_of=None):
    targets = pd.Series([utc_timestamp(t).floor("h") for t in predicted_for])
    if targets.empty:
        return np.array([], dtype=int), pd.DataFrame(columns=FEATURES_H)
    now = utc_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    cutoff = min(now.floor("h"), targets.min())
    if recent_hourly is None or recent_hourly.empty:
        raise ValueError("No occupancy history available")
    history = recent_hourly.copy()
    if "zone_id" not in history:
        raise ValueError("History requires zone_id")
    history = history[pd.to_numeric(history.zone_id, errors="coerce") == int(zone_id)].copy()
    history["hour"] = pd.to_datetime(history.hour, utc=True)
    history = normalise_hourly(history[history.hour < cutoff])
    if history.empty:
        raise ValueError("No completed-hour history for zone")
    origin = history.hour.max()
    buckets = np.array([bucket_for_gap(float((t - origin) / _H)) for t in targets])
    row = pd.DataFrame({
        "zone_id": int(zone_id), "hour": targets, "origin": origin,
        "capacity": float(history.iloc[-1].capacity),
        "zone_type_standard": float(zone_meta.get("zone_type_standard", 1)),
    })
    row = _with_history_features(row, _history_grid(history))
    for col in _PROFILE_COLUMNS:
        row[col] = np.nan
    _attach_profiles(row, np.ones(len(row), dtype=bool), profiles)
    row["anomaly_vs_prof"] = row["last"] - row.prof_last_hod
    return buckets, row
