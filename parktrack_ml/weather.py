"""
Weather module.

Fetches hourly temperature and precipitation from Open-Meteo (free, no API key),
stores and reads observations via the ParkTrack API (/weather/new, /weather).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

import requests
import pandas as pd

from .config import API_URL, API_TOKEN, TEMP_FALLBACK_BY_MONTH
from .api_client import ParkTrackClient

logger = logging.getLogger(__name__)

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _client() -> ParkTrackClient:
    return ParkTrackClient(API_URL, API_TOKEN)


def _fetch_open_meteo(
    lat: float,
    lon: float,
    start: Optional[date] = None,
    end: Optional[date] = None,
    forecast: bool = False,
) -> dict:
    params = {
        "latitude":  round(lat, 6),
        "longitude": round(lon, 6),
        "hourly":    "temperature_2m,precipitation",
        "timezone":  "UTC",
    }
    if forecast:
        url = FORECAST_URL
        params["forecast_days"] = 2
    else:
        url = ARCHIVE_URL
        params["start_date"] = start.isoformat()
        params["end_date"]   = end.isoformat()

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_hourly(camera_id: int, data: dict) -> list[tuple]:
    times  = data["hourly"]["time"]
    temps  = data["hourly"]["temperature_2m"]
    precip = data["hourly"]["precipitation"]
    rows = []
    for t, temp, prec in zip(times, temps, precip):
        if temp is None:
            continue
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        rows.append((camera_id, dt, float(temp), float(prec or 0)))
    return rows


def _zone_to_camera_map(client: ParkTrackClient) -> dict[int, int]:
    zones = client.get_zones()
    result = {}
    for z in zones:
        zid = z.get("parking_zone_id") or z.get("id") or z.get("zone_id")
        cid = z.get("camera_id")
        if zid and cid:
            result[int(zid)] = int(cid)
    return result


def backfill(days: int = 200) -> int:
    """Fetch historical weather for all cameras and POST to API. Called once on startup."""
    client = _client()
    try:
        cameras = client.get_cameras()
    except Exception as exc:
        logger.warning("Could not fetch cameras from API: %s", exc)
        return 0

    end_d   = date.today()
    start_d = end_d - timedelta(days=days)
    total   = 0

    for cam in cameras:
        cam_id = cam.get("camera_id") or cam.get("id")
        lat    = cam.get("latitude")
        lon    = cam.get("longitude")
        if not (cam_id and lat is not None and lon is not None):
            continue
        try:
            data = _fetch_open_meteo(lat, lon, start_d, end_d)
            rows = _parse_hourly(cam_id, data)
            for _, observed_at, temp, prec in rows:
                try:
                    if client.post_weather(cam_id, observed_at, temp, prec):
                        total += 1
                except Exception:
                    pass
            logger.info("  camera %d: %d rows fetched", cam_id, len(rows))
        except Exception as exc:
            logger.warning("  camera %d: %s", cam_id, exc)

    return total


def fetch_latest() -> int:
    """Fetch 2-day weather forecast for all cameras and POST to API. Called hourly."""
    client = _client()
    try:
        cameras = client.get_cameras()
    except Exception as exc:
        logger.warning("Could not fetch cameras from API: %s", exc)
        return 0

    total = 0
    for cam in cameras:
        cam_id = cam.get("camera_id") or cam.get("id")
        lat    = cam.get("latitude")
        lon    = cam.get("longitude")
        if not (cam_id and lat is not None and lon is not None):
            continue
        try:
            data = _fetch_open_meteo(lat, lon, forecast=True)
            rows = _parse_hourly(cam_id, data)
            for _, observed_at, temp, prec in rows:
                try:
                    if client.post_weather(cam_id, observed_at, temp, prec):
                        total += 1
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("  camera %d: %s", cam_id, exc)

    return total


def load_for_training() -> pd.DataFrame:
    """Load weather joined to zones via API, for use during model training."""
    empty = pd.DataFrame(columns=["zone_id", "hour", "temperature", "is_precipitation"])
    client = _client()

    try:
        cam_to_zone = {v: k for k, v in _zone_to_camera_map(client).items()}
    except Exception as exc:
        logger.warning("Could not build zone-camera map: %s", exc)
        return empty

    if not cam_to_zone:
        return empty

    try:
        records = client.get_weather()
    except Exception as exc:
        logger.warning("Could not fetch weather from API: %s", exc)
        return empty

    if not records:
        return empty

    rows = []
    for r in records:
        cam_id  = r.get("camera_id")
        zone_id = cam_to_zone.get(cam_id)
        if not zone_id:
            continue
        observed_at = r.get("observed_at")
        temp        = r.get("temperature")
        prec        = r.get("precipitation", 0)
        if observed_at is None or temp is None:
            continue
        rows.append({
            "zone_id":          zone_id,
            "hour":             pd.Timestamp(observed_at, tz="UTC").floor("h"),
            "temperature":      float(temp),
            "is_precipitation": int(float(prec) > 0),
        })

    return pd.DataFrame(rows) if rows else empty


def get_at(zone_id: int, target_dt: datetime) -> Tuple[Optional[float], Optional[int]]:
    """Return (temperature, is_precipitation) for a zone at target_dt, or (None, None)."""
    client = _client()

    try:
        cam_to_zone = {v: k for k, v in _zone_to_camera_map(client).items()}
        zone_to_cam = {v: k for k, v in cam_to_zone.items()}
        camera_id   = zone_to_cam.get(zone_id)
    except Exception:
        return (None, None)

    if not camera_id:
        return (None, None)

    from_dt = target_dt - timedelta(hours=1)
    to_dt   = target_dt + timedelta(hours=1)

    try:
        records = client.get_weather(camera_id=camera_id, from_dt=from_dt, to_dt=to_dt)
    except Exception:
        return (None, None)

    if not records:
        return (None, None)

    target_ts = target_dt.timestamp()
    best = min(
        records,
        key=lambda r: abs(
            pd.Timestamp(r["observed_at"]).timestamp() - target_ts
        ),
    )
    return (float(best["temperature"]), int(float(best.get("precipitation", 0)) > 0))
