"""
Weather data module.

Fetches hourly temperature and precipitation from Open-Meteo (free, no API key).
Stores results in weather_observations table in the parktrack DB.
Also provides helpers for joining weather with occupancy data during training/inference.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

import requests
import psycopg2
import psycopg2.extras
import pandas as pd

from .config import DB_CONFIG

logger = logging.getLogger(__name__)

ARCHIVE_URL  = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'

# DDL — run once via setup()
_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS weather_observations (
        id            SERIAL PRIMARY KEY,
        camera_id     INTEGER NOT NULL REFERENCES cameras(camera_id),
        observed_at   TIMESTAMPTZ NOT NULL,
        temperature   FLOAT NOT NULL,
        precipitation FLOAT NOT NULL DEFAULT 0.0,
        source        VARCHAR(50) NOT NULL DEFAULT 'open_meteo',
        fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (camera_id, observed_at)
    );
    CREATE INDEX IF NOT EXISTS idx_weather_camera_time
        ON weather_observations (camera_id, observed_at);
"""


def _connect():
    return psycopg2.connect(**DB_CONFIG)


def _table_exists() -> bool:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name   = 'weather_observations'
                )
            """)
            return cur.fetchone()[0]
    finally:
        conn.close()


def setup():
    """Create weather_observations if it doesn't exist, then grant access to the current user."""
    if _table_exists():
        logger.info("weather_observations table ready")
        return True

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
        conn.commit()
        logger.info("weather_observations table created")
        return True
    except Exception as exc:
        logger.warning(
            f"Could not create weather_observations table: {exc}. "
            "Weather features will use seasonal fallback. "
            "Ask the DB admin to run: migrations/001_weather_observations.sql"
        )
        return False
    finally:
        conn.close()


def _camera_locations() -> List[dict]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT camera_id, latitude, longitude FROM cameras WHERE is_active = TRUE"
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _fetch_open_meteo(lat: float, lon: float,
                      start: Optional[date] = None,
                      end:   Optional[date] = None,
                      forecast: bool = False) -> dict:
    params = {
        'latitude':  round(lat, 6),
        'longitude': round(lon, 6),
        'hourly':    'temperature_2m,precipitation',
        'timezone':  'UTC',
    }
    if forecast:
        url = FORECAST_URL
        params['forecast_days'] = 2
    else:
        url = ARCHIVE_URL
        params['start_date'] = start.isoformat()
        params['end_date']   = end.isoformat()

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_rows(camera_id: int, data: dict) -> List[Tuple]:
    times  = data['hourly']['time']
    temps  = data['hourly']['temperature_2m']
    precip = data['hourly']['precipitation']
    rows = []
    for t, temp, prec in zip(times, temps, precip):
        if temp is None:
            continue
        # Open-Meteo returns naive UTC strings like "2025-11-01T00:00"
        rows.append((camera_id, t + '+00:00', float(temp), float(prec or 0)))
    return rows


def _upsert(rows: List[Tuple]) -> int:
    if not rows:
        return 0
    conn = _connect()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO weather_observations
                    (camera_id, observed_at, temperature, precipitation)
                VALUES %s
                ON CONFLICT (camera_id, observed_at) DO NOTHING
                """,
                rows,
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def backfill(days: int = 200) -> int:
    """Fetch historical weather for all cameras. Call once on startup."""
    if not _table_exists():
        logger.warning("Skipping weather backfill — table not found")
        return 0

    cameras = _camera_locations()
    end_d   = date.today()
    start_d = end_d - timedelta(days=days)
    total   = 0

    for cam in cameras:
        try:
            data = _fetch_open_meteo(cam['latitude'], cam['longitude'], start_d, end_d)
            rows = _parse_rows(cam['camera_id'], data)
            n    = _upsert(rows)
            total += n
            logger.info(f"  camera {cam['camera_id']}: {n} rows")
        except Exception as exc:
            logger.warning(f"  camera {cam['camera_id']}: {exc}")

    return total


def fetch_latest() -> int:
    """Fetch 2-day weather forecast for all cameras. Call hourly."""
    if not _table_exists():
        return 0
    cameras = _camera_locations()
    total   = 0
    for cam in cameras:
        try:
            data = _fetch_open_meteo(cam['latitude'], cam['longitude'], forecast=True)
            rows = _parse_rows(cam['camera_id'], data)
            total += _upsert(rows)
        except Exception as exc:
            logger.warning(f"  camera {cam['camera_id']}: {exc}")
    return total


def load_for_training() -> pd.DataFrame:
    """Load weather joined to zones for training. Returns empty DataFrame if table is missing."""
    if not _table_exists():
        return pd.DataFrame(columns=['zone_id', 'hour', 'temperature', 'is_precipitation'])

    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    pz.parking_zone_id AS zone_id,
                    w.observed_at,
                    w.temperature,
                    (w.precipitation > 0)::int AS is_precipitation
                FROM weather_observations w
                JOIN parking_zones pz ON pz.camera_id = w.camera_id
                WHERE pz.is_active = TRUE
                ORDER BY pz.parking_zone_id, w.observed_at
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=['zone_id', 'hour', 'temperature', 'is_precipitation'])

    df = pd.DataFrame([dict(r) for r in rows])
    df['observed_at'] = pd.to_datetime(df['observed_at'], utc=True)
    df['hour'] = df['observed_at'].dt.floor('h')
    return df[['zone_id', 'hour', 'temperature', 'is_precipitation']]


def get_at(zone_id: int, target_dt: datetime) -> Tuple[Optional[float], Optional[int]]:
    """Return (temperature, is_precipitation) for zone at target_dt, or (None, None)."""
    if not _table_exists():
        return (None, None)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT w.temperature, (w.precipitation > 0)::int
                FROM weather_observations w
                JOIN parking_zones pz ON pz.camera_id = w.camera_id
                WHERE pz.parking_zone_id = %s
                  AND w.observed_at BETWEEN %s - INTERVAL '1 hour' AND %s + INTERVAL '1 hour'
                ORDER BY ABS(EXTRACT(EPOCH FROM (w.observed_at - %s)))
                LIMIT 1
            """, (zone_id, target_dt, target_dt, target_dt))
            row = cur.fetchone()
    finally:
        conn.close()

    return (row[0], row[1]) if row else (None, None)
