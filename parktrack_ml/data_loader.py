"""
Database access layer for the ML module.
All queries go through this file — change DB schema here, not elsewhere.
"""
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from .config import DB_CONFIG


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _connect():
    return psycopg2.connect(**DB_CONFIG)


def _query_df(query: str, params=None, columns: list = None) -> pd.DataFrame:
    """Execute a query and return a DataFrame without pandas/psycopg2 warnings."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=columns or [])
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Zone metadata (static, cached after first load)
# ---------------------------------------------------------------------------

def load_zone_meta() -> pd.DataFrame:
    """
    Return DataFrame: zone_id, capacity, zone_type_standard (0/1).
    Only active zones.
    """
    df = _query_df(
        """
        SELECT parking_zone_id AS zone_id, capacity, zone_type
        FROM parking_zones
        WHERE is_active = TRUE
        ORDER BY parking_zone_id
        """,
        columns=['zone_id', 'capacity', 'zone_type'],
    )
    df['zone_type_standard'] = (df['zone_type'] == 'standard').astype(int)
    return df[['zone_id', 'capacity', 'zone_type_standard']]


# ---------------------------------------------------------------------------
# Occupancy observations
# ---------------------------------------------------------------------------

def load_observations(
    zone_ids: Optional[List[int]] = None,
    from_dt:  Optional[datetime]  = None,
    to_dt:    Optional[datetime]  = None,
) -> pd.DataFrame:
    """
    Load raw observations.
    Returns: zone_id, observed_at (tz-aware UTC), occupied, capacity, occupancy_rate
    """
    conditions: list = []
    params: list = []

    if zone_ids:
        conditions.append('zone_id = ANY(%s)')
        params.append(zone_ids)
    if from_dt:
        conditions.append('observed_at >= %s')
        params.append(from_dt)
    if to_dt:
        conditions.append('observed_at < %s')
        params.append(to_dt)

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    df = _query_df(
        f"""
        SELECT zone_id, observed_at, occupied, capacity
        FROM occupancy_observations
        {where}
        ORDER BY zone_id, observed_at
        """,
        params=params or None,
        columns=['zone_id', 'observed_at', 'occupied', 'capacity'],
    )

    if not df.empty:
        df['observed_at'] = pd.to_datetime(df['observed_at'], utc=True)
        df['occupancy_rate'] = df['occupied'] / df['capacity'].clip(lower=1)

    return df


def load_recent_observations(zone_id: int, before_dt: datetime, hours: int = 25) -> pd.DataFrame:
    """
    Load the most recent `hours` hours of observations for one zone.
    Used at inference time to build lag/MA features.
    """
    from_dt = before_dt - timedelta(hours=hours)
    return load_observations(zone_ids=[zone_id], from_dt=from_dt, to_dt=before_dt)


# ---------------------------------------------------------------------------
# Hourly aggregation
# ---------------------------------------------------------------------------

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw observations to hourly buckets per zone.
    Returns: zone_id, hour (datetime), occupancy_rate, capacity
    """
    df = df.copy()
    # Floor to hour; works for both tz-aware and naive timestamps
    df['hour'] = df['observed_at'].dt.floor('h')

    hourly = (
        df.groupby(['zone_id', 'hour'])
        .agg(
            occupancy_rate=('occupancy_rate', 'mean'),
            capacity=('capacity', 'max'),
        )
        .reset_index()
    )
    return hourly.sort_values(['zone_id', 'hour']).reset_index(drop=True)
