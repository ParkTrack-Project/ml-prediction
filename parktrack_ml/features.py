import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import (
    LAG_HOURS, MA_WINDOWS, FEATURE_NAMES,
    THRESHOLD_LOW, THRESHOLD_MEDIUM, TEMP_FALLBACK_BY_MONTH,
)


def label_occupancy(rate: float) -> int:
    if rate < THRESHOLD_LOW:
        return 0
    if rate < THRESHOLD_MEDIUM:
        return 1
    return 2


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['hour']
    df = df.copy()
    df['hour']         = dt.dt.hour
    df['day_of_week']  = dt.dt.dayofweek
    df['month']        = dt.dt.month
    df['day_of_month'] = dt.dt.day
    df['quarter']      = dt.dt.quarter
    df['is_weekend']   = (dt.dt.dayofweek >= 5).astype(int)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(['zone_id', 'hour'])
    for h in LAG_HOURS:
        df[f'occupancy_lag_{h}h'] = df.groupby('zone_id')['occupancy_rate'].shift(h)
    return df


def _add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(['zone_id', 'hour'])
    for w in MA_WINDOWS:
        df[f'occupancy_ma_{w}h'] = (
            df.groupby('zone_id')['occupancy_rate']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    return df


def build_training_dataset(
    hourly_df:    pd.DataFrame,
    zone_meta_df: pd.DataFrame,
    weather_df:   Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the training dataset from hourly occupancy + zone metadata + optional weather.
    Drops rows where lag features are NaN (insufficient history).
    Returns DataFrame with FEATURE_NAMES columns and 'label'.
    """
    # Rename datetime column before adding time features
    df = hourly_df.rename(columns={'hour': 'hour'}).copy()
    # stash the original datetime for merges
    df['_hour_dt'] = df['hour']
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_ma_features(df)

    # Zone metadata
    meta = zone_meta_df[['zone_id', 'capacity', 'zone_type_standard']].copy()
    df = df.merge(meta, on='zone_id', how='left', suffixes=('_obs', '_meta'))
    if 'capacity_meta' in df.columns:
        df['capacity'] = df['capacity_meta'].fillna(df['capacity_obs'])
        df.drop(columns=['capacity_obs', 'capacity_meta'], inplace=True)
    df['zone_type_standard'] = df['zone_type_standard'].fillna(1).astype(int)

    # Weather
    if weather_df is not None and not weather_df.empty:
        df = df.merge(
            weather_df[['zone_id', 'hour', 'temperature', 'is_precipitation']],
            left_on=['zone_id', '_hour_dt'],
            right_on=['zone_id', 'hour'],
            how='left',
            suffixes=('', '_w'),
        )
        if 'hour_w' in df.columns:
            df.drop(columns=['hour_w'], inplace=True)
        # Fill gaps with seasonal fallback
        df['is_precipitation'] = df['is_precipitation'].fillna(0).astype(int)
        df['temperature'] = df.apply(
            lambda r: r['temperature'] if pd.notna(r['temperature'])
            else TEMP_FALLBACK_BY_MONTH.get(int(r['month']), 10),
            axis=1,
        )
    else:
        df['temperature']    = df['month'].map(TEMP_FALLBACK_BY_MONTH).fillna(10)
        df['is_precipitation'] = 0

    df.drop(columns=['_hour_dt'], inplace=True, errors='ignore')

    df['label'] = df['occupancy_rate'].apply(label_occupancy)

    lag_ma_cols = [c for c in FEATURE_NAMES if 'lag' in c or 'ma_' in c]
    df = df.dropna(subset=lag_ma_cols).reset_index(drop=True)

    return df


def build_prediction_vector(
    zone_id:      int,
    predicted_for: datetime,
    recent_hourly: pd.DataFrame,
    zone_meta:    Dict,
    weather:      Tuple[Optional[float], Optional[int]] = (None, None),
) -> np.ndarray:
    """
    Build a single feature vector for inference.

    weather: (temperature, is_precipitation) — pass result from weather.get_at().
             Falls back to seasonal average if None.
    """
    dt = pd.Timestamp(predicted_for)
    if dt.tzinfo is None:
        dt = dt.tz_localize('UTC')

    # Time
    feats: Dict = {
        'hour':         dt.hour,
        'day_of_week':  dt.dayofweek,
        'month':        dt.month,
        'day_of_month': dt.day,
        'quarter':      dt.quarter,
        'is_weekend':   int(dt.dayofweek >= 5),
    }

    # Zone
    feats['capacity']           = int(zone_meta.get('capacity', 10))
    feats['zone_type_standard'] = int(zone_meta.get('zone_type_standard', 1))

    # Weather
    temp, is_prec = weather
    feats['temperature']     = float(temp) if temp is not None else float(TEMP_FALLBACK_BY_MONTH.get(dt.month, 10))
    feats['is_precipitation'] = int(is_prec) if is_prec is not None else 0

    # Occupancy lags & MAs
    FALLBACK = 0.5
    if recent_hourly.empty:
        for h in LAG_HOURS:
            feats[f'occupancy_lag_{h}h'] = FALLBACK
        for w in MA_WINDOWS:
            feats[f'occupancy_ma_{w}h'] = FALLBACK
    else:
        history   = recent_hourly.set_index('hour')['occupancy_rate'].sort_index()
        pred_hour = dt.floor('h')
        history   = history[history.index < pred_hour]

        def get_lag(h):
            t = pred_hour - pd.Timedelta(hours=h)
            if t in history.index:
                return float(history[t])
            prior = history[history.index <= t]
            return float(prior.iloc[-1]) if not prior.empty else FALLBACK

        def get_ma(w):
            window = history[history.index >= pred_hour - pd.Timedelta(hours=w)]
            return float(window.mean()) if not window.empty else FALLBACK

        for h in LAG_HOURS:
            feats[f'occupancy_lag_{h}h'] = get_lag(h)
        for w in MA_WINDOWS:
            feats[f'occupancy_ma_{w}h'] = get_ma(w)

    return np.array([feats[f] for f in FEATURE_NAMES], dtype=float)
