import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import (
    LAG_HOURS, MA_WINDOWS, FEATURE_NAMES,
    THRESHOLD_LOW, THRESHOLD_MEDIUM, TEMP_FALLBACK_BY_MONTH,
)

# Russian federal holidays (month, day) — fixed dates only
_RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (1, 6), (1, 7), (1, 8),   # New Year + Christmas
    (2, 23),                   # Defender of the Fatherland Day
    (3, 8),                    # International Women's Day
    (5, 1),                    # Spring & Labor Day
    (5, 9),                    # Victory Day
    (6, 12),                   # Russia Day
    (11, 4),                   # National Unity Day
}


def label_occupancy(rate: float) -> int:
    if rate < THRESHOLD_LOW:
        return 0
    if rate < THRESHOLD_MEDIUM:
        return 1
    return 2


def _is_holiday(month: int, day: int) -> int:
    return int((month, day) in _RU_HOLIDAYS)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['hour']
    df = df.copy()
    h   = dt.dt.hour
    dow = dt.dt.dayofweek
    mon = dt.dt.month

    df['hour']         = h
    df['day_of_week']  = dow
    df['month']        = mon
    df['day_of_month'] = dt.dt.day
    df['quarter']      = dt.dt.quarter
    df['is_weekend']   = (dow >= 5).astype(int)

    df['hour_sin']  = np.sin(2 * np.pi * h  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * h  / 24)
    df['dow_sin']   = np.sin(2 * np.pi * dow / 7)
    df['dow_cos']   = np.cos(2 * np.pi * dow / 7)
    df['month_sin'] = np.sin(2 * np.pi * mon / 12)
    df['month_cos'] = np.cos(2 * np.pi * mon / 12)

    df['is_holiday'] = df.apply(
        lambda r: _is_holiday(int(r['month']), int(r['day_of_month'])), axis=1
    )
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
    df = hourly_df.copy()
    df['_hour_dt'] = df['hour']
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_ma_features(df)

    meta = zone_meta_df[['zone_id', 'capacity', 'zone_type_standard']].copy()
    df = df.merge(meta, on='zone_id', how='left', suffixes=('_obs', '_meta'))
    if 'capacity_meta' in df.columns:
        df['capacity'] = df['capacity_meta'].fillna(df['capacity_obs'])
        df.drop(columns=['capacity_obs', 'capacity_meta'], inplace=True)
    df['zone_type_standard'] = df['zone_type_standard'].fillna(1).astype(int)

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
        df['is_precipitation'] = df['is_precipitation'].fillna(0).astype(int)
        df['temperature'] = df.apply(
            lambda r: r['temperature'] if pd.notna(r['temperature'])
            else TEMP_FALLBACK_BY_MONTH.get(int(r['month']), 10),
            axis=1,
        )
    else:
        df['temperature']      = df['month'].map(TEMP_FALLBACK_BY_MONTH).fillna(10)
        df['is_precipitation'] = 0

    df.drop(columns=['_hour_dt'], inplace=True, errors='ignore')
    df['label'] = df['occupancy_rate'].apply(label_occupancy)

    lag_ma_cols = [c for c in FEATURE_NAMES if 'lag' in c or 'ma_' in c]
    df = df.dropna(subset=lag_ma_cols).reset_index(drop=True)

    return df


def build_prediction_vector(
    zone_id:       int,
    predicted_for: datetime,
    recent_hourly: pd.DataFrame,
    zone_meta:     Dict,
    weather:       Tuple[Optional[float], Optional[int]] = (None, None),
) -> np.ndarray:
    dt = pd.Timestamp(predicted_for)
    if dt.tzinfo is None:
        dt = dt.tz_localize('UTC')

    h   = dt.hour
    dow = dt.dayofweek
    mon = dt.month

    feats: Dict = {
        'hour':         h,
        'day_of_week':  dow,
        'month':        mon,
        'day_of_month': dt.day,
        'quarter':      dt.quarter,
        'is_weekend':   int(dow >= 5),
        'hour_sin':     float(np.sin(2 * np.pi * h   / 24)),
        'hour_cos':     float(np.cos(2 * np.pi * h   / 24)),
        'dow_sin':      float(np.sin(2 * np.pi * dow / 7)),
        'dow_cos':      float(np.cos(2 * np.pi * dow / 7)),
        'month_sin':    float(np.sin(2 * np.pi * mon / 12)),
        'month_cos':    float(np.cos(2 * np.pi * mon / 12)),
        'is_holiday':   _is_holiday(mon, dt.day),
        'zone_id':      int(zone_id),
        'capacity':           int(zone_meta.get('capacity', 10)),
        'zone_type_standard': int(zone_meta.get('zone_type_standard', 1)),
    }

    temp, is_prec = weather
    feats['temperature']     = float(temp) if temp is not None else float(TEMP_FALLBACK_BY_MONTH.get(dt.month, 10))
    feats['is_precipitation'] = int(is_prec) if is_prec is not None else 0

    FALLBACK = 0.5
    hourly_avgs = zone_meta.get('hourly_avgs', {})

    def _hist_avg(hour_of_day: int) -> float:
        h = hour_of_day % 24
        return float(hourly_avgs.get(str(h), hourly_avgs.get(h, FALLBACK)))

    if recent_hourly.empty:
        # No recent data — use per-zone per-hour historical averages so predictions
        # vary by time of day (3am ≠ 4pm) instead of a flat constant.
        for lag_h in LAG_HOURS:
            feats[f'occupancy_lag_{lag_h}h'] = _hist_avg(dt.hour - lag_h)
        for w in MA_WINDOWS:
            feats[f'occupancy_ma_{w}h'] = float(np.mean([_hist_avg(dt.hour - i) for i in range(1, w + 1)]))
    else:
        history   = recent_hourly.set_index('hour')['occupancy_rate'].sort_index()
        history   = history[~history.index.duplicated(keep='last')]
        pred_hour = dt.floor('h')
        history   = history[history.index < pred_hour]

        def get_lag(lag_h: int) -> float:
            t = pred_hour - pd.Timedelta(hours=lag_h)
            if t in history.index:
                return float(history[t])
            # No direct match (future slot) — try same hour yesterday.
            # Yesterday's 22:00 is far more predictive than the multi-day
            # average, because it captures the real daily on/off pattern.
            t_yesterday = t - pd.Timedelta(hours=24)
            if t_yesterday in history.index:
                return float(history[t_yesterday])
            return _hist_avg(t.hour)

        def get_ma(w: int) -> float:
            window = history[history.index >= pred_hour - pd.Timedelta(hours=w)]
            if not window.empty:
                return float(window.mean())
            # Try same window yesterday
            window_yesterday = history[
                (history.index >= pred_hour - pd.Timedelta(hours=w + 24)) &
                (history.index <  pred_hour - pd.Timedelta(hours=24))
            ]
            if not window_yesterday.empty:
                return float(window_yesterday.mean())
            hours = [(dt.hour - i) % 24 for i in range(1, w + 1)]
            return float(np.mean([_hist_avg(h) for h in hours]))

        for lag_h in LAG_HOURS:
            feats[f'occupancy_lag_{lag_h}h'] = get_lag(lag_h)
        for w in MA_WINDOWS:
            feats[f'occupancy_ma_{w}h'] = get_ma(w)

    return np.array([feats[f] for f in FEATURE_NAMES], dtype=float)
