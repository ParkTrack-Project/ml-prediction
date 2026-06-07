"""
Generates 24-hour occupancy forecasts for all active zones
and publishes them to the ParkTrack API (/forecasts/new).
"""
from __future__ import annotations

import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from .config import API_URL, API_TOKEN, ML_MODEL_TYPE, ML_MODEL_VERSION, FORECAST_HOURS
from .api_client import ParkTrackClient
from .interfaces import predict
from .data_loader import load_recent_observations, aggregate_hourly

logger = logging.getLogger(__name__)

_client: ParkTrackClient | None = None

_EMPTY_HOURLY = pd.DataFrame(columns=["zone_id", "hour", "occupancy_rate", "capacity"])


def _get_client() -> ParkTrackClient:
    global _client
    if _client is None:
        _client = ParkTrackClient(API_URL, API_TOKEN)
    return _client


def _active_zone_ids() -> list[int]:
    zones = _get_client().get_zones()
    return sorted(
        int(z.get("parking_zone_id") or z.get("id") or z.get("zone_id"))
        for z in zones
        if z.get("is_active", True)
        and (z.get("parking_zone_id") or z.get("id") or z.get("zone_id"))
    )


def _predict_and_post(zone_id: int, target_dt: datetime, recent_hourly: pd.DataFrame) -> bool:
    try:
        result = predict(zone_id=zone_id, predicted_for=target_dt, recent_hourly=recent_hourly)
        _get_client().post_forecast(
            zone_id=zone_id,
            model_type=ML_MODEL_TYPE,
            model_version=ML_MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            predicted_for=target_dt,
            predicted_occupied=result.predicted_occupied,
            probability_free_space=result.probability_free_space,
            confidence=result.confidence,
            capacity=result.capacity,
            metadata={
                "occupancy_class": result.occupancy_class,
                "prob_low":        round(result.prob_low, 4),
                "prob_medium":     round(result.prob_medium, 4),
                "prob_high":       round(result.prob_high, 4),
            },
        )
        return True
    except Exception as exc:
        logger.warning("Forecast failed zone=%d at %s: %s", zone_id, target_dt, exc)
        return False


def run() -> None:
    """Generate forecasts for all zones for the next 24 hours and POST to ParkTrack API."""
    if not API_TOKEN:
        logger.warning("API_TOKEN not set — skipping forecast posting")
        return

    now      = datetime.now(timezone.utc)
    zone_ids = _active_zone_ids()

    base  = now.replace(second=0, microsecond=0)
    slots = [
        base.replace(minute=m) + timedelta(hours=h)
        for h in range(FORECAST_HOURS + 1) for m in (0, 30)
        if base.replace(minute=m) + timedelta(hours=h) > now
    ][:FORECAST_HOURS * 2]

    # Pre-fetch recent observations ONCE per zone, not once per (zone, slot).
    # Without this we'd make zones×slots = potentially 720+ API calls per run.
    recent_by_zone: dict[int, pd.DataFrame] = {}
    for zid in zone_ids:
        try:
            raw = load_recent_observations(zid, now, hours=25)
            recent_by_zone[zid] = aggregate_hourly(raw) if not raw.empty else _EMPTY_HOURLY
        except Exception as exc:
            logger.warning("Failed to load recent obs zone=%d: %s", zid, exc)
            recent_by_zone[zid] = _EMPTY_HOURLY

    tasks   = [(z, t) for z in zone_ids for t in slots]
    success = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_predict_and_post, z, t, recent_by_zone.get(z, _EMPTY_HOURLY)): (z, t)
            for z, t in tasks
        }
        for f in as_completed(futures):
            if f.result():
                success += 1

    logger.info("Forecasts posted: %d/%d", success, len(tasks))
