"""
Generates 24-hour occupancy forecasts for all active zones
and publishes them to the ParkTrack API (/forecasts/new).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List

import psycopg2

from .config import API_URL, API_TOKEN, ML_MODEL_VERSION, DB_CONFIG
from .api_client import ParkTrackClient
from .interfaces import predict, PredictOutput

logger = logging.getLogger(__name__)

_client: ParkTrackClient = None


def _get_client() -> ParkTrackClient:
    global _client
    if _client is None:
        _client = ParkTrackClient(API_URL, API_TOKEN)
    return _client


def _active_zone_ids() -> List[int]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parking_zone_id FROM parking_zones WHERE is_active = TRUE ORDER BY parking_zone_id"
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _predict_and_post(zone_id: int, target_dt: datetime) -> bool:
    try:
        result = predict(zone_id=zone_id, predicted_for=target_dt)
        _get_client().post_forecast(
            zone_id=zone_id,
            model_version=ML_MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            predicted_for=target_dt,
            predicted_occupied=result.predicted_occupied,
            probability_free_space=result.probability_free_space,
            confidence=result.confidence,
            capacity=result.capacity,
            metadata={
                'occupancy_class': result.occupancy_class,
                'prob_low':        round(result.prob_low, 4),
                'prob_medium':     round(result.prob_medium, 4),
                'prob_high':       round(result.prob_high, 4),
            },
        )
        return True
    except Exception as exc:
        logger.warning(f"Forecast failed zone={zone_id} at {target_dt}: {exc}")
        return False


def run():
    """Generate forecasts for all zones for the next 24 hours and POST to ParkTrack API."""
    if not API_TOKEN:
        logger.warning("API_TOKEN not set — skipping forecast posting")
        return

    now      = datetime.now(timezone.utc)
    zone_ids = _active_zone_ids()

    base  = now.replace(second=0, microsecond=0)
    slots = [
        base.replace(minute=m) + timedelta(hours=h)
        for h in range(25) for m in (0, 30)
        if base.replace(minute=m) + timedelta(hours=h) > now
    ][:48]

    tasks   = [(z, t) for z in zone_ids for t in slots]
    success = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_predict_and_post, z, t): (z, t) for z, t in tasks}
        for f in as_completed(futures):
            if f.result():
                success += 1

    logger.info(f"Forecasts posted: {success}/{len(tasks)}")
