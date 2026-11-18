"""Generate half-hour forecasts and publish them to the ParkTrack API."""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import local

from .config import API_URL, API_TOKEN, ML_MODEL_TYPE, ML_MODEL_VERSION, FORECAST_HOURS, LOOKBACK_HOURS
from .api_client import ParkTrackClient
from .interfaces import predict_many
from .data_loader import load_recent_observations, aggregate_hourly

logger = logging.getLogger(__name__)

_clients = local()


def _get_client() -> ParkTrackClient:
    if not hasattr(_clients, "client"):
        _clients.client = ParkTrackClient(API_URL, API_TOKEN)
    return _clients.client


def _active_zone_ids() -> list[int]:
    zones = _get_client().get_zones()
    return sorted({
        int(z.get("parking_zone_id") or z.get("id") or z.get("zone_id"))
        for z in zones
        if z.get("is_active", True)
        and (z.get("parking_zone_id") or z.get("id") or z.get("zone_id"))
    })


def _counts() -> dict[str, int]:
    return {
        "created": 0, "conflicts": 0, "failures": 0,
        "history_failures": 0, "prediction_failures": 0, "skipped_zones": 0,
    }


def _forecast_zone(zone_id: int, slots: list[datetime], now: datetime) -> dict[str, int]:
    counts = _counts()
    try:
        raw = load_recent_observations(zone_id, now, hours=LOOKBACK_HOURS, as_of=now)
        if raw.empty:
            raise ValueError("empty observation history")
        recent_hourly = aggregate_hourly(raw)
        if recent_hourly.empty:
            raise ValueError("empty hourly history")
    except Exception as exc:
        logger.error("History unavailable zone=%d; skipping zone: %s", zone_id, exc)
        counts["history_failures"] = counts["skipped_zones"] = 1
        return counts

    try:
        results = predict_many(
            zone_id=zone_id, predicted_for=slots, recent_hourly=recent_hourly, as_of=now,
        )
        if len(results) != len(slots):
            raise ValueError("prediction count does not match forecast slots")
        for result in results:
            probabilities = (
                result.confidence, result.prob_low, result.prob_medium,
                result.prob_high, result.probability_free_space,
            )
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
                raise ValueError("nonfinite or out-of-range prediction probability")
            if (not math.isfinite(result.capacity) or result.capacity <= 0
                    or int(result.capacity) != result.capacity
                    or not math.isfinite(result.predicted_occupied)
                    or int(result.predicted_occupied) != result.predicted_occupied
                    or not 0 <= result.predicted_occupied <= result.capacity):
                raise ValueError("invalid predicted count or capacity")
    except Exception as exc:
        logger.error("Prediction failed zone=%d; skipping zone: %s", zone_id, exc)
        counts["prediction_failures"] = counts["skipped_zones"] = 1
        return counts

    for target_dt, result in zip(slots, results):
        try:
            forecast_id = _get_client().post_forecast(
                zone_id=zone_id,
                model_type=ML_MODEL_TYPE,
                model_version=ML_MODEL_VERSION,
                generated_at=now,
                predicted_for=target_dt,
                predicted_occupied=result.predicted_occupied,
                probability_free_space=result.probability_free_space,
                confidence=result.confidence,
                capacity=result.capacity,
                metadata={
                    "as_of": now.isoformat(),
                    "forecast_semantics": "hourly_mean_proxy",
                    "predicted_occupied_method": "regression",
                    "occupancy_class_method": "regression_class",
                    "class_probability_method": "classifier",
                    "confidence_method": "classifier_probability_for_regression_class",
                    "probability_free_space_method": "binary_classifier",
                    "probability_free_space_calibration": "not_calibrated",
                    "occupancy_class": result.occupancy_class,
                    "prob_low": round(result.prob_low, 4),
                    "prob_medium": round(result.prob_medium, 4),
                    "prob_high": round(result.prob_high, 4),
                },
            )
            if forecast_id is None:
                counts["conflicts"] += 1
                logger.warning(
                    "Duplicate forecast generation (409) zone=%d at %s; "
                    "existing version retained", zone_id, target_dt,
                )
            else:
                counts["created"] += 1
        except Exception as exc:
            counts["failures"] += 1
            logger.error("Forecast POST failed zone=%d at %s: %s", zone_id, target_dt, exc)
    return counts


def run() -> dict[str, int]:
    """Publish future half-hour slots; failures counts POST errors only."""
    summary = _counts()
    if not API_TOKEN:
        logger.warning("API_TOKEN not set — skipping forecast posting")
        return summary

    now      = datetime.now(timezone.utc)
    zone_ids = _active_zone_ids()

    first = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
    slots = [first + timedelta(minutes=30 * (i + 1)) for i in range(FORECAST_HOURS * 2)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_forecast_zone, zone_id, slots, now): zone_id for zone_id in zone_ids}
        for future in as_completed(futures):
            try:
                counts = future.result()
            except Exception as exc:
                logger.error("Forecast worker failed zone=%d: %s", futures[future], exc)
                summary["prediction_failures"] += 1
                summary["skipped_zones"] += 1
                continue
            for key, count in counts.items():
                summary[key] += count

    logger.info(
        "Forecasts created=%d conflicts=%d POST failures=%d history failures=%d "
        "prediction failures=%d skipped zones=%d",
        summary["created"], summary["conflicts"], summary["failures"],
        summary["history_failures"], summary["prediction_failures"], summary["skipped_zones"],
    )
    return summary
