"""ParkTrack API client with Bearer token auth."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)


class ParkTrackClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_zones(self) -> list[dict]:
        resp = self._session.get(f"{self._base}/zones", timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def get_occupancy(
        self,
        zone_id: int | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        view: str = "points",
    ) -> list[dict]:
        params: dict[str, Any] = {"view": view}
        if zone_id is not None:
            params["zone_id"] = zone_id
        if from_dt is not None:
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if to_dt is not None:
            params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        resp = self._session.get(
            f"{self._base}/occupancy", params=params, timeout=self._timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def post_forecast(
        self,
        zone_id: int,
        generated_at: datetime,
        predicted_for: datetime,
        predicted_occupied: int,
        probability_free_space: float,
        confidence: float,
        capacity: int | None = None,
        model_version: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """Return forecast_id on success, None on 409 conflict."""
        payload: dict[str, Any] = {
            "zone_id": zone_id,
            "model_type": "ml_model",
            "generated_at": _fmt(generated_at),
            "predicted_for": _fmt(predicted_for),
            "predicted_occupied": predicted_occupied,
            "probability_free_space": round(float(probability_free_space), 4),
            "confidence": round(float(confidence), 4),
        }
        if capacity is not None:
            payload["capacity"] = capacity
        if model_version is not None:
            payload["model_version"] = model_version
        if metadata is not None:
            payload["metadata"] = metadata

        resp = self._session.post(
            f"{self._base}/forecasts/new", json=payload, timeout=self._timeout
        )

        if resp.status_code == 409:
            log.debug("Forecast already exists for zone=%d predicted_for=%s", zone_id, _fmt(predicted_for))
            return None

        resp.raise_for_status()
        return resp.json()["forecast_id"]


def _fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
