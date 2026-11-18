"""ParkTrack API client with Bearer token auth."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import sleep
from typing import Any

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)


_GET_ATTEMPTS = 3
_GET_BACKOFF_SECONDS = 0.5
_GET_BACKOFF_MAX = 4.0


def _raise_for_status(response: requests.Response) -> None:
    response.raise_for_status()
    if 300 <= response.status_code < 400:
        raise requests.HTTPError("Unexpected redirect from ParkTrack API", response=response)


class ParkTrackClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=0))
        self._session.mount("http://", HTTPAdapter(max_retries=0))
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    def _get(self, path: str, **kwargs) -> list[dict]:
        for attempt in range(_GET_ATTEMPTS):
            response = None
            try:
                response = self._session.get(
                    f"{self._base}{path}", timeout=self._timeout,
                    stream=True, allow_redirects=False, **kwargs,
                )
                _raise_for_status(response)
                return response.json()
            except requests.RequestException as exc:
                transient = isinstance(exc, (
                    requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                )) and not isinstance(exc, requests.exceptions.SSLError)
                if isinstance(exc, requests.HTTPError):
                    transient = response is not None and (
                        response.status_code == 429 or 500 <= response.status_code < 600
                    )
                if not transient or attempt + 1 == _GET_ATTEMPTS:
                    raise
            finally:
                if response is not None:
                    response.close()
            sleep(min(_GET_BACKOFF_SECONDS * 2**attempt, _GET_BACKOFF_MAX))

    # ------------------------------------------------------------------
    # Zones & cameras
    # ------------------------------------------------------------------

    def get_zones(self) -> list[dict]:
        return self._get("/zones")

    def get_cameras(self) -> list[dict]:
        return self._get("/cameras")

    # ------------------------------------------------------------------
    # Occupancy
    # ------------------------------------------------------------------

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
            params["from"] = _fmt(from_dt)
        if to_dt is not None:
            params["to"] = _fmt(to_dt)

        return self._get("/occupancy", params=params)

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def get_weather(
        self,
        camera_id: int | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        latest_only: bool = False,
    ) -> list[dict]:
        params: dict[str, Any] = {}
        if camera_id is not None:
            params["camera_id"] = camera_id
        if from_dt is not None:
            params["from"] = _fmt(from_dt)
        if to_dt is not None:
            params["to"] = _fmt(to_dt)
        if latest_only:
            params["latest_only"] = "true"

        return self._get("/weather", params=params)

    def post_weather(
        self,
        camera_id: int,
        observed_at: datetime,
        temperature: float,
        precipitation: float,
    ) -> bool:
        """POST /weather/new. Returns False on 409 (already exists), True on success."""
        payload: dict[str, Any] = {
            "camera_id": camera_id,
            "observed_at": _fmt(observed_at.replace(minute=0, second=0, microsecond=0)),
            "temperature": round(float(temperature), 2),
            "precipitation": round(float(max(0.0, precipitation)), 2),
        }
        resp = self._session.post(
            f"{self._base}/weather/new", json=payload, timeout=self._timeout, allow_redirects=False,
        )
        if resp.status_code == 409:
            return False
        _raise_for_status(resp)
        return True

    # ------------------------------------------------------------------
    # Forecasts
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
        model_type: str = "ml_model",
        model_version: str | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """Return forecast_id on success, None on 409 conflict."""
        payload: dict[str, Any] = {
            "zone_id": zone_id,
            "model_type": model_type,
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
            f"{self._base}/forecasts/new", json=payload, timeout=self._timeout, allow_redirects=False,
        )

        if resp.status_code == 409:
            log.debug(
                "Forecast already exists for zone=%d predicted_for=%s",
                zone_id,
                _fmt(predicted_for),
            )
            return None

        _raise_for_status(resp)
        return resp.json()["forecast_id"]


def _fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if dt.microsecond else "%Y-%m-%dT%H:%M:%SZ"
    return dt.strftime(fmt)
