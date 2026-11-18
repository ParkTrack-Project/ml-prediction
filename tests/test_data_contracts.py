import io
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import NewConnectionError, ProtocolError, ReadTimeoutError
from urllib3.response import HTTPResponse

with patch("dotenv.load_dotenv"):
    from parktrack_ml.api_client import ParkTrackClient, _fmt
    from parktrack_ml.data_loader import (
        _fetch_chunked,
        _parse_occupancy,
        aggregate_hourly,
        load_recent_observations,
    )


UTC = timezone.utc
HOUR = datetime(2026, 1, 10, 10, tzinfo=UTC)


def observation(occupied=2, capacity=10, **extra):
    return {
        "zone_id": 1,
        "observed_at": HOUR.isoformat(),
        "occupied": occupied,
        "capacity": capacity,
        **extra,
    }


def response(status=200):
    return HTTPResponse(
        body=io.BytesIO(b"[]"), status=status,
        headers={"Content-Length": "2", "Retry-After": "3600"},
        preload_content=False,
    )


class APIContractsTests(unittest.TestCase):
    def setUp(self):
        self.client = ParkTrackClient("http://parktrack.invalid/api/v1", "test-token")
        self.client._session.trust_env = False
        self.addCleanup(self.client._session.close)

    def test_fmt_converts_aware_time_and_accepts_naive_utc(self):
        east = timezone(timedelta(hours=5, minutes=30))
        self.assertEqual(_fmt(datetime(2026, 1, 10, 0, 15, tzinfo=east)), "2026-01-09T18:45:00Z")
        self.assertEqual(_fmt(datetime(2026, 1, 10, 10)), "2026-01-10T10:00:00Z")

    def test_fmt_preserves_fractional_seconds_and_distinct_versions(self):
        first = HOUR.replace(microsecond=1)
        second = HOUR.replace(microsecond=999999)
        self.assertEqual(_fmt(first), "2026-01-10T10:00:00.000001Z")
        self.assertEqual(_fmt(second), "2026-01-10T10:00:00.999999Z")
        self.assertNotEqual(_fmt(first), _fmt(second))
        self.assertEqual(_fmt(first.astimezone(timezone(timedelta(hours=5, minutes=30)))), _fmt(first))

    def test_get_payload_formats_offsets_as_utc(self):
        east = timezone(timedelta(hours=3))
        with patch.object(self.client._session, "get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = []
            self.client.get_occupancy(1, HOUR.astimezone(east), HOUR + timedelta(hours=1))
        self.assertEqual(get.call_args.kwargs["params"], {
            "view": "points", "zone_id": 1,
            "from": "2026-01-10T10:00:00Z", "to": "2026-01-10T11:00:00Z",
        })

    def test_get_recovers_after_rate_limit_and_server_error(self):
        with patch.object(HTTPConnectionPool, "_make_request", side_effect=[
            response(429), response(503), response(),
        ]) as request, patch("parktrack_ml.api_client.sleep") as sleep:
            self.assertEqual(self.client.get_zones(), [])
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.5, 1.0])

    def test_get_retries_transient_network_errors(self):
        for error in (
            NewConnectionError(None, "temporary connection failure"),
            ProtocolError("temporary disconnect"),
            ReadTimeoutError(None, "/zones", "temporary timeout"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(HTTPConnectionPool, "_make_request", side_effect=[
                    error, response(),
                ]) as request, patch("parktrack_ml.api_client.sleep"):
                    self.assertEqual(self.client.get_zones(), [])
                self.assertEqual(request.call_count, 2)

    def test_get_stops_after_three_attempts(self):
        with patch.object(HTTPConnectionPool, "_make_request", side_effect=[
            response(500), response(500), response(500),
        ]) as request, patch("parktrack_ml.api_client.sleep"):
            with self.assertRaises(requests.HTTPError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 3)

    def test_get_network_failure_stops_after_three_attempts(self):
        with patch.object(HTTPConnectionPool, "_make_request", side_effect=NewConnectionError(
            None, "temporary connection failure",
        )) as request, patch("parktrack_ml.api_client.sleep"):
            with self.assertRaises(requests.ConnectionError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 3)

    def test_get_does_not_retry_nontransient_client_error(self):
        with patch.object(HTTPConnectionPool, "_make_request", return_value=response(400)) as request:
            with self.assertRaises(requests.HTTPError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 1)

    def test_post_is_not_retried_on_server_or_connection_error(self):
        for failure, exception in (
            (response(503), requests.HTTPError),
            (NewConnectionError(None, "temporary connection failure"), requests.ConnectionError),
        ):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(HTTPConnectionPool, "_make_request", side_effect=[failure]) as request:
                    with self.assertRaises(exception):
                        self.client.post_weather(1, HOUR, 2.0, 0.0)
                self.assertEqual(request.call_count, 1)

    def test_post_forecast_formats_utc_and_preserves_conflict_result(self):
        with patch.object(self.client._session, "post") as post:
            post.return_value.status_code = 409
            result = self.client.post_forecast(
                1, HOUR.astimezone(timezone(timedelta(hours=-4))), HOUR,
                2, 0.8, 0.7,
            )
        self.assertIsNone(result)
        self.assertEqual(post.call_args.kwargs["json"]["generated_at"], "2026-01-10T10:00:00Z")

    def test_retry_backoff_has_a_cap(self):
        with patch.object(HTTPConnectionPool, "_make_request", side_effect=[
            response(503), response(503), response(),
        ]), patch("parktrack_ml.api_client._GET_BACKOFF_SECONDS", 10), patch(
            "parktrack_ml.api_client.sleep",
        ) as sleep:
            self.assertEqual(self.client.get_zones(), [])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [4.0, 4.0])

    def test_get_retries_interrupted_response_body(self):
        broken = response()
        with patch.object(broken, "stream", side_effect=ProtocolError("body interrupted")), patch.object(
            broken, "close", wraps=broken.close,
        ) as close, patch.object(HTTPConnectionPool, "_make_request", side_effect=[
            broken, response(),
        ]) as request, patch("parktrack_ml.api_client.sleep"):
            self.assertEqual(self.client.get_zones(), [])
        self.assertEqual(request.call_count, 2)
        close.assert_called()

    def test_get_stops_after_three_body_failures(self):
        broken = [response() for _ in range(3)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for raw in broken:
                stack.enter_context(patch.object(raw, "stream", side_effect=ProtocolError("body interrupted")))
            request = stack.enter_context(patch.object(HTTPConnectionPool, "_make_request", side_effect=broken))
            sleep = stack.enter_context(patch("parktrack_ml.api_client.sleep"))
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_get_shares_one_attempt_budget_for_connection_status_and_body_failures(self):
        broken = response()
        with patch.object(broken, "stream", side_effect=ProtocolError("body interrupted")), patch.object(
            HTTPConnectionPool, "_make_request", side_effect=[
                NewConnectionError(None, "temporary failure"), response(429), broken,
            ],
        ) as request, patch("parktrack_ml.api_client.sleep") as sleep:
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_post_does_not_retry_response_body_failure(self):
        broken = response()
        with patch.object(broken, "stream", side_effect=ProtocolError("body interrupted")), patch.object(
            HTTPConnectionPool, "_make_request", return_value=broken,
        ) as request, patch("parktrack_ml.api_client.sleep") as sleep:
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                self.client.post_forecast(1, HOUR, HOUR, 2, 0.8, 0.7)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_redirects_do_not_replay_get_or_post(self):
        for method in (self.client.get_zones, lambda: self.client.post_weather(1, HOUR, 2.0, 0.0)):
            with self.subTest(method=method):
                redirect = response(307)
                redirect.headers["Location"] = "http://parktrack.invalid/other"
                with patch.object(HTTPConnectionPool, "_make_request", return_value=redirect) as request:
                    with self.assertRaises(requests.HTTPError):
                        method()
                self.assertEqual(request.call_count, 1)

    def test_get_does_not_retry_invalid_json(self):
        invalid = HTTPResponse(body=io.BytesIO(b"invalid"), status=200, preload_content=False)
        with patch.object(HTTPConnectionPool, "_make_request", return_value=invalid) as request:
            with self.assertRaises(requests.exceptions.JSONDecodeError):
                self.client.get_zones()
        self.assertEqual(request.call_count, 1)

    def test_post_forecast_preserves_generated_at_microseconds(self):
        with patch.object(self.client._session, "post") as post:
            post.return_value.status_code = 201
            post.return_value.json.return_value = {"forecast_id": 42}
            self.assertEqual(self.client.post_forecast(1, HOUR.replace(microsecond=123), HOUR, 2, 0.8, 0.7), 42)
        self.assertEqual(post.call_args.kwargs["json"]["generated_at"], "2026-01-10T10:00:00.000123Z")


class OccupancyContractsTests(unittest.TestCase):
    def test_invalid_counts_are_dropped_instead_of_imputed(self):
        invalid = []
        for field in ("occupied", "capacity"):
            for value in (None, "broken", np.nan, np.inf, -np.inf, 1.5, False, True, 2**63):
                invalid.append(observation(**{field: value}))
            missing = observation()
            del missing[field]
            invalid.append(missing)
        invalid.extend([observation(occupied=-1), observation(capacity=0), observation(capacity=-1)])
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING") as logs:
            result = _parse_occupancy(invalid + [observation(occupied="0", capacity="10")])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["occupied"], 0)
        self.assertIn(f"Dropped {len(invalid)} invalid", logs.output[0])

    def test_absent_required_fields_and_invalid_identity_are_dropped(self):
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING"):
            self.assertTrue(_parse_occupancy([{}]).empty)
            self.assertTrue(_parse_occupancy([
                observation(zone_id=np.inf), observation(zone_id=1.5),
                observation(observed_at="bad timestamp"),
            ]).empty)

    def test_boolean_counts_are_rejected_when_entire_column_is_boolean(self):
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING"):
            result = _parse_occupancy([observation(occupied=False, capacity=True)])
        self.assertTrue(result.empty)

    def test_over_capacity_is_clipped_and_logged(self):
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING") as logs:
            result = _parse_occupancy([observation(occupied=12)])
        self.assertEqual(result.iloc[0]["occupied"], 10)
        self.assertEqual(result.iloc[0]["occupancy_rate"], 1.0)
        self.assertTrue(any("Clipped 1" in message for message in logs.output))

    def test_missing_and_invalid_confidence_remain_unknown(self):
        rows = [observation(zone_id=1)]
        for zone_id, confidence in enumerate((None, "bad", np.inf, -0.1, 1.1), start=2):
            rows.append(observation(zone_id=zone_id, confidence=confidence))
        result = _parse_occupancy(rows)
        self.assertTrue(result["confidence"].isna().all())

    def test_same_occupied_prefers_valid_confidence_without_ranking_its_value(self):
        result = _parse_occupancy([
            observation(), observation(confidence=0.2), observation(confidence=0.9),
            observation(occupied=0, zone_id=2),
            observation(occupied=0, zone_id=2, confidence=0),
        ])
        self.assertEqual(result["confidence"].tolist(), [0.2, 0.0])

    def test_conflicting_counts_without_ingestion_keep_first_even_if_empty(self):
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING") as logs:
            result = _parse_occupancy([
                observation(occupied=0, confidence=0), observation(occupied=7, confidence=0.99),
            ])
        self.assertEqual(result.iloc[0]["occupied"], 0)
        self.assertIn("Conflicting occupied counts in 1", logs.output[0])

    def test_conflicting_counts_use_latest_ingestion_in_utc(self):
        for field in ("ingested_at", "received_at", "created_at"):
            with self.subTest(field=field):
                with self.assertLogs("parktrack_ml.data_loader", level="WARNING"):
                    result = _parse_occupancy([
                        observation(occupied=8, confidence=0.99, **{field: "2026-01-10T13:05:00+03:00"}),
                        observation(occupied=0, confidence=0, **{field: "2026-01-10T10:10:00Z"}),
                        observation(occupied=9, confidence=1, **{field: "broken"}),
                    ])
                self.assertEqual(result.iloc[0]["occupied"], 0)

    def test_equal_ingestion_timestamps_use_stable_first(self):
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING"):
            result = _parse_occupancy([
                observation(occupied=0, ingested_at=HOUR.isoformat()),
                observation(occupied=4, confidence=1, ingested_at=HOUR.isoformat()),
            ])
        self.assertEqual(result.iloc[0]["occupied"], 0)

    def test_mixed_time_formats_and_offsets_identify_same_observation(self):
        result = _parse_occupancy([
            observation(observed_at="2026-01-10T13:00:00+03:00"),
            observation(observed_at="2026-01-10T10:00:00.000Z", confidence=0.4),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["observed_at"], pd.Timestamp(HOUR))
        self.assertEqual(result.iloc[0]["confidence"], 0.4)

    def test_duplicates_across_chunks_keep_latest_ingestion(self):
        client = Mock()
        client.get_occupancy.side_effect = [
            [observation(occupied=8, ingested_at="2026-01-10T10:01:00Z")],
            [observation(occupied=0, ingested_at="2026-01-10T10:02:00Z")],
        ]
        with self.assertLogs("parktrack_ml.data_loader", level="WARNING"):
            result = _fetch_chunked(client, 1, HOUR, HOUR + timedelta(days=5))
        self.assertEqual(client.get_occupancy.call_count, 2)
        self.assertEqual(result.iloc[0]["occupied"], 0)

    def test_hourly_mean_counts_and_observed_free_fraction(self):
        raw = _parse_occupancy([
            observation(occupied=0, confidence=0),
            observation(occupied=5, confidence=0.2, observed_at=HOUR + timedelta(minutes=10)),
            observation(occupied=10, confidence=0.8, observed_at=HOUR + timedelta(minutes=20)),
        ])
        row = aggregate_hourly(raw).iloc[0]
        self.assertEqual(row["hour"], pd.Timestamp(HOUR))
        self.assertEqual(row["n_obs"], 3)
        self.assertEqual(row["occupancy_rate"], 0.5)
        self.assertEqual(row["capacity"], 10)
        self.assertEqual(row["confidence"], 0.5)
        self.assertAlmostEqual(row["free_fraction"], 2 / 3)

    def test_hourly_missing_confidence_is_nan_with_cars_and_without(self):
        raw = _parse_occupancy([
            observation(occupied=0, confidence=0), observation(occupied=4, zone_id=2),
        ])
        self.assertTrue(aggregate_hourly(raw)["confidence"].isna().all())
        self.assertTrue(aggregate_hourly(raw.drop(columns="confidence"))["confidence"].isna().all())

    def test_empty_schema_preserves_counts_and_free_fraction(self):
        result = aggregate_hourly(_parse_occupancy([]))
        self.assertTrue(result.empty)
        self.assertTrue({"n_obs", "confidence", "free_fraction"}.issubset(result.columns))


class RecentObservationsTests(unittest.TestCase):
    def load_at(self, before, *, now=HOUR + timedelta(minutes=45), **kwargs):
        raw = _parse_occupancy([
            observation(observed_at=HOUR - timedelta(seconds=1)),
            observation(observed_at=HOUR),
            observation(observed_at=HOUR + timedelta(minutes=20)),
        ])
        with patch("parktrack_ml.data_loader.datetime") as clock, patch(
            "parktrack_ml.data_loader.load_observations", return_value=raw,
        ) as load:
            clock.now.return_value = now
            result = load_recent_observations(1, before, **kwargs)
        clock.now.assert_called_once_with(UTC)
        return result, load.call_args.kwargs

    def test_future_prediction_uses_now_completed_hour_and_strict_filter(self):
        result, params = self.load_at(HOUR + timedelta(days=1), hours=25)
        self.assertEqual(params, {
            "zone_ids": [1], "from_dt": HOUR - timedelta(hours=25), "to_dt": HOUR,
        })
        self.assertEqual(result["observed_at"].tolist(), [pd.Timestamp(HOUR - timedelta(seconds=1))])

    def test_as_of_limits_history_and_normalizes_offset(self):
        east = timezone(timedelta(hours=3))
        result, params = self.load_at(
            HOUR + timedelta(days=1),
            as_of=(HOUR - timedelta(minutes=10)).astimezone(east),
        )
        self.assertEqual(params["to_dt"], HOUR - timedelta(hours=1))
        self.assertTrue(result.empty)

    def test_future_as_of_cannot_extend_real_now(self):
        _, params = self.load_at(HOUR + timedelta(days=2), as_of=HOUR + timedelta(days=1))
        self.assertEqual(params["to_dt"], HOUR)

    def test_historical_exact_cutoff_is_available(self):
        cutoff = HOUR + timedelta(minutes=20)
        result, params = self.load_at(cutoff, completed_hour=False)
        self.assertEqual(params["to_dt"], cutoff)
        self.assertEqual(len(result), 2)
        self.assertTrue((result["observed_at"] < cutoff).all())

    def test_naive_before_is_interpreted_as_utc_and_floored(self):
        _, params = self.load_at(datetime(2026, 1, 10, 9, 37))
        self.assertEqual(params["to_dt"], HOUR - timedelta(hours=1))

    def test_old_positional_call_and_empty_results_are_supported(self):
        with patch("parktrack_ml.data_loader.datetime") as clock, patch(
            "parktrack_ml.data_loader.load_observations", return_value=_parse_occupancy([]),
        ) as load:
            clock.now.return_value = HOUR + timedelta(minutes=45)
            result = load_recent_observations(1, HOUR, 12)
        self.assertTrue(result.empty)
        self.assertEqual(load.call_args.kwargs["from_dt"], HOUR - timedelta(hours=12))

    def test_mocked_api_payload_is_filtered_before_aggregation(self):
        client = Mock()
        client.get_occupancy.return_value = [
            observation(occupied=0, confidence=0, observed_at=HOUR - timedelta(minutes=30)),
            observation(occupied=5, confidence=0.8, observed_at=HOUR - timedelta(minutes=10)),
            observation(occupied=10, confidence=1, observed_at=HOUR),
            observation(occupied=10, confidence=1, observed_at=HOUR + timedelta(minutes=10)),
        ]
        with patch("parktrack_ml.data_loader.datetime") as clock, patch(
            "parktrack_ml.data_loader._client", return_value=client,
        ):
            clock.now.return_value = HOUR + timedelta(minutes=45)
            raw = load_recent_observations(1, HOUR + timedelta(days=1))
        self.assertEqual(client.get_occupancy.call_args.kwargs["to_dt"], HOUR)
        hourly = aggregate_hourly(raw)
        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly.iloc[0]["n_obs"], 2)
        self.assertEqual(hourly.iloc[0]["occupancy_rate"], 0.25)
        self.assertEqual(hourly.iloc[0]["free_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
