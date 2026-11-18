import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier, Lock, get_ident, local
from unittest.mock import Mock, patch

import pandas as pd

with patch("dotenv.load_dotenv"):
    from parktrack_ml import forecaster
    from parktrack_ml.interfaces import PredictOutput


NOW = datetime(2026, 11, 18, 10, 12, 34, 123456, tzinfo=timezone.utc)


def history(zone_id):
    return pd.DataFrame([{
        "zone_id": zone_id,
        "observed_at": NOW.replace(minute=0, second=0, microsecond=0) - timedelta(minutes=10),
        "occupied": 3, "capacity": 10, "occupancy_rate": 0.3, "confidence": 0.8,
    }])


def prediction(**changes):
    result = PredictOutput(
        occupancy_class="Medium", class_code=1, confidence=0.6,
        prob_low=0.3, prob_medium=0.6, prob_high=0.1,
        predicted_occupied=5, probability_free_space=0.9, capacity=10,
    )
    return replace(result, **changes)


class ForecasterTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(forecaster, "API_TOKEN", "test-token"))
        self.stack.enter_context(patch.object(forecaster, "FORECAST_HOURS", 48))
        self.clock = self.stack.enter_context(patch.object(forecaster, "datetime"))
        self.clock.now.return_value = NOW
        self.zones = self.stack.enter_context(patch.object(forecaster, "_active_zone_ids", return_value=[1]))
        self.client = Mock()
        self.client.post_forecast.return_value = 123
        self.stack.enter_context(patch.object(forecaster, "_get_client", return_value=self.client))
        self.history = self.stack.enter_context(patch.object(
            forecaster, "load_recent_observations",
            side_effect=lambda zone_id, *args, **kwargs: history(zone_id),
        ))
        self.predict = self.stack.enter_context(patch.object(
            forecaster, "predict_many",
            side_effect=lambda **kwargs: [prediction()] * len(kwargs["predicted_for"]),
        ))

    def test_run_predicts_once_per_zone_and_posts_96_unique_future_slots(self):
        self.zones.return_value = [1, 2]
        summary = forecaster.run()
        self.assertEqual(summary["created"], 192)
        self.assertEqual(summary["conflicts"], 0)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["skipped_zones"], 0)
        self.assertEqual(self.predict.call_count, 2)
        self.assertEqual(self.history.call_count, 2)
        self.clock.now.assert_called_once_with(timezone.utc)
        for call in self.predict.call_args_list:
            args = call.kwargs
            self.assertEqual(args["as_of"], NOW)
            self.assertEqual(args["recent_hourly"]["zone_id"].tolist(), [args["zone_id"]])
            slots = args["predicted_for"]
            self.assertEqual(len(slots), 96)
            self.assertEqual(len(set(slots)), 96)
            self.assertEqual(slots, sorted(slots))
            self.assertTrue(all(NOW < slot <= NOW + timedelta(hours=48) for slot in slots))
            self.assertTrue(all(slot.minute in (0, 30) and slot.second == slot.microsecond == 0 for slot in slots))
            self.assertTrue(all(b - a == timedelta(minutes=30) for a, b in zip(slots, slots[1:])))
        for call in self.history.call_args_list:
            self.assertEqual(call.args[1], NOW)
            self.assertEqual(call.kwargs, {"hours": forecaster.LOOKBACK_HOURS, "as_of": NOW})
        self.assertEqual(self.client.post_forecast.call_count, 192)

    def test_generated_at_and_metadata_use_single_run_time(self):
        forecaster.run()
        for call in self.client.post_forecast.call_args_list:
            payload = call.kwargs
            self.assertEqual(payload["generated_at"], NOW)
            self.assertEqual(payload["metadata"]["as_of"], NOW.isoformat())
            self.assertEqual(payload["metadata"]["forecast_semantics"], "hourly_mean_proxy")
            self.assertEqual(payload["metadata"]["predicted_occupied_method"], "regression")
            self.assertEqual(payload["metadata"]["occupancy_class_method"], "regression_class")
            self.assertEqual(payload["metadata"]["class_probability_method"], "classifier")
            self.assertEqual(payload["metadata"]["probability_free_space_calibration"], "not_calibrated")
            self.assertNotIn("data_until", payload["metadata"])
            self.assertNotIn("artifact_dir", payload["metadata"])

    def test_exact_half_hour_boundary_starts_with_next_slot(self):
        now = NOW.replace(minute=30, second=0, microsecond=0)
        self.clock.now.return_value = now
        forecaster.run()
        slots = self.predict.call_args.kwargs["predicted_for"]
        self.assertEqual(slots[0], now + timedelta(minutes=30))
        self.assertEqual(slots[-1], now + timedelta(hours=48))

    def test_conflicts_and_post_failures_are_separate_and_not_retried(self):
        self.stack.enter_context(patch.object(forecaster, "FORECAST_HOURS", 2))
        self.client.post_forecast.side_effect = [10, None, RuntimeError("temporary failure"), 11]
        with self.assertLogs("parktrack_ml.forecaster", level="WARNING") as logs:
            summary = forecaster.run()
        self.assertEqual(summary["created"], 2)
        self.assertEqual(summary["conflicts"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["prediction_failures"], 0)
        self.assertEqual(self.client.post_forecast.call_count, 4)
        self.assertTrue(any("409" in line and "existing version retained" in line for line in logs.output))
        self.assertTrue(any("POST failed" in line for line in logs.output))

    def test_failed_history_is_logged_and_never_predicts_or_posts(self):
        self.history.side_effect = RuntimeError("history unavailable")
        with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
            summary = forecaster.run()
        self.assertEqual(summary["history_failures"], 1)
        self.assertEqual(summary["skipped_zones"], 1)
        self.assertEqual(summary["created"], 0)
        self.assertEqual(summary["failures"], 0)
        self.predict.assert_not_called()
        self.client.post_forecast.assert_not_called()

    def test_empty_history_skips_zone_without_predictions_or_posts(self):
        self.history.side_effect = None
        self.history.return_value = history(1).iloc[:0]
        with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
            summary = forecaster.run()
        self.assertEqual(summary["history_failures"], 1)
        self.assertEqual(summary["skipped_zones"], 1)
        self.predict.assert_not_called()
        self.client.post_forecast.assert_not_called()

    def test_empty_aggregation_skips_zone(self):
        with patch.object(forecaster, "aggregate_hourly", return_value=pd.DataFrame()), self.assertLogs(
            "parktrack_ml.forecaster", level="ERROR",
        ):
            summary = forecaster.run()
        self.assertEqual(summary["history_failures"], 1)
        self.predict.assert_not_called()
        self.client.post_forecast.assert_not_called()

    def test_failed_history_does_not_prevent_other_zone_from_posting(self):
        self.zones.return_value = [1, 2]

        def load(zone_id, *args, **kwargs):
            if zone_id == 1:
                raise RuntimeError("history unavailable")
            return history(zone_id)

        self.history.side_effect = load
        with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
            summary = forecaster.run()
        self.assertEqual(summary["created"], 96)
        self.assertEqual(summary["skipped_zones"], 1)
        self.predict.assert_called_once()
        self.assertEqual(self.predict.call_args.kwargs["zone_id"], 2)

    def test_prediction_failure_skips_zone_without_posts(self):
        self.predict.side_effect = RuntimeError("model unavailable")
        with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
            summary = forecaster.run()
        self.assertEqual(summary["prediction_failures"], 1)
        self.assertEqual(summary["skipped_zones"], 1)
        self.client.post_forecast.assert_not_called()

    def test_prediction_count_mismatch_never_partially_posts(self):
        self.predict.side_effect = None
        self.predict.return_value = [prediction()]
        with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
            summary = forecaster.run()
        self.assertEqual(summary["prediction_failures"], 1)
        self.client.post_forecast.assert_not_called()

    def test_invalid_last_prediction_never_posts_earlier_valid_predictions(self):
        for changes in (
            {"probability_free_space": float("nan")}, {"confidence": float("inf")},
            {"prob_high": -0.1}, {"prob_low": 1.1},
            {"capacity": 0}, {"predicted_occupied": float("nan")},
            {"predicted_occupied": 11}, {"predicted_occupied": 2.5},
        ):
            with self.subTest(changes=changes):
                self.predict.side_effect = lambda **kwargs: (
                    [prediction()] * (len(kwargs["predicted_for"]) - 1) + [prediction(**changes)]
                )
                with self.assertLogs("parktrack_ml.forecaster", level="ERROR"):
                    summary = forecaster.run()
                self.assertEqual(summary["prediction_failures"], 1)
                self.client.post_forecast.assert_not_called()

    def test_pool_submits_one_zone_task_and_uses_at_most_four_workers(self):
        self.zones.return_value = [1, 2, 3, 4, 5]
        with patch.object(forecaster, "ThreadPoolExecutor", wraps=ThreadPoolExecutor) as executor:
            summary = forecaster.run()
        executor.assert_called_once_with(max_workers=4)
        self.assertEqual(summary["created"], 480)
        self.assertEqual(self.predict.call_count, 5)
        self.assertEqual(self.history.call_count, 5)

    def test_missing_token_skips_all_work(self):
        with patch.object(forecaster, "API_TOKEN", ""), self.assertLogs(
            "parktrack_ml.forecaster", level="WARNING",
        ):
            summary = forecaster.run()
        self.assertEqual(summary["created"], 0)
        self.zones.assert_not_called()
        self.history.assert_not_called()
        self.predict.assert_not_called()
        self.client.post_forecast.assert_not_called()


class ClientThreadSafetyTests(unittest.TestCase):
    def test_each_thread_owns_and_reuses_its_client(self):
        barrier = Barrier(4)
        client_threads = []
        lock = Lock()

        def make_client(*args):
            with lock:
                client_threads.append(get_ident())
            return Mock()

        def clients_in_thread(_):
            barrier.wait(timeout=5)
            first = forecaster._get_client()
            second = forecaster._get_client()
            return get_ident(), first, second

        with patch.object(forecaster, "_clients", local()), patch.object(
            forecaster, "ParkTrackClient", side_effect=make_client,
        ) as factory, ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(clients_in_thread, range(4)))
        self.assertEqual(factory.call_count, 4)
        self.assertEqual(len(set(client_threads)), 4)
        self.assertEqual(len({id(first) for _, first, _ in results}), 4)
        for _, first, second in results:
            self.assertIs(first, second)

    def test_active_zones_are_deduplicated(self):
        client = Mock()
        client.get_zones.return_value = [
            {"parking_zone_id": 2}, {"id": 2}, {"zone_id": 1},
            {"id": 3, "is_active": False}, {},
        ]
        with patch.object(forecaster, "_get_client", return_value=client):
            self.assertEqual(forecaster._active_zone_ids(), [1, 2])


if __name__ == "__main__":
    unittest.main()
