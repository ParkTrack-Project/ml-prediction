import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from parktrack_ml.features_h import (
    FEATURES_H, _calendar, _history_grid, add_profile_features,
    bucket_for_gap, build_horizon_dataset, build_inference_vector,
    compute_profiles, label_rate,
)
from parktrack_ml.predict import predict_many_for_zone


def history(periods=24 * 35):
    times = pd.date_range("2026-06-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({
        "zone_id": 8, "hour": times,
        "occupancy_rate": (np.sin(np.arange(periods) / 24) + 1) / 2,
        "capacity": 20, "n_obs": 20, "confidence": 0.9,
    })


class OriginFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.data = history()
        self.meta = pd.DataFrame([{"zone_id": 8, "capacity": 20, "zone_type_standard": 1}])
        self.zone = {"capacity": 20, "zone_type_standard": 1}

    def test_gap_two_uses_fresh_measurement(self):
        origin = self.data.hour.max()
        H, values = build_inference_vector(8, origin + pd.Timedelta(hours=2), self.data, self.zone, {}, origin + pd.Timedelta(hours=1))
        self.assertEqual(H, 3)
        values = dict(zip(FEATURES_H, values))
        self.assertEqual(values["last"], self.data.iloc[-1].occupancy_rate)
        self.assertEqual(values["exact_gap"], 2)

    def test_completed_hours_and_zone_filter(self):
        as_of = self.data.hour.max()
        altered = pd.concat([
            self.data,
            self.data.assign(zone_id=9, occupancy_rate=1),
        ], ignore_index=True)
        _, values = build_inference_vector(8, as_of, altered, self.zone, {}, as_of + pd.Timedelta(minutes=5))
        self.assertEqual(values[FEATURES_H.index("last")], self.data.iloc[-2].occupancy_rate)

    def test_ffill_is_bounded_and_causal(self):
        small = self.data.iloc[:10].drop(index=[3, 5, 6, 7])
        grid = _history_grid(small)
        self.assertEqual(grid.loc[(8, self.data.hour.iloc[3]), "value"], self.data.occupancy_rate.iloc[2])
        self.assertTrue(np.isnan(grid.loc[(8, self.data.hour.iloc[7]), "value"]))
        changed = small.copy()
        changed.loc[changed.hour > self.data.hour.iloc[4], "occupancy_rate"] = 0
        pd.testing.assert_series_equal(grid.loc[(8, slice(None)), "value"].iloc[:5],
                                       _history_grid(changed).loc[(8, slice(None)), "value"].iloc[:5])

    def test_training_and_inference_vectors_match(self):
        d = build_horizon_dataset(self.data, self.meta, 3)
        d = add_profile_features(d, np.ones(len(d), dtype=bool), source=self.data)
        row = d[(d.exact_gap == 2) & (d.hour > self.data.hour.min() + pd.Timedelta(days=20))].iloc[0]
        block = row.origin.floor("7D")
        profiles = compute_profiles(self.data[self.data.hour < block], as_of=block)
        H, vector = build_inference_vector(8, row.hour, self.data, self.zone, profiles,
                                           as_of=row.origin + pd.Timedelta(hours=1))
        self.assertEqual(H, 3)
        np.testing.assert_allclose(vector, row[FEATURES_H].to_numpy(dtype=float), equal_nan=True)

    def test_future_labels_do_not_change_earlier_profiles(self):
        d = build_horizon_dataset(self.data, self.meta, 3)
        first = add_profile_features(d, np.ones(len(d), dtype=bool), source=self.data)
        boundary = self.data.hour.iloc[24 * 25]
        changed = self.data.copy()
        changed.loc[changed.hour >= boundary, "occupancy_rate"] = 1
        second = add_profile_features(d, np.ones(len(d), dtype=bool), source=changed)
        early = first.origin < boundary
        cols = ["prof", "prof_hod", "zone_mean", "prof_last_hod", "anomaly_vs_prof"]
        pd.testing.assert_frame_equal(first.loc[early, cols], second.loc[early, cols])

    def test_local_weekday_boundary(self):
        cal = _calendar(pd.Series(pd.to_datetime(["2026-11-20T20:00Z", "2026-11-20T21:00Z"])))
        np.testing.assert_array_equal(cal["hod"], [23, 0])
        np.testing.assert_array_equal(cal["dow"], [4, 5])
        np.testing.assert_array_equal(cal["is_weekend"], [0, 1])

    def test_timestamp_resolutions_produce_identical_samples(self):
        expected = build_horizon_dataset(self.data, self.meta, 3)
        for unit in ("us", "ms"):
            data = self.data.copy()
            data["hour"] = data.hour.astype(f"datetime64[{unit}, UTC]")
            pd.testing.assert_frame_equal(build_horizon_dataset(data, self.meta, 3), expected)

    def test_hourly_zone_type_column_does_not_break_merge(self):
        data = self.data.assign(zone_type_standard=0)
        meta = self.meta.assign(zone_type_standard=0)
        samples = build_horizon_dataset(data, meta, 1)
        self.assertEqual(samples.zone_type_standard.unique().tolist(), [0])

    def test_capacity_is_known_at_origin_in_training_and_inference(self):
        data = self.data.copy()
        changed_at = data.hour.iloc[-10]
        data.loc[data.hour >= changed_at, "capacity"] = 40
        samples = build_horizon_dataset(data, self.meta.assign(capacity=99), 1)
        self.assertEqual(samples.loc[samples.origin < changed_at, "capacity"].unique().tolist(), [20])
        self.assertEqual(samples.loc[samples.origin >= changed_at, "capacity"].unique().tolist(), [40])
        _, values = build_inference_vector(8, data.hour.max() + pd.Timedelta(hours=1), data,
                                           self.zone, {}, data.hour.max() + pd.Timedelta(hours=1))
        self.assertEqual(values[FEATURES_H.index("capacity")], 40)

    def test_training_preserves_supplied_static_type_without_using_its_capacity(self):
        from parktrack_ml import train
        data = history(24 * 65)
        model = Mock()
        model.temperature = {1: 1.0}
        model.fit_one.return_value = (2, 2)
        model.predict.side_effect = lambda H, X: (np.tile([0.6, 0.3, 0.1], (len(X), 1)), np.full(len(X), 0.2))
        with patch.object(train, "HORIZONS", [1]), patch.object(train, "HorizonModels", return_value=model):
            train.fit_all(data, self.meta.assign(zone_type_standard=0, capacity=999), save=False)
        fit = model.fit_one.call_args.args[1]
        self.assertEqual(fit.zone_type_standard.unique().tolist(), [0])
        self.assertEqual(fit.capacity.unique().tolist(), [20])

    def test_bad_targets_and_stale_history_rejected(self):
        for value in [np.nan, np.inf, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                label_rate([value])
        with self.assertRaises(ValueError):
            bucket_for_gap(73)
        with self.assertRaises(ValueError):
            build_inference_vector(8, self.data.hour.max(), self.data.iloc[:0], self.zone, {})


class BatchPredictionTests(unittest.TestCase):
    class Model:
        def __init__(self):
            self.calls = []
            self.free_calls = []

        def predict(self, H, X):
            self.calls.append((H, len(X)))
            gap = X[:, FEATURES_H.index("exact_gap")]
            rate = gap / 10
            return np.tile([0.05, 0.05, 0.9], (len(X), 1)), rate

        def predict_free(self, H, X):
            self.free_calls.append((H, len(X)))
            return np.full(len(X), 0.8)

    def test_unique_hours_batched_and_interpolated(self):
        data = history()
        as_of = data.hour.max() + pd.Timedelta(hours=1)
        target = as_of + pd.Timedelta(hours=3)
        model = self.Model()
        outputs = predict_many_for_zone(8, [target, target + pd.Timedelta(minutes=30), target + pd.Timedelta(hours=1)],
                                       model, {"capacity": 20}, {}, data, as_of)
        self.assertEqual(model.calls, [(6, 2)])
        self.assertEqual(model.free_calls, [(6, 2)])
        self.assertAlmostEqual(outputs[1]["rate_hat"], 0.45)
        self.assertEqual(outputs[1]["predicted_occupied"], 9)
        self.assertEqual(outputs[1]["class"], "Medium")
        self.assertEqual(outputs[1]["confidence"], 0.05)
        self.assertEqual(outputs[1]["probability_free_space"], 0.8)

    def test_implicit_historical_request_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_many_for_zone(8, [pd.Timestamp("2020-01-01", tz="UTC")], self.Model(), {"capacity": 20}, {}, history())


if __name__ == "__main__":
    unittest.main()
