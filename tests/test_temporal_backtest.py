import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

bt = importlib.import_module("parktrack_ml.backtest")
from parktrack_ml.features_h import FEATURES_H, compute_profiles, training_gaps
from parktrack_ml.config import LOCAL_TIMEZONE


class FakeModels:
    def __init__(self):
        self.temperature = {}
        self.calls = []

    def fit_one(self, H, train, val=None, rounds=None):
        if self.calls or val is None or rounds is not None:
            raise AssertionError("Expected exactly one fit with validation, without refitting")
        self.calls.append((H, train.copy(deep=True), val.copy(deep=True)))
        self.temperature[H] = 1.25
        return 7, 11

    def predict(self, H, X):
        if self.temperature[H] != 1.25:
            raise AssertionError("Selected classifier calibration was replaced")
        return np.tile([0.6, 0.3, 0.1], (len(X), 1)), np.full(len(X), 0.66)

    def predict_free(self, H, X):
        return np.full(len(X), 0.25)


class HourlyInputTests(unittest.TestCase):
    def csv(self, text):
        return bt.load_hourly_csv(io.StringIO(text))

    def test_csv_defaults_unknown_confidence_and_preserves_free_fraction(self):
        result = self.csv("zone_id,hour,occupancy_rate,capacity,free_fraction\n"
                          "1,2026-03-01T00:00:00Z,0.5,10,0.75\n")
        self.assertEqual(result.n_obs.tolist(), [1])
        self.assertTrue(result.confidence.isna().all())
        self.assertEqual(result.free_fraction.tolist(), [0.75])
        self.assertEqual(str(result.hour.dt.tz), "UTC")

    def test_csv_cleans_invalid_rows_and_optional_values(self):
        result = self.csv("zone_id,hour,occupancy_rate,capacity,n_obs,cv_confidence,free_fraction\n"
                          "1,2026-03-01T02:00:00Z,0.5,10,bad,bad,1.2\n"
                          "1,2026-03-01T01:00:00Z,0.3,10,4,0.8,0.25\n"
                          "1,broken,0.5,10,2,0.9,0.2\n"
                          "1,2026-03-01T03:00:00Z,-0.1,10,2,0.9,0.2\n"
                          "1,2026-03-01T04:00:00Z,0.2,0,2,0.9,0.2\n")
        self.assertEqual(result.hour.dt.hour.tolist(), [1, 2])
        self.assertEqual(result.n_obs.tolist(), [4, 1])
        self.assertEqual(result.iloc[0].confidence, 0.8)
        self.assertTrue(pd.isna(result.iloc[1].confidence))
        self.assertTrue(pd.isna(result.iloc[1].free_fraction))

    def test_csv_rejects_duplicates_unaligned_hours_and_missing_columns(self):
        header = "zone_id,hour,occupancy_rate,capacity\n"
        row = "1,2026-03-01T00:00:00Z,0.5,10\n"
        for text in (header + row + row,
                     header + "1,2026-03-01T00:30:00Z,0.5,10\n",
                     "zone_id,hour\n1,2026-03-01\n"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.csv(text)

    def test_metadata_has_empty_schema_and_latest_capacity_instead_of_mode(self):
        empty = bt.zone_meta_from_hourly(pd.DataFrame())
        self.assertEqual(list(empty.columns), ["zone_id", "capacity", "zone_type_standard"])
        self.assertTrue(empty.empty)
        hourly = pd.DataFrame({"zone_id": [1, 1, 1],
                               "hour": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"], utc=True),
                               "occupancy_rate": [0.2] * 3, "capacity": [10, 10, 20]})
        self.assertEqual(bt.zone_meta_from_hourly(hourly.iloc[::-1]).capacity.tolist(), [20])

    def test_unknown_confidence_does_not_exclude_zones(self):
        self.assertEqual(bt.unreliable_zones(pd.DataFrame()), [])
        self.assertEqual(bt.unreliable_zones(pd.DataFrame({"zone_id": [1]})), [])
        data = pd.DataFrame({"zone_id": [1, 2, 3, 4], "confidence": [np.nan, 0.1, "bad", 1.5]})
        self.assertEqual(bt.unreliable_zones(data), [2])


class TemporalIntegrationTests(unittest.TestCase):
    split = pd.Timestamp("2026-03-01", tz="UTC")
    horizons = [1, 6, 24]

    def hourly(self):
        hours = pd.date_range(self.split - pd.Timedelta(days=28),
                              self.split + pd.Timedelta(days=4), freq="h", inclusive="left")
        frames = []
        for zone in (1, 2, 3):
            fit = hours < self.split - pd.Timedelta(days=14)
            capacity = np.where(hours < self.split - pd.Timedelta(days=18), 6, 10)
            capacity = np.where(fit, capacity, 100)
            if zone == 3:
                capacity = np.full(len(hours), 8)
            frames.append(pd.DataFrame({
                "zone_id": zone, "hour": hours,
                "occupancy_rate": 0.15 + ((np.arange(len(hours)) + zone * 3) % 24) / 32,
                "capacity": capacity, "n_obs": 4,
                "confidence": np.where(fit, 0.1 if zone == 2 else 0.9, 0.99),
                "free_fraction": 0.75,
            }))
        return pd.concat(frames, ignore_index=True)

    def run_backtest(self, hourly, *, global_capacity=9999, **kwargs):
        instances, metas, sources = [], [], []
        build, profiles = bt.build_horizon_dataset, bt.add_profile_features

        def factory():
            model = FakeModels()
            instances.append(model)
            return model

        def capture_build(data, meta, H):
            metas.append(meta.copy(deep=True))
            return build(data, meta, H)

        def capture_profiles(data, mask, source=None):
            sources.append(source.copy(deep=True))
            return profiles(data, mask, source=source)

        global_meta = pd.DataFrame({"zone_id": [1, 2, 3], "capacity": global_capacity, "zone_type_standard": 0})
        with patch.object(bt, "HORIZONS", self.horizons), patch.object(bt, "HorizonModels", side_effect=factory), \
                patch.object(bt, "build_horizon_dataset", side_effect=capture_build), \
                patch.object(bt, "add_profile_features", side_effect=capture_profiles):
            result = bt.backtest(hourly, global_meta, self.split, **kwargs)
        return result, instances, metas, sources

    def test_purges_origins_freezes_profiles_and_keeps_selected_artifact(self):
        hourly = self.hourly()
        capacity_at_hour = hourly.set_index(["zone_id", "hour"]).capacity
        result, models, metas, sources = self.run_backtest(hourly)
        val_start = self.split - pd.Timedelta(days=14)
        self.assertEqual(set(result.H), set(self.horizons))
        self.assertTrue((result.origin >= self.split).all())
        self.assertTrue((result.hour >= self.split).all())
        self.assertFalse(result.zone_id.eq(2).any())
        np.testing.assert_array_equal(result.exact_gap, (result.hour - result.origin) / pd.Timedelta(hours=1))
        for model, meta, source in zip(models, metas, sources):
            self.assertEqual(len(model.calls), 1)
            H, fit, val = model.calls[0]
            self.assertTrue((fit.hour < val_start).all())
            self.assertTrue((fit.origin < val_start).all())
            self.assertTrue((val.hour < self.split).all())
            self.assertTrue((val.origin >= val_start).all())
            self.assertTrue((source.hour < val_start).all())
            self.assertEqual(set(source.zone_id), {1, 3})
            self.assertEqual(meta.set_index("zone_id").capacity.to_dict(), {1: 10, 3: 8})
            for name, subset in (("fit", fit), ("val", val), ("test", result[result.H == H])):
                with self.subTest(H=H, partition=name):
                    origin_keys = pd.MultiIndex.from_arrays([subset.zone_id, subset.origin])
                    np.testing.assert_array_equal(subset.capacity.to_numpy(),
                                                  capacity_at_hour.reindex(origin_keys).to_numpy())
            self.assertEqual(set(fit.loc[fit.zone_id == 1, "capacity"]), {6, 10})
            self.assertTrue(val.loc[val.zone_id == 1, "capacity"].eq(100).all())
            self.assertEqual(set(result.loc[result.H == H, "exact_gap"]), set(training_gaps(H)))
            frozen = compute_profiles(source)
            sample = val.iloc[0]
            local = sample.hour.tz_convert(LOCAL_TIMEZONE)
            expected = frozen[int(sample.zone_id)]["dow_hod"][f"{local.dayofweek}_{local.hour}"]
            self.assertAlmostEqual(sample.prof, expected)
        self.assertTrue(result.temperature.eq(1.25).all())
        self.assertTrue(result.pred.eq(0).all())
        self.assertTrue(result.loc[result.zone_id == 1, "pred_reg"].eq(1).all())
        self.assertTrue(result.loc[result.zone_id == 1, "conf"].eq(0.3).all())
        np.testing.assert_allclose(result[["prob_low", "prob_medium", "prob_high"]].sum(axis=1), 1)
        np.testing.assert_array_equal(result.free_target,
                                      (result.occupancy_rate < (result.capacity - 0.5) / result.capacity).astype(int))
        self.assertTrue(result.free_target_kind.eq("hourly_mean_below_rounding_threshold").all())
        self.assertTrue(result.free_calibration.eq("not_calibrated").all())
        self.assertTrue(result.free_fraction.eq(0.75).all())

    def test_validation_perturbation_cannot_change_fit_features_or_profiles(self):
        original = self.hourly()
        changed = original.copy()
        mask = (changed.hour >= self.split - pd.Timedelta(days=14)) & (changed.hour < self.split)
        changed.loc[mask, "occupancy_rate"] = 1 - changed.loc[mask, "occupancy_rate"]
        changed.loc[mask, "capacity"] = 777
        changed.loc[mask, "confidence"] = 0
        _, before, before_meta, before_source = self.run_backtest(original)
        _, after, after_meta, after_source = self.run_backtest(changed)
        for first, second, meta_a, meta_b, source_a, source_b in zip(
                before, after, before_meta, after_meta, before_source, after_source):
            pd.testing.assert_frame_equal(first.calls[0][1], second.calls[0][1])
            pd.testing.assert_frame_equal(meta_a, meta_b)
            pd.testing.assert_frame_equal(source_a, source_b)
            self.assertFalse(first.calls[0][2].label.equals(second.calls[0][2].label))

    def test_test_perturbation_and_global_meta_cannot_change_fit_or_validation(self):
        original = self.hourly()
        changed = original.copy()
        mask = changed.hour >= self.split
        changed.loc[mask, "occupancy_rate"] = 0.99
        changed.loc[mask, "capacity"] = 888
        changed.loc[mask, "confidence"] = 0
        _, before, before_meta, _ = self.run_backtest(original)
        _, after, after_meta, _ = self.run_backtest(changed, global_capacity=1)
        for first, second, meta_a, meta_b in zip(before, after, before_meta, after_meta):
            pd.testing.assert_frame_equal(first.calls[0][1], second.calls[0][1])
            pd.testing.assert_frame_equal(first.calls[0][2], second.calls[0][2])
            pd.testing.assert_frame_equal(meta_a, meta_b)

    def test_test_end_bounds_output_and_empty_fit_fails_before_training(self):
        end = self.split + pd.Timedelta(days=2)
        result, _, _, _ = self.run_backtest(self.hourly(), test_end=end)
        self.assertTrue((result.hour < end).all())
        with self.assertRaisesRegex(ValueError, "test_end"):
            self.run_backtest(self.hourly(), test_end=self.split)
        data = self.hourly()
        with self.assertRaisesRegex(RuntimeError, "No fit observations"):
            self.run_backtest(data[data.hour >= self.split - pd.Timedelta(days=14)])


class ReportTests(unittest.TestCase):
    def rows(self):
        data = pd.DataFrame({
            "H": [1, 1, 1], "label": [0, 1, 2], "occupancy_rate": [0.1, 0.5, 0.9],
            "capacity": [10] * 3, "pred": [0, 0, 0], "pred_reg": [0, 1, 2],
            "rate_hat": [0.2, 0.6, 0.8], "conf": [0.6, 0.3, 0.1], "cls_conf": [0.6] * 3,
            "prob_low": [0.6] * 3, "prob_medium": [0.3] * 3, "prob_high": [0.1] * 3,
            "last": [0.1, 0.4, 0.7], "prof": [0.2, 0.5, 0.8], "lag_week": [np.nan, np.nan, 0.9],
            "free_p": [0.2, 0.4, 0.8], "free_target": [1, 1, 1],
        })
        return pd.concat([data, data.assign(H=6)], ignore_index=True)

    def test_report_returns_all_stream_common_coverage_and_proxy_metrics_for_every_horizon(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            result = bt.report(self.rows())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.H.tolist(), [1, 6])
        for row in result.itertuples(index=False):
            self.assertEqual(row.n_all, 3)
            self.assertEqual(row.n_common, 1)
            self.assertAlmostEqual(row.coverage_common, 1 / 3)
            self.assertAlmostEqual(row.acc_classifier, 1 / 3)
            self.assertEqual(row.acc_reg, 1)
            self.assertAlmostEqual(row.MAE_model, 0.1)
            self.assertEqual(row.n_week_ago, 1)
            self.assertAlmostEqual(row.coverage_week_ago, 1 / 3)
            self.assertEqual(row.recall_classifier_high, 0)
            self.assertEqual(row.recall_reg_high, 1)
            self.assertAlmostEqual(row.predicted_p05, 0.24)
            self.assertAlmostEqual(row.predicted_p95, 0.78)
            self.assertAlmostEqual(row.actual_p05, 0.14)
            self.assertAlmostEqual(row.actual_p95, 0.86)
            self.assertAlmostEqual(row.brier_free_proxy, (0.64 + 0.36 + 0.04) / 3)
            self.assertEqual(row.free_calibration, "not_calibrated")
            self.assertTrue(np.isfinite(row.calibration_ece_classifier))
            self.assertTrue(np.isfinite(row.calibration_ece_reg))
        self.assertEqual(printed.getvalue().count("Calibration"), 4)
        self.assertIn("not instantaneous", printed.getvalue())

    def test_missing_baselines_and_free_head_remain_visible_as_missing(self):
        rows = self.rows()
        rows[["last", "prof", "lag_week", "free_p"]] = np.nan
        with contextlib.redirect_stdout(io.StringIO()):
            result = bt.report(rows)
        self.assertTrue(result.n_all.eq(3).all())
        self.assertTrue(result.n_common.eq(0).all())
        self.assertTrue(result.n_free.eq(0).all())
        self.assertTrue(result.brier_free_proxy.isna().all())
        self.assertTrue(result.acc_reg_common.isna().all())
        self.assertEqual(bt.report(pd.DataFrame()).columns.tolist(), ["H", "n_all", "n_common"])


class CLITests(unittest.TestCase):
    def test_one_or_multiple_chronological_splits_write_predictions_gzip(self):
        with tempfile.TemporaryDirectory(prefix="temporal-backtest-") as folder:
            for dates in (["2026-03-01"], ["2026-03-01", "2026-03-03"]):
                with self.subTest(dates=dates):
                    output = Path(folder) / "predictions.csv.gz"
                    hourly = pd.DataFrame({"zone_id": [1]})

                    def predictions(data, meta, split, *, test_end=None):
                        return pd.DataFrame({"H": [1], "split": [split], "rate_hat": [0.5],
                                             "prob_low": [0.6], "origin": [split]})

                    args = ["backtest", "--hourly-csv", "offline.csv.gz", "--split", *dates,
                            "--output", str(output)]
                    with patch.object(sys, "argv", args), patch.object(bt, "load_hourly_csv", return_value=hourly), \
                            patch.object(bt, "load_hourly_api") as api, \
                            patch.object(bt, "backtest", side_effect=predictions) as backtest, \
                            patch.object(bt, "report") as report:
                        bt.main()
                    api.assert_not_called()
                    self.assertEqual(backtest.call_count, len(dates))
                    self.assertTrue(backtest.call_args_list[0].args[1].empty)
                    if len(dates) > 1:
                        self.assertEqual(backtest.call_args_list[0].kwargs["test_end"], pd.Timestamp(dates[1], tz="UTC"))
                    self.assertIsNone(backtest.call_args_list[-1].kwargs["test_end"])
                    saved = pd.read_csv(output)
                    self.assertEqual(len(saved), len(dates))
                    self.assertIn("prob_low", saved)
                    self.assertEqual(output.read_bytes()[:2], b"\x1f\x8b")
                    pd.testing.assert_frame_equal(report.call_args.args[0].reset_index(drop=True),
                                                  pd.concat([predictions(hourly, None, pd.Timestamp(d, tz="UTC"))
                                                             for d in dates], ignore_index=True))

    def test_unordered_or_duplicate_splits_fail_before_loading(self):
        for dates in (["2026-03-03", "2026-03-01"], ["2026-03-01", "2026-03-01"]):
            with self.subTest(dates=dates):
                with patch.object(sys, "argv", ["backtest", "--split", *dates]), \
                        patch.object(bt, "load_hourly_api") as api, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        bt.main()
                self.assertEqual(error.exception.code, 2)
                api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
