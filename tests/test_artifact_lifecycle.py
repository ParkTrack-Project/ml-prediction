import importlib.util
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

import numpy as np
import pandas as pd

from parktrack_ml import interfaces
from parktrack_ml import model_h
from parktrack_ml.model_h import HorizonModels, apply_temperature


class Booster:
    def __init__(self, values, *, best_iteration=7, before_save=None):
        self.values = values
        self.best_iteration = best_iteration
        self.before_save = before_save

    def predict(self, X, **kwargs):
        return np.tile(self.values, (len(X), 1)) if isinstance(self.values, list) else np.full(len(X), self.values)

    def save_model(self, path):
        if self.before_save is not None:
            self.before_save(path)
        Path(path).write_text(json.dumps(self.values), encoding="utf-8")


def load_booster(*, model_file):
    return Booster(json.loads(Path(model_file).read_text(encoding="utf-8")))


def models(value=0.8):
    result = HorizonModels()
    for H in (1, 6):
        result.cls[H] = Booster([0.6, 0.3, 0.1])
        result.reg[H] = Booster(value)
        result.temperature[H] = 1.7
    result.free[1] = Booster(0.4)
    return result


def metadata(capacity=10):
    return {"version": "test", "zone_meta": {"3": {"capacity": capacity}},
            "profiles": {"3": {"mean": capacity / 100}}}


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parktrack-artifacts-")
        self.addCleanup(self.temp.cleanup)
        self.pointer = Path(self.temp.name) / "models" / "model_v4.json"
        self.lgb = SimpleNamespace(Booster=Mock(side_effect=load_booster))
        self.modules = patch.dict(sys.modules, {"lightgbm": self.lgb})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def read_meta(self):
        return json.loads(self.pointer.read_text(encoding="utf-8"))

    def test_versioned_roundtrip_keeps_previous_boosters_and_classifier(self):
        first = models()
        first.save(self.pointer, metadata())
        old_meta = self.read_meta()
        old_dir = self.pointer.parent / old_meta["artifact_dir"]
        old_files = {p.name: p.read_bytes() for p in old_dir.iterdir()}

        models(0.2).save(self.pointer, metadata(20))
        new_meta = self.read_meta()
        self.assertNotEqual(old_meta["artifact_dir"], new_meta["artifact_dir"])
        self.assertEqual(Path(new_meta["artifact_dir"]).parent, Path("runs"))
        UUID(Path(new_meta["artifact_dir"]).name)
        self.assertEqual(old_files, {p.name: p.read_bytes() for p in old_dir.iterdir()})
        self.assertFalse(list(self.pointer.parent.glob("*.lgb")))
        self.assertEqual(new_meta["version"], "test")
        self.assertEqual(new_meta["free_target"], "hourly_mean_below_rounding_threshold")
        self.assertEqual(new_meta["free_calibration"], "not_calibrated")

        loaded, saved_meta = HorizonModels.load(self.pointer)
        X = np.zeros((2, len(model_h.FEATURES_H)))
        probs, rate = loaded.predict(1, X)
        np.testing.assert_allclose(probs, apply_temperature(first.cls[1].predict(X), 1.7))
        np.testing.assert_allclose(rate, 0.2)
        np.testing.assert_allclose(loaded.predict_free(1, X), 0.4)
        self.assertTrue(np.isnan(loaded.predict_free(6, X)).all())
        self.assertEqual(saved_meta, new_meta)
        self.assertEqual(loaded.temperature, first.temperature)

    def test_staging_is_invisible_until_all_boosters_are_saved(self):
        models().save(self.pointer, metadata())
        previous = self.pointer.read_bytes()

        def inspect_stage(path):
            self.assertIn(".staging-", Path(path).parent.name)
            self.assertEqual(self.pointer.read_bytes(), previous)
            loaded, _ = HorizonModels.load(self.pointer)
            self.assertEqual(loaded.reg[1].values, 0.8)

        next_models = models(0.2)
        next_models.free[1].before_save = inspect_stage
        next_models.save(self.pointer, metadata(20))
        loaded, _ = HorizonModels.load(self.pointer)
        self.assertEqual(loaded.reg[1].values, 0.2)

    def test_failed_booster_write_preserves_pointer_and_old_artifact(self):
        models().save(self.pointer, metadata())
        previous = self.pointer.read_bytes()
        for head in ("cls", "reg", "free"):
            with self.subTest(head=head):
                candidate = models(0.2)
                getattr(candidate, head)[1].before_save = Mock(side_effect=OSError("write failed"))
                with self.assertRaisesRegex(OSError, "write failed"):
                    candidate.save(self.pointer, metadata(20))
                self.assertEqual(self.pointer.read_bytes(), previous)
                self.assertFalse(list((self.pointer.parent / "runs").glob(".staging-*")))
                loaded, _ = HorizonModels.load(self.pointer)
                self.assertEqual(loaded.reg[1].values, 0.8)

    def test_failed_directory_rename_or_pointer_replace_keeps_previous_run(self):
        models().save(self.pointer, metadata())
        previous = self.pointer.read_bytes()
        for operation in ("rename", "replace"):
            with self.subTest(operation=operation):
                with patch.object(model_h.os, operation, side_effect=OSError("publish failed")):
                    with self.assertRaisesRegex(OSError, "publish failed"):
                        models(0.2).save(self.pointer, metadata(20))
                self.assertEqual(self.pointer.read_bytes(), previous)
                self.assertFalse(list(self.pointer.parent.glob(".model-v4-*")))
                self.assertFalse(list((self.pointer.parent / "runs").glob(".staging-*")))
                loaded, _ = HorizonModels.load(self.pointer)
                self.assertEqual(loaded.reg[1].values, 0.8)

    def test_failed_first_save_does_not_create_pointer(self):
        candidate = models()
        candidate.reg[1].before_save = Mock(side_effect=OSError("write failed"))
        with self.assertRaises(OSError):
            candidate.save(self.pointer, metadata())
        self.assertFalse(self.pointer.exists())
        self.assertEqual(list((self.pointer.parent / "runs").iterdir()), [])

    def test_legacy_flat_artifact_loads_without_free_head(self):
        self.pointer.parent.mkdir()
        legacy = models()
        for H in legacy.cls:
            legacy.cls[H].save_model(self.pointer.parent / f"v4_h{H}_cls.lgb")
            legacy.reg[H].save_model(self.pointer.parent / f"v4_h{H}_reg.lgb")
        self.pointer.write_text(json.dumps({**metadata(), "features": model_h.FEATURES_H,
                                           "horizons": [1, 6], "temperature": {"1": 1.7}}))
        loaded, _ = HorizonModels.load(self.pointer)
        self.assertEqual(loaded.free, {})
        self.assertTrue(np.isnan(loaded.predict_free(1, np.zeros((3, 2)))).all())

    def test_declared_missing_free_booster_fails_load(self):
        models().save(self.pointer, metadata())
        original_loader = self.lgb.Booster.side_effect

        def fail_free(*, model_file):
            if model_file.endswith("_free.lgb"):
                raise OSError("corrupt free booster")
            return original_loader(model_file=model_file)

        self.lgb.Booster.side_effect = fail_free
        with self.assertRaisesRegex(OSError, "corrupt free booster"):
            HorizonModels.load(self.pointer)

    def test_corrupt_sha_rejects_each_head_before_loading_any_booster(self):
        for head, changed_values in (("cls", [0.1, 0.2, 0.7]), ("reg", 0.1), ("free", 0.9)):
            with self.subTest(head=head):
                models().save(self.pointer, metadata())
                self.lgb.Booster.reset_mock()
                HorizonModels.load(self.pointer)
                self.assertEqual(self.lgb.Booster.call_count, 5)
                self.lgb.Booster.reset_mock()
                artifact = self.pointer.parent / self.read_meta()["artifact_dir"] / f"v4_h1_{head}.lgb"
                artifact.write_text(json.dumps(changed_values), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, f"Corrupt model artifact: v4_h1_{head}"):
                    HorizonModels.load(self.pointer)
                self.lgb.Booster.assert_not_called()

    def test_archived_manifest_exactly_preserves_published_metadata_after_next_save(self):
        supplied = {**metadata(), "trained_at": "2026-11-18T00:00:00Z",
                    "data_until": "2026-11-17T23:00:00Z", "fit_until": "2026-11-01T00:00:00Z",
                    "val": {"1": {"acc": 0.75, "n_val": 200}},
                    "training_config": {"parameters": {"num_threads": 4}},
                    "source_sha256": {"train.py": "source-fingerprint"}}
        models().save(self.pointer, supplied)
        published = self.read_meta()
        directory = self.pointer.parent / published["artifact_dir"]
        manifest = directory / "manifest.json"
        archived_bytes = manifest.read_bytes()
        self.assertEqual(json.loads(archived_bytes), published)
        for key, value in supplied.items():
            self.assertEqual(published[key], value)
        expected_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in directory.glob("*.lgb")}
        self.assertEqual(len(expected_hashes), 5)
        self.assertEqual(published["artifact_sha256"], expected_hashes)

        models(0.2).save(self.pointer, metadata(20))
        current = self.read_meta()
        self.assertNotEqual(current["artifact_dir"], published["artifact_dir"])
        self.assertEqual(manifest.read_bytes(), archived_bytes)
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), published)
        current_manifest = self.pointer.parent / current["artifact_dir"] / "manifest.json"
        self.assertEqual(json.loads(current_manifest.read_text(encoding="utf-8")), current)

    def test_predict_free_clips_probability(self):
        candidate = models()
        for value, expected in ((-0.1, 0.0), (1.1, 1.0)):
            candidate.free[1] = Booster(value)
            np.testing.assert_array_equal(candidate.predict_free(1, np.zeros((2, 3))), [expected] * 2)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.state = patch.object(interfaces, "_snapshot", None)
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_parallel_lazy_load_publishes_one_complete_snapshot(self):
        candidate = models()
        barrier = Barrier(8)

        def read():
            barrier.wait(timeout=5)
            return interfaces._load_once()

        with patch.object(HorizonModels, "load", return_value=(candidate, metadata())) as load:
            with ThreadPoolExecutor(max_workers=8) as pool:
                snapshots = list(pool.map(lambda _: read(), range(8)))
        load.assert_called_once()
        self.assertTrue(all(snapshot is snapshots[0] for snapshot in snapshots))
        self.assertIs(snapshots[0][0], candidate)
        self.assertEqual(snapshots[0][1]["zone_meta"][3]["capacity"], 10)

    def test_failed_lazy_load_can_be_retried(self):
        with patch.object(HorizonModels, "load", side_effect=OSError("load failed")):
            with self.assertRaises(OSError):
                interfaces._load_once()
        self.assertIsNone(interfaces._snapshot)
        with patch.object(HorizonModels, "load", return_value=(models(), metadata())):
            self.assertIsNotNone(interfaces._load_once())

    def test_failed_reload_retains_entire_old_snapshot(self):
        with patch.object(HorizonModels, "load", return_value=(models(), metadata())):
            previous = interfaces._load_once()
        for result in (OSError("booster load failed"), (models(0.2), {"zone_meta": {"bad": {}}})):
            with self.subTest(result=type(result).__name__):
                loader = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
                with patch.object(HorizonModels, "load", loader):
                    with self.assertRaises((OSError, ValueError)):
                        interfaces.reload()
                self.assertIs(interfaces._load_once(), previous)

    def assert_future_model_rejected(self, field):
        cutoff = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
        target = cutoff + timedelta(hours=3)
        meta = {**metadata(), "data_until": cutoff.isoformat(), "trained_at": cutoff.isoformat(),
                field: (cutoff + timedelta(seconds=1)).isoformat()}
        for name, targets in (("predict", target), ("predict_many", [target])):
            with self.subTest(public_method=name, field=field):
                with patch.object(interfaces, "_snapshot", None), \
                        patch.object(HorizonModels, "load", return_value=(models(), meta)) as load, \
                        patch.object(interfaces, "predict_for_zone") as scalar, \
                        patch.object(interfaces, "predict_many_for_zone") as batch:
                    with self.assertRaisesRegex(ValueError, f"Model {field} is later"):
                        getattr(interfaces, name)(3, targets, recent_hourly=object(), as_of=cutoff)
                    load.assert_called_once()
                    scalar.assert_not_called()
                    batch.assert_not_called()

    def test_public_as_of_rejects_data_until_later_than_cutoff(self):
        self.assert_future_model_rejected("data_until")

    def test_public_as_of_rejects_trained_at_later_than_cutoff(self):
        self.assert_future_model_rejected("trained_at")

    def test_public_as_of_accepts_equal_instants_with_different_offsets(self):
        cutoff = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
        offset_cutoff = cutoff.astimezone(timezone(timedelta(hours=3)))
        meta = {**metadata(), "data_until": cutoff.isoformat(), "trained_at": offset_cutoff.isoformat()}
        raw = {"class": "Medium", "class_code": 1, "confidence": 0.3,
               "all_probabilities": {"Low": 0.6, "Medium": 0.3, "High": 0.1},
               "predicted_occupied": 5, "probability_free_space": 0.4, "capacity": 10}
        with patch.object(HorizonModels, "load", return_value=(models(), meta)) as load, \
                patch.object(interfaces, "predict_for_zone", return_value=raw) as scalar, \
                patch.object(interfaces, "predict_many_for_zone", return_value=[raw]) as batch:
            target = cutoff + timedelta(hours=3)
            single = interfaces.predict(3, target, recent_hourly=object(), as_of=offset_cutoff)
            multiple = interfaces.predict_many(3, [target], recent_hourly=object(), as_of=offset_cutoff)
        load.assert_called_once()
        self.assertEqual(multiple, [single])
        self.assertIs(scalar.call_args.kwargs["as_of"], offset_cutoff)
        self.assertIs(batch.call_args.kwargs["as_of"], offset_cutoff)

    def test_predict_in_flight_uses_one_snapshot_and_forwards_as_of(self):
        old, new = models(), models(0.2)
        with patch.object(HorizonModels, "load", return_value=(old, metadata())):
            interfaces.reload()
        entered, release = Event(), Event()
        target = datetime(2026, 11, 19, tzinfo=timezone.utc)
        as_of = datetime(2026, 11, 18, tzinfo=timezone.utc)
        recent = object()

        def infer(zone_id, predicted_for, model, zone_meta, profiles, **kwargs):
            self.assertIs(kwargs["recent_hourly"], recent)
            self.assertEqual(kwargs["as_of"], as_of)
            self.assertEqual((zone_id, predicted_for), (3, target))
            capacity = 10 if model is old else 20
            self.assertEqual(zone_meta["capacity"], capacity)
            self.assertEqual(profiles[3]["mean"], capacity / 100)
            if model is old:
                entered.set()
                self.assertTrue(release.wait(timeout=5))
            return {"class": "High", "class_code": 2, "confidence": 0.1,
                    "all_probabilities": {"Low": 0.6, "Medium": 0.3, "High": 0.1},
                    "predicted_occupied": 8, "probability_free_space": 0.4, "capacity": capacity}

        with patch.object(interfaces, "predict_for_zone", side_effect=infer):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(interfaces.predict, 3, target, recent, as_of)
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    with patch.object(HorizonModels, "load", return_value=(new, metadata(20))):
                        interfaces.reload()
                finally:
                    release.set()
                self.assertEqual(pending.result(timeout=5).capacity, 10)
            output = interfaces.predict(3, target, recent, as_of)
        self.assertEqual(output.capacity, 20)
        self.assertEqual(output.confidence, output.prob_high)
        self.assertLess(output.confidence, output.prob_low)


class TrainingContractTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({name: np.zeros(4) for name in model_h.FEATURES_H})
        frame["capacity"] = [10, 10, 10, 1]
        frame["label"] = [0, 1, 2, 2]
        frame["occupancy_rate"] = [0.94, 0.95, 0.96, 0.5]
        frame["n_obs"] = [0, 4, 100, 1]
        return frame

    def lgb(self):
        def train(params, dataset, rounds, **kwargs):
            objective = params["objective"]
            return Booster([0.6, 0.3, 0.1] if objective == "multiclass" else 0.4,
                           best_iteration={"multiclass": 7, "regression": 9, "binary": 5}[objective])
        return SimpleNamespace(Dataset=Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(**kwargs)),
                               train=Mock(side_effect=train), early_stopping=Mock(return_value=object()))

    def test_weights_categorical_and_head_specific_params_with_validation(self):
        features = [*model_h.FEATURES_H, "zone_id"] if "zone_id" not in model_h.FEATURES_H else model_h.FEATURES_H
        params = {**model_h.LGBM_V4_PARAMS, "num_threads": 4}
        lgb = self.lgb()
        with patch.object(model_h, "FEATURES_H", features), patch.object(model_h, "LGBM_V4_PARAMS", params):
            frame = self.frame()
            candidate = HorizonModels()
            with patch.dict(sys.modules, {"lightgbm": lgb}):
                self.assertEqual(candidate.fit_one(1, frame, frame), (7, 9))
        calls = lgb.train.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].args[0], {**params, "objective": "multiclass", "num_class": 3})
        self.assertEqual(calls[1].args[0], {**params, "objective": "regression", "num_leaves": 63,
                                           "min_child_samples": 20, "max_depth": 8})
        self.assertEqual(calls[2].args[0], {**params, "objective": "binary", "metric": "binary_logloss"})
        np.testing.assert_array_equal(calls[2].args[1].label, [1, 0, 0, 0])
        for call in lgb.Dataset.call_args_list:
            np.testing.assert_array_equal(call.kwargs["weight"], [1, 2, 8, 1])
            self.assertEqual(call.kwargs["categorical_feature"], ["zone_id"])
        for call in calls:
            self.assertEqual(call.args[2], model_h.MAX_ROUNDS)
            self.assertEqual(len(call.kwargs["valid_sets"]), 1)
            self.assertEqual(len(call.kwargs["callbacks"]), 1)
        self.assertEqual(lgb.early_stopping.call_count, 3)
        self.assertIn(1, candidate.temperature)
        self.assertIn(1, candidate.free)

    def test_fixed_rounds_free_uses_classifier_rounds_and_weights_are_optional(self):
        lgb = self.lgb()
        candidate = HorizonModels()
        frame = self.frame().drop(columns="n_obs")
        with patch.dict(sys.modules, {"lightgbm": lgb}):
            self.assertEqual(candidate.fit_one(1, frame, rounds=(11, 13)), (11, 13))
        self.assertEqual([call.args[2] for call in lgb.train.call_args_list], [11, 13, 11])
        self.assertTrue(all(call.kwargs["weight"] is None for call in lgb.Dataset.call_args_list))
        expected_cat = ["zone_id"] if "zone_id" in model_h.FEATURES_H else []
        self.assertTrue(all(call.kwargs["categorical_feature"] == expected_cat for call in lgb.Dataset.call_args_list))
        lgb.early_stopping.assert_not_called()

    def test_free_head_can_be_disabled(self):
        lgb = self.lgb()
        candidate = HorizonModels(train_free=False)
        with patch.dict(sys.modules, {"lightgbm": lgb}):
            candidate.fit_one(1, self.frame(), rounds=(11, 13))
        self.assertEqual(lgb.train.call_count, 2)
        self.assertEqual(candidate.free, {})


class ServiceStartupTests(unittest.TestCase):
    def load_service(self):
        scheduler = Mock()
        modules = {}
        for name in ("apscheduler", "apscheduler.schedulers", "apscheduler.triggers",
                     "apscheduler.schedulers.blocking", "apscheduler.triggers.cron", "apscheduler.events",
                     "parktrack_ml.weather", "parktrack_ml.forecaster", "parktrack_ml.train"):
            modules[name] = ModuleType(name)
        modules["apscheduler.schedulers.blocking"].BlockingScheduler = Mock(return_value=scheduler)
        modules["apscheduler.triggers.cron"].CronTrigger = Mock()
        modules["apscheduler.events"].EVENT_JOB_ERROR = 1
        modules["parktrack_ml.weather"].backfill = Mock(return_value=4)
        modules["parktrack_ml.weather"].fetch_latest = Mock()
        modules["parktrack_ml.forecaster"].run = Mock()
        modules["parktrack_ml.train"].train = Mock()
        path = Path(model_h.__file__).with_name("service.py")
        spec = importlib.util.spec_from_file_location("parktrack_ml._service_lifecycle_test", path)
        service = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch("logging.basicConfig"):
            spec.loader.exec_module(service)
        return service, scheduler

    def test_startup_never_backfills_and_main_schedules_background_job(self):
        service, scheduler = self.load_service()
        with patch.object(service.os.path, "exists", return_value=True), patch.object(service, "reload_model"):
            service.main()
        service.backfill_weather.assert_not_called()
        service.train.assert_not_called()
        service.generate_forecasts.assert_called_once()
        jobs = [call for call in scheduler.add_job.call_args_list if call.kwargs.get("id") == "weather_backfill"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].args, (service._backfill_weather, "date"))
        scheduler.start.assert_called_once()
        service._backfill_weather()
        service.backfill_weather.assert_called_once_with(days=service.TRAIN_DAYS_BACK)

    def test_no_model_and_failed_initial_training_still_schedules_backfill(self):
        service, scheduler = self.load_service()
        service.train.side_effect = RuntimeError("no data")
        with patch.object(service.os.path, "exists", return_value=False), patch.object(service, "reload_model"):
            service.main()
        service.backfill_weather.assert_not_called()
        service.generate_forecasts.assert_not_called()
        self.assertIn("weather_backfill", [call.kwargs.get("id") for call in scheduler.add_job.call_args_list])
        scheduler.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
