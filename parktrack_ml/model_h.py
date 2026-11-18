"""Per-horizon classifier, occupancy regressor and optional hourly mean availability head."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import LGBM_V4_PARAMS, MAX_ROUNDS
from .features_h import FEATURES_H


def _calibrate_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising log-loss of p^(1/T); T > 1 softens over-confident probabilities."""
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    best_t, best_ll = 1.0, np.inf
    for T in np.arange(0.6, 3.01, 0.05):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)).mean()
        if ll < best_ll:
            best_t, best_ll = float(T), ll
    return best_t


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-9, 1.0)) / T
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


class HorizonModels:
    def __init__(self, *, train_free: bool = True) -> None:
        self.cls: Dict[int, object] = {}
        self.reg: Dict[int, object] = {}
        self.free: Dict[int, object] = {}
        self.temperature: Dict[int, float] = {}
        self.train_free = train_free

    # -- training ----------------------------------------------------------

    def fit_one(
        self, H: int, train: pd.DataFrame, val: Optional[pd.DataFrame] = None,
        rounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Fit models for bucket H. With `val`, early stopping + temperature calibration.
        With `rounds` (cls, reg), trains a fixed number of rounds. Returns the rounds used."""
        import lightgbm as lgb

        X, ylab, yrate = train[FEATURES_H].values, train["label"].values, train["occupancy_rate"].values
        p_cls = {**LGBM_V4_PARAMS, "objective": "multiclass", "num_class": 3}
        p_reg = {**LGBM_V4_PARAMS, "objective": "regression", "num_leaves": 63,
                 "min_child_samples": 20, "max_depth": 8}
        p_free = {**LGBM_V4_PARAMS, "objective": "binary", "metric": "binary_logloss"}

        def dataset(frame, values, label, reference=None):
            weight = np.clip(np.sqrt(frame["n_obs"].values), 1, 8) if "n_obs" in frame else None
            return lgb.Dataset(values, label=label, weight=weight, reference=reference,
                               feature_name=FEATURES_H,
                               categorical_feature=["zone_id"] if "zone_id" in FEATURES_H else [])

        def free_target(frame):
            capacity = frame["capacity"].values
            return (frame["occupancy_rate"].values < (capacity - 0.5) / capacity).astype(int)

        ds_c = dataset(train, X, ylab)
        ds_r = dataset(train, X, yrate)
        ds_free = dataset(train, X, free_target(train)) if self.train_free else None

        if val is not None:
            Xv = val[FEATURES_H].values
            vc = dataset(val, Xv, val["label"].values, ds_c)
            vr = dataset(val, Xv, val["occupancy_rate"].values, ds_r)
            stop = [lgb.early_stopping(50, verbose=False)]
            self.cls[H] = lgb.train(p_cls, ds_c, MAX_ROUNDS, valid_sets=[vc], callbacks=stop)
            self.reg[H] = lgb.train(p_reg, ds_r, MAX_ROUNDS, valid_sets=[vr],
                                    callbacks=[lgb.early_stopping(50, verbose=False)])
            raw = self.cls[H].predict(Xv, num_iteration=self.cls[H].best_iteration)
            self.temperature[H] = _calibrate_temperature(raw, val["label"].values)
            if ds_free is not None:
                vf = dataset(val, Xv, free_target(val), ds_free)
                self.free[H] = lgb.train(p_free, ds_free, MAX_ROUNDS, valid_sets=[vf],
                                       callbacks=[lgb.early_stopping(50, verbose=False)])
            return self.cls[H].best_iteration, self.reg[H].best_iteration

        if rounds is None:
            raise ValueError("rounds is required when validation data is omitted")
        rc, rr = rounds
        self.cls[H] = lgb.train(p_cls, ds_c, rc)
        self.reg[H] = lgb.train(p_reg, ds_r, rr)
        if ds_free is not None:
            self.free[H] = lgb.train(p_free, ds_free, rc)
        self.temperature.setdefault(H, 1.0)
        return rc, rr

    # -- inference ---------------------------------------------------------

    def predict(self, H: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (calibrated class probabilities [n,3], occupancy rate [n])."""
        probs = apply_temperature(self.cls[H].predict(X), self.temperature.get(H, 1.0))
        rate = np.clip(self.reg[H].predict(X), 0.0, 1.0)
        return probs, rate

    def predict_free(self, H: int, X: np.ndarray) -> np.ndarray:
        """Uncalibrated probability for the hourly mean proxy, not instantaneous availability."""
        if H not in self.free:
            return np.full(len(X), np.nan)
        return np.clip(self.free[H].predict(X), 0.0, 1.0)

    # -- persistence -------------------------------------------------------

    def save(self, meta_path: str, meta: dict) -> None:
        meta_path = os.path.abspath(meta_path)
        base = os.path.dirname(meta_path)
        runs = os.path.join(base, "runs")
        os.makedirs(runs, exist_ok=True)
        artifact_dir = os.path.join(runs, uuid4().hex)
        stage = tempfile.mkdtemp(prefix=".staging-", dir=runs)
        tmp = None
        try:
            if self.cls.keys() != self.reg.keys() or not self.free.keys() <= self.cls.keys():
                raise ValueError("incomplete horizon models")
            for H in self.cls:
                self.cls[H].save_model(os.path.join(stage, f"v4_h{H}_cls.lgb"))
                self.reg[H].save_model(os.path.join(stage, f"v4_h{H}_reg.lgb"))
                if H in self.free:
                    self.free[H].save_model(os.path.join(stage, f"v4_h{H}_free.lgb"))
            meta = {**meta, "horizons": sorted(self.cls),
                    "temperature": {str(k): v for k, v in self.temperature.items()},
                    "features": FEATURES_H, "artifact_dir": os.path.relpath(artifact_dir, base),
                    "free_horizons": sorted(self.free),
                    "free_target": "hourly_mean_below_rounding_threshold",
                    "free_calibration": "not_calibrated"}
            meta["artifact_sha256"] = {
                name: hashlib.sha256(Path(stage, name).read_bytes()).hexdigest()
                for name in os.listdir(stage) if name.endswith(".lgb")
            }
            with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as manifest:
                json.dump(meta, manifest)
            os.rename(stage, artifact_dir)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=base,
                                             prefix=".model-v4-", suffix=".tmp", delete=False) as f:
                tmp = f.name
                json.dump(meta, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, meta_path)
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage)
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, meta_path: str) -> Tuple["HorizonModels", dict]:
        import lightgbm as lgb

        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("features") != FEATURES_H:
            raise RuntimeError("v4 feature set changed — retrain: python -m parktrack_ml.train")
        base = os.path.join(os.path.dirname(os.path.abspath(meta_path)), meta.get("artifact_dir", "."))
        for name, expected in meta.get("artifact_sha256", {}).items():
            with open(os.path.join(base, name), "rb") as artifact:
                if hashlib.sha256(artifact.read()).hexdigest() != expected:
                    raise RuntimeError(f"Corrupt model artifact: {name}")
        m = cls()
        for H in meta["horizons"]:
            m.cls[H] = lgb.Booster(model_file=os.path.join(base, f"v4_h{H}_cls.lgb"))
            m.reg[H] = lgb.Booster(model_file=os.path.join(base, f"v4_h{H}_reg.lgb"))
        for H in meta.get("free_horizons", []):
            m.free[H] = lgb.Booster(model_file=os.path.join(base, f"v4_h{H}_free.lgb"))
        m.temperature = {int(k): v for k, v in meta["temperature"].items()}
        return m, meta
