"""
Honest holdout backtest for v4, by horizon, against naive baselines.

Fits before --split minus 14 days, selects on the following 14 days and tests after
the split. Forecast origins are purged at both boundaries. Profiles and zone
metadata use only the fit period; the selected models are never refitted.

Usage:
    python -m parktrack_ml.backtest --hourly-csv ../Phisics/SmartParking/analysis/data/hourly_clean.csv.gz --split 2026-08-15
    python -m parktrack_ml.backtest --split 2026-09-10          # loads data from the ParkTrack API
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import HORIZONS, MIN_ZONE_CONFIDENCE, TRAIN_DAYS_BACK
from .features_h import (
    FEATURES_H, add_profile_features, build_horizon_dataset, label_rate,
    normalise_hourly, utc_timestamp,
)
from .model_h import HorizonModels

log = logging.getLogger("backtest")


def load_hourly_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["zone_id", "hour", "occupancy_rate", "capacity"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing hourly columns: {sorted(missing)}")
    if "confidence" not in df:
        df["confidence"] = df.get("cv_confidence", np.nan)
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")
    df["confidence"] = df.confidence.where(df.confidence.between(0, 1))
    if "free_fraction" in df:
        df["free_fraction"] = pd.to_numeric(df.free_fraction, errors="coerce")
        df["free_fraction"] = df.free_fraction.where(df.free_fraction.between(0, 1))
    df = normalise_hourly(df)
    columns = required + ["n_obs", "confidence"]
    columns += [c for c in ("free_fraction", "zone_type_standard") if c in df]
    return df[columns]


def load_hourly_api(days: int) -> pd.DataFrame:
    from .data_loader import load_observations, load_zone_meta, aggregate_hourly
    ids = load_zone_meta()["zone_id"].tolist()
    to_dt = datetime.now(tz=timezone.utc)
    return aggregate_hourly(load_observations(ids, to_dt - timedelta(days=days), to_dt))


def unreliable_zones(hourly: pd.DataFrame) -> list[int]:
    if hourly.empty or "confidence" not in hourly:
        return []
    values = pd.to_numeric(hourly.confidence, errors="coerce")
    conf = values.where(values.between(0, 1)).groupby(hourly.zone_id).mean().dropna()
    return sorted(int(z) for z in conf[conf < MIN_ZONE_CONFIDENCE].index)


def zone_meta_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame({"zone_id": pd.Series(dtype="int64"),
                             "capacity": pd.Series(dtype="float64"),
                             "zone_type_standard": pd.Series(dtype="int64")})
    data = normalise_hourly(hourly)
    latest = data.sort_values("hour").drop_duplicates("zone_id", keep="last").copy()
    if "zone_type_standard" not in latest:
        latest["zone_type_standard"] = 1
    return latest[["zone_id", "capacity", "zone_type_standard"]].sort_values("zone_id").reset_index(drop=True)


def _rounded_class(rate, capacity) -> np.ndarray:
    rate, capacity = np.asarray(rate, dtype=float), np.asarray(capacity, dtype=float)
    return label_rate(np.rint(np.clip(rate, 0, 1) * capacity) / capacity)


def backtest(hourly: pd.DataFrame, zone_meta_df: pd.DataFrame, split: pd.Timestamp,
             *, test_end=None) -> pd.DataFrame:
    """Return holdout prediction rows. Supplied global metadata is intentionally ignored."""
    hourly = normalise_hourly(hourly)
    split = utc_timestamp(split)
    val_start = split - pd.Timedelta(days=14)
    end = utc_timestamp(test_end) if test_end is not None else None
    if end is not None:
        if end <= split:
            raise ValueError("test_end must be after split")
        hourly = hourly[hourly.hour < end]
    fit_source = hourly[hourly.hour < val_start]
    bad = unreliable_zones(fit_source)
    log.info("Unreliable zones (mean detection confidence < %.2f), not trained on: %s", MIN_ZONE_CONFIDENCE, bad)
    hourly = hourly[~hourly["zone_id"].isin(bad)]
    fit_source = fit_source[~fit_source.zone_id.isin(bad)]
    if fit_source.empty:
        raise RuntimeError("No fit observations before validation start")
    fit_meta = zone_meta_from_hourly(fit_source)
    rows = []
    for H in HORIZONS:
        d = build_horizon_dataset(hourly, fit_meta, H)
        eligible = ((d.hour < val_start)
                    | ((d.hour < split) & (d.origin >= val_start))
                    | ((d.hour >= split) & (d.origin >= split)))
        d = d.loc[eligible].reset_index(drop=True)
        fit_mask = (d.hour < val_start).to_numpy()
        d = add_profile_features(d, fit_mask, source=fit_source)
        fit = d.loc[fit_mask]
        val = d[(d.hour >= val_start) & (d.hour < split)]
        test = d[d.hour >= split]
        if fit.empty or val.empty or test.empty:
            raise RuntimeError(f"Insufficient purged rows for H={H}: fit={len(fit)} val={len(val)} test={len(test)}")
        m = HorizonModels()
        m.fit_one(H, fit, val)
        probs, rate = m.predict(H, test[FEATURES_H].values)
        pred_reg = _rounded_class(rate, test.capacity)
        t = test[["zone_id", "hour", "origin", "exact_gap", "label", "occupancy_rate",
                  "last", "prof", "lag_week", "capacity"]].copy()
        t["H"], t["split"] = H, split
        t["pred"], t["pred_reg"], t["rate_hat"] = probs.argmax(1), pred_reg, rate
        t["conf"] = probs[np.arange(len(test)), pred_reg]
        t["cls_conf"] = probs.max(1)
        t[["prob_low", "prob_medium", "prob_high"]] = probs
        t["free_p"] = m.predict_free(H, test[FEATURES_H].values)
        t["free_target"] = (test.occupancy_rate < (test.capacity - 0.5) / test.capacity).astype(int)
        t["free_target_kind"] = "hourly_mean_below_rounding_threshold"
        t["free_calibration"] = "not_calibrated"
        t["temperature"] = m.temperature[H]
        if "free_fraction" in test:
            t["free_fraction"] = test.free_fraction
        rows.append(t)
        log.info("H=%d  fit=%d val=%d test=%d  T=%.2f", H, len(fit), len(val), len(test), m.temperature[H])
    return pd.concat(rows, ignore_index=True)


def _calibration(confidence: pd.Series, correct: pd.Series) -> tuple[pd.DataFrame, float]:
    bins = pd.cut(confidence, [0, .5, .6, .7, .8, .9, 1], include_lowest=True)
    data = pd.DataFrame({"confidence": confidence, "correct": correct})
    table = data.groupby(bins, observed=True).agg(n=("correct", "size"),
                                                 accuracy=("correct", "mean"),
                                                 confidence=("confidence", "mean"))
    ece = ((table.accuracy - table.confidence).abs() * table.n).sum() / len(data)
    return table, float(ece)


def report(r: pd.DataFrame) -> pd.DataFrame:
    """Return metrics without rounding; free Brier uses an hourly mean proxy target."""
    if r.empty:
        return pd.DataFrame(columns=["H", "n_all", "n_common"])
    out, calibrations = [], []
    keys = ["split", "H"] if "split" in r else ["H"]
    baselines = {"persist": "last", "profile": "prof", "week_ago": "lag_week"}
    for _, g in r.groupby(keys, sort=True):
        common = g.dropna(subset=list(baselines.values()))
        metrics = {key: g[key].iloc[0] for key in keys}
        metrics.update(n_all=len(g), n_common=len(common), coverage_common=len(common) / len(g),
                       acc_classifier=float((g.pred == g.label).mean()),
                       acc_reg=float((g.pred_reg == g.label).mean()),
                       MAE_model=float((g.rate_hat - g.occupancy_rate).abs().mean()),
                       acc_classifier_common=float((common.pred == common.label).mean()),
                       acc_reg_common=float((common.pred_reg == common.label).mean()),
                       MAE_model_common=float((common.rate_hat - common.occupancy_rate).abs().mean()),
                       always_low=float((g.label == 0).mean()),
                       predicted_p05=float(g.rate_hat.quantile(.05)),
                       predicted_p95=float(g.rate_hat.quantile(.95)),
                       actual_p05=float(g.occupancy_rate.quantile(.05)),
                       actual_p95=float(g.occupancy_rate.quantile(.95)))
        for name, column in baselines.items():
            available = g[g[column].notna()]
            metrics[f"n_{name}"] = len(available)
            metrics[f"coverage_{name}"] = len(available) / len(g)
            for suffix, subset in (("", available), ("_common", common)):
                pred = _rounded_class(subset[column], subset.capacity)
                metrics[f"acc_{name}{suffix}"] = float((pred == subset.label).mean()) if len(subset) else np.nan
                metrics[f"MAE_{name}{suffix}"] = float((subset[column] - subset.occupancy_rate).abs().mean())
        probs = g[["prob_low", "prob_medium", "prob_high"]].to_numpy()
        labels = g.label.to_numpy(dtype=int)
        metrics["logloss_classifier"] = float(-np.log(np.clip(probs[np.arange(len(g)), labels], 1e-9, 1)).mean())
        metrics["brier_classifier"] = float(((probs - np.eye(3)[labels]) ** 2).sum(axis=1).mean())
        for name, pred, confidence in (("classifier", g.pred, g.cls_conf), ("reg", g.pred_reg, g.conf)):
            table, metrics[f"calibration_ece_{name}"] = _calibration(confidence, pred == g.label)
            calibrations.append((dict((key, metrics[key]) for key in keys), name, table))
            for code, label in enumerate(("low", "medium", "high")):
                mask = g.label == code
                metrics[f"n_{label}"] = int(mask.sum())
                metrics[f"recall_{name}_{label}"] = float((pred[mask] == code).mean())
        free = g[g.free_p.notna()]
        metrics["n_free"] = len(free)
        metrics["brier_free_proxy"] = float(((free.free_p - free.free_target) ** 2).mean())
        metrics["free_target_kind"] = "hourly_mean_below_rounding_threshold"
        metrics["free_calibration"] = "not_calibrated"
        out.append(metrics)
    df = pd.DataFrame(out)
    print("\nHoldout metrics (all stream, baseline coverage and common rows):")
    print(df.round(3).to_string(index=False))
    print("\nPredicted/actual p05 and p95 describe rate distributions, not prediction intervals.")
    print("Free Brier target: hourly_mean_below_rounding_threshold (proxy, not instantaneous availability).")
    for group, name, table in calibrations:
        print(f"\nCalibration {group}, {name}:")
        print(table.round(3).to_string())
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-csv", default=None)
    ap.add_argument("--split", required=True, nargs="+", help="One or more strictly chronological UTC dates")
    ap.add_argument("--days", type=int, default=TRAIN_DAYS_BACK)
    ap.add_argument("--output", default=None, help="Prediction dataset CSV or CSV.gz")
    a = ap.parse_args()
    splits = [utc_timestamp(value) for value in a.split]
    if any(right <= left for left, right in zip(splits, splits[1:])):
        ap.error("--split dates must be strictly chronological")
    hourly = load_hourly_csv(a.hourly_csv) if a.hourly_csv else load_hourly_api(a.days)
    empty_meta = zone_meta_from_hourly(hourly.iloc[:0])
    rows = [backtest(hourly, empty_meta, split,
                     test_end=splits[i + 1] if i + 1 < len(splits) else None)
            for i, split in enumerate(splits)]
    predictions = pd.concat(rows, ignore_index=True)
    if a.output:
        predictions.to_csv(a.output, index=False, compression="infer")
    report(predictions)


if __name__ == "__main__":
    main()
