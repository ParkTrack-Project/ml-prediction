"""
Visualize model performance — generates PNG charts from backtest results.

Usage:
    python -m parktrack_ml.visualize
    python -m parktrack_ml.visualize --days 60 --out ./charts
    python -m parktrack_ml.visualize --days 30 --zone-id 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("visualize")

CLASS_NAMES = ["Low", "Medium", "High"]
COLORS = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
PALETTE = [COLORS[c] for c in CLASS_NAMES]


def _ensure_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.error("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, out_dir: str, plt) -> str:
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.zeros((3, 3), dtype=int)
    for a in range(3):
        for p in range(3):
            cm[a, p] = int(((df["actual_class"] == a) & (df["predicted_class"] == p)).sum())

    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    total = cm.sum()
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            color = "white" if v > total * 0.15 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_by_hour(df: pd.DataFrame, out_dir: str, plt) -> str:
    df = df.copy()
    df["hod"] = pd.to_datetime(df["hour"], utc=True).dt.hour
    hod = df.groupby("hod")["correct"].mean().reindex(range(24), fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(hod.index, hod.values * 100, color="#2196F3", alpha=0.8, width=0.7)
    ax.axhline(df["correct"].mean() * 100, color="red", linestyle="--",
               linewidth=1.5, label=f"Overall {df['correct'].mean():.1%}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Hour of Day")
    ax.set_xticks(range(24))
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "accuracy_by_hour.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_zone_metrics(df: pd.DataFrame, out_dir: str, plt) -> str:
    zones = sorted(df["zone_id"].unique())
    accs = [df[df["zone_id"] == z]["correct"].mean() * 100 for z in zones]
    maes = [df[df["zone_id"] == z]["abs_error"].mean() for z in zones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.barh([f"Zone {z}" for z in zones], accs, color="#4CAF50", alpha=0.8)
    ax1.axvline(df["correct"].mean() * 100, color="red", linestyle="--", linewidth=1.5,
                label=f"Overall {df['correct'].mean():.1%}")
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Accuracy per Zone")
    ax1.set_xlim(0, 105)
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh([f"Zone {z}" for z in zones], maes, color="#FF9800", alpha=0.8)
    ax2.axvline(df["abs_error"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Overall MAE {df['abs_error'].mean():.2f}")
    ax2.set_xlabel("MAE (spots)")
    ax2.set_title("MAE per Zone")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "per_zone_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, out_dir: str, plt) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    actual_counts = [int((df["actual_class"] == i).sum()) for i in range(3)]
    pred_counts   = [int((df["predicted_class"] == i).sum()) for i in range(3)]

    x = np.arange(3)
    w = 0.35
    ax1.bar(x - w/2, actual_counts, w, label="Actual",    color=PALETTE, alpha=0.9)
    ax1.bar(x + w/2, pred_counts,   w, label="Predicted", color=PALETTE, alpha=0.5,
            edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution: Actual vs Predicted")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Per-class F1
    f1s = []
    for cls in range(3):
        y_true = (df["actual_class"] == cls)
        y_pred = (df["predicted_class"] == cls)
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)

    bars = ax2.bar(CLASS_NAMES, [v * 100 for v in f1s], color=PALETTE, alpha=0.85)
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_title("F1 Score per Class")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    path = os.path.join(out_dir, "class_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(out_dir: str, plt) -> str | None:
    from .config import MODEL_FILE
    from .model import LGBMWrapper

    try:
        model = LGBMWrapper.load(MODEL_FILE)
        importance = model.feature_importance()  # dict sorted by gain desc
    except Exception as exc:
        log.warning("Could not load model for feature importance: %s", exc)
        return None

    names = list(importance.keys())[:15]
    vals  = list(importance.values())[:15]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3"] * len(names)
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Gain (feature importance)")
    ax.set_title("Top-15 Features by Importance (Gain)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_predicted_vs_actual(df: pd.DataFrame, out_dir: str, plt) -> str:
    # Sample up to 2000 points for readability
    sample = df.sample(min(2000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(6, 6))
    c_colors = [PALETTE[int(c)] for c in sample["actual_class"]]
    ax.scatter(sample["actual_occupied"], sample["predicted_occupied"],
               c=c_colors, alpha=0.35, s=15)
    lim = max(sample["actual_occupied"].max(), sample["predicted_occupied"].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual occupied spots")
    ax.set_ylabel("Predicted occupied spots")
    ax.set_title("Predicted vs Actual (spots)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend()

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=PALETTE[i], label=CLASS_NAMES[i]) for i in range(3)]
    ax.legend(handles=legend_els + [ax.get_lines()[0]], loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML performance charts")
    parser.add_argument("--days",    type=int, default=30)
    parser.add_argument("--zone-id", type=int, default=None)
    parser.add_argument("--out",     type=str, default="./charts",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    plt = _ensure_mpl()
    os.makedirs(args.out, exist_ok=True)

    log.info("Running backtest (%d days)...", args.days)
    from .evaluate import evaluate
    zone_ids = [args.zone_id] if args.zone_id else None
    df = evaluate(days=args.days, zone_ids=zone_ids)

    if df.empty:
        log.error("No data — cannot generate charts.")
        sys.exit(1)

    generated = []

    log.info("Generating charts → %s", args.out)
    generated.append(plot_confusion_matrix(df, args.out, plt))
    generated.append(plot_accuracy_by_hour(df, args.out, plt))
    generated.append(plot_per_zone_metrics(df, args.out, plt))
    generated.append(plot_class_distribution(df, args.out, plt))
    generated.append(plot_predicted_vs_actual(df, args.out, plt))

    fi_path = plot_feature_importance(args.out, plt)
    if fi_path:
        generated.append(fi_path)

    print()
    print("Charts saved:")
    for p in generated:
        if p:
            print(f"  {p}")
    print()
    print(f"Overall accuracy : {df['correct'].mean():.1%}")
    print(f"MAE (spots)      : {df['abs_error'].mean():.2f}")
    print(f"Samples          : {len(df):,}")


if __name__ == "__main__":
    main()
