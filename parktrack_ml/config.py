import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ParkTrack API
# ---------------------------------------------------------------------------

API_URL   = os.environ.get("API_URL",   "https://api.parktrack.live/api/v1")
API_TOKEN = os.environ.get("API_TOKEN", "")

# ---------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH         = os.environ.get("MODEL_PATH", os.path.join(_BASE, "models"))
MODEL_FILE         = os.path.join(MODEL_PATH, "model.lgb")
MODEL_WEIGHTS_FILE = os.path.join(MODEL_PATH, "model_weights.json")  # LR fallback
SCALER_FILE        = os.path.join(MODEL_PATH, "scaler_params.json")
ZONE_META_FILE     = os.path.join(MODEL_PATH, "zone_meta.json")

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

LAG_HOURS  = [1, 2, 3, 6, 12, 24]
MA_WINDOWS = [3, 6, 12, 24]

FEATURE_NAMES = [
    # Raw time
    "hour", "day_of_week", "month", "day_of_month", "quarter", "is_weekend",
    # Cyclical time — help the model learn circular patterns (23h ≈ 0h)
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    # Calendar
    "is_holiday",
    # Zone
    "zone_id", "capacity", "zone_type_standard",
    # Weather
    "temperature", "is_precipitation",
    # Occupancy history
    "occupancy_lag_1h",  "occupancy_lag_2h",  "occupancy_lag_3h",
    "occupancy_lag_6h",  "occupancy_lag_12h", "occupancy_lag_24h",
    "occupancy_ma_3h",   "occupancy_ma_6h",   "occupancy_ma_12h", "occupancy_ma_24h",
]

CATEGORICAL_FEATURES = ["zone_id"]

THRESHOLD_LOW    = 0.33
THRESHOLD_MEDIUM = 0.67

CLASS_CENTER_RATES = {0: 0.15, 1: 0.50, 2: 0.83}

TEMP_FALLBACK_BY_MONTH = {
    1: -5, 2: -4, 3: 2, 4: 9, 5: 15, 6: 20,
    7: 22, 8: 20, 9: 14, 10: 7, 11: 1, 12: -3,
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         3,
    "num_leaves":        63,
    "n_estimators":      500,
    "learning_rate":     0.05,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbose":           -1,
}

TRAIN_DAYS_BACK  = int(os.environ.get("TRAIN_DAYS_BACK", "200"))
ML_MODEL_TYPE    = "lightgbm"
ML_MODEL_VERSION = "3.0"

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

RETRAIN_HOUR_UTC = 2
