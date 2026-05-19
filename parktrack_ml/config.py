import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ParkTrack API  (matching ParkTrack-Project/ml-prediction conventions)
# ---------------------------------------------------------------------------

API_URL   = os.environ.get('API_URL',   'https://api.parktrack.live')
API_TOKEN = os.environ.get('API_TOKEN', '')

# ---------------------------------------------------------------------------
# Database  (direct access for bulk training data; credentials via env only)
# ---------------------------------------------------------------------------

def _db_config():
    missing = [v for v in ('DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD') if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Required env vars not set: {', '.join(missing)}")
    return {
        'host':            os.environ['DB_HOST'],
        'port':            int(os.environ.get('DB_PORT', '5432')),
        'dbname':          os.environ['DB_NAME'],
        'user':            os.environ['DB_USER'],
        'password':        os.environ['DB_PASSWORD'],
        'connect_timeout': 20,
    }

DB_CONFIG = _db_config()

# ---------------------------------------------------------------------------
# Model storage  (matching MODEL_PATH convention from ml-prediction)
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH         = os.environ.get('MODEL_PATH', os.path.join(_BASE, 'models'))
MODEL_WEIGHTS_FILE = os.path.join(MODEL_PATH, 'model_weights.json')
SCALER_FILE        = os.path.join(MODEL_PATH, 'scaler_params.json')
ZONE_META_FILE     = os.path.join(MODEL_PATH, 'zone_meta.json')

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

LAG_HOURS  = [1, 2, 3, 6, 12, 24]
MA_WINDOWS = [3, 6, 12, 24]

FEATURE_NAMES = [
    'hour', 'day_of_week', 'month', 'day_of_month', 'quarter', 'is_weekend',
    'capacity', 'zone_type_standard',
    'temperature', 'is_precipitation',
    'occupancy_lag_1h',  'occupancy_lag_2h',  'occupancy_lag_3h',
    'occupancy_lag_6h',  'occupancy_lag_12h', 'occupancy_lag_24h',
    'occupancy_ma_3h', 'occupancy_ma_6h', 'occupancy_ma_12h', 'occupancy_ma_24h',
]

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

MODEL_PARAMS = {
    'learning_rate':  0.05,
    'iterations':     1000,
    'regularization': 0.01,
}

TRAIN_DAYS_BACK  = int(os.environ.get('TRAIN_DAYS_BACK', '200'))
ML_MODEL_TYPE    = 'logistic_regression'
ML_MODEL_VERSION = '2.1'

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

RETRAIN_HOUR_UTC = 2
