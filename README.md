# parktrack-ml

ML microservice for parking occupancy forecasting. Consists of two independent scripts:

- **train** — fetches historical occupancy data from ParkTrack API, trains a LightGBM model, saves it to disk.
- **predict** — loads the saved model, generates forecasts for configurable future horizons, posts them to `POST /forecasts/new`.

Both scripts are packaged as separate Docker images and can be scheduled as cron jobs.

---

## Project structure

```
parktrack_ml/
├── api_client.py   # ParkTrack API wrapper (Bearer token auth)
├── features.py     # Feature engineering
├── train.py        # Training script entry point
└── predict.py      # Prediction script entry point
Dockerfile.train
Dockerfile.predict
docker-compose.yml
.env.example
requirements.txt
```

---

## Quick start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and set API_URL and API_TOKEN
```

### 2. Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python -m parktrack_ml.train

# Predict (requires a trained model)
python -m parktrack_ml.predict
```

### 3. Run with Docker Compose

```bash
# Train
docker compose --profile train up --build

# Predict
docker compose --profile predict up --build
```

---

## Environment variables

| Variable            | Required | Default                      | Description                                      |
|---------------------|----------|------------------------------|--------------------------------------------------|
| `API_URL`           | yes      | —                            | ParkTrack API base URL (e.g. `https://api.parktrack.live/api/v1`) |
| `API_TOKEN`         | yes      | —                            | Bearer token with `forecasts.write` permission   |
| `MODEL_PATH`        | no       | `models/forecast_model.pkl`  | Path to save/load the trained model              |
| `FORECAST_HORIZONS` | no       | `15,30,60`                   | Comma-separated list of forecast horizons (minutes) |
| `TRAIN_DAYS_BACK`   | no       | `90`                         | Days of historical data to use for training      |

---

## Docker images

Images are published to GHCR automatically via GitHub Actions on every push to `main` or `development`:

| Image | Tag |
|-------|-----|
| `ghcr.io/parktrack-project/parktrack-ml-train` | `latest` / `development` / `sha-xxxxxxx` |
| `ghcr.io/parktrack-project/parktrack-ml-predict` | `latest` / `development` / `sha-xxxxxxx` |

---

## Model details

- **Algorithm**: LightGBM (`LGBMRegressor`)
- **Features**: `zone_id`, `hour`, `minute`, `day_of_week`, `is_weekend`, `month`, `horizon_minutes`
- **Target**: `occupied` N minutes into the future
- **Training split**: 85% train / 15% validation (time-ordered, no shuffle)
- **Metric**: Mean Absolute Error (MAE) on validation set

The model artifact (`.pkl`) includes the trained model, feature column names, configured horizons, and a per-zone capacity map derived from training data.

---

## Cron scheduling

Train weekly, predict every 15 minutes (example crontab):

```cron
# Retrain every Sunday at 02:00 UTC
0 2 * * 0 docker compose -f /opt/parktrack-ml/docker-compose.yml --profile train up --build

# Predict every 15 minutes
*/15 * * * * docker compose -f /opt/parktrack-ml/docker-compose.yml --profile predict up
```

---

## Integration with ParkTrack deploy

Add to the `deploy/docker-compose.yml`:

```yaml
  ml-predict:
    image: ghcr.io/parktrack-project/parktrack-ml-predict:latest
    environment:
      - API_URL=http://api-server:8000/api/v1
      - API_TOKEN=${ML_API_TOKEN}
      - MODEL_PATH=/models/forecast_model.pkl
      - FORECAST_HORIZONS=15,30,60
    volumes:
      - ml_models:/models
    depends_on:
      - api-server
    restart: unless-stopped

volumes:
  ml_models:
```
