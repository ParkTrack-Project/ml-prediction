"""
SmartParking ML service.

Runs as a standalone process (or Docker container).
Handles weather collection, model training, and forecast publishing — all internally scheduled.

Environment variables:
    DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
    API_URL
    API_TOKEN   — token with forecasts.write permission
"""
import os
import sys
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR

from .config import MODEL_WEIGHTS_FILE, RETRAIN_HOUR_UTC, TRAIN_DAYS_BACK
from .weather import setup as setup_weather_table, backfill as backfill_weather, fetch_latest as fetch_weather
from .forecaster import run as generate_forecasts
from .train import train
from .interfaces import reload as reload_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('service')


def _retrain():
    logger.info("Retraining model...")
    try:
        train()
        reload_model()
        logger.info("Model retrained and reloaded")
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}")


def _on_job_error(event):
    logger.error(f"Job '{event.job_id}' raised {event.exception!r}")


def startup():
    logger.info("=== SmartParking ML service startup ===")

    logger.info("Setting up weather table...")
    setup_weather_table()

    logger.info("Backfilling weather history...")
    n = backfill_weather(days=TRAIN_DAYS_BACK)
    logger.info(f"  {n} weather rows upserted")

    if not os.path.exists(MODEL_WEIGHTS_FILE):
        logger.info("No saved model found — training from scratch...")
        train()

    reload_model()

    logger.info("Generating initial forecasts...")
    generate_forecasts()

    logger.info("Startup complete")


def main():
    startup()

    scheduler = BlockingScheduler(timezone='UTC')
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)

    scheduler.add_job(
        generate_forecasts,
        CronTrigger(minute='0,30'),
        id='forecasts',
        name='Generate & post forecasts',
        misfire_grace_time=60,
        coalesce=True,
    )
    scheduler.add_job(
        fetch_weather,
        CronTrigger(minute=5),
        id='weather',
        name='Fetch weather from Open-Meteo',
        misfire_grace_time=120,
        coalesce=True,
    )
    scheduler.add_job(
        _retrain,
        CronTrigger(hour=RETRAIN_HOUR_UTC, minute=0),
        id='retrain',
        name='Retrain model',
        misfire_grace_time=600,
        coalesce=True,
    )

    logger.info("Scheduler running — forecasts every :00/:30, weather every :05, retrain daily at 02:00 UTC")
    scheduler.start()


if __name__ == '__main__':
    main()
