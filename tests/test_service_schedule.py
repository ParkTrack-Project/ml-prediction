import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

with patch("dotenv.load_dotenv"):
    from parktrack_ml import api_client


class FakeCronTrigger:
    def __init__(self, *, minute, hour=None, timezone=None):
        self.timezone = ZoneInfo(timezone or "Europe/Moscow")
        self.minutes = {int(value) for value in str(minute).split(",")}
        self.hour = hour

    def next_after(self, now):
        candidate = now.astimezone(self.timezone).replace(second=0, microsecond=0) + timedelta(minutes=1)
        while candidate.minute not in self.minutes or (self.hour is not None and candidate.hour != self.hour):
            candidate += timedelta(minutes=1)
        return candidate


class ServiceScheduleTests(unittest.TestCase):
    def load_service(self):
        scheduler = Mock()
        modules = {name: ModuleType(name) for name in (
            "apscheduler", "apscheduler.schedulers", "apscheduler.schedulers.blocking",
            "apscheduler.triggers", "apscheduler.triggers.cron", "apscheduler.events",
            "parktrack_ml.weather", "parktrack_ml.forecaster", "parktrack_ml.train",
        )}
        factory = Mock(return_value=scheduler)
        modules["apscheduler.schedulers.blocking"].BlockingScheduler = factory
        modules["apscheduler.triggers.cron"].CronTrigger = FakeCronTrigger
        modules["apscheduler.events"].EVENT_JOB_ERROR = 1
        modules["parktrack_ml.weather"].backfill = Mock()
        modules["parktrack_ml.weather"].fetch_latest = Mock()
        modules["parktrack_ml.forecaster"].run = Mock()
        modules["parktrack_ml.train"].train = Mock()
        spec = importlib.util.spec_from_file_location(
            "parktrack_ml._service_schedule_test", Path(api_client.__file__).with_name("service.py"),
        )
        service = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch("logging.basicConfig"):
            spec.loader.exec_module(service)
        return service, scheduler, factory

    def test_all_cron_jobs_use_utc_even_when_default_zone_is_moscow(self):
        service, scheduler, factory = self.load_service()
        with patch.object(service, "startup"), patch.object(service, "RETRAIN_HOUR_UTC", 2):
            service.main()
        factory.assert_called_once_with(timezone="UTC")
        jobs = {call.kwargs["id"]: call.args[1] for call in scheduler.add_job.call_args_list}
        now = datetime(2026, 11, 18, 0, 10, tzinfo=timezone.utc)
        expected = {"forecasts": (0, 30), "weather": (1, 5), "retrain": (2, 0)}
        for job_id, (hour, minute) in expected.items():
            with self.subTest(job_id=job_id):
                trigger = jobs[job_id]
                self.assertEqual(trigger.timezone.key, "UTC")
                self.assertEqual(trigger.next_after(now), now.replace(hour=hour, minute=minute))
        scheduler.start.assert_called_once()

    def test_backfill_remains_a_background_date_job(self):
        service, scheduler, _ = self.load_service()
        with patch.object(service, "startup") as startup:
            service.main()
        startup.assert_called_once()
        jobs = [call for call in scheduler.add_job.call_args_list if call.kwargs["id"] == "weather_backfill"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].args, (service._backfill_weather, "date"))
        self.assertIsNone(jobs[0].kwargs["misfire_grace_time"])
        service.backfill_weather.assert_not_called()
        service.train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
