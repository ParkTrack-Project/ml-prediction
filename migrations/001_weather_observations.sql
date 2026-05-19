-- Migration 001: weather_observations table
-- Run as a superuser / DB owner before starting the ML service.
--
--   psql -h <host> -U <admin_user> -d parktrack -f migrations/001_weather_observations.sql

CREATE TABLE IF NOT EXISTS weather_observations (
    id            SERIAL PRIMARY KEY,
    camera_id     INTEGER NOT NULL REFERENCES cameras(camera_id),
    observed_at   TIMESTAMPTZ NOT NULL,
    temperature   FLOAT NOT NULL,
    precipitation FLOAT NOT NULL DEFAULT 0.0,
    source        VARCHAR(50) NOT NULL DEFAULT 'open_meteo',
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (camera_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_weather_camera_time
    ON weather_observations (camera_id, observed_at);

-- Grant access to the ML service DB user (replace <ml_user> with the actual username):
-- GRANT SELECT, INSERT ON weather_observations TO <ml_user>;
-- GRANT USAGE, SELECT ON SEQUENCE weather_observations_id_seq TO <ml_user>;
