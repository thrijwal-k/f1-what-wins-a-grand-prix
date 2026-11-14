"""Download lap, tyre and weather detail for each race from FastF1 (2018 onward), and the weekend's practice laps.

Car telemetry is NOT loaded: it is huge and the plan does not need it. For each
finished race this writes three small parquet files under data/raw/fastf1/:
    <year>_<round>_laps.parquet, _results.parquet, _weather.parquet
with column names already normalised, so the database build does not depend on
FastF1's own naming. Practice sessions go to <year>_<round>_practice.parquet (one file per weekend, all of
its practice sessions in one table with a `session` column), in a separate pass so a rate-limit stop in one
does not hold up the other.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# FastF1 allows about 500 API calls per hour. Each race uses roughly 10, so a
# full download hits the limit about every 50 races. When that happens we wait
# and carry on instead of failing the rest of the season.
RATE_LIMIT_WAIT_S = 300
RATE_LIMIT_MAX_WAITS = 24  # 24 x 5 minutes = up to 2 hours per race


def is_rate_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "RateLimitExceededError" or "calls/h" in str(exc)


def call_with_rate_limit_wait(fn, what: str, sleep=time.sleep,
                              wait_s: float = RATE_LIMIT_WAIT_S,
                              max_waits: int = RATE_LIMIT_MAX_WAITS):
    """Run fn(); if FastF1's hourly call limit is hit, wait and try again."""
    waits = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit(exc) and waits < max_waits:
                waits += 1
                log.warning("FastF1 hourly call limit reached at %s; waiting %.0f min (%s/%s)",
                            what, wait_s / 60, waits, max_waits)
                sleep(wait_s)
                continue
            raise


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def _seconds(series: pd.Series) -> pd.Series:
    """Timedelta column -> float seconds (NaN where missing)."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def normalise_laps(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["abbreviation"] = _col(df, "Driver").astype("string")
    out["driver_number"] = pd.to_numeric(_col(df, "DriverNumber"), errors="coerce")
    out["lap_number"] = pd.to_numeric(_col(df, "LapNumber"), errors="coerce")
    out["lap_time_s"] = _seconds(_col(df, "LapTime"))
    out["stint"] = pd.to_numeric(_col(df, "Stint"), errors="coerce")
    out["compound"] = _col(df, "Compound").astype("string")
    out["tyre_life"] = pd.to_numeric(_col(df, "TyreLife"), errors="coerce")
    out["fresh_tyre"] = _col(df, "FreshTyre").astype("boolean")
    out["pit_in_s"] = _seconds(_col(df, "PitInTime"))
    out["pit_out_s"] = _seconds(_col(df, "PitOutTime"))
    out["track_status"] = _col(df, "TrackStatus").astype("string")
    out["position"] = pd.to_numeric(_col(df, "Position"), errors="coerce")
    out["is_accurate"] = _col(df, "IsAccurate").astype("boolean")
    return out.reset_index(drop=True)


PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")


def normalise_practice(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """A practice session's laps in the same columns as a race's, plus `session` (FP1/FP2/FP3) and `session_time_s`
    (seconds into the session when the lap was completed, which lets track evolution be allowed for)."""
    out = normalise_laps(df)
    out["session_time_s"] = _seconds(_col(df, "Time")).reset_index(drop=True)
    out.insert(0, "session", session)
    return out


def practice_sessions_of(event, wanted=PRACTICE_SESSIONS) -> list[str]:
    """Which of FP1-FP3 this event held, read from its schedule row (sprint weekends have only FP1; some are cut short)."""
    held = []
    for i in range(1, 6):
        name = str(event.get(f"Session{i}", "") or "")
        if name.startswith("Practice"):
            code = "FP" + name.split()[-1]
            if code in wanted:
                held.append(code)
    return held


def normalise_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    """Per lap, the session clock when the car crossed the line (FastF1's `Time`): the gap between two cars on a lap is the
    difference of these. Kept apart from the laps file so races already downloaded need only this small extra pass."""
    out = pd.DataFrame(index=df.index)
    out["abbreviation"] = _col(df, "Driver").astype("string")
    out["lap_number"] = pd.to_numeric(_col(df, "LapNumber"), errors="coerce")
    out["lap_end_s"] = _seconds(_col(df, "Time"))
    return out.reset_index(drop=True)


def lap_times_path(year: int, rnd: int):
    return config.fastf1_raw_dir() / f"{year}_{rnd:02d}_laptimes.parquet"


def normalise_results(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["abbreviation"] = _col(df, "Abbreviation").astype("string")
    out["driver_number"] = pd.to_numeric(_col(df, "DriverNumber"), errors="coerce")
    out["full_name"] = _col(df, "FullName").astype("string")
    out["team_name"] = _col(df, "TeamName").astype("string")
    out["position"] = pd.to_numeric(_col(df, "Position"), errors="coerce")
    out["grid_position"] = pd.to_numeric(_col(df, "GridPosition"), errors="coerce")
    return out.reset_index(drop=True)


def normalise_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["time_s"] = _seconds(_col(df, "Time"))
    for src, dst in [("AirTemp", "air_temp"), ("TrackTemp", "track_temp"),
                     ("Humidity", "humidity"), ("Pressure", "pressure"),
                     ("WindSpeed", "wind_speed"), ("WindDirection", "wind_direction")]:
        out[dst] = pd.to_numeric(_col(df, src), errors="coerce")
    out["rainfall"] = _col(df, "Rainfall").astype("boolean")
    return out.reset_index(drop=True)


def _write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _paths(year: int, rnd: int):
    base = config.fastf1_raw_dir()
    return {kind: base / f"{year}_{rnd:02d}_{kind}.parquet" for kind in ("laps", "results", "weather")}


def practice_path(year: int, rnd: int):
    return config.fastf1_raw_dir() / f"{year}_{rnd:02d}_practice.parquet"


def download(from_season: int = config.FASTF1_FIRST_SEASON, to_season: int | None = None,
             force: bool = False) -> None:
    import fastf1  # imported here so the rest of the project works without it

    cache = config.fastf1_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    to_season = to_season or config.current_season()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    failed_log = config.fastf1_raw_dir() / "_failed.log"

    for year in range(max(from_season, config.FASTF1_FIRST_SEASON), to_season + 1):
        try:
            schedule = call_with_rate_limit_wait(
                lambda y=year: fastf1.get_event_schedule(y, include_testing=False), f"{year} schedule")
        except Exception as exc:  # noqa: BLE001 - keep going, record it
            log.warning("FastF1 schedule for %s failed: %s", year, exc)
            continue
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            if rnd < 1:
                continue
            if pd.Timestamp(event["EventDate"]).normalize() >= today:
                continue  # race not finished yet
            paths = _paths(year, rnd)
            if all(p.exists() for p in paths.values()) and not force:
                continue
            def load_race(year=year, rnd=rnd, paths=paths):
                session = fastf1.get_session(year, rnd, "R")
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                _write(normalise_laps(session.laps), paths["laps"])
                _write(normalise_results(session.results), paths["results"])
                weather = session.weather_data
                _write(normalise_weather(weather if weather is not None else pd.DataFrame()),
                       paths["weather"])

            try:
                call_with_rate_limit_wait(load_race, f"{year} round {rnd}")
                log.info("FastF1 %s round %s saved", year, rnd)
            except Exception as exc:  # noqa: BLE001
                log.warning("FastF1 %s round %s failed: %s", year, rnd, exc)
                failed_log.parent.mkdir(parents=True, exist_ok=True)
                with failed_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now(timezone.utc).isoformat()} {year} {rnd} {exc}\n")


def download_practice(from_season: int = config.FASTF1_FIRST_SEASON, to_season: int | None = None,
                      force: bool = False, sessions=PRACTICE_SESSIONS) -> None:
    """Practice laps for every finished weekend. Laps only (no telemetry, weather or messages), one file per weekend.
    Resumable: a weekend whose file exists is skipped; a failed weekend is logged and tried again next time."""
    import fastf1

    cache = config.fastf1_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    to_season = to_season or config.current_season()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    failed_log = config.fastf1_raw_dir() / "_failed.log"

    for year in range(max(from_season, config.FASTF1_FIRST_SEASON), to_season + 1):
        try:
            schedule = call_with_rate_limit_wait(
                lambda y=year: fastf1.get_event_schedule(y, include_testing=False), f"{year} schedule")
        except Exception as exc:  # noqa: BLE001
            log.warning("FastF1 schedule for %s failed: %s", year, exc)
            continue
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            if rnd < 1 or pd.Timestamp(event["EventDate"]).normalize() >= today:
                continue
            path = practice_path(year, rnd)
            if path.exists() and not force:
                continue

            def load_weekend(year=year, rnd=rnd, event=event, path=path):
                frames = []
                for code in practice_sessions_of(event, sessions):
                    session = fastf1.get_session(year, rnd, code)
                    session.load(laps=True, telemetry=False, weather=False, messages=False)
                    laps = session.laps
                    if laps is not None and len(laps):
                        frames.append(normalise_practice(laps, code))
                cols = list(normalise_practice(pd.DataFrame(), "FP1").columns)
                _write(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols), path)

            try:
                call_with_rate_limit_wait(load_weekend, f"{year} round {rnd} practice")
                log.info("FastF1 %s round %s practice saved", year, rnd)
            except Exception as exc:  # noqa: BLE001
                log.warning("FastF1 %s round %s practice failed: %s", year, rnd, exc)
                failed_log.parent.mkdir(parents=True, exist_ok=True)
                with failed_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now(timezone.utc).isoformat()} {year} {rnd} practice {exc}\n")


def download_lap_times(from_season: int = config.FASTF1_FIRST_SEASON, to_season: int | None = None, force: bool = False) -> None:
    """The line-crossing clock of every race lap, one small file per race. Races loaded before with the same cache usually
    need no new API calls. Resumable; a failed race is logged and tried again next time."""
    import fastf1

    cache = config.fastf1_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    to_season = to_season or config.current_season()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    failed_log = config.fastf1_raw_dir() / "_failed.log"

    for year in range(max(from_season, config.FASTF1_FIRST_SEASON), to_season + 1):
        try:
            schedule = call_with_rate_limit_wait(
                lambda y=year: fastf1.get_event_schedule(y, include_testing=False), f"{year} schedule")
        except Exception as exc:  # noqa: BLE001
            log.warning("FastF1 schedule for %s failed: %s", year, exc)
            continue
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            if rnd < 1 or pd.Timestamp(event["EventDate"]).normalize() >= today:
                continue
            path = lap_times_path(year, rnd)
            if path.exists() and not force:
                continue

            def load_race(year=year, rnd=rnd, path=path):
                session = fastf1.get_session(year, rnd, "R")
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                _write(normalise_lap_times(session.laps), path)

            try:
                call_with_rate_limit_wait(load_race, f"{year} round {rnd} lap times")
                log.info("FastF1 %s round %s lap times saved", year, rnd)
            except Exception as exc:  # noqa: BLE001
                log.warning("FastF1 %s round %s lap times failed: %s", year, rnd, exc)
                failed_log.parent.mkdir(parents=True, exist_ok=True)
                with failed_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now(timezone.utc).isoformat()} {year} {rnd} laptimes {exc}\n")
