"""Paths and settings shared by every script.

Everything lives under one data folder (default: <project>/data). Set the
F1_DATA_DIR environment variable to put it somewhere else, which the tests do.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
FIRST_SEASON = 1950
FASTF1_FIRST_SEASON = 2018
SPRINT_FIRST_SEASON = 2021   # the first season with sprint races
PAGE_LIMIT = 100  # the largest page size the API accepts
# Seconds between requests. Jolpica is rate limited, so stay polite; raise this
# (for example F1_REQUEST_DELAY=2) if you see lots of "HTTP 429" waits.
MIN_SECONDS_BETWEEN_REQUESTS = float(os.environ.get("F1_REQUEST_DELAY", "1.0"))
USER_AGENT = "f1-project (personal portfolio project)"


def data_dir() -> Path:
    return Path(os.environ.get("F1_DATA_DIR", ROOT / "data"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def jolpica_raw_dir() -> Path:
    return raw_dir() / "jolpica"


def fastf1_raw_dir() -> Path:
    return raw_dir() / "fastf1"


def fastf1_cache_dir() -> Path:
    return data_dir() / "fastf1_cache"


def db_path() -> Path:
    return data_dir() / "f1.duckdb"


def current_season() -> int:
    return date.today().year


def outputs_dir() -> Path:
    """Where analysis tables and charts are written (kept in Git, they are small)."""
    return Path(os.environ.get("F1_OUTPUT_DIR", ROOT / "outputs"))


def forecasts_dir() -> Path:
    """Where the logged forecasts live: a folder of files that are never edited, committed to Git so the date can be checked."""
    return Path(os.environ.get("F1_FORECAST_DIR", ROOT / "forecasts"))
