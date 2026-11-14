"""Download Formula 1 history from the Jolpica-F1 API (the Ergast successor).

Design rules:
- Every response is saved untouched under data/raw/jolpica/, so the database
  can always be rebuilt without downloading again.
- Requests are throttled, and 429 / 5xx responses back off and retry.
- Downloads are resumable: finished past seasons are never fetched twice. The
  current season is re-fetched each run so new races appear.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)

# Keys that hold per-race lists and must be merged when a race spans pages.
LIST_KEYS = ("Results", "QualifyingResults", "PitStops", "SprintResults")


class Client:
    """Tiny throttled HTTP client for the Jolpica API."""

    def __init__(self, delay: float | None = None, session: requests.Session | None = None):
        self.delay = config.MIN_SECONDS_BETWEEN_REQUESTS if delay is None else delay
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self._last = 0.0
        self.requests_made = 0

    def get(self, path: str, params: dict | None = None, max_retries: int = 12) -> dict:
        url = f"{config.JOLPICA_BASE}/{path}"
        for attempt in range(max_retries):
            pause = self.delay - (time.monotonic() - self._last)
            if pause > 0:
                time.sleep(pause)
            self._last = time.monotonic()
            self.requests_made += 1
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                wait = min(300, 5 * 2**attempt)
                log.warning("Network error for %s (%s); waiting %ss", url, exc, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(600, 15 * 2**attempt)
                log.warning("HTTP %s from %s; waiting %.0fs", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Giving up on {url} after {max_retries} attempts")


def merge_races(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge pages of a paginated response: one race can span several pages."""
    index = {(r["season"], r["round"]): r for r in existing}
    for race in new:
        key = (race["season"], race["round"])
        if key not in index:
            index[key] = race
            existing.append(race)
        else:
            target = index[key]
            for list_key in LIST_KEYS:
                if list_key in race:
                    target.setdefault(list_key, []).extend(race[list_key])
    return existing


def fetch_all(client, path: str) -> list[dict]:
    """Fetch every page of an endpoint and return the merged list of races."""
    offset = 0
    merged: list[dict] = []
    while True:
        payload = client.get(path, {"limit": config.PAGE_LIMIT, "offset": offset})["MRData"]
        total = int(payload["total"])
        merge_races(merged, payload["RaceTable"]["Races"])
        offset += config.PAGE_LIMIT
        if offset >= total:
            return merged


def _save(path: Path, races: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(races), encoding="utf-8")
    tmp.replace(path)  # atomic: a stopped download never leaves a half file


def raw_path(kind: str, name: str) -> Path:
    return config.jolpica_raw_dir() / kind / f"{name}.json"


def download_season(client, season: int, current: int, today: str | None = None,
                    force: bool = False) -> None:
    """Download one season: race list, results, qualifying, (2021+) sprint results and (2011+) pit stops."""
    today = today or date.today().isoformat()
    refetch = force or season == current

    jobs = [
        ("races", str(season), f"{season}.json"),
        ("results", str(season), f"{season}/results.json"),
        ("qualifying", str(season), f"{season}/qualifying.json"),
    ]
    if season >= config.SPRINT_FIRST_SEASON:  # one small file per season; empty when no sprint was held
        jobs.append(("sprint", str(season), f"{season}/sprint.json"))
    races: list[dict] = []
    for kind, name, path in jobs:
        target = raw_path(kind, name)
        if target.exists() and not refetch:
            if kind == "races":
                races = json.loads(target.read_text(encoding="utf-8"))
            continue
        data = fetch_all(client, path)
        _save(target, data)
        if kind == "races":
            races = data

    if season < 2011:  # Jolpica has no pit stop data before 2011
        return
    for race in races:
        if race["date"] > today:
            continue  # not run yet
        rnd = int(race["round"])
        target = raw_path("pitstops", f"{season}_{rnd:02d}")
        if target.exists() and not force:
            stale = season == current and not json.loads(target.read_text(encoding="utf-8"))
            if not stale:
                continue
        _save(target, fetch_all(client, f"{season}/{rnd}/pitstops.json"))


def download_all(from_season: int = config.FIRST_SEASON, to_season: int | None = None,
                 force: bool = False, client: Client | None = None) -> None:
    client = client or Client()
    current = config.current_season()
    to_season = to_season or current
    for season in range(from_season, to_season + 1):
        before = client.requests_made
        download_season(client, season, current, force=force)
        log.info("Jolpica %s done (%s requests)", season, client.requests_made - before)
