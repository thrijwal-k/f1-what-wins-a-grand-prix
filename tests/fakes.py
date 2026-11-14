"""Small synthetic raw files in the exact shape the real sources return."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PEOPLE = {
    "farina": ("FAR", "Giuseppe", "Farina", 1),
    "fagioli": ("FAG", "Luigi", "Fagioli", 2),
    "parnell": ("PAR", "Reg", "Parnell", 3),
    "max_verstappen": ("VER", "Max", "Verstappen", 33),
    "hamilton": ("HAM", "Lewis", "Hamilton", 44),
    "bottas": ("BOT", "Valtteri", "Bottas", 77),
}
SPRINT_ROUNDS = (5, 10)
POINTS = {1950: [8, 6, 4], 2021: [25, 18, 15]}


def _driver(driver_id):
    code, given, family, number = PEOPLE[driver_id]
    return {"driverId": driver_id, "code": code, "givenName": given, "familyName": family,
            "dateOfBirth": "1980-01-01", "nationality": "Testland", "permanentNumber": str(number)}


def _race_header(season, rnd):
    return {"season": str(season), "round": str(rnd), "raceName": f"Test GP {season}-{rnd}",
            "date": _date(season, rnd),
            "time": "13:00:00Z",
            "Circuit": {"circuitId": f"circuit_{rnd % 5}", "circuitName": f"Circuit {rnd % 5}",
                        "Location": {"lat": "10.5", "long": "20.5", "locality": "Town", "country": "Testland"}}}


def _date(season, rnd):
    # one race every 10 days, always in the past, ascending with the round
    return (pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=10 * rnd)).strftime("%Y-%m-%d")


def result_rows(season, order, laps=5):
    pts = POINTS.get(season, POINTS[2021])
    rows = []
    for i, driver_id in enumerate(order, start=1):
        rows.append({
            "number": str(PEOPLE[driver_id][3]), "position": str(i), "positionText": str(i),
            "points": str(pts[i - 1]), "Driver": _driver(driver_id),
            "Constructor": {"constructorId": "team_a", "name": "Team A", "nationality": "Testland"},
            "grid": str(i), "laps": str(laps), "status": "Finished",
            "Time": {"millis": str(5_400_000 + i * 1000), "time": f"1:30:0{i}.000"},
            "FastestLap": {"rank": str(i), "lap": "3", "Time": {"time": "1:20.000"}}})
    return rows


def write_jolpica(data_dir: Path, seasons=(1950, 2021)):
    base = data_dir / "raw" / "jolpica"
    for kind in ("races", "results", "qualifying", "sprint", "pitstops"):
        (base / kind).mkdir(parents=True, exist_ok=True)
    for season in seasons:
        n_rounds = {1950: 7, 2021: 22}.get(season, 2)
        orders = (["farina", "fagioli", "parnell"] if season == 1950
                  else ["max_verstappen", "hamilton", "bottas"])
        races = [_race_header(season, r) for r in range(1, n_rounds + 1)]
        results = [dict(_race_header(season, r), Results=result_rows(season, orders)) for r in range(1, n_rounds + 1)]
        (base / "races" / f"{season}.json").write_text(json.dumps(races))
        (base / "results" / f"{season}.json").write_text(json.dumps(results))
        quali = []
        if season >= 2021:
            for r in range(1, n_rounds + 1):
                quali.append(dict(_race_header(season, r), QualifyingResults=[
                    {"number": "33", "position": "1", "Driver": _driver("max_verstappen"),
                     "Constructor": {"constructorId": "team_a", "name": "Team A", "nationality": "Testland"},
                     "Q1": "1:20.1", "Q2": "1:19.9", "Q3": "1:19.5"}]))
        (base / "qualifying" / f"{season}.json").write_text(json.dumps(quali))
        sprints = []
        if season >= 2021:  # rounds 5 and 10 are sprint weekends in the fake data
            for r in SPRINT_ROUNDS:
                sprints.append(dict(_race_header(season, r), SprintResults=[
                    dict(row, points=str(p)) for row, p in zip(result_rows(season, orders), (3, 2, 1))]))
        (base / "sprint" / f"{season}.json").write_text(json.dumps(sprints))
        if season >= 2011:
            for r in (1, 2):
                stops = dict(_race_header(season, r), PitStops=[
                    {"driverId": "max_verstappen", "stop": "1", "lap": "2", "time": "14:05:01", "duration": "22.4"},
                    {"driverId": "hamilton", "stop": "1", "lap": "3", "time": "14:06:01", "duration": "1:02.5"}])
                (base / "pitstops" / f"{season}_{r:02d}.json").write_text(json.dumps([stops]))


def write_fastf1(data_dir: Path, rounds=(1, 2), season=2021, practice=True):
    base = data_dir / "raw" / "fastf1"
    base.mkdir(parents=True, exist_ok=True)
    drivers = [("VER", 33), ("HAM", 44), ("BOT", 77)]
    for rnd in rounds:
        laps = []
        for abbr, num in drivers:
            for lap in range(1, 6):
                laps.append({"abbreviation": abbr, "driver_number": num, "lap_number": lap,
                             "lap_time_s": 80.0 + lap * 0.1, "stint": 1 if lap <= 3 else 2,
                             "compound": "MEDIUM" if lap <= 3 else "HARD", "tyre_life": float(lap),
                             "fresh_tyre": True, "pit_in_s": None, "pit_out_s": None,
                             "track_status": "1", "position": 1, "is_accurate": True})
        pd.DataFrame(laps).to_parquet(base / f"{season}_{rnd:02d}_laps.parquet", index=False)
        pd.DataFrame([{"abbreviation": a, "driver_number": n, "full_name": a, "team_name": "Team A",
                       "position": i, "grid_position": i} for i, (a, n) in enumerate(drivers, start=1)]
                     ).to_parquet(base / f"{season}_{rnd:02d}_results.parquet", index=False)
        pd.DataFrame([{"time_s": 60.0 * t, "air_temp": 25.0, "track_temp": 35.0, "humidity": 50.0,
                       "pressure": 1010.0, "wind_speed": 2.0, "wind_direction": 90.0, "rainfall": False}
                      for t in range(3)]).to_parquet(base / f"{season}_{rnd:02d}_weather.parquet", index=False)
        pd.DataFrame([{"abbreviation": a, "lap_number": lap, "lap_end_s": 100.0 * lap + 0.3 * i}
                      for i, (a, _) in enumerate(drivers) for lap in range(1, 6)]
                     ).to_parquet(base / f"{season}_{rnd:02d}_laptimes.parquet", index=False)
        if practice:
            rows = []
            for session in ("FP1", "FP2"):
                for abbr, num in drivers + [("RES", 99)]:
                    for lap in range(1, 7):
                        rows.append({"session": session, "abbreviation": abbr, "driver_number": num, "lap_number": lap,
                                     "lap_time_s": 82.0 + 0.1 * lap, "stint": 1, "compound": "MEDIUM", "tyre_life": float(lap),
                                     "fresh_tyre": True, "pit_in_s": None, "pit_out_s": None, "track_status": "1",
                                     "position": None, "is_accurate": True, "session_time_s": 600.0 + 100.0 * lap})
            pd.DataFrame(rows).to_parquet(base / f"{season}_{rnd:02d}_practice.parquet", index=False)
