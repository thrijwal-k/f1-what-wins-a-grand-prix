"""Build the DuckDB database from the untouched raw downloads.

The database is always rebuilt from scratch into a temporary file and swapped in
at the end, so a failed build never damages the existing database.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import duckdb
import pandas as pd

from . import config

log = logging.getLogger(__name__)

SCHEMAS: dict[str, dict[str, str]] = {
    "circuits": {"circuit_id": "VARCHAR", "circuit_name": "VARCHAR", "locality": "VARCHAR",
                 "country": "VARCHAR", "lat": "DOUBLE", "lon": "DOUBLE"},
    "races": {"race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "race_name": "VARCHAR",
              "circuit_id": "VARCHAR", "race_date": "DATE", "race_time": "VARCHAR"},
    "drivers": {"driver_id": "VARCHAR", "code": "VARCHAR", "given_name": "VARCHAR",
                "family_name": "VARCHAR", "birth_date": "DATE", "nationality": "VARCHAR"},
    "constructors": {"constructor_id": "VARCHAR", "name": "VARCHAR", "nationality": "VARCHAR"},
    "results": {"race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "driver_id": "VARCHAR",
                "driver_code": "VARCHAR", "constructor_id": "VARCHAR", "car_number": "BIGINT",
                "grid": "BIGINT", "position": "BIGINT", "position_text": "VARCHAR",
                "classified": "BOOLEAN", "points": "DOUBLE", "laps": "BIGINT", "status": "VARCHAR",
                "time_millis": "BIGINT", "fastest_lap_rank": "BIGINT",
                "fastest_lap_number": "BIGINT", "fastest_lap_time": "VARCHAR"},
    "sprint_results": {"race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "driver_id": "VARCHAR",
                       "driver_code": "VARCHAR", "constructor_id": "VARCHAR", "car_number": "BIGINT",
                       "grid": "BIGINT", "position": "BIGINT", "position_text": "VARCHAR",
                       "classified": "BOOLEAN", "points": "DOUBLE", "laps": "BIGINT", "status": "VARCHAR",
                       "time_millis": "BIGINT"},
    "qualifying": {"race_id": "BIGINT", "driver_id": "VARCHAR", "constructor_id": "VARCHAR",
                   "position": "BIGINT", "q1": "VARCHAR", "q2": "VARCHAR", "q3": "VARCHAR"},
    "pit_stops": {"race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "driver_id": "VARCHAR",
                  "stop": "BIGINT", "lap": "BIGINT", "local_time": "VARCHAR", "duration_s": "DOUBLE"},
    "fastf1_results": {"race_id": "BIGINT", "abbreviation": "VARCHAR", "driver_id": "VARCHAR",
                       "driver_number": "BIGINT", "full_name": "VARCHAR", "team_name": "VARCHAR",
                       "position": "BIGINT", "grid_position": "BIGINT"},
    "laps": {"race_id": "BIGINT", "abbreviation": "VARCHAR", "driver_id": "VARCHAR",
             "lap_number": "BIGINT", "lap_time_s": "DOUBLE", "stint": "BIGINT", "compound": "VARCHAR",
             "tyre_life": "DOUBLE", "fresh_tyre": "BOOLEAN", "pit_in_s": "DOUBLE",
             "pit_out_s": "DOUBLE", "track_status": "VARCHAR", "position": "BIGINT",
             "is_accurate": "BOOLEAN", "lap_end_s": "DOUBLE"},
    "practice_laps": {"race_id": "BIGINT", "session": "VARCHAR", "abbreviation": "VARCHAR", "driver_id": "VARCHAR",
                      "lap_number": "BIGINT", "lap_time_s": "DOUBLE", "stint": "BIGINT", "compound": "VARCHAR",
                      "tyre_life": "DOUBLE", "fresh_tyre": "BOOLEAN", "pit_in_s": "DOUBLE",
                      "pit_out_s": "DOUBLE", "track_status": "VARCHAR", "is_accurate": "BOOLEAN",
                      "session_time_s": "DOUBLE"},
    "weather": {"race_id": "BIGINT", "time_s": "DOUBLE", "air_temp": "DOUBLE", "track_temp": "DOUBLE",
                "humidity": "DOUBLE", "pressure": "DOUBLE", "wind_speed": "DOUBLE",
                "wind_direction": "DOUBLE", "rainfall": "BOOLEAN"},
    "driver_map": {"season": "BIGINT", "abbreviation": "VARCHAR", "driver_id": "VARCHAR"},
    "mismatches": {"kind": "VARCHAR", "race_id": "BIGINT", "detail": "VARCHAR"},
    "build_info": {"built_at": "VARCHAR", "jolpica_files": "BIGINT", "fastf1_races": "BIGINT"},
}


# ---------------------------------------------------------------- helpers

def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_time_str(text) -> float | None:
    """'22.456' -> 22.456 ; '1:02.300' -> 62.3 ; anything else -> None."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return None


def coerce(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Give a DataFrame exactly the schema's columns, in order, with clean dtypes."""
    out = pd.DataFrame(index=df.index)
    for col, typ in schema.items():
        series = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index, dtype="object")
        if typ == "BIGINT":
            series = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif typ == "DOUBLE":
            series = pd.to_numeric(series, errors="coerce").astype("float64")
        elif typ == "VARCHAR":
            series = series.astype("string")
        elif typ == "DATE":
            series = pd.to_datetime(series, errors="coerce")
        elif typ == "BOOLEAN":
            series = series.astype("boolean")
        out[col] = series
    return out


def write_table(con, name: str, df: pd.DataFrame) -> None:
    schema = SCHEMAS[name]
    clean = coerce(df, schema)
    con.register("_df", clean)
    cols = ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in schema.items())
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT {cols} FROM _df')
    con.unregister("_df")


# ------------------------------------------------------------ raw readers

def _json_files(kind: str):
    folder = config.jolpica_raw_dir() / kind
    return sorted(folder.glob("*.json")) if folder.exists() else []


def _races_from(kind: str):
    for path in _json_files(kind):
        yield from json.loads(path.read_text(encoding="utf-8"))


def _race_id(race: dict) -> int:
    return int(race["season"]) * 100 + int(race["round"])


def read_races() -> tuple[pd.DataFrame, pd.DataFrame]:
    races, circuits = [], []
    for race in _races_from("races"):
        circuit = race.get("Circuit", {})
        loc = circuit.get("Location", {})
        races.append({"race_id": _race_id(race), "season": race["season"], "round": race["round"],
                      "race_name": race.get("raceName"), "circuit_id": circuit.get("circuitId"),
                      "race_date": race.get("date"), "race_time": race.get("time")})
        circuits.append({"circuit_id": circuit.get("circuitId"), "circuit_name": circuit.get("circuitName"),
                         "locality": loc.get("locality"), "country": loc.get("country"),
                         "lat": _float(loc.get("lat")), "lon": _float(loc.get("long"))})
    circuits_df = pd.DataFrame(circuits, columns=list(SCHEMAS["circuits"]))
    circuits_df = circuits_df.drop_duplicates("circuit_id", keep="last")
    return pd.DataFrame(races, columns=list(SCHEMAS["races"])), circuits_df


def read_results() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    rows, drivers, constructors = [], [], []
    for race in _races_from("results"):
        rid = _race_id(race)
        for res in race.get("Results", []):
            drv, con = res["Driver"], res["Constructor"]
            fastest = res.get("FastestLap", {})
            pos_text = res.get("positionText")
            rows.append({
                "race_id": rid, "season": race["season"], "round": race["round"],
                "driver_id": drv["driverId"], "driver_code": drv.get("code"),
                "constructor_id": con["constructorId"], "car_number": _int(res.get("number")),
                "grid": _int(res.get("grid")), "position": _int(res.get("position")),
                "position_text": pos_text, "classified": bool(pos_text and str(pos_text).isdigit()),
                "points": _float(res.get("points")), "laps": _int(res.get("laps")),
                "status": res.get("status"), "time_millis": _int(res.get("Time", {}).get("millis")),
                "fastest_lap_rank": _int(fastest.get("rank")), "fastest_lap_number": _int(fastest.get("lap")),
                "fastest_lap_time": fastest.get("Time", {}).get("time"),
            })
            drivers.append(drv)
            constructors.append(con)
    return pd.DataFrame(rows, columns=list(SCHEMAS["results"])), drivers, constructors


def read_sprint_results() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Sprint races (2021 on). Same fields as a Sunday result; the sprint's own grid and finishing order."""
    rows, drivers, constructors = [], [], []
    for race in _races_from("sprint"):
        rid = _race_id(race)
        for res in race.get("SprintResults", []):
            drv, con = res["Driver"], res["Constructor"]
            pos_text = res.get("positionText")
            rows.append({
                "race_id": rid, "season": race["season"], "round": race["round"],
                "driver_id": drv["driverId"], "driver_code": drv.get("code"),
                "constructor_id": con["constructorId"], "car_number": _int(res.get("number")),
                "grid": _int(res.get("grid")), "position": _int(res.get("position")),
                "position_text": pos_text, "classified": bool(pos_text and str(pos_text).isdigit()),
                "points": _float(res.get("points")), "laps": _int(res.get("laps")),
                "status": res.get("status"), "time_millis": _int(res.get("Time", {}).get("millis"))})
            drivers.append(drv)
            constructors.append(con)
    return pd.DataFrame(rows, columns=list(SCHEMAS["sprint_results"])), drivers, constructors


def read_qualifying() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    rows, drivers, constructors = [], [], []
    for race in _races_from("qualifying"):
        rid = _race_id(race)
        for res in race.get("QualifyingResults", []):
            rows.append({"race_id": rid, "driver_id": res["Driver"]["driverId"],
                         "constructor_id": res["Constructor"]["constructorId"],
                         "position": _int(res.get("position")), "q1": res.get("Q1"),
                         "q2": res.get("Q2"), "q3": res.get("Q3")})
            drivers.append(res["Driver"])
            constructors.append(res["Constructor"])
    return pd.DataFrame(rows, columns=list(SCHEMAS["qualifying"])), drivers, constructors


def read_pit_stops() -> pd.DataFrame:
    rows = []
    for race in _races_from("pitstops"):
        for stop in race.get("PitStops", []):
            rows.append({"race_id": _race_id(race), "season": race["season"], "round": race["round"],
                         "driver_id": stop.get("driverId"), "stop": _int(stop.get("stop")),
                         "lap": _int(stop.get("lap")), "local_time": stop.get("time"),
                         "duration_s": parse_time_str(stop.get("duration"))})
    return pd.DataFrame(rows, columns=list(SCHEMAS["pit_stops"]))


def build_people(driver_dicts: list[dict], constructor_dicts: list[dict]):
    drivers = pd.DataFrame([{
        "driver_id": d["driverId"], "code": d.get("code"), "given_name": d.get("givenName"),
        "family_name": d.get("familyName"), "birth_date": d.get("dateOfBirth"),
        "nationality": d.get("nationality")} for d in driver_dicts],
        columns=list(SCHEMAS["drivers"])).drop_duplicates("driver_id", keep="last")
    constructors = pd.DataFrame([{
        "constructor_id": c["constructorId"], "name": c.get("name"),
        "nationality": c.get("nationality")} for c in constructor_dicts],
        columns=list(SCHEMAS["constructors"])).drop_duplicates("constructor_id", keep="last")
    return drivers, constructors


def read_fastf1(suffix: str) -> pd.DataFrame:
    folder = config.fastf1_raw_dir()
    frames = []
    for path in sorted(folder.glob(f"*_{suffix}.parquet")) if folder.exists() else []:
        year, rnd = path.name.split("_")[:2]
        df = pd.read_parquet(path)
        df["race_id"] = int(year) * 100 + int(rnd)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------- driver mapping

def build_driver_map(results: pd.DataFrame, ff1_results: pd.DataFrame):
    """Link FastF1 driver abbreviations to Jolpica driver ids, race by race.

    First match on the three-letter code; anything left over is matched on car
    number. Whatever is still unmatched is returned so it can be logged.
    Returns (per_race_map, season_map, unmatched).
    """
    empty = pd.DataFrame(columns=["race_id", "abbreviation", "driver_id"])
    if ff1_results.empty or "abbreviation" not in ff1_results.columns:
        return empty, pd.DataFrame(columns=["season", "abbreviation", "driver_id"]), \
            pd.DataFrame(columns=["race_id", "abbreviation", "driver_number"])

    ff1 = coerce(ff1_results.assign(race_id=ff1_results["race_id"]), {
        "race_id": "BIGINT", "abbreviation": "VARCHAR", "driver_number": "BIGINT"})
    res = coerce(results, {"race_id": "BIGINT", "driver_id": "VARCHAR", "driver_code": "VARCHAR",
                           "car_number": "BIGINT"})

    by_code = ff1.merge(res[["race_id", "driver_code", "driver_id"]],
                        left_on=["race_id", "abbreviation"], right_on=["race_id", "driver_code"], how="left")
    matched = by_code[by_code["driver_id"].notna()][["race_id", "abbreviation", "driver_id"]]

    left = by_code[by_code["driver_id"].isna()][["race_id", "abbreviation", "driver_number"]]
    by_number = left.merge(res[["race_id", "car_number", "driver_id"]],
                           left_on=["race_id", "driver_number"], right_on=["race_id", "car_number"], how="left")
    matched_number = by_number[by_number["driver_id"].notna()][["race_id", "abbreviation", "driver_id"]]
    unmatched = by_number[by_number["driver_id"].isna()][["race_id", "abbreviation", "driver_number"]]

    per_race = pd.concat([matched, matched_number], ignore_index=True).drop_duplicates()
    season_map = per_race.assign(season=per_race["race_id"] // 100)[["season", "abbreviation", "driver_id"]]
    season_map = season_map.drop_duplicates().reset_index(drop=True)
    return per_race, season_map, unmatched


# -------------------------------------------------------------------- build

def build(log_fn=print) -> dict:
    """Rebuild the whole database from raw files. Returns row counts."""
    races, circuits = read_races()
    results, res_drivers, res_constructors = read_results()
    qualifying, q_drivers, q_constructors = read_qualifying()
    sprints, s_drivers, s_constructors = read_sprint_results()
    pit_stops = read_pit_stops()
    drivers, constructors = build_people(res_drivers + q_drivers + s_drivers,
                                         res_constructors + q_constructors + s_constructors)

    ff1_results = read_fastf1("results")
    ff1_laps = read_fastf1("laps")
    ff1_weather = read_fastf1("weather")
    ff1_practice = read_fastf1("practice")
    ff1_times = read_fastf1("laptimes")

    per_race, season_map, unmatched = build_driver_map(results, ff1_results)
    if not ff1_results.empty:
        ff1_results = ff1_results.merge(per_race, on=["race_id", "abbreviation"], how="left")
    if not ff1_laps.empty:
        ff1_laps = ff1_laps.merge(per_race, on=["race_id", "abbreviation"], how="left")
        if not ff1_times.empty and "lap_end_s" in ff1_times.columns:
            ff1_times = ff1_times.drop_duplicates(["race_id", "abbreviation", "lap_number"])
            ff1_laps = ff1_laps.merge(ff1_times[["race_id", "abbreviation", "lap_number", "lap_end_s"]],
                                      on=["race_id", "abbreviation", "lap_number"], how="left")
    if not ff1_practice.empty and "abbreviation" in ff1_practice.columns:
        # a reserve driver who only drove practice has no race result to link to: his laps keep a null driver_id
        ff1_practice = ff1_practice.merge(per_race, on=["race_id", "abbreviation"], how="left")

    config.data_dir().mkdir(parents=True, exist_ok=True)
    final = config.db_path()
    temp = final.with_name(final.name + ".building")
    temp.unlink(missing_ok=True)

    con = duckdb.connect(str(temp))
    try:
        for name, df in [("circuits", circuits), ("races", races), ("drivers", drivers),
                         ("constructors", constructors), ("results", results),
                         ("qualifying", qualifying), ("sprint_results", sprints), ("pit_stops", pit_stops),
                         ("fastf1_results", ff1_results), ("laps", ff1_laps), ("practice_laps", ff1_practice),
                         ("weather", ff1_weather), ("driver_map", season_map)]:
            write_table(con, name, df)

        unmatched_rows = pd.DataFrame({
            "kind": "driver_map_unmatched", "race_id": unmatched["race_id"],
            "detail": unmatched["abbreviation"].astype(str) + " (#" + unmatched["driver_number"].astype(str) + ")"})
        write_table(con, "mismatches", unmatched_rows)

        con.execute("""
            CREATE OR REPLACE TABLE stints AS
            SELECT race_id, driver_id, stint, mode(compound) AS compound,
                   min(lap_number) AS first_lap, max(lap_number) AS last_lap,
                   count(*) AS n_laps, min(tyre_life) AS start_tyre_life
            FROM laps
            WHERE stint IS NOT NULL AND driver_id IS NOT NULL
            GROUP BY race_id, driver_id, stint
        """)

        # Cross-source comparisons go to the mismatch log; nothing is overwritten.
        con.execute("""
            INSERT INTO mismatches
            SELECT 'finish_order', r.race_id,
                   r.driver_id || ': jolpica P' || r.position || ' vs fastf1 P' || f.position
            FROM results r
            JOIN fastf1_results f ON f.race_id = r.race_id AND f.driver_id = r.driver_id
            WHERE r.classified AND f.position IS NOT NULL AND r.position <> f.position
        """)
        con.execute("""
            INSERT INTO mismatches
            SELECT 'lap_count', l.race_id,
                   'fastf1 max lap ' || l.mx || ' vs jolpica ' || r.mx
            FROM (SELECT race_id, max(lap_number) AS mx FROM laps GROUP BY race_id) l
            JOIN (SELECT race_id, max(laps) AS mx FROM results GROUP BY race_id) r
              ON r.race_id = l.race_id
            WHERE l.mx <> r.mx
        """)

        info = pd.DataFrame([{
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "jolpica_files": sum(len(_json_files(k)) for k in ("races", "results", "qualifying", "sprint", "pitstops")),
            "fastf1_races": int(ff1_results["race_id"].nunique()) if not ff1_results.empty else 0}])
        write_table(con, "build_info", info)

        counts = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                  for t in list(SCHEMAS) + ["stints"]}
    finally:
        con.close()

    final.unlink(missing_ok=True)
    temp.replace(final)
    log_fn("Database built: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    return counts
