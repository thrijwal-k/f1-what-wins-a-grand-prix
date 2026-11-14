"""Derived tables: one row per driver per race, per team per race, and per race.

The base tables (results, qualifying, pit_stops, laps ...) mirror the sources. The analysis keeps needing the
same joined view of them, and each analysis module used to rebuild it. These tables build it once, from the
definitions the analysis already uses (status classes from race_events, clean-lap race pace from race_pace,
wet and neutralised races from tyres), so a dashboard, the forecast log and the analysis all read the same rows.

    race_summary   one row per race with results: starters, winner, pole sitter, safety-car laps, wet flag
    driver_race    one row per driver per race: grid, result, status class, qualifying, sprint, pit stops, race pace
    team_race      one row per constructor per race: the same, summed or taken over the team's cars

Rules that matter when reading them
  - A null is "not known", never "zero". Pace and tyre columns are null before 2018 (no lap data), pit-stop
    columns before 2011, qualifying columns where the qualifying session is missing. `n_stops` is 0 only when
    the race has pit-stop data and the driver made none.
  - `wet` in race_summary is null when the race has no lap data; it does not mean dry.
  - Before 1965 a car could be shared by several drivers. Those rows are collapsed to the best-placed one
    (`shared_drive` marks them), so points before 1965 are not the sum of the source rows.
  - `started` is true for any car with a classified result or points, even if the source status says
    "Withdrew" or "Did not start"; `status_class` still shows the source's wording.
  - `race_pace` is the median of ln(lap time / field median on the same lap) over clean laps, so lower is
    faster and 0.01 is about 1% of a lap. See race_pace.py for the limits.
  - `pole` means the first place on the starting grid (grid == 1), after penalties; `quali_position` is the
    qualifying result itself.

Run:  python -m src.derive        (also run by python -m src.run_all after the build)
"""
from __future__ import annotations

import logging
import sys

import duckdb
import numpy as np
import pandas as pd

from . import build_db, config
from .analysis import race_events, race_pace, tyres

log = logging.getLogger(__name__)

RETIRED_CLASSES = ("mechanical", "incident", "unspecified")

RACE_SUMMARY = {
    "race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "circuit_id": "VARCHAR", "race_date": "DATE",
    "n_starters": "BIGINT", "n_classified": "BIGINT", "n_retired": "BIGINT",
    "winner_id": "VARCHAR", "grid_pole_id": "VARCHAR", "quali_pole_id": "VARCHAR", "pole_won": "BOOLEAN",
    "has_laps": "BOOLEAN", "sc_laps": "BIGINT", "vsc_laps": "BIGINT", "red_laps": "BIGINT",
    "neutral": "BOOLEAN", "wet": "BOOLEAN", "wet_borderline": "BOOLEAN", "rain_share": "DOUBLE",
}

DRIVER_RACE = {
    "race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "circuit_id": "VARCHAR", "race_date": "DATE",
    "driver_id": "VARCHAR", "driver_code": "VARCHAR", "constructor_id": "VARCHAR", "car_number": "BIGINT",
    "grid": "BIGINT", "grid_pit_lane": "BOOLEAN", "position": "BIGINT", "classified": "BOOLEAN",
    "status": "VARCHAR", "status_class": "VARCHAR", "finished": "BOOLEAN", "completed": "DOUBLE",
    "points": "DOUBLE", "laps": "BIGINT", "time_millis": "BIGINT", "fastest_lap_rank": "BIGINT",
    "started": "BOOLEAN", "retired": "BOOLEAN", "disqualified": "BOOLEAN",
    "won": "BOOLEAN", "podium": "BOOLEAN", "points_finish": "BOOLEAN", "places_gained": "BIGINT",
    "shared_drive": "BOOLEAN",
    "quali_position": "BIGINT", "quali_stage": "VARCHAR", "quali_best_s": "DOUBLE", "quali_gap_pct": "DOUBLE",
    "sprint_position": "BIGINT", "sprint_points": "DOUBLE",
    "n_stops": "BIGINT", "first_stop_lap": "BIGINT", "first_stop_frac": "DOUBLE", "pit_time_s": "DOUBLE",
    "race_pace": "DOUBLE", "pace_laps": "BIGINT", "laps_led": "BIGINT", "stints": "BIGINT",
    "tyre_sequence": "VARCHAR",
}

TEAM_RACE = {
    "race_id": "BIGINT", "season": "BIGINT", "round": "BIGINT", "constructor_id": "VARCHAR",
    "n_cars": "BIGINT", "grid_best": "BIGINT", "grid_mean": "DOUBLE", "best_position": "BIGINT",
    "points": "DOUBLE", "sprint_points": "DOUBLE", "n_classified": "BIGINT", "n_retired": "BIGINT",
    "n_mechanical": "BIGINT", "n_incident": "BIGINT", "wins": "BIGINT", "podiums": "BIGINT", "poles": "BIGINT",
    "quali_gap_pct_best": "DOUBLE", "quali_gap_pct_mean": "DOUBLE",
    "race_pace_best": "DOUBLE", "race_pace_mean": "DOUBLE", "pace_cars": "BIGINT",
    "stops_mean": "DOUBLE", "laps_led": "BIGINT",
}

SCHEMAS = {"race_summary": RACE_SUMMARY, "driver_race": DRIVER_RACE, "team_race": TEAM_RACE}


# ---------------------------------------------------------------- helpers

def _has(con, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]).fetchone()[0])


def _count(con, table: str) -> int:
    return int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]) if _has(con, table) else 0


def _write(con, name: str, df: pd.DataFrame) -> None:
    schema = SCHEMAS[name]
    clean = build_db.coerce(df, schema)
    con.register("_derived", clean)
    cols = ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in schema.items())
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT {cols} FROM _derived')
    con.unregister("_derived")


def _best_time(row) -> float:
    times = [build_db.parse_time_str(row[c]) for c in ("q1", "q2", "q3")]
    times = [t for t in times if t is not None]
    return min(times) if times else np.nan


def _stage(row) -> str | None:
    for col, label in (("q3", "Q3"), ("q2", "Q2"), ("q1", "Q1")):
        if build_db.parse_time_str(row[col]) is not None:
            return label
    return None


# ------------------------------------------------------------ building blocks

def result_rows(con) -> pd.DataFrame:
    """Results, one row per driver per race, with the status class. Shared drives collapse to the best-placed row."""
    res = con.execute("""
        SELECT r.race_id, r.season, r.round, ra.circuit_id, ra.race_date, r.driver_id, r.driver_code,
               r.constructor_id, r.car_number, r.grid, r.position, r.classified, r.points, r.laps, r.status,
               r.time_millis, r.fastest_lap_rank
        FROM results r JOIN races ra ON ra.race_id = r.race_id
    """).df()
    if res.empty:
        return res.assign(status_class=[], finished=[], completed=[], shared_drive=[])
    dup = res.duplicated(["race_id", "driver_id"], keep=False)
    res = res.assign(_pos=pd.to_numeric(res["position"], errors="coerce").fillna(1e9))
    res = res.sort_values(["race_id", "driver_id", "_pos"]).drop_duplicates(["race_id", "driver_id"], keep="first")
    res["shared_drive"] = dup.loc[res.index].to_numpy()
    res = race_events.add_status(res.drop(columns="_pos"))
    return res.reset_index(drop=True)


def qualifying_rows(con) -> pd.DataFrame:
    if not _has(con, "qualifying") or _count(con, "qualifying") == 0:
        return pd.DataFrame(columns=["race_id", "driver_id", "quali_position", "quali_stage", "quali_best_s",
                                     "quali_gap_pct"])
    q = con.execute("SELECT race_id, driver_id, position, q1, q2, q3 FROM qualifying").df()
    q = q.drop_duplicates(["race_id", "driver_id"], keep="first")
    q["quali_best_s"] = q.apply(_best_time, axis=1)
    q["quali_stage"] = q.apply(_stage, axis=1)
    pole = q.groupby("race_id")["quali_best_s"].transform("min")
    q["quali_gap_pct"] = 100.0 * (q["quali_best_s"] / pole - 1.0)
    return q.rename(columns={"position": "quali_position"})[
        ["race_id", "driver_id", "quali_position", "quali_stage", "quali_best_s", "quali_gap_pct"]]


def sprint_rows(con) -> pd.DataFrame:
    if not _has(con, "sprint_results") or _count(con, "sprint_results") == 0:
        return pd.DataFrame(columns=["race_id", "driver_id", "sprint_position", "sprint_points"])
    s = con.execute("SELECT race_id, driver_id, position AS sprint_position, points AS sprint_points "
                    "FROM sprint_results").df()
    return s.drop_duplicates(["race_id", "driver_id"], keep="first")


def pit_rows(con, results: pd.DataFrame) -> pd.DataFrame:
    """Stops per driver per race. A race with pit-stop data gives every starter a count (0 if none)."""
    cols = ["race_id", "driver_id", "n_stops", "first_stop_lap", "first_stop_frac", "pit_time_s"]
    if not _has(con, "pit_stops") or _count(con, "pit_stops") == 0:
        return pd.DataFrame(columns=cols)
    p = con.execute("SELECT race_id, driver_id, lap, duration_s FROM pit_stops").df()
    p["duration_s"] = p["duration_s"].clip(upper=race_events.PIT_CLIP_S)
    g = p.groupby(["race_id", "driver_id"]).agg(n_stops=("lap", "size"), first_stop_lap=("lap", "min"),
                                                pit_time_s=("duration_s", "sum")).reset_index()
    base = results[results["race_id"].isin(p["race_id"].unique())][["race_id", "driver_id", "laps"]]
    out = base.merge(g, on=["race_id", "driver_id"], how="left")
    out["n_stops"] = out["n_stops"].fillna(0)
    race_laps = out.groupby("race_id")["laps"].transform("max")
    out["first_stop_frac"] = out["first_stop_lap"] / race_laps
    return out[cols]


def lap_rows(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race pace, laps led and tyre sequence per driver per race, and the race-level flags (2018 on)."""
    pace_cols = ["race_id", "driver_id", "race_pace", "pace_laps", "laps_led", "stints", "tyre_sequence"]
    flag_cols = ["race_id", "neutral", "red", "sc_laps", "vsc_laps", "red_laps", "wet_laps", "rain_share",
                 "wet", "wet_borderline"]
    if not _has(con, "laps") or _count(con, "laps") == 0:
        return pd.DataFrame(columns=pace_cols), pd.DataFrame(columns=flag_cols)
    laps = con.execute("""
        SELECT race_id, driver_id, lap_number, lap_time_s, stint, compound, pit_in_s, pit_out_s,
               track_status, position, is_accurate
        FROM laps WHERE driver_id IS NOT NULL
    """).df()
    pace = race_pace.race_pace(laps).rename(columns={"pace": "race_pace", "laps": "pace_laps"})
    led = (laps[laps["position"] == 1].groupby(["race_id", "driver_id"]).size().rename("laps_led").reset_index())
    stint_df = laps[laps["stint"].notna()].groupby(["race_id", "driver_id", "stint"])["compound"].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else "UNKNOWN").reset_index()
    seq = (stint_df.sort_values("stint").groupby(["race_id", "driver_id"])["compound"]
           .agg(stints="size", tyre_sequence=lambda s: ">".join(s)).reset_index())
    keys = laps[["race_id", "driver_id"]].drop_duplicates()
    per_driver = keys.merge(pace, on=["race_id", "driver_id"], how="left") \
                     .merge(led, on=["race_id", "driver_id"], how="left") \
                     .merge(seq, on=["race_id", "driver_id"], how="left")
    per_driver["laps_led"] = per_driver["laps_led"].fillna(0)
    flags = tyres.race_flags(laps, tyres.load_rain_share(con))
    return per_driver[pace_cols], flags[flag_cols]


# ------------------------------------------------------------------ tables

def build_driver_race(results, quali, sprint, pits, laps_pd) -> pd.DataFrame:
    d = results.copy()
    pos = pd.to_numeric(d["position"], errors="coerce")
    grid = pd.to_numeric(d["grid"], errors="coerce")
    d["grid_pit_lane"] = grid == 0
    # A classified result or points prove the car started, whatever the status text says (Monaco 1996, Frentzen:
    # status "Withdrew" with 3 points). `status_class` is left as the source gives it.
    d["started"] = ((d["status_class"] != "not_started") | d["classified"].fillna(False).astype(bool)
                    | (pd.to_numeric(d["points"], errors="coerce") > 0))
    d["retired"] = d["status_class"].isin(RETIRED_CLASSES)
    d["disqualified"] = d["status_class"] == "disqualified"
    d["won"] = (pos == 1) & d["classified"].fillna(False).astype(bool)
    d["podium"] = (pos <= 3) & d["classified"].fillna(False).astype(bool)
    d["points_finish"] = pd.to_numeric(d["points"], errors="coerce") > 0
    ok = d["started"] & (grid > 0) & d["classified"].fillna(False).astype(bool)
    d["places_gained"] = (grid - pos).where(ok)
    for frame in (quali, sprint, pits, laps_pd):
        d = d.merge(frame, on=["race_id", "driver_id"], how="left")
    return d


def build_team_race(dr: pd.DataFrame) -> pd.DataFrame:
    s = dr[dr["started"] & ~dr["disqualified"]].copy()
    s["_grid"] = pd.to_numeric(s["grid"], errors="coerce").where(lambda x: x > 0)
    s["_pos"] = pd.to_numeric(s["position"], errors="coerce").where(s["classified"].fillna(False).astype(bool))
    s["_mech"] = s["status_class"].isin(["mechanical", "unspecified"])
    s["_inc"] = s["status_class"] == "incident"
    g = s.groupby(["race_id", "season", "round", "constructor_id"])
    out = g.agg(n_cars=("driver_id", "size"), grid_best=("_grid", "min"), grid_mean=("_grid", "mean"),
                best_position=("_pos", "min"), points=("points", "sum"), sprint_points=("sprint_points", "sum"),
                n_classified=("classified", "sum"), n_retired=("retired", "sum"), n_mechanical=("_mech", "sum"),
                n_incident=("_inc", "sum"), wins=("won", "sum"), podiums=("podium", "sum"),
                quali_gap_pct_best=("quali_gap_pct", "min"), quali_gap_pct_mean=("quali_gap_pct", "mean"),
                race_pace_best=("race_pace", "min"), race_pace_mean=("race_pace", "mean"),
                pace_cars=("race_pace", "count"), stops_mean=("n_stops", "mean"),
                laps_led=("laps_led", "sum")).reset_index()
    poles = s[s["_grid"] == 1].groupby(["race_id", "constructor_id"]).size().rename("poles").reset_index()
    out = out.merge(poles, on=["race_id", "constructor_id"], how="left")
    out["poles"] = out["poles"].fillna(0)
    return out


def build_race_summary(dr: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    if dr.empty:
        return pd.DataFrame(columns=list(RACE_SUMMARY))
    started = dr[dr["started"] & ~dr["disqualified"]]
    base = dr.drop_duplicates("race_id")[["race_id", "season", "round", "circuit_id", "race_date"]].copy()
    counts = started.groupby("race_id").agg(n_starters=("driver_id", "size"), n_classified=("classified", "sum"),
                                            n_retired=("retired", "sum")).reset_index()
    winner = dr[dr["won"]].drop_duplicates("race_id")[["race_id", "driver_id"]].rename(columns={"driver_id": "winner_id"})
    gp = dr[pd.to_numeric(dr["grid"], errors="coerce") == 1].drop_duplicates("race_id")[
        ["race_id", "driver_id"]].rename(columns={"driver_id": "grid_pole_id"})
    qp = dr[pd.to_numeric(dr["quali_position"], errors="coerce") == 1].drop_duplicates("race_id")[
        ["race_id", "driver_id"]].rename(columns={"driver_id": "quali_pole_id"})
    out = base.merge(counts, on="race_id", how="left").merge(winner, on="race_id", how="left") \
              .merge(gp, on="race_id", how="left").merge(qp, on="race_id", how="left")
    out["pole_won"] = [None if pd.isna(a) or pd.isna(b) else bool(a == b)
                       for a, b in zip(out["grid_pole_id"], out["winner_id"])]
    has = flags[["race_id", "neutral", "sc_laps", "vsc_laps", "red_laps", "wet", "wet_borderline", "rain_share"]]
    out = out.merge(has, on="race_id", how="left")
    out["has_laps"] = out["race_id"].isin(flags["race_id"])
    return out


def build(con=None, log_fn=print) -> dict[str, int]:
    """Build race_summary, driver_race and team_race inside the database. Returns the row counts."""
    own = con is None
    if own:
        con = duckdb.connect(str(config.db_path()))
    try:
        results = result_rows(con)
        quali, sprint = qualifying_rows(con), sprint_rows(con)
        pits = pit_rows(con, results)
        laps_pd, flags = lap_rows(con)
        dr = build_driver_race(results, quali, sprint, pits, laps_pd)
        tr = build_team_race(dr) if not dr.empty else pd.DataFrame(columns=list(TEAM_RACE))
        rs = build_race_summary(dr, flags)
        for name, frame in (("race_summary", rs), ("driver_race", dr), ("team_race", tr)):
            _write(con, name, frame)
        counts = {t: _count(con, t) for t in SCHEMAS}
    finally:
        if own:
            con.close()
    log_fn("Derived tables built: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


def main() -> int:
    path = config.db_path()
    if not path.exists():
        print(f"No database at {path}. Run the build first (python -m src.run_all --skip-jolpica --skip-fastf1).")
        return 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    con = duckdb.connect(str(path))
    try:
        build(con, log_fn=log.info)
        q = lambda sql: con.execute(sql).fetchone()
        log.info("driver_race: %s with race pace, %s with qualifying, %s with pit-stop counts, %s with lap data",
                 q("SELECT count(*) FROM driver_race WHERE race_pace IS NOT NULL")[0],
                 q("SELECT count(*) FROM driver_race WHERE quali_position IS NOT NULL")[0],
                 q("SELECT count(*) FROM driver_race WHERE n_stops IS NOT NULL")[0],
                 q("SELECT count(*) FROM driver_race WHERE stints IS NOT NULL")[0])
        log.info("race_summary: %s races with lap data, %s wet, pole won %s of %s (grid pole, 1965 on)",
                 q("SELECT count(*) FROM race_summary WHERE has_laps")[0],
                 q("SELECT count(*) FROM race_summary WHERE wet")[0],
                 q("SELECT count(*) FROM race_summary WHERE pole_won AND season >= 1965")[0],
                 q("SELECT count(*) FROM race_summary WHERE pole_won IS NOT NULL AND season >= 1965")[0])
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
