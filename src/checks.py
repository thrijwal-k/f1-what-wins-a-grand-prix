"""Data quality checks, run after every database build.

Each check has a level:
  error - the data is wrong or incomplete; the run exits with a failure code
  warn  - worth a look, but can be legitimate (old shared drives, odd sessions)
Results are printed and saved in the data_checks table.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb

from . import config


@dataclass
class Check:
    name: str
    level: str  # "error" or "warn"
    ok: bool
    detail: str = ""


def _scalar(con, sql: str) -> int:
    return con.execute(sql).fetchone()[0]


def _examples(con, sql: str, limit: int = 5) -> str:
    rows = con.execute(f"{sql} LIMIT {limit}").fetchall()
    return "; ".join(str(r) for r in rows)


def run_checks(con) -> list[Check]:
    checks: list[Check] = []

    def zero(name: str, level: str, count_sql: str, examples_sql: str | None = None, what: str = "rows"):
        n = _scalar(con, count_sql)
        detail = "" if n == 0 else f"{n} {what}"
        if n and examples_sql:
            detail += f" e.g. {_examples(con, examples_sql)}"
        checks.append(Check(name, level, n == 0, detail))

    # ---- structure of the race calendar
    zero("race_ids_unique", "error",
         "SELECT count(*) - count(DISTINCT race_id) FROM races", what="duplicate race ids")
    zero("rounds_contiguous", "error",
         "SELECT count(*) FROM (SELECT season, max(round) mx, count(DISTINCT round) c FROM races GROUP BY season) WHERE mx <> c",
         "SELECT season, max(round), count(DISTINCT round) FROM races GROUP BY season HAVING max(round) <> count(DISTINCT round)",
         what="seasons with missing rounds")
    seasons = con.execute("SELECT min(season), max(season), count(DISTINCT season) FROM races").fetchone()
    lo, hi, n = seasons
    if n:
        ok = lo == config.FIRST_SEASON and (hi - lo + 1) == n
        checks.append(Check("seasons_complete", "error", ok,
                            "" if ok else f"seasons {lo}-{hi}, {n} present (expected from {config.FIRST_SEASON}, no gaps)"))
    else:
        checks.append(Check("seasons_complete", "error", False, "no races loaded"))
    zero("race_dates_ascending", "error",
         "SELECT count(*) FROM races a JOIN races b ON a.season = b.season AND a.round + 1 = b.round WHERE a.race_date >= b.race_date",
         what="rounds out of date order")

    # ---- results
    # Shared drives (one car, several drivers) happened until 1964, so a driver
    # can legitimately appear twice in a race, or two drivers share a position,
    # before then.
    zero("results_unique", "error",
         "SELECT count(*) FROM (SELECT race_id, driver_id FROM results WHERE race_id >= 196500 GROUP BY 1, 2 HAVING count(*) > 1)",
         "SELECT race_id, driver_id FROM results WHERE race_id >= 196500 GROUP BY 1, 2 HAVING count(*) > 1",
         what="duplicated driver results (1965 onward)")
    zero("results_unique_early", "warn",
         "SELECT count(*) FROM (SELECT race_id, driver_id FROM results WHERE race_id < 196500 GROUP BY 1, 2 HAVING count(*) > 1)",
         what="driver results repeated before 1965 (shared drives)")
    zero("finish_positions_unique", "error",
         "SELECT count(*) FROM (SELECT race_id, position FROM results WHERE position IS NOT NULL AND season >= 1965 GROUP BY 1, 2 HAVING count(*) > 1)",
         "SELECT race_id, position, string_agg(driver_id, ' / ') FROM results WHERE position IS NOT NULL AND season >= 1965 GROUP BY 1, 2 HAVING count(*) > 1",
         what="repeated finishing positions (1965 onward)")
    zero("finish_positions_unique_early", "warn",
         "SELECT count(*) FROM (SELECT race_id, position FROM results WHERE position IS NOT NULL AND season < 1965 GROUP BY 1, 2 HAVING count(*) > 1)",
         what="repeated finishing positions before 1965 (shared drives)")
    zero("race_has_winner", "error",
         "SELECT count(*) FROM (SELECT race_id FROM results GROUP BY race_id HAVING sum(CASE WHEN position = 1 THEN 1 ELSE 0 END) = 0)",
         "SELECT race_id FROM results GROUP BY race_id HAVING sum(CASE WHEN position = 1 THEN 1 ELSE 0 END) = 0",
         what="races with no winner")
    zero("race_has_results", "error",
         "SELECT count(*) FROM races r WHERE r.race_date < current_date AND r.season < extract(year FROM current_date) AND NOT EXISTS (SELECT 1 FROM results x WHERE x.race_id = r.race_id)",
         "SELECT race_id FROM races r WHERE r.race_date < current_date AND r.season < extract(year FROM current_date) AND NOT EXISTS (SELECT 1 FROM results x WHERE x.race_id = r.race_id)",
         what="past races without results")
    zero("race_has_results_current_season", "warn",
         "SELECT count(*) FROM races r WHERE r.race_date < current_date AND r.season = extract(year FROM current_date) AND NOT EXISTS (SELECT 1 FROM results x WHERE x.race_id = r.race_id)",
         what="current-season races without results yet")
    zero("race_has_enough_starters", "error",
         "SELECT count(*) FROM (SELECT race_id FROM results GROUP BY race_id HAVING count(*) < 2)",
         what="races with fewer than 2 results")
    zero("grid_and_position_ranges", "error",
         "SELECT count(*) FROM results WHERE grid < 0 OR grid > 45 OR position < 1 OR position > 45",
         "SELECT race_id, driver_id, grid, position FROM results WHERE grid < 0 OR grid > 45 OR position < 1 OR position > 45",
         what="rows with an impossible grid or position")
    zero("points_not_negative", "error",
         "SELECT count(*) FROM results WHERE points < 0", what="rows with negative points")
    zero("winner_points_within_era", "warn",
         """SELECT count(*) FROM results WHERE position = 1 AND points >
            CASE WHEN season <= 1990 THEN 9 WHEN season <= 2009 THEN 10 WHEN season = 2014 THEN 50 ELSE 26 END""",
         """SELECT race_id, driver_id, points FROM results WHERE position = 1 AND points >
            CASE WHEN season <= 1990 THEN 9 WHEN season <= 2009 THEN 10 WHEN season = 2014 THEN 50 ELSE 26 END""",
         what="winners with more points than their era allowed")

    # ---- sprint races (2021 on)
    zero("sprint_results_unique", "error",
         "SELECT count(*) FROM (SELECT race_id, driver_id FROM sprint_results GROUP BY 1, 2 HAVING count(*) > 1)",
         what="duplicated sprint results")
    zero("sprint_race_in_calendar", "error",
         "SELECT count(DISTINCT s.race_id) FROM sprint_results s WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.race_id = s.race_id)",
         what="sprints for a race that is not on the calendar")
    zero("sprint_positions_valid", "error",
         "SELECT count(*) FROM sprint_results WHERE grid < 0 OR grid > 45 OR position < 1 OR position > 45",
         "SELECT race_id, driver_id, grid, position FROM sprint_results WHERE grid < 0 OR grid > 45 OR position < 1 OR position > 45",
         what="sprint rows with an impossible grid or position")
    zero("sprint_has_winner", "error",
         "SELECT count(*) FROM (SELECT race_id FROM sprint_results GROUP BY race_id HAVING sum(CASE WHEN position = 1 THEN 1 ELSE 0 END) <> 1)",
         what="sprints without exactly one winner")
    zero("sprints_per_season_plausible", "warn",
         "SELECT count(*) FROM (SELECT season FROM sprint_results GROUP BY season HAVING count(DISTINCT race_id) > 8)",
         what="seasons with more than 8 sprints")
    zero("sprint_data_loaded", "warn",
         "SELECT CASE WHEN (SELECT count(*) FROM sprint_results) = 0 AND (SELECT count(*) FROM races WHERE season >= 2022) > 0 THEN 1 ELSE 0 END",
         what="(no sprint results loaded: run python -m src.run_all --skip-fastf1 to download them)")

    # ---- pit stops
    zero("pit_stop_durations_plausible", "warn",
         "SELECT count(*) FROM pit_stops WHERE duration_s < 8 OR duration_s > 300",
         "SELECT race_id, driver_id, stop, duration_s FROM pit_stops WHERE duration_s < 8 OR duration_s > 300",
         what="pit stops outside 8 to 300 seconds")
    zero("pit_stop_laps_within_race", "warn",
         "SELECT count(*) FROM pit_stops p JOIN (SELECT race_id, max(laps) mx FROM results GROUP BY race_id) r USING (race_id) WHERE p.lap > r.mx",
         what="pit stops after the final lap")

    # ---- laps (FastF1)
    zero("laps_unique", "error",
         "SELECT count(*) FROM (SELECT race_id, abbreviation, lap_number FROM laps GROUP BY 1, 2, 3 HAVING count(*) > 1)",
         what="duplicated laps")
    zero("laps_sane", "error",
         "SELECT count(*) FROM laps WHERE lap_number < 1 OR lap_time_s <= 0",
         what="laps with a number below 1 or a non-positive time")
    zero("lap_end_times_coverage", "warn",
         "SELECT count(*) FROM (SELECT race_id FROM laps GROUP BY race_id HAVING count(lap_end_s) < 0.8 * count(*))",
         "SELECT race_id FROM laps GROUP BY race_id HAVING count(lap_end_s) < 0.8 * count(*)",
         what="races where under 80% of laps have a line-crossing time (run python -m src.run_all --skip-jolpica)")
    zero("practice_laps_sane", "error",
         "SELECT count(*) FROM practice_laps WHERE lap_number < 1 OR lap_time_s <= 0",
         what="practice laps with a number below 1 or a non-positive time")
    zero("practice_laps_unique", "error",
         "SELECT count(*) FROM (SELECT race_id, session, abbreviation, lap_number FROM practice_laps GROUP BY 1, 2, 3, 4 HAVING count(*) > 1)",
         what="duplicated practice laps")
    zero("practice_coverage", "warn",
         "SELECT count(*) FROM races r WHERE r.season >= 2018 AND r.race_date < current_date AND EXISTS (SELECT 1 FROM laps l WHERE l.race_id = r.race_id) AND NOT EXISTS (SELECT 1 FROM practice_laps p WHERE p.race_id = r.race_id)",
         "SELECT race_id FROM races r WHERE r.season >= 2018 AND r.race_date < current_date AND EXISTS (SELECT 1 FROM laps l WHERE l.race_id = r.race_id) AND NOT EXISTS (SELECT 1 FROM practice_laps p WHERE p.race_id = r.race_id)",
         what="races with FastF1 race laps but no practice laps (run python -m src.run_all --skip-jolpica)")
    zero("fastf1_coverage", "warn",
         "SELECT count(*) FROM races r WHERE r.season >= 2018 AND r.race_date < current_date AND NOT EXISTS (SELECT 1 FROM laps l WHERE l.race_id = r.race_id)",
         "SELECT race_id FROM races r WHERE r.season >= 2018 AND r.race_date < current_date AND NOT EXISTS (SELECT 1 FROM laps l WHERE l.race_id = r.race_id)",
         what="races since 2018 with no FastF1 laps")

    # ---- the two sources agree
    zero("driver_map_conflicts", "error",
         "SELECT count(*) FROM (SELECT season, abbreviation FROM driver_map GROUP BY 1, 2 HAVING count(DISTINCT driver_id) > 1)",
         what="abbreviations mapped to two drivers in one season")
    for kind, label in [("driver_map_unmatched", "FastF1 drivers with no Jolpica match"),
                        ("finish_order", "drivers with a different finishing position in the two sources"),
                        ("lap_count", "races where the two sources disagree on race length")]:
        zero(f"sources_agree_{kind}", "warn",
             f"SELECT count(*) FROM mismatches WHERE kind = '{kind}'",
             f"SELECT race_id, detail FROM mismatches WHERE kind = '{kind}'", what=label)

    # ---- spot checks against facts that never change
    def spot(name: str, season_sql: str, test_sql: str):
        if _scalar(con, season_sql) == 0:
            checks.append(Check(name, "error", True, "skipped: season not loaded"))
        else:
            checks.append(Check(name, "error", _scalar(con, test_sql) == 1, "" if _scalar(con, test_sql) == 1 else "known fact not matched"))

    spot("spot_1950_has_7_races", "SELECT count(*) FROM races WHERE season = 1950",
         "SELECT CAST(count(*) = 7 AS INT) FROM races WHERE season = 1950")
    spot("spot_1950_round1_winner_farina", "SELECT count(*) FROM results WHERE race_id = 195001",
         "SELECT CAST(count(*) > 0 AS INT) FROM results WHERE race_id = 195001 AND position = 1 AND driver_id = 'farina'")
    spot("spot_2021_has_22_races", "SELECT count(*) FROM races WHERE season = 2021",
         "SELECT CAST(count(*) = 22 AS INT) FROM races WHERE season = 2021")
    spot("spot_2021_final_race_winner_verstappen", "SELECT count(*) FROM results WHERE race_id = 202122",
         "SELECT CAST(count(*) > 0 AS INT) FROM results WHERE race_id = 202122 AND position = 1 AND driver_id = 'max_verstappen'")

    # ---- derived tables (src/derive.py): they must reconcile with the base tables they come from
    derived = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if {"driver_race", "team_race", "race_summary"} <= derived:
        zero("driver_race_unique", "error",
             "SELECT count(*) - count(DISTINCT (race_id, driver_id)) FROM driver_race", what="duplicated driver-races")
        zero("driver_race_matches_results", "error",
             "SELECT abs((SELECT count(*) FROM driver_race) - (SELECT count(*) FROM (SELECT DISTINCT race_id, driver_id FROM results)))",
             what="rows between driver_race and results")
        zero("team_race_unique", "error",
             "SELECT count(*) - count(DISTINCT (race_id, constructor_id)) FROM team_race", what="duplicated team-races")
        zero("race_summary_matches_races", "error",
             "SELECT abs((SELECT count(*) FROM race_summary) - (SELECT count(DISTINCT race_id) FROM results))",
             what="races between race_summary and results")
        zero("derived_points_match_results", "error",
             """SELECT CAST(abs((SELECT coalesce(sum(points), 0) FROM results WHERE season >= 1965)
                              - (SELECT coalesce(sum(points), 0) FROM driver_race WHERE season >= 1965)) > 1e-6 AS INT)
                       + CAST(abs((SELECT coalesce(sum(points), 0) FROM driver_race WHERE season >= 1965 AND NOT disqualified)
                              - (SELECT coalesce(sum(points), 0) FROM team_race WHERE season >= 1965)) > 1e-6 AS INT)""",
             what="point totals that differ between results, driver_race and team_race (1965 onward)")
        zero("race_pace_plausible", "warn",
             "SELECT count(*) FROM driver_race WHERE race_pace IS NOT NULL AND abs(race_pace) > 0.10",
             "SELECT race_id, driver_id, race_pace FROM driver_race WHERE race_pace IS NOT NULL AND abs(race_pace) > 0.10",
             what="driver-races with a race pace more than 10% from the field")
    else:
        checks.append(Check("derived_tables_built", "warn", False,
                            "driver_race, team_race or race_summary missing: run python -m src.derive"))
    return checks


def save_and_print(con, checks: list[Check]) -> int:
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("CREATE OR REPLACE TABLE data_checks (check_name VARCHAR, level VARCHAR, ok BOOLEAN, detail VARCHAR, run_at VARCHAR)")
    con.executemany("INSERT INTO data_checks VALUES (?, ?, ?, ?, ?)",
                    [(c.name, c.level, c.ok, c.detail, run_at) for c in checks])
    errors = [c for c in checks if not c.ok and c.level == "error"]
    warns = [c for c in checks if not c.ok and c.level == "warn"]
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.level == "error" else "warn")
        print(f"[{mark}] {c.name}" + (f" - {c.detail}" if c.detail else ""))
    print(f"\n{len(checks)} checks: {len(errors)} errors, {len(warns)} warnings")
    return 1 if errors else 0


def main() -> int:
    path = config.db_path()
    if not path.exists():
        print(f"No database at {path}. Run the build first.")
        return 1
    con = duckdb.connect(str(path))
    try:
        return save_and_print(con, run_checks(con))
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
