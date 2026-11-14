import duckdb
import numpy as np
import pandas as pd

from src import build_db, checks, config, derive
from tests import fakes


def built(data_dir, with_fastf1=True):
    fakes.write_jolpica(data_dir)
    if with_fastf1:
        fakes.write_fastf1(data_dir)
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    derive.build(con, log_fn=lambda *_: None)
    return con


def test_one_row_per_driver_per_race_and_per_team_per_race(data_dir):
    con = built(data_dir)
    n_results = con.execute("SELECT count(*) FROM (SELECT DISTINCT race_id, driver_id FROM results)").fetchone()[0]
    assert con.execute("SELECT count(*) FROM driver_race").fetchone()[0] == n_results
    assert con.execute("SELECT count(*) - count(DISTINCT (race_id, driver_id)) FROM driver_race").fetchone()[0] == 0
    assert con.execute("SELECT count(*) - count(DISTINCT (race_id, constructor_id)) FROM team_race").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM race_summary").fetchone()[0] == 7 + 22


def test_points_add_up_from_results_to_team_and_race(data_dir):
    con = built(data_dir)
    src = con.execute("SELECT sum(points) FROM results WHERE season >= 1965").fetchone()[0]
    dr = con.execute("SELECT sum(points) FROM driver_race WHERE season >= 1965").fetchone()[0]
    tr = con.execute("SELECT sum(points) FROM team_race WHERE season >= 1965").fetchone()[0]
    assert src == dr == tr


def test_result_flags_and_places_gained(data_dir):
    con = built(data_dir)
    row = con.execute("SELECT won, podium, points_finish, places_gained, started, retired, status_class, grid_pit_lane "
                      "FROM driver_race WHERE race_id = 202101 AND driver_id = 'max_verstappen'").fetchone()
    assert row == (True, True, True, 0, True, False, "finished", False)
    other = con.execute("SELECT won, podium FROM driver_race WHERE race_id = 202101 AND driver_id = 'bottas'").fetchone()
    assert other == (False, True)


def test_qualifying_sprint_and_pit_columns_are_null_when_unknown(data_dir):
    con = built(data_dir)
    q = con.execute("SELECT quali_position, quali_stage, quali_best_s, quali_gap_pct FROM driver_race "
                    "WHERE race_id = 202101 AND driver_id = 'max_verstappen'").fetchone()
    assert q[0] == 1 and q[1] == "Q3" and abs(q[2] - 79.5) < 1e-9 and q[3] == 0
    assert con.execute("SELECT quali_position FROM driver_race WHERE race_id = 202101 AND driver_id = 'bottas'"
                       ).fetchone()[0] is None
    sp = con.execute("SELECT sprint_position, sprint_points FROM driver_race WHERE race_id = 202105 "
                     "AND driver_id = 'hamilton'").fetchone()
    assert sp == (2, 2.0)
    assert con.execute("SELECT sprint_position FROM driver_race WHERE race_id = 202101 AND driver_id = 'hamilton'"
                       ).fetchone()[0] is None
    # no pit-stop data in 1950, so stops are unknown, not zero
    assert con.execute("SELECT count(*) FROM driver_race WHERE season = 1950 AND n_stops IS NOT NULL").fetchone()[0] == 0


def test_a_race_with_pit_data_gives_every_starter_a_stop_count(data_dir):
    con = built(data_dir)
    rows = con.execute("SELECT driver_id, n_stops FROM driver_race WHERE race_id = 202101 ORDER BY driver_id").fetchall()
    assert all(n is not None for _, n in rows)
    stops = con.execute("SELECT count(*) FROM pit_stops WHERE race_id = 202101").fetchone()[0]
    assert sum(n for _, n in rows) == stops
    assert con.execute("SELECT count(*) FROM driver_race WHERE race_id = 202105 AND n_stops IS NULL").fetchone()[0] == 3


def test_lap_columns_exist_only_where_laps_exist(data_dir):
    con = built(data_dir)
    led = con.execute("SELECT driver_id, laps_led, stints, tyre_sequence FROM driver_race WHERE race_id = 202101 "
                      "ORDER BY driver_id").fetchall()
    assert all(r[1] == 5 and r[2] == 2 and r[3] == "MEDIUM>HARD" for r in led)    # fake laps: everyone "leads" every lap
    assert con.execute("SELECT count(*) FROM driver_race WHERE race_id = 202103 AND stints IS NOT NULL").fetchone()[0] == 0
    rs = con.execute("SELECT has_laps, wet FROM race_summary WHERE race_id IN (202101, 202103) ORDER BY race_id").fetchall()
    assert rs[0][0] is True and rs[1] == (False, None)


def test_race_pace_is_relative_to_the_field_and_lower_is_faster(data_dir):
    con = built(data_dir, with_fastf1=False)
    rng = np.random.default_rng(0)
    rows = []
    for d in range(10):
        for lap in range(1, 31):
            rows.append({"race_id": 202101, "driver_id": f"d{d}", "lap_number": lap,
                         "lap_time_s": 90.0 * (1 + 0.002 * d) + rng.normal(0, 0.01), "stint": 1, "compound": "MEDIUM",
                         "tyre_life": float(lap), "pit_in_s": None, "pit_out_s": None, "track_status": "1",
                         "position": d + 1, "is_accurate": True})
    con.register("_l", pd.DataFrame(rows))
    con.execute("INSERT INTO laps (race_id, driver_id, lap_number, lap_time_s, stint, compound, tyre_life, pit_in_s, "
                "pit_out_s, track_status, position, is_accurate) SELECT race_id, driver_id, lap_number, lap_time_s, "
                "stint, compound, tyre_life, pit_in_s, pit_out_s, track_status, position, is_accurate FROM _l")
    pace, flags = derive.lap_rows(con)
    p = pace.set_index("driver_id")["race_pace"]
    assert p["d0"] < p["d9"] and p.is_monotonic_increasing
    assert abs((p["d9"] - p["d0"]) - np.log(1.018)) < 0.003
    assert set(pace["pace_laps"].dropna()) == {29}                 # lap 1 is not a clean lap
    assert bool(flags.loc[flags["race_id"] == 202101, "wet"].iloc[0]) is False


def test_status_classes_and_retirements(data_dir):
    con = built(data_dir)
    con.execute("UPDATE results SET status = 'Engine', classified = false, position = 20 "
                "WHERE race_id = 202101 AND driver_id = 'bottas'")
    con.execute("UPDATE results SET status = 'Collision', classified = false WHERE race_id = 202101 AND driver_id = 'hamilton'")
    derive.build(con, log_fn=lambda *_: None)
    got = dict(con.execute("SELECT driver_id, status_class FROM driver_race WHERE race_id = 202101").fetchall())
    assert got["bottas"] == "mechanical" and got["hamilton"] == "incident" and got["max_verstappen"] == "finished"
    t = con.execute("SELECT n_retired, n_mechanical, n_incident, n_classified FROM team_race WHERE race_id = 202101"
                    ).fetchone()
    assert t == (2, 1, 1, 1)
    r = con.execute("SELECT n_retired, n_classified FROM race_summary WHERE race_id = 202101").fetchone()
    assert r == (2, 1)


def test_shared_drives_collapse_to_the_best_placed_row(data_dir):
    con = built(data_dir)
    con.execute("INSERT INTO results SELECT * FROM results WHERE race_id = 195001 AND driver_id = 'farina'")
    con.execute("UPDATE results SET position = 9, points = 0 WHERE rowid = (SELECT max(rowid) FROM results "
                "WHERE race_id = 195001 AND driver_id = 'farina')")
    derive.build(con, log_fn=lambda *_: None)
    rows = con.execute("SELECT position, shared_drive FROM driver_race WHERE race_id = 195001 AND driver_id = 'farina'"
                       ).fetchall()
    assert rows == [(1, True)]


def test_pole_and_winner_in_race_summary(data_dir):
    con = built(data_dir)
    row = con.execute("SELECT winner_id, grid_pole_id, quali_pole_id, pole_won, n_starters FROM race_summary "
                      "WHERE race_id = 202101").fetchone()
    assert row == ("max_verstappen", "max_verstappen", "max_verstappen", True, 3)
    assert con.execute("SELECT quali_pole_id FROM race_summary WHERE race_id = 195001").fetchone()[0] is None


def test_rebuilding_gives_the_same_tables(data_dir):
    con = built(data_dir)
    a = con.execute("SELECT * FROM driver_race ORDER BY race_id, driver_id").df()
    derive.build(con, log_fn=lambda *_: None)
    b = con.execute("SELECT * FROM driver_race ORDER BY race_id, driver_id").df()
    pd.testing.assert_frame_equal(a, b)


def test_a_database_without_laps_or_qualifying_still_builds(data_dir):
    con = built(data_dir, with_fastf1=False)
    con.execute("DELETE FROM qualifying")
    counts = derive.build(con, log_fn=lambda *_: None)
    assert counts["driver_race"] > 0 and counts["race_summary"] == 7 + 22
    assert con.execute("SELECT count(*) FROM driver_race WHERE race_pace IS NOT NULL").fetchone()[0] == 0


def _errors(con):
    # the fake data has only two seasons, so the calendar check always fails on it
    return {c.name for c in checks.run_checks(con) if not c.ok and c.level == "error"} - {"seasons_complete"}


def test_checks_pass_on_a_good_build_and_catch_broken_derived_tables(data_dir):
    con = built(data_dir)
    assert not _errors(con)
    con.execute("UPDATE driver_race SET points = points + 1 WHERE race_id = 202101 AND driver_id = 'bottas'")
    assert "derived_points_match_results" in _errors(con)
    derive.build(con, log_fn=lambda *_: None)
    con.execute("DELETE FROM driver_race WHERE race_id = 202101 AND driver_id = 'bottas'")
    assert "driver_race_matches_results" in _errors(con)
    derive.build(con, log_fn=lambda *_: None)
    con.execute("INSERT INTO team_race SELECT * FROM team_race WHERE race_id = 202101")
    assert "team_race_unique" in _errors(con)


def test_missing_derived_tables_are_a_warning_not_an_error(data_dir):
    fakes.write_jolpica(data_dir)
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    found = {c.name: c for c in checks.run_checks(con)}
    assert found["derived_tables_built"].level == "warn" and not found["derived_tables_built"].ok
    assert not _errors(con)


def test_a_withdrawn_driver_who_scored_points_still_counts_as_a_starter(data_dir):
    con = built(data_dir)
    con.execute("UPDATE results SET status = 'Withdrew', classified = false WHERE race_id = 202101 "
                "AND driver_id = 'bottas'")                       # still has the points the fake race gave him
    derive.build(con, log_fn=lambda *_: None)
    row = con.execute("SELECT status_class, started FROM driver_race WHERE race_id = 202101 AND driver_id = 'bottas'"
                      ).fetchone()
    assert row == ("not_started", True)
    assert con.execute("SELECT n_cars FROM team_race WHERE race_id = 202101").fetchone()[0] == 3
    assert not _errors(con)
