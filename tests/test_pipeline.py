import duckdb
import pandas as pd

from src import build_db, checks, config, fastf1_loader, jolpica
from tests import fakes


def build_fake(data_dir, with_fastf1=True):
    fakes.write_jolpica(data_dir)
    if with_fastf1:
        fakes.write_fastf1(data_dir)
    return build_db.build(log_fn=lambda *_: None)


def failing_errors(con):
    return [c for c in checks.run_checks(con) if not c.ok and c.level == "error"]


# ------------------------------------------------------------------ build

def test_build_creates_expected_tables(data_dir):
    counts = build_fake(data_dir)
    assert counts["races"] == 7 + 22
    assert counts["results"] == (7 + 22) * 3
    assert counts["pit_stops"] == 4  # 2 rounds x 2 stops, 2021 only
    assert counts["sprint_results"] == 2 * 3  # two sprint weekends, three drivers each
    assert counts["laps"] == 2 * 3 * 5
    assert counts["stints"] == 2 * 3 * 2
    assert counts["driver_map"] == 3
    assert counts["practice_laps"] == 2 * 2 * 4 * 6        # two weekends, FP1 and FP2, three drivers plus a reserve


def test_laps_get_jolpica_driver_ids(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    ids = {r[0] for r in con.execute("SELECT DISTINCT driver_id FROM laps").fetchall()}
    assert ids == {"max_verstappen", "hamilton", "bottas"}
    assert con.execute("SELECT count(*) FROM laps WHERE driver_id IS NULL").fetchone()[0] == 0


def test_sprint_results_are_stored_with_their_own_grid_and_order(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    rows = con.execute("SELECT race_id, driver_id, grid, position, classified, points FROM sprint_results "
                       "WHERE race_id = 202105 ORDER BY position").fetchall()
    assert rows[0][:5] == (202105, "max_verstappen", 1, 1, True) and rows[0][5] == 3.0
    assert {r[0] for r in con.execute("SELECT DISTINCT race_id FROM sprint_results").fetchall()} == {202105, 202110}


def test_checks_catch_a_sprint_for_an_unknown_race_and_two_winners(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("UPDATE sprint_results SET race_id = 209901 WHERE race_id = 202110")
    assert "sprint_race_in_calendar" in {c.name for c in failing_errors(con)}
    con.execute("UPDATE sprint_results SET position = 1 WHERE race_id = 202105 AND position = 2")
    assert "sprint_has_winner" in {c.name for c in failing_errors(con)}


def test_missing_sprint_data_is_a_warning_not_an_error(data_dir):
    fakes.write_jolpica(data_dir, seasons=range(1950, 2023))
    for f in (data_dir / "raw" / "jolpica" / "sprint").glob("*.json"):
        f.write_text("[]")
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    results = {c.name: c for c in checks.run_checks(con)}
    assert not results["sprint_data_loaded"].ok and results["sprint_data_loaded"].level == "warn"
    assert "sprint_data_loaded" not in {c.name for c in failing_errors(con)}


def test_pit_stop_durations_parsed(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    durations = sorted(r[0] for r in con.execute("SELECT DISTINCT duration_s FROM pit_stops").fetchall())
    assert durations == [22.4, 62.5]


def test_build_without_fastf1_still_works(data_dir):
    counts = build_fake(data_dir, with_fastf1=False)
    assert counts["laps"] == 0 and counts["races"] == 29


def test_rebuild_is_repeatable(data_dir):
    first = build_fake(data_dir)
    second = build_db.build(log_fn=lambda *_: None)
    assert first == second


def test_driver_map_falls_back_to_car_number():
    results = pd.DataFrame([{"race_id": 202101, "driver_id": "hamilton", "driver_code": None, "car_number": 44}])
    ff1 = pd.DataFrame([{"race_id": 202101, "abbreviation": "HAM", "driver_number": 44}])
    per_race, season_map, unmatched = build_db.build_driver_map(results, ff1)
    assert per_race.iloc[0]["driver_id"] == "hamilton" and unmatched.empty


def test_unmatched_driver_is_reported():
    results = pd.DataFrame([{"race_id": 202101, "driver_id": "hamilton", "driver_code": "HAM", "car_number": 44}])
    ff1 = pd.DataFrame([{"race_id": 202101, "abbreviation": "XXX", "driver_number": 99}])
    _, _, unmatched = build_db.build_driver_map(results, ff1)
    assert list(unmatched["abbreviation"]) == ["XXX"]


# ----------------------------------------------------------------- checks

def test_checks_pass_on_clean_data(data_dir):
    fakes.write_jolpica(data_dir, seasons=range(1950, 2022))  # every season, no gaps
    fakes.write_fastf1(data_dir)
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    assert failing_errors(con) == []


def test_checks_catch_duplicate_result(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("INSERT INTO results SELECT * FROM results WHERE race_id = 202101 AND position = 1")
    assert "results_unique" in {c.name for c in failing_errors(con)}


def test_checks_catch_missing_round(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("DELETE FROM races WHERE race_id = 202105")
    assert "rounds_contiguous" in {c.name for c in failing_errors(con)}


def test_checks_catch_missing_season(data_dir):
    fakes.write_jolpica(data_dir, seasons=(2021,))
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    assert "seasons_complete" in {c.name for c in failing_errors(con)}


def test_checks_catch_wrong_known_fact(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("UPDATE results SET driver_id = 'hamilton' WHERE race_id = 202122 AND position = 1")
    assert "spot_2021_final_race_winner_verstappen" in {c.name for c in failing_errors(con)}


def test_checks_catch_impossible_position(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("UPDATE results SET grid = 99 WHERE race_id = 202101 AND position = 2")
    assert "grid_and_position_ranges" in {c.name for c in failing_errors(con)}


def test_source_disagreement_goes_to_mismatch_log(data_dir):
    fakes.write_jolpica(data_dir)
    fakes.write_fastf1(data_dir)
    path = data_dir / "raw" / "fastf1" / "2021_01_results.parquet"
    df = pd.read_parquet(path)
    df.loc[df.abbreviation == "HAM", "position"] = 3
    df.loc[df.abbreviation == "BOT", "position"] = 2
    df.to_parquet(path, index=False)
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()))
    assert con.execute("SELECT count(*) FROM mismatches WHERE kind = 'finish_order'").fetchone()[0] == 2
    warns = {c.name for c in checks.run_checks(con) if not c.ok and c.level == "warn"}
    assert "sources_agree_finish_order" in warns


# ------------------------------------------------------------- downloading

class FakeClient:
    """Stands in for the API: counts requests and answers from a dict."""

    def __init__(self, answers):
        self.answers = answers
        self.requests_made = 0

    def get(self, path, params=None, **_):
        self.requests_made += 1
        races = self.answers.get(path, [])
        total = sum(len(r.get("Results", r.get("QualifyingResults", r.get("PitStops", [1])))) for r in races) \
            if "pitstops" in path or "results" in path or "qualifying" in path else len(races)
        return {"MRData": {"total": str(total), "RaceTable": {"Races": races}}}


def race(season, rnd, **extra):
    return {"season": str(season), "round": str(rnd), "date": f"{season}-03-0{rnd}", **extra}


def make_answers(season):
    races = [race(season, 1), race(season, 2)]
    return {
        f"{season}.json": races,
        f"{season}/results.json": [race(season, 1, Results=[{"x": 1}]), race(season, 2, Results=[{"x": 1}])],
        f"{season}/qualifying.json": [],
        f"{season}/1/pitstops.json": [race(season, 1, PitStops=[{"x": 1}])],
        f"{season}/2/pitstops.json": [race(season, 2, PitStops=[{"x": 1}])],
    }


def test_finished_season_is_not_downloaded_twice(data_dir):
    client = FakeClient(make_answers(2021))
    jolpica.download_season(client, 2021, current=2026, today="2026-10-07")
    first = client.requests_made
    assert first > 0
    jolpica.download_season(client, 2021, current=2026, today="2026-10-07")
    assert client.requests_made == first


def test_sprint_file_is_requested_from_2021_and_downloaded_once(data_dir):
    answers = make_answers(2021)
    answers["2021/sprint.json"] = [race(2021, 1, SprintResults=[{"x": 1}])]
    client = FakeClient(answers)
    jolpica.download_season(client, 2021, current=2026, today="2026-10-07")
    saved = data_dir / "raw" / "jolpica" / "sprint" / "2021.json"
    assert saved.exists()
    first = client.requests_made
    jolpica.download_season(client, 2021, current=2026, today="2026-10-07")
    assert client.requests_made == first
    client2 = FakeClient(make_answers(2019))
    jolpica.download_season(client2, 2019, current=2026, today="2026-10-07")
    assert not (data_dir / "raw" / "jolpica" / "sprint" / "2019.json").exists()


def test_a_season_downloaded_before_sprints_existed_only_fetches_the_sprint_file(data_dir):
    client = FakeClient(make_answers(2022))
    jolpica.download_season(client, 2022, current=2026, today="2026-10-07")
    (data_dir / "raw" / "jolpica" / "sprint" / "2022.json").unlink()
    before = client.requests_made
    jolpica.download_season(client, 2022, current=2026, today="2026-10-07")
    assert client.requests_made == before + 1


def test_current_season_is_refreshed(data_dir):
    client = FakeClient(make_answers(2026))
    jolpica.download_season(client, 2026, current=2026, today="2026-10-07")
    first = client.requests_made
    jolpica.download_season(client, 2026, current=2026, today="2026-10-07")
    assert client.requests_made > first


def test_future_races_get_no_pit_stop_request(data_dir):
    client = FakeClient(make_answers(2026))
    jolpica.download_season(client, 2026, current=2026, today="2026-02-01")  # both races are later
    assert not (data_dir / "raw" / "jolpica" / "pitstops").exists()


def test_merge_races_joins_pages():
    page1 = [{"season": "2021", "round": "1", "Results": [{"n": 1}, {"n": 2}]}]
    page2 = [{"season": "2021", "round": "1", "Results": [{"n": 3}]},
             {"season": "2021", "round": "2", "Results": [{"n": 1}]}]
    merged = jolpica.merge_races(page1, page2)
    assert [len(r["Results"]) for r in merged] == [3, 1]


def test_parse_time_str():
    assert build_db.parse_time_str("22.456") == 22.456
    assert build_db.parse_time_str("1:02.300") == 62.3
    assert build_db.parse_time_str("") is None
    assert build_db.parse_time_str("n/a") is None


def test_normalise_laps_from_fastf1_style_frame():
    raw = pd.DataFrame({
        "Driver": ["VER", "HAM"], "DriverNumber": ["33", "44"], "LapNumber": [1.0, 2.0],
        "LapTime": pd.to_timedelta([80.5, None], unit="s"), "Stint": [1.0, 1.0],
        "Compound": ["SOFT", "MEDIUM"], "TyreLife": [1.0, 2.0], "FreshTyre": [True, False],
        "PitInTime": pd.to_timedelta([None, 3000.0], unit="s"),
        "PitOutTime": pd.to_timedelta([None, None], unit="s"),
        "TrackStatus": ["1", "12"], "Position": [1.0, 2.0], "IsAccurate": [True, False]})
    out = fastf1_loader.normalise_laps(raw)
    assert list(out["lap_time_s"].iloc[:1]) == [80.5] and pd.isna(out["lap_time_s"].iloc[1])
    assert out["pit_in_s"].iloc[1] == 3000.0
    assert list(out["abbreviation"]) == ["VER", "HAM"]
    assert out["driver_number"].tolist() == [33, 44]


def test_normalise_tolerates_missing_columns():
    out = fastf1_loader.normalise_laps(pd.DataFrame({"Driver": ["VER"], "LapNumber": [1]}))
    assert pd.isna(out["compound"].iloc[0]) and out["lap_number"].iloc[0] == 1
    assert list(fastf1_loader.normalise_weather(pd.DataFrame()).columns)[0] == "time_s"


def test_failed_error_check_gives_nonzero_exit_code(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("DELETE FROM races WHERE race_id = 202105")
    assert checks.save_and_print(con, checks.run_checks(con)) == 1
    stored = con.execute("SELECT count(*) FROM data_checks WHERE NOT ok AND level = 'error'").fetchone()[0]
    assert stored >= 1


def test_warnings_alone_give_zero_exit_code(data_dir):
    fakes.write_jolpica(data_dir, seasons=range(1950, 2022))
    build_db.build(log_fn=lambda *_: None)  # no FastF1 files: only warnings expected
    con = duckdb.connect(str(config.db_path()))
    assert checks.save_and_print(con, checks.run_checks(con)) == 0


# ------------------------------------------------------- FastF1 rate limit

class RateLimitExceededError(Exception):
    pass


def test_rate_limit_is_waited_out_then_succeeds():
    calls, naps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitExceededError("any API: 500 calls/h")
        return "ok"

    result = fastf1_loader.call_with_rate_limit_wait(flaky, "test", sleep=naps.append, wait_s=300)
    assert result == "ok" and naps == [300, 300]


def test_rate_limit_gives_up_after_max_waits():
    import pytest

    def always():
        raise RateLimitExceededError("any API: 500 calls/h")

    naps = []
    with pytest.raises(RateLimitExceededError):
        fastf1_loader.call_with_rate_limit_wait(always, "test", sleep=naps.append, max_waits=3)
    assert len(naps) == 3


def test_other_errors_are_not_retried():
    import pytest
    naps = []

    def broken():
        raise ValueError("bad data")

    with pytest.raises(ValueError):
        fastf1_loader.call_with_rate_limit_wait(broken, "test", sleep=naps.append)
    assert naps == []


# ------------------------------------------------------- shared drives (pre-1960)

def test_duplicate_driver_result_before_1960_is_only_a_warning(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("INSERT INTO results SELECT * FROM results WHERE race_id = 195001 AND position = 1")
    results = {c.name: c for c in checks.run_checks(con)}
    assert results["results_unique"].ok
    assert not results["results_unique_early"].ok and results["results_unique_early"].level == "warn"


def test_repeated_position_check_names_the_drivers(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()))
    con.execute("UPDATE results SET position = 1 WHERE race_id = 202101 AND position = 2")
    detail = {c.name: c for c in checks.run_checks(con)}["finish_positions_unique"].detail
    assert "202101" in detail and "max_verstappen" in detail and "hamilton" in detail


def test_practice_laps_link_to_drivers_and_a_reserve_keeps_a_null_id(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    ids = {r[0] for r in con.execute("SELECT DISTINCT driver_id FROM practice_laps WHERE abbreviation <> 'RES'").fetchall()}
    assert ids == {"max_verstappen", "hamilton", "bottas"}
    assert con.execute("SELECT count(*) FROM practice_laps WHERE abbreviation = 'RES' AND driver_id IS NOT NULL").fetchone()[0] == 0
    assert {r[0] for r in con.execute("SELECT DISTINCT session FROM practice_laps").fetchall()} == {"FP1", "FP2"}
    assert not [c for c in checks.run_checks(con) if c.name.startswith("practice_laps") and not c.ok]


def test_build_without_practice_files_gives_an_empty_table_and_a_coverage_warning(data_dir):
    fakes.write_jolpica(data_dir)
    fakes.write_fastf1(data_dir, practice=False)
    counts = build_db.build(log_fn=lambda *_: None)
    assert counts["practice_laps"] == 0
    con = duckdb.connect(str(config.db_path()), read_only=True)
    res = {c.name: c for c in checks.run_checks(con)}
    assert res["practice_laps_sane"].ok and res["practice_laps_unique"].ok
    assert not res["practice_coverage"].ok and res["practice_coverage"].level == "warn"


def test_normalise_practice_adds_session_and_session_time():
    raw = pd.DataFrame({"Driver": ["VER", "HAM"], "DriverNumber": ["33", "44"], "LapNumber": [1.0, 2.0],
                        "LapTime": pd.to_timedelta([90.0, 91.0], unit="s"), "Time": pd.to_timedelta([1200.0, 1291.0], unit="s")})
    out = fastf1_loader.normalise_practice(raw, "FP2")
    assert list(out["session"]) == ["FP2", "FP2"] and list(out["session_time_s"]) == [1200.0, 1291.0]
    assert out.columns[0] == "session"


def test_practice_sessions_come_from_the_schedule_row():
    full = {"Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3", "Session4": "Qualifying", "Session5": "Race"}
    sprint = {"Session1": "Practice 1", "Session2": "Qualifying", "Session3": "Sprint Shootout", "Session4": "Sprint", "Session5": "Race"}
    assert fastf1_loader.practice_sessions_of(full) == ["FP1", "FP2", "FP3"]
    assert fastf1_loader.practice_sessions_of(sprint) == ["FP1"]
    assert fastf1_loader.practice_sessions_of(full, ("FP2",)) == ["FP2"]


def test_line_crossing_times_are_merged_onto_the_laps(data_dir):
    build_fake(data_dir)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    assert con.execute("SELECT count(*) FROM laps WHERE lap_end_s IS NULL").fetchone()[0] == 0
    gap = con.execute("SELECT max(lap_end_s) - min(lap_end_s) FROM laps WHERE lap_number = 3 AND race_id = 202101").fetchone()[0]
    assert abs(gap - 0.6) < 1e-9
    res = {c.name: c for c in checks.run_checks(con)}
    assert res["lap_end_times_coverage"].ok


def test_missing_line_crossing_files_leave_the_column_empty_with_a_warning(data_dir):
    fakes.write_jolpica(data_dir)
    fakes.write_fastf1(data_dir)
    for p in (data_dir / "raw" / "fastf1").glob("*_laptimes.parquet"):
        p.unlink()
    build_db.build(log_fn=lambda *_: None)
    con = duckdb.connect(str(config.db_path()), read_only=True)
    assert con.execute("SELECT count(lap_end_s) FROM laps").fetchone()[0] == 0
    res = {c.name: c for c in checks.run_checks(con)}
    assert not res["lap_end_times_coverage"].ok and res["lap_end_times_coverage"].level == "warn"


def test_normalise_lap_times_reads_the_session_clock():
    raw = pd.DataFrame({"Driver": ["VER", "HAM"], "LapNumber": [3.0, 3.0], "Time": pd.to_timedelta([300.5, 301.2], unit="s")})
    out = fastf1_loader.normalise_lap_times(raw)
    assert list(out["lap_end_s"]) == [300.5, 301.2] and list(out["lap_number"]) == [3, 3]
