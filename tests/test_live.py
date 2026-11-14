import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import forecast as F
from src.sim import live as L
from . import lapworld

NOW = datetime(2030, 3, 1, 12, 0, tzinfo=timezone.utc)
FUTURE_ID = 999999


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("live")
    w = lapworld.world(seed=6, seasons=(2022, 2023, 2024, 2025), races=7, n_laps=36, n_drivers=14, sc_races=(3, 4), wet_race=7)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=15)
    truths, par = C.load_truths(d / "cal")
    con.execute("ALTER TABLE races ADD COLUMN race_name VARCHAR")
    con.execute("ALTER TABLE races ADD COLUMN race_date DATE")
    con.execute("ALTER TABLE races ADD COLUMN race_time VARCHAR")
    con.execute("UPDATE races SET race_name = 'GP ' || race_id, race_date = DATE '2025-01-01', race_time = '13:00:00Z'")
    last = truths[-1]
    con.execute("INSERT INTO races VALUES (?, 2026, 1, ?, 'Future GP', DATE '2030-03-03', '14:00:00Z')", [FUTURE_ID, last.circuit_id])
    rows = [(FUTURE_ID, dr, co, i + 1, f"1:30.{i:03d}") for i, (dr, co) in enumerate(zip(last.drivers, last.constructors))]
    con.execute("CREATE TABLE qualifying (race_id BIGINT, driver_id VARCHAR, constructor_id VARCHAR, position INTEGER, q1 VARCHAR)")
    con.executemany("INSERT INTO qualifying VALUES (?, ?, ?, ?, ?)", rows)
    return con, truths, d


def make(setup, folder, now=NOW, **kw):
    con, truths, d = setup
    return L.make_forecast(con, FUTURE_ID, sims=40, history_sims=20, now=now, folder=folder, cal_dir=d / "cal", truths=truths,
                           progress=False, min_history=8, **kw)


@pytest.fixture
def mini(monkeypatch):
    monkeypatch.setattr(F, "MIN_RECAL_CARS", 30)
    monkeypatch.setattr(F, "MIN_HISTORY", 8)


def test_next_race_finds_the_race_without_results(setup):
    con, _, _ = setup
    n = L.next_race(con, NOW)
    assert n["race_id"] == FUTURE_ID and n["ready"] and n["qualifying_rows"] == 14


def test_forecast_is_written_and_probabilities_are_sensible(setup, tmp_path, mini):
    path = make(setup, tmp_path / "f")
    rec = json.loads(path.read_text())
    assert rec["prev_sha256"] == L.GENESIS and rec["race_id"] == FUTURE_ID
    for m in L.LOGGED:
        assert sum(d["models"][m]["p_win"] for d in rec["drivers"]) == pytest.approx(1.0, abs=0.02)
        assert 0 <= min(d["models"][m]["p_top10"] for d in rec["drivers"]) <= 1
    assert L.verify(tmp_path / "f") == []


def test_chain_links_forecasts_and_verify_catches_edits(setup, tmp_path, mini):
    folder = tmp_path / "f"
    a = make(setup, folder)
    b = make(setup, folder, now=datetime(2030, 3, 2, 12, 0, tzinfo=timezone.utc))
    rb = json.loads(b.read_text())
    assert rb["prev_sha256"] == json.loads(a.read_text())["sha256"]
    assert L.verify(folder) == []
    rec = json.loads(a.read_text())
    rec["drivers"][0]["models"][L.PRIMARY]["p_win"] = 0.99
    a.write_text(json.dumps(rec))
    assert any("edited" in p for p in L.verify(folder))
    with pytest.raises(L.ForecastError):
        make(setup, folder, now=datetime(2030, 3, 2, 13, 0, tzinfo=timezone.utc))


def test_verify_catches_a_removed_forecast(setup, tmp_path, mini):
    folder = tmp_path / "f"
    a = make(setup, folder)
    make(setup, folder, now=datetime(2030, 3, 2, 12, 0, tzinfo=timezone.utc))
    a.unlink()
    assert L.verify(folder)


def test_refuses_after_the_start_and_after_results(setup, tmp_path, mini):
    con, truths, _ = setup
    with pytest.raises(L.ForecastError, match="passed"):
        make(setup, tmp_path / "f", now=datetime(2030, 3, 3, 15, 0, tzinfo=timezone.utc))
    with pytest.raises(L.ForecastError, match="already has results"):
        L.make_forecast(con, truths[-1].race_id, now=NOW, folder=tmp_path / "g", truths=truths, progress=False)


def test_refuses_without_qualifying(setup, tmp_path, mini):
    con, _, _ = setup
    con.execute("DELETE FROM qualifying WHERE race_id = ?", [FUTURE_ID])
    try:
        with pytest.raises(L.ForecastError, match="Qualifying"):
            make(setup, tmp_path / "f")
    finally:
        last = setup[1][-1]
        con.executemany("INSERT INTO qualifying VALUES (?, ?, ?, ?, ?)",
                        [(FUTURE_ID, dr, co, i + 1, f"1:30.{i:03d}") for i, (dr, co) in enumerate(zip(last.drivers, last.constructors))])


def test_scoring_uses_the_last_forecast_before_the_start(setup, tmp_path, mini):
    con, truths, _ = setup
    folder = tmp_path / "f"
    make(setup, folder)
    rec = json.loads(next(folder.glob("*.json")).read_text())
    last = truths[-1]
    # give the future race a result: the first driver on the grid wins
    for i, dr in enumerate(rec["drivers"]):
        con.execute("INSERT INTO results (race_id, season, round, driver_id, position) VALUES (?, 2026, 1, ?, ?)",
                    [FUTURE_ID, dr["driver_id"], i + 1])
    try:
        races, summ = L.score(con, folder)
        assert set(races["model"]) == set(L.LOGGED) and (races["race_id"] == FUTURE_ID).all()
        assert (races["winner"] == rec["drivers"][0]["driver_id"]).all()
        assert summ.loc[summ["model"] == L.BASELINE, "podium_skill"].iloc[0] == pytest.approx(0.0)
        assert "\n".join(L.write_scores(con, folder)).startswith("Scored 1")
        assert (folder / "scores.csv").exists()
    finally:
        con.execute("DELETE FROM results WHERE race_id = ?", [FUTURE_ID])


def test_a_forecast_made_after_the_start_is_never_scored(setup, tmp_path, mini):
    con, _, _ = setup
    folder = tmp_path / "f"
    path = make(setup, folder)
    rec = json.loads(path.read_text())
    body = {k: v for k, v in rec.items() if k != "sha256"}
    body["created_utc"] = "2030-03-04T00:00:00"
    path.write_text(json.dumps({**body, "sha256": L.sha256_of(body)}))
    con.execute("INSERT INTO results (race_id, season, round, driver_id, position) VALUES (?, 2026, 1, ?, 1)",
                [FUTURE_ID, rec["drivers"][0]["driver_id"]])
    try:
        assert L.scored_forecasts(con, folder) == []
    finally:
        con.execute("DELETE FROM results WHERE race_id = ?", [FUTURE_ID])


def test_code_fingerprint_changes_with_the_settings(tmp_path):
    (tmp_path / "params.json").write_text("{}")
    a = L.code_fingerprint(tmp_path)
    (tmp_path / "params.json").write_text('{"x":1}')
    assert a != L.code_fingerprint(tmp_path)
