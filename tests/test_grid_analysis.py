import duckdb
import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis import grid


def synthetic(seed=7):
    """Races where finishing order follows grid order, tightly at 'monaco', loosely at 'monza'."""
    rng = np.random.default_rng(seed)
    noise = {"monaco": 0.8, "monza": 7.0, "spa": 3.0}
    rows = []
    race_no = 0
    for season in range(1950, 2025):
        for rnd, circuit in enumerate(["monaco", "monza", "spa"], start=1):
            race_no += 1
            race_id = season * 100 + rnd
            slots = np.arange(1, 21)
            order = np.argsort(slots + rng.normal(0, noise[circuit], 20))
            for finish, idx in enumerate(order, start=1):
                rows.append({"race_id": race_id, "season": season, "round": rnd, "circuit_id": circuit,
                             "circuit_name": circuit.title(), "driver_id": f"d{idx}",
                             "grid": int(slots[idx]), "position": finish})
    return pd.DataFrame(rows)


def test_wilson_interval():
    low, high = grid.wilson(50, 100)
    assert 0.40 < low < 0.41 and 0.59 < high < 0.60
    assert all(np.isnan(v) for v in grid.wilson(0, 0))
    low, high = grid.wilson(0, 10)
    assert low == pytest.approx(0, abs=1e-9) and 0.25 < high < 0.35


def test_era_of():
    assert grid.era_of(1950) == "1950-69" and grid.era_of(2021) == "2010-21" and grid.era_of(2026) == "2022-now"


def test_prepare_counts_shared_drives_once_and_handles_pit_lane():
    raw = pd.DataFrame({
        "race_id": [195001] * 10, "season": 1950, "round": 1, "circuit_id": "x", "circuit_name": "X",
        "driver_id": [f"d{i}" for i in range(10)],
        "grid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        "position": [1, 2, 3, 3, 5, 6, 7, 8, 9, 10]})  # positions 3 shared
    df = grid.prepare(raw)
    assert len(df) == 9
    assert df.loc[df["driver_id"] == "d9", "grid_slot"].iloc[0] == 10  # pit lane = behind the last slot


def test_small_races_are_dropped():
    raw = pd.DataFrame({"race_id": [1] * 5, "season": 2000, "round": 1, "circuit_id": "x", "circuit_name": "X",
                        "driver_id": list("abcde"), "grid": [1, 2, 3, 4, 5], "position": [1, 2, 3, 4, 5]})
    assert grid.prepare(raw).empty


def test_race_summary_and_circuit_contrast():
    df = grid.prepare(synthetic())
    races = grid.race_summary(df)
    assert len(races) == 75 * 3
    assert races["spearman"].between(-1, 1).all()
    circuits = grid.circuit_summary(races, since=2000, min_races=8).set_index("circuit_id")
    assert circuits.loc["monaco", "mean_spearman"] > circuits.loc["monza", "mean_spearman"] + 0.3
    assert circuits.loc["monaco", "pole_win_rate"] > circuits.loc["monza", "pole_win_rate"]
    row = circuits.loc["monaco"]
    assert row["pole_ci_low"] <= row["pole_win_rate"] <= row["pole_ci_high"]


def test_grid_slot_table_is_monotone_for_a_predictable_field():
    df = grid.prepare(synthetic())
    slots = grid.by_grid_slot(df)
    d = slots[slots["era"] == "2010-21"].sort_values("slot")
    assert d["slot"].corr(d["mean_finish"]) > 0.95
    assert d["mean_finish"].iloc[0] < d["mean_finish"].iloc[-1] - 8
    assert d["win_rate"].iloc[0] > d["win_rate"].iloc[-1]


def test_era_and_season_summaries():
    races = grid.race_summary(grid.prepare(synthetic()))
    eras = grid.era_summary(races)
    assert list(eras["era"].astype(str)) == grid.ERA_LABELS
    seasons = grid.season_summary(races)
    assert len(seasons) == 75 and seasons["spearman_rolling5"].notna().sum() > 60


def test_full_run_writes_tables_charts_and_findings(data_dir, tmp_path):
    df = synthetic()
    con = duckdb.connect(str(tmp_path / "fake.duckdb"))
    results = df[["race_id", "season", "round", "driver_id", "grid", "position"]]
    races = df.drop_duplicates("race_id")[["race_id", "season", "round", "circuit_id"]]
    circuits = df.drop_duplicates("circuit_id")[["circuit_id", "circuit_name"]]
    con.register("r_df", results); con.register("ra_df", races); con.register("c_df", circuits)
    con.execute("CREATE TABLE results AS SELECT * FROM r_df")
    con.execute("CREATE TABLE races AS SELECT * FROM ra_df")
    con.execute("CREATE TABLE circuits AS SELECT * FROM c_df")
    out = tmp_path / "out"
    lines = grid.run(con, out)
    for name in ["race_summary.csv", "grid_slot_by_era.csv", "era_summary.csv", "season_summary.csv",
                 "circuit_summary.csv", "findings.txt", "expected_finish_by_grid.png",
                 "grid_grip_by_season.png", "pole_win_rate_by_era.png", "pole_win_rate_by_circuit.png"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert any("Monaco" in l for l in lines) and any("Monza" in l for l in lines)
