import pandas as pd
import pytest

from src.analysis import grid, grid_checks as gc, sprint_check as sc
from tests.test_grid_checks import make_con, world

NOISE = {c: 3.0 for c in "abcdef"}


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    rows, quali = world(seasons=range(2018, 2024), noise=NOISE, swap=0.5)
    return make_con(tmp_path_factory.mktemp("s") / "f.duckdb", rows, quali)


@pytest.fixture(scope="module")
def parts(con):
    df = gc.prepare_checks(gc.load(con), drop_indy=True)
    return df, grid.race_summary(df)


IDS = {202102: "test race A", 202104: "test race B", 209999: "not in the data"}


def test_default_list_is_the_six_2021_22_sprint_weekends():
    assert len(sc.SPRINT_GRID_RACES) == 6
    seasons = sorted(r // 100 for r in sc.SPRINT_GRID_RACES)
    assert seasons == [2021, 2021, 2021, 2022, 2022, 2022]


def test_missing_races_are_skipped_and_fields_filled(parts):
    df, races = parts
    t = sc.sprint_races(df, races, IDS)
    assert list(t["race_id"]) == [202102, 202104]
    assert t["winner"].notna().all() and t["spearman"].between(-1, 1).all()
    assert (t["grid_pole_won"] == (t["grid_pole"] == t["winner"])).all()


def test_sensitivity_removes_exactly_the_listed_races(parts):
    _, races = parts
    s = sc.sensitivity(races, IDS)
    full = s[(s["scope"] == "2021 and 2022") & (s["version"] == "all races")].iloc[0]
    cut = s[(s["scope"] == "2021 and 2022") & (s["version"] != "all races")].iloc[0]
    assert full["races"] == 12 and cut["races"] == 10
    expected = races[(races["season"].between(2021, 2022)) & ~races["race_id"].isin(IDS)]
    assert cut["pole_win_rate"] == pytest.approx(expected[expected["has_pole"]]["pole_won"].mean())


def test_run_writes_files_and_findings(con, tmp_path):
    lines = sc.run(con, tmp_path, ids=IDS)
    for name in ["sprint_grid_races.csv", "sprint_sensitivity.csv", "findings.txt"]:
        assert (tmp_path / name).stat().st_size > 0
    text = "\n".join(lines)
    assert "SCOPE" in text and "SPRINT-GRID RACES: 2" in text and "DOES EXCLUDING THEM" in text


def test_ids_come_from_sprint_table_and_agree_note(tmp_path):
    rows, quali = world(seasons=range(2021, 2024), noise=NOISE, swap=0.5)
    con = make_con(tmp_path / "d.duckdb", rows, quali)
    assert sc.derive_ids(con) is None
    con.execute("CREATE TABLE sprint_results AS SELECT race_id, season FROM results WHERE race_id IN (202101, 202202, 202302)")
    ids = sc.derive_ids(con)
    assert set(ids) == {202101, 202202}          # 2023 sprint grids were normal qualifying
    lines = sc.run(con, tmp_path / "o")
    assert "DO NOT match" in lines[0]
    con.close()
