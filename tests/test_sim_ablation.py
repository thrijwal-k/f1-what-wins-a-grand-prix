import numpy as np
import pandas as pd
import pytest

from src.sim import ablation as A
from src.sim import calibrate as C
from . import lapworld


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("a")
    w = lapworld.world(seed=8, seasons=(2022, 2023, 2024), races=6, n_laps=34, n_drivers=12, sc_races=(3,), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=12)
    truths, par = C.load_truths(d / "cal")
    return truths, par, d


def test_combo_names():
    assert A.combo_name((False, False, False)) == "pace only" and A.combo_name((True, True, True)) == "everything known"
    assert A.combo_name((False, True, False)) == "pace + safety cars"


def test_ablation_runs_all_eight_combinations_and_knowing_more_helps(setup):
    truths, par, d = setup
    races = A.ablate(truths, par, S=40, progress=False)
    assert races["combo"].nunique() == 8
    s = A.summarise(races)
    a = s[s["split"] == "all"].set_index("combo")
    assert a.loc["everything known", "mean_rank_error"] < a.loc["pace only", "mean_rank_error"]


def test_shapley_gains_add_up_to_the_total_improvement(setup):
    truths, par, d = setup
    s = A.summarise(A.ablate(truths, par, S=30, progress=False))
    sh = A.shapley(s, "mean_rank_error")
    g = sh[sh["split"] == "all"]["gain"].sum()
    a = s[s["split"] == "all"].set_index("combo")["mean_rank_error"]
    assert g == pytest.approx(a["pace only"] - a["everything known"])


def test_run_writes_files(setup, tmp_path):
    truths, par, d = setup
    lines = A.run(tmp_path / "o", cal_dir=d / "cal", S=30, progress=False)
    names = {p.name for p in (tmp_path / "o").iterdir()}
    assert {"ablation_races.csv", "ablation_summary.csv", "ablation_shapley.csv", "ablation_ladder.png", "findings.txt"} <= names
    assert any("SHARE OF THE PLACES" in l for l in lines)
