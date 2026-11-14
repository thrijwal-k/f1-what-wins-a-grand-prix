import math

import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import diagnose as D
from src.sim.engine import Params
from . import lapworld


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("dg")
    w = lapworld.world(seed=9, seasons=(2022, 2023, 2024), races=6, n_laps=34, n_drivers=12, sc_races=(3,), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=12)
    truths, par = C.load_truths(d / "cal")
    return truths, par, d, con


def test_shapley_splits_the_whole_change_exactly():
    vals = {(a, b, c): 10.0 * a + 3.0 * b + 1.0 * c + 2.0 * a * b for a in (False, True) for b in (False, True) for c in (False, True)}
    sh = D.shapley_of(vals)
    assert sum(sh.values()) == pytest.approx(vals[(True, True, True)] - vals[(False, False, False)])
    assert sh["strategies"] == pytest.approx(11.0) and sh["safety cars"] == pytest.approx(4.0) and sh["retirements"] == pytest.approx(1.0)


def test_scaled_leaves_negative_inherit_flag_and_scales_circuit_thresholds():
    p = Params(sigma_pit=0.4, sigma_pit_top=-1.0, thr=0.6, circuit_thr={"x": 1.0})
    q = D.scaled(p, ("sigma_pit", "sigma_pit_top"), 0.5)
    assert q.sigma_pit == pytest.approx(0.2) and q.sigma_pit_top == -1.0
    r = D.scaled(p, ("thr",), 2.0)
    assert r.thr == pytest.approx(1.2) and r.circuit_thr["x"] == pytest.approx(2.0)


def test_lever_reach_interpolates_and_flags_unreachable():
    base = {"lever": "fitted settings", "fields": "", "multiplier": 1.0, "races": 10, "pole_win": 0.40, "real_pole_win": 0.55, "early_gap": 0.0,
            "late_gap": 0.0, "abs_gap": 0.02, "grid_finish_gap": 0.0, "passes_ratio": 1.0, "close_ratio": 1.0, "lap1_ratio": 1.0}
    rows = [base] + [{**base, "lever": "a", "multiplier": m, "pole_win": 0.40 + (1.0 - m) * 0.3, "passes_ratio": 1.0 - (1.0 - m) * 0.2} for m in (0.5, 0.75)] \
        + [{**base, "lever": "b", "multiplier": m, "pole_win": 0.40 + (1.0 - m) * 0.01} for m in (0.5, 0.75)]
    r = D.lever_reach(pd.DataFrame(rows)).set_index("lever")
    assert bool(r.loc["a", "reached"]) and r.loc["a", "multiplier_needed"] == pytest.approx(0.5)
    assert r.loc["a", "passes_ratio"] == pytest.approx(0.9)
    assert not bool(r.loc["b", "reached"])


def test_pole_by_margin_buckets_and_both_combos():
    rows = pd.DataFrame({"combo": ["everything known"] * 4 + ["pace only"] * 4, "margin": [-0.5, 0.1, 0.3, 0.9] * 2,
                         "pole_won": [False, True, True, True] * 2, "pole_win": [0.2, 0.4, 0.6, 0.8] * 2})
    t = D.pole_by_margin(rows)
    assert set(t["combo"]) == {"everything known", "pace only"} and (t["races"] == 1).all() and len(t) == 8
    top = t[(t["combo"] == "everything known") & (t["pace_margin"] == D.MARGIN_LABELS[-1])].iloc[0]
    assert top["real_pole_win"] == 1.0 and top["simulated_pole_win"] == pytest.approx(0.8)


def test_diagnose_runs_end_to_end_and_writes_everything(setup):
    truths, par, d, con = setup
    out = d / "diag"
    lines = D.run(con, out, d / "cal", S=20, S_lever=8, events=3, progress=False)
    for f in ("factor_races", "factor_summary", "factor_effects", "lever_sweep", "lever_reach", "pole_by_pace_margin", "stops_by_season",
              "lead_loss_by_combo"):
        assert (out / f"{f}.csv").exists(), f
    assert (out / "findings.txt").exists() and (out / "pole_by_unknown.png").exists()
    assert any(l.startswith("FACTORS (all") for l in lines) and any(l.startswith("LEVERS at L0") for l in lines)
    s = pd.read_csv(out / "factor_summary.csv")
    a = s[s["split"] == "all"]
    assert len(a) == 8 and a["pole_win"].between(0, 1).all() and a["races"].nunique() == 1
    sw = pd.read_csv(out / "lever_sweep.csv")
    assert (sw["lever"] == "fitted settings").sum() == 1 and sw["lever"].nunique() == 1 + len(D.LEVERS)
    # taking the lap noise away must make the pole sitter win at least as often
    quiet = sw[(sw["lever"] == "lap noise, whole field") & (sw["multiplier"] == 0.5)]["pole_win"].iloc[0]
    base = sw[sw["lever"] == "fitted settings"]["pole_win"].iloc[0]
    assert quiet >= base - 0.03
    e = pd.read_csv(out / "factor_effects.csv")
    g = e[(e["split"] == "all") & (e["metric"] == "pole_win")]
    assert g["shapley"].sum() == pytest.approx(g["all_known"].iloc[0] - g["none_known"].iloc[0])


def test_diagnose_without_database_skips_pole_loss_only(setup):
    truths, par, d, con = setup
    out = d / "diag2"
    lines = D.run(None, out, d / "cal", S=12, S_lever=6, progress=False)
    assert (out / "findings.txt").exists()
    assert pd.read_csv(out / "stops_by_season.csv").shape[0] >= 1
    assert not any(l.startswith("POLE KEPT") for l in lines)


def test_stops_by_season_has_real_and_sampled(setup):
    truths, par, d, con = setup
    t = D.stops_by_season(truths, S=6)
    assert {"real_stops_per_car", "sampled_stops_per_car", "real_never_stop", "sampled_never_stop"} <= set(t.columns)
    assert t[["real_stops_per_car", "sampled_stops_per_car"]].notna().all().all()
