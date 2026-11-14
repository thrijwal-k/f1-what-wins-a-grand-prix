import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import engine as E
from src.sim import refit as R
from src.sim.engine import BIG, Params, make_spec, simulate
from . import lapworld


def _truth(rid=202201, season=2023, n=8, L=44, seed=0, stop_at=None, neutral=None, retire=None, par=None, pace=None):
    """A race made by the engine itself, so order, gaps and stops agree."""
    rng = np.random.default_rng(seed)
    pace = np.sort(rng.normal(0, 0.4, n))[rng.permutation(n)] if pace is None else np.asarray(pace, float)
    stops = np.array([stop_at[i] if stop_at is not None else 14 + 2 * (i % 4) for i in range(n)])
    strategies = [[(E.MEDIUM, 0, int(stops[i])), (E.HARD, 0)] for i in range(n)]
    nz = np.zeros(L + 1, int) if neutral is None else np.asarray(neutral, int)
    ret = np.full(n, BIG) if retire is None else np.asarray(retire, int)
    spec = make_spec(L, np.arange(1, n + 1), pace, strategies, neutral=nz, retire=ret, pit_loss=22.0)
    res = simulate(spec, par or Params(), S=1, seed=seed)
    pos = res.pos[0].T.astype(float)
    pos = np.where(pos > 0, pos, np.nan)
    gap = res.gap[0].T.astype(float)
    return C.RaceTruth(race_id=rid, season=season, circuit_id="x", n_laps=L, drivers=[f"d{i}" for i in range(n)],
                       constructors=[f"c{i // 2}" for i in range(n)], slot=np.arange(1, n + 1), pace=pace,
                       comp=np.array([[E.MEDIUM, E.HARD]] * n), stops=stops.reshape(n, 1), start_age=np.zeros((n, 2)), retire=ret,
                       neutral=nz, finish=res.finish[0], pos_actual=pos, off=np.array([-0.5, 0.0, 0.4]), deg=np.array([0.10, 0.06, 0.04]),
                       fuel=-0.056, pit_loss=22.0, pace_known=np.ones(n, bool), gap_ahead=gap)


# ------------------------------------------------------------------ gaps behind the leader

def test_behind_leader_adds_the_gaps_down_the_order_and_breaks_at_unknowns():
    pos = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, np.nan]])
    gap = np.array([[np.nan, 2.0, 3.0, 4.0], [1.5, np.nan, 2.5, np.nan]])
    bl = R.behind_leader(pos, gap)
    assert bl[0].tolist() == [0.0, 2.0, 5.0, 9.0]
    assert bl[1][1] == 0.0 and bl[1][0] == 1.5 and bl[1][2] == 4.0 and np.isnan(bl[1][3])
    gap2 = gap.copy(); gap2[0, 2] = np.nan
    b2 = R.behind_leader(pos, gap2)
    assert b2[0][1] == 2.0 and np.isnan(b2[0][2]) and np.isnan(b2[0][3])         # one unknown breaks everything behind it


def test_behind_leader_of_the_engine_matches_its_own_gaps():
    t = _truth(seed=2)
    bl = R.real_bl(t)
    lap = 20
    order = np.argsort(t.pos_actual[lap - 1])
    d = np.diff(bl[lap - 1][order])
    assert np.allclose(d, t.gap_ahead[lap - 1][order][1:], atol=1e-9)


def test_wander_windows_skip_stops_neutral_laps_and_retirements():
    t = _truth(seed=1, stop_at=[10] * 8)               # everybody stops on lap 10
    win = R.wander_windows(t)
    assert win is not None
    # no window overlaps lap 10 (+- 1) for either car
    for k in np.unique(win["k"]):
        assert not (k - 1 <= 10 <= k + R.W + 1)
    t2 = _truth(seed=1, stop_at=[30] * 8, neutral=[0] + [0] * 20 + [2] * 3 + [0] * 21)
    w2 = R.wander_windows(t2)
    for k in np.unique(w2["k"]):
        assert not (k - 1 <= 23 and k + R.W + 1 >= 21)
    t3 = _truth(seed=1, stop_at=[40] * 8, retire=[BIG] * 7 + [25])
    w3 = R.wander_windows(t3)
    bad = (w3["a"] == 7) | (w3["b"] == 7)
    assert (w3["k"][bad] + R.W + 1 < 25).all()


def test_wander_windows_real_change_is_the_change_in_the_gap_between_the_two_cars():
    t = _truth(seed=3, stop_at=[40] * 8)
    win = R.wander_windows(t)
    rb = R.real_bl(t)
    i = 0
    a, b, k = win["a"][i], win["b"][i], win["k"][i]
    exp = (rb[k + R.W - 1, b] - rb[k + R.W - 1, a]) - (rb[k - 1, b] - rb[k - 1, a])
    assert win["real"][i] == pytest.approx(exp)
    res = simulate(t.spec(), Params(), S=3, seed=1)
    d = R.sim_wander(R.sim_bl(res), win)
    assert d.shape == (3, len(win["a"]))


def test_robust_sd_ignores_outliers():
    x = np.concatenate([np.random.default_rng(0).normal(0, 1, 2000), [500.0, -400.0]])
    assert 0.9 < R.robust_sd(x) < 1.1 and np.std(x) > 5


# ------------------------------------------------------------------ the closing rate

def test_closing_events_and_gain_have_the_right_sign():
    n = 6
    stops = [12, 20, 40, 41, 42, 43]                    # car 0 stops 8 laps before car 1; car 1 is directly behind it
    t = _truth(seed=4, n=n, stop_at=stops, L=50, pace=[0.0, 0.05, 0.3, 0.5, 0.7, 0.9], par=Params(sigma_start=0.0, sigma_lap=0.05, sigma_stint=0.0, drift_sd=0.0, slot_gap=0.3))
    ev = R.closing_events(t)
    assert ev is not None
    assert set(np.unique(ev["sign"])) <= {-1, 1}
    g = R.closing_gain(R.real_bl(t), ev)
    assert g.shape == (len(ev["a"]),)
    # a first stopper gains on the old tyres: positive on average in a world with wear and no warm-up
    assert np.nanmean(g) > 0


# ------------------------------------------------------------------ settings

def test_scaled_keeps_circuit_differences_and_scales_the_rest():
    par = Params(thr=1.0, drift_sd=0.4, sigma_lap=0.5, circuit_thr={"a": 0.6, "b": 1.4}, warm_amp=0.9)
    q = R.scaled(par, drift=0.5, noise=0.8, thr=0.5, warm=0.0)
    assert q.drift_sd == pytest.approx(0.2) and q.sigma_lap == pytest.approx(0.4) and q.thr == pytest.approx(0.5)
    assert q.circuit_thr["b"] / q.circuit_thr["a"] == pytest.approx(1.4 / 0.6)
    assert q.circuit_thr["a"] == pytest.approx(0.3) and q.warm_scale == 0.0 and q.warm_amp == 0.9
    assert R.scaled(par).thr == par.thr and R.scaled(par).warm_scale == par.warm_scale


def test_warm_scale_zero_switches_the_warm_up_off_and_one_changes_nothing():
    spec = make_spec(30, [1, 2], [0.0, 0.0], [[(E.MEDIUM, 0, 10), (E.HARD, 0)], [(E.MEDIUM, 0)]])
    base = Params(warm_amp=1.0, warm_decay=0.9)
    a = simulate(spec, base, S=20, seed=3)
    b = simulate(spec, replace(base, warm_scale=1.0), S=20, seed=3)
    off = simulate(spec, replace(base, warm_scale=0.0), S=20, seed=3)
    none = simulate(spec, Params(warm_amp=0.0), S=20, seed=3)
    assert (a.finish == b.finish).all() and (a.done == b.done).all()
    assert (off.finish == none.finish).all() and np.allclose(off.gap[~np.isnan(off.gap)], none.gap[~np.isnan(none.gap)])
    # with the race's own warm-up set, the scale still applies
    spec2 = replace(spec, warm=2.0)
    assert (simulate(spec2, replace(base, warm_scale=0.0), S=20, seed=3).finish == none.finish).all()


def test_match_threshold_finds_the_factor_that_gives_the_target_pass_rate():
    truths = [_truth(rid=202300 + i, seed=i, n=10, L=40) for i in range(6)]
    par = Params(thr=1.0, circuit_thr={"x": 1.0}, drift_sd=0.3, sigma_lap=0.5)
    target = R.pass_rate_of(truths, R.scaled(par, thr=0.6), S=12, seed=0)
    f = R.match_threshold(truths, par, target, S=12, seed=0, iters=8)
    got = R.pass_rate_of(truths, R.scaled(par, thr=f), S=12, seed=0)
    assert 0.45 < f < 0.8
    assert abs(got - target) / target < 0.12
    # a lower threshold means more passes
    assert R.pass_rate_of(truths, R.scaled(par, thr=0.3), 12, 0) > R.pass_rate_of(truths, R.scaled(par, thr=1.2), 12, 0)


def test_loss_prefers_the_wander_that_matches_and_penalises_a_loose_one():
    real = {"wander_robust_sd": 3.3, "close_share": 0.2, "pass_rate": 0.04}
    good = {"wander_robust_sd": 3.4, "close_share": 0.2, "pass_rate": 0.04, "grip_gap": 0.01}
    loose = {**good, "wander_robust_sd": 6.4}
    assert R.loss_of(good, real)["loss"] < R.loss_of(loose, real)["loss"]
    assert R.loss_of(loose, real)["wander_term"] > 5
    few = {**good, "close_share": 0.14}
    assert R.loss_of(few, real)["close_term"] > R.loss_of(good, real)["close_term"]


def test_closing_loss_is_zero_when_the_closing_rates_match():
    r = {"closing_a_first": 0.78, "closing_b_first": 0.53}
    assert R.closing_loss(r, r) == 0.0
    assert R.closing_loss({"closing_a_first": 0.0, "closing_b_first": -0.4}, r) > 10


# ------------------------------------------------------------------ the whole thing

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("refit")
    w = lapworld.world(seed=13, seasons=(2022, 2023, 2024), races=6, n_laps=40, n_drivers=14, sc_races=(3,), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=10)
    return d


def test_the_refit_writes_its_tables_and_replaces_the_params(world):
    d = world
    before = json.loads((d / "cal" / "params.json").read_text())
    lines = R.run(out_dir=d / "refit", cal_dir=d / "cal", S=6, S_eval=6, S_check=4, drift_grid=(1.0, 0.0), noise_grid=(1.0,), warm_grid=(1.0, 0.0), progress=False)
    for f in ("refit_grid.csv", "warm_up_check.csv", "wander_check.csv", "pass_direction.csv", "lead_changes.csv", "findings.txt"):
        assert (d / "refit" / f).exists()
    assert (d / "cal" / "params_before_refit.json").exists()
    after = json.loads((d / "cal" / "params.json").read_text())
    assert "refit" in after and after["refit"]["warm_scale"] in (0.0, 1.0)
    assert after["params"]["drift_sd"] == pytest.approx(before["params"]["drift_sd"] * after["refit"]["drift_factor"])
    assert after["params"]["warm_scale"] == after["refit"]["warm_scale"]
    assert any(l.startswith("REFIT") for l in lines) and any(l.startswith("WARM-UP") for l in lines)
    grid = pd.read_csv(d / "refit" / "refit_grid.csv")
    assert grid["chosen"].sum() == len(grid["split"].unique())          # one chosen row per split
    m = grid[(grid["setting"] == "matched passing") & (grid["split"] == "all")]
    assert len(m) == 2 and m["threshold_factor"].between(R.THR_RANGE[0] * 0.99, R.THR_RANGE[1] * 1.01).all()
    # the params still load and run
    truths, par = C.load_truths(d / "cal")
    assert par.warm_scale == after["refit"]["warm_scale"]
    simulate(truths[0].spec(), par.for_circuit(truths[0].circuit_id), S=2, seed=0)


def test_running_the_refit_twice_gives_the_same_answer(world):
    d = world
    a = json.loads((d / "cal" / "params.json").read_text())
    R.run(out_dir=d / "refit2", cal_dir=d / "cal", S=6, S_eval=6, S_check=4, drift_grid=(1.0, 0.0), noise_grid=(1.0,), warm_grid=(1.0, 0.0), progress=False)
    b = json.loads((d / "cal" / "params.json").read_text())
    assert a["refit"] == b["refit"] and a["params"]["drift_sd"] == b["params"]["drift_sd"] and a["params"]["thr"] == b["params"]["thr"]


def test_pass_direction_and_lead_changes_have_sensible_shapes(world):
    d = world
    truths, par = C.load_truths(d / "cal")
    pd_ = R.pass_direction(truths[:6], {"a": par}, S=3, seed=0)
    assert {"real", "a"} <= set(pd_["setting"])
    rates = pd_[pd_["pace_advantage_of_car_behind_s"].str.startswith("(")]
    assert rates["swap_rate"].between(0, 1).all()
    lc = R.lead_changes(truths[:6], {"a": par}, S=3, seed=0)
    w = lc[lc["half"] == "whole race"]
    assert (w["per_race"] >= 0).all() and set(w["kind"]) == {"lose", "regain"}
    # the pole sitter cannot regain a lead more often than he loses it plus one (he may start on it)
    for s in ("real", "a"):
        x = w[w["setting"] == s].set_index("kind")["per_race"]
        assert x["regain"] <= x["lose"] + 1.0


def test_wander_check_has_real_and_every_setting(world):
    d = world
    truths, par = C.load_truths(d / "cal")
    wins = {t.race_id: R.wander_windows(t) for t in truths}
    evs = {t.race_id: R.closing_events(t) for t in truths}
    cw = R.wander_check(truths, wins, evs, {"v2.7": par, "v2.8": R.scaled(par, drift=0.0)}, S=4, seed=0)
    assert {"real", "v2.7", "v2.8"} == set(cw["setting"])
    a = cw[cw["split"] == "all"].set_index("setting")["robust_sd_s"]
    assert a["v2.8"] <= a["v2.7"] * 1.05              # less drift never makes the gaps wander more
