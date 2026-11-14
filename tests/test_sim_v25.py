"""Simulator v2.5: start noise by slot, out-lap share of a stop, recency-weighted pool, no-stop fix, stop spread,
retirement by grid band, drift share, pass-by-position loss and the undercut / start-shape fits."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import engine as E
from src.sim.engine import BIG, MEDIUM, Params, make_spec, simulate
from src.sim.sample import Pool, RET_BANDS, first_stop_gap, sample_inputs
from . import lapworld

QUIET = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=6.0)


def strat(stop=None):
    return [(MEDIUM, 0)] if stop is None else [(MEDIUM, 0, stop), (MEDIUM, 0)]


# ------------------------------------------------------------------ engine

def test_outlap_share_moves_the_cost_to_the_next_lap_without_changing_the_total():
    spec = make_spec(40, np.array([1, 2]), np.zeros(2), [strat(10), strat(25)], fuel=0.0, deg=np.zeros(3))
    r0 = simulate(spec, replace(QUIET, outlap_share=0.0), S=1)
    r1 = simulate(spec, replace(QUIET, outlap_share=1.0), S=1)
    # the car on the stop lap: with the whole cost on the out-lap it is still in front at the end of its in-lap, behind a lap later
    assert r0.pos[0, 0, 9] == 2 and r1.pos[0, 0, 9] == 1
    assert r1.pos[0, 0, 10] == 2
    # once both have stopped (lap 25 on) the gap between the cars is the same either way
    assert np.allclose(np.nan_to_num(r0.gap[0, :, 30]), np.nan_to_num(r1.gap[0, :, 30]), atol=1e-4)


def test_pole_sitter_noise_is_smaller_with_start_front():
    n = 6
    spec = make_spec(2, np.arange(1, n + 1), np.zeros(n), [strat()] * n, fuel=0.0, deg=np.zeros(3))
    wide = Params(sigma_start=2.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=1.0, thr=6.0)
    kept = [(simulate(spec, replace(wide, start_front=f), S=400, seed=3).pos[:, 0, 0] == 1).mean() for f in (1.0, 0.0)]
    assert kept[1] > 0.95 and kept[0] < 0.8


def test_start_front_one_changes_nothing():
    n = 6
    spec = make_spec(10, np.arange(1, n + 1), np.zeros(n), [strat()] * n)
    a = simulate(spec, Params(), S=20, seed=4)
    b = simulate(spec, Params(start_front=1.0, start_ramp=3.0), S=20, seed=4)
    assert (a.finish == b.finish).all()


# ----------------------------------------------------------------- sampler

@pytest.fixture(scope="module")
def truths(tmp_path_factory):
    d = tmp_path_factory.mktemp("v25")
    w = lapworld.world(seed=5, seasons=(2022, 2023, 2024, 2025), races=6, n_laps=36, n_drivers=14, sc_races=(3, 4), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=15)
    ts, par = C.load_truths(d / "cal")
    return ts


def test_cars_with_no_stop_in_the_template_take_a_neighbours_strategy(truths):
    t = truths[12]
    stops0 = t.stops.copy()
    stops0[np.argsort(t.slot)[4], :] = BIG
    pool = Pool([replace(p, stops=np.where(np.arange(p.n)[:, None] == np.argsort(p.slot)[4], BIG, p.stops)) for p in truths[:11]])
    s = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, 40, np.random.default_rng(0), retirements=False, stop_jitter=0.0)
    assert (s["stops"][:, :, 0] < BIG).all()


def test_recent_races_count_for_more(truths):
    pool = Pool(truths[:20], season=2025)
    w = pool._weights(pool.truths, "no_such_circuit")
    seasons = np.array([p.season for p in pool.truths])
    assert w[seasons == seasons.max()].mean() > w[seasons == seasons.min()].mean()
    assert Pool(truths[:20])._weights(truths[:20], "x").std() == pytest.approx(0.0)       # no season, no recency weighting


def test_retirement_rate_is_lower_at_the_front_when_the_pool_says_so(truths):
    rows = []
    for p in truths[:20]:
        ret = np.where(p.slot <= 3, BIG, np.where(np.arange(p.n) % 2 == 0, 5, BIG))
        rows.append(replace(p, retire=ret))
    rates = Pool(rows).retire_rates()
    assert len(rates) == len(RET_BANDS) and rates[0] < rates[1] and rates[0] < rates[2]
    t = truths[12]
    s = sample_inputs(Pool(rows), t.n, t.n_laps, t.circuit_id, t.slot, 400, np.random.default_rng(1))
    front = np.argsort(t.slot)[:3]
    assert (s["retire"][:, front] < BIG).mean() < (s["retire"][:, np.argsort(t.slot)[10:]] < BIG).mean()


def test_stop_jitter_spreads_stops_and_keeps_them_valid(truths):
    t = truths[12]
    pool = Pool.before(truths, t)
    plain = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, 60, np.random.default_rng(2), retirements=False, stop_jitter=0.0)
    wide = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, 60, np.random.default_rng(2), retirements=False, stop_jitter=0.06)
    g0 = np.nanmean([first_stop_gap(plain["stops"][i], t.slot, t.n_laps) for i in range(60)])
    g1 = np.nanmean([first_stop_gap(wide["stops"][i], t.slot, t.n_laps) for i in range(60)])
    assert g1 > g0
    st = wide["stops"]
    valid = st < BIG
    assert (st[valid] >= 2).all() and (st[valid] <= t.n_laps - 1).all()
    both = valid[:, :, 1:] & valid[:, :, :-1]
    assert (st[:, :, 1:][both] > st[:, :, :-1][both]).all()


def test_the_pools_own_stop_jitter_is_non_negative_and_capped(truths):
    from src.sim.sample import JITTER_MAX, _stop_jitter
    j = _stop_jitter(Pool.before(truths, truths[15]))
    assert 0.0 <= j <= JITTER_MAX


# ------------------------------------------------------------- calibration

def test_start_shape_picks_a_front_share_from_its_grid_and_reports_bands(truths):
    ts = [t for t in truths if np.isfinite(C.lap1_move(t))][:10]
    target = C.real_lap1_move(ts)
    sig, front, bp, bl, tab = C.fit_start_shape(ts, Params(), target, S=10, fronts=(1.0, 0.5), bads=((0.0, 0.0), (0.06, 2.0)))
    assert front in (1.0, 0.5) and sig >= 0.0 and bp in (0.0, 0.06)
    assert set(tab["slots"]) == {"1-2", "3-6", "7-12", "13+"} and len(tab) == 16
    real = C.real_lap1_bands(ts)
    assert real.shape == (4, 2) and (real[:, 1] > 0).all()


def test_band_pass_term_is_part_of_the_loss(truths):
    ts = truths[:8]
    bands = C.real_pass_bands(ts)
    assert bands.shape == (len(C.BANDS), 2) and (bands[:, 1] > 0).all()
    par = Params()
    rate = C.real_pass_rate(ts)
    plain = C.fit_loss(ts, par, rate, 8, 1)
    withb = C.fit_loss(ts, par, rate, 8, 1, real_bands=bands)
    assert withb[2] >= plain[2] and np.isfinite(withb[0])


def test_fit_passing_searches_drift_share_and_front(truths):
    ts = truths[:8]
    base = Params(drift_sd=0.3, drift_rho=0.8, sigma_lap=0.3)
    par, grid, edge = C.fit_passing(ts, base, S=8, thr_grid=(0.4, 0.8), start_grid=(1.0,), slot_grid=(0.3, 0.6), noise_grid=(1.0, 1.5),
                                    gap_grid=(0.6,), drift_grid=(0.5, 1.0), front_grid=(1.0, 0.5), bad_grid=((0.0, 0.0), (0.06, 2.0)), rounds=1)
    assert 0.5 <= par.drift_sd / base.drift_sd <= 1.0
    # the total variance of a lap does not change with the drift share
    noise = grid.attrs["drift_share"]
    assert noise in (0.5, 1.0) and "drift" not in edge
    assert par.start_front in (1.0, 0.5)
    assert len(grid.attrs["start_shape"]) > 0


def test_fit_outlap_returns_a_share_from_its_grid_or_leaves_it_alone(truths):
    ts = truths[:10]
    empty = C.fit_outlap(ts, Params(outlap_share=0.0), pd.DataFrame())
    assert empty[0] == 0.0 and empty[1].empty


def test_fit_outlap_runs_on_real_looking_events_and_the_share_changes_the_undercut(tmp_path):
    from src.analysis import race_flow
    w = lapworld.world(seed=4, seasons=(2022, 2023, 2024, 2025), races=8, n_laps=50, n_drivers=16, sc_races=(3, 4), wet_race=6)
    con = lapworld.build_con(tmp_path / "w.duckdb", w)
    laps, flags, _ = C.load_inputs(con)
    cal = C.calibrate(con, S=10, thr_grid=(0.4, 0.8), start_grid=(1.0, 2.0))
    train = [t for t in cal["truths"] if C.split_of(t.season) == "train"]
    ids = {t.race_id for t in train}
    ev = race_flow.undercut_table(laps[laps["race_id"].isin(ids)], flags[flags["race_id"].isin(ids)])
    share, tab = C.fit_outlap(train, replace(cal["params"], outlap_share=0.0), ev, S=6)
    assert share in C.OUTLAP_GRID and len(tab) == len(C.OUTLAP_GRID)
    assert tab["simulated_close"].max() - tab["simulated_close"].min() > 0.02       # the split does move the undercut
    assert tab.loc[tab["outlap_share"] == share, "loss"].iloc[0] == tab["loss"].min()
    assert np.isfinite(cal["params"].outlap_share) and cal["params"].outlap_share in C.OUTLAP_GRID
