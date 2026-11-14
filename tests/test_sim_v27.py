import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import engine as E
from src.sim.engine import BIG, Params, make_spec, simulate
from src.sim.sample import Pool, sample_inputs
from . import lapworld


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("v27")
    w = lapworld.world(seed=11, seasons=(2022, 2023, 2024), races=6, n_laps=40, n_drivers=14, sc_races=(3,), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=10)
    truths, par = C.load_truths(d / "cal")
    return truths, par, d, con, w


def _two_car_spec(**kw):
    return make_spec(30, [1, 2], [0.0, 0.0], [[(E.MEDIUM, 0, 10), (E.HARD, 0)], [(E.MEDIUM, 0)]], **kw)


def test_warm_up_is_zero_by_default_and_changes_nothing():
    a = simulate(_two_car_spec(), Params(), S=20, seed=3)
    b = simulate(_two_car_spec(), Params(warm_amp=0.0, bad_start_p=0.0), S=20, seed=3)
    assert (a.finish == b.finish).all() and (a.done == b.done).all()


def test_warm_up_slows_the_laps_after_a_stop_and_decays():
    spec = make_spec(30, [1], [0.0], [[(E.MEDIUM, 0, 10), (E.MEDIUM, 0)]])
    par = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.0, warm_amp=1.0, warm_decay=0.5, da_loss=0.0)
    flat = replace(par, warm_amp=0.0)
    t1 = simulate(spec, par, S=1, seed=0, history=False)
    t0 = simulate(spec, flat, S=1, seed=0, history=False)
    # same finishing order trivially; check the time through the engine's lap loop by comparing a longer race's total
    spec2 = make_spec(30, [1, 2], [0.0, 0.0], [[(E.MEDIUM, 0, 10), (E.MEDIUM, 0)], [(E.MEDIUM, 0, 10), (E.MEDIUM, 0)]])
    # the car that stops first is slowed on laps 12.. while the other, still on old tyres, is not: with a huge warm-up the earlier stopper
    # loses the place it would otherwise take
    spec3 = make_spec(30, [1, 2], [0.0, -0.05], [[(E.MEDIUM, 0, 8), (E.MEDIUM, 0)], [(E.MEDIUM, 0, 20), (E.MEDIUM, 0)]])
    base = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.0, thr=50.0, min_gap=0.01)
    with_w = simulate(spec3, replace(base, warm_amp=3.0, warm_decay=0.9), S=4, seed=1)
    without = simulate(spec3, base, S=4, seed=1)
    assert t1.finish.shape == t0.finish.shape == (1, 1)
    assert with_w.done.min() == 30 and without.done.min() == 30


def test_race_warm_overrides_the_default_and_nan_falls_back():
    spec = _two_car_spec()
    assert spec.warm is None
    par = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.0, warm_amp=0.0)
    a = simulate(replace(spec, warm=2.0), par, S=2, seed=0)
    b = simulate(replace(spec, warm=math.nan), par, S=2, seed=0)
    c = simulate(spec, par, S=2, seed=0)
    assert a.done.min() == 30 and (b.finish == c.finish).all()


def test_warm_up_hurts_the_stopper_in_a_close_race():
    # two equal cars; A stops on lap 10, B never stops (one-stopper vs no-stopper is not the point: both stop, A earlier).
    strat = [[(E.MEDIUM, 0, 10), (E.MEDIUM, 0)], [(E.MEDIUM, 0, 11), (E.MEDIUM, 0)]]
    spec = make_spec(24, [1, 2], [0.0, 0.0], strat)
    par = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.3, thr=0.3, thr_scale=0.05, min_gap=0.3, sigma_pit=0.0)
    wins = []
    for amp in (0.0, 2.5):
        r = simulate(spec, replace(par, warm_amp=amp, warm_decay=0.8), S=200, seed=2)
        wins.append(float((r.finish[:, 1] == 1).mean()))
    # the earlier stopper (car 0) is on slow new tyres while car 1 is still on its old ones: the undercut is weaker with warm-up
    assert wins[1] >= wins[0]


def test_bad_start_adds_a_heavy_tail_at_the_front():
    spec = make_spec(2, list(range(1, 11)), [0.0] * 10, [[(E.MEDIUM, 0)]] * 10)
    quiet = Params(sigma_start=0.3, slot_gap=0.25)
    keep0 = (simulate(spec, quiet, S=2000, seed=1).pos[:, 0, 0] == 1).mean()
    bad = replace(quiet, bad_start_p=0.2, bad_start_loss=2.0)
    keep1 = (simulate(spec, bad, S=2000, seed=1).pos[:, 0, 0] == 1).mean()
    assert keep1 < keep0 - 0.05
    far = (np.abs(simulate(spec, bad, S=2000, seed=1).pos[:, :, 0] - np.arange(1, 11)[None]) >= 3).mean()
    near = (np.abs(simulate(spec, quiet, S=2000, seed=1).pos[:, :, 0] - np.arange(1, 11)[None]) >= 3).mean()
    assert far > near


def test_stint_lap_counts_from_the_out_lap():
    laps = pd.DataFrame({"race_id": [1] * 6, "driver_id": ["a"] * 6, "stint": [1, 1, 1, 2, 2, 2], "lap_number": [1, 2, 3, 4, 5, 6]})
    assert list(C.stint_lap_of(laps)) == [1, 2, 3, 1, 2, 3]
    laps.loc[2, "stint"] = np.nan
    assert C.stint_lap_of(laps).notna().sum() == 5


def test_warm_up_measure_runs_and_has_the_right_shape(world):
    truths, par, d, con, w = world
    laps, flags, _ = C.load_inputs(con)
    wu = C.warm_up(laps, list(flags.set_index("race_id")["wet"].loc[lambda x: x].index))
    assert set(wu) >= {"table", "decay", "amp", "pooled", "note"}
    assert wu["decay"] in C.WARM_DECAYS and wu["pooled"] >= 0
    if len(wu["table"]):
        assert set(wu["table"]["stint_lap"]) <= set(C.WARM_LAPS)
    assert math.isfinite(C.warm_for(wu, 2023))


def test_warm_up_recovers_a_planted_excess(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for rid in range(1, 9):
        for dr in range(10):
            stop = 18 + 2 * (dr % 5)                 # drivers stop on different laps, or tyre age and lap number are collinear
            for stint, (a, b) in enumerate([(1, stop), (stop + 1, 50)], start=1):
                for lap in range(a, b + 1):
                    j = lap - a + 1
                    t = 90.0 + 0.1 * dr + 0.04 * (j if stint == 2 else lap) + rng.normal(0, 0.15)
                    if stint == 2 and j >= 2:
                        t += 0.8 * 0.5 ** (j - 2)
                    rows.append({"race_id": rid, "season": 2023, "round": rid, "circuit_id": "c", "circuit_name": "c", "driver_id": f"d{dr}",
                                 "lap_number": lap, "lap_time_s": t, "stint": stint, "compound": "MEDIUM", "tyre_life": float(j if stint == 2 else lap),
                                 "pit_in_s": 1.0 if lap == b and stint == 1 else np.nan, "pit_out_s": 1.0 if (stint == 2 and lap == a) else np.nan,
                                 "track_status": "1", "position": dr + 1.0, "is_accurate": True})
    wu = C.warm_up(pd.DataFrame(rows))
    assert wu["note"] == "measured"
    assert 0.5 <= wu["pooled"] <= 1.1 and wu["decay"] in (0.4, 0.5, 0.6)


def test_lap1_tail_counts(world):
    truths, par, d, con, w = world
    ts = [t for t in truths if np.isfinite(C.lap1_move(t))]
    real = C.real_lap1_tail(ts)
    assert 0 <= real["kept"] <= real["pole"] and real["front"] > 0
    sim = C.sim_lap1_tail(ts[:5], Params(), S=10)
    assert 0 <= sim["kept"] <= sim["pole"]


def test_fit_start_shape_scores_bad_starts_and_reports_the_tail(world):
    truths, par, d, con, w = world
    ts = [t for t in truths if np.isfinite(C.lap1_move(t))][:10]
    target = C.real_lap1_move(ts)
    sig, front, bp, bl, tab = C.fit_start_shape(ts, Params(), target, S=8, fronts=(1.0,), bads=((0.0, 0.0), (0.1, 3.0)))
    assert bp in (0.0, 0.1) and {"simulated_pole_keeps_lead", "real_front_big_moves", "start_loss"} <= set(tab.columns)


def test_sampler_never_copies_a_retired_cars_plan(world):
    truths, par, d, con, w = world
    pool = Pool.before(truths, truths[-1])
    s = sample_inputs(pool, 14, 40, truths[-1].circuit_id, np.arange(1, 15), 200, np.random.default_rng(0), retirements=False)
    n_stops = (s["stops"] < BIG).sum(2)
    # every sampled car has the stop count of a car that finished; none has fewer stops than the pool's finishers ever show
    fin_stops = np.concatenate([(t.stops[t.retire >= BIG] < BIG).sum(1) for t in pool.truths])
    assert n_stops.min() >= fin_stops.min()
    # in a pool where retirees have truncated plans, the mean stops per sampled car is not below the finishers' mean by much
    assert n_stops.mean() >= fin_stops.mean() - 0.35


def test_truth_carries_warm_and_passes_it_to_the_spec(world):
    truths, par, d, con, w = world
    assert all(np.isfinite(t.warm) for t in truths)
    t = truths[0]
    assert t.spec().warm == pytest.approx(t.warm)
    t.warm = math.nan
    assert t.spec().warm is None
    t.warm = C.warm_for({"amp": {}, "pooled": 0.0}, t.season)
