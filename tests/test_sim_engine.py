import numpy as np
import pytest
from scipy.stats import spearmanr

from src.sim import engine as E
from src.sim.engine import BIG, HARD, MEDIUM, Params, make_spec, simulate, spearman_masked

QUIET = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=6.0)   # passing only when clearly faster


def strat(stop=None, comp=MEDIUM):
    return [(comp, 0)] if stop is None else [(comp, 0, stop), (comp, 0)]


def test_equal_cars_keep_grid_order_without_noise():
    n = 8
    spec = make_spec(40, np.arange(1, n + 1), np.zeros(n), [strat()] * n, fuel=0.0, deg=np.zeros(3))
    r = simulate(spec, QUIET, S=3)
    assert (r.finish == np.arange(1, n + 1)).all()


def test_much_faster_car_works_through_the_field():
    n = 8
    pace = np.zeros(n)
    pace[-1] = -2.0                       # last on the grid, two seconds a lap quicker
    spec = make_spec(50, np.arange(1, n + 1), pace, [strat()] * n, fuel=0.0, deg=np.zeros(3))
    r = simulate(spec, Params(sigma_start=0.0, sigma_lap=0.1, sigma_stint=0.0), S=60, seed=1)
    assert np.median(r.finish[:, -1]) <= 2


def test_slightly_faster_car_is_held_up():
    n = 6
    pace = np.zeros(n)
    pace[-1] = -0.2                       # below the passing threshold
    spec = make_spec(50, np.arange(1, n + 1), pace, [strat()] * n, fuel=0.0, deg=np.zeros(3))
    r = simulate(spec, Params(sigma_start=0.0, sigma_lap=0.05, sigma_stint=0.0), S=60, seed=2)
    assert np.median(r.finish[:, -1]) >= 5


def test_an_extra_stop_costs_about_the_pit_loss():
    n = 2
    spec = make_spec(40, [1, 2], [0.0, 0.0], [strat(20), strat(None)], fuel=0.0, deg=np.zeros(3),
                     off=np.zeros(3), pit_loss=22.0)
    spec2 = make_spec(40, [1, 2], [0.0, 0.0], [strat(20), strat(None)], fuel=0.0, deg=np.zeros(3),
                      off=np.zeros(3), pit_loss=22.0)
    r = simulate(spec, QUIET, S=2)
    assert (r.finish[:, 0] == 2).all() and r.pit[:, 0].sum() == 2     # the stopper finishes behind and pitted once per sim


def test_retired_cars_finish_last_in_order_of_laps_done():
    n = 5
    ret = np.full(n, BIG)
    ret[0], ret[1] = 10, 20               # the pole sitter goes out first
    spec = make_spec(30, np.arange(1, n + 1), np.zeros(n), [strat()] * n, retire=ret, fuel=0.0, deg=np.zeros(3))
    r = simulate(spec, QUIET, S=2)
    assert (r.finish[:, 0] == 5).all() and (r.finish[:, 1] == 4).all()
    assert (r.done[:, 0] == 9).all() and (r.done[:, 1] == 19).all()
    assert (r.pos[:, 0, 9:] == 0).all()


def test_safety_car_closes_the_field_and_cheapens_a_stop():
    n = 6
    neutral = np.zeros(31, int)
    neutral[10:13] = 2
    pace = np.zeros(n)
    base = make_spec(30, np.arange(1, n + 1), pace, [strat()] * n, neutral=neutral, fuel=0.0, deg=np.zeros(3))
    # big grid gaps, then a safety car: afterwards the gaps are small, so a car that stops under it loses little
    par = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=5.0)
    r = simulate(base, par, S=1, history=True)
    assert (r.finish == np.arange(1, n + 1)).all()
    green = make_spec(30, np.arange(1, n + 1), pace, [strat(11)] + [strat()] * (n - 1), fuel=0.0, deg=np.zeros(3), off=np.zeros(3))
    under = make_spec(30, np.arange(1, n + 1), pace, [strat(11)] + [strat()] * (n - 1), neutral=neutral, fuel=0.0,
                      deg=np.zeros(3), off=np.zeros(3))
    pg = simulate(green, par, S=1).finish[0, 0]
    pu = simulate(under, par, S=1).finish[0, 0]
    assert pu < pg


def test_a_stop_onto_fresh_tyres_can_undercut():
    """Car B leads A by 1.2 s on worn tyres; A stops first, and B stops three laps later: A should come out ahead more often
    than when A stops after B."""
    wear = np.array([0.0, 0.15, 0.0])
    par = Params(sigma_start=0.0, sigma_lap=0.08, sigma_stint=0.0, slot_gap=1.2, thr=0.5)

    def ahead(a_stop, b_stop, seed):
        spec = make_spec(30, [2, 1], [0.0, 0.0], [strat(a_stop), strat(b_stop)], fuel=0.0, deg=wear, off=np.zeros(3), pit_loss=20.0)
        r = simulate(spec, par, S=300, seed=seed)
        return float((r.finish[:, 0] < r.finish[:, 1]).mean())

    first = ahead(18, 21, 1)
    second = ahead(21, 18, 1)
    assert first > second + 0.2


def test_spearman_masked_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.permutation(10).astype(float)
    y = x + rng.normal(0, 3, 10)
    mask = np.ones(10, bool)
    mask[[2, 7]] = False
    got = spearman_masked(x[None], y[None], mask[None])[0]
    assert got == pytest.approx(spearmanr(x[mask], y[mask])[0])


def test_broadcast_per_simulation_inputs():
    n = 4
    spec = make_spec(20, np.arange(1, n + 1), np.zeros(n), [strat()] * n, fuel=0.0, deg=np.zeros(3))
    spec.retire = np.stack([np.full((n,), BIG), np.array([5, BIG, BIG, BIG])])   # sim 1 loses the pole car
    r = simulate(spec, QUIET, S=2)
    assert r.finish[0, 0] == 1 and r.finish[1, 0] == 4


def test_for_circuit_swaps_only_the_passing_threshold():
    p = Params(thr=1.0, circuit_thr={"monaco": 3.0})
    assert p.for_circuit("monaco").thr == 3.0 and p.for_circuit("monza").thr == 1.0
    assert p.for_circuit("monaco").sigma_lap == p.sigma_lap


def test_pit_cost_spread_changes_who_gains_from_a_close_stop():
    """With a noisy stop cost two otherwise identical cars that stop together no longer come out in grid order every time."""
    n = 2
    spec = make_spec(30, [1, 2], np.zeros(n), [strat(15)] * n, fuel=0.0, deg=np.zeros(3), pit_loss=20.0)
    calm = simulate(spec, Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=0.3), S=200, seed=1)
    noisy = simulate(spec, Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=0.3, sigma_pit=1.5), S=200, seed=1)
    assert (calm.finish[:, 0] == 1).all()
    assert 0.0 < (noisy.finish[:, 0] == 1).mean() < 1.0


def test_lap_one_noise_comes_from_sigma_start_alone():
    n = 6
    spec = make_spec(2, np.arange(1, n + 1), np.zeros(n), [strat()] * n, fuel=0.0, deg=np.zeros(3))
    calm = simulate(spec, Params(sigma_start=0.0, sigma_lap=5.0, sigma_stint=0.0, slot_gap=0.25), S=50, seed=0)
    assert (calm.pos[:, :, 0] == np.arange(1, n + 1)).all()           # lap-1 order is the grid when sigma_start is zero


def test_noise_multipliers_by_position_scale_the_noise():
    n = 8
    spec = make_spec(30, np.arange(1, n + 1), np.linspace(0, 0.5, n), [strat()] * n, fuel=0.0, deg=np.zeros(3))
    sharp = dict(sigma_start=0.0, sigma_lap=1.5, sigma_stint=0.0, thr=0.3, thr_scale=0.02)
    calm = simulate(spec, Params(**sharp, noise_top=0.0, noise_mid=0.0, noise_rest=0.0), S=12, seed=1)
    assert (calm.finish == calm.finish[0]).all()                      # no noise at all: every simulation is the same
    loud = simulate(spec, Params(**sharp), S=12, seed=1)
    assert not (loud.finish == loud.finish[0]).all()


def test_cover_rule_pulls_the_car_ahead_in_only_for_sampled_stops():
    spec = make_spec(40, [1, 2], [0.0, 0.0], [strat(28), strat(20)], fuel=0.0, deg=np.zeros(3), off=np.zeros(3))
    par = Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=6.0, cover_prob=1.0)
    plain = simulate(spec, par, S=3)
    assert plain.pit[:, 0, 27].all() and not plain.pit[:, 0, 20].any()      # the planned lap
    spec.cover = True
    covered = simulate(spec, par, S=3)
    assert covered.pit[:, 0, 20].all() and not covered.pit[:, 0, 27].any()  # the lap after the rival's stop
    off = simulate(spec, Params(**{**par.__dict__, "cover_prob": 0.0}), S=3)
    assert off.pit[:, 0, 27].all()


def test_pace_drift_makes_good_and_bad_spells_last():
    spec = make_spec(60, np.arange(1, 3), np.zeros(2), [strat()] * 2, fuel=0.0, deg=np.zeros(3))
    base = dict(sigma_start=0.0, sigma_lap=0.3, sigma_stint=0.0, min_gap=0.0, thr=99.0)
    plain = simulate(spec, Params(**base), S=300, seed=2)
    drift = simulate(spec, Params(**base, drift_sd=0.4, drift_rho=0.95), S=300, seed=2)
    end = lambda r: np.nanmean(np.abs(r.gap[:, :, -1]))
    assert end(drift) > 1.5 * end(plain)          # spells that last add up over the race; lap-to-lap noise mostly cancels


def test_default_settings_leave_the_engine_unchanged():
    n = 6
    spec = make_spec(30, np.arange(1, n + 1), np.linspace(0, 0.4, n), [strat()] * n)
    a = simulate(spec, Params(), S=20, seed=5)
    b = simulate(spec, Params(drift_sd=0.0, thr_sd=0.0), S=20, seed=5)
    assert (a.finish == b.finish).all()


def test_passing_difficulty_varies_from_race_to_race():
    from src.sim import calibrate as C
    n = 8
    spec = make_spec(50, np.arange(1, n + 1), np.linspace(0, 0.5, n)[::-1], [strat()] * n, fuel=0.0, deg=np.zeros(3))
    base = dict(thr=0.5, sigma_lap=0.3)
    fixed = simulate(spec, Params(**base), S=200, seed=3)
    varied = simulate(spec, Params(**base, thr_sd=0.8), S=200, seed=3)
    pf, _ = C.sim_passes(fixed, 50)
    pv, _ = C.sim_passes(varied, 50)
    assert pv.var() > pf.var()


def test_dirty_air_pushes_a_follower_back_until_it_is_out_of_range():
    spec = make_spec(20, np.arange(1, 3), np.zeros(2), [strat()] * 2, fuel=0.0, deg=np.zeros(3))
    base = dict(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.5, min_gap=0.5, thr=99.0)
    plain = simulate(spec, Params(**base), S=2, seed=0)
    air = simulate(spec, Params(**base, da_loss=0.4, da_range=2.0), S=2, seed=0)
    assert plain.gap[0, 1, -1] == pytest.approx(0.5)               # no dirty air: the follower stays where it started
    assert 1.8 < air.gap[0, 1, -1] <= 2.0 + 1e-9                   # each lap costs 0.4 (1 - gap / 2): the gap opens up to the range
    assert (np.diff(air.gap[0, 1, 1:]) >= -1e-9).all()


def test_dirty_air_makes_a_pass_need_a_bigger_pace_advantage_and_is_off_by_default():
    spec = make_spec(12, np.arange(1, 3), np.array([0.0, -0.6]), [strat()] * 2, fuel=0.0, deg=np.zeros(3))
    base = dict(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=1.5, min_gap=0.4, thr=0.3, thr_scale=0.05)
    free = simulate(spec, Params(**base), S=4, seed=0)
    held = simulate(spec, Params(**base, da_loss=1.5, da_range=2.0), S=4, seed=0)
    assert (free.finish[:, 1] == 1).all()                         # 0.6 s a lap quicker passes at once
    assert (held.finish[:, 1] == 2).all()                         # but loses most of that in the dirty air, so never gets by
    assert (simulate(spec, Params(**base, da_loss=0.0), S=4, seed=0).finish == free.finish).all()


def test_dirty_air_is_not_charged_on_neutralised_laps():
    n = 3
    neutral = np.zeros(11, int)
    neutral[3:6] = 2
    spec = make_spec(10, np.arange(1, n + 1), np.zeros(n), [strat()] * n, fuel=0.0, deg=np.zeros(3), neutral=neutral)
    base = dict(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.5, min_gap=0.5, thr=99.0, sc_close=0.0)
    r = simulate(spec, Params(**base, da_loss=0.5, da_range=2.0), S=1, seed=0)
    assert r.gap[0, 1, 1] > r.gap[0, 1, 0]                         # lap 2 is green: the follower loses time
    assert np.allclose(r.gap[0, 1:, 1:5], r.gap[0, 1:, 1:2])        # laps 3-5 are neutralised: every car runs the same time, gaps stay
