import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import engine as E
from src.sim import replay as R
from src.sim.sample import Pool, sample_inputs, _episodes
from . import lapworld


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("r")
    w = lapworld.world(seed=5, seasons=(2022, 2023, 2024, 2025), races=6, n_laps=36, n_drivers=14, sc_races=(3, 4), wet_race=6)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=15)
    return con, d


def test_pool_only_holds_earlier_races(setup):
    con, d = setup
    truths, _ = C.load_truths(d / "cal")
    t = truths[10]
    pool = Pool.before(truths, t)
    assert pool.truths and all(p.race_id < t.race_id for p in pool.truths)


def test_sampled_inputs_have_the_right_shapes_and_valid_stops(setup):
    con, d = setup
    truths, _ = C.load_truths(d / "cal")
    t = truths[12]
    pool = Pool.before(truths, t)
    s = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, 30, np.random.default_rng(0))
    assert s["comp"].shape[:2] == (30, t.n) and s["neutral"].shape == (30, t.n_laps + 1)
    st = s["stops"]
    valid = st < E.BIG
    assert (st[valid] >= 2).all() and (st[valid] <= t.n_laps - 1).all()
    both = valid[:, :, 1:] & valid[:, :, :-1]
    assert (st[:, :, 1:][both] > st[:, :, :-1][both]).all()          # strictly increasing
    assert s["retire"].min() >= 1
    assert set(np.unique(s["neutral"])) <= {0, 1, 2}


def test_episodes_keeps_duration():
    nv = np.zeros(31, int)
    nv[10:13] = 2
    nv[20:22] = 1
    assert _episodes(nv) == [(10, 3, 2), (20, 2, 1)]


def test_finish_metrics_perfect_and_reversed():
    n = 10
    actual = np.arange(1, n + 1)
    perfect = np.tile(actual, (20, 1))
    m = R.finish_metrics(perfect, actual)
    assert m["spearman"] == pytest.approx(1.0) and m["winner_top1"] and m["brier_top3"] == pytest.approx(0.0)
    bad = R.finish_metrics(np.tile(actual[::-1], (20, 1)), actual)
    assert bad["spearman"] == pytest.approx(-1.0) and not bad["winner_top1"] and bad["winner_logscore"] > 2


def test_grid_baseline_uses_the_grid(setup):
    con, d = setup
    truths, _ = C.load_truths(d / "cal")
    t = truths[-1]
    b = R.grid_baseline(Pool.before(truths, t), t.slot, t.finish)
    assert 0 <= b["base_brier_top3"] < 0.3 and 0 < b["base_winner_logscore"] < 10


def test_sim_laps_table_feeds_the_stage2_functions(setup):
    con, d = setup
    truths, par = C.load_truths(d / "cal")
    t = truths[3]
    res = R.run_l0(t, par, 6, 1)
    laps = R.sim_laps(t, res, 0)
    assert {"driver_id", "lap_number", "position", "pit_in_s", "track_status"} <= set(laps.columns)
    ev, cats = R.sim_events(t, res, 3)
    assert len(cats) == 3 and all(isinstance(c, str) for c in cats)


def test_full_replay_writes_tables_and_beats_or_matches_the_grid_with_known_pace(setup):
    con, d = setup
    lines = R.run(con, d / "rep", cal_dir=d / "cal", S=40, s_events=4, progress=False)
    assert any(l.startswith("L0") for l in lines) and any(l.startswith("L1") for l in lines)
    names = {p.name for p in (d / "rep").iterdir()}
    assert {"replay_races.csv", "replay_summary.csv", "grip_curves.csv", "circuit_check.csv", "findings.txt",
            "replay_levels.png", "replay_grip.png"} <= names
    s = pd.read_csv(d / "rep" / "replay_summary.csv")
    l0 = s[(s["level"].str.startswith("L0")) & (s["split"] == "all")].iloc[0]
    l1 = s[(s["level"].str.startswith("L1")) & (s["split"] == "all")].iloc[0]
    assert l0["finish_order_correlation"] >= l1["finish_order_correlation"] - 0.05
    assert l0["finish_order_correlation"] > l0["grid_only_correlation"] - 0.05
    p = pd.read_csv(d / "rep" / "pass_check.csv")
    assert {"real_close_share", "simulated_close_share", "real_lap1_places_moved", "simulated_lap1_places_moved"} <= set(p.columns)
    assert p["real_close_share"].notna().any() and p["real_lap1_places_moved"].notna().any()
    assert any(l.startswith("CAR-LAPS WITHIN 1 s") for l in lines) and any(l.startswith("PLACES MOVED ON LAP 1") for l in lines)
    assert {"position_check.csv", "stop_timing_check.csv"} <= names
    assert any(l.startswith("PASSES BY RUNNING POSITION") for l in lines) and any(l.startswith("STOP TIMING") for l in lines)
    pos = pd.read_csv(d / "rep" / "position_check.csv")
    assert set(pos["split"]) >= {"all"} and len(pos[pos["split"] == "all"]) == len(C.BANDS)


def test_stop_timing_check_compares_real_stops_with_the_strategies_the_simulator_draws(setup):
    con, d = setup
    truths, _ = C.load_truths(d / "cal")
    tab = R.stop_timing_check(truths, S=6)
    a = tab[tab["split"] == "all"].set_index("statistic")
    assert {"stops per car", "share of cars that never stop", "gap between neighbours' first stops (share of race)"} <= set(a.index)
    assert a.loc["stops per car", "real"] > 0.5 and a.loc["stops per car", "sampled"] > 0.5
    med = a.loc["first stop, 50th percentile (share of race)"]
    assert 0.0 < med["real"] < 1.0 and 0.0 < med["sampled"] < 1.0
    q = [a.loc[f"first stop, {int(x * 100)}th percentile (share of race)", "sampled"] for x in R.STOP_QUANTILES]
    assert q == sorted(q)


def test_first_stops_reads_shares_and_neighbour_gaps():
    stops = np.array([[10, E.BIG], [20, E.BIG], [E.BIG, E.BIG], [30, E.BIG]])
    slot = np.array([1, 2, 3, 4])
    frac, ns, diff = R._first_stops(stops, slot, 40, np.ones(4, bool))
    assert np.allclose(frac[[0, 1, 3]], [0.25, 0.5, 0.75]) and np.isnan(frac[2]) and ns.tolist() == [1, 1, 0, 1]
    assert diff.tolist() == [0.25]                                  # only slots 1 and 2 both stop
    _, _, diff2 = R._first_stops(stops, slot, 40, np.array([True, False, True, True]))
    assert len(diff2) == 0                                          # a retired neighbour is left out


def test_lap1_check_has_every_band_and_finite_numbers(setup):
    con, d = setup
    truths, par = C.load_truths(d / "cal")
    tab = R.lap1_check(truths, par, S=10)
    assert {"1-2", "3-6", "7-12", "13+"} <= set(tab["slots"]) and {"all", "train"} <= set(tab["split"])
    assert np.isfinite(tab[["real_places_moved", "simulated_places_moved"]].to_numpy()).all()
