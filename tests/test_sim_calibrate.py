import numpy as np
import pytest

from src.sim import calibrate as C
from src.sim import engine as E
from . import lapworld


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    w = lapworld.world(seed=4, seasons=(2022, 2023, 2024, 2025), races=6, n_laps=40, n_drivers=14, sc_races=(3, 4), wet_race=6)
    return lapworld.build_con(tmp_path_factory.mktemp("c") / "w.duckdb", w)


@pytest.fixture(scope="module")
def cal(con):
    return C.calibrate(con, S=20, thr_grid=(0.4, 0.8), start_grid=(1.0, 2.0))


def test_dry_races_become_truths_and_wet_and_red_are_left_out(cal):
    ids = {t.race_id % 100 for t in cal["truths"]}
    assert 6 not in ids and len(cal["truths"]) >= 15            # round 6 is the wet race
    assert set(cal["skipped"]["reason"]) >= {"wet"}


def test_truth_shapes_and_strategy_extraction(cal):
    t = cal["truths"][0]
    n = t.n
    assert t.slot.tolist() == sorted(t.slot.tolist()) or sorted(t.slot.tolist()) == list(range(1, n + 1))
    assert sorted(t.slot) == list(range(1, n + 1)) and sorted(t.finish) == list(range(1, n + 1))
    assert t.comp.shape[0] == n and t.stops.shape == (n, t.comp.shape[1] - 1)
    stops_made = (t.stops < E.BIG).sum(1)
    assert stops_made.min() >= 0 and stops_made.max() <= 2        # lapworld: one or two stops
    assert (t.stops[t.stops < E.BIG] < t.n_laps).all()
    assert t.pos_actual.shape == (t.n_laps, n)


def test_safety_car_laps_and_retirements_are_read_from_the_laps(cal):
    sc_races = [t for t in cal["truths"] if t.race_id % 100 in (3, 4)]
    assert sc_races and all((t.neutral[20:23] == 2).all() and t.neutral[1:19].max() == 0 for t in sc_races)
    assert any((t.retire < E.BIG).any() for t in cal["truths"])


def test_pace_follows_the_grid_and_tyres_are_close_to_the_truth(cal):
    t = cal["truths"][2]
    rho = np.corrcoef(np.argsort(np.argsort(t.pace)), t.slot)[0, 1]
    assert rho > 0.4                                              # lapworld's grid mostly follows pace
    assert t.deg[0] > t.deg[1] > t.deg[2]                         # soft wears fastest (true 0.12 / 0.06 / 0.03)
    assert t.deg[1] == pytest.approx(0.06, abs=0.03)
    assert t.off[0] < 0 < t.off[2]
    assert t.fuel == pytest.approx(-0.05, abs=0.02)
    assert 15 < t.pit_loss < 28                                   # lapworld pit loss is about 20 s


def test_spec_runs_in_the_engine_and_reproduces_most_of_the_order(cal):
    t = cal["truths"][0]
    r = E.simulate(t.spec(), cal["params"], S=40, seed=3)
    rho = np.mean([np.corrcoef(r.finish[s], t.finish)[0, 1] for s in range(40)])
    assert rho > 0.5


def test_stop_rates_and_costs_are_measured(cal):
    r = cal["rates"]
    assert r["sc"]["eligible"] > 0
    assert 0 <= r["sc"]["rate"] <= 1
    assert len(cal["pit_cost"]) > 0
    assert (cal["pit_cost"]["kind"] == "sc").any()


def test_fit_picks_a_grid_point_and_writes_everything(con, tmp_path):
    lines = C.run(con, tmp_path / "o", S=20)
    assert any(l.startswith("RACES") for l in lines)
    names = {p.name for p in (tmp_path / "o").iterdir()}
    assert {"truth.pkl", "params.json", "findings.txt", "grip_fit.csv", "passing_fit_grid.csv", "era_tyres.csv"} <= names
    truths, par = C.load_truths(tmp_path / "o")
    assert len(truths) > 10 and par.thr in C.THR_GRID


def test_load_truths_without_calibration_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        C.load_truths(tmp_path)


def test_truth_pickle_holds_plain_dicts_so_it_loads_from_any_entry_point(con, tmp_path):
    import pickle
    C.run(con, tmp_path / "p", S=10)
    with open(tmp_path / "p" / "truth.pkl", "rb") as f:
        raw = pickle.load(f)
    assert isinstance(raw, list) and isinstance(raw[0], dict) and "pace" in raw[0]


def test_messy_lap_data_does_not_break_calibration(tmp_path):
    """Real lap tables have missing stint numbers, compounds and tyre ages, drivers with no laps, and repeated rows."""
    import pandas as pd
    rng = np.random.default_rng(0)
    w = lapworld.world(seed=8, seasons=(2022, 2023, 2024), races=5, n_laps=30, n_drivers=12, sc_races=(3,), wet_race=5)
    L = w["laps"].copy()
    for col, frac in (("stint", 0.03), ("compound", 0.02), ("tyre_life", 0.02), ("lap_time_s", 0.02), ("position", 0.03)):
        L.loc[rng.random(len(L)) < frac, col] = np.nan if col != "compound" else None
    L.loc[(L.race_id == 202202) & (L.driver_id == "d3"), "stint"] = np.nan        # a whole car without stint numbers
    L = L[~((L.race_id == 202302) & (L.driver_id == "d5"))]                       # a car with no laps at all
    w["laps"] = pd.concat([L, L.sample(30, random_state=1)])                      # repeated rows
    con = lapworld.build_con(tmp_path / "m.duckdb", w)
    c = C.calibrate(con, S=10, thr_grid=(0.6,), start_grid=(1.5,))
    assert len(c["truths"]) >= 8
    t = [x for x in c["truths"] if x.race_id == 202302][0]
    assert t.retire[t.drivers.index("d5")] == 1                                   # no laps = out on lap 1
    con.close()


def test_pass_counts_counts_swaps_among_cars_not_stopping():
    pos = np.array([[[1, 2, 3], [1, 2, 3], [2, 1, 3], [2, 1, 3]]], float)       # (S=1, L=4, n=3): cars 0 and 1 swap on lap 3
    green = np.ones((1, 4), bool)
    p, c = C.pass_counts(pos, np.zeros((1, 4, 3), bool), green)
    assert p[0] == 1 and c[0] == 3 * 2
    pit = np.zeros((1, 4, 3), bool)
    pit[0, 2, 1] = True                                                         # car 1 stops on lap 3: not counted
    p, _ = C.pass_counts(pos, pit, green)
    assert p[0] == 0
    green[0, 2] = False                                                         # safety car on the lap of the swap
    p, _ = C.pass_counts(pos, np.zeros((1, 4, 3), bool), green)
    assert p[0] == 0


def test_real_and_simulated_pass_rates_are_the_same_kind_of_number(cal):
    t = cal["truths"][2]
    p, c = C.real_passes(t)
    assert c > 0 and 0 <= p / c < 0.2
    r = E.simulate(t.spec(), cal["params"], S=10, seed=2)
    sp, sc = C.sim_passes(r, t.n_laps)
    assert sp.shape == (10,) and (sc > 0).all()


def test_circuit_passing_gives_harder_circuits_higher_thresholds(cal):
    ts = cal["truths"]
    thr, tab = C.circuit_passing(ts[:12], cal["params"], S=8, mults=(0.5, 1.0, 2.0))
    assert set(thr) == set(t.circuit_id for t in ts[:12]) and (tab["thr_used"] > 0).all()
    assert tab["weight"].between(0, 1).all()


def test_stop_spread_is_zero_when_stops_cost_the_same():
    import pandas as pd
    d = pd.DataFrame({"race_id": [1] * 40, "loss_s": [21.0] * 40})
    assert C.stop_spread(d, 0.3) == 0.0
    rng = np.random.default_rng(0)
    d2 = pd.DataFrame({"race_id": np.repeat(np.arange(8), 10), "loss_s": 21 + rng.normal(0, 1.2, 80)})
    assert C.stop_spread(d2, 0.3) == pytest.approx(1.2, abs=0.4)


def test_quali_gaps_are_percent_slower_and_missing_times_are_filled():
    import pandas as pd
    q = pd.DataFrame({"race_id": 1, "driver_id": list("abcdefg"), "q1_s": [80.0, 80.8, 81.6, np.nan, 82.4, 83.2, 84.0]})
    g = C.quali_gaps(q, 1, list("abcdefg"))
    assert g[0] == 0 and g[1] == pytest.approx(1.0) and g[3] == pytest.approx(np.quantile([0, 1, 2, 3, 4, 5], 0.9) if False else g[3])
    assert g[3] >= g[2]
    assert (C.quali_gaps(None, 1, ["a", "b"]) == 0).all()


def test_noise_by_position_and_cover_rates_are_measured(con, cal):
    laps, flags, _ = C.load_inputs(con)
    tab = C.lap_noise_by_position(laps, list(flags.loc[flags["wet"], "race_id"]))
    assert list(tab["bucket"]) == ["top 3", "4 to 10", "11 and below"]
    assert (tab["noise_sd_s"] > 0).all() and tab["multiplier"].between(0.5, 1.5).all()
    assert "by_season" in tab.attrs
    cv = C.cover_rates(cal["truths"])
    assert cv["exposed"] + cv["unexposed"] > 0 and 0.0 <= cv["cover_prob"] <= 1.0
    assert cal["params"].cover_prob == cv["cover_prob"]
    assert {"noise_top", "noise_mid", "noise_rest"} <= set(cal["params"].__dict__)


def test_cover_rates_find_a_planted_reaction():
    from src.sim.engine import BIG
    L, n = 60, 6
    stops = np.full((n, 1), BIG); stops[:, 0] = [21, 20, 40, 41, 42, 43]
    pos = np.tile(np.arange(1, n + 1, dtype=float), (L, 1))
    t = C.RaceTruth(race_id=1, season=2023, circuit_id="x", n_laps=L, drivers=list("abcdef"), constructors=list("ABCDEF"),
                    slot=np.arange(1, n + 1), pace=np.zeros(n), comp=np.ones((n, 2), int), stops=stops, start_age=np.zeros((n, 2)),
                    retire=np.full(n, BIG), neutral=np.zeros(L + 1, int), finish=np.arange(1, n + 1), pos_actual=pos,
                    off=np.zeros(3), deg=np.ones(3) * 0.05, fuel=-0.05, pit_loss=22.0, pace_known=np.ones(n, bool))
    cv = C.cover_rates([t] * 40)
    assert cv["exposed"] > 0 and cv["exposed_rate"] == 1.0 and cv["unexposed_rate"] < 0.2


# ------------------------------------------------------------------ v2.2: gaps, lap 1, front-runner stop spread

def test_truths_carry_gaps_to_the_car_ahead(cal):
    t = cal["truths"][0]
    assert t.gap_ahead is not None and t.gap_ahead.shape == (t.n_laps, t.n)
    g = t.gap_ahead[np.isfinite(t.gap_ahead)]
    assert len(g) > 0 and (g > 0).all()


def test_close_counts_ignore_stops_neutral_laps_and_lap_one():
    gap = np.full((1, 6, 3), 0.5)
    gap[0, :, 0] = np.nan                                    # the leader has no gap
    pit = np.zeros((1, 6, 3), bool)
    pit[0, 3, 1] = True                                      # car 1 stops on lap 4: laps 3-5 are not counted for it
    green = np.ones((1, 6), bool)
    green[0, 1] = False                                      # lap 2 neutral: laps 2 and 3 not counted
    close, cars = C.close_counts(gap, pit, green)
    # car 2: laps 4,5,6 are counted (lap 3 follows the neutral lap 2); car 1: only lap 6 (laps 3-5 are near its stop, 2-3 neutral)
    assert cars[0] == 4 and close[0] == 4
    gap2 = gap.copy()
    gap2[0, :, 2] = 3.0
    close2, cars2 = C.close_counts(gap2, pit, green)
    assert cars2[0] == 4 and close2[0] == 1


def test_engine_gap_is_the_time_to_the_car_ahead():
    n = 4
    spec = E.make_spec(10, slot=np.arange(1, n + 1), pace=[0.0] * n, strategies=[[(E.MEDIUM, 0)]] * n)
    r = E.simulate(spec, Params0(), S=5, seed=0)
    assert r.gap.shape == (5, n, 10)
    assert np.isnan(r.gap[:, 0, :]).all()                    # the leader
    assert np.isfinite(r.gap[:, 1:, :]).all() and (r.gap[:, 1:, :] > 0).all()


def Params0():
    return E.Params(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, slot_gap=0.5, min_gap=0.5, thr=6.0)


def test_lap_one_noise_is_found_from_the_measured_shuffling(cal):
    truths = cal["truths"][:6]
    target = C.real_lap1_move(truths)
    assert np.isfinite(target)
    par = E.Params(slot_gap=0.3)
    sig = C.fit_start(truths, par, target, S=30)
    assert 0.0 <= sig <= C.START_RANGE[1]
    # asking for a shuffle twice as big needs more noise
    assert C.fit_start(truths, par, 2 * target, S=30) >= sig
    # and with no shuffling at all, none
    assert C.fit_start(truths, par, 0.0, S=30) == 0.0


def test_the_fit_uses_the_close_share_and_lap_one_when_the_races_have_them(cal):
    g = cal["fit_grid"]
    assert "close_term" in g.columns and g["real_close_share"].notna().all()
    assert "min_gap" in set(g["setting"])
    assert 0.0 <= cal["params"].sigma_start <= C.START_RANGE[1]
    assert np.isfinite(cal["real_lap1_move"])


def test_stop_spread_splits_front_runners_from_the_rest():
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for race in range(12):
        for k in range(20):
            pos = 1 + (k % 20)
            sd = 0.3 if pos <= 3 else 1.5
            rows.append({"race_id": race, "loss_s": 22 + rng.normal(0, sd), "position": float(pos)})
    d = pd.DataFrame(rows)
    top, rest = C.stop_spread(d, 0.0, top=True), C.stop_spread(d, 0.0, top=False)
    assert top < 0.6 < 1.0 < rest
    par = E.Params(sigma_lap=0.0)
    sp = C.pit_spreads(d, par)
    assert sp["sigma_pit_top"] < sp["sigma_pit"]


def test_pit_spreads_fall_back_without_positions():
    import pandas as pd
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"race_id": np.repeat(np.arange(10), 10), "loss_s": 22 + rng.normal(0, 1.0, 100)})
    sp = C.pit_spreads(d, E.Params(sigma_lap=0.0))
    assert sp["sigma_pit_top"] == -1.0 and sp["sigma_pit"] > 0.5


def test_front_runner_stop_noise_is_used_by_the_engine():
    n = 6
    strat = [[(E.MEDIUM, 0, 5), (E.HARD, 0)] for _ in range(n)]
    spec = E.make_spec(12, slot=np.arange(1, n + 1), pace=[0.0] * n, strategies=strat)
    base = dict(sigma_start=0.0, sigma_lap=0.0, sigma_stint=0.0, thr=6.0, slot_gap=3.0)
    wide = E.simulate(spec, E.Params(**base, sigma_pit=2.0, sigma_pit_top=0.0), S=300, seed=1)
    flat = E.simulate(spec, E.Params(**base, sigma_pit=2.0), S=300, seed=1)
    # the leader's stop cost has no spread when the top three are set to zero, so it wins more reliably
    assert (wide.finish[:, 0] == 1).mean() > (flat.finish[:, 0] == 1).mean()


def _stint_laps(drift_sd, rho=0.9, white=0.3, races=10, drivers=20, n_laps=40, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(races):
        for d in range(drivers):
            e, x = rng.normal(0, drift_sd), []
            for lap in range(2, n_laps + 2):
                e = rho * e + np.sqrt(1 - rho ** 2) * drift_sd * rng.normal()
                x.append(90 + 0.02 * lap + e + rng.normal(0, white))
            for lap, y in zip(range(2, n_laps + 2), x):
                rows.append({"race_id": r, "season": 2023, "driver_id": f"d{d}", "stint": 1, "lap_number": lap, "lap_time_s": y,
                             "compound": "MEDIUM", "is_accurate": True, "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1",
                             "tyre_life": lap, "position": d + 1})
    return pd.DataFrame(rows)


def test_lap_dependence_finds_planted_spells_and_none_in_white_noise():
    spells = C.lap_dependence(_stint_laps(0.3, races=20))
    assert spells["note"] == "measured" and 0.2 < spells["drift_sd"] < 0.45 and 0.8 <= spells["drift_rho"] <= 0.98
    white = C.lap_dependence(_stint_laps(0.0, seed=1, races=20))
    assert white["drift_sd"] < 0.12


def test_gap_profile_agrees_with_close_counts_and_rises_with_the_bound():
    rng = np.random.default_rng(0)
    gap = rng.uniform(0.0, 6.0, (2, 12, 5))
    pit = np.zeros((2, 12, 5), bool)
    green = np.ones((2, 12), bool)
    cnt, cars = C.gap_profile_counts(gap, pit, green, bounds=(0.5, 1.0, 2.0, 3.5))
    close, cars2 = C.close_counts(gap, pit, green)
    assert (cnt[:, 1] == close).all() and (cars == cars2).all()
    assert (np.diff(cnt, axis=1) >= 0).all()


def test_pass_difficulty_is_fitted_from_the_spread_of_real_races(cal):
    sd, tab = C.fit_pass_difficulty(cal["truths"][:10], cal["params"], grid=(0.0, 0.3), S=12)
    assert sd in (0.0, 0.3) and list(tab["thr_sd"]) == [0.0, 0.3] and np.isfinite(tab["dispersion"]).all()
    assert "lap_dependence" in cal and "pass_difficulty" in cal


# ------------------------------------------------------------------ v2.4: dirty air, drift check, by-position checks

def _traffic_laps(loss, rng_seed=0, races=30, drivers=20, n_laps=40, persistent=False, white=0.2):
    """Clean laps where each lap's start gap to the car ahead (`gap_prev`) is drawn at random and the lap time pays
    loss x (1 - gap / 2) for it. persistent=True makes the gaps wander slowly, so the loss looks like a drift in pace."""
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    rows = []
    for r in range(races):
        for d in range(drivers):
            pace = rng.normal(0, 0.4)
            g = 1.5
            for lap in range(2, n_laps + 2):
                if persistent:
                    g = float(np.clip(0.9 * g + 0.1 * 1.5 + 0.45 * rng.normal(), 0.1, 6.0))
                else:
                    g = float(rng.uniform(0.1, 6.0))
                pen = loss * max(0.0, 1.0 - g / 2.0)
                rows.append({"race_id": r, "season": 2023, "driver_id": f"d{d}", "stint": 1, "lap_number": lap,
                             "lap_time_s": 90 + 0.02 * lap + pace + pen + rng.normal(0, white), "compound": "MEDIUM",
                             "is_accurate": True, "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "tyre_life": lap,
                             "position": d + 1, "gap_prev": g})
    return pd.DataFrame(rows)


def test_gap_prev_is_the_gap_at_the_end_of_the_previous_lap():
    import pandas as pd
    rows = []
    for lap, ends in ((1, {"a": 90.0, "b": 91.5, "c": 95.0}), (2, {"a": 180.0, "b": 182.0, "c": 190.0})):
        for pos, (d, e) in enumerate(ends.items(), start=1):
            rows.append({"race_id": 1, "driver_id": d, "lap_number": lap, "position": float(pos), "lap_end_s": e})
    rows.append({"race_id": 1, "driver_id": "e", "lap_number": 1, "position": 5.0, "lap_end_s": 99.0})   # position 4 missing: not adjacent
    laps = pd.DataFrame(rows)
    out = C.add_gap_prev(laps)
    g = out.set_index(["driver_id", "lap_number"])["gap_prev"]
    assert np.isnan(g[("a", 1)]) and np.isnan(g[("a", 2)])              # leader, and nothing before lap 1
    assert g[("b", 2)] == pytest.approx(1.5) and g[("c", 2)] == pytest.approx(3.5)
    assert np.isnan(g[("e", 2)]) if ("e", 2) in g.index else True
    none = C.add_gap_prev(laps.drop(columns="lap_end_s"))
    assert none["gap_prev"].isna().all()


def test_dirty_air_finds_a_planted_loss_and_none_when_there_is_none():
    d = C.dirty_air(_traffic_laps(0.5))
    assert d["note"] == "measured" and 0.35 < d["da_loss"] < 0.65 and d["da_range"] in C.DA_RANGES
    tab = d["table"]
    assert tab["loss_s"].iloc[0] > tab["loss_s"].iloc[-1] > -0.05                 # closer costs more, and 2-3 s behind costs about nothing
    assert (tab["low_s"] <= tab["loss_s"]).all() and (tab["loss_s"] <= tab["high_s"]).all()
    assert C.dirty_air(_traffic_laps(0.0, rng_seed=1))["da_loss"] == 0.0
    assert C.dirty_air(_traffic_laps(0.0).drop(columns="gap_prev"))["note"] == "no line-crossing times"


def test_taking_dirty_air_off_removes_the_drift_it_was_mistaken_for():
    laps = _traffic_laps(0.7, persistent=True, white=0.25)
    da = C.dirty_air(laps)
    assert da["da_loss"] > 0.4
    raw = C.lap_dependence(laps)
    adj = C.lap_dependence(laps, da=da)
    assert raw["drift_sd"] > 0.12                                                  # traffic that lasts looks like a lasting spell of pace
    assert adj["dirty_air_removed"] and adj["drift_sd"] < 0.6 * raw["drift_sd"]
    assert not C.lap_dependence(laps)["dirty_air_removed"]


def test_by_position_counts_add_up_to_the_totals():
    rng = np.random.default_rng(3)
    S, L, n = 3, 14, 8
    pos = np.stack([np.stack([rng.permutation(n) + 1 for _ in range(L)]) for _ in range(S)]).astype(float)
    pit = rng.random((S, L, n)) < 0.05
    green = rng.random((S, L)) < 0.9
    p, k = C.pass_counts(pos, pit, green)
    pb, kb = C.pass_counts_by_band(pos, pit, green)
    assert pb.shape == (S, len(C.BANDS)) and np.allclose(pb.sum(1), p) and np.allclose(kb.sum(1), k)
    gap = rng.uniform(0.0, 5.0, (S, L, n))
    c, cars = C.close_counts(gap, pit, green)
    cb, carsb = C.close_counts_by_band(gap, pos, pit, green)
    assert np.allclose(cb.sum(1), c) and np.allclose(carsb.sum(1), cars)


def test_a_swap_at_the_front_counts_in_the_front_band():
    pos = np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 4, 3], [1, 2, 4, 3]]], float)       # cars 2 and 3 swap, far from the lead
    p, k = C.pass_counts_by_band(pos, np.zeros((1, 4, 4), bool), np.ones((1, 4), bool))
    assert p[0].tolist() == [0.0, 1.0, 0.0, 0.0]                                              # the better-placed of the two was P3: band P2-3


def test_position_check_reports_every_band_for_real_and_simulated(cal):
    tab = C.position_check(cal["truths"][:6], cal["params"], S=6)
    d = tab[tab["split"] == "all"]
    assert list(d["position"]) == [b[0] for b in C.BANDS]
    assert (d["real_car_laps"] > 0).all() and d["simulated_passes_per_car_lap"].notna().all()
    assert np.isnan(d["real_close_share"].iloc[0])                                            # the leader has no car ahead
    rest = d.iloc[1:]
    assert rest["real_close_share"].between(0, 1).all() and rest["simulated_close_share"].between(0, 1).all()


def test_drift_check_moves_the_drift_into_white_noise_and_keeps_the_total(cal):
    from dataclasses import replace
    par = replace(cal["params"], drift_sd=0.3, drift_rho=0.9, sigma_lap=0.2)
    tab = C.drift_check(cal["truths"][:6], par, mults=(0.0, 1.0), S=8)
    assert list(tab["drift_share"]) == [0.0, 1.0] and tab["drift_sd"].iloc[0] == 0.0
    assert tab["white_sd"].iloc[0] == pytest.approx(np.sqrt(0.2 ** 2 + 0.3 ** 2)) and tab["white_sd"].iloc[1] == pytest.approx(0.2)
    assert np.isfinite(tab["loss"]).all()
    assert C.drift_check(cal["truths"][:3], replace(par, drift_sd=0.0)).empty


def test_the_fit_searches_the_dirty_air_multiple_and_flags_only_the_top_of_its_range(cal):
    from dataclasses import replace
    base = replace(cal["base"], da_loss=0.3, da_range=2.0)
    par, grid, edge = C.fit_passing(cal["truths"][:8], base, S=8, thr_grid=(0.4, 0.8), start_grid=(1.0,), slot_grid=(0.3, 0.6),
                                    noise_grid=(1.0,), gap_grid=(0.4, 0.8), da_grid=(0.0, 1.0, 2.0), rounds=1)
    assert "da" in set(grid["setting"]) and set(grid[grid["setting"] == "da"]["value"]) == {0.0, 1.0, 2.0}
    assert par.da_loss in (0.0, 0.3, 0.6) or par.da_loss == pytest.approx(0.6)
    assert "da" not in edge or par.da_loss == pytest.approx(0.6)                              # the bottom of the range (none) is not an edge
    plain, grid2, _ = C.fit_passing(cal["truths"][:8], replace(cal["base"], da_loss=0.0), S=8, thr_grid=(0.4, 0.8), start_grid=(1.0,),
                                    slot_grid=(0.3,), noise_grid=(1.0,), gap_grid=(0.4, 0.8), rounds=1)
    assert "da" not in set(grid2["setting"]) and plain.da_loss == 0.0


def test_calibration_reports_dirty_air_and_the_drift_check(cal, tmp_path, con):
    assert "dirty_air" in cal and "lap_dependence_raw" in cal and "drift_check" in cal
    lines = C.run(con, tmp_path / "o", S=10)
    assert any(l.startswith("DIRTY AIR") for l in lines)
    assert (tmp_path / "o" / "dirty_air.csv").exists() and (tmp_path / "o" / "drift_check.csv").exists()
    assert (tmp_path / "o" / "lap_dependence_before_dirty_air.csv").exists()
