import math
import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import forecast as F
from src.sim import practice as P
from . import lapworld


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("p")
    w = lapworld.world(seed=9, seasons=(2022, 2023, 2024, 2025), races=7, n_laps=34, n_drivers=14, sc_races=(3,), wet_race=7)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=12)
    truths, par = C.load_truths(d / "cal")
    return con, truths, par, w


def test_long_runs_keep_only_stints_with_enough_clean_laps():
    rows = []
    for stint, n in ((1, 7), (2, 3)):
        for lap in range(1, n + 1):
            rows.append({"driver_id": "a", "stint": stint, "lap_time_s": 90.0 + 0.05 * lap, "compound": "MEDIUM", "tyre_life": lap,
                         "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "session_time_s": 100.0 * lap})
    rows.append({"driver_id": "a", "stint": 1, "lap_time_s": 99.0, "compound": "MEDIUM", "tyre_life": 8, "pit_in_s": np.nan,
                 "pit_out_s": np.nan, "track_status": "1", "session_time_s": 900.0})          # a cool-down lap
    lr = P.long_runs(pd.DataFrame(rows))
    assert set(lr["stint"]) == {1} and len(lr) == 7


def test_session_effects_recover_driver_pace_through_fuel_and_tyres():
    rng = np.random.default_rng(2)
    true = rng.normal(0, 0.4, 10)
    rows = []
    for d in range(10):
        t = float(rng.uniform(0, 3000))
        for stint, comp in ((1, "MEDIUM"), (2, "HARD")):
            for age in range(1, 9):
                lt = 90 + true[d] + {"MEDIUM": 0.0, "HARD": 0.4}[comp] + {"MEDIUM": 0.06, "HARD": 0.03}[comp] * age + rng.normal(0, 0.08)
                t += lt
                rows.append({"driver_id": f"d{d}", "stint": stint, "lap_time_s": lt, "compound": comp, "tyre_life": age,
                             "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "session_time_s": t})
    eff = P.session_effects(pd.DataFrame(rows)).set_index("driver_id")["effect_s"]
    est = np.array([eff[f"d{d}"] for d in range(10)])
    assert np.corrcoef(est, true - np.median(true))[0, 1] > 0.95 and abs(np.median(est)) < 1e-9


def test_session_with_too_few_drivers_gives_nothing():
    d = pd.DataFrame({"driver_id": ["a"] * 6, "stint": 1, "lap_time_s": 90.0, "compound": "MEDIUM", "tyre_life": range(1, 7),
                      "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "session_time_s": 100.0})
    assert P.session_effects(d).empty


def test_fill_uses_team_mate_and_none_without_practice():
    feat = {"a": (0.2, 20, 0.5), "b": (math.nan if False else float("nan"), 0, float("nan")), "c": (-0.1, 15, 0.1)}
    lng, bst, cnt = P.fill(feat, ["a", "b", "c"], ["T1", "T1", "T2"])
    assert lng[1] == pytest.approx(0.2) and bst[1] == pytest.approx(0.5)
    assert P.fill({}, ["a"], ["T1"]) == (None, None, None)
    assert P.fill(None, ["a"], ["T1"]) == (None, None, None)


def test_truths_carry_practice_pace_that_follows_the_race_pace(setup):
    con, truths, par, w = setup
    with_p = [t for t in truths if t.practice_long is not None]
    assert len(with_p) == len(truths)
    r = np.mean([np.corrcoef(t.practice_long, t.pace)[0, 1] for t in with_p])
    assert r > 0.6
    assert with_p[0].practice_best.shape == (with_p[0].n,)


def test_load_practice_without_a_table_is_empty():
    import duckdb
    assert P.load_practice(duckdb.connect(":memory:")).empty and P.features(pd.DataFrame()) == {}


def test_practice_variant_learns_the_practice_weight_and_falls_back_without_it():
    rng = np.random.default_rng(5)
    recs = []
    for _ in range(20):
        n = 12
        true = rng.normal(0, 0.6, n)
        recs.append(F.Record(pred=rng.normal(0, 0.6, n), prof=rng.normal(0, 0.6, n), oracle=true, known=np.ones(n, bool),
                             cons=np.array([f"c{i // 2}" for i in range(n)]), plong=true + rng.normal(0, 0.05, n),
                             pbest=rng.normal(0, 1, n)))
    coef, _ = F.fit_blend(recs, F.PRACTICE)
    assert coef[4] > 0.8 and abs(coef[5]) < 0.15
    assert F.fit_blend([], F.PRACTICE)[0].shape == (6,)
    # records without practice are ignored by the practice variant, so too few of them gives the default blend
    plain = [F.Record(pred=r.pred, prof=r.prof, oracle=r.oracle, known=r.known, cons=r.cons) for r in recs]
    assert F.fit_blend(plain, F.PRACTICE)[0][4] == 0.0


def test_walk_forward_scores_the_practice_variant_and_the_signal_table_reads_sensibly(setup):
    con, truths, par, w = setup
    races, probs = F.walk_forward(truths, par, S=40, min_history=10, progress=False)
    assert F.PRACTICE in set(races["level"])
    g = probs[probs["model"] == F.PRACTICE].groupby("race_id")["p_win"].sum()
    assert len(g) > 0 and np.allclose(g, 1.0)
    sig = F.practice_signal(truths, min_history=10)
    all_row = sig[sig["season"] == 0].iloc[0]
    assert all_row["corr_practice"] > 0.5 and all_row["races"] > 5
    from src.sim import replay as R
    lines = F.findings(R.summarise(races), pd.DataFrame(), signal=sig)
    assert lines[0].startswith("PRACTICE SIGNAL")
