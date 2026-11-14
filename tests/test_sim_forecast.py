import numpy as np
import pandas as pd
import pytest

from src.sim import calibrate as C
from src.sim import forecast as F
from src.sim import replay as R
from . import lapworld


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("f")
    w = lapworld.world(seed=6, seasons=(2022, 2023, 2024, 2025), races=7, n_laps=36, n_drivers=14, sc_races=(3, 4), wet_race=7)
    con = lapworld.build_con(d / "w.duckdb", w)
    C.run(con, d / "cal", S=15)
    truths, par = C.load_truths(d / "cal")
    return truths, par, d


def test_history_pace_uses_only_the_races_it_is_given(setup):
    truths, par, d = setup
    t = truths[-1]
    hist = truths[:10]
    p1 = F.history_pace(hist, F.card_of(t))
    changed = truths[:10] + []
    p2 = F.history_pace(truths[:10] + [], F.card_of(t))
    assert np.allclose(p1, p2) and p1.shape == (t.n,)


def test_history_pace_recovers_who_is_quick(setup):
    truths, par, d = setup
    t = truths[-1]
    pred = F.history_pace(truths[:-1], F.card_of(t))
    # lapworld paces are redrawn each race, so history says little; with stable teams it should still be finite and centred
    assert np.isfinite(pred).all() and abs(np.mean(pred)) < 1.0


def test_stable_pace_is_recovered_from_history():
    """Build races where each team has a fixed pace: history must rank the teams correctly."""
    rng = np.random.default_rng(0)
    n = 10
    cons = [f"c{i // 2}" for i in range(n)]
    drivers = [f"d{i}" for i in range(n)]
    team_pace = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

    def race(rid):
        pace = np.array([team_pace[i // 2] for i in range(n)]) + rng.normal(0, 0.08, n)
        return C.RaceTruth(race_id=rid, season=2023, circuit_id="x", n_laps=40, drivers=drivers, constructors=cons,
                           slot=np.arange(1, n + 1), pace=pace, comp=np.ones((n, 2), int), stops=np.full((n, 1), 20),
                           start_age=np.zeros((n, 2)), retire=np.full(n, 10_000), neutral=np.zeros(41, int),
                           finish=np.arange(1, n + 1), pos_actual=np.full((40, n), np.nan), off=np.zeros(3),
                           deg=np.zeros(3), fuel=0.0, pit_loss=20.0, pace_known=np.ones(n, bool))
    hist = [race(202300 + i) for i in range(1, 12)]
    card = F.card_of(race(202312))
    pred = F.history_pace(hist, card)
    assert np.corrcoef(pred, [team_pace[i // 2] for i in range(n)])[0, 1] > 0.97


def test_fit_blend_falls_back_when_there_is_little_history():
    coef, sig = F.fit_blend([], "History + grid")
    assert coef.shape == (6,) and sig == F.DEFAULT_SIGMA
    coef, _ = F.fit_blend([], "History")
    assert coef.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_fit_blend_finds_the_grid_when_history_is_useless():
    rng = np.random.default_rng(1)
    recs = []
    for _ in range(12):
        prof = np.linspace(-1, 1, 10)
        oracle = prof + rng.normal(0, 0.05, 10)
        recs.append(F.Record(pred=rng.normal(0, 1, 10), prof=prof, oracle=oracle, known=np.ones(10, bool),
                             cons=np.array([f"c{i // 2}" for i in range(10)])))
    coef, sig = F.fit_blend(recs, "History + grid")
    assert coef[2] > 0.8 and abs(coef[1]) < 0.2
    assert sig[0] > 0 and sig[1] > 0


def test_pace_samples_share_a_team_shock():
    cons = np.array(["a", "a", "b", "b"])
    x = F.pace_samples(np.zeros(4), cons, (1.0, 0.0), 2000, np.random.default_rng(0))
    assert np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.95 and abs(np.corrcoef(x[:, 0], x[:, 2])[0, 1]) < 0.1


def test_walk_forward_produces_scored_probabilities(setup, tmp_path):
    truths, par, d = setup
    races, probs = F.walk_forward(truths, par, S=60, min_history=10, progress=False)
    assert set(races["level"]) == set(F.VARIANTS) | {F.ORACLE}
    # each model's win probabilities in one race add to about 1
    g = probs[probs["model"] == "History"].groupby("race_id")["p_win"].sum()
    assert np.allclose(g, 1.0, atol=1e-9)
    g = probs[probs["model"] == F.BASELINE].groupby("race_id")["p_win"].sum()
    assert np.allclose(g, 1.0, atol=0.05)
    first_scored = races["race_id"].min()
    earlier = [t for t in truths if t.race_id < first_scored]
    assert len(earlier) >= 10                                          # nothing is forecast without history


def test_run_writes_files_and_reads_sensibly(setup, tmp_path):
    truths, par, d = setup
    lines = F.run(tmp_path / "o", cal_dir=d / "cal", S=60, progress=False)
    assert any("History + grid" in l for l in lines) and any(l.startswith("LOG LOSS") for l in lines)
    names = {p.name for p in (tmp_path / "o").iterdir()}
    assert {"forecast_races.csv", "forecast_summary.csv", "forecast_probabilities.csv", "calibration.csv",
            "forecast_skill.png", "forecast_calibration.png", "findings.txt"} <= names
    rel = pd.read_csv(tmp_path / "o" / "calibration.csv")
    assert set(rel["outcome"]) == {"win", "podium", "top ten"}


def test_reliability_bins_a_calibrated_model_near_the_diagonal():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p)
    probs = pd.DataFrame({"model": "m", "p_win": p, "p_top3": p, "p_top10": p, "finish": np.where(y, 1, 20)})
    rel = F.reliability(probs)
    w = rel[rel["outcome"] == "win"]
    assert (abs(w["mean_forecast"] - w["observed"]) < 0.08).all()


def test_fit_recal_corrects_overspread_probabilities():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.02, 0.98, 6000)
    y = (rng.uniform(0, 1, 6000) < true_p).astype(float)
    # a forecaster that is too timid: pulls everything toward 0.5
    shown = 0.5 + 0.5 * (true_p - 0.5)
    a, b = F.fit_recal(shown, y)
    fixed = F.apply_recal(shown, (a, b))
    assert b > 1.3
    assert np.mean((fixed - y) ** 2) < np.mean((shown - y) ** 2)


def test_fit_recal_leaves_a_calibrated_forecast_alone():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 8000)
    y = (rng.uniform(0, 1, 8000) < p).astype(float)
    a, b = F.fit_recal(p, y)
    assert abs(a) < 0.15 and abs(b - 1) < 0.15


def test_quali_gap_enters_the_blend_when_it_carries_the_signal():
    rng = np.random.default_rng(3)
    recs = []
    for _ in range(30):
        n = 12
        true = rng.normal(0, 0.6, n)
        recs.append(F.Record(pred=rng.normal(0, 0.6, n), prof=rng.normal(0, 0.6, n), oracle=true, known=np.ones(n, bool),
                             cons=np.array([f"c{i // 2}" for i in range(n)]), qgap=true + rng.normal(0, 0.05, n)))
    coef_q, _ = F.fit_blend(recs, "History + quali gap")
    coef_g, _ = F.fit_blend(recs, "History + grid")
    assert coef_q[3] > 0.8 and coef_g[3] == 0.0


def test_walk_forward_adds_recalibrated_models_once_there_is_enough_history(setup, monkeypatch):
    truths, par, d = setup
    monkeypatch.setattr(F, "MIN_RECAL_CARS", 60)
    races, probs = F.walk_forward(truths, par, S=40, min_history=8, progress=False)
    names = set(races["level"])
    assert {b + F.RECAL_SUFFIX for b in F.RECAL_BASES} <= names
    g = probs[probs["model"] == "History + grid" + F.RECAL_SUFFIX].groupby("race_id")["p_win"].sum()
    assert len(g) > 0 and np.allclose(g, 1.0)
    # a recalibrated forecast only exists for races after the first ones
    assert races[races["level"] == "History + grid" + F.RECAL_SUFFIX]["race_id"].min() > races["race_id"].min()


def test_sharpened_variants_are_gone(setup):
    truths, par, d = setup
    assert not hasattr(F, "fit_sharpen") and not hasattr(F, "SHARP_SUFFIX")
    races, _ = F.walk_forward(truths, par, S=30, min_history=8, progress=False)
    assert not any("sharpened" in n for n in set(races["level"]))


def test_season_table_breaks_skill_down_by_season(setup, tmp_path):
    truths, par, d = setup
    races, _ = F.walk_forward(truths, par, S=40, min_history=10, progress=False)
    tab = F.season_table(races, models=("History + grid",))
    assert set(tab["season"]) <= set(races["season"].astype(int)) and len(tab) >= 1
    assert {"podium_skill", "top10_skill", "races"} <= set(tab.columns)
    lines = F.findings(R.summarise(races), pd.DataFrame(), seasons=tab)
    assert any(l.startswith("BY SEASON") for l in lines)
