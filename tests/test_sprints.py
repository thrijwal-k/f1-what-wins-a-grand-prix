import numpy as np
import pandas as pd
import pytest

from src.analysis import sprints as sp
from . import lapworld

SPRINT_ROUNDS = (2, 4)


def sprint_world(seed=1, seasons=(2021, 2022, 2023, 2024), races=6, n_laps=30, shuffle=0.35, n_drivers=14):
    """Lap world plus sprint results on rounds 2 and 4: a noisy copy of the Sunday order, own grid."""
    w = lapworld.world(seed=seed, seasons=seasons, races=races, n_laps=n_laps, n_drivers=n_drivers,
                       sc_races=(3,), wet_race=5)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for rid, g in w["results"][w["results"]["race_id"] % 100 == 0].groupby("race_id"):
        pass
    for rid, g in w["results"].groupby("race_id"):
        if rid % 100 not in SPRINT_ROUNDS:
            continue
        score = g["position"].to_numpy() + rng.normal(0, shuffle * len(g) / 3, len(g))
        order = np.argsort(score)
        pos = np.empty(len(g), int)
        pos[order] = np.arange(1, len(g) + 1)
        for (_, r), p in zip(g.iterrows(), pos):
            rows.append({"race_id": rid, "season": r["season"], "round": r["round"], "driver_id": r["driver_id"],
                         "constructor_id": r["constructor_id"], "grid": int(r["grid"]), "position": int(p),
                         "classified": bool(p <= len(g) - 2)})
    w["sprint_results"] = pd.DataFrame(rows)
    return w


@pytest.fixture(scope="module")
def sw():
    return sprint_world()


@pytest.fixture()
def con(tmp_path, sw):
    c = lapworld.build_con(tmp_path / "s.duckdb", sw)
    yield c
    c.close()


def test_load_sprints_and_empty(con, tmp_path):
    df = sp.load_sprints(con)
    assert df["race_id"].nunique() == 8 and df["position"].notna().all()
    con2 = lapworld.build_con(tmp_path / "e.duckdb", {k: v for k, v in sprint_world().items() if k != "sprint_results"})
    assert sp.load_sprints(con2).empty
    con2.close()


def test_sprint_grid_ids_rule():
    ids = {202102, 202204, 202302, 202404}
    assert sp.sprint_grid_ids(ids) == {202102, 202204}


def test_table_and_summary(con):
    from src.analysis import grid
    df = grid.prepare(sp.load_sprints(con))
    rel = sp.reliability(df)
    table = sp.sprint_table(df, rel)
    assert len(table) == 8
    summ = sp.sprint_summary(table)
    assert len(summ) >= 1


def test_run_writes_everything(con, tmp_path):
    lines = sp.run(con, tmp_path / "out", reps=40, perms=60, n_boot=20)
    assert lines and any(l.strip() for l in lines)
    names = {p.name for p in (tmp_path / "out").iterdir()}
    for f in ["findings.txt", "sprint_races.csv", "sprint_summary.csv", "sprint_vs_sunday.csv",
              "sunday_race_measures.csv", "sprint_weekend_effect.csv", "car_driver_with_sprints.csv",
              "teammate_agreement.csv"]:
        assert f in names, f
    m = pd.read_csv(tmp_path / "out" / "sunday_race_measures.csv")
    # grid-based measures are blank for the sprint-grid races (2021-22 sprint weekends)
    grid_races = m[m["race_id"].isin([202102, 202104, 202202, 202204])]
    assert grid_races["pole_won"].isna().all()


def test_no_sprint_data(tmp_path):
    w = {k: v for k, v in sprint_world().items() if k != "sprint_results"}
    c = lapworld.build_con(tmp_path / "n.duckdb", w)
    lines = sp.run(c, tmp_path / "o")
    c.close()
    assert any("run_all" in l for l in lines)


def test_weekend_effect_planted_and_null():
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2021, 2026):
        for i in range(24):
            sprint = i % 3 == 0
            rows.append({"race_id": season * 100 + i, "season": season, "circuit_id": f"c{i % 8}",
                         "sprint": sprint, "finish_rate": 0.8 + rng.normal(0, 0.02),
                         "stops_per_car": 1.8 + rng.normal(0, 0.05) - (0.5 if sprint else 0)})
    m = pd.DataFrame(rows)
    for col in sp.EFFECT_METRICS:
        if col[0] not in m:
            m[col[0]] = np.nan
    eff = sp.weekend_effect(m, reps=100, perms=200)
    e = eff.set_index("key") if "key" in eff.columns else eff
    assert len(eff) >= 2
    s = eff[eff.astype(str).apply(lambda r: "stops_per_car" in " ".join(r.astype(str)), axis=1)].iloc[0]
    assert float(s["effect"]) < -0.3


def test_paired_excludes_sprint_grid_races(con):
    from src.analysis import grid, grid_checks
    sprint_df = grid.prepare(sp.load_sprints(con))
    srel = sp.reliability(sprint_df)
    raw = grid_checks.load(con)
    sun = sp.reliability(grid_checks.prepare_checks(raw, drop_indy=True))
    ids = set(srel["race_id"])
    full = sp.paired_comparison(srel, sun, None, set(), 30)
    cut = sp.paired_comparison(srel, sun, None, sp.sprint_grid_ids(ids), 30)
    assert full["weekends"].max() == 8 and cut["weekends"].max() == 4
    assert "q_value" in cut.columns and (cut["q_value"] >= cut["p_value"] - 1e-12).all()


def test_fe_effect_recovers_planted_effect():
    rng = np.random.default_rng(1)
    n = 200
    season = rng.integers(2021, 2025, n)
    circuit = rng.integers(0, 6, n)
    treat = (rng.random(n) < 0.3).astype(float)
    y = 0.4 * treat + 0.2 * (season - 2021) + 0.1 * circuit + rng.normal(0, 0.05, n)
    X = sp._design(season, circuit)
    assert sp.fe_effect(y, treat, X) == pytest.approx(0.4, abs=0.05)


def test_teammate_agreement_high_when_drivers_consistent():
    rng = np.random.default_rng(2)
    skill = {f"d{i}": (1 if i % 2 == 0 else -1) * (0.5 + i / 20) for i in range(10)}

    def results(rounds, base):
        rows = []
        for season in (2021, 2022, 2023):
            for rnd in rounds:
                for team in range(5):
                    for k in (0, 1):
                        d = f"d{team * 2 + k}"
                        rows.append({"race_id": season * 100 + rnd, "season": season, "driver_id": d,
                                     "constructor_id": f"c{team}", "grid": 1,
                                     "score": base + team * 4 - skill[d] * 3 + rng.normal(0, 0.5)})
        df = pd.DataFrame(rows)
        df["position"] = df.groupby("race_id")["score"].rank(method="first").astype(int)
        return df.drop(columns="score")

    sunday = results(range(1, 9), 0)
    sprint = results((2, 4, 6), 0)
    q = pd.DataFrame(columns=["race_id", "driver_id", "position"])
    from src.analysis import car_driver
    srows = car_driver.build_rows(sunday, q)
    srows["weekend"] = srows["race_id"]
    prows = sp.sprint_rows(sprint)
    assert len(sp.teammate_gaps(prows)) > 0
    out = sp.teammate_agreement(srows, prows, reps=40)
    same = out.iloc[0]
    assert same["value"] > 0.8 and same["pairs"] > 10
