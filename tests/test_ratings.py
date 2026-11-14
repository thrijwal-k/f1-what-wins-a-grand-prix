import duckdb
import numpy as np
import pandas as pd

from src.analysis import ratings


# ----------------------------------------------------------- simulated world

def fake_world(seed=11, n_pool=40, seasons=range(1994, 2016), rounds=8, n_teams=8):
    """Drivers with known skill, teams with known car strength, qualifying times generated from both."""
    rng = np.random.default_rng(seed)
    skill = {f"d{i}": rng.normal(0, 0.012) for i in range(n_pool)}
    drivers = list(skill)
    res_rows, q_rows = [], []
    for season in seasons:
        active = list(rng.choice(drivers, size=2 * n_teams, replace=False))
        rng.shuffle(active)
        car = {t: rng.normal(0, 0.02) for t in range(n_teams)}
        for rnd in range(1, rounds + 1):
            race_id = season * 100 + rnd
            times = {}
            for j, d in enumerate(active):
                team = j // 2
                base = 80 * np.exp(car[team] - skill[d] + rng.normal(0, 0.004))
                times[d] = (team, base)
            order = sorted(times, key=lambda d: times[d][1])
            top14 = set(order[:14])
            top10 = set(order[:10])
            for pos, d in enumerate(order, start=1):
                team, t = times[d]
                fmt = lambda x: f"{int(x // 60)}:{x % 60:06.3f}"
                q2 = fmt(t * np.exp(rng.normal(0, 0.003))) if d in top14 else None
                q3 = fmt(t * np.exp(rng.normal(0, 0.003))) if d in top10 else None
                q_rows.append({"race_id": race_id, "driver_id": d, "constructor_id": f"c{team}",
                               "position": pos, "q1": fmt(t), "q2": q2, "q3": q3})
                res_rows.append({"race_id": race_id, "season": season, "round": rnd, "driver_id": d,
                                 "constructor_id": f"c{team}", "grid": pos, "position": pos})
    names = pd.DataFrame({"driver_id": drivers, "given_name": drivers, "family_name": "Test"})
    return pd.DataFrame(res_rows), pd.DataFrame(q_rows), names, skill


# ----------------------------------------------------------- observations

def test_rank_obs_sign_ties_and_quali_preference():
    results = pd.DataFrame({
        "race_id": [195001] * 4 + [200001] * 2, "season": [1950] * 4 + [2000] * 2,
        "constructor_id": ["x", "x", "y", "y", "z", "z"],
        "driver_id": ["a", "b", "c", "d", "e", "f"], "grid": [1, 3, 5, 5, 9, 2]})  # c, d share (tie)
    quali = pd.DataFrame({"race_id": [200001] * 2, "driver_id": ["e", "f"], "constructor_id": ["z", "z"],
                          "position": [1, 2], "q1": None, "q2": None, "q3": None})
    obs = ratings.build_rank_obs(results, quali)
    assert len(obs) == 2  # the tied pair c/d is dropped
    old = obs[obs["race_id"] == 195001].iloc[0]
    assert (old["a"], old["b"]) == ("a", "b") and old["y"] > 0   # a on grid 1 ahead of b on 3
    new = obs[obs["race_id"] == 200001].iloc[0]
    assert (new["a"], new["b"]) == ("e", "f") and new["y"] > 0   # qualifying 1 vs 2 beats grid 9 vs 2


def test_teams_with_three_cars_are_ignored():
    results = pd.DataFrame({
        "race_id": [196001] * 5, "season": [1960] * 5, "constructor_id": ["x", "x", "x", "y", "y"],
        "driver_id": ["a", "b", "c", "d", "e"], "grid": [1, 2, 3, 4, 5]})
    quali = pd.DataFrame(columns=["race_id", "driver_id", "constructor_id", "position", "q1", "q2", "q3"])
    obs = ratings.build_rank_obs(results, quali)
    assert list(zip(obs["a"], obs["b"])) == [("d", "e")]


def test_time_obs_uses_latest_common_stage_and_drops_outliers():
    q = pd.DataFrame({
        "race_id": [1] * 6, "constructor_id": ["x", "x", "y", "y", "z", "z"],
        "driver_id": ["a", "b", "c", "d", "e", "f"],
        "q1": ["1:20.000", "1:20.800", "1:20.000", "1:20.000", "1:20.000", "1:40.000"],
        "q2": ["1:19.000", "1:19.500", "1:19.000", None, None, None],
        "q3": ["1:18.000", None, None, None, None, None]})
    obs = ratings.build_time_obs(q).set_index("a")
    assert abs(obs.loc["a", "y"] - np.log(79.5 / 79.0)) < 1e-9   # a vs b: both in Q2, not Q1
    assert abs(obs.loc["c", "y"]) < 1e-9                         # c vs d: identical Q1 times, zero gap
    assert "e" not in obs.index                                  # 25% gap dropped as an outlier
    assert (obs["y"] >= 0).all()


# ------------------------------------------------------------------ model

def test_model_recovers_known_skills():
    results, quali, names, skill = fake_world()
    obs = ratings.build_time_obs(quali)
    fit, noise, prior = ratings.fit_ratings(obs)
    truth = pd.Series(skill)
    est = fit.set_index("driver_id")
    common = est.index.intersection(truth.index)
    corr = np.corrcoef(est.loc[common, "rating"], truth[common] - truth[common].mean())[0, 1]
    assert corr > 0.9
    centred = truth[common] - truth[common].mean()
    mean_shift = (est.loc[common, "rating"] - centred).mean()
    cover = (np.abs(est.loc[common, "rating"] - mean_shift - centred) <= 1.96 * est.loc[common, "sd"]).mean()
    assert cover > 0.85
    assert 0.005 < prior < 0.02 and 0.002 < noise < 0.02


def test_more_data_gives_narrower_intervals():
    _, quali, _, _ = fake_world(seasons=range(1994, 2016), rounds=8)
    few = ratings.build_time_obs(quali[quali["race_id"] % 100 <= 2])
    many = ratings.build_time_obs(quali)
    sd_few = ratings.fit_ratings(few)[0]["sd"].mean()
    sd_many = ratings.fit_ratings(many)[0]["sd"].mean()
    assert sd_many < sd_few


def test_rank_model_orders_drivers_like_the_truth():
    results, quali, names, skill = fake_world(seed=5)
    obs = ratings.build_rank_obs(results, quali)
    fit = ratings.fit_ratings(obs)[0].set_index("driver_id")
    truth = pd.Series(skill)
    common = fit.index.intersection(truth.index)
    assert np.corrcoef(fit.loc[common, "rating"], truth[common])[0, 1] > 0.85


# ------------------------------------------------------------- end to end

def test_full_run_writes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(ratings, "RANK_LAST_SEASON", 2015)  # the fake world only has 1994-2015
    results, quali, names, skill = fake_world(seed=3)
    con = duckdb.connect(str(tmp_path / "fake.duckdb"))
    con.register("r_df", results); con.register("q_df", quali); con.register("n_df", names)
    con.execute("CREATE TABLE results AS SELECT * FROM r_df")
    con.execute("CREATE TABLE qualifying AS SELECT * FROM q_df")
    con.execute("CREATE TABLE drivers AS SELECT * FROM n_df")
    races = pd.DataFrame({"race_id": sorted(results["race_id"].unique())})
    races["season"] = races["race_id"] // 100
    races["circuit_id"] = "somewhere"
    con.register("ra_df", races)
    con.execute("CREATE TABLE races AS SELECT * FROM ra_df")
    out = tmp_path / "out"
    lines = ratings.run(con, out)
    for name in ["ratings_rank_model_1950_93.csv", "ratings_time_model_1994_on.csv", "teammate_head_to_head.csv",
                 "ratings_rank_model_1950_93.png", "ratings_time_model_1994_on.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    table = pd.read_csv(out / "ratings_time_model_1994_on.csv")
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert (table["ci_low"] < table["rating"]).all() and (table["rating"] < table["ci_high"]).all()
    assert any("Time model" in l for l in lines) and any("Rank model" in l for l in lines)
