import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis import grid, grid_checks as gc


def world(seed=3, noise=None, seasons=range(1950, 2025), swap=0.3):
    """Races where finishing follows the grid, tightly at monaco, loosely at monza, with Indy 500s before 1961."""
    rng = np.random.default_rng(seed)
    noise = noise or {"monaco": 0.8, "monza": 7.0, "spa": 3.0}
    rows, quali = [], []
    for season in seasons:
        circuits = list(noise) + (["indianapolis"] if season <= 1960 else [])
        for rnd, circuit in enumerate(circuits, start=1):
            race_id = season * 100 + rnd
            sigma = noise.get(circuit, 12.0)
            slots = np.arange(1, 21)
            order = np.argsort(slots + rng.normal(0, sigma, 20))
            grid_slot = slots.copy()
            if rng.random() < swap:  # grid penalty: pole and second swapped relative to qualifying
                grid_slot[[0, 1]] = [2, 1]
            for finish, idx in enumerate(order, start=1):
                rows.append({"race_id": race_id, "season": season, "round": rnd, "circuit_id": circuit,
                             "circuit_name": circuit.title(), "driver_id": f"d{idx}", "constructor_id": f"c{idx // 2}",
                             "grid": int(grid_slot[idx]), "position": finish, "classified": finish <= 16})
                quali.append({"race_id": race_id, "driver_id": f"d{idx}", "position": int(slots[idx])})
    return pd.DataFrame(rows), pd.DataFrame(quali)


def make_con(path, rows, quali):
    con = duckdb.connect(str(path))
    results = rows[["race_id", "season", "round", "driver_id", "constructor_id", "grid", "position", "classified"]]
    races = rows.drop_duplicates("race_id")[["race_id", "season", "round", "circuit_id"]]
    circuits = rows.drop_duplicates("circuit_id")[["circuit_id", "circuit_name"]]
    for name, df in [("results", results), ("races", races), ("circuits", circuits), ("qualifying", quali)]:
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    return con


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    rows, quali = world()
    con = make_con(tmp_path_factory.mktemp("g") / "f.duckdb", rows, quali)
    raw = gc.load(con)
    return raw, gc.prepare_checks(raw, drop_indy=False)


def test_load_flags_indy_and_qualifying(loaded):
    raw, _ = loaded
    assert raw["is_indy"].sum() == 11 * 20
    assert raw[raw["season"] > 1960]["is_indy"].sum() == 0
    assert raw["q_pos"].notna().all() and raw["classified"].dtype == bool


def test_coverage_counts_indy_rows(loaded):
    _, with_indy = loaded
    cov = gc.coverage(with_indy).set_index("era")
    assert cov.loc["1950-69", "indy_rows"] == 220 and cov.loc["1970-89", "indy_rows"] == 0
    assert cov.loc["2010-21", "share_not_classified"] == pytest.approx(4 / 20)
    assert cov.loc["2010-21", "mean_field_size"] == 20 and cov.loc["2010-21", "share_grid_missing_or_zero"] == 0


def test_indy_sensitivity_removing_indy_tightens_early_era(loaded):
    raw, _ = loaded
    s = gc.indy_sensitivity(raw).set_index(["era", "version"])
    with_i, without = s.loc[("1950-69", "with Indianapolis 500")], s.loc[("1950-69", "without")]
    assert with_i["races"] == without["races"] + 11
    assert without["mean_spearman"] > with_i["mean_spearman"]
    assert s.loc[("1970-89", "with Indianapolis 500"), "races"] == s.loc[("1970-89", "without"), "races"]


def test_reliability_identity_and_finishers_only(loaded):
    _, with_indy = loaded
    df = with_indy[~with_indy["is_indy"]]
    table = gc.reliability_table(gc.race_reliability(df))
    assert len(table) == 5
    prod = table["pole_finish_rate"] * table["win_given_finish"]
    assert np.allclose(prod, table["pole_win_rate"])
    assert (table["finish_ci_low"] <= table["pole_finish_rate"] + 1e-9).all()
    assert (table["pole_finish_rate"] <= table["finish_ci_high"] + 1e-9).all()
    assert (table["spearman_finishers_only"] != table["spearman_all_starters"]).all()


def test_pole_that_retires_cannot_win():
    df = pd.DataFrame({"race_id": 1, "season": 2000, "era": "1990-2009", "grid": list(range(1, 11)),
                       "grid_slot": list(range(1, 11)), "position": [10, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       "classified": [False] + [True] * 9})
    r = gc.race_reliability(df).iloc[0]
    assert r["has_pole"] and not r["pole_finished"] and not r["pole_won"]


def test_permutation_detects_a_real_circuit_effect_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    groups = np.tile(np.arange(6), 40)
    strata = np.repeat(np.arange(40), 6)
    real = rng.normal(0, 1, 240) + np.where(groups == 0, 2.0, 0.0)
    fake = rng.normal(0, 1, 240)
    a = gc.circuit_permutation(real, groups, strata, n_perm=300)
    b = gc.circuit_permutation(fake, groups, strata, n_perm=300)
    assert a["p_value"] < 0.01 and a["share_explained"] > a["share_by_chance"] * 3
    assert b["p_value"] > 0.05


def test_permutation_ignores_a_season_effect():
    """Every circuit has the same two seasons, one easy and one hard: shuffling inside a season removes it."""
    groups = np.tile(np.arange(4), 20)
    strata = np.repeat(np.arange(20), 4)
    values = np.repeat(np.arange(20) % 2, 4).astype(float)
    res = gc.circuit_permutation(values, groups, strata, n_perm=200)
    assert res["share_explained"] == pytest.approx(0, abs=1e-9)


def test_shrink_rates_pull_small_samples_in_and_keep_big_ones():
    wins = np.array([1, 50, 10, 5])
    n = np.array([1, 100, 20, 20])
    shrunk, strength = gc.shrink_rates(wins, n)
    raw = wins / n
    pbar = wins.sum() / n.sum()
    assert abs(shrunk[0] - pbar) < abs(raw[0] - pbar) * 0.6   # the 1-race circuit moves most
    assert abs(shrunk[1] - raw[1]) < abs(raw[0] - shrunk[0])  # the 100-race circuit barely moves
    same, strength_same = gc.shrink_rates(np.array([5, 10, 20]), np.array([10, 20, 40]))
    assert strength_same > 1e6 and np.allclose(same, 0.5)    # identical rates: everything shrinks to the mean


def test_shrink_means_reliability_grows_with_sample_size():
    means = np.array([0.9, 0.5, 0.1, 0.6])
    n = np.array([3, 30, 3, 30])
    shrunk, rel = gc.shrink_means(means, n, within_var=0.04)
    assert rel[1] > rel[0] and rel[3] > rel[2]
    assert abs(shrunk[0] - 0.9) > abs(shrunk[1] - 0.5)


def test_circuit_heterogeneity_finds_monaco_above_monza(loaded):
    _, with_indy = loaded
    df = with_indy[~with_indy["is_indy"]]
    races = grid.race_summary(df)
    table, tests = gc.circuit_heterogeneity(races, n_perm=200)
    t = table.set_index("circuit_id")
    assert t.loc["monaco", "spearman_shrunk"] > t.loc["monza", "spearman_shrunk"] + 0.25
    assert (t["spearman_reliability"].between(0, 1)).all()
    row = tests.set_index("measure").loc["grid-vs-finish correlation"]
    assert row["p_value"] < 0.01 and row["share_explained"] > row["share_by_chance"]


def test_circuit_heterogeneity_quiet_when_circuits_are_alike(tmp_path):
    rows, quali = world(seed=4, noise={"a": 3.0, "b": 3.0, "c": 3.0, "d": 3.0}, seasons=range(2000, 2025))
    races = grid.race_summary(grid.prepare(rows))
    _, tests = gc.circuit_heterogeneity(races, n_perm=300)
    assert tests["p_value"].min() > 0.01


def test_dominance_follows_a_team_that_wins_everything():
    rows = []
    for season in (2000, 2001):
        for rnd in range(1, 11):
            winner = "c0" if season == 2000 else f"c{rnd % 5}"
            for pos in range(1, 11):
                rows.append({"race_id": season * 100 + rnd, "season": season, "round": rnd, "circuit_id": "x",
                             "circuit_name": "X", "driver_id": f"d{pos}", "constructor_id": winner if pos == 1 else "z",
                             "grid": pos, "position": pos, "classified": True})
    df = grid.prepare(pd.DataFrame(rows))
    races = grid.race_summary(df)
    dom = gc.season_dominance(df, races).set_index("season")
    assert dom.loc[2000, "top_team_win_share"] == 1.0 and dom.loc[2001, "top_team_win_share"] == pytest.approx(0.2)


def test_dominance_correlations_shape():
    seasons = pd.DataFrame({"season": range(1990, 2010), "n_races": 16, "top_team_win_share": np.linspace(0.2, 0.9, 20),
                            "pole_win_rate": np.linspace(0.3, 0.8, 20), "mean_spearman": np.linspace(0.5, 0.8, 20)})
    out = gc.dominance_correlations(seasons)
    assert list(out["from_season"]) == [1950, 1990] and (out["rank_corr_with_pole_win_rate"] > 0.99).all()


def test_pole_definitions_detect_grid_penalties(loaded):
    _, with_indy = loaded
    out = gc.pole_definitions(with_indy[~with_indy["is_indy"]]).set_index("era")
    assert 0.1 < out.loc["2010-21", "grid_pole_differs_share"] < 0.5
    assert out.loc["2010-21", "differing_win_rate_grid_pole"] >= 0


def test_full_run_writes_everything(tmp_path):
    rows, quali = world(seed=9)
    con = make_con(tmp_path / "f.duckdb", rows, quali)
    out = tmp_path / "out"
    lines = gc.run(con, out, n_perm=100)
    for name in ["coverage_by_era.csv", "indy_sensitivity.csv", "reliability_by_era.csv", "circuit_shrinkage.csv",
                 "circuit_tests.csv", "season_dominance.csv", "dominance_correlations.csv",
                 "pole_by_grid_vs_qualifying.csv", "reliability_decomposition.png", "circuit_shrinkage.png",
                 "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    for head in ["COVERAGE", "INDIANAPOLIS 500", "RELIABILITY", "CIRCUITS SINCE", "TEAM DOMINANCE", "POLE BY GRID"]:
        assert head in text, head


def test_run_without_a_qualifying_table(tmp_path):
    rows, quali = world(seed=2, seasons=range(2000, 2012))
    con = duckdb.connect(str(tmp_path / "f.duckdb"))
    results = rows[["race_id", "season", "round", "driver_id", "constructor_id", "grid", "position", "classified"]]
    races = rows.drop_duplicates("race_id")[["race_id", "season", "round", "circuit_id"]]
    circuits = rows.drop_duplicates("circuit_id")[["circuit_id", "circuit_name"]]
    for name, df in [("results", results), ("races", races), ("circuits", circuits)]:
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    lines = gc.run(con, tmp_path / "out", n_perm=50)
    assert not any("POLE BY GRID" in l for l in lines)
