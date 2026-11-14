import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis import car_driver as cd, car_driver_robust as rb
from . import whatworld


def design(seed=1, seasons=range(2010, 2022), teams=8, pool=40, races=10):
    rng = np.random.default_rng(seed)
    recs = []
    for season in seasons:
        active = list(rng.choice([f"d{i}" for i in range(pool)], size=2 * teams, replace=False))
        for rnd in range(races):
            for j, d in enumerate(active):
                recs.append({"race_id": season * 100 + rnd, "season": season, "driver_id": d,
                             "constructor_id": f"c{j // 2}", "car": f"c{j // 2}|{season}"})
    df = pd.DataFrame(recs)
    df["era"] = "2010-21"
    return df


def test_simulated_outcomes_are_race_centred_scores_with_the_truth_built_in():
    dz = design()
    score, truth = rb.simulate(dz, "well specified", np.random.default_rng(0))
    s = pd.Series(score).groupby(dz["race_id"]).mean()
    assert s.abs().max() < 1e-9
    assert truth["car"] == pytest.approx(0.50, abs=0.12) and truth["driver"] == pytest.approx(0.12, abs=0.07)


def test_each_world_changes_the_truth_in_the_stated_direction():
    dz = design()
    base = rb.simulate(dz, "well specified", np.random.default_rng(1))[1]
    drift = rb.simulate(dz, "driver form drifts", np.random.default_rng(1))[1]
    dev = rb.simulate(dz, "in-season development", np.random.default_rng(1))[1]
    rel = rb.simulate(dz, "reliability varies", np.random.default_rng(1))[1]
    assert drift["driver"] > base["driver"] + 0.04
    assert dev["car"] > base["car"] + 0.03
    assert rel["car"] < base["car"]              # retirements add a lot of luck, which dilutes every share


def test_the_model_is_right_in_its_own_world_and_misses_drifting_drivers():
    out = rb.stress_era(design(), "2010-21", n_rep=2).set_index("scenario")
    assert abs(out.loc["well specified", "bias_driver"]) < 0.08 and abs(out.loc["well specified", "bias_car"]) < 0.12
    assert out.loc["driver form drifts", "bias_driver"] < out.loc["well specified", "bias_driver"] - 0.02
    assert set(out.index) == set(rb.SCENARIOS)


def test_extended_model_finds_form_only_when_it_exists():
    dz = design(seasons=range(2000, 2012), races=12)
    dz = dz.assign(**{"era": "2010-21"})
    s_plain, _ = rb.simulate(dz, "well specified", np.random.default_rng(3))
    s_drift, _ = rb.simulate(dz, "driver form drifts", np.random.default_rng(3))
    plain = rb.fit_extended(dz.assign(score=s_plain), "score")
    drift = rb.fit_extended(dz.assign(score=s_drift), "score")
    assert drift["driver_form"] > plain["driver_form"] + 0.03
    assert drift["driver_total"] > plain["driver_total"]
    parts = ["car", "car_development", "driver_persistent", "driver_form", "driver_car_fit", "other"]
    assert abs(sum(plain[k] for k in parts) - 1) < 1e-6


def test_race_bootstrap_brackets_the_point_and_allowance_only_widens():
    dz = design(races=8)
    score, _ = rb.simulate(dz, "well specified", np.random.default_rng(2))
    dz = dz.assign(finish_score=score, start_score=score)
    boots = rb.race_bootstrap(dz, "finish_score", n_rep=10)
    point = cd.fit_era(dz, "finish_score")
    assert len(boots) == 10 and boots["car"].min() - 0.05 < point["car"] < boots["car"].max() + 0.05
    stress = rb.stress_era(dz, "2010-21", n_rep=1)
    table = rb.intervals_table(dz, stress, n_boot=6)
    assert (table["driver_boot_low"] <= table["driver_share"] + 1e-12).all() and (table["driver_share"] <= table["driver_boot_high"] + 1e-12).all()
    assert (table["driver_allowed_low"] <= table["driver_boot_low"] + 1e-12).all()
    assert (table["driver_allowed_high"] >= table["driver_boot_high"] - 1e-12).all()
    data = rb.stress_era(dz, "2010-21", n_rep=1, sizes={"drift": 0.01, "fit": 0.01, "dev": 0.01}, scenarios=rb.DATA_SCENARIOS)
    assert set(data["world"]) == {"data-sized"} and len(data) == 3
    t2 = rb.intervals_table(dz, stress, n_boot=6, stress_data=data)
    assert (t2["driver_data_low"] >= t2["driver_allowed_low"] - 1e-12).all()
    assert (t2["driver_data_low"] <= t2["driver_boot_low"] + 1e-12).all()


def test_full_quick_run_writes_everything(tmp_path):
    res, qual, _, _ = whatworld.world(seed=9, races=24)
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    races = pd.DataFrame({"race_id": res["race_id"].unique(), "circuit_id": "alpha"}).assign(season=lambda d: d["race_id"] // 100)
    for name, df in (("results", res), ("qualifying", qual.assign(q1="1:30.0")), ("races", races)):
        con.register("df_", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_")
        con.unregister("df_")
    out = tmp_path / "out"
    lines = rb.run(con, out, quick=True)
    for name in ["stress_tests.csv", "driver_form_vs_skill.csv", "driver_share_intervals.csv", "stress_tests_data_sized.csv", "driver_share_intervals.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    assert "STRESS TESTS" in text and "WHAT THE EXTENDED MODEL FINDS" in text and "INTERVALS FOR THE DRIVER AND CAR SHARES" in text
