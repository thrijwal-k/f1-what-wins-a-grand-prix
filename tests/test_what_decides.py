import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis import race_events as ev, what_decides as wd
from . import whatworld


def test_status_classes():
    c = ev.classify_status
    assert c("Finished") == "finished" and c("+1 Lap") == "lapped" and c("+3 Laps") == "lapped" and c("Lapped") == "lapped"
    assert c("Engine") == "mechanical" and c("Gearbox") == "mechanical" and c("Retired") == "unspecified"
    assert c("Collision") == "incident" and c("Accident") == "incident" and c("Spun off") == "incident"
    assert c("Collision damage") == "incident"
    assert c("Disqualified") == "disqualified" and c("Excluded") == "disqualified"
    assert c("Did not start") == "not_started" and c("Withdrew") == "not_started" and c("Illness") == "not_started"
    assert c(None) == "unspecified" and c("") == "unspecified"


def test_stop_features_and_safety_car_features():
    res = pd.DataFrame({"race_id": [1, 1, 1], "driver_id": ["a", "b", "c"], "laps": [50, 50, 20]})
    pits = pd.DataFrame({"race_id": [1, 1, 1], "driver_id": ["a", "a", "b"], "stop": [1, 2, 1], "lap": [10, 30, 20],
                         "duration_s": [22.0, 24.0, 200.0]})
    f = ev.stop_features(res, pits).set_index("driver_id")
    assert f.loc["a", "n_stops"] == 2 and f.loc["a", "first_stop"] == pytest.approx(0.2)
    assert f.loc["b", "pit_time"] == pytest.approx(ev.PIT_CLIP_S)       # a repair is clipped
    assert f.loc["c", "n_stops"] == 0                                      # no stop recorded in a covered race
    laps = pd.DataFrame([{"race_id": 1, "lap_number": l, "track_status": "4" if 12 <= l <= 15 else "1"}
                         for l in range(1, 51) for _ in range(4)])
    pits2 = pd.DataFrame({"race_id": [1, 1, 1], "driver_id": ["a", "b", "c"], "stop": [1, 1, 1], "lap": [13, 10, 40],
                          "duration_s": [20.0, 22.0, 22.0]})
    sc = ev.safety_car_features(res, pits2, laps).set_index("driver_id")
    assert sc.loc["a", "free_stops"] == 1 and sc.loc["a", "unlucky_stops"] == 0
    assert sc.loc["b", "free_stops"] == 0 and sc.loc["b", "unlucky_stops"] == 1       # stopped two laps before the SC
    assert sc.loc["c", "free_stops"] == 0 and sc.loc["c", "unlucky_stops"] == 0


def test_shapley_shares_add_to_the_joint_r2_and_split_duplicates_evenly():
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=2000), rng.normal(size=2000)
    y = x1 + 0.5 * x2 + rng.normal(size=2000)
    out = wd.shapley_r2({"a": x1, "b": x2}, y)
    X = np.column_stack([x1, x2])
    beta = np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)[0]
    joint = float(((X - X.mean(0)) @ beta @ ((X - X.mean(0)) @ beta))) / float(((y - y.mean()) ** 2).sum())
    assert sum(out.values()) == pytest.approx(joint, rel=1e-6) and out["a"] > out["b"] > 0
    twin = wd.shapley_r2({"a": x1, "a2": x1.copy()}, x1 + rng.normal(size=2000))
    assert twin["a"] == pytest.approx(twin["a2"], rel=1e-6)


def _table(**kw):
    res, qual, pits, laps = whatworld.world(**kw)
    return wd.build_table(res, qual, pits, laps)


def test_build_table_drops_rulings_and_has_every_block_input():
    res, qual, pits, laps = whatworld.world(seed=3, races=6)
    res.loc[0, "status"] = "Disqualified"
    res.loc[1, "status"] = "Did not start"
    t = wd.build_table(res, qual, pits, laps)
    assert len(t) == len(res) - 2
    assert {"status_class", "completed", "d_n_stops", "free_stops", "finish_score", "start_score", "period"} <= set(t.columns)
    assert set(t["period"]) == {"2018-21"}
    # scores are recomputed on the rows that remain: every race is centred at zero
    assert abs(t.groupby("race_id")["finish_score"].mean()).max() < 1e-9


def test_shares_add_to_one_and_blocks_find_what_was_built_in():
    t = _table(seed=2, races=24, strategy_effect=0.8, free_bonus=0.8)
    rec = wd.decompose_period(t, n_stab=0, n_boot=20)
    blocks = ["Qualifying form", "Mechanical retirements", "Accidents", "Safety-car timing", "Strategy"]
    total = rec["car_share"] + rec["driver_share"] + sum(rec[f"{b}_share"] for b in blocks) + rec["unexplained_share"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert rec["Mechanical retirements_share"] > rec["Accidents_share"] > 0          # 8% against 4% of rows
    assert rec["Strategy_share"] > 0.01 and rec["Safety-car timing_share"] > 0.003
    assert rec["status_mechanical"] > rec["status_incident"]


def test_without_a_strategy_effect_the_strategy_block_is_small():
    with_effect = wd.decompose_period(_table(seed=5, races=24, strategy_effect=0.9), n_stab=0, n_boot=0)
    without = wd.decompose_period(_table(seed=5, races=24, strategy_effect=0.0), n_stab=0, n_boot=0)
    assert with_effect["Strategy_share"] > 3 * without["Strategy_share"]
    assert without["Strategy_share"] < 0.02


def test_blocks_that_cannot_be_measured_are_left_out():
    res, qual, pits, laps = whatworld.world(seed=4, races=20, with_pits=False, with_laps=False)
    t = wd.build_table(res, qual, pits, laps)
    rec = wd.decompose_period(t, n_stab=0, n_boot=0)
    assert "Strategy" not in rec["blocks_measured"] and "Safety-car timing" not in rec["blocks_measured"]
    assert np.isnan(rec["Strategy_share"]) and np.isnan(rec["Safety-car timing_share"])
    assert rec["unexplained_share"] > 0


def test_car_effect_on_the_grid_matches_the_race_and_finishers_only_has_less_reliability():
    rec = wd.decompose_period(_table(seed=6, races=24, p_mech=0.2), n_stab=0, n_boot=0)
    assert rec["car_effect_corr_grid_vs_race"] > 0.7
    assert 0 < rec["car_share_finishers_only"] < 1


def test_full_run_writes_everything(tmp_path):
    res, qual, pits, laps = whatworld.world(seed=7, races=24)
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    for name, df in (("results", res), ("qualifying", qual), ("pit_stops", pits), ("laps", laps)):
        con.register("df_", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_")
        con.unregister("df_")
    out = tmp_path / "out"
    lines = wd.run(con, out, n_stab=3, n_boot=10)
    for name in ["decomposition_by_period.csv", "decomposition_by_period.png", "decomposition_chart_values.csv", "status_classification.csv", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    assert "WHAT DECIDES A RACE RESULT" in text and "2018-21" in text and "READ WITH CARE" in text
    assert "WHO IS BEHIND THE RETIREMENTS" in text and "STRATEGY BLOCK" in text
    st = pd.read_csv(out / "status_classification.csv")
    assert {"status", "status_class", "total"} <= set(st.columns) and st["total"].iloc[0] >= st["total"].iloc[-1]
    values = pd.read_csv(out / "decomposition_chart_values.csv")
    assert values.drop(columns="period").sum(axis=1).iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_too_little_data_is_reported_not_crashed(tmp_path):
    res, qual, pits, laps = whatworld.world(seed=8, races=2, seasons=(2018,))
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    for name, df in (("results", res), ("qualifying", qual), ("pit_stops", pits), ("laps", laps)):
        con.register("df_", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_")
        con.unregister("df_")
    lines = wd.run(con, tmp_path / "o", n_stab=0, n_boot=0)
    assert "Not enough" in lines[0]


def test_retirements_are_attributed_to_the_car_when_the_car_causes_them():
    rows = _table(seed=11, races=30)
    rng = np.random.default_rng(0)
    bad = {c: rng.random() < 0.4 for c in rows["car"].unique()}
    flip = rows["car"].map(bad) & (rng.random(len(rows)) < 0.45)
    rows = rows.copy()
    rows.loc[flip, "status_class"] = "mechanical"
    rows.loc[~flip & (rows["status_class"] == "mechanical"), "status_class"] = "finished"
    rec = wd.decompose_period(rows, n_stab=0, n_boot=0)
    assert rec["mech_car_share"] > 0.2 and rec["mech_car_share"] > 2 * rec["mech_driver_share"]
    assert 0 <= rec["car_effect_corr_all_vs_finishers"] <= 1


def test_bare_retired_is_kept_apart_and_merges_the_retirement_blocks():
    n = 400
    base = pd.DataFrame({"completed": np.linspace(0, 1, n), "status_class": ["finished"] * 200 + ["mechanical"] * 40 + ["incident"] * 40 + ["unspecified"] * 120})
    assert wd.merge_retirements(base)
    few = base.assign(status_class=["finished"] * 200 + ["mechanical"] * 100 + ["incident"] * 90 + ["unspecified"] * 10)
    assert not wd.merge_retirements(few)
    blocks = wd.make_blocks(base, np.zeros(n))
    assert "Retirements" in blocks and "Mechanical retirements" not in blocks and "Accidents" not in blocks
    blocks = wd.make_blocks(few, np.zeros(n))
    assert "Mechanical retirements" in blocks and "Accidents" in blocks and "Retirements" not in blocks
