import duckdb
import numpy as np
import pandas as pd

from src.analysis import car_driver as cd
from src.analysis import car_driver_checks as chk


def fake_rows(seed=0, seasons=range(2010, 2024), teams=8, pool=40, races=6, move=True):
    rng = np.random.default_rng(seed)
    skill = {f"d{i}": rng.normal(0, 0.4) for i in range(pool)}
    base = {t: rng.normal(0, 0.9) for t in range(teams)}
    recs, quali = [], []
    for season in seasons:
        active = list(rng.choice(list(skill), size=2 * teams, replace=False)) if move else [f"d{i}" for i in range(2 * teams)]
        dev = {t: rng.normal(0, 0.4) for t in range(teams)}
        for rnd in range(races):
            rid = season * 100 + rnd
            perf = {d: base[j // 2] + dev[j // 2] + skill[d] + rng.normal(0, 0.6) for j, d in enumerate(active)}
            order = sorted(perf, key=perf.get, reverse=True)
            for pos, d in enumerate(order, 1):
                j = active.index(d)
                recs.append({"race_id": rid, "season": season, "driver_id": d, "constructor_id": f"c{j // 2}",
                             "grid": pos, "position": pos})
                t = 90 * np.exp(-0.02 * perf[d])
                quali.append({"race_id": rid, "driver_id": d, "position": pos, "q1": f"{int(t // 60)}:{t % 60:06.3f}"})
    return pd.DataFrame(recs), pd.DataFrame(quali)


def to_rows(res, qual):
    rows = cd.build_rows(res, qual)
    return chk.add_lap_gap(rows, qual)


def test_mobility_counts_movers_and_connectivity():
    res, qual = fake_rows()
    rows = to_rows(res, qual)
    m = chk.mobility(rows)
    assert m["drivers"] == rows["driver_id"].nunique()
    assert 0 < m["share_drivers_with_2plus_teams"] <= 1
    assert m["largest_piece_share_of_nodes"] > 0.9
    static = chk.mobility(to_rows(*fake_rows(move=False)))
    assert static["share_drivers_with_2plus_teams"] < m["share_drivers_with_2plus_teams"]
    assert static["connected_pieces"] >= m["connected_pieces"]


def test_recovery_is_close_when_drivers_move_and_poor_when_they_do_not():
    scen = {"driver 20%": {"team": 0.25, "car": 0.25, "driver": 0.20}}
    moving = to_rows(*fake_rows(seasons=range(2010, 2024), races=10))
    still = to_rows(*fake_rows(seasons=range(2010, 2024), races=10, move=False))
    r_move = chk.recovery_table(moving, n_rep=3, scenarios=scen).iloc[0]
    r_still = chk.recovery_table(still, n_rep=3, scenarios=scen).iloc[0]
    assert abs(r_move["driver_bias"]) < 0.10
    assert abs(r_still["driver_bias"]) > abs(r_move["driver_bias"])


def test_rolling_windows_cover_the_last_season():
    rows = to_rows(*fake_rows(seasons=range(2000, 2024)))
    roll = chk.rolling_table(rows, window=10, step=4)
    assert roll["last_season"].max() == 2023
    assert set(roll["outcome"]) == {"qualifying", "race"}
    assert np.allclose(roll[["car_share", "driver_share", "other_share"]].sum(axis=1), 1)


def test_lap_gap_is_standardised_and_drops_outliers():
    res, qual = fake_rows(seasons=range(2015, 2017))
    qual.loc[qual.index[0], "q1"] = "2:30.000"   # a car 65% slower is dropped
    rows = to_rows(res, qual)
    assert rows["gap_score"].notna().sum() == len(rows) - 1
    assert abs(rows["gap_score"].std() - 1) < 0.05


def test_placebo_removes_driver_share_when_it_is_real():
    rows = to_rows(*fake_rows(seasons=range(2000, 2022), races=10))
    rows["era"] = "2010-21"
    tab = chk.placebo_table(rows, n_rep=1)
    q = tab[tab["outcome"] == "qualifying"].iloc[0]
    assert q["driver_share_real"] > 0.04
    assert q["driver_share_shuffled"] < q["driver_share_real"] / 2, dict(q)


def test_two_car_filter_drops_customer_cars():
    res, qual = fake_rows(seasons=range(2010, 2022), races=10)
    rows = to_rows(res, qual)
    extra = rows[rows["constructor_id"] == "c0"].copy()
    extra["driver_id"] = "customer_" + extra["driver_id"]      # a third and fourth car for c0
    rows = pd.concat([rows, extra], ignore_index=True)
    rows["era"] = "2010-21"
    tab = chk.two_car_table(rows)
    assert (tab["share_of_rows_kept"] < 1).all()


def test_full_run_writes_everything(tmp_path):
    res, qual = fake_rows(seasons=range(2000, 2022), races=6)
    races = pd.DataFrame({"race_id": sorted(res["race_id"].unique())})
    races["season"] = races["race_id"] // 100
    races["circuit_id"] = "x"
    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    for name, df in (("results", res), ("qualifying", qual), ("races", races)):
        con.register(name + "_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    out = tmp_path / "out"
    lines = chk.run(con, out, n_rep=1)
    for name in ["mobility_by_era.csv", "recovery_by_era.csv", "rank_vs_laptime_by_era.csv",
                 "rolling_shares.csv", "rolling_shares.png", "placebo_by_era.csv", "two_car_teams_by_era.csv",
                 "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert any(l.startswith("MOBILITY") for l in lines) and any(l.startswith("RECOVERY") for l in lines)
