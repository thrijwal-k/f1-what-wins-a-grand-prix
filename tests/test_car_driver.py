import pytest
import duckdb
import numpy as np
import pandas as pd

from src.analysis import car_driver


def sim_rows(seed=1, sd_car=1.0, sd_driver=0.5, sd_noise=0.7, seasons=range(2010, 2022), teams=10, pool=60, races=8):
    """Drivers and cars with known effect sizes; the score is their sum plus noise."""
    rng = np.random.default_rng(seed)
    skill = {f"d{i}": rng.normal(0, sd_driver) for i in range(pool)}
    recs = []
    for season in seasons:
        active = list(rng.choice(list(skill), size=2 * teams, replace=False))
        car = {t: rng.normal(0, sd_car) for t in range(teams)}
        for rnd in range(races):
            for j, d in enumerate(active):
                recs.append({"race_id": season * 100 + rnd, "season": season, "driver_id": d,
                             "constructor_id": f"c{j // 2}", "car": f"c{j // 2}|{season}", "era": "2010-21",
                             "score": car[j // 2] + skill[d] + rng.normal(0, sd_noise)})
    return pd.DataFrame(recs)


def test_rankit_is_symmetric_and_ordered():
    pos = pd.Series([1, 2, 3, 4])
    z = car_driver.rankit(pos, pd.Series([1, 1, 1, 1]))
    assert list(z) == sorted(z, reverse=True)
    assert abs(z.sum()) < 1e-9


def test_model_recovers_known_shares():
    rows = sim_rows()
    d, nd = car_driver._codes(rows["driver_id"])
    c, nc = car_driver._codes(rows["car"])
    share = car_driver.shares(car_driver.fit_components(rows["score"].to_numpy(), d, c, nd, nc))
    truth = np.array([1.0, 0.25, 0.49]) / 1.74
    est = np.array([share["car"], share["driver"], share["other"]])
    assert np.all(np.abs(est - truth) < 0.07), (est, truth)
    assert abs(share["car"] + share["driver"] + share["other"] - 1) < 1e-9


def test_more_driver_skill_means_more_driver_share():
    low = sim_rows(sd_driver=0.2)
    high = sim_rows(sd_driver=0.9)

    def driver_share(rows):
        d, nd = car_driver._codes(rows["driver_id"])
        c, nc = car_driver._codes(rows["car"])
        return car_driver.shares(car_driver.fit_components(rows["score"].to_numpy(), d, c, nd, nc))["driver"]

    assert driver_share(high) > driver_share(low) + 0.1


def test_stability_subsets_never_repeat_a_season_and_bracket_the_truth():
    rows = sim_rows(seasons=range(2000, 2020)).rename(columns={"score": "finish_score"})
    reps = car_driver.stability_era(rows, "finish_score", n_rep=12)
    assert len(reps) == 12 and np.allclose(reps[["car", "driver", "other"]].sum(axis=1), 1)
    truth_car = 1.0 / 1.74
    assert reps["car"].min() - 0.05 < truth_car < reps["car"].max() + 0.05


def test_full_run_writes_everything(tmp_path):
    rng = np.random.default_rng(4)
    res, qual = [], []
    for season in range(2010, 2020):
        skill = {d: rng.normal(0, 0.4) for d in range(24)}
        car = {t: rng.normal(0, 1.0) for t in range(12)}
        for rnd in range(1, 9):
            rid = season * 100 + rnd
            perf = {d: car[d // 2] + skill[d] + rng.normal(0, 0.6) for d in range(24)}
            order = sorted(perf, key=perf.get, reverse=True)
            for pos, d in enumerate(order, start=1):
                res.append({"race_id": rid, "season": season, "driver_id": f"d{d}",
                            "constructor_id": f"c{d // 2}", "grid": pos, "position": pos})
                qual.append({"race_id": rid, "driver_id": f"d{d}", "position": pos,
                             "q1": f"1:{20 + pos * 0.1 + rng.normal(0, 0.05):06.3f}"})
    races = pd.DataFrame({"race_id": sorted({r["race_id"] for r in res})})
    races["season"] = races["race_id"] // 100
    races["circuit_id"] = "somewhere"
    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    for name, df in (("results", pd.DataFrame(res)), ("qualifying", pd.DataFrame(qual)), ("races", races)):
        con.register(name + "_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    out = tmp_path / "out"
    lines = car_driver.run(con, out, n_boot=3)
    for name in ["variance_shares_by_era.csv", "variance_shares_by_era.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    table = pd.read_csv(out / "variance_shares_by_era.csv")
    assert set(table["outcome"]) == {"qualifying", "race"}
    assert np.allclose(table[["car_share", "driver_share", "other_share"]].sum(axis=1), 1)
    assert (table["car_share"] > table["driver_share"]).all()
    assert len(lines) >= len(table)
    assert "SPREADS IN GRID PLACES" in "\n".join(lines)
    assert "QUALIFYING BY LAP-TIME GAP" in "\n".join(lines)      # must come through the real load_inputs path
    assert len(pd.read_csv(out / "variance_shares_qualifying_by_laptime.csv")) >= 1
    for part in ("car", "driver", "other"):
        assert (table[f"{part}_sd_places"] > 0).all()


def persistent_world(seed=0, persist_sd=1.0, season_sd=0.4, drv_sd=0.4, noise=0.8, n_seasons=20, teams=10, stay=4):
    """Constructors keep their strength across seasons and drivers stay with a team for several years."""
    rng = np.random.default_rng(seed)
    base = {t: rng.normal(0, persist_sd) for t in range(teams)}
    slots = [(t, k) for t in range(teams) for k in range(2)]
    holder = {s: f"d{s[0]}_{s[1]}_0" for s in slots}
    skill, rows, gen = {}, [], 0
    for season in range(n_seasons):
        if season and season % stay == 0:
            gen += 1
            perm = rng.permutation(len(slots))
            holder = {slots[i]: f"d{slots[j][0]}_{slots[j][1]}_{gen}" for i, j in enumerate(perm)}
        car = {t: base[t] + rng.normal(0, season_sd) for t in range(teams)}
        for _ in range(10):
            for (t, _k), d in holder.items():
                skill.setdefault(d, rng.normal(0, drv_sd))
                e = rng.normal(0, noise)
                rows.append({"season": season, "driver_id": d, "constructor_id": f"c{t}", "car": f"c{t}|{season}",
                             "era": "x", "score": car[t] + skill[d] + e,
                             "_car": car[t], "_driver": skill[d], "_noise": e})
    df = pd.DataFrame(rows)
    # truth = the variances actually realised in this simulated sample (only 10 teams are drawn, so the
    # realised car variance can differ a lot from its expectation)
    parts = df[["_car", "_driver", "_noise"]].var()
    return df.drop(columns=["_car", "_driver", "_noise"]), (parts["_driver"] / parts.sum(), parts["_car"] / parts.sum())


def test_persistent_team_strength_is_not_credited_to_drivers():
    """Regression test: without a team-level effect the driver share came out near 57% when it was 8%."""
    for seed in range(3):
        rows, (true_driver, true_car) = persistent_world(seed)
        est = car_driver.fit_era(rows, "score")
        # The bug this guards against was off by about 50 points, far outside these bounds.
        assert abs(est["driver"] - true_driver) < 0.07, (seed, est, true_driver)
        assert abs(est["car"] - true_car) < 0.09, (seed, est, true_car)


def test_absolute_spreads_scale_with_true_effect_sizes_and_shares_do_not_tell_them_apart():
    """Same driver spread, much smaller car spread: the driver SD is unchanged but its share jumps."""
    wide = car_driver.fit_era(sim_rows(sd_car=1.5, sd_driver=0.5, seed=3), "score")
    narrow = car_driver.fit_era(sim_rows(sd_car=0.5, sd_driver=0.5, seed=3), "score")
    assert abs(wide["driver_sd"] - narrow["driver_sd"]) < 0.12
    assert narrow["driver"] > wide["driver"] + 0.1
    assert wide["car_sd"] > narrow["car_sd"] + 0.5


def test_places_per_unit_makes_pooled_spread_the_uniform_rank_spread():
    k = car_driver.places_per_unit(1.0, field=20)
    assert abs(k - 20 * np.sqrt((1 - 1 / 400) / 12)) < 1e-9
    assert car_driver.places_per_unit(2.0) == car_driver.places_per_unit(1.0) / 2


def test_qualifying_by_time_uses_lap_gaps_and_skips_when_there_are_none():
    rng = np.random.default_rng(2)
    rows, q = [], []
    for season in range(1995, 2010):
        car = {t: rng.normal(0, 0.01) for t in range(8)}
        skill = {d: rng.normal(0, 0.004) for d in range(16)}
        for rnd in range(6):
            rid = season * 100 + rnd
            for d in range(16):
                t = 80 * np.exp(car[d // 2] - skill[d] + rng.normal(0, 0.003))
                rows.append({"race_id": rid, "season": season, "driver_id": f"d{d}", "constructor_id": f"c{d // 2}",
                             "car": f"c{d // 2}|{season}", "era": "1990-2009", "start_score": 0.0,
                             "finish_score": 0.0, "_t": t})
    df = pd.DataFrame(rows)
    for rid, g in df.groupby("race_id"):
        order = g["_t"].rank()
        for (_, r), o in zip(g.iterrows(), order):
            q.append({"race_id": rid, "driver_id": r["driver_id"], "position": int(o),
                      "q1": f"{int(r['_t'] // 60)}:{r['_t'] % 60:06.3f}"})
    out = car_driver.qualifying_by_time(df.drop(columns="_t"), pd.DataFrame(q))
    assert list(out["era"]) == ["1990-2009"]
    assert out["car_share"].iloc[0] > out["driver_share"].iloc[0]
    assert abs(out[["car_share", "driver_share", "other_share"]].iloc[0].sum() - 1) < 1e-9
    r = out.iloc[0]
    assert r["car_low"] <= r["car_high"] and r["driver_low"] <= r["driver_high"] and r["car_low"] - 0.05 < r["car_share"] < r["car_high"] + 0.05
    assert car_driver.qualifying_by_time(df.drop(columns="_t"), pd.DataFrame({"race_id": [1]})).empty


def test_spreads_in_places_are_the_shares_in_other_units():
    """Ranks always spread evenly over the field, so the parts add up (in squares) to the uniform-rank spread
    and places = sqrt(share) x that spread. The table must not be read as an independent measure."""
    rows = sim_rows(sd_car=1.0, sd_driver=0.5, seed=4)
    est = car_driver.fit_era(rows, "score")
    k = car_driver.places_per_unit(est["total_sd"])
    total_places = est["total_sd"] * k
    for part in ("car", "driver", "other"):
        assert est[f"{part}_sd"] * k == pytest.approx(np.sqrt(est[part]) * total_places, rel=0.02)
