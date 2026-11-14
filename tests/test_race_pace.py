import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis import race_pace, ratings
from . import lapworld


def laps_world(seed=0, n_races=14, n_drivers=8, n_laps=30, gap=0.005):
    """Pairs of team-mates; the first of each pair is `gap` (as a fraction) faster in every race."""
    rng = np.random.default_rng(seed)
    rows, teams = [], []
    for r in range(n_races):
        rid = 201800 + r + 1
        car = rng.normal(0, 0.01, n_drivers // 2)
        for d in range(n_drivers):
            teams.append({"race_id": rid, "driver_id": f"d{d}", "constructor_id": f"c{d // 2}"})
            off = car[d // 2] - (gap if d % 2 == 0 else 0.0)
            for lap in range(1, n_laps + 1):
                t = 90 * np.exp(off + rng.normal(0, 0.002)) - 0.05 * lap
                rows.append({"race_id": rid, "season": 2018, "driver_id": f"d{d}", "lap_number": lap, "lap_time_s": t,
                             "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "is_accurate": True})
    return pd.DataFrame(rows), pd.DataFrame(teams)


def test_clean_laps_drop_pit_laps_lap_one_neutral_inaccurate_and_slow_laps():
    base = {"race_id": 1, "driver_id": "d", "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1", "is_accurate": True}
    rows = [dict(base, lap_number=lap, lap_time_s=90.0) for lap in range(1, 11)]
    rows[3] = dict(rows[3], pit_in_s=1.0)
    rows[4] = dict(rows[4], pit_out_s=1.0)
    rows[5] = dict(rows[5], track_status="4")
    rows[6] = dict(rows[6], is_accurate=False)
    rows[7] = dict(rows[7], lap_time_s=120.0)            # more than 107% of his median
    kept = race_pace.clean_laps(pd.DataFrame(rows))
    assert sorted(kept["lap_number"]) == [2, 3, 9, 10]


def test_race_pace_is_relative_to_the_field_and_ignores_fuel_trend():
    laps, _ = laps_world()
    pace = race_pace.race_pace(laps)
    first = pace[pace["race_id"] == 201801].set_index("driver_id")["pace"]
    assert first["d0"] < first["d1"]                      # faster than his team-mate
    assert abs(first["d1"] - first["d0"] - 0.005) < 0.002
    assert pace["laps"].min() >= race_pace.MIN_LAPS


def test_drivers_with_too_few_clean_laps_or_races_with_a_small_field_are_dropped():
    laps, _ = laps_world(n_laps=12)
    assert race_pace.race_pace(laps).empty                # 11 clean laps < MIN_LAPS
    small, _ = laps_world(n_drivers=6)
    assert race_pace.race_pace(small).empty               # fewer than 8 cars on a lap


def test_pace_observations_pair_team_mates_only_and_ratings_recover_the_gap():
    laps, teams = laps_world(seed=3)
    obs = race_pace.build_pace_obs(race_pace.race_pace(laps), teams)
    assert len(obs) == 14 * 4 and (obs["y"] > 0).all()
    assert obs["y"].mean() == pytest.approx(0.005, abs=0.001)
    fit, noise, prior = ratings.fit_ratings(obs)
    r = fit.set_index("driver_id")["rating"]
    assert r["d0"] > 0 > r["d1"]
    assert 0 < noise < 0.01


def test_a_huge_gap_is_dropped_as_a_problem_not_a_rating():
    laps, teams = laps_world(seed=4, gap=0.08)
    obs = race_pace.build_pace_obs(race_pace.race_pace(laps), teams)
    assert obs.empty


def test_compare_flags_a_driver_whose_race_rating_differs_from_qualifying():
    race_t = pd.DataFrame({"driver_id": list("abc"), "name": list("ABC"), "rating": [0.5, 0.0, -0.5],
                           "sd": [0.1, 0.1, 0.1], "n_obs": [40, 40, 10]})
    quali_t = pd.DataFrame({"driver_id": list("abc"), "rating": [0.1, 0.0, -0.5], "sd": [0.1, 0.1, 0.1],
                            "n_obs": [40, 40, 40]})
    out = race_pace.compare(race_t, quali_t)
    assert list(out["driver_id"]) == ["a", "b"]           # c has too few race comparisons
    assert out.iloc[0]["race_minus_quali"] == pytest.approx(0.4) and out.iloc[0]["z"] > 2


def test_full_run_writes_everything_and_is_honest_when_data_is_thin(tmp_path):
    w = lapworld.world(seed=5)
    con = lapworld.build_con(tmp_path / "w.duckdb", w)
    con.execute("CREATE TABLE qualifying(race_id INTEGER, driver_id VARCHAR, constructor_id VARCHAR, position INTEGER, "
                "q1 VARCHAR, q2 VARCHAR, q3 VARCHAR)")
    drivers = pd.DataFrame({"driver_id": [f"d{i}" for i in range(14)], "given_name": "D", "family_name": [f"N{i}" for i in range(14)]})
    con.register("drivers_df", drivers)
    con.execute("CREATE TABLE drivers AS SELECT * FROM drivers_df")
    out = tmp_path / "out"
    lines = race_pace.run(con, out)
    for name in ["ratings_race_pace_2018_on.csv", "race_pace_by_driver_race.csv", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert "RACE PACE" in "\n".join(lines) or "Not enough" in "\n".join(lines)


def _cmp(n=30, sd=0.08, spread=0.25, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, spread, n)
    return pd.DataFrame({"driver_id": [f"d{i}" for i in range(n)], "name": [f"Driver {i}" for i in range(n)],
                         "rating_race": t + rng.normal(0, sd, n), "rating_quali": t + rng.normal(0, sd, n),
                         "sd_race": sd, "sd_quali": sd, "race_minus_quali": 0.0, "z": 0.0})


def test_noise_ceiling_is_high_for_precise_ratings_and_low_for_noisy_ones():
    precise = race_pace.noise_ceiling(_cmp(sd=0.02).assign(sd_race=0.02, sd_quali=0.02), spread=0.25)
    noisy = race_pace.noise_ceiling(_cmp(sd=0.4).assign(sd_race=0.4, sd_quali=0.4), spread=0.25)
    assert precise[0] <= precise[1] <= precise[2]
    assert precise[1] > 0.9 and noisy[1] < 0.5


def test_findings_say_whether_the_observed_correlation_is_below_the_noise_ceiling():
    cmp_ = _cmp()
    race_t = pd.DataFrame({"name": ["A"], "rating": [0.1], "ci_low": [0.0], "ci_high": [0.2], "n_obs": [50]})
    obs = pd.DataFrame({"x": range(5)})
    low = "\n".join(race_pace.findings(obs, race_t, 0.005, 0.0025, cmp_, 0.10, 10, (0.5, 0.7, 0.85)))
    ok = "\n".join(race_pace.findings(obs, race_t, 0.005, 0.0025, cmp_, 0.75, 10, (0.5, 0.7, 0.85)))
    assert "below that range" in low and "tyre age, strategy and traffic matter" in low
    assert "within that range" in ok and "not established" in ok
    assert "chart shows those with at least" in low


def test_chart_always_names_the_top_rated_driver(tmp_path):
    cmp_ = _cmp()
    cmp_.loc[7, ["rating_race", "rating_quali"]] = [0.9, 0.9]       # best driver, no race-vs-qualifying gap
    cmp_["race_minus_quali"] = (cmp_["rating_race"] - cmp_["rating_quali"]).where(cmp_.index != 7, 0.0)
    import matplotlib.text as mtext
    seen = []
    orig = mtext.Annotation.__init__
    def spy(self, text, *a, **k):
        seen.append(text); orig(self, text, *a, **k)
    mtext.Annotation.__init__ = spy
    try:
        race_pace.chart_compare(cmp_, tmp_path / "c.png", 0.6, (0.5, 0.7, 0.85))
    finally:
        mtext.Annotation.__init__ = orig
    assert "7" in seen
