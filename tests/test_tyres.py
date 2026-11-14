import numpy as np
import pandas as pd
import pytest

from src.analysis import tyres
from . import lapworld


@pytest.fixture(scope="module")
def laps_and_flags():
    w = lapworld.world(seed=1)
    laps = w["laps"].merge(w["races"], on="race_id")
    laps["circuit_name"] = laps["circuit_id"]
    rain = w["weather"].groupby("race_id")["rainfall"].mean()
    return laps, tyres.race_flags(laps, rain), w


def test_race_flags_detect_safety_car_and_rain(laps_and_flags):
    _, flags, _ = laps_and_flags
    f = flags.set_index("race_id")
    assert bool(f.at[201803, "neutral"]) and bool(f.at[201804, "neutral"]) and not bool(f.at[201801, "neutral"])
    assert bool(f.at[201806, "wet"]) and not bool(f.at[201805, "wet"])
    assert not flags["red"].any()
    assert (f.loc[[201803, 201804], "sc_laps"] == 3).all() and (f.loc[[201801, 201805], "sc_laps"] == 0).all()


def test_a_stray_status_on_one_car_does_not_flag_the_race():
    rows = [{"race_id": 1, "lap_number": lap, "driver_id": f"d{d}", "compound": "SOFT",
             "track_status": "4" if (lap == 5 and d == 0) else "1"} for lap in range(1, 11) for d in range(10)]
    f = tyres.race_flags(pd.DataFrame(rows)).iloc[0]
    assert not f["neutral"] and f["sc_laps"] == 0


def test_status_diagnostics_by_season(laps_and_flags):
    laps, flags, _ = laps_and_flags
    diag = tyres.status_diagnostics(flags, laps).set_index("season")
    assert list(diag.index) == [2018, 2019, 2020] and (diag["share_safety_car"] == 2 / 6).all()


def test_status_codes_for_red_flag_and_vsc():
    laps = pd.DataFrame({"race_id": [1, 1, 2], "lap_number": [1, 2, 1], "track_status": ["1", "15", "6"],
                         "compound": "SOFT"})
    f = tyres.race_flags(laps).set_index("race_id")
    assert bool(f.at[1, "red"]) and bool(f.at[1, "neutral"]) and bool(f.at[2, "neutral"]) and not bool(f.at[2, "red"])


def test_clean_laps_drop_pit_sc_first_lap_and_wet(laps_and_flags):
    laps, _, _ = laps_and_flags
    c = tyres.clean_laps(laps)
    assert (c["lap_number"] > 1).all() and c["pit_in_s"].isna().all() and c["pit_out_s"].isna().all()
    assert (c["track_status"] == "1").all() and set(c["compound"]) <= set(tyres.DRY)


def test_model_recovers_known_wear_fuel_and_compound_gaps(laps_and_flags):
    laps, flags, _ = laps_and_flags
    per_race = tyres.tyre_by_race(laps, flags.set_index("race_id")["wet"])
    assert len(per_race) >= 12 and 201806 not in set(per_race["race_id"])   # the wet race is skipped
    assert per_race["lap_trend_s_per_lap"].median() == pytest.approx(lapworld.FUEL, abs=0.01)
    for c in tyres.DRY:
        assert per_race[f"deg_{c}"].median() == pytest.approx(lapworld.WEAR[c], abs=0.015), c
    true_hard = (lapworld.OFFSET["HARD"] + 10 * lapworld.WEAR["HARD"]) - (lapworld.OFFSET["MEDIUM"] + 10 * lapworld.WEAR["MEDIUM"])
    assert per_race["gap_HARD_vs_MEDIUM"].median() == pytest.approx(true_hard, abs=0.05)
    assert per_race["deg_SOFT"].median() > per_race["deg_MEDIUM"].median() > per_race["deg_HARD"].median()


def test_fit_race_refuses_thin_or_unidentified_data():
    d = pd.DataFrame({"driver_id": ["a"] * 40, "lap_number": range(2, 42), "lap_time_s": 90.0, "compound": "SOFT",
                      "tyre_life": range(1, 41)})
    assert tyres.fit_race(d) is None         # one driver, far too few laps


def test_season_and_circuit_summaries(laps_and_flags):
    laps, flags, _ = laps_and_flags
    per_race = tyres.tyre_by_race(laps, flags.set_index("race_id")["wet"])
    season = tyres.season_tyre_summary(per_race)
    assert list(season["season"]) == [2018, 2019, 2020] and (season["deg_SOFT_median"] > season["deg_HARD_median"]).all()
    circuit = tyres.circuit_tyre_summary(per_race, min_races=3)
    assert set(circuit["circuit_id"]) <= set(lapworld.CIRCUITS) and (circuit["races"] >= 3).all()


def test_pit_loss_recovers_true_loss_and_skips_safety_car_stops(laps_and_flags):
    laps, _, _ = laps_and_flags
    stops = tyres.pit_loss(laps)
    assert 330 < len(stops) < 400
    assert stops["loss_s"].median() == pytest.approx(lapworld.IN_EXTRA + lapworld.OUT_EXTRA, abs=0.6)
    table = tyres.pit_loss_by_circuit(stops, min_stops=30)
    assert len(table) == 3 and (table["median_loss_s"].between(19, 21)).all()


def test_pit_loss_ignores_stops_under_a_safety_car():
    rows = []
    for lap in range(1, 21):
        rows.append({"race_id": 1, "season": 2019, "circuit_id": "x", "circuit_name": "X", "driver_id": "a",
                     "lap_number": lap, "lap_time_s": 90.0 + (5 if lap == 10 else 0) + (15 if lap == 11 else 0),
                     "pit_in_s": 1.0 if lap == 10 else np.nan, "pit_out_s": 1.0 if lap == 11 else np.nan,
                     "track_status": "4" if lap in (10, 11) else "1", "is_accurate": True})
    assert tyres.pit_loss(pd.DataFrame(rows)).empty
    for r in rows:
        r["track_status"] = "1"
    out = tyres.pit_loss(pd.DataFrame(rows))
    assert len(out) == 1 and out["loss_s"].iloc[0] == pytest.approx(20.0)


def test_stop_mix_counts_stops_for_classified_finishers_and_winner_comparison(laps_and_flags):
    laps, flags, w = laps_and_flags
    from src.analysis import grid_checks
    results = w["results"].assign(circuit_id="x")
    mix, winner = tyres.stop_mix(laps, results, flags)
    assert list(mix["season"]) == [2018, 2019, 2020]
    assert ((mix["mean_stops"] >= 1) & (mix["mean_stops"] <= 2)).all()
    shares = mix[["share_0_stops", "share_1_stop", "share_2_stops", "share_3plus_stops"]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    total = winner[winner["season"] == 0].iloc[0]
    assert total["winner_fewer_share"] + total["winner_same_share"] + total["winner_more_share"] == pytest.approx(1.0)


def test_full_run_writes_everything(tmp_path):
    w = lapworld.world(seed=2)
    con = lapworld.build_con(tmp_path / "w.duckdb", w)
    out = tmp_path / "out"
    lines = tyres.run(con, out)
    for name in ["tyre_by_race.csv", "tyre_by_season.csv", "tyre_by_circuit.csv", "pit_loss_stops.csv",
                 "pit_loss_by_circuit.csv", "pit_loss_wet_by_circuit.csv", "pit_loss_dry_vs_wet.csv", "stops_per_car_by_season.csv", "winner_stops_vs_field.csv",
                 "race_flags.csv", "tyre_degradation_by_season.png", "pit_loss_by_circuit.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    for head in ["TYRE MODEL", "PIT LOSS", "STOPS PER CAR", "WINNER'S STOPS"]:
        assert head in text, head


def test_old_2018_compound_names_are_relabelled_by_softness():
    rows = []
    for race, names in ((1, ["ULTRASOFT", "SUPERSOFT", "SOFT"]), (2, ["SOFT", "MEDIUM", "HARD"]),
                        (3, ["ULTRASOFT", "SOFT"]), (4, ["HYPERSOFT", "SOFT", "MEDIUM"])):
        for c in names:
            rows += [{"race_id": race, "compound": c}] * 120
    out = tyres.normalise_compounds(pd.DataFrame(rows))
    unique = lambda r: sorted(out.loc[out["race_id"] == r, "compound"].unique())
    assert unique(1) == ["HARD", "MEDIUM", "SOFT"]
    assert unique(2) == ["HARD", "MEDIUM", "SOFT"]
    assert unique(3) == ["UNKNOWN"]                    # only two compounds: ambiguous, left out
    assert unique(4) == ["HARD", "MEDIUM", "SOFT"]
    first = out[out["race_id"] == 1]["compound"].tolist()
    assert first[:120] == ["SOFT"] * 120 and first[-120:] == ["HARD"] * 120


@pytest.fixture(scope="module")
def confounded():
    """Softs early, hards late, non-linear track improvement, early traffic for the back of the grid."""
    out = []
    for seed in (1, 2, 3):
        w = lapworld.world(seed=seed, confound=True, races=8, seasons=(2019, 2020, 2021), wet_race=99, sc_races=())
        laps = w["laps"].merge(w["races"], on="race_id")
        laps["circuit_name"] = laps["circuit_id"]
        out.append(tyres.tyre_by_race(laps))
    return pd.concat(out)


def test_model_still_recovers_the_truth_when_strategy_and_track_evolution_are_confounded(confounded):
    pr = confounded
    for c in tyres.DRY:
        assert pr[f"deg_{c}"].median() == pytest.approx(lapworld.WEAR[c], abs=0.015), c
    assert abs(pr["gap_SOFT_vs_MEDIUM"].median()) < 0.1
    assert pr["gap_HARD_vs_MEDIUM"].median() == pytest.approx(0.2, abs=0.1)
    assert pr["lap_trend_s_per_lap"].median() < -0.04       # fuel burn plus a track that keeps improving


def test_findings_report_the_compound_sanity_checks(confounded):
    season = tyres.season_tyre_summary(confounded)
    circuit = tyres.circuit_tyre_summary(confounded, min_races=3)
    lines = tyres.findings(confounded, season, circuit, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    sanity = [l for l in lines if "SANITY" in l]
    assert len(sanity) == 1 and "soft wears faster than hard in" in sanity[0]
    assert circuit[["deg_soft", "deg_medium", "deg_hard"]].notna().all().all()


def _races(n_circuits=8, per_circuit=10, step=0.0, circuit_effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_circuits):
        eff = rng.normal(0, circuit_effect) if circuit_effect else 0.0
        for k in range(per_circuit):
            season = 2018 + (k % 9)
            rows.append({"circuit_id": f"c{c}", "season": season,
                         "deg_MEDIUM": 0.03 + eff + (step if season >= 2022 else 0) + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


def test_era_step_test_finds_a_real_step_and_not_a_fake_one():
    real = tyres.era_step_test(_races(step=0.03), "deg_MEDIUM", n_perm=500)
    fake = tyres.era_step_test(_races(step=0.0, seed=4), "deg_MEDIUM", n_perm=500)
    assert real["difference"] == pytest.approx(0.03, abs=0.015) and real["p_value"] < 0.02
    assert fake["p_value"] > 0.05
    assert tyres.era_step_test(_races().query("season < 2022"), "deg_MEDIUM") is None


def test_era_step_ignores_a_change_in_which_circuits_were_raced():
    """Late seasons visit only the high-wear circuits: the raw medians differ, but inside each circuit nothing changed."""
    rng = np.random.default_rng(1)
    rows = []
    for c in range(8):
        base = 0.01 if c < 4 else 0.08
        for season in range(2018, 2027):
            if season >= 2022 and c < 4 and season != 2022:
                continue
            rows.append({"circuit_id": f"c{c}", "season": season, "deg_MEDIUM": base + rng.normal(0, 0.005)})
    df = pd.DataFrame(rows)
    raw_gap = df[df.season >= 2022]["deg_MEDIUM"].median() - df[df.season < 2022]["deg_MEDIUM"].median()
    res = tyres.era_step_test(df, "deg_MEDIUM", n_perm=500)
    assert raw_gap > 0.02 and res["p_value"] > 0.05


def test_split_half_reliability_is_high_when_circuits_differ_and_low_when_they_do_not():
    strong = tyres.split_half_reliability(_races(circuit_effect=0.05, per_circuit=12), "deg_MEDIUM", n_rep=100)
    none = tyres.split_half_reliability(_races(circuit_effect=0.0, per_circuit=12, seed=2), "deg_MEDIUM", n_rep=100)
    assert strong["split_half_corr"] > 0.7 and abs(none["split_half_corr"]) < 0.4
    assert strong["split_half_low"] <= strong["split_half_corr"] <= strong["split_half_high"]


def test_warning_when_compound_pace_gaps_are_not_separated():
    rng = np.random.default_rng(0)
    pr = pd.DataFrame({"race_id": range(40), "season": 2022, "circuit_id": "x", "circuit_name": "X",
                       "lap_trend_s_per_lap": -0.05, "deg_SOFT": 0.05, "deg_MEDIUM": 0.05, "deg_HARD": 0.04,
                       "gap_SOFT_vs_MEDIUM": rng.normal(0, 0.3, 40), "gap_HARD_vs_MEDIUM": rng.normal(0, 0.3, 40)})
    lines = tyres.findings(pr, tyres.season_tyre_summary(pr), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert any("WARNING" in l for l in lines)
    pr["gap_HARD_vs_MEDIUM"] = 0.3
    lines = tyres.findings(pr, tyres.season_tyre_summary(pr), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert not any("WARNING" in l for l in lines)


def test_chart_breaks_the_line_over_a_season_with_too_few_races(tmp_path):
    season = pd.DataFrame({"season": [2018, 2019, 2020, 2021]})
    for c in tyres.DRY:
        season[f"deg_{c}_median"] = 0.05
        season[f"deg_{c}_q25"], season[f"deg_{c}_q75"] = 0.04, 0.06
        season[f"deg_{c}_races"] = [10, 3, 10, 10]
    tyres.chart_degradation(season, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_borderline_wet_flag_marks_thin_wet_races_only():
    def race(rid, wet_laps, rain=0.0):
        return [{"race_id": rid, "lap_number": lap, "driver_id": f"d{d}", "track_status": "1",
                 "compound": "INTERMEDIATE" if (lap * 20 + d) < wet_laps else "SOFT"}
                for lap in range(1, 51) for d in range(20)]
    laps = pd.DataFrame(race(1, 60) + race(2, 300) + race(3, 0))
    f = tyres.race_flags(laps).set_index("race_id")
    assert f.at[1, "wet"] and f.at[1, "wet_borderline"]
    assert f.at[2, "wet"] and not f.at[2, "wet_borderline"]
    assert not f.at[3, "wet"] and not f.at[3, "wet_borderline"]


def test_season_adjusted_split_half_is_higher_when_an_era_step_hides_circuit_differences():
    rng = np.random.default_rng(2)
    rows = []
    for c in range(10):
        for k in range(8):
            season = 2018 + k
            rows.append({"circuit_id": f"c{c}", "season": season,
                         "deg_MEDIUM": 0.02 + 0.004 * c + (0.05 if season >= 2022 else 0) + rng.normal(0, 0.004)})
    df = pd.DataFrame(rows)
    raw = tyres.split_half_reliability(df, "deg_MEDIUM")["split_half_corr"]
    adj = tyres.split_half_reliability(df, "deg_MEDIUM", adjust_season=True)["split_half_corr"]
    assert adj > raw + 0.15 and adj > 0.7


def _wear_races(sizes, seed=0, true_sd=0.01, noise=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for c, n in enumerate(sizes):
        eff = rng.normal(0, true_sd)
        for k in range(n):
            rows.append({"circuit_id": f"c{c}", "circuit_name": f"C{c}", "season": 2018 + (k % 8),
                         "deg_SOFT": np.nan, "deg_MEDIUM": 0.04 + eff + rng.normal(0, noise), "deg_HARD": np.nan})
    return pd.DataFrame(rows)


def test_circuit_wear_shrinkage_pulls_thin_circuits_in_and_keeps_thick_ones():
    df = _wear_races([2, 2, 2, 2, 30, 30, 30, 30], seed=5, true_sd=0.02)
    out = tyres.circuit_wear_shrunk(df).set_index("circuit_id")
    thin = out.loc[["c0", "c1", "c2", "c3"]]
    thick = out.loc[["c4", "c5", "c6", "c7"]]
    assert (thin["shrunk_offset"].abs() <= thin["raw_offset"].abs() + 1e-12).all()
    assert thick["reliability"].min() > thin["reliability"].max()
    assert (thick["shrunk_offset"] - thick["raw_offset"]).abs().max() < 0.2 * thick["raw_offset"].abs().max() + 0.003
    assert set(out["compound"]) == {"medium"}


def test_circuit_wear_shrinks_everything_to_zero_when_circuits_do_not_differ():
    df = _wear_races([6] * 8, seed=8, true_sd=0.0, noise=0.02)
    out = tyres.circuit_wear_shrunk(df)
    # shrinkage is towards the overall mean, so it is the spread that must collapse
    assert out["shrunk_offset"].std() < 0.6 * out["raw_offset"].std()


def test_wear_level_by_era_splits_at_the_step():
    df = _wear_races([10, 10], seed=1)
    df.loc[df["season"] >= 2022, "deg_MEDIUM"] += 0.03
    out = tyres.wear_level_by_era(df).set_index("era")
    assert out.loc["2022-now", "median_wear"] > out.loc["2018-2021", "median_wear"] + 0.015
    assert (out["q25"] <= out["median_wear"]).all() and (out["median_wear"] <= out["q75"]).all()


def test_pit_loss_by_circuit_adds_shrunk_columns_and_pulls_thin_circuits_in():
    rng = np.random.default_rng(3)
    rows = []
    for c, (n_races, mu) in enumerate([(1, 30.0)] + [(8, 22.0 + 0.3 * i) for i in range(6)]):
        for r in range(n_races):
            for d in range(4):
                rows.append({"circuit_id": f"c{c}", "circuit_name": f"C{c}", "race_id": c * 100 + r,
                             "loss_s": mu + rng.normal(0, 1.0)})
    out = tyres.pit_loss_by_circuit(pd.DataFrame(rows), min_stops=3).set_index("circuit_id")
    assert {"loss_shrunk_s", "weight_on_circuit"} <= set(out.columns)
    assert out.loc["c0", "loss_shrunk_s"] < out.loc["c0", "median_loss_s"] - 1.0
    assert out.loc["c0", "weight_on_circuit"] < out.loc["c6", "weight_on_circuit"]


def test_wet_stops_are_split_out_and_circuits_seen_only_in_the_wet_vanish_from_the_dry_table():
    rows = []
    for race, circuit, wet, loss in ((1, "a", False, 21.0), (2, "a", False, 22.0), (3, "a", False, 21.5),
                                     (4, "b", True, 30.0), (5, "b", True, 31.0)):
        for k in range(40):
            rows.append({"race_id": race, "season": 2019, "circuit_id": circuit, "circuit_name": circuit.upper(),
                         "driver_id": f"d{k}", "in_lap": 10, "loss_s": loss + 0.01 * k, "position": float(k + 1)})
    stops = pd.DataFrame(rows)
    wet = pd.Series({1: False, 2: False, 3: False, 4: True, 5: True})
    dry, wet_stops = tyres.split_wet_stops(stops, wet)
    assert set(dry["circuit_id"]) == {"a"} and set(wet_stops["circuit_id"]) == {"b"}
    assert list(tyres.pit_loss_by_circuit(dry, min_stops=30)["circuit_id"]) == ["a"]
    t = tyres.dry_vs_wet_table(dry, wet_stops).set_index("conditions")
    assert t.at["wet", "median_loss_s"] > t.at["dry", "median_loss_s"] + 5
    assert t.at["dry", "races"] == 3 and t.at["wet", "stops"] == 80
    # unknown races (no flag) count as dry rather than being dropped silently
    d2, w2 = tyres.split_wet_stops(stops, pd.Series({4: True, 5: True}))
    assert len(d2) == 120 and len(w2) == 80


def test_line_end_labels_never_overlap_and_keep_their_order():
    ends = [(0.0475, 2026.0, "Soft"), (0.0476, 2026.0, "Medium"), (0.036, 2026.0, "Hard")]
    out = tyres.spread_labels(ends, (0.0, 0.13))
    ys = {text: y for y, _, text in out}
    assert ys["Medium"] - ys["Soft"] >= 0.04 * 0.13 - 1e-9
    assert ys["Soft"] < ys["Medium"] and ys["Hard"] < ys["Soft"]
    far = tyres.spread_labels([(0.02, 2026.0, "A"), (0.09, 2025.0, "B")], (0.0, 0.13))
    assert [round(y, 4) for y, _, _ in far] == [0.02, 0.09]
    assert tyres.spread_labels([], (0, 1)) == []
