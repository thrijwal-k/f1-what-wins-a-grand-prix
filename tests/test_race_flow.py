import math
import numpy as np
import pandas as pd
import pytest

from src.analysis import race_flow as rf
from . import lapworld


def race(leaders, n_laps=10, pit=None, status=None, retire_after=None):
    """Two cars, a and b. leaders[lap] says who leads after that lap. pit = {driver: in_lap}."""
    pit, status = pit or {}, status or {}
    rows = []
    for lap in range(1, n_laps + 1):
        for d in "ab":
            if retire_after and d in retire_after and lap > retire_after[d]:
                continue
            lead = leaders[lap - 1]
            if retire_after and lead not in [x for x in "ab" if not (x in retire_after and lap > retire_after[x])]:
                lead = "b" if lead == "a" else "a"
            rows.append({"race_id": 1, "season": 2019, "driver_id": d, "lap_number": lap,
                         "position": 1 if d == lead else 2,
                         "pit_in_s": 1.0 if pit.get(d) == lap else np.nan,
                         "pit_out_s": 1.0 if pit.get(d) == lap - 1 else np.nan,
                         "track_status": status.get(lap, "1")})
    return pd.DataFrame(rows)


RES = pd.DataFrame({"driver_id": ["a", "b"], "grid": [1, 2], "position": [2, 1]})   # b wins from second on the grid


def test_pole_sitter_won():
    res = pd.DataFrame({"driver_id": ["a", "b"], "grid": [1, 2], "position": [1, 2]})
    assert rf.classify_lead(race(["a"] * 10), res) == rf.POLE_WON


def test_winner_leading_from_the_first_lap_took_it_at_the_start():
    assert rf.classify_lead(race(["b"] * 10), RES) == "Took the lead at the start"


def test_lead_changes_in_a_pit_cycle():
    laps = race(["a"] * 5 + ["b"] * 5, pit={"a": 5})
    assert rf.classify_lead(laps, RES) == "Pit-stop cycle"


def test_lead_changes_under_a_safety_car_beats_pit_cycle():
    laps = race(["a"] * 5 + ["b"] * 5, pit={"a": 5}, status={5: "4", 6: "4"})
    assert rf.classify_lead(laps, RES) == "Safety car or red flag"


def test_leader_retiring_beats_everything_else():
    laps = race(["a"] * 5 + ["b"] * 5, pit={"a": 5}, status={5: "4"}, retire_after={"a": 5})
    assert rf.classify_lead(laps, RES) == "Leader retired"


def test_plain_pass_on_track():
    assert rf.classify_lead(race(["a"] * 5 + ["b"] * 5), RES) == "On-track pass"


def test_lead_regained_late_counts_the_final_takeover():
    laps = race(["a"] * 3 + ["b"] * 2 + ["a"] * 2 + ["b"] * 3, pit={"a": 7})
    assert rf.classify_lead(laps, RES) == "Pit-stop cycle"


def test_winner_not_leading_at_the_end_is_unclear():
    assert rf.classify_lead(race(["a"] * 10), RES) == "Unclear"


def undercut_race(a_pit=13, b_pit=10, n_laps=30, flip_after=13, status=None):
    rows = []
    for lap in range(1, n_laps + 1):
        a_pos = 1 if lap <= flip_after else 2
        for d, pit, pos in (("a", a_pit, a_pos), ("b", b_pit, 3 - a_pos)):
            rows.append({"race_id": 1, "season": 2019, "driver_id": d, "lap_number": lap, "position": pos,
                         "pit_in_s": 1.0 if lap == pit else np.nan, "pit_out_s": 1.0 if lap == pit + 1 else np.nan,
                         "track_status": (status or {}).get(lap, "1")})
    return pd.DataFrame(rows)


def test_undercut_event_is_found_and_scored():
    ev = rf.undercut_events(undercut_race())
    assert len(ev) == 1
    e = ev[0]
    assert e["role"] == "chaser_first" and e["gap_laps"] == 3 and e["chaser_ahead"] is True


def test_undercut_failed_when_the_chaser_stays_behind():
    ev = rf.undercut_events(undercut_race(flip_after=99))
    assert len(ev) == 1 and ev[0]["chaser_ahead"] is False


def test_leader_stopping_first_is_the_other_role():
    ev = rf.undercut_events(undercut_race(a_pit=10, b_pit=13, flip_after=99))
    assert len(ev) == 1 and ev[0]["role"] == "leader_first" and ev[0]["chaser_ahead"] is False


def test_undercut_ignores_pairs_with_a_safety_car_or_a_long_gap():
    assert rf.undercut_events(undercut_race(status={12: "4"})) == []
    assert rf.undercut_events(undercut_race(a_pit=25, b_pit=10)) == []


def _events(n_races=40, seed=0, a_rate=0.5, b_rate=0.1, gap=1, season=2019, start=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        for role, rate in (("chaser_first", a_rate), ("leader_first", b_rate)):
            for _ in range(3):
                rows.append({"race_id": start + r, "season": season, "role": role, "gap_laps": gap,
                             "chaser_ahead": bool(rng.random() < rate)})
    return pd.DataFrame(rows)


def test_undercut_summary_difference_and_interval():
    s = rf.undercut_summary(_events(), reps=300)
    row = s[(s["period"] == "all") & (s["scope"] == "all stops")].iloc[0]
    assert row["undercut_advantage"] == pytest.approx(row["chaser_ahead_if_first"] - row["chaser_ahead_if_second"])
    assert 0.2 < row["undercut_advantage"] < 0.6
    assert row["advantage_ci_low"] < row["undercut_advantage"] < row["advantage_ci_high"]


def test_undercut_bands_split_by_stop_gap():
    near = _events(gap=1, a_rate=0.4, b_rate=0.3, seed=1)
    far = _events(gap=5, a_rate=0.9, b_rate=0.1, seed=2, start=1000)
    s = rf.undercut_summary(pd.concat([near, far]), reps=300).set_index(["period", "scope"])
    assert s.loc[("all", "stops 1-2 laps apart"), "undercut_advantage"] < 0.3
    assert s.loc[("all", "stops 3+ laps apart"), "undercut_advantage"] > 0.6
    # standardised sits between the bands, and is not just the pooled number
    std = s.loc[("all", "gap-standardised"), "undercut_advantage"]
    assert 0.1 < std < 0.8


def test_undercut_change_row_and_clustered_interval_is_wider():
    ev = pd.concat([_events(season=2019, a_rate=0.3, b_rate=0.3, seed=3),
                    _events(season=2023, a_rate=0.8, b_rate=0.2, seed=4, start=500)])
    s = rf.undercut_summary(ev, reps=400)
    ch = s[(s["period"].str.startswith("change")) & (s["scope"] == "all stops")].iloc[0]
    assert ch["undercut_advantage"] > 0.4 and ch["p_value"] < 0.05
    # six pairs share a race, so the race-clustered interval must be wider than a naive pair-level one
    d = ev[ev["season"] == 2019]
    pa = d[d["role"] == "chaser_first"]["chaser_ahead"]
    pb = d[d["role"] == "leader_first"]["chaser_ahead"]
    naive = 1.96 * math.sqrt(pa.mean() * (1 - pa.mean()) / len(pa) + pb.mean() * (1 - pb.mean()) / len(pb))
    row = s[(s["period"] == "2018-21") & (s["scope"] == "all stops")].iloc[0]
    assert row["advantage_ci_high"] - row["advantage_ci_low"] > 0.5 * 2 * naive


def test_undercut_summary_handles_empty():
    assert rf.undercut_summary(pd.DataFrame()).empty


def test_takeover_shift_tests_detect_a_planted_change_and_stay_quiet_otherwise():
    rng = np.random.default_rng(0)
    cats = rf.CATEGORIES
    same = pd.DataFrame({"race_id": range(120), "season": [2019] * 60 + [2023] * 60, "wet": False,
                         "category": rng.choice([rf.POLE_WON] + cats[:3], size=120)})
    t = rf.takeover_shift_tests(same, reps=300)
    assert (t["p_value"] > 0.001).all() and len(t) == 1 + len(cats) + 1
    planted = same.assign(category=[rf.POLE_WON] * 10 + [cats[1]] * 50 + [rf.POLE_WON] * 50 + [cats[0]] * 10)
    t2 = rf.takeover_shift_tests(planted, reps=300).set_index("test")
    assert t2.loc["Pole sitter won", "p_value"] < 0.001
    assert t2.loc["Mix of pole losses (all categories together)", "p_value"] < 0.05


def test_grip_recovery_finds_a_late_recovery_and_none_when_flat():
    rng = np.random.default_rng(1)
    rows = []
    for r in range(40):
        for grp, shape in (("Dry, 2018-21", {0.0: .9, 0.4: .5, 0.95: .7}),
                           ("Dry, 2022-now", {0.0: .9, 0.4: .5, 0.95: .5})):
            for f, v in shape.items():
                rows.append({"race_id": f"{grp}{r}", "group": grp, "fraction": f, "spearman": v + rng.normal(0, .05)})
    out = rf.grip_recovery(pd.DataFrame(rows), reps=300)
    rec = out[out["measure"].str.startswith("recovery")].set_index("comparison")
    assert rec.loc["Dry, 2018-21", "ci_low"] > 0.1
    assert rec.loc["Dry, 2022-now", "ci_low"] < 0 < rec.loc["Dry, 2022-now", "ci_high"]
    diff = out[out["p_value"].notna()]
    assert len(diff) == 3 and diff[diff["measure"].str.contains("95%")]["p_value"].iloc[0] < 0.05


@pytest.fixture(scope="module")
def world_tables():
    from src.analysis import grid_checks, tyres
    w = lapworld.world(seed=1)
    laps = w["laps"].merge(w["races"], on="race_id")
    laps["circuit_name"] = laps["circuit_id"]
    flags = tyres.race_flags(laps, w["weather"].groupby("race_id")["rainfall"].mean())
    results = w["results"].merge(w["races"][["race_id", "circuit_id"]], on="race_id")
    results["circuit_name"] = results["circuit_id"]
    results["is_indy"] = False
    results["q_pos"] = np.nan
    return laps, results, flags


def test_grip_decays_from_the_start_and_is_high_after_lap_one(world_tables):
    laps, results, flags = world_tables
    decay = rf.grip_decay(laps, results, flags)
    assert set(decay["group"]) == {"Dry, 2018-21", "Wet races"}
    summary = rf.grip_summary(decay)
    d = summary[summary["group"] == "Dry, 2018-21"].set_index("fraction")["mean_spearman"]
    assert d.loc[0.0] > 0.85 and d.loc[0.0] > d.loc[0.4] + 0.2
    assert (decay["n_running"] >= 8).all()


def test_takeover_table_covers_every_race_and_categories_sum(world_tables):
    laps, results, flags = world_tables
    per_race = rf.lead_takeovers(laps, results, flags)
    assert len(per_race) == 18 and set(per_race["category"]) <= set([rf.POLE_WON] + rf.CATEGORIES)
    s = rf.takeover_summary(per_race)
    a = s[s["period"] == "all"]
    assert a["races"].sum() == 18
    assert a.loc[a["category"] != rf.POLE_WON, "share_of_pole_losses"].sum() == pytest.approx(1.0)


def test_conditions_split_dry_neutralised_and_wet(world_tables):
    laps, results, flags = world_tables
    table = rf.conditions_table(results, flags).set_index("condition")
    assert set(table.index) == {"Dry, green throughout", "Dry, with safety car, VSC or red flag", "Wet"}
    assert table["races"].sum() == 18 and table.loc["Wet", "races"] == 3
    assert (table["pole_ci_low"] <= table["pole_win_rate"]).all()


def test_full_run_writes_everything(tmp_path):
    w = lapworld.world(seed=3)
    con = lapworld.build_con(tmp_path / "w.duckdb", w)
    out = tmp_path / "out"
    lines = rf.run(con, out)
    for name in ["grip_decay.csv", "grip_decay_by_race.csv", "lead_takeover.csv", "lead_takeover_by_race.csv",
                 "undercut.csv", "undercut_events.csv", "conditions.csv", "grip_recovery.csv", "lead_takeover_shift_tests.csv", "grip_decay.png", "how_pole_is_lost.png",
                 "races_without_laps.csv", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    for head in ["LAP-DATA COVERAGE", "GRIP DECAY", "HOW THE WINNER BEAT", "UNDERCUT", "CONDITIONS"]:
        assert head in text, head


def test_lap_coverage_lists_races_with_results_but_no_laps():
    results = pd.DataFrame({"race_id": [201801, 201802, 201803, 201803], "season": 2018})
    laps = pd.DataFrame({"race_id": [201801, 201803]})
    out = rf.lap_coverage(results, laps)
    assert list(out["race_id"]) == [201802]


def test_borderline_wet_is_reported_as_a_subset_of_wet(world_tables):
    laps, results, flags = world_tables
    flags = flags.copy()
    flags["wet_borderline"] = False
    wet_ids = flags.index[flags["wet"]] if flags.index.name == "race_id" else flags.loc[flags["wet"], "race_id"]
    flags.loc[flags["race_id"] == list(wet_ids)[0], "wet_borderline"] = True if "race_id" in flags.columns else None
    table = rf.conditions_table(results, flags).set_index("condition")
    assert rf.BORDERLINE_LABEL in table.index
    assert table.loc[rf.BORDERLINE_LABEL, "races"] == 1 and table.loc["Wet", "races"] == 3


def test_race_without_flags_is_left_out_of_conditions(world_tables):
    laps, results, flags = world_tables
    table = rf.conditions_table(results, flags[flags["race_id"] != flags["race_id"].iloc[0]])
    assert "Unknown" not in set(table["condition"]) and table["races"].sum() <= 17
