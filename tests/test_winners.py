import duckdb
import numpy as np
import pandas as pd

from src.analysis import winners


def race(rid, season, rows):
    """rows: (driver, team, start, finish)."""
    return [{"race_id": rid, "season": season, "driver_id": d, "constructor_id": t, "start": s, "finish": f,
             "is_indy": False, "grid": s, "q_pos": s, "position": f, "classified": True, "circuit_id": "x",
             "circuit_name": "X", "round": 1} for d, t, s, f in rows]


PATTERNS = {
    # pole sitter wins
    "pole": [("a1", "A", 1, 1), ("a2", "A", 3, 2), ("b1", "B", 2, 3), ("b2", "B", 4, 4), ("c1", "C", 5, 5), ("c2", "C", 6, 6)],
    # pole sitter's team-mate wins, having qualified behind him
    "mate": [("a1", "A", 1, 2), ("a2", "A", 3, 1), ("b1", "B", 2, 3), ("b2", "B", 4, 4), ("c1", "C", 5, 5), ("c2", "C", 6, 6)],
    # the second-quickest team wins, with the better qualifier of that team
    "second": [("a1", "A", 1, 2), ("a2", "A", 3, 3), ("b1", "B", 2, 1), ("b2", "B", 4, 4), ("c1", "C", 5, 5), ("c2", "C", 6, 6)],
    # a team that qualified third or lower wins
    "back": [("a1", "A", 1, 2), ("a2", "A", 3, 3), ("b1", "B", 2, 4), ("b2", "B", 4, 5), ("c1", "C", 5, 1), ("c2", "C", 6, 6)],
}


def world(counts, season=2012):
    rows, rid = [], season * 100
    for name, k in counts.items():
        for _ in range(k):
            rid += 1
            rows += race(rid, season, PATTERNS[name])
    return pd.DataFrame(rows)


def test_each_pattern_lands_in_its_category_and_teammate_order_is_recorded():
    src = winners.win_sources(world({"pole": 1, "mate": 1, "second": 1, "back": 1}))
    assert list(src["category"]) == ["Pole sitter", "Pole sitter's team-mate", "Next-quickest team", "Further back"]
    assert list(src["winner_team_rank_on_grid"]) == [1, 1, 2, 3]
    ahead = list(src["winner_started_ahead_of_teammate"])
    assert ahead == [1.0, 0.0, 1.0, 1.0]


def test_by_era_shares_sum_to_one_and_quickest_team_is_pole_plus_teammate():
    src = winners.win_sources(world({"pole": 8, "mate": 4, "second": 5, "back": 3}))
    t = winners.by_era(src).iloc[0]
    assert t["races"] == 20 and t["era"] == "2010-21"
    shares = sum(t[f"{c}_share"] for c in winners.CATEGORIES)
    assert abs(shares - 1) < 1e-9
    assert abs(t["quickest_team_share"] - 12 / 20) < 1e-9 and abs(t["top2_team_share"] - 17 / 20) < 1e-9
    assert t["quickest_team_low"] < 0.6 < t["quickest_team_high"]
    assert abs(t["winner_ahead_of_teammate_share"] - 16 / 20) < 1e-9


def test_era_with_too_few_races_is_left_out_and_races_without_a_winner_or_grid_are_skipped():
    src = winners.win_sources(world({"pole": 5}))
    assert winners.by_era(src).empty
    no_win = world({"pole": 1}).assign(finish=lambda d: d["finish"] + 1)
    no_grid = world({"pole": 1}).assign(start=np.nan)
    assert winners.win_sources(no_win).empty and winners.win_sources(no_grid).empty


def test_shared_drive_and_three_car_teams_give_no_teammate_number():
    rows = race(201201, 2012, [("a1", "A", 1, 1), ("a2", "A", 3, 2), ("a3", "A", 5, 4), ("b1", "B", 2, 3)])
    src = winners.win_sources(pd.DataFrame(rows))
    assert src["winner_started_ahead_of_teammate"].isna().all()


def test_full_run_writes_everything(tmp_path):
    d = world({"pole": 8, "mate": 4, "second": 5, "back": 3})
    res = d.rename(columns={"finish": "position"})[["race_id", "season", "round", "driver_id", "constructor_id", "grid", "position"]]
    res = res.assign(classified=True)
    races = pd.DataFrame({"race_id": sorted(d["race_id"].unique())})
    races["season"] = races["race_id"] // 100
    races["circuit_id"] = "x"
    qual = d[["race_id", "driver_id", "start"]].rename(columns={"start": "position"})
    circuits = pd.DataFrame({"circuit_id": ["x"], "circuit_name": ["X"]})
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    for name, df in (("results", res), ("races", races), ("qualifying", qual), ("circuits", circuits)):
        con.register(name + "_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    out = tmp_path / "out"
    lines = winners.run(con, out)
    for name in ["win_sources_by_era.csv", "win_sources_by_race.csv", "win_sources_by_era.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    assert "WHERE WINNERS COME FROM" in text and "WINNER vs TEAM-MATE" in text and "2010-21 (20 races)" in text
