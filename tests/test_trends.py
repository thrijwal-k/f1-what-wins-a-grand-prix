import numpy as np
import pandas as pd
import pytest

from src.analysis import trends as tr
from tests.test_grid_checks import make_con, world


def fake_seasons(step_metric=None, step=0.0, step_year=2014, first=1950, last=2025, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(first, last + 1):
        row = {"season": s, "races": 20}
        for key in tr.METRIC_KEYS:
            v = 0.5 + rng.normal(0, 0.02)
            if key == step_metric and s >= step_year:
                v += step
            row[key] = v
        rows.append(row)
    return pd.DataFrame(rows)


def test_bh_qvalues_known_values():
    q = tr.bh_qvalues(np.array([0.01, 0.04, 0.03, 0.20, np.nan]))
    assert q[0] == pytest.approx(0.04) and q[2] == pytest.approx(0.0533333, abs=1e-6)
    assert q[1] == pytest.approx(0.0533333, abs=1e-6) and q[3] == pytest.approx(0.2)
    assert np.isnan(q[4])


def test_placebo_breaks_stay_clear_of_real_rule_years():
    seasons = pd.Index(range(1990, 2026))
    out = tr._placebo_breaks(seasons, 3, [1994, 2009, 2014])
    for b in out:
        assert not any(b - 3 <= q <= b + 2 for q in (1994, 2009, 2014))
    assert 1993 not in out and 2014 not in out and min(out) >= 1993 and max(out) <= 2023


def test_planted_step_is_found_and_only_there():
    seasons = fake_seasons(step_metric="finish_rate", step=0.15, step_year=2014)
    t = tr.rule_change_tests(seasons)
    hit = t[(t["window"] == 3) & (t["rule_year"] == 2014) & (t["metric"] == "finish_rate")].iloc[0]
    assert hit["tested"] and hit["z"] > 3 and hit["z_modern"] > 3 and hit["p_value"] < 0.05 and hit["q_value"] < 0.05 and hit["unusual"]
    assert hit["change"] == pytest.approx(0.15, abs=0.03)
    main = t[(t["window"] == 3) & t["tested"]]
    flagged = main[main["unusual"]]
    assert set(zip(flagged["rule_year"], flagged["metric"])) == {(2014, "finish_rate")}


def test_no_effect_means_nothing_is_flagged():
    t = tr.rule_change_tests(fake_seasons(seed=5))
    main = t[(t["window"] == 3) & t["tested"]]
    assert not main["unusual"].any()
    assert len(main) == len(tr.METRIC_KEYS) * 5   # 2026 is not testable (no full window)


def test_unfinished_season_is_shown_but_not_tested():
    seasons = fake_seasons(last=2026)
    t = tr.rule_change_tests(seasons)
    r = t[(t["rule_year"] == 2026) & (t["window"] == 3)]
    assert len(r) == len(tr.METRIC_KEYS) and not r["tested"].any()
    assert r["p_value"].isna().all() and (r["post_seasons"] == 1).all()


def test_stability_counts_other_windows():
    seasons = fake_seasons(step_metric="grid_corr", step=0.2, step_year=2017)
    t = tr.rule_change_tests(seasons)
    s = tr.stability(t)
    row = s[(s["rule_year"] == 2017) & (s["metric"] == "grid_corr")].iloc[0]
    assert row["other_windows"] == 2 and row["same_direction"] == 2 and row["also_unusual"] >= 1


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    rows, quali = world(seasons=range(1990, 2027), noise={c: 3.0 for c in "abcdef"})
    con = make_con(tmp_path_factory.mktemp("t") / "f.duckdb", rows, quali)
    out = tmp_path_factory.mktemp("o")
    lines = tr.run(con, out)
    return out, lines


def test_season_measures_have_sensible_ranges(pipeline):
    out, _ = pipeline
    s = pd.read_csv(out / "season_measures.csv")
    assert len(s) == 37
    assert s["pole_win_rate"].between(0, 1).all() and s["finish_rate"].between(0, 1).all()
    assert s["grid_corr"].between(-1, 1).all() and (s["winner_variety"] > 0).all()
    assert s["finish_rate"].round(3).eq(0.8).all()     # the fake world classifies 16 of 20 every race


def test_run_writes_everything_and_reports(pipeline):
    out, lines = pipeline
    for name in ["season_measures.csv", "rule_change_tests.csv", "rule_change_stability.csv",
                 "what_decides_a_race_by_season.png", "rule_change_effects.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0
    text = "\n".join(lines)
    assert "RULE CHANGES" in text and "READ WITH CARE" in text and "SEASONS:" in text
    assert "2026" in text and "not tested" in text and "WHAT THE TEST COULD SEE" in text


def test_nearest_misses_are_unflagged_and_sorted_by_q():
    t = tr.rule_change_tests(fake_seasons(step_metric="finish_rate", step=0.15, step_year=2014, seed=2))
    miss = tr.nearest_misses(t, 3)
    assert len(miss) == 3 and not miss["unusual"].any()
    assert miss["q_value"].is_monotonic_increasing


def test_trend_adjusted_check_ignores_a_pure_trend_but_finds_a_jump():
    first, last = 1950, 2025
    rng = np.random.default_rng(3)
    rows = []
    for s in range(first, last + 1):
        row = {"season": s, "races": 20}
        for key in tr.METRIC_KEYS:
            row[key] = 0.2 + 0.004 * (s - first) + rng.normal(0, 0.01) if key == "finish_rate" else 0.5 + rng.normal(0, 0.01)
        if s >= 2014:
            row["pole_win_rate"] += 0.12
        rows.append(row)
    t = tr.trend_adjusted_tests(pd.DataFrame(rows)).set_index(["metric", "rule_year"])
    assert abs(t.loc[("finish_rate", 2014), "z"]) < 2.5          # a steady trend is not a rule effect
    assert t.loc[("pole_win_rate", 2014), "z"] > 3 and t.loc[("pole_win_rate", 2014), "p_value"] < 0.1


def test_findings_mention_nearest_misses_and_trend_check():
    seasons = fake_seasons(seed=4)
    tests = tr.rule_change_tests(seasons)
    text = "\n".join(tr.findings(seasons, tests, tr.stability(tests), trend_tests=tr.trend_adjusted_tests(seasons)))
    assert "closest to being flagged" in text and "trend-adjusted check" in text
