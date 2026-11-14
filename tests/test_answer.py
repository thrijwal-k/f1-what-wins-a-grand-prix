import pandas as pd
import pytest

from src.analysis import answer


def _write(root, folder, name, df):
    (root / folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(root / folder / name, index=False)


def _decomp():
    rows = []
    for p, car in (("2018-21", 0.30), ("2022-now", 0.27)):
        rows.append({"period": p, "car_share": car, "car_low": car - 0.02, "car_high": car + 0.02, "driver_share": 0.08,
                     "driver_low": 0.07, "driver_high": 0.11, "Qualifying form_share": 0.03, "Qualifying form_low": 0.02,
                     "Qualifying form_high": 0.04, "Strategy_share": 0.04, "Strategy_low": 0.03, "Strategy_high": 0.05,
                     "Safety-car timing_share": 0.01, "Safety-car timing_low": 0.0, "Safety-car timing_high": 0.02,
                     "Mechanical retirements_share": 0.2, "Mechanical retirements_low": 0.18, "Mechanical retirements_high": 0.22,
                     "Accidents_share": 0.1, "Accidents_low": 0.08, "Accidents_high": 0.12,
                     "unexplained_share": 0.27, "unexplained_low": 0.24, "unexplained_high": 0.3})
    return pd.DataFrame(rows)


def test_table_and_text_come_from_the_other_outputs(tmp_path):
    _write(tmp_path, "what_decides", "decomposition_by_period.csv", _decomp())
    _write(tmp_path, "car_vs_driver_robust", "driver_share_intervals.csv",
           pd.DataFrame([{"era": "2022-now", "outcome": "race", "driver_allowed_low": 0.02, "driver_allowed_high": 0.19}]))
    _write(tmp_path, "strategy_teammates", "strategy_effects.csv",
           pd.DataFrame([{"period": "2011-now", "variant": "all pairs", "term": t, "estimate": e, "low": e - 0.3, "high": e + 0.3,
                          "pairs": 900, "races": 300} for t, e in (("stops", 0.5), ("grid", 0.4))]))
    _write(tmp_path, "who_wins", "win_sources_by_era.csv",
           pd.DataFrame([{"era": "2022-now", "races": 109, "Pole sitter_share": 0.6, "quickest_team_share": 0.65, "Further back_share": 0.08}]))
    lines = answer.run(tmp_path)
    text = "\n".join(lines)
    assert "car (team and season) 27% (25%-29%)" in text and "driver 8% (7%-11%)" in text
    assert "allowing for a wrong model" in text and "2%-19%" in text
    assert "one more stop than the team-mate +0.50 places" in text and "the pole sitter won 60%" in text
    assert "MISSING" not in text
    table = pd.read_csv(tmp_path / "stage2_answer" / "answer_table.csv")
    assert set(table["period"]) == {"2018-21", "2022-now"} and len(table) == 16
    assert table.loc[(table["period"] == "2022-now") & (table["factor"] == "Driver"), "allowed_high"].iloc[0] == pytest.approx(0.19)


def test_missing_inputs_are_named_not_fatal(tmp_path):
    lines = answer.run(tmp_path)
    assert "has not been run" in lines[0] and "MISSING INPUTS" in lines[-1]
    _write(tmp_path, "what_decides", "decomposition_by_period.csv", _decomp())
    text = "\n".join(answer.run(tmp_path))
    assert "2022-now: car (team and season)" in text and "MISSING INPUTS (run these first): car_vs_driver_robust, strategy_teammates, who_wins" in text
