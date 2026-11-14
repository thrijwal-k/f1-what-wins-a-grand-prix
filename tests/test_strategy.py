import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analysis import strategy as st, what_decides as wd
from . import whatworld


def _pairs(**kw):
    res, qual, pits, laps = whatworld.world(**kw)
    return st.build_pairs(wd.build_table(res, qual, pits, laps))


def test_pairs_are_two_car_lead_lap_finishers_with_signed_differences():
    p = _pairs(seed=1, races=16)
    assert len(p) > 100 and (p["driver_a"] < p["driver_b"]).all()
    assert p.groupby(["race_id", "constructor_id"]).size().max() == 1
    # the grid and gain columns are differences of positions, so they are integers and within the field size
    assert p["gain"].abs().max() < 16 and p["grid"].abs().max() < 16


def test_a_built_in_stop_effect_is_found_and_no_effect_is_not():
    with_effect = _pairs(seed=2, races=40, strategy_effect=1.0)
    none = _pairs(seed=2, races=40, strategy_effect=0.0)
    e1 = st.fit_effects(with_effect, st.usable_terms(with_effect), n_boot=200).set_index("term")
    e0 = st.fit_effects(none, st.usable_terms(none), n_boot=200).set_index("term")
    assert e1.loc["stops", "low"] > 0 and e1.loc["stops", "estimate"] > 1.0
    assert e0.loc["stops", "low"] < 0 < e0.loc["stops", "high"]
    assert e1.loc["grid", "low"] > 0          # starting ahead helps


def test_the_grid_term_is_the_yardstick_and_signs_flip_cleanly():
    p = _pairs(seed=3, races=30)
    terms = st.usable_terms(p)
    a = st.fit_effects(p, terms, n_boot=50).set_index("term")["estimate"]
    flipped = p.copy()
    for c in ["gain", "grid", "stops", "first", "pit_time", "free", "unlucky"]:
        flipped[c] = -flipped[c]
    b = st.fit_effects(flipped, terms, n_boot=50).set_index("term")["estimate"]
    assert np.allclose(a, b, atol=1e-9)             # swapping the two cars changes nothing


def test_safety_car_terms_need_lap_data():
    p = _pairs(seed=4, races=20, with_laps=False)
    assert "free" not in st.usable_terms(p) and "unlucky" not in st.usable_terms(p)
    q = _pairs(seed=4, races=20, free_bonus=1.0, sc_prob=0.8)
    assert "free" in st.usable_terms(q)


def test_plain_counts_and_effects_table_have_every_period_that_has_data():
    p = _pairs(seed=5, races=40, strategy_effect=1.0)
    eff = st.effects_table(p, n_boot=50)
    assert set(eff["period"]) == {"2018-21", "2011-now"}          # the world only has 2018-21 seasons
    assert {"all pairs", f"started within {st.NEAR_GRID} places", "no slow stops"} <= set(eff["variant"])
    sp = st.split_table(p)
    row = sp[(sp["period"] == "2018-21") & sp["comparison"].str.startswith("more stops")].iloc[0]
    assert row["pairs"] > 20 and row["share"] > 0.5            # more stops really did help in this world


def test_full_run_and_thin_data(tmp_path):
    res, qual, pits, laps = whatworld.world(seed=6, races=40, strategy_effect=1.0)
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    for name, df in (("results", res), ("qualifying", qual), ("pit_stops", pits), ("laps", laps)):
        con.register("df_", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_")
        con.unregister("df_")
    out = tmp_path / "out"
    lines = st.run(con, out, n_boot=50)
    for name in ["teammate_pairs.csv", "strategy_effects.csv", "strategy_plain_counts.csv", "strategy_effects.png", "findings.txt"]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    text = "\n".join(lines)
    assert "STRATEGY AMONG TEAM-MATES" in text and "One more stop than the team-mate" in text and "READ WITH CARE" in text
    res2, qual2, pits2, laps2 = whatworld.world(seed=7, races=2, seasons=(2018,))
    con2 = duckdb.connect(str(tmp_path / "w2.duckdb"))
    for name, df in (("results", res2), ("qualifying", qual2), ("pit_stops", pits2), ("laps", laps2)):
        con2.register("df_", df)
        con2.execute(f"CREATE TABLE {name} AS SELECT * FROM df_")
        con2.unregister("df_")
    assert "Not enough" in st.run(con2, tmp_path / "o2", n_boot=10)[0]
