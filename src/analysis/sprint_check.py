"""Stage 2, part 6: do the 2021-22 sprint weekends distort the grid analysis?

Run:  python -m src.analysis.sprint_check
Writes tables and findings.txt to outputs/sprint_check/.

Scope of every analysis in this project
  - Parts 1-6 analyse only the Sunday Grand Prix, and the lap and tyre data (FastF1) is
    fetched for the race session only. Points are not used. Sprint results are loaded
    (sprint_results table) and analysed in part 7, python -m src.analysis.sprints.
  - On a sprint weekend the Sunday grid is whatever the results table says it was:
      2021 and 2022: the finishing order of the Saturday sprint (so the "pole sitter" on
        the grid is the sprint winner, not the fastest qualifier);
      2023 onward: Friday qualifying, as on any other weekend (the sprint has its own
        shootout and does not feed the Sunday grid), so those races need no special case.
  - So the only races with a different grid definition are the six sprint weekends of
    2021 and 2022. This check measures how much they matter.

The six races are taken from the sprint_results table (every sprint weekend up to 2022). A hand-
written list of the same six is kept as a fallback for a database built before sprints were loaded,
and the output says whether the two agree: Britain, Italy and Brazil in 2021; Emilia-Romagna,
Austria and Brazil in 2022.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import grid, grid_checks

SPRINT_GRID_RACES = {
    202110: "2021 British GP", 202114: "2021 Italian GP", 202119: "2021 Brazilian GP",
    202204: "2022 Emilia Romagna GP", 202211: "2022 Austrian GP", 202221: "2022 Brazilian GP",
}
SCOPES = [("2018 to now", 2018, 9999), ("2021 and 2022", 2021, 2022)]


def sprint_races(df: pd.DataFrame, races: pd.DataFrame, ids: dict | None = None) -> pd.DataFrame:
    """The sprint-grid races: who started first on the grid, who was fastest in qualifying, who won."""
    ids = SPRINT_GRID_RACES if ids is None else ids
    rows = []
    for race_id, label in ids.items():
        g = df[df["race_id"] == race_id]
        if g.empty:
            continue
        grid1 = g[g["grid"] == 1]
        q1 = g[g["q_pos"] == 1] if g["q_pos"].notna().any() else g.iloc[0:0]
        win = g[g["position"] == 1]
        r = races[races["race_id"] == race_id]
        rows.append({
            "race_id": race_id, "race": label,
            "grid_pole": grid1["driver_id"].iloc[0] if len(grid1) else None,
            "qualifying_p1": q1["driver_id"].iloc[0] if len(q1) else None,
            "winner": win["driver_id"].iloc[0] if len(win) else None,
            "grid_pole_is_qualifying_p1": bool(len(grid1) and len(q1) and grid1["driver_id"].iloc[0] == q1["driver_id"].iloc[0]),
            "grid_pole_won": bool(len(grid1) and len(win) and grid1["driver_id"].iloc[0] == win["driver_id"].iloc[0]),
            "qualifying_p1_won": bool(len(q1) and len(win) and q1["driver_id"].iloc[0] == win["driver_id"].iloc[0]),
            "spearman": float(r["spearman"].iloc[0]) if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def sensitivity(races: pd.DataFrame, ids: dict | None = None, scopes=SCOPES) -> pd.DataFrame:
    """Headline grid numbers with all races, and without the sprint-grid races."""
    ids = SPRINT_GRID_RACES if ids is None else ids
    rows = []
    for scope, first, last in scopes:
        part = races[(races["season"] >= first) & (races["season"] <= last)]
        for version, d in [("all races", part), ("without the sprint-grid races", part[~part["race_id"].isin(ids)])]:
            pole = d[d["has_pole"]]
            lo, hi = grid.wilson(float(pole["pole_won"].sum()), len(pole))
            rows.append({"scope": scope, "version": version, "races": len(d),
                         "pole_win_rate": float(pole["pole_won"].mean()) if len(pole) else np.nan,
                         "pole_ci_low": lo, "pole_ci_high": hi,
                         "mean_spearman": float(d["spearman"].mean()) if len(d) else np.nan})
    return pd.DataFrame(rows)


def findings(sprints: pd.DataFrame, sens: pd.DataFrame) -> list[str]:
    L = ["SCOPE: this check uses Sunday Grand Prix races only (sprints themselves are analysed by python -m src.analysis.sprints)."]
    L.append(f"SPRINT-GRID RACES: {len(sprints)} of the 2021-22 sprint weekends found in the data")
    for _, r in sprints.iterrows():
        same = "same driver" if r["grid_pole_is_qualifying_p1"] else "different driver"
        L.append(f"  {r['race']}: grid pole {r['grid_pole']} ({same} as fastest in qualifying, {r['qualifying_p1']}); "
                 f"winner {r['winner']}; grid correlation {r['spearman']:.2f}")
    if len(sprints):
        L.append(f"  grid pole won {int(sprints['grid_pole_won'].sum())} of {len(sprints)}; "
                 f"the qualifying P1 driver won {int(sprints['qualifying_p1_won'].sum())} of {len(sprints)}; "
                 f"the two were different drivers in {int((~sprints['grid_pole_is_qualifying_p1']).sum())}")
    L.append("DOES EXCLUDING THEM CHANGE THE HEADLINES?")
    for scope in sens["scope"].unique():
        a = sens[(sens["scope"] == scope) & (sens["version"] == "all races")]
        b = sens[(sens["scope"] == scope) & (sens["version"] != "all races")]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        L.append(f"  {scope}: pole won {a['pole_win_rate']:.1%} of {int(a['races'])} races with them, "
                 f"{b['pole_win_rate']:.1%} of {int(b['races'])} without ({(b['pole_win_rate'] - a['pole_win_rate']) * 100:+.1f} points); "
                 f"grid correlation {a['mean_spearman']:.3f} against {b['mean_spearman']:.3f}")
    L.append("READ WITH CARE: six races is too few to say anything about sprint weekends themselves. This only shows whether "
             "their different grid definition moves the headline numbers; it does not study sprint racing.")
    return L


def derive_ids(con, last_season: int = 2022) -> dict | None:
    """Sprint-grid races from the sprint_results table, labelled from the calendar. None if no sprint data."""
    has = bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'sprint_results'").fetchone()[0])
    if not has:
        return None
    d = con.execute("""SELECT DISTINCT s.race_id, s.season, coalesce(ra.race_name, ra.circuit_id, '') AS name
                       FROM sprint_results s LEFT JOIN races ra ON ra.race_id = s.race_id
                       WHERE s.season <= ? ORDER BY s.race_id""", [last_season]).df() \
        if _has_col(con, "races", "race_name") else \
        con.execute("""SELECT DISTINCT s.race_id, s.season, coalesce(ra.circuit_id, '') AS name
                       FROM sprint_results s LEFT JOIN races ra ON ra.race_id = s.race_id
                       WHERE s.season <= ? ORDER BY s.race_id""", [last_season]).df()
    if d.empty:
        return None
    return {int(r.race_id): f"{int(r.season)} {r['name']}".strip() for _, r in d.iterrows()}


def _has_col(con, table: str, col: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                            [table, col]).fetchone()[0])


def run(con, out_dir: Path | None = None, ids: dict | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sprint_check"
    out.mkdir(parents=True, exist_ok=True)
    source_note = None
    if ids is None:
        derived = derive_ids(con)
        if derived is None:
            source_note = ("Sprint-grid races: hand-written list of six (no sprint results in the database; "
                           "run python -m src.run_all --skip-fastf1 to load them).")
        else:
            ids = derived
            same = set(derived) == set(SPRINT_GRID_RACES)
            source_note = (f"Sprint-grid races: {len(derived)} taken from the sprint results table; "
                           + ("they match the hand-written list of six." if same else
                              f"they DO NOT match the hand-written list of six (only in data: {sorted(set(derived) - set(SPRINT_GRID_RACES))}; "
                              f"only in list: {sorted(set(SPRINT_GRID_RACES) - set(derived))})."))
    raw = grid_checks.load(con)
    df = grid_checks.prepare_checks(raw, drop_indy=True)
    races = grid.race_summary(df)
    sprints = sprint_races(df, races, ids)
    sens = sensitivity(races, ids)
    sprints.to_csv(out / "sprint_grid_races.csv", index=False)
    sens.to_csv(out / "sprint_sensitivity.csv", index=False)
    lines = findings(sprints, sens)
    if source_note:
        lines = [source_note] + lines
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    path = config.db_path()
    if not path.exists():
        print(f"No database at {path}. Run stage 1 first.")
        return 1
    con = duckdb.connect(str(path), read_only=True)
    try:
        lines = run(con)
    finally:
        con.close()
    print("\n".join(lines))
    print(f"\nTables written to {config.outputs_dir() / 'sprint_check'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
