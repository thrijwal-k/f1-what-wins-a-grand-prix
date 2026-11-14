"""Stage 2, part 7: who wins, in terms of cars and drivers (not the whole field).

The variance split in car_driver.py describes finishing position across the whole field, which is
dominated by the midfield. The project question is about winning, so this module looks only at the
winner of each race and asks where the winner came from on the grid, using qualifying alone to
define the fastest car of the day:

    pole sitter                       the winner started from the front of the grid
    pole sitter's team-mate           the winner's team was the quickest in qualifying but the other car won
    next-quickest team                the winner's team was second on the grid (best car of each team)
    further back                      the winner's team was third or lower

and, for the driver question, how often the winner was the one of his team's two cars that
qualified ahead. A winner who beats his team-mate on the grid and in the race half of the time tells
you nothing about drivers; a team that wins whoever drives tells you about the car.

Limits. Qualifying position is a noisy yardstick for the car (one lap, track evolution, penalties
move the grid: the grid pole and the qualifying P1 differ in a few percent of races), and "team-mate
qualified ahead" uses the position of the two cars that started, so teams with three or more cars
and shared drives are left out of that last number.

Run:  python -m src.analysis.winners
Writes to outputs/who_wins/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from .grid import ERA_LABELS, SERIES, TEXT, TEXT2, _frame, _plt, era_of, wilson

CATEGORIES = ["Pole sitter", "Pole sitter's team-mate", "Next-quickest team", "Further back"]
MIN_RACES = 15


def load(con) -> pd.DataFrame:
    from . import grid_checks
    raw = grid_checks.load(con)
    d = raw[~raw["is_indy"]].copy()
    grid = pd.to_numeric(d["grid"], errors="coerce")
    d["start"] = pd.to_numeric(d["q_pos"], errors="coerce").where(lambda s: s.notna(), grid.where(grid > 0))
    d["finish"] = pd.to_numeric(d["position"], errors="coerce")
    return d


def win_sources(d: pd.DataFrame) -> pd.DataFrame:
    """One row per race with a winner and a known front of the grid."""
    rows = []
    for race_id, g in d.groupby("race_id"):
        g = g[g["start"].notna()]
        win = g[g["finish"] == 1]
        if g.empty or win.empty:
            continue
        win = win.sort_values("start").iloc[0]
        best = g.groupby("constructor_id")["start"].min().sort_values()
        pole_team = best.index[0]
        pole_driver = g.sort_values("start").iloc[0]["driver_id"]
        team_rank = int(best.rank(method="min")[win["constructor_id"]])
        if win["driver_id"] == pole_driver:
            cat = CATEGORIES[0]
        elif win["constructor_id"] == pole_team:
            cat = CATEGORIES[1]
        elif team_rank == 2:
            cat = CATEGORIES[2]
        else:
            cat = CATEGORIES[3]
        mates = g[(g["constructor_id"] == win["constructor_id"]) & (g["driver_id"] != win["driver_id"])]
        mate_ahead = np.nan
        if len(mates) == 1 and len(g[g["constructor_id"] == win["constructor_id"]]) == 2 \
                and mates.iloc[0]["start"] != win["start"]:
            mate_ahead = float(win["start"] < mates.iloc[0]["start"])
        rows.append({"race_id": race_id, "season": int(win["season"]), "era": era_of(int(win["season"])),
                     "winner": win["driver_id"], "winner_team": win["constructor_id"],
                     "winner_start": float(win["start"]), "winner_team_rank_on_grid": team_rank,
                     "category": cat, "winner_started_ahead_of_teammate": mate_ahead})
    return pd.DataFrame(rows)


def by_era(src: pd.DataFrame) -> pd.DataFrame:
    out = []
    for era in ERA_LABELS:
        s = src[src["era"] == era]
        if len(s) < MIN_RACES:
            continue
        n = len(s)
        rec = {"era": era, "races": n}
        for cat in CATEGORIES:
            k = int((s["category"] == cat).sum())
            lo, hi = wilson(k, n)
            rec[f"{cat}_n"], rec[f"{cat}_share"], rec[f"{cat}_low"], rec[f"{cat}_high"] = k, k / n, lo, hi
        front = int(s["category"].isin(CATEGORIES[:2]).sum())
        top2 = int(s["category"].isin(CATEGORIES[:3]).sum())
        rec["quickest_team_share"], rec["top2_team_share"] = front / n, top2 / n
        rec["quickest_team_low"], rec["quickest_team_high"] = wilson(front, n)
        m = s["winner_started_ahead_of_teammate"].dropna()
        rec["teammate_races"] = len(m)
        rec["winner_ahead_of_teammate_share"] = float(m.mean()) if len(m) else np.nan
        rec["winner_ahead_low"], rec["winner_ahead_high"] = wilson(float(m.sum()), len(m)) if len(m) else (np.nan, np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def chart_sources(table: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    eras = list(table["era"])
    fig, ax = _frame(plt, "Where race winners come from, by how they qualified",
                     "Share of wins by the winner's position on the grid: the pole sitter, the pole sitter's team-mate, "
                     "the next-quickest team, or further back. The first two together are 'the quickest car of the day'.",
                     size=(9.5, 0.62 * len(eras) + 3.2))
    fig.subplots_adjust(top=1 - 1.45 / (0.62 * len(eras) + 3.2), left=0.12, bottom=0.2)
    y = np.arange(len(eras))[::-1]
    left = np.zeros(len(eras))
    colours = [SERIES[0], SERIES[3], SERIES[2], "#c9c8c2"]
    for cat, colour in zip(CATEGORIES, colours):
        w = table[f"{cat}_share"].to_numpy()
        ax.barh(y, w, left=left, height=0.62, color=colour, edgecolor="#fcfcfb", linewidth=2, label=cat)
        for yi, l, wi in zip(y, left, w):
            if wi >= 0.07:
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if colour in (SERIES[0], SERIES[2]) else TEXT)
        left += w
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(y)
    ax.set_yticklabels(eras)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.1), labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)


def findings(table: pd.DataFrame) -> list[str]:
    L = ["WHERE WINNERS COME FROM (qualifying decides which car was quickest; one row per race with a winner)"]
    for _, r in table.iterrows():
        mate = r[f"{CATEGORIES[1]}_share"]
        L.append(f"  {r['era']} ({int(r['races'])} races): pole sitter {r['Pole sitter_share']:.0%}, pole sitter's team-mate "
                 f"{mate:.0%}, next-quickest team {r['Next-quickest team_share']:.0%}, "
                 f"further back {r['Further back_share']:.0%}; quickest team won {r['quickest_team_share']:.0%} "
                 f"({r['quickest_team_low']:.0%}-{r['quickest_team_high']:.0%})")
    L.append("WINNER vs TEAM-MATE ON THE GRID (winners with exactly one team-mate). 50% is NOT the no-driver-effect baseline: starting ahead of "
             "your team-mate helps you win even between equal drivers, and every pole win counts. Read it as an upper bound on the driver's "
             "part, and compare with how often the better-qualified team-mate finishes ahead (grid_vs_result)")
    for _, r in table.iterrows():
        if r["teammate_races"] >= MIN_RACES:
            L.append(f"  {r['era']}: the winner out-qualified his team-mate in {r['winner_ahead_of_teammate_share']:.0%} "
                     f"of {int(r['teammate_races'])} wins ({r['winner_ahead_low']:.0%}-{r['winner_ahead_high']:.0%})")
    L.append("READ WITH CARE: qualifying is a noisy measure of the quickest car, so 'further back' includes wins by a "
             "car that had a bad qualifying; grid penalties and wet qualifying also move it.")
    return L


def run(con, out_dir: Path | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "who_wins"
    out.mkdir(parents=True, exist_ok=True)
    src = win_sources(load(con))
    table = by_era(src)
    src.to_csv(out / "win_sources_by_race.csv", index=False)
    table.to_csv(out / "win_sources_by_era.csv", index=False)
    if len(table):
        chart_sources(table, out / "win_sources_by_era.png")
    lines = findings(table) if len(table) else ["Not enough races with a winner and a known grid."]
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    db = config.db_path()
    if not Path(db).exists():
        sys.exit(f"No database at {db}. Run stage 1 first: python -m src.run_all")
    con = duckdb.connect(str(db), read_only=True)
    try:
        print("\n".join(run(con)))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
