"""Stage 2, part 1: how much does the starting grid decide the race?

Questions answered (all from the DuckDB database):
  - Expected finishing position from each grid slot, by era.
  - How tightly grid order predicts finishing order, race by race, season by
    season and circuit by circuit (Spearman rank correlation).
  - How often pole wins, with a 95% interval, so that "Monaco vs Monza" is not
    decided by a handful of races.

Run:  python -m src.analysis.grid
Writes tables (CSV) and charts (PNG) to outputs/grid_vs_result/.

Method notes
  - One row per finishing position per race: before 1965 two drivers could share
    a car and a position, and they are counted once.
  - Grid 0 (pit-lane start) is treated as starting behind the last grid slot.
  - Races with fewer than 8 classified-or-not finishers are ignored.
  - The Indianapolis 500 is excluded before 1961: it counted towards the championship, but with
    33-car fields of one-off entries it is a different kind of race (see grid_checks.py).
"""
from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config

ERAS = [(1950, 1969, "1950-69"), (1970, 1989, "1970-89"), (1990, 2009, "1990-2009"),
        (2010, 2021, "2010-21"), (2022, 9999, "2022-now")]
ERA_LABELS = [label for _, _, label in ERAS]
MIN_STARTERS = 8
MAX_SLOT = 20
HIGHLIGHT_CIRCUITS = ("monaco", "monza")


def era_of(season: int) -> str:
    for first, last, label in ERAS:
        if first <= season <= last:
            return label
    return "other"


def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if not n:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# ------------------------------------------------------------------ data

def load_results(con) -> pd.DataFrame:
    return con.execute("""
        SELECT r.race_id, r.season, r.round, ra.circuit_id,
               coalesce(c.circuit_name, ra.circuit_id) AS circuit_name,
               r.driver_id, r.grid, r.position
        FROM results r
        JOIN races ra ON ra.race_id = r.race_id
        LEFT JOIN circuits c ON c.circuit_id = ra.circuit_id
        WHERE r.position IS NOT NULL
          AND NOT (ra.circuit_id = 'indianapolis' AND r.season <= 1960)
    """).df()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(0).astype(float)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df[df["position"].notna()].copy()
    df = df.drop_duplicates(["race_id", "position"], keep="first")  # shared drives count once
    max_grid = df.groupby("race_id")["grid"].transform("max")
    df["grid_slot"] = np.where(df["grid"] > 0, df["grid"], max_grid + 1)
    df["n_starters"] = df.groupby("race_id")["position"].transform("size")
    df = df[df["n_starters"] >= MIN_STARTERS].copy()
    df["era"] = df["season"].astype(int).map(era_of)
    return df.reset_index(drop=True)


# ------------------------------------------------------------ summaries

def race_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in df.groupby("race_id", sort=True):
        pole = g[g["grid"] == 1]
        winner = g[g["position"] == 1]
        first = g.iloc[0]
        rows.append({
            "race_id": race_id, "season": int(first["season"]), "round": int(first["round"]),
            "circuit_id": first["circuit_id"], "circuit_name": first["circuit_name"],
            "era": first["era"], "n_starters": len(g),
            "spearman": g["grid_slot"].corr(g["position"], method="spearman"),
            "has_pole": len(pole) > 0,
            "pole_won": bool((pole["position"] == 1).any()),
            "winner_grid": float(winner["grid_slot"].iloc[0]) if len(winner) else math.nan,
        })
    return pd.DataFrame(rows)


def by_grid_slot(df: pd.DataFrame, max_slot: int = MAX_SLOT) -> pd.DataFrame:
    d = df.assign(slot=df["grid_slot"].clip(upper=max_slot).astype(int))
    out = d.groupby(["era", "slot"]).agg(
        n=("position", "size"),
        mean_finish=("position", "mean"),
        win_rate=("position", lambda s: (s == 1).mean()),
        podium_rate=("position", lambda s: (s <= 3).mean()),
    ).reset_index()
    out["era"] = pd.Categorical(out["era"], categories=ERA_LABELS, ordered=True)
    return out.sort_values(["era", "slot"]).reset_index(drop=True)


def summarise(races: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Pole win rate (with interval), grid-vs-finish correlation and winner's grid slot."""
    g = races.groupby(by, observed=True)
    out = g.agg(n_races=("race_id", "size"),
                pole_races=("has_pole", "sum"),
                pole_wins=("pole_won", "sum"),
                mean_spearman=("spearman", "mean"),
                mean_winner_grid=("winner_grid", "mean")).reset_index()
    out["pole_win_rate"] = out["pole_wins"] / out["pole_races"].where(out["pole_races"] > 0)
    bounds = [wilson(w, n) for w, n in zip(out["pole_wins"], out["pole_races"])]
    out["pole_ci_low"] = [b[0] for b in bounds]
    out["pole_ci_high"] = [b[1] for b in bounds]
    return out


def era_summary(races: pd.DataFrame) -> pd.DataFrame:
    out = summarise(races, "era")
    out["era"] = pd.Categorical(out["era"], categories=ERA_LABELS, ordered=True)
    return out.sort_values("era").reset_index(drop=True)


def season_summary(races: pd.DataFrame) -> pd.DataFrame:
    out = summarise(races, "season").sort_values("season").reset_index(drop=True)
    out["spearman_rolling5"] = out["mean_spearman"].rolling(5, min_periods=3, center=True).mean()
    return out


def circuit_summary(races: pd.DataFrame, since: int = 2000, min_races: int = 8) -> pd.DataFrame:
    recent = races[races["season"] >= since]
    if recent.empty:
        return pd.DataFrame()
    names = recent.groupby("circuit_id")["circuit_name"].agg(lambda s: s.iloc[-1])
    out = summarise(recent, "circuit_id")
    out["circuit_name"] = out["circuit_id"].map(names)
    out = out[out["n_races"] >= min_races]
    return out.sort_values("mean_spearman", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- charts

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order
BLUE, ORANGE = SERIES[0], SERIES[1]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    return plt


def _frame(plt, title: str, subtitle: str, size=(9, 5.6)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.12)
    fig.text(0.09, 0.955, title, fontsize=13.5, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.09, 0.905, textwrap.fill(subtitle, 105), fontsize=9.5, color=TEXT2, ha="left", va="top",
             linespacing=1.45)
    return fig, ax


def chart_expected_finish(slots: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    eras = [e for e in ERA_LABELS if ((slots["era"] == e) & (slots["n"] >= 20)).any()]
    pole = {e: slots[(slots["era"] == e) & (slots["slot"] == 1)]["mean_finish"] for e in eras}
    if len(eras) >= 2 and all(len(v) for v in pole.values()):
        first, last = eras[0], eras[-1]
        subtitle = (f"Pole sitters finish {pole[last].iloc[0]:.1f} on average in {last}, "
                    f"against {pole[first].iloc[0]:.1f} in {first}. Dashed line: finish equals grid.")
    else:
        subtitle = "Dashed line: finishing position equals grid position."
    fig, ax = _frame(plt, "Where each grid slot ends up", subtitle)
    ax.plot([1, MAX_SLOT], [1, MAX_SLOT], ls="--", lw=1, color=AXIS, zorder=1)
    for i, era in enumerate(ERA_LABELS):
        d = slots[(slots["era"] == era) & (slots["n"] >= 20)].sort_values("slot")
        if d.empty:
            continue
        ax.plot(d["slot"], d["mean_finish"], color=SERIES[i], lw=2, marker="o", ms=4, label=era, zorder=3)
    ax.set_xlabel("Grid slot (20 = 20th or lower)")
    ax.set_ylabel("Mean finishing position")
    ax.set_xlim(0.5, MAX_SLOT + 0.5)
    ax.set_xticks(range(1, MAX_SLOT + 1, 1))
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)


def chart_grip_by_season(seasons: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    s = seasons.dropna(subset=["mean_spearman"])
    fig, ax = _frame(plt, "How tightly qualifying order predicts the race result",
                     "Spearman rank correlation between grid and finishing order, average per season. "
                     "1 = the grid order is the finishing order; thick line = 5-season average.")
    ax.plot(s["season"], s["mean_spearman"], color=AXIS, lw=1.2, marker="o", ms=3, label="Season", zorder=2)
    ax.plot(s["season"], s["spearman_rolling5"], color=BLUE, lw=2.4, label="5-season average", zorder=3)
    low = float(s["mean_spearman"].min())
    ax.set_ylim(max(0.0, math.floor((low - 0.05) * 10) / 10), 1)
    ax.set_xlabel("Season")
    ax.set_ylabel("Grid vs finish correlation")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)


def chart_pole_by_era(eras: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    e = eras.dropna(subset=["pole_win_rate"])
    fig, ax = _frame(plt, "How often the pole sitter wins, by era",
                     "Share of races won from pole, with 95% interval. Labels give the rate and the number of races.")
    x = np.arange(len(e))
    rate = e["pole_win_rate"].to_numpy()
    err = np.vstack([rate - e["pole_ci_low"].to_numpy(), e["pole_ci_high"].to_numpy() - rate])
    ax.bar(x, rate, width=0.6, color=BLUE, zorder=3)
    ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=TEXT2, elinewidth=1.2, capsize=4, zorder=4)
    for xi, r, n, hi in zip(x, rate, e["pole_races"], e["pole_ci_high"]):
        ax.text(xi, hi + 0.015, f"{r:.0%}\nn={int(n)}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(e["era"].astype(str))
    ax.set_ylim(0, min(1.0, float(e["pole_ci_high"].max()) + 0.15))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Races won from pole")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def chart_circuits(circuits: pd.DataFrame, path: Path, since: int = 2000) -> None:
    plt = _plt()
    c = circuits.dropna(subset=["pole_win_rate"]).sort_values("pole_win_rate", ascending=True)
    height = max(3.4, 0.32 * len(c) + 2.1)
    fig, ax = _frame(plt, "How often pole wins, circuit by circuit",
                     f"Races won from pole since {since}, with 95% interval (circuits with at least 8 races). "
                     "Monaco and Monza highlighted.", size=(9, height))
    fig.subplots_adjust(top=1 - 1.5 / height, left=0.27, right=0.95, bottom=0.9 / height)
    y = np.arange(len(c))
    rate = c["pole_win_rate"].to_numpy()
    err = np.vstack([rate - c["pole_ci_low"].to_numpy(), c["pole_ci_high"].to_numpy() - rate])
    colours = [ORANGE if cid in HIGHLIGHT_CIRCUITS else BLUE for cid in c["circuit_id"]]
    ax.barh(y, rate, height=0.62, color=colours, zorder=3)
    ax.errorbar(rate, y, xerr=err, fmt="none", ecolor=TEXT2, elinewidth=1.1, capsize=3, zorder=4)
    for yi, r, n, hi in zip(y, rate, c["pole_races"], c["pole_ci_high"]):
        ax.text(hi + 0.012, yi, f"{r:.0%} (n={int(n)})", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(c["circuit_name"], fontsize=8.5)
    ax.set_xlim(0, min(1.15, float(c["pole_ci_high"].max()) + 0.22))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Races won from pole")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main

def findings(eras: pd.DataFrame, circuits: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in eras.iterrows():
        if r["pole_races"]:
            lines.append(f"{r['era']}: pole won {r['pole_win_rate']:.0%} of {int(r['pole_races'])} races "
                         f"(95% interval {r['pole_ci_low']:.0%} to {r['pole_ci_high']:.0%}); "
                         f"grid-vs-finish correlation {r['mean_spearman']:.2f}")
    for cid in HIGHLIGHT_CIRCUITS:
        row = circuits[circuits["circuit_id"] == cid]
        if len(row):
            r = row.iloc[0]
            lines.append(f"{r['circuit_name']}: pole won {r['pole_win_rate']:.0%} of {int(r['pole_races'])} races "
                         f"({r['pole_ci_low']:.0%} to {r['pole_ci_high']:.0%}); correlation {r['mean_spearman']:.2f}")
    if len(circuits) >= 2:
        top, bottom = circuits.iloc[0], circuits.iloc[-1]
        lines.append(f"Most qualifying-decided circuit: {top['circuit_name']} (correlation {top['mean_spearman']:.2f}); "
                     f"least: {bottom['circuit_name']} ({bottom['mean_spearman']:.2f})")
    return lines


def run(con, out_dir: Path | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "grid_vs_result"
    out.mkdir(parents=True, exist_ok=True)

    df = prepare(load_results(con))
    races = race_summary(df)
    slots = by_grid_slot(df)
    eras = era_summary(races)
    seasons = season_summary(races)
    circuits = circuit_summary(races)

    for name, table in [("race_summary", races), ("grid_slot_by_era", slots), ("era_summary", eras),
                        ("season_summary", seasons), ("circuit_summary", circuits)]:
        table.to_csv(out / f"{name}.csv", index=False)

    chart_expected_finish(slots, out / "expected_finish_by_grid.png")
    chart_grip_by_season(seasons, out / "grid_grip_by_season.png")
    chart_pole_by_era(eras, out / "pole_win_rate_by_era.png")
    if len(circuits):
        chart_circuits(circuits, out / "pole_win_rate_by_circuit.png")

    lines = findings(eras, circuits)
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
    out = config.outputs_dir() / "grid_vs_result"
    print("\n".join(lines))
    print(f"\nTables and charts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
