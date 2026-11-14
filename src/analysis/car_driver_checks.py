"""Stage 2, part 3b: how far can the car-vs-driver split be trusted?

Four checks on the model in car_driver.py:

  1 mobility   how many drivers change constructor, how connected the data is. A driver is only
               separated from a car by moving or by having different team-mates, so this is the
               raw material of the driver share.
  2 recovery   take the real drivers, constructors and seasons of each era, invent outcomes with
               KNOWN car, driver and noise shares, refit the model, and compare. This shows, era
               by era, how much of the answer the design alone can support, and in which direction
               the model errs when it cannot.
  3 rolling    the same shares over 10-season windows instead of five fixed eras, so a result that
               depends on where an era boundary falls shows up as a jump.
  4 lap times  qualifying measured as the gap in lap time (first qualifying session) instead of as
               a rank. If the split depends on the yardstick, the two will disagree.

  5 placebo    shuffle which driver drove in which race inside each constructor-season. That keeps the
               cars and destroys any real driver effect, so the driver share should fall to about zero.
               If it does not, the pipeline is manufacturing driver share out of something else.
  6 two cars   keep only teams that ran exactly two cars in a race. Customer and privateer cars share a
               constructor name with the works team but are not the same car; the model would read their
               persistent slowness as driver skill. If the driver share drops a lot here, that was happening.

Run:  python -m src.analysis.car_driver_checks
Writes to outputs/car_vs_driver_checks/. Takes several minutes.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .. import config
from ..build_db import parse_time_str
from . import car_driver as cd
from .grid import ERA_LABELS

SCENARIOS = {
    "driver 8%": {"team": 0.30, "car": 0.30, "driver": 0.08},    # noise = the rest (32%)
    "driver 20%": {"team": 0.25, "car": 0.25, "driver": 0.20},   # noise = the rest (30%)
}
MAX_ABS_GAP = 0.20      # drop qualifying lap gaps above 20%
WINDOW, STEP = 10, 3


# ------------------------------------------------------------------ 1 mobility

def mobility(era_rows: pd.DataFrame) -> dict:
    d = era_rows.dropna(subset=["finish_score"])
    per_driver = d.groupby("driver_id").agg(races=("race_id", "nunique"), teams=("constructor_id", "nunique"),
                                            seasons=("season", "nunique"))
    codes_d, nd = cd._codes(d["driver_id"])
    codes_t, nt = cd._codes(d["constructor_id"])
    pairs = pd.DataFrame({"d": codes_d, "t": codes_t}).drop_duplicates()
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs["d"], nd + pairs["t"])), shape=(nd + nt, nd + nt))
    n_comp, labels = connected_components(graph, directed=False)
    biggest = pd.Series(labels).value_counts().iloc[0]
    return {"drivers": nd, "constructors": nt,
            "share_drivers_with_2plus_teams": float((per_driver["teams"] >= 2).mean()),
            "share_drivers_one_off_under_6_races": float((per_driver["races"] < 6).mean()),
            "mean_seasons_per_driver": float(per_driver["seasons"].mean()),
            "mean_races_per_driver": float(per_driver["races"].mean()),
            "connected_pieces": int(n_comp), "largest_piece_share_of_nodes": float(biggest / (nd + nt))}


def mobility_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label in ERA_LABELS:
        era_rows = rows[rows["era"] == label]
        if len(era_rows) >= cd.MIN_ROWS:
            out.append({"era": label, **mobility(era_rows)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 2 recovery

def simulate_outcome(era_rows: pd.DataFrame, truth: dict, rng: np.random.Generator) -> pd.Series:
    """Invent outcomes on the real design with known shares (total variance 1)."""
    noise_share = 1 - sum(truth.values())
    team = pd.Series(rng.normal(0, np.sqrt(truth["team"]), era_rows["constructor_id"].nunique()),
                     index=era_rows["constructor_id"].unique())
    car = pd.Series(rng.normal(0, np.sqrt(truth["car"]), era_rows["car"].nunique()), index=era_rows["car"].unique())
    drv = pd.Series(rng.normal(0, np.sqrt(truth["driver"]), era_rows["driver_id"].nunique()),
                    index=era_rows["driver_id"].unique())
    y = (era_rows["constructor_id"].map(team).to_numpy() + era_rows["car"].map(car).to_numpy()
         + era_rows["driver_id"].map(drv).to_numpy() + rng.normal(0, np.sqrt(noise_share), len(era_rows)))
    return pd.Series(y, index=era_rows.index)


def recovery_table(rows: pd.DataFrame, n_rep: int = 4, seed: int = 0, scenarios: dict | None = None) -> pd.DataFrame:
    scenarios = scenarios or SCENARIOS
    rng = np.random.default_rng(seed)
    out = []
    for label in ERA_LABELS:
        era_rows = rows[(rows["era"] == label) & rows["finish_score"].notna()].copy()
        if len(era_rows) < cd.MIN_ROWS:
            continue
        for name, truth in scenarios.items():
            ests = []
            for _ in range(n_rep):
                era_rows["sim"] = simulate_outcome(era_rows, truth, rng)
                s = cd.fit_era(era_rows, "sim")
                ests.append([s["car"], s["driver"], s["other"]])
            est = np.array(ests)
            true_car = truth["team"] + truth["car"]
            out.append({"era": label, "scenario": name,
                        "true_car": true_car, "true_driver": truth["driver"], "true_other": 1 - true_car - truth["driver"],
                        "est_car": est[:, 0].mean(), "est_driver": est[:, 1].mean(), "est_other": est[:, 2].mean(),
                        "driver_sd_over_reps": est[:, 1].std(), "driver_bias": est[:, 1].mean() - truth["driver"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 5 placebo, 6 two cars

def placebo_table(rows: pd.DataFrame, n_rep: int = 2, seed: int = 0) -> pd.DataFrame:
    """Shuffle driver labels among the rows of each constructor-season; the driver share should vanish."""
    rng = np.random.default_rng(seed)
    out = []
    for label in ERA_LABELS:
        era_rows = rows[rows["era"] == label]
        if len(era_rows) < cd.MIN_ROWS:
            continue
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            real = cd.fit_era(era_rows, column)
            shuffled = []
            for _ in range(n_rep):
                fake = era_rows.copy()
                fake["driver_id"] = fake.groupby("car")["driver_id"].transform(lambda x: rng.permutation(x.to_numpy()))
                shuffled.append(cd.fit_era(fake, column)["driver"])
            out.append({"era": label, "outcome": outcome, "driver_share_real": real["driver"],
                        "driver_share_shuffled": float(np.mean(shuffled))})
    return pd.DataFrame(out)


def two_car_table(rows: pd.DataFrame) -> pd.DataFrame:
    n = rows.groupby(["race_id", "constructor_id"])["driver_id"].transform("nunique")
    two = rows[n == 2]
    out = []
    for label in ERA_LABELS:
        all_rows, sub = rows[rows["era"] == label], two[two["era"] == label]
        if len(sub) < cd.MIN_ROWS:
            continue
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            a, b = cd.fit_era(all_rows, column), cd.fit_era(sub, column)
            out.append({"era": label, "outcome": outcome, "share_of_rows_kept": len(sub) / len(all_rows),
                        "driver_all": a["driver"], "driver_two_car_teams": b["driver"],
                        "car_all": a["car"], "car_two_car_teams": b["car"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 3 rolling

def rolling_table(rows: pd.DataFrame, window: int = WINDOW, step: int = STEP) -> pd.DataFrame:
    """Shares in windows of `window` seasons, moved on by `step`; the last window always ends at the latest season."""
    first, last = int(rows["season"].min()), int(rows["season"].max())
    starts = list(range(first, last - window + 2, step))
    if not starts or starts[-1] + window - 1 < last:
        starts.append(max(first, last - window + 1))
    out = []
    for start in starts:
        end = min(start + window - 1, last)
        win = rows[(rows["season"] >= start) & (rows["season"] <= end)]
        if len(win) < cd.MIN_ROWS:
            continue
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            s = cd.fit_era(win, column)
            out.append({"first_season": start, "last_season": end, "mid": (start + end) / 2, "outcome": outcome,
                        "car_share": s["car"], "driver_share": s["driver"], "other_share": s["other"],
                        "car_persistent_share": s["car_persistent"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 4 lap times

def add_lap_gap(rows: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """Qualifying as the lap-time gap in the first session: higher = faster. Needs a time for the driver."""
    q = quali[["race_id", "driver_id", "q1"]].drop_duplicates(["race_id", "driver_id"]).copy()
    q["t"] = q["q1"].map(parse_time_str).astype(float)
    q = q[q["t"] > 0]
    q["lt"] = np.log(q["t"])
    q["gap_score"] = -(q["lt"] - q.groupby("race_id")["lt"].transform("median"))
    q.loc[q["gap_score"].abs() > MAX_ABS_GAP, "gap_score"] = np.nan
    sd = q["gap_score"].std()
    q["gap_score"] = q["gap_score"] / sd          # unit variance, so shares are comparable
    return rows.merge(q[["race_id", "driver_id", "gap_score"]], on=["race_id", "driver_id"], how="left")


def compare_measures(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    timed = rows[rows["gap_score"].notna()]
    for label in ERA_LABELS:
        era_rows = timed[timed["era"] == label]
        if len(era_rows) < cd.MIN_ROWS:
            continue
        by_rank = cd.fit_era(era_rows, "start_score")
        by_time = cd.fit_era(era_rows, "gap_score")
        out.append({"era": label, "driver_races_with_times": len(era_rows),
                    "car_by_rank": by_rank["car"], "car_by_time": by_time["car"],
                    "driver_by_rank": by_rank["driver"], "driver_by_time": by_time["driver"],
                    "other_by_rank": by_rank["other"], "other_by_time": by_time["other"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ chart

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def chart_rolling(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=160, sharex=True)
    fig.subplots_adjust(top=0.74, left=0.07, right=0.97, bottom=0.12, wspace=0.18)
    fig.text(0.07, 0.96, "Does the answer depend on the era boundaries? Shares over 10-season windows",
             fontsize=13.5, fontweight="bold", va="top")
    sub = ("Each point is one window of ten seasons, placed at its midpoint. Where a line jumps or wanders, "
           "the split is not stable; compare with the recovery table before trusting it.")
    fig.text(0.07, 0.905, textwrap.fill(sub, 120), fontsize=9.2, color=TEXT2, va="top", linespacing=1.45)
    for ax, key, title in zip(axes, ("car_share", "driver_share"), ("Car share", "Driver share")):
        for outcome, colour in (("qualifying", BLUE), ("race", ORANGE)):
            t = table[table["outcome"] == outcome].sort_values("mid")
            ax.plot(t["mid"], t[key], color=colour, lw=2, marker="o", ms=5, label=outcome.capitalize())
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
        ax.set_ylim(0, max(0.8, float(table[key].max()) + 0.05))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("Midpoint of window")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main

def findings(mob: pd.DataFrame, rec: pd.DataFrame, cmp_: pd.DataFrame, roll: pd.DataFrame,
             placebo: pd.DataFrame | None = None, two: pd.DataFrame | None = None) -> list[str]:
    lines = ["MOBILITY (who can be told apart from their car)"]
    for _, r in mob.iterrows():
        lines.append(f"  {r['era']}: {r['drivers']} drivers, {r['constructors']} constructors; "
                     f"{r['share_drivers_with_2plus_teams']:.0%} of drivers raced for 2+ teams; "
                     f"{r['share_drivers_one_off_under_6_races']:.0%} had under 6 races; "
                     f"{r['mean_seasons_per_driver']:.1f} seasons per driver; "
                     f"largest connected piece holds {r['largest_piece_share_of_nodes']:.0%} of drivers and teams")
    lines.append("RECOVERY ON THE REAL DESIGN (invented outcomes with known shares; est = what the model returns)")
    for _, r in rec.iterrows():
        lines.append(f"  {r['era']}, truth driver {r['true_driver']:.0%}: est driver {r['est_driver']:.0%} "
                     f"(bias {r['driver_bias']:+.0%}, spread {r['driver_sd_over_reps']:.0%}); "
                     f"car truth {r['true_car']:.0%} est {r['est_car']:.0%}")
    lines.append("QUALIFYING BY RANK vs BY LAP-TIME GAP (same drivers and races)")
    for _, r in cmp_.iterrows():
        lines.append(f"  {r['era']}: car {r['car_by_rank']:.0%} vs {r['car_by_time']:.0%}; "
                     f"driver {r['driver_by_rank']:.0%} vs {r['driver_by_time']:.0%}; "
                     f"other {r['other_by_rank']:.0%} vs {r['other_by_time']:.0%}  ({r['driver_races_with_times']} driver-races)")
    lines.append("ROLLING 10-SEASON WINDOWS: driver share (qualifying / race)")
    wide = roll.pivot(index="first_season", columns="outcome", values="driver_share")
    for first, r in wide.iterrows():
        last = int(roll[roll["first_season"] == first]["last_season"].iloc[0])
        lines.append(f"  {first}-{last}: {r['qualifying']:.0%} / {r['race']:.0%}")
    if placebo is not None and len(placebo):
        lines.append("PLACEBO (driver labels shuffled inside each constructor-season; driver share should fall to ~0)")
        for _, r in placebo.iterrows():
            lines.append(f"  {r['era']} {r['outcome']}: real {r['driver_share_real']:.0%}, shuffled {r['driver_share_shuffled']:.0%}")
    if two is not None and len(two):
        lines.append("ALL CARS vs ONLY TEAMS WITH EXACTLY TWO CARS (customer and privateer cars removed)")
        for _, r in two.iterrows():
            lines.append(f"  {r['era']} {r['outcome']}: driver {r['driver_all']:.0%} -> {r['driver_two_car_teams']:.0%}; "
                         f"car {r['car_all']:.0%} -> {r['car_two_car_teams']:.0%}; rows kept {r['share_of_rows_kept']:.0%}")
    return lines


def run(con, out_dir: Path | None = None, n_rep: int = 4) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "car_vs_driver_checks"
    out.mkdir(parents=True, exist_ok=True)
    results, quali = cd.load_inputs(con)
    qtimes = con.execute("""SELECT q.race_id, q.driver_id, q.q1 FROM qualifying q JOIN races ra USING (race_id)
                            WHERE NOT (ra.circuit_id = 'indianapolis' AND ra.season <= 1960)""").df()
    rows = add_lap_gap(cd.build_rows(results, quali), qtimes)
    mob = mobility_table(rows)
    rec = recovery_table(rows, n_rep=n_rep)
    cmp_ = compare_measures(rows)
    roll = rolling_table(rows)
    placebo = placebo_table(rows)
    two = two_car_table(rows)
    placebo.to_csv(out / "placebo_by_era.csv", index=False)
    two.to_csv(out / "two_car_teams_by_era.csv", index=False)
    mob.to_csv(out / "mobility_by_era.csv", index=False)
    rec.to_csv(out / "recovery_by_era.csv", index=False)
    cmp_.to_csv(out / "rank_vs_laptime_by_era.csv", index=False)
    roll.to_csv(out / "rolling_shares.csv", index=False)
    if len(roll):
        chart_rolling(roll, out / "rolling_shares.png")
    lines = findings(mob, rec, cmp_, roll, placebo, two)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> None:
    db = config.db_path()
    if not Path(db).exists():
        sys.exit(f"No database at {db}. Run stage 1 first: python -m src.run_all")
    con = duckdb.connect(str(db), read_only=True)
    print("\n".join(run(con)))


if __name__ == "__main__":
    main()
