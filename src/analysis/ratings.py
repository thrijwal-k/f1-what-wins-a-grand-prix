"""Stage 2, part 2: driver skill with the car taken out, using team-mate comparisons.

Idea: two drivers in the same team in the same race drive (nearly) the same car, so
the gap between them is mostly driver. Drivers who change teams link the
comparisons together, so one rating per driver can be estimated across careers.

Two models, same method, different yardsticks:
  rank model  1950 to 1993  gap in qualifying position (grid position where no qualifying
                            data), as a share of the field. Output: grid places on a 20-car grid.
                            Stops in 1993 on purpose: a place on the grid is worth very different
                            amounts of pace in a spread-out field and in today's tightly bunched
                            one, so places cannot be compared across eras. Modern drivers are
                            judged by the time model instead.
  time model  1994 to now   gap in qualifying lap time, ln(slower / faster).
                            Output: percent of a lap.

The method is a Bayesian linear model with a Gaussian prior on every driver's rating.
The posterior is exact (no sampler needed), so each rating comes with a standard
deviation. Prior and noise sizes are learned from the data by EM.
A positive rating means "faster than the average team-mate".

Limits, stated plainly: one rating per driver for a whole career (no improvement or
decline over time); only drivers who had team-mates are rated; links between eras are
long chains of team-mates, so cross-era comparisons carry wide intervals; only teams with exactly
two cars in a race are used (so customer entries are excluded) and the Indianapolis 500 is
excluded before 1961; the 1950s and 60s are still the least reliable part because privateer
entries sometimes carry the same constructor name; team orders,
car reliability and unequal equipment are not modelled.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from .. import config
from ..build_db import parse_time_str

MAX_ABS_TIME_GAP = 0.20   # drop team-mate gaps above 20% of a lap: crashes, red flags, damage
PLACES = 20               # report the rank model as places on a 20-car grid
MIN_OBS_CHART = 40
RANK_LAST_SEASON = 1993  # the rank model stops here; lap times take over from 1994


# ----------------------------------------------------------- observations

def _pairs(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    m = df.merge(df, on=keys, suffixes=("_a", "_b"))
    return m[m["driver_id_a"] < m["driver_id_b"]].copy()


def _two_car_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only teams with exactly two cars in the race.

    Three or more cars from one constructor are mostly customer or privateer entries (not true
    team-mates), and a team of k cars gives k(k-1)/2 pairs that are not independent."""
    n = df.groupby(["race_id", "constructor_id"])["driver_id"].transform("nunique")
    return df[n == 2]


def build_rank_obs(results: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """One row per team-mate pair per race: y = (pos_b - pos_a) / field, positive when a is ahead."""
    d = results[["race_id", "season", "constructor_id", "driver_id", "grid"]].copy()
    q = quali[["race_id", "driver_id", "position"]].rename(columns={"position": "q_pos"})
    d = d.merge(q.drop_duplicates(["race_id", "driver_id"]), on=["race_id", "driver_id"], how="left")
    grid = pd.to_numeric(d["grid"], errors="coerce")
    d["pos"] = pd.to_numeric(d["q_pos"], errors="coerce").where(lambda s: s.notna(), grid.where(grid > 0))
    d = _two_car_teams(d[d["pos"].notna()]).copy()
    d["field"] = d.groupby("race_id")["pos"].transform("max")
    m = _pairs(d, ["race_id", "season", "constructor_id", "field"])
    m = m[m["pos_a"] != m["pos_b"]]  # equal positions = a shared drive, no information
    out = pd.DataFrame({"race_id": m["race_id"], "season": m["season"], "constructor_id": m["constructor_id"],
                        "a": m["driver_id_a"], "b": m["driver_id_b"],
                        "y": (m["pos_b"] - m["pos_a"]) / m["field"]})
    return out.reset_index(drop=True)


def build_time_obs(quali: pd.DataFrame) -> pd.DataFrame:
    """One row per team-mate pair per race: y = ln(time_b / time_a), positive when a is faster.

    Uses the latest qualifying stage both drivers set a time in (Q3, else Q2, else Q1)."""
    q = quali[["race_id", "constructor_id", "driver_id", "q1", "q2", "q3"]].copy()
    for col in ("q1", "q2", "q3"):
        q[col] = q[col].map(parse_time_str).astype(float)
    q = _two_car_teams(q)
    m = _pairs(q, ["race_id", "constructor_id"])
    y = pd.Series(np.nan, index=m.index)
    for stage in ("q3", "q2", "q1"):
        ta, tb = m[f"{stage}_a"], m[f"{stage}_b"]
        ok = y.isna() & (ta > 0) & (tb > 0)
        y[ok] = np.log(tb[ok] / ta[ok])
    m = m.assign(y=y)
    m = m[m["y"].notna() & (m["y"].abs() <= MAX_ABS_TIME_GAP)]
    return pd.DataFrame({"race_id": m["race_id"], "season": (m["race_id"] // 100).astype(int),
                         "constructor_id": m["constructor_id"], "a": m["driver_id_a"],
                         "b": m["driver_id_b"], "y": m["y"]}).reset_index(drop=True)


# ------------------------------------------------------------------ model

def fit_ratings(obs: pd.DataFrame, n_iter: int = 60, tol: float = 1e-7) -> tuple[pd.DataFrame, float, float]:
    """Bayesian linear model y = s_a - s_b + noise with s ~ N(0, tau^2); variances by EM.

    Returns (ratings with mean and sd per driver, noise sd, prior sd)."""
    drivers = sorted(set(obs["a"]) | set(obs["b"]))
    index = {d: i for i, d in enumerate(drivers)}
    n, k = len(obs), len(drivers)
    ia = obs["a"].map(index).to_numpy()
    ib = obs["b"].map(index).to_numpy()
    rows = np.arange(n)
    X = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[rows, rows], np.r_[ia, ib])), shape=(n, k))
    y = obs["y"].to_numpy(dtype=float)
    xtx = (X.T @ X).toarray()
    xty = X.T @ y

    s2 = max(float(y.var()), 1e-12)
    t2 = s2 / 4
    for _ in range(n_iter):
        sigma = np.linalg.inv(xtx / s2 + np.eye(k) / t2)
        mean = sigma @ (xty / s2)
        resid = y - X @ mean
        trace_term = float((sigma[ia, ia] + sigma[ib, ib] - 2 * sigma[ia, ib]).sum())
        s2_new = (float(resid @ resid) + trace_term) / n
        t2_new = (float(mean @ mean) + float(np.trace(sigma))) / k
        done = abs(s2_new - s2) / s2 < tol and abs(t2_new - t2) / t2 < tol
        s2, t2 = s2_new, t2_new
        if done:
            break
    sigma = np.linalg.inv(xtx / s2 + np.eye(k) / t2)
    mean = sigma @ (xty / s2)
    counts = np.bincount(np.r_[ia, ib], minlength=k)
    seasons = pd.concat([obs[["a", "season"]].rename(columns={"a": "driver_id"}),
                         obs[["b", "season"]].rename(columns={"b": "driver_id"})])
    span = seasons.groupby("driver_id")["season"].agg(first_season="min", last_season="max")
    out = pd.DataFrame({"driver_id": drivers, "rating": mean, "sd": np.sqrt(np.diag(sigma)), "n_obs": counts})
    out = out.merge(span, left_on="driver_id", right_index=True, how="left")
    return out, math.sqrt(s2), math.sqrt(t2)


def tidy(ratings: pd.DataFrame, names: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = ratings.merge(names, on="driver_id", how="left")
    out["name"] = out["name"].fillna(out["driver_id"])
    out["rating"] = out["rating"] * scale
    out["sd"] = out["sd"] * scale
    out["ci_low"] = out["rating"] - 1.96 * out["sd"]
    out["ci_high"] = out["rating"] + 1.96 * out["sd"]
    out = out.sort_values("rating", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "name", "driver_id", "rating", "sd", "ci_low", "ci_high", "n_obs",
                "first_season", "last_season"]]


def head_to_head(obs: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    g = obs.assign(a_ahead=obs["y"] > 0).groupby(["season", "constructor_id", "a", "b"]).agg(
        n_races=("y", "size"), a_ahead=("a_ahead", "sum"), mean_gap=("y", "mean")).reset_index()
    g["a_share"] = g["a_ahead"] / g["n_races"]
    lookup = dict(zip(names["driver_id"], names["name"]))
    g["driver_a"] = g["a"].map(lookup).fillna(g["a"])
    g["driver_b"] = g["b"].map(lookup).fillna(g["b"])
    return g[["season", "constructor_id", "driver_a", "driver_b", "n_races", "a_ahead", "a_share", "mean_gap"]]


# ----------------------------------------------------------------- charts

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
BLUE = "#2a78d6"


def overlap_with_leader(top: pd.DataFrame) -> int:
    leader = top.iloc[0]
    return int((top.iloc[1:]["ci_high"] >= leader["ci_low"]).sum())


def chart_ratings(table: pd.DataFrame, path: Path, title: str, unit: str, since: str, top: int = 20,
                  min_obs: int = MIN_OBS_CHART) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
    t = table[table["n_obs"] >= min_obs].head(top).reset_index(drop=True)
    if t.empty:
        return
    t["rank"] = np.arange(1, len(t) + 1)   # rank among drivers with enough comparisons
    n_overlap = overlap_with_leader(t) if len(t) > 1 else 0
    if n_overlap == 0:
        verdict = f"{t.iloc[0]['name']} rates highest and no other interval overlaps theirs."
    else:
        verdict = (f"{t.iloc[0]['name']} rates highest, but {n_overlap} of the other {len(t) - 1} intervals "
                   f"overlap theirs, so the order within the group is not settled.")
    subtitle = (f"{verdict} Dots show the rating, lines the 95% interval, {since}, drivers with at least "
                f"{min_obs} comparisons. Positive = ahead of the average team-mate.")
    height = max(4.5, 0.34 * len(t) + 2.2)
    fig, ax = plt.subplots(figsize=(9, height), dpi=160)
    fig.subplots_adjust(top=1 - 1.7 / height, left=0.24, right=0.96, bottom=0.9 / height)
    fig.text(0.04, 1 - 0.15 / height, title, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.04, 1 - 0.55 / height, textwrap.fill(subtitle, 110), fontsize=9.2, color=TEXT2, va="top",
             linespacing=1.45)
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t["ci_low"], t["ci_high"], color=AXIS, lw=2.2, zorder=2)
    ax.scatter(t["rating"], y, color=BLUE, s=38, zorder=3)
    ax.axvline(0, color=TEXT2, lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}. {n}" for r, n in zip(t["rank"], t["name"])], fontsize=9)
    ax.set_xlabel(unit)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------- main

def load_inputs(con):
    # The Indianapolis 500 counted towards the world championship until 1960: dozens of
    # one-off entries sharing a few chassis names, so "same constructor" does not mean team-mate.
    results = con.execute("""SELECT r.race_id, r.season, r.driver_id, r.constructor_id, r.grid FROM results r
                             JOIN races ra USING (race_id)
                             WHERE NOT (ra.circuit_id = 'indianapolis' AND r.season <= 1960)""").df()
    quali = con.execute("""SELECT q.race_id, q.driver_id, q.constructor_id, q.position, q.q1, q.q2, q.q3
                           FROM qualifying q JOIN races ra USING (race_id)
                           WHERE NOT (ra.circuit_id = 'indianapolis' AND ra.season <= 1960)""").df()
    names = con.execute("""SELECT driver_id, trim(coalesce(given_name, '') || ' ' || coalesce(family_name, '')) AS name
                           FROM drivers""").df()
    return results, quali, names


def run(con, out_dir: Path | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "driver_vs_car"
    out.mkdir(parents=True, exist_ok=True)
    results, quali, names = load_inputs(con)
    lines: list[str] = []

    rank_obs = build_rank_obs(results, quali)
    rank_obs = rank_obs[rank_obs["season"] <= RANK_LAST_SEASON].reset_index(drop=True)
    if len(rank_obs):
        fit, noise, prior = fit_ratings(rank_obs)
        table = tidy(fit, names, PLACES)
        table.to_csv(out / "ratings_rank_model_1950_93.csv", index=False)
        head_to_head(rank_obs, names).to_csv(out / "teammate_head_to_head.csv", index=False)
        chart_ratings(table, out / "ratings_rank_model_1950_93.png",
                      "Best drivers once the car is taken out: grid places, 1950 to 1993",
                      "Grid places per race ahead of the average team-mate (20-car grid)", "1950 to 1993")
        shown = table[table["n_obs"] >= MIN_OBS_CHART].head(10).reset_index(drop=True)
        shown["rank"] = shown.index + 1
        lines.append(f"Rank model: {len(rank_obs)} team-mate comparisons, {len(table)} drivers; "
                     f"typical race-to-race noise {noise * PLACES:.1f} places, spread of true skill {prior * PLACES:.1f} places.")
        lines += [f"  {int(r['rank'])}. {r['name']}: {r['rating']:+.2f} places ({r['ci_low']:+.2f} to {r['ci_high']:+.2f}), "
                  f"{int(r['n_obs'])} comparisons, {int(r['first_season'])}-{int(r['last_season'])}"
                  for _, r in shown.iterrows()]

    time_obs = build_time_obs(quali)
    if len(time_obs) >= 50:
        fit, noise, prior = fit_ratings(time_obs)
        table = tidy(fit, names, 100.0)
        table.to_csv(out / "ratings_time_model_1994_on.csv", index=False)
        chart_ratings(table, out / "ratings_time_model_1994_on.png",
                      "Best drivers once the car is taken out: qualifying pace since 1994",
                      "Percent of a lap faster than the average team-mate", "1994 to now")
        shown = table[table["n_obs"] >= MIN_OBS_CHART].head(10).reset_index(drop=True)
        shown["rank"] = shown.index + 1
        lines.append(f"Time model: {len(time_obs)} team-mate comparisons, {len(table)} drivers; "
                     f"typical gap noise {noise * 100:.2f}% of a lap, spread of true skill {prior * 100:.2f}%.")
        lines += [f"  {int(r['rank'])}. {r['name']}: {r['rating']:+.2f}% ({r['ci_low']:+.2f} to {r['ci_high']:+.2f}), "
                  f"{int(r['n_obs'])} comparisons, {int(r['first_season'])}-{int(r['last_season'])}"
                  for _, r in shown.iterrows()]

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
    print(f"\nTables and charts written to {config.outputs_dir() / 'driver_vs_car'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
