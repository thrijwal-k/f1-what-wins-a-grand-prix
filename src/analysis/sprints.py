"""Stage 2, part 7: sprint races.

Run:  python -m src.analysis.sprints
Writes tables (CSV), charts (PNG) and findings.txt to outputs/sprints/.
Needs the sprint results in the database: python -m src.run_all --skip-fastf1 (downloads them, rebuilds).

Everything before this part looked at the Sunday Grand Prix only. This part asks four things.

  A. What do sprints look like?  Grid, winner, how well the grid predicts the finish, per sprint.
  B. Sprint against Sunday, same weekend.  A sprint is a third of the distance with no required
     stop, so how far the grid order dissolves in it, set beside the Sunday race at the same share of
     the distance (from the lap data) and at the end, separates what is distance from what is
     strategy. Only pairs where the two grids are independent are used.
  C. Does a sprint weekend change the Sunday race?  One practice session instead of three, and
     (from 2023) a Sunday grid that is set before the sprint. Each Sunday measure is compared with
     other Sundays after removing the season level and the circuit level, so Spa on a sprint
     weekend is compared with Spa on other weekends, and 2023 with 2023. The p-value comes from
     shuffling the sprint labels inside each season, the interval from resampling races.
  D. Do sprints add information about drivers and cars?  The car/driver model of part 3 is fitted on
     Sunday races, on sprints alone and on both, and team-mate gaps in sprints are compared with
     team-mate gaps in Sunday races at other weekends, against the same comparison between two sets of
     Sunday races.

Grid definition: in 2021 and 2022 the Sunday grid on a sprint weekend was the sprint finishing
order, so those six races are left out of every Sunday comparison that uses the grid (they stay in
the measures that do not: stops, pit loss, wear, finish rate, neutralisations).
"""
from __future__ import annotations

import math
import textwrap
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import car_driver, grid, grid_checks, race_flow, trends, tyres
from .grid import BLUE, ORANGE, SERIES, TEXT, TEXT2, AXIS, GRID, _frame, _plt, wilson

SPRINT_GRID_LAST_SEASON = 2022       # up to here the Sunday grid on a sprint weekend was the sprint result
FIRST_SEASON = 2018
BOOT_REPS = 1000
PERM_REPS = 2000
SPRINT_OFFSET = 10_000_000           # sprint rows get race_id + this when stacked with Sunday rows
SINCE = 2021
MIN_ROWS_FIT = 150

EFFECT_METRICS = [
    # key, label, uses the Sunday grid?
    ("pole_won", "Pole sitter wins", True),
    ("spearman_all", "Grid vs finishing order", True),
    ("grip_40", "Grid vs running order at 40% of distance", True),
    ("finish_rate", "Cars classified", False),
    ("stops_per_car", "Stops per finishing car (dry)", False),
    ("pit_loss_s", "Time lost per pit stop", False),
    ("wear_medium", "Medium-tyre wear per lap", False),
    ("neutralised", "Safety car, VSC or red flag", False),
]


# ------------------------------------------------------------------ data

def load_sprints(con) -> pd.DataFrame:
    """Sprint results in the shape grid_checks.load gives for Sunday races (no qualifying column)."""
    has = bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'sprint_results'").fetchone()[0])
    if not has:
        return pd.DataFrame()
    df = con.execute("""
        SELECT s.race_id, s.season, s.round, ra.circuit_id,
               coalesce(c.circuit_name, ra.circuit_id) AS circuit_name,
               s.driver_id, s.constructor_id, s.grid, s.position,
               coalesce(s.classified, FALSE) AS classified, NULL::INTEGER AS q_pos
        FROM sprint_results s
        JOIN races ra ON ra.race_id = s.race_id
        LEFT JOIN circuits c ON c.circuit_id = ra.circuit_id
        WHERE s.position IS NOT NULL
    """).df()
    df["is_indy"] = False
    return df


def reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Per race: pole won, grid correlations and the share classified."""
    rel = grid_checks.race_reliability(df)
    share = df.groupby("race_id")["classified"].apply(lambda s: float(s.astype(bool).mean())).rename("classified_share")
    return rel.merge(share, left_on="race_id", right_index=True, how="left")


def sprint_grid_ids(sprint_ids, last_season: int = SPRINT_GRID_LAST_SEASON) -> set[int]:
    """Races whose Sunday grid was the sprint result: every sprint weekend up to `last_season`."""
    return {int(r) for r in sprint_ids if int(r) // 100 <= last_season}


# --------------------------------------------------------------- A. sprints

def sprint_table(df: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    r = rel.set_index("race_id")
    for race_id, g in df.groupby("race_id", sort=True):
        pole = g[g["grid"] == 1]
        win = g[g["position"] == 1]
        first = g.iloc[0]
        rows.append({
            "race_id": race_id, "season": int(first["season"]), "circuit": first["circuit_name"],
            "starters": len(g),
            "grid_defined_by": "qualifying" if int(first["season"]) <= 2022 else "sprint shootout",
            "grid_pole": pole["driver_id"].iloc[0] if len(pole) else None,
            "winner": win["driver_id"].iloc[0] if len(win) else None,
            "winner_grid": float(win["grid_slot"].iloc[0]) if len(win) else math.nan,
            "pole_won": bool(r.at[race_id, "pole_won"]),
            "grid_correlation_all": float(r.at[race_id, "spearman_all"]),
            "grid_correlation_finishers": float(r.at[race_id, "spearman_finishers"]),
            "classified_share": float(r.at[race_id, "classified_share"]),
        })
    return pd.DataFrame(rows)


def sprint_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, d in [("all sprints", table)] + [(f"{s}", g) for s, g in table.groupby("season")]:
        if d.empty:
            continue
        lo, hi = wilson(float(d["pole_won"].sum()), len(d))
        rows.append({"group": label, "sprints": len(d), "pole_win_rate": float(d["pole_won"].mean()),
                     "pole_ci_low": lo, "pole_ci_high": hi,
                     "mean_grid_correlation_all": float(d["grid_correlation_all"].mean()),
                     "mean_grid_correlation_finishers": float(d["grid_correlation_finishers"].mean()),
                     "mean_classified_share": float(d["classified_share"].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------- B. sprint against Sunday

def _boot_diff(d: np.ndarray, reps: int, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(1)
    p = min(1.0, 2 * min(float(np.mean(m <= 0)), float(np.mean(m >= 0))))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), p


def paired_comparison(sprint_rel: pd.DataFrame, sunday_rel: pd.DataFrame, decay: pd.DataFrame | None,
                      sprint_grid: set[int], reps: int = BOOT_REPS, seed: int = 3) -> pd.DataFrame:
    """Sprint minus Sunday, weekend by weekend, with a resampled-weekend interval."""
    ids = sorted((set(sprint_rel["race_id"]) & set(sunday_rel["race_id"])) - set(sprint_grid))
    if len(ids) < 3:
        return pd.DataFrame()
    s = sprint_rel.set_index("race_id").loc[ids]
    u = sunday_rel.set_index("race_id").loc[ids]
    d30 = d40 = None
    if decay is not None and len(decay):
        by = decay.pivot_table(index="race_id", columns="fraction", values="spearman")
        d30 = by[0.3].reindex(ids) if 0.3 in by.columns else None
        d40 = by[0.4].reindex(ids) if 0.4 in by.columns else None
    measures = [
        ("Grid vs finishing order, all starters (Sunday: end of race)", s["spearman_all"], u["spearman_all"]),
        ("Grid vs finishing order, finishers only (Sunday: end of race)", s["spearman_finishers"], u["spearman_finishers"]),
    ]
    if d30 is not None:
        measures.append(("Sprint finishers vs Sunday running order at 30% of the distance", s["spearman_finishers"], d30))
    if d40 is not None:
        measures.append(("Sprint finishers vs Sunday running order at 40% of the distance", s["spearman_finishers"], d40))
    measures += [
        ("Pole sitter wins", s["pole_won"].astype(float), u["pole_won"].astype(float)),
        ("Share of cars classified", s["classified_share"], u["classified_share"]),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for label, a, b in measures:
        ok = a.notna() & b.notna()
        if ok.sum() < 3:
            continue
        a, b = a[ok].to_numpy(float), b[ok].to_numpy(float)
        lo, hi, p = _boot_diff(a - b, reps, rng)
        rows.append({"measure": label, "weekends": int(ok.sum()), "sprint": float(a.mean()), "sunday": float(b.mean()),
                     "difference": float((a - b).mean()), "ci_low": lo, "ci_high": hi, "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = trends.bh_qvalues(out["p_value"].to_numpy())   # across the measures compared
    return out


# ------------------------------------------- C. does a sprint weekend change Sunday?

def stops_by_race(laps: pd.DataFrame, results: pd.DataFrame, flags: pd.DataFrame) -> pd.Series:
    s = tyres._stops_per_driver(laps, results, flags)
    s = s[s["classified"].astype(bool)]
    return s.groupby("race_id")["stops"].mean() if len(s) else pd.Series(dtype=float)


def sunday_metrics(rel: pd.DataFrame, meta: pd.DataFrame, sprint_ids: set[int], sprint_grid: set[int],
                   decay: pd.DataFrame | None = None, flags: pd.DataFrame | None = None,
                   stops: pd.Series | None = None, pit: pd.DataFrame | None = None,
                   wear: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per Sunday race since 2018: whether it was a sprint weekend, and the outcome measures."""
    out = rel[rel["season"] >= FIRST_SEASON][["race_id", "season", "pole_won", "has_pole", "spearman_all",
                                              "classified_share"]].copy()
    out = out.merge(meta[["race_id", "circuit_id"]].drop_duplicates("race_id"), on="race_id", how="left")
    out["sprint"] = out["race_id"].isin(sprint_ids)
    out["sprint_grid"] = out["race_id"].isin(sprint_grid)
    out["pole_won"] = out["pole_won"].astype(float).where(out["has_pole"])
    out = out.rename(columns={"classified_share": "finish_rate"}).drop(columns="has_pole")
    if decay is not None and len(decay):
        g40 = decay[decay["fraction"] == 0.4].drop_duplicates("race_id").set_index("race_id")["spearman"]
        out["grip_40"] = out["race_id"].map(g40)
    else:
        out["grip_40"] = np.nan
    out["neutralised"] = out["race_id"].map(flags.set_index("race_id")["neutral"].astype(float)) if flags is not None else np.nan
    out["stops_per_car"] = out["race_id"].map(stops) if stops is not None else np.nan
    if pit is not None and len(pit):
        out["pit_loss_s"] = out["race_id"].map(pit.groupby("race_id")["loss_s"].median())
    else:
        out["pit_loss_s"] = np.nan
    if wear is not None and len(wear):
        out["wear_medium"] = out["race_id"].map(wear.drop_duplicates("race_id").set_index("race_id")["deg_MEDIUM"])
    else:
        out["wear_medium"] = np.nan
    for key, _, grid_based in EFFECT_METRICS:
        if grid_based:
            out.loc[out["sprint_grid"], key] = np.nan      # a different grid definition: not comparable
    return out.reset_index(drop=True)


def _design(season: np.ndarray, circuit: np.ndarray) -> np.ndarray:
    s = pd.get_dummies(pd.Series(season), dtype=float).to_numpy()
    c = pd.get_dummies(pd.Series(circuit), dtype=float, drop_first=True).to_numpy()
    return np.hstack([s, c]) if c.shape[1] else s


def _residualiser(X: np.ndarray) -> np.ndarray:
    """Matrix that removes whatever the fixed effects explain."""
    return np.eye(len(X)) - X @ np.linalg.pinv(X)


def fe_effect(y: np.ndarray, treat: np.ndarray, X: np.ndarray) -> float:
    M = _residualiser(X)
    t, yy = M @ treat, M @ y
    den = float(t @ t)
    return float((t @ yy) / den) if den > 1e-9 else math.nan


def weekend_effect(metrics: pd.DataFrame, reps: int = BOOT_REPS, perms: int = PERM_REPS, seed: int = 13) -> pd.DataFrame:
    """Sprint-weekend minus other Sundays, within season and circuit, for each measure."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, label, _ in EFFECT_METRICS:
        d = metrics.dropna(subset=[key]).reset_index(drop=True)
        n_sprint = int(d["sprint"].sum())
        if n_sprint < 3 or n_sprint == len(d):
            continue
        y = d[key].to_numpy(float)
        treat = d["sprint"].to_numpy(float)
        season, circuit = d["season"].to_numpy(), d["circuit_id"].to_numpy()
        X = _design(season, circuit)
        est = fe_effect(y, treat, X)
        if math.isnan(est):
            continue
        # p: shuffle the sprint labels inside each season (keeps the number of sprints per season)
        M = _residualiser(X)
        yy = M @ y
        by_season = [np.flatnonzero(season == s) for s in np.unique(season)]
        hits = 0
        for _ in range(perms):
            t = treat.copy()
            for idx in by_season:
                t[idx] = rng.permutation(t[idx])
            tr = M @ t
            den = float(tr @ tr)
            e = (tr @ yy) / den if den > 1e-9 else 0.0
            hits += abs(e) >= abs(est) - 1e-12
        p = (hits + 1) / (perms + 1)
        boots = []
        for _ in range(reps):
            i = rng.integers(0, len(d), size=len(d))
            tb = treat[i]
            if tb.sum() == 0 or tb.sum() == len(tb):
                continue
            Xb = _design(season[i], circuit[i])
            b = fe_effect(y[i], tb, Xb)
            if not math.isnan(b):
                boots.append(b)
        boots = np.array(boots)
        both = set(d.loc[d["sprint"], "circuit_id"]) & set(d.loc[~d["sprint"], "circuit_id"])
        se = float(boots.std(ddof=1)) if len(boots) > 5 else math.nan
        rows.append({"measure": key, "label": label, "races": len(d), "sprint_weekends": n_sprint,
                     "circuits_with_both": len(both),
                     "mean_sprint_weekend": float(y[treat == 1].mean()), "mean_other": float(y[treat == 0].mean()),
                     "raw_difference": float(y[treat == 1].mean() - y[treat == 0].mean()),
                     "effect": est,
                     "ci_low": float(np.percentile(boots, 2.5)) if len(boots) > 5 else math.nan,
                     "ci_high": float(np.percentile(boots, 97.5)) if len(boots) > 5 else math.nan,
                     "smallest_detectable": 2.8 * se if not math.isnan(se) else math.nan,
                     "spread_sd": float(y.std(ddof=1)), "p_value": float(p)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = trends.bh_qvalues(out["p_value"].to_numpy())
        for col in ("effect", "ci_low", "ci_high", "raw_difference"):
            out[col] = out[col].where(out[col].abs() > 1e-9, 0.0)   # float dust from residualising
    return out


# --------------------------------------------- D. drivers and cars with sprints

def sprint_rows(sprints: pd.DataFrame) -> pd.DataFrame:
    """Sprint results in the form the car/driver model takes; sprint race ids are shifted so they cannot clash."""
    s = sprints[["race_id", "season", "constructor_id", "driver_id", "grid", "position"]].copy()
    s["weekend"] = s["race_id"]
    s["race_id"] = s["race_id"] + SPRINT_OFFSET
    empty_q = pd.DataFrame({"race_id": pd.Series(dtype="int64"), "driver_id": pd.Series(dtype="object"),
                            "position": pd.Series(dtype="float64")})
    rows = car_driver.build_rows(s.drop(columns="weekend"), empty_q)
    rows["weekend"] = rows["race_id"] - SPRINT_OFFSET
    return rows


def car_driver_with_sprints(results: pd.DataFrame, quali: pd.DataFrame, sprints: pd.DataFrame, since: int = SINCE,
                            n_boot: int = car_driver.N_BOOT) -> pd.DataFrame:
    """Car, driver and everything-else shares from Sunday races, from sprints, and from both, over the same seasons."""
    res = results[results["season"] >= since]
    sun = car_driver.build_rows(res, quali[quali["race_id"].isin(res["race_id"])])
    sun["weekend"] = sun["race_id"]
    spr = sprint_rows(sprints[sprints["season"] >= since])
    label = f"{since}-now"
    sets = [("Sunday races", sun), ("Sprints", spr), ("Sunday races and sprints", pd.concat([sun, spr], ignore_index=True))]
    out = []
    for name, rows in sets:
        rows = rows.assign(era=label)
        for outcome, col in (("race", "finish_score"), ("qualifying", "start_score")):
            n = int(rows[col].notna().sum())
            if n < MIN_ROWS_FIT:
                continue
            point = car_driver.fit_era(rows, col)
            boots = car_driver.stability_era(rows, col, n_boot)
            rec = {"data": name, "outcome": outcome, "driver_races": n, "drivers": int(rows["driver_id"].nunique()),
                   "car_seasons": int(rows["car"].nunique())}
            for part in car_driver.PARTS:
                rec[f"{part}_share"] = point[part]
                rec[f"{part}_low"] = float(boots[part].quantile(0.025))
                rec[f"{part}_high"] = float(boots[part].quantile(0.975))
            out.append(rec)
    return pd.DataFrame(out)


def teammate_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """Per race and constructor with exactly two scored cars: the first driver's score minus the second's."""
    d = rows[rows["finish_score"].notna()]
    out = []
    for (race_id, team), g in d.groupby(["race_id", "constructor_id"]):
        if len(g) != 2:
            continue
        g = g.sort_values("driver_id")
        out.append({"weekend": int(g["weekend"].iloc[0]), "season": int(g["season"].iloc[0]), "constructor_id": team,
                    "a": g["driver_id"].iloc[0], "b": g["driver_id"].iloc[1],
                    "gap": float(g["finish_score"].iloc[0] - g["finish_score"].iloc[1])})
    return pd.DataFrame(out, columns=["weekend", "season", "constructor_id", "a", "b", "gap"])


def _split_corrs(pairs: list[tuple[np.ndarray, np.ndarray]], rng, boot: bool) -> tuple[float, float]:
    """(sprint-style correlation, Sunday-benchmark correlation) over pair-seasons for one random split."""
    n = len(pairs)
    pick = rng.integers(0, n, size=n) if boot else np.arange(n)
    sp, bm, cm = [], [], []
    for i in pick:
        sprint_gaps, other = pairs[i]
        k = len(sprint_gaps)
        perm = rng.permutation(len(other))
        bench, comp = other[perm[:k]], other[perm[k:]]
        sp.append(sprint_gaps.mean())
        bm.append(bench.mean())
        cm.append(comp.mean())
    sp, bm, cm = np.array(sp), np.array(bm), np.array(cm)
    if sp.std() == 0 or bm.std() == 0 or cm.std() == 0:
        return math.nan, math.nan
    return float(np.corrcoef(sp, cm)[0, 1]), float(np.corrcoef(bm, cm)[0, 1])


def teammate_agreement(sunday_rows: pd.DataFrame, sprint_rows_: pd.DataFrame, reps: int = 300, seed: int = 7,
                       min_sprints: int = 2, min_extra: int = 3) -> pd.DataFrame:
    """Do team-mate gaps in sprints tell the same story as team-mate gaps in Sunday races?

    1. Same weekend: how often does the team-mate who finished ahead in the sprint also finish ahead on Sunday,
       against how often the team-mate ahead in one Sunday race is ahead in the next (two non-sprint Sundays).
    2. Over a season: correlation, across team-mate pairs, between the mean sprint gap and the mean gap in
       Sunday races at other weekends, against the same correlation using a random equal number of those Sundays
       in place of the sprints (comparing with the same remaining Sundays each time)."""
    sun = teammate_gaps(sunday_rows[sunday_rows["season"] >= SINCE])
    spr = teammate_gaps(sprint_rows_)
    if sun.empty or spr.empty:
        return pd.DataFrame()
    key = ["weekend", "season", "constructor_id", "a", "b"]
    rows = []
    both = spr.merge(sun, on=key, suffixes=("_sprint", "_sunday"))
    both = both[(both["gap_sprint"] != 0) & (both["gap_sunday"] != 0)]
    if len(both):
        agree = int((np.sign(both["gap_sprint"]) == np.sign(both["gap_sunday"])).sum())
        lo, hi = wilson(agree, len(both))
        rows.append({"comparison": "Same weekend: team-mate ahead in the sprint is ahead on Sunday",
                     "pairs": len(both), "value": agree / len(both), "ci_low": lo, "ci_high": hi})
    sprint_weekends = set(spr["weekend"])
    other = sun[~sun["weekend"].isin(sprint_weekends)].sort_values("weekend")
    agree_n, total_n = 0, 0
    for _, g in other.groupby(["season", "constructor_id", "a", "b"]):
        gaps = np.sign(g["gap"].to_numpy())
        for x, y in zip(gaps[:-1], gaps[1:]):
            if x != 0 and y != 0:
                total_n += 1
                agree_n += int(x == y)
    if total_n:
        lo, hi = wilson(agree_n, total_n)
        rows.append({"comparison": "Benchmark: team-mate ahead in one Sunday is ahead in the next (non-sprint weekends)",
                     "pairs": total_n, "value": agree_n / total_n, "ci_low": lo, "ci_high": hi})
    pairs = []
    for k, g in spr.groupby(["season", "constructor_id", "a", "b"]):
        o = other[(other["season"] == k[0]) & (other["constructor_id"] == k[1]) & (other["a"] == k[2]) & (other["b"] == k[3])]
        if len(g) >= min_sprints and len(o) >= len(g) + min_extra:
            pairs.append((g["gap"].to_numpy(), o["gap"].to_numpy()))
    if len(pairs) >= 8:
        rng = np.random.default_rng(seed)
        point = np.array([_split_corrs(pairs, rng, boot=False) for _ in range(reps)])
        boot = np.array([_split_corrs(pairs, rng, boot=True) for _ in range(reps)])
        for label, col in (("Season level: mean sprint gap vs mean gap in other Sundays (correlation across pairs)", 0),
                           ("Benchmark: the same with an equal number of other Sundays in place of the sprints", 1)):
            rows.append({"comparison": label, "pairs": len(pairs), "value": float(np.nanmean(point[:, col])),
                         "ci_low": float(np.nanpercentile(boot[:, col], 2.5)),
                         "ci_high": float(np.nanpercentile(boot[:, col], 97.5))})
        diff = boot[:, 0] - boot[:, 1]
        rows.append({"comparison": "Sprint minus benchmark correlation", "pairs": len(pairs),
                     "value": float(np.nanmean(diff)), "ci_low": float(np.nanpercentile(diff[~np.isnan(diff)], 2.5)),
                     "ci_high": float(np.nanpercentile(diff[~np.isnan(diff)], 97.5))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts

def chart_sprint_vs_sunday(table: pd.DataFrame, path: Path) -> None:
    t = table[table["measure"].str.contains("Grid vs|Sprint finishers")].copy()
    if t.empty:
        return
    plt = _plt()
    short = {"Grid vs finishing order, all starters (Sunday: end of race)": "Sunday at the end\n(all starters)",
             "Grid vs finishing order, finishers only (Sunday: end of race)": "Sunday at the end\n(finishers only)",
             "Sprint finishers vs Sunday running order at 30% of the distance": "Sunday at 30%\nof the distance",
             "Sprint finishers vs Sunday running order at 40% of the distance": "Sunday at 40%\nof the distance"}
    n = int(t["weekends"].max())
    sprint_all = t[t["measure"].str.contains("all starters")]
    sprint_fin = t[t["measure"].str.contains("finishers only")]
    sub = (f"{n} sprint weekends with independent grids, same weekends in each pair. Beside each pair: "
           f"sprint minus Sunday, with its 95% range.")
    fig, ax = _frame(plt, "Does a sprint keep the grid order better than a Grand Prix?", sub, size=(9.5, 5.6))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.scatter([i], [r["sprint"]], s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=1.2)
        ax.scatter([i], [r["sunday"]], s=70, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.2)
        ax.plot([i, i], [r["sprint"], r["sunday"]], color=AXIS, lw=1.2, zorder=2)
        ax.text(i + 0.08, max(r["sprint"], r["sunday"]) - 0.012 if abs(r["sprint"] - r["sunday"]) < 0.04 else (r["sprint"] + r["sunday"]) / 2,
                f"{r['difference']:+.2f}\n({r['ci_low']:+.2f} to {r['ci_high']:+.2f})", fontsize=8.5, color=TEXT2, va="center")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([short.get(m, m) for m in t["measure"]], fontsize=9)
    ax.set_xlim(-0.5, len(t) - 0.2)
    ax.set_ylabel("Grid vs order correlation")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=BLUE, label="Sprint"),
                       Line2D([0], [0], marker="o", ls="", color=ORANGE, label="Sunday race")],
              frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    lo = float(min(t["sprint"].min(), t["sunday"].min()))
    hi = float(max(t["sprint"].max(), t["sunday"].max()))
    ax.set_ylim(lo - 0.04, hi + 0.04)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path)
    plt.close(fig)


def chart_weekend_effect(effect: pd.DataFrame, path: Path) -> None:
    if effect.empty:
        return
    plt = _plt()
    e = effect.copy()
    for col in ("effect", "ci_low", "ci_high", "smallest_detectable"):
        e[f"{col}_sd"] = e[col] / e["spread_sd"]
    sub = ("Sprint-weekend Sundays against other Sundays, same circuit and season. Line = 95% range; shaded = too small "
           "to detect. Brackets: sprint weekends used.")
    fig, ax = _frame(plt, "Does a sprint weekend change the Sunday race?", sub, size=(9.5, 5.6))
    fig.subplots_adjust(top=0.80, left=0.40)
    y = np.arange(len(e))[::-1]
    det = float(np.nanmedian(e["smallest_detectable_sd"]))
    if not math.isnan(det):
        ax.axvspan(-det, det, color=GRID, alpha=0.7, zorder=0)
    ax.axvline(0, color=AXIS, lw=1.0)
    for yi, (_, r) in zip(y, e.iterrows()):
        ax.plot([r["ci_low_sd"], r["ci_high_sd"]], [yi, yi], color=AXIS, lw=2.2, zorder=2, solid_capstyle="round")
        ax.scatter([r["effect_sd"]], [yi], s=60, color=ORANGE if r["p_value"] <= 0.05 else BLUE, zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{textwrap.fill(l, 34)} ({int(n)})" for l, n in zip(e["label"], e["sprint_weekends"])], fontsize=8.5)
    ax.set_xlabel("Effect of a sprint weekend, in units of the usual spread")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def chart_shares(table: pd.DataFrame, path: Path) -> None:
    t = table[table["outcome"] == "race"]
    if t.empty:
        return
    plt = _plt()
    colours = {"car": SERIES[0], "driver": SERIES[1], "other": AXIS}
    names = {"car": "Car", "driver": "Driver", "other": "Everything else"}
    sets = list(t["data"])
    sub = ("Share of variation in finishing position (part 3 model), same seasons. "
           "Sprints add few results, so ranges are wide (see CSV).")
    fig, ax = _frame(plt, "Do sprints change who gets the credit, car or driver?", sub, size=(9.5, 3.9))
    fig.subplots_adjust(top=0.70, left=0.27, bottom=0.2)
    y = np.arange(len(sets))[::-1]
    left = np.zeros(len(sets))
    for part in ("car", "driver", "other"):
        w = t[f"{part}_share"].to_numpy()
        ax.barh(y, w, left=left, color=colours[part], height=0.6, edgecolor="white", linewidth=2, label=names[part])
        for yi, l, wi in zip(y, left, w):
            if wi >= 0.1:
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if part != "other" else TEXT)
        left = left + w
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({int(n)})" for s, n in zip(sets, t["driver_races"])], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------- findings

def _q(x) -> str:
    return "not significant" if (x is None or math.isnan(x) or x >= 0.05) else "significant"


def findings(sprints: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, effect: pd.DataFrame,
             shares: pd.DataFrame, mates: pd.DataFrame) -> list[str]:
    L: list[str] = []
    if sprints.empty:
        return ["NO SPRINT RESULTS in the database. Run:  python -m src.run_all --skip-fastf1  (downloads them and rebuilds)."]
    allrow = summary[summary["group"] == "all sprints"].iloc[0]
    L.append(f"SPRINTS: {int(allrow['sprints'])} races, {int(sprints['season'].min())} to {int(sprints['season'].max())}")
    L.append(f"  pole sitter (the sprint's own grid) won {allrow['pole_win_rate']:.0%} "
             f"({allrow['pole_ci_low']:.0%} to {allrow['pole_ci_high']:.0%}); grid correlation "
             f"{allrow['mean_grid_correlation_all']:.2f} all starters, {allrow['mean_grid_correlation_finishers']:.2f} finishers only; "
             f"{allrow['mean_classified_share']:.0%} of cars classified")
    for _, r in summary[summary["group"] != "all sprints"].iterrows():
        L.append(f"  {r['group']}: {int(r['sprints'])} sprints, pole won {r['pole_win_rate']:.0%}, grid correlation "
                 f"{r['mean_grid_correlation_all']:.2f}")
    if len(paired):
        L.append(f"SPRINT AGAINST SUNDAY, SAME WEEKEND ({int(paired['weekends'].max())} weekends with independent grids; "
                 "difference = sprint minus Sunday, 95% interval)")
        for _, r in paired.iterrows():
            L.append(f"  {r['measure']}: sprint {r['sprint']:.2f}, Sunday {r['sunday']:.2f}; difference {r['difference']:+.2f} "
                     f"({r['ci_low']:+.2f} to {r['ci_high']:+.2f}), p = {r['p_value']:.2f}, q = {r['q_value']:.2f}, {_q(r['q_value'])}")
        L.append("  Significance here uses q, which allows for the number of measures compared; the measures overlap, so q is cautious.")
        L.append("  Reading: a sprint is about a third of the distance with no required stop. If it matches the Sunday race at "
                 "30-40% of the distance, distance alone explains how far the order dissolves; if it keeps more order than that, "
                 "stops and tyre strategy are doing some of the scrambling in a Grand Prix.")
    if len(effect):
        L.append(f"DOES A SPRINT WEEKEND CHANGE THE SUNDAY RACE? ({int(effect['sprint_weekends'].max())} sprint weekends; compared "
                 "with other Sundays at the same circuit and season; p from shuffling labels inside seasons)")
        for _, r in effect.iterrows():
            L.append(f"  {r['label']}: sprint weekends {r['mean_sprint_weekend']:.3g} against {r['mean_other']:.3g}; adjusted effect "
                     f"{r['effect']:+.3g} ({r['ci_low']:+.3g} to {r['ci_high']:+.3g}), p = {r['p_value']:.2f}, q = {r['q_value']:.2f}; "
                     f"smallest effect these races could detect about {r['smallest_detectable']:.3g}")
        small = effect[effect["p_value"] < 0.05]
        L.append(f"  {len(small)} of {len(effect)} measures have p < 0.05 (about {0.05 * len(effect):.1f} expected by luck); "
                 "'not significant' here means 'no evidence of a difference at this sample size', not 'no difference'.")
    if len(shares):
        L.append(f"CAR, DRIVER AND EVERYTHING ELSE, {car_driver_label(shares)}")
        for _, r in shares.iterrows():
            L.append(f"  {r['data']}, {r['outcome']}: car {r['car_share']:.0%} ({r['car_low']:.0%}-{r['car_high']:.0%}), "
                     f"driver {r['driver_share']:.0%} ({r['driver_low']:.0%}-{r['driver_high']:.0%}), "
                     f"everything else {r['other_share']:.0%}; {int(r['driver_races'])} driver-races")
    if len(mates):
        L.append("TEAM-MATE GAPS: do sprints tell the same story about drivers as Sunday races?")
        for _, r in mates.iterrows():
            L.append(f"  {r['comparison']}: {r['value']:.2f} ({r['ci_low']:.2f} to {r['ci_high']:.2f}), {int(r['pairs'])} cases")
    L.append("READ WITH CARE: there are few sprint weekends, so intervals are wide. These comparisons are observational: "
             "which circuits get a sprint is a choice, which is why the weekend comparison is made within circuit and season.")
    return L


def car_driver_label(shares: pd.DataFrame) -> str:
    return "same seasons for every row (2021 on)"


# ----------------------------------------------------------------------- run

def run(con, out_dir: Path | None = None, reps: int = BOOT_REPS, perms: int = PERM_REPS,
        n_boot: int = car_driver.N_BOOT) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sprints"
    out.mkdir(parents=True, exist_ok=True)

    raw_sprints = load_sprints(con)
    if raw_sprints.empty:
        lines = findings(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    sprint_df = grid.prepare(raw_sprints)
    sprint_rel = reliability(sprint_df)
    table = sprint_table(sprint_df, sprint_rel)
    summary = sprint_summary(table)
    sprint_ids = set(int(x) for x in raw_sprints["race_id"].unique())
    sprint_grid = sprint_grid_ids(sprint_ids)

    sunday_raw = grid_checks.load(con)
    sunday_df = grid_checks.prepare_checks(sunday_raw, drop_indy=True)
    sunday_rel = reliability(sunday_df)
    meta = sunday_raw.drop_duplicates("race_id")[["race_id", "circuit_id"]]

    laps = pd.DataFrame()
    decay = flags = stops = pit = wear = None
    if tyres._has(con, "laps"):
        laps = tyres.load_laps(con)
    if len(laps):
        laps = tyres.normalise_compounds(laps[laps["season"] >= tyres.FIRST_SEASON])
        flags = tyres.race_flags(laps, tyres.load_rain_share(con))
        res_recent = sunday_raw[sunday_raw["season"] >= tyres.FIRST_SEASON]
        decay = race_flow.grip_decay(laps, res_recent, flags)
        stops = stops_by_race(laps, res_recent, flags)
        pit = tyres.pit_loss(laps)
        wear = tyres.tyre_by_race(laps, flags.set_index("race_id")["wet"])

    paired = paired_comparison(sprint_rel, sunday_rel, decay, sprint_grid, reps)
    metrics = sunday_metrics(sunday_rel, meta, sprint_ids, sprint_grid, decay, flags, stops, pit, wear)
    effect = weekend_effect(metrics, reps, perms)

    results = sunday_raw[~sunday_raw["is_indy"]][["race_id", "season", "driver_id", "constructor_id", "grid", "position"]]
    quali = con.execute("SELECT race_id, driver_id, position FROM qualifying").df() \
        if tyres._has(con, "qualifying") else pd.DataFrame(columns=["race_id", "driver_id", "position"])
    shares = car_driver_with_sprints(results, quali, raw_sprints, n_boot=n_boot)
    sun_rows = car_driver.build_rows(results[results["season"] >= SINCE], quali)
    sun_rows["weekend"] = sun_rows["race_id"]
    mates = teammate_agreement(sun_rows, sprint_rows(raw_sprints[raw_sprints["season"] >= SINCE]))

    for name, t in [("sprint_races", table), ("sprint_summary", summary), ("sprint_vs_sunday", paired),
                    ("sunday_race_measures", metrics), ("sprint_weekend_effect", effect),
                    ("car_driver_with_sprints", shares), ("teammate_agreement", mates)]:
        t.to_csv(out / f"{name}.csv", index=False)
    if len(paired):
        chart_sprint_vs_sunday(paired, out / "sprint_vs_sunday.png")
    if len(effect):
        chart_weekend_effect(effect, out / "sprint_weekend_effect.png")
    if len(shares):
        chart_shares(shares, out / "car_driver_with_sprints.png")

    lines = findings(table, summary, paired, effect, shares, mates)
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'sprints'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
