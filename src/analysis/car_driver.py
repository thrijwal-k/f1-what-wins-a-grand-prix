"""Stage 2, part 3: how much of a result is the car, how much the driver, how much everything else?

For each era the model splits the variation in one outcome into three parts:
    car     the constructor's effect: what it keeps from season to season plus its season-specific
            shape (both are shared by every driver of that constructor in that season)
    driver  a fixed effect for each driver across the era
    other   what is left: luck, reliability on the day, strategy, incidents, form

Two outcomes are split with the same model, so they can be compared:
    qualifying   starting position (qualifying position, grid where there is none)
    race         official finishing position

Both are turned into a score that does not depend on field size: the rankit, the normal
score of (field - position + 0.5) / field, within each race. Higher = better.

Method: a Gaussian mixed model y = car + driver + noise, with the three variances learned
by EM and the effects solved exactly (no sampler). A share is a variance divided by the
sum of the three. The low/high columns are a stability range: the 2.5th to 97.5th percentile of
the same shares refitted on 40 random 70% subsets of the era's seasons. It shows how much the
answer depends on which seasons are included; it is not a formal confidence interval.

Run:  python -m src.analysis.car_driver
Writes to outputs/car_vs_driver/.

Why the persistent part matters: a constructor that stays strong for years looks, to a model that
sees only independent seasons, like a group of drivers who happen to be good. Giving the model a
constructor effect that lasts the whole era removes that bias (tested on simulated data).

Limits, stated plainly. Drivers are only separated from cars because they change teams or
have team-mates, so the driver share is a lower bound when drivers rarely move. One effect
per driver per era means improvement and decline inside an era are counted as noise.
Customer cars share a constructor name with the works team. The Indianapolis 500 is
excluded before 1961.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from .. import config
from .grid import ERAS, ERA_LABELS, era_of

N_BOOT = 40               # number of random subsets of seasons per era
MIN_ROWS = 400            # skip an era with fewer driver-races than this
PARTS = ["car", "driver", "other"]


# ------------------------------------------------------------------- data

def rankit(pos: pd.Series, race: pd.Series) -> pd.Series:
    field = pos.groupby(race).transform("count")
    return pd.Series(norm.ppf((field - pos + 0.5) / field), index=pos.index)


def build_rows(results: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per race with a qualifying score and a race score."""
    d = results[["race_id", "season", "constructor_id", "driver_id", "grid", "position"]].copy()
    q = quali[["race_id", "driver_id", "position"]].rename(columns={"position": "q_pos"})
    d = d.merge(q.drop_duplicates(["race_id", "driver_id"]), on=["race_id", "driver_id"], how="left")
    grid = pd.to_numeric(d["grid"], errors="coerce")
    d["start"] = pd.to_numeric(d["q_pos"], errors="coerce").where(lambda s: s.notna(), grid.where(grid > 0))
    d["finish"] = pd.to_numeric(d["position"], errors="coerce")
    d = d.drop_duplicates(["race_id", "driver_id"])
    d["start_score"] = rankit(d["start"], d["race_id"])
    d["finish_score"] = rankit(d["finish"], d["race_id"])
    d["car"] = d["constructor_id"] + "|" + d["season"].astype(str)
    d["era"] = d["season"].map(era_of)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ model

def fit_components(y: np.ndarray, driver: np.ndarray, car: np.ndarray, n_driver: int, n_car: int,
                   team: np.ndarray | None = None, n_team: int = 0, n_iter: int = 100, tol: float = 1e-7,
                   extra: list | None = None, return_effects: bool = False, init: dict | None = None) -> dict:
    """y = driver + car-season [+ team for the whole era] [+ extra effects] + noise.

    The optional team effect is the strength a constructor keeps from season to season. Without it,
    a constructor that stays strong for years is credited to the drivers who stay with it (checked
    on simulated data: the driver share came out at 57% when the truth was 8%).
    `extra` is a list of further random effects, each (name, codes, size) or (name, codes, size, weights);
    with weights the effect enters as weight x effect (a random slope, e.g. a car getting better or worse
    through a season), and its returned variance is per unit of weight.
    Returns the variances: car (season-specific), team (persistent), driver, other, and each extra."""
    n = len(y)
    groups = [("driver", driver, n_driver, None), ("car", car, n_car, None)]
    if team is not None and n_team:
        groups.append(("team", team, n_team, None))
    for item in (extra or []):
        name, codes, size = item[:3]
        groups.append((name, codes, size, item[3] if len(item) > 3 else None))
    offsets, start = [], 0
    for _, _, size, _ in groups:
        offsets.append(start)
        start += size
    k = start
    idx = [off + g[1] for g, off in zip(groups, offsets)]
    wts = [np.ones(n) if g[3] is None else np.asarray(g[3], float) for g in groups]
    rows = np.arange(n)
    Z = sparse.csr_matrix((np.concatenate(wts), (np.tile(rows, len(groups)), np.concatenate(idx))), shape=(n, k))
    ztz = (Z.T @ Z).toarray()
    zty = Z.T @ y
    s2 = max(float(y.var()), 1e-9)
    tau = {g[0]: s2 / (len(groups) + 1) for g in groups}
    if init:                                          # warm start from an earlier fit (bootstrap refits)
        s2 = float(init.get("other", s2))
        for name in tau:
            tau[name] = float(init.get(name, tau[name]))
    prev = np.inf
    for _ in range(n_iter):
        prior = np.concatenate([np.full(g[2], 1 / tau[g[0]]) for g in groups])
        sigma = np.linalg.inv(ztz / s2 + np.diag(prior))
        mean = sigma @ zty / s2
        resid = y - Z @ mean
        trace = 0.0
        for i in range(len(groups)):
            for j in range(len(groups)):
                trace += float((wts[i] * wts[j] * sigma[idx[i], idx[j]]).sum())
        s2 = max((float(resid @ resid) + trace) / n, 1e-9)
        for g, off in zip(groups, offsets):
            seg = slice(off, off + g[2])
            tau[g[0]] = max((float(mean[seg] @ mean[seg]) + float(np.trace(sigma[seg, seg]))) / g[2], 1e-9)
        total = s2 + sum(tau.values())
        if abs(prev - total) < tol:
            break
        prev = total
    out = {"car": tau["car"], "team": tau.get("team", 0.0), "driver": tau["driver"], "other": s2}
    for item in (extra or []):
        out[item[0]] = tau[item[0]]
    if return_effects:
        prior = np.concatenate([np.full(g[2], 1 / tau[g[0]]) for g in groups])
        mean = np.linalg.solve(ztz / s2 + np.diag(prior), zty / s2)
        out["fitted"] = Z @ mean
        out["resid"] = y - out["fitted"]
        out["effects"] = {g[0]: mean[off:off + g[2]] for g, off in zip(groups, offsets)}
    return out


def shares(var: dict) -> dict:
    """Shares of the total. 'car' is everything that belongs to the constructor: persistent plus season-specific."""
    total = var["car"] + var.get("team", 0.0) + var["driver"] + var["other"]
    return {"car": (var["car"] + var.get("team", 0.0)) / total, "driver": var["driver"] / total,
            "other": var["other"] / total, "car_persistent": var.get("team", 0.0) / total,
            "car_sd": float(np.sqrt(var["car"] + var.get("team", 0.0))), "driver_sd": float(np.sqrt(var["driver"])),
            "other_sd": float(np.sqrt(var["other"])), "total_sd": float(np.sqrt(total))}


FIELD_FOR_PLACES = 20


def places_per_unit(total_sd: float, field: int = FIELD_FOR_PLACES) -> float:
    """Grid places per unit of score, so that the pooled score spread equals the spread of positions in a
    field of `field` cars (uniform ranks: sd = field * sqrt((1 - 1/field^2) / 12)). Approximate, and the
    same factor is applied to car, driver and everything else. Because the total is fixed by construction,
    the result is the shares re-expressed (places = sqrt(share) x about 5.8), not an independent measure."""
    return field * float(np.sqrt((1 - 1 / field ** 2) / 12)) / max(total_sd, 1e-9)


def _codes(values: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values)
    return codes, len(uniques)


def fit_era(rows: pd.DataFrame, column: str) -> dict:
    d = rows[rows[column].notna()]
    driver, nd = _codes(d["driver_id"])
    car, nc = _codes(d["car"])
    team, nt = _codes(d["constructor_id"])
    return shares(fit_components(d[column].to_numpy(float), driver, car, nd, nc, team, nt))


def stability_era(rows: pd.DataFrame, column: str, n_rep: int, seed: int = 0, keep: float = 0.7) -> pd.DataFrame:
    """Refit on random subsets of seasons (drawn without replacement); returns one row of shares per subset.

    A bootstrap that draws a season twice would copy its drivers too and overstate how consistent
    drivers are, so the subsets never repeat a season. The spread across subsets is a stability
    range, not a formal confidence interval."""
    rng = np.random.default_rng(seed)
    by_season = {s: g for s, g in rows[rows[column].notna()].groupby("season")}
    seasons = list(by_season)
    size = max(2, int(round(keep * len(seasons))))
    out = []
    for _ in range(n_rep):
        pick = rng.choice(seasons, size=size, replace=False)
        out.append(fit_era(pd.concat([by_season[s] for s in pick], ignore_index=True), column))
    return pd.DataFrame(out)


def run_tables(rows: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    for label in ERA_LABELS:
        era_rows = rows[rows["era"] == label]
        if len(era_rows) < MIN_ROWS:
            continue
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            point = fit_era(era_rows, column)
            boots = stability_era(era_rows, column, n_boot)
            rec = {"era": label, "outcome": outcome, "driver_races": int(era_rows[column].notna().sum()),
                   "drivers": int(era_rows["driver_id"].nunique()), "car_seasons": int(era_rows["car"].nunique())}
            rec["car_persistent_share"] = point["car_persistent"]
            k = places_per_unit(point["total_sd"])
            for part in PARTS:
                rec[f"{part}_sd_places"] = point[f"{part}_sd"] * k
            rec["total_sd_score"] = point["total_sd"]
            for part in PARTS:
                rec[f"{part}_share"] = point[part]
                rec[f"{part}_low"] = float(boots[part].quantile(0.025))
                rec[f"{part}_high"] = float(boots[part].quantile(0.975))
            out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ charts

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
COLOURS = {"car": "#2a78d6", "driver": "#eb6834", "other": "#c9c8c2"}
LABELS = {"car": "Car (team and season)", "driver": "Driver", "other": "Everything else"}


def chart_shares(table: pd.DataFrame, path: Path, by_time: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "text.color": TEXT, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    eras = [e for e in ERA_LABELS if e in set(table["era"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.62 * len(eras) + 2.9), dpi=160, sharey=True)
    fig.subplots_adjust(top=1 - 1.75 / (0.62 * len(eras) + 2.9), left=0.11, right=0.975, bottom=0.20, wspace=0.14)
    race = table[(table["outcome"] == "race")].set_index("era")
    last = eras[-1]
    sub = (f"Share of the variation in each outcome, split by a model that separates car from driver. "
           f"In {last} the car accounts for {race.loc[last, 'car_share']:.0%} of the race result and the driver "
           f"for {race.loc[last, 'driver_share']:.0%}. Ranges across season subsets are in the CSV.")
    fig.text(0.11, 1 - 0.12 / (0.62 * len(eras) + 2.9), "What decides a result: the car, the driver or everything else",
             fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.11, 1 - 0.5 / (0.62 * len(eras) + 2.9), textwrap.fill(sub, 130), fontsize=9.2, color=TEXT2,
             va="top", linespacing=1.45)
    y = np.arange(len(eras))[::-1]
    for ax, outcome, title in zip(axes, ("qualifying", "race"), ("Qualifying position", "Race finishing position")):
        t = table[table["outcome"] == outcome].set_index("era").reindex(eras)
        left = np.zeros(len(eras))
        for part in PARTS:
            w = t[f"{part}_share"].to_numpy()
            ax.barh(y, w, left=left, height=0.62, color=COLOURS[part], edgecolor=SURFACE, linewidth=2,
                    label=LABELS[part])
            for yi, l, wi in zip(y, left, w):
                if not np.isfinite(wi):
                    continue
                if wi >= 0.12:
                    ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=9,
                            color="white" if part != "other" else TEXT)
                else:  # thin segment: label above it, in text colour
                    ax.text(l + wi / 2, yi + 0.40, f"{wi:.0%}", ha="center", va="bottom", fontsize=9, color=TEXT)
            left = left + np.nan_to_num(w)
        ax.set_title(title, loc="left", fontsize=10.5, color=TEXT, pad=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(eras)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.55, 0.0))
    if by_time is not None and len(by_time):
        note = "Qualifying driver share by lap-time gap instead of position: " + ", ".join(
            f"{r['era']} {r['driver_share']:.0%}" for _, r in by_time.iterrows()) + "."
        fig.text(0.11, 0.075, textwrap.fill(note, 150), fontsize=8.3, color=TEXT2, va="top")
    fig.savefig(path)
    plt.close(fig)


# -------------------------------------------------------------------- main

def load_inputs(con):
    results = con.execute("""SELECT r.race_id, r.season, r.driver_id, r.constructor_id, r.grid, r.position
                             FROM results r JOIN races ra USING (race_id)
                             WHERE NOT (ra.circuit_id = 'indianapolis' AND r.season <= 1960)""").df()
    quali = con.execute("""SELECT q.race_id, q.driver_id, q.position, q.q1
                           FROM qualifying q JOIN races ra USING (race_id)
                           WHERE NOT (ra.circuit_id = 'indianapolis' AND ra.season <= 1960)""").df()
    return results, quali


def qualifying_by_time(rows: pd.DataFrame, quali: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Qualifying shares using the lap-time gap in Q1 instead of the position: a place is worth very
    different amounts of time in a spread-out field and in a tightly bunched one, so the time measure is
    the fairer one wherever times exist (1990 on in practice)."""
    from .car_driver_checks import add_lap_gap
    if "q1" not in quali.columns:
        return pd.DataFrame()
    timed = add_lap_gap(rows, quali)
    timed = timed[timed["gap_score"].notna()]
    out = []
    for label in ERA_LABELS:
        era_rows = timed[timed["era"] == label]
        if len(era_rows) < MIN_ROWS:
            continue
        s = fit_era(era_rows, "gap_score")
        rec = {"era": label, "driver_races_with_times": len(era_rows), "car_share": s["car"],
               "driver_share": s["driver"], "other_share": s["other"]}
        if n_boot > 0 and era_rows["season"].nunique() >= 3:
            boots = stability_era(era_rows, "gap_score", n_boot)
            for part in PARTS:
                rec[f"{part}_low"] = float(boots[part].quantile(0.025))
                rec[f"{part}_high"] = float(boots[part].quantile(0.975))
        out.append(rec)
    return pd.DataFrame(out)


def findings(table: pd.DataFrame, by_time: pd.DataFrame | None = None) -> list[str]:
    lines = []
    for _, r in table.iterrows():
        lines.append(f"{r['era']} {r['outcome']}: car {r['car_share']:.0%} (range {r['car_low']:.0%}-{r['car_high']:.0%}), "
                     f"driver {r['driver_share']:.0%} (range {r['driver_low']:.0%}-{r['driver_high']:.0%}), "
                     f"everything else {r['other_share']:.0%}; {r['driver_races']} driver-races, "
                     f"{r['drivers']} drivers, {r['car_seasons']} car-seasons")
    if "driver_sd_places" in table.columns:
        lines.append("SPREADS IN GRID PLACES (standard deviation on a 20-car field). This is the shares in other units, not new "
                     "information: ranks always spread evenly over the field, so the three parts always add up (in squares) to "
                     "about 5.8 places. It cannot say whether cars bunched up; only the lap-time table below is in real units:")
        for _, r in table.iterrows():
            lines.append(f"  {r['era']} {r['outcome']}: car {r['car_sd_places']:.1f}, driver {r['driver_sd_places']:.1f}, "
                         f"everything else {r['other_sd_places']:.1f} places")
    if by_time is not None and len(by_time):
        lines.append("QUALIFYING BY LAP-TIME GAP (same drivers and races as the timed rows; the fairer yardstick where times exist):")
        for _, r in by_time.iterrows():
            rng = lambda part: (f" (range {r[f'{part}_low']:.0%}-{r[f'{part}_high']:.0%})"
                                if f"{part}_low" in r.index and pd.notna(r[f"{part}_low"]) else "")
            lines.append(f"  {r['era']}: car {r['car_share']:.0%}{rng('car')}, driver {r['driver_share']:.0%}{rng('driver')}, "
                         f"everything else {r['other_share']:.0%}{rng('other')} ({int(r['driver_races_with_times'])} driver-races)")
    return lines


def run(con, out_dir: Path | None = None, n_boot: int = N_BOOT) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "car_vs_driver"
    out.mkdir(parents=True, exist_ok=True)
    results, quali = load_inputs(con)
    rows = build_rows(results, quali)
    table = run_tables(rows, n_boot)
    table.to_csv(out / "variance_shares_by_era.csv", index=False)
    by_time = qualifying_by_time(rows, quali, n_boot)
    by_time.to_csv(out / "variance_shares_qualifying_by_laptime.csv", index=False)
    if len(table):
        chart_shares(table, out / "variance_shares_by_era.png", by_time)
    lines = findings(table, by_time)
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
