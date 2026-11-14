"""Stage 2, deep dive 1: one table of what decides a race result, with 'everything else' opened up.

car_driver.py splits a result into car, driver and "everything else" (61-84% of the race result). This
module opens the last part. For each period it

  1 fits the same car + driver mixed model to the finishing position (rankit score), and keeps what the
    model cannot explain, car-season by car-season and driver by driver (the residual);
  2 explains that residual with blocks of things that happened on the day, and shares the explained
    variance between the blocks by Shapley averaging (each block's contribution averaged over every order
    in which the blocks could be added, so correlated blocks are not credited by their position in a list):
        Qualifying form        the car-and-driver-free part of the grid slot (fit of the same model to the grid)
        Mechanical retirements retired with a failure (and how early)
        Accidents              retired after contact or an off (and how early)
        Safety-car timing      stops made under a neutralisation, stops made just before one (2018 on)
        Strategy               stops, first-stop timing and pit-lane time (net of the number of stops) against the race median (2011 on)
  3 reports each block as a share of the whole result: share = (unexplained share of the car-driver model)
    x (block's fraction of the residual). What no block explains stays "unexplained".

Reading it. The car share contains the car's reliability: a car that keeps breaking down has a low car
effect, so mechanical retirements here are only the part that is luck within a car's own record. The same
holds for strategy: a team's habitual strategy is part of its car effect; the strategy block is how much a
car's strategy differed from the race median on the day. The blocks are associations with the result,
not proof of cause (strategy for the same team-mates is tested separately in strategy.py).

Periods. 1950-69, 1970-89, 1990-2010, 2011-17 (pit stops exist), 2018-21 and 2022-now (lap data exist).
A block that cannot be measured in a period is left out and its share stays inside "unexplained".

Ranges. Car and driver: 20 refits on random 70% subsets of the period's seasons (as in car_driver.py).
Blocks: 200 resamples of whole races, with the car and driver effects held fixed (so they show the
sampling error of the second step only). Neither is a formal confidence interval.

Run:  python -m src.analysis.what_decides
Writes to outputs/what_decides/.
"""
from __future__ import annotations

import itertools
import sys
import textwrap
from math import factorial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import car_driver as cd
from . import race_events as ev

PERIODS = [(1950, 1969, "1950-69"), (1970, 1989, "1970-89"), (1990, 2010, "1990-2010"),
           (2011, 2017, "2011-17"), (2018, 2021, "2018-21"), (2022, 9999, "2022-now")]
PERIOD_LABELS = [p[2] for p in PERIODS]
MIN_ROWS = 400
N_STAB = 20              # season-subset refits for the car and driver ranges
N_BLOCK_BOOT = 200       # race resamples for the block ranges
MIN_COVERAGE = 0.5       # a block is used in a period only if at least this share of its rows are measured
BLOCK_ORDER = ["Qualifying form", "Mechanical retirements", "Accidents", "Retirements", "Safety-car timing", "Strategy"]
MERGE_UNSPECIFIED = 0.25      # above this share of retirements with no stated cause, mechanical and accident are not split


def merge_retirements(d: pd.DataFrame) -> bool:
    """True when too many retirements carry no stated cause (a bare "Retired") for a mechanical/accident split to mean much."""
    ret = d["status_class"].isin(["mechanical", "incident", "unspecified"])
    return bool(ret.sum() > 0 and (d["status_class"] == "unspecified").sum() / ret.sum() > MERGE_UNSPECIFIED)
SEED = 20261022


def period_of(season: int) -> str:
    for a, b, label in PERIODS:
        if a <= season <= b:
            return label
    return "other"


# ------------------------------------------------------------------- data

def load_inputs(con):
    def has(name):
        return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0])
    results = con.execute("""SELECT race_id, season, constructor_id, driver_id, grid, position, laps, status
                             FROM results""").df()
    quali = con.execute("SELECT race_id, driver_id, position FROM qualifying").df() if has("qualifying") \
        else pd.DataFrame(columns=["race_id", "driver_id", "position"])
    pits = con.execute("SELECT race_id, driver_id, stop, lap, duration_s FROM pit_stops").df() if has("pit_stops") \
        else pd.DataFrame(columns=["race_id", "driver_id", "stop", "lap", "duration_s"])
    laps = con.execute("SELECT race_id, lap_number, track_status FROM laps").df() if has("laps") \
        else pd.DataFrame(columns=["race_id", "lap_number", "track_status"])
    return results, quali, pits, laps


def _rescore(d: pd.DataFrame) -> pd.DataFrame:
    """Recompute the rankit scores after rows were dropped, ranking within each race over the rows that remain."""
    d = d.copy()
    for col, score in (("start", "start_score"), ("finish", "finish_score")):
        pos = d[col].where(d[col] > 0)
        rank = pos.groupby(d["race_id"]).rank(method="first")
        d[score] = cd.rankit(rank, d["race_id"]).where(pos.notna())
    return d


def build_table(results: pd.DataFrame, quali: pd.DataFrame, pits: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver-race: scores, status, strategy and safety-car features, period."""
    res = ev.add_status(results)
    rows = cd.build_rows(results, quali)
    info = res[["race_id", "driver_id", "status_class", "finished", "completed"]].drop_duplicates(["race_id", "driver_id"])
    rows = rows.merge(info, on=["race_id", "driver_id"], how="left")
    rows = rows[~rows["status_class"].isin(["disqualified", "not_started"])].copy()
    rows = _rescore(rows)
    rows["period"] = rows["season"].map(period_of)
    rows["era"] = rows["period"]
    stops = ev.stop_features(results, pits)
    rows = rows.merge(stops, on=["race_id", "driver_id"], how="left")
    rows = ev.deviations(rows, ["n_stops", "first_stop", "pit_excess"])
    for c in ("n_stops", "first_stop", "pit_excess"):
        rows[f"d_{c}"] = rows[f"d_{c}"].where(rows[c].notna())       # keep 'not measured' visible
    sc = ev.safety_car_features(results, pits, laps)
    rows = rows.merge(sc, on=["race_id", "driver_id"], how="left")
    return rows.reset_index(drop=True)


# ------------------------------------------------------------------ model

def fit_parts(rows: pd.DataFrame, column: str) -> dict:
    """Mixed-model fit of one score column. Returns shares and, aligned to the rows used, the car part,
    the driver part and the residual of every row."""
    d = rows[rows[column].notna()]
    driver, nd = cd._codes(d["driver_id"])
    car, nc = cd._codes(d["car"])
    team, nt = cd._codes(d["constructor_id"])
    fit = cd.fit_components(d[column].to_numpy(float), driver, car, nd, nc, team, nt, return_effects=True)
    eff = fit["effects"]
    car_part = eff["car"][car] + eff["team"][team]
    driver_part = eff["driver"][driver]
    return {"shares": cd.shares(fit), "index": d.index, "car_part": car_part, "driver_part": driver_part,
            "resid": fit["resid"]}


def shapley_r2(blocks: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Share of the variance of y explained by each block, Shapley-averaged over all orders of adding blocks.
    The values add up to the R-squared of all blocks together."""
    names = list(blocks)
    k = len(names)
    yc = y - y.mean()
    tot = float(yc @ yc)
    if tot <= 0 or k == 0:
        return {n: 0.0 for n in names}
    cols = {n: np.asarray(b, float).reshape(len(y), -1) for n, b in blocks.items()}
    cols = {n: b - b.mean(axis=0) for n, b in cols.items()}
    cache: dict[int, float] = {0: 0.0}

    def r2(mask: int) -> float:
        if mask in cache:
            return cache[mask]
        X = np.column_stack([cols[n] for i, n in enumerate(names) if mask >> i & 1])
        beta, *_ = np.linalg.lstsq(X, yc, rcond=None)
        fit = X @ beta
        cache[mask] = float(fit @ fit) / tot
        return cache[mask]

    out = {}
    for i, n in enumerate(names):
        total = 0.0
        for mask in range(1 << k):
            if mask >> i & 1:
                continue
            size = bin(mask).count("1")
            w = factorial(size) * factorial(k - size - 1) / factorial(k)
            total += w * (r2(mask | (1 << i)) - r2(mask))
        out[n] = total
    return out


def make_blocks(d: pd.DataFrame, grid_resid: np.ndarray) -> dict[str, np.ndarray]:
    """The explanatory blocks that exist for these rows (a block needs MIN_COVERAGE measured rows)."""
    blocks: dict[str, np.ndarray] = {}
    late = (1 - d["completed"].to_numpy(float))
    blocks["Qualifying form"] = np.nan_to_num(grid_resid)
    if merge_retirements(d):
        ret = d["status_class"].isin(["mechanical", "incident", "unspecified"]).to_numpy(float)
        if ret.sum() > 5:
            blocks["Retirements"] = np.column_stack([ret, ret * late])
    else:
        mech = d["status_class"].isin(["mechanical", "unspecified"]).to_numpy(float)   # few unspecified: counted as failures
        inc = (d["status_class"] == "incident").to_numpy(float)
        if mech.sum() > 5:
            blocks["Mechanical retirements"] = np.column_stack([mech, mech * late])
        if inc.sum() > 5:
            blocks["Accidents"] = np.column_stack([inc, inc * late])
    if "free_stops" in d and d["free_stops"].notna().mean() >= MIN_COVERAGE:
        blocks["Safety-car timing"] = d[["free_stops", "unlucky_stops"]].fillna(0).to_numpy(float)
    if "d_n_stops" in d and d["d_n_stops"].notna().mean() >= MIN_COVERAGE:
        blocks["Strategy"] = d[["d_n_stops", "d_first_stop", "d_pit_excess"]].fillna(0).to_numpy(float)
    return blocks


def decompose_period(rows: pd.DataFrame, n_stab: int = N_STAB, n_boot: int = N_BLOCK_BOOT, seed: int = SEED) -> dict:
    """Everything for one period: shares, block contributions, ranges, side statistics."""
    fin = fit_parts(rows, "finish_score")
    grid = fit_parts(rows, "start_score")
    d = rows.loc[fin["index"]].copy()
    grid_resid = pd.Series(grid["resid"], index=grid["index"]).reindex(d.index).to_numpy()
    r = fin["resid"]
    blocks = make_blocks(d, grid_resid)
    contrib = shapley_r2(blocks, r)
    other = fin["shares"]["other"]
    explained = sum(contrib.values())
    rec = {"driver_races": len(d), "drivers": int(d["driver_id"].nunique()), "car_seasons": int(d["car"].nunique()),
           "car_share": fin["shares"]["car"], "driver_share": fin["shares"]["driver"], "other_share": other,
           "blocks_measured": ", ".join(blocks), "block_r2_total": explained}
    for name in BLOCK_ORDER:
        rec[f"{name}_share"] = other * contrib.get(name, np.nan) if name in contrib else np.nan
    rec["unexplained_share"] = other * (1 - explained)
    # ranges for the blocks: resample whole races
    rng = np.random.default_rng(seed)
    race_codes, uniq = pd.factorize(d["race_id"])
    by_race = [np.flatnonzero(race_codes == i) for i in range(len(uniq))]
    boots = {n: [] for n in blocks}
    boots["_unexplained"] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_race), len(by_race))
        idx = np.concatenate([by_race[i] for i in pick])
        c = shapley_r2({n: b[idx] for n, b in blocks.items()}, r[idx])
        for n in blocks:
            boots[n].append(other * c[n])
        boots["_unexplained"].append(other * (1 - sum(c.values())))
    def q(values, p):
        return float(np.quantile(values, p)) if len(values) else float("nan")
    for n in blocks:
        rec[f"{n}_low"], rec[f"{n}_high"] = q(boots[n], 0.025), q(boots[n], 0.975)
    rec["unexplained_low"], rec["unexplained_high"] = q(boots["_unexplained"], 0.025), q(boots["_unexplained"], 0.975)
    # car and driver ranges: season subsets
    if n_stab:
        stab = cd.stability_era(rows, "finish_score", n_stab, seed=seed)
        for part in ("car", "driver"):
            rec[f"{part}_low"] = float(stab[part].quantile(0.025))
            rec[f"{part}_high"] = float(stab[part].quantile(0.975))
    # side statistics
    rec["car_share_grid"] = grid["shares"]["car"]
    rec["driver_share_grid"] = grid["shares"]["driver"]
    both = pd.DataFrame({"cf": pd.Series(fin["car_part"], index=fin["index"]),
                         "cg": pd.Series(grid["car_part"], index=grid["index"]),
                         "df": pd.Series(fin["driver_part"], index=fin["index"]),
                         "dg": pd.Series(grid["driver_part"], index=grid["index"])}).dropna()
    rec["car_effect_corr_grid_vs_race"] = float(both["cf"].corr(both["cg"])) if len(both) > 10 else np.nan
    rec["driver_effect_corr_grid_vs_race"] = float(both["df"].corr(both["dg"])) if len(both) > 10 else np.nan
    only = d[d["finished"]].copy()
    if len(only) >= MIN_ROWS:
        only = _rescore(only)
        sh = cd.fit_era(only, "finish_score")
        rec["car_share_finishers_only"], rec["driver_share_finishers_only"] = sh["car"], sh["driver"]
        fo = fit_parts(only, "finish_score")
        a = pd.Series(fin["car_part"], index=fin["index"]).groupby(d["car"]).mean()
        b = pd.Series(fo["car_part"], index=fo["index"]).groupby(only.loc[fo["index"], "car"]).mean()
        both_cars = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        rec["car_effect_corr_all_vs_finishers"] = float(both_cars["a"].corr(both_cars["b"])) if len(both_cars) > 5 else np.nan
    else:
        rec["car_share_finishers_only"] = rec["driver_share_finishers_only"] = np.nan
        rec["car_effect_corr_all_vs_finishers"] = np.nan
    # who is behind the retirements: how much of the chance of retiring belongs to the car, to the driver, to luck
    merged = merge_retirements(d)
    rec["retirements_merged"] = merged
    for key, cls in (("mech", ["mechanical", "unspecified"]), ("inc", ["incident"]), ("ret", ["mechanical", "incident", "unspecified"])):
        ind = d["status_class"].isin(cls).astype(float)
        if (merged and key in ("mech", "inc")) or (not merged and key == "ret"):
            ind = ind * 0                      # not meaningful for this period; left empty
        if ind.sum() >= 30:
            sh = cd.fit_era(d.assign(_ind=ind), "_ind")
            rec[f"{key}_car_share"], rec[f"{key}_driver_share"], rec[f"{key}_luck_share"] = sh["car"], sh["driver"], sh["other"]
        else:
            rec[f"{key}_car_share"] = rec[f"{key}_driver_share"] = rec[f"{key}_luck_share"] = np.nan
    mix = d["status_class"].value_counts(normalize=True)
    for c in ("finished", "lapped", "mechanical", "incident", "unspecified"):
        rec[f"status_{c}"] = float(mix.get(c, 0.0))
    for name, b in blocks.items():
        rec[f"{name}_r2_alone"] = float(shapley_r2({name: b}, r)[name])
    return rec


def run_tables(rows: pd.DataFrame, n_stab: int = N_STAB, n_boot: int = N_BLOCK_BOOT) -> pd.DataFrame:
    out = []
    for label in PERIOD_LABELS:
        sub = rows[rows["period"] == label]
        if len(sub) < MIN_ROWS:
            continue
        rec = decompose_period(sub, n_stab, n_boot)
        rec["period"] = label
        out.append(rec)
    table = pd.DataFrame(out)
    if len(table):
        table = table[["period"] + [c for c in table.columns if c != "period"]]
    return table


# ------------------------------------------------------------------ output

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
SEGMENTS = [("Car (team and season)", ["car_share"], "#2a78d6"),
            ("Driver", ["driver_share"], "#eb6834"),
            ("Qualifying form", ["Qualifying form_share"], "#1baf7a"),
            ("Strategy", ["Strategy_share"], "#eda100"),
            ("Race-day events (failures, accidents, safety-car timing)",
             ["Mechanical retirements_share", "Accidents_share", "Retirements_share", "Safety-car timing_share"], "#e87ba4"),
            ("Unexplained", ["unexplained_share"], "#c9c8c2")]


def segment_values(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"period": table["period"]})
    for label, cols, _ in SEGMENTS:
        out[label] = table[[c for c in cols if c in table.columns]].fillna(0).sum(axis=1)
    return out


def chart(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False})
    seg = segment_values(table)
    n = len(seg)
    h = 0.62 * n + 3.0
    fig, ax = plt.subplots(figsize=(11, h), dpi=160)
    fig.subplots_adjust(top=1 - 1.5 / h, left=0.10, right=0.975, bottom=1.15 / h)
    last = seg.iloc[-1]
    fig.text(0.10, 1 - 0.12 / h, "What decides a race result, with 'everything else' opened up", fontsize=13.5,
             fontweight="bold", va="top")
    sub = (f"Share of the variation in finishing position. In {last['period']}: car {last['Car (team and season)']:.0%}, "
           f"driver {last['Driver']:.0%}, qualifying form {last['Qualifying form']:.0%}, strategy {last['Strategy']:.0%}, "
           f"race-day events {last[SEGMENTS[4][0]]:.0%}, unexplained {last['Unexplained']:.0%}. A block that cannot be "
           f"measured in a period (strategy before 2011, safety-car timing before 2018) stays inside unexplained.")
    fig.text(0.10, 1 - 0.5 / h, textwrap.fill(sub, 135), fontsize=9.2, color=TEXT2, va="top", linespacing=1.45)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for label, _, colour in SEGMENTS:
        w = seg[label].to_numpy()
        ax.barh(y, w, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
        for yi, l, wi in zip(y, left, w):
            if wi >= 0.06:
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=9,
                        color=TEXT if colour in ("#c9c8c2", "#eda100", "#e87ba4") else "white")
        left = left + w
    ax.set_yticks(y)
    ax.set_yticklabels(seg["period"])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.54, 0.0))
    fig.savefig(path)
    plt.close(fig)


def _pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def findings(table: pd.DataFrame) -> list[str]:
    if not len(table):
        return ["Not enough driver-races to decompose."]
    L = ["WHAT DECIDES A RACE RESULT (shares of the variation in finishing position; ranges: car and driver over season "
         "subsets, blocks over resampled races)"]
    for _, r in table.iterrows():
        parts = [f"car {_pct(r['car_share'])} ({_pct(r.get('car_low'))}-{_pct(r.get('car_high'))})",
                 f"driver {_pct(r['driver_share'])} ({_pct(r.get('driver_low'))}-{_pct(r.get('driver_high'))})"]
        for name in BLOCK_ORDER:
            v = r.get(f"{name}_share")
            if pd.notna(v):
                parts.append(f"{name.lower()} {_pct(v)} ({_pct(r[f'{name}_low'])}-{_pct(r[f'{name}_high'])})")
        parts.append(f"unexplained {_pct(r['unexplained_share'])} ({_pct(r['unexplained_low'])}-{_pct(r['unexplained_high'])})")
        L.append(f"  {r['period']} ({int(r['driver_races'])} driver-races): " + ", ".join(parts))
    L.append("WHICH BLOCKS EXIST IN EACH PERIOD")
    for _, r in table.iterrows():
        L.append(f"  {r['period']}: {r['blocks_measured']}")
    L.append("HOW THE RACE ENDED (share of driver-races)")
    for _, r in table.iterrows():
        L.append(f"  {r['period']}: finished lead lap {_pct(r['status_finished'])}, lapped {_pct(r['status_lapped'])}, "
                 f"mechanical {_pct(r['status_mechanical'])}, accident {_pct(r['status_incident'])}, cause not stated "
                 f"{_pct(r['status_unspecified'])}")
    L.append("CAR AND DRIVER ON THE GRID vs IN THE RACE")
    for _, r in table.iterrows():
        L.append(f"  {r['period']}: car share {_pct(r['car_share_grid'])} on the grid, {_pct(r['car_share'])} in the race; "
                 f"driver share {_pct(r['driver_share_grid'])} on the grid, {_pct(r['driver_share'])} in the race; "
                 f"a car's effect on the grid and in the race correlate {r['car_effect_corr_grid_vs_race']:.2f}, a driver's "
                 f"{r['driver_effect_corr_grid_vs_race']:.2f}")
    L.append("ALL CARS vs ONLY CARS THAT FINISHED. Both shares rise when retirements are removed, partly just because the noise "
             "shrinks, so read this for direction, not size; the last number is how closely the cars' order agrees between the two fits")
    for _, r in table.iterrows():
        L.append(f"  {r['period']}: car {_pct(r['car_share'])} all cars, {_pct(r['car_share_finishers_only'])} finishers only; "
                 f"driver {_pct(r['driver_share'])} against {_pct(r['driver_share_finishers_only'])}; "
                 f"car-season effects agree at {r['car_effect_corr_all_vs_finishers']:.2f}")
    L.append("WHO IS BEHIND THE RETIREMENTS (share of the variation in whether a car retired: its car-season, its driver, luck)")
    for _, r in table.iterrows():
        if r.get("retirements_merged"):
            L.append(f"  {r['period']}: any retirement (cause too often not stated to split) {_pct(r['ret_car_share'])} car, "
                     f"{_pct(r['ret_driver_share'])} driver, {_pct(r['ret_luck_share'])} luck")
        else:
            L.append(f"  {r['period']}: mechanical {_pct(r['mech_car_share'])} car, {_pct(r['mech_driver_share'])} driver, "
                     f"{_pct(r['mech_luck_share'])} luck; accident {_pct(r['inc_car_share'])} car, {_pct(r['inc_driver_share'])} driver, "
                     f"{_pct(r['inc_luck_share'])} luck")
    L.append("STRATEGY BLOCK: it only sees how many stops, how early the first stop and how quick the pit work were, compared with the "
             "race median, for cars that finished. It cannot see timing against rivals, tyre choice or reaction to a safety car, so its "
             "share is a floor for what strategy does, not an estimate of it (strategy.py tests team-mates directly).")
    L.append("READ WITH CARE: the car share includes the car's reliability and habitual strategy; the blocks are associations "
             "with the result, not proof of cause; the incident/mechanical split reads a free-text status and is approximate; "
             "a retirement is counted by how early it came, but not why.")
    return L


def status_table(results: pd.DataFrame) -> pd.DataFrame:
    """Every distinct status with the class it was given and how often it occurs in each period, so the
    classification can be checked by eye."""
    d = ev.add_status(results)
    d["period"] = d["season"].map(period_of)
    t = d.groupby(["status", "status_class", "period"]).size().unstack("period", fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False).reset_index()


def run(con, out_dir: Path | None = None, n_stab: int = N_STAB, n_boot: int = N_BLOCK_BOOT) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "what_decides"
    out.mkdir(parents=True, exist_ok=True)
    results, quali, pits, laps = load_inputs(con)
    status_table(results).to_csv(out / "status_classification.csv", index=False)
    rows = build_table(results, quali, pits, laps)
    table = run_tables(rows, n_stab, n_boot)
    table.to_csv(out / "decomposition_by_period.csv", index=False)
    if len(table):
        segment_values(table).to_csv(out / "decomposition_chart_values.csv", index=False)
        chart(table, out / "decomposition_by_period.png")
    lines = findings(table)
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
