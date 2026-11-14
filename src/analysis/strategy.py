"""Stage 2, deep dive 2: does strategy decide results? A team-mate experiment.

Comparing teams' strategies mixes strategy with car speed. Two cars of the same team in the same race share the
car, the circuit, the weather and (mostly) the safety cars, so the way their results differ is the closest thing
the data offers to an experiment. For each race in which a team's two cars both finished on the lead lap
(so neither was damaged or slowed by a problem big enough to lose a lap), the module compares the two:

    gain      places the first car finished ahead of the second (negative if behind)
    grid      places the first car started ahead of the second            (the control)
    stops     stops of the first car minus stops of the second
    first     lap of its first stop minus the other's                      (positive = later)
    pit time  pit-lane time of the first minus the second, seconds, net of the number of stops (a car that
              stops twice is not "slower in the pits" for it)
    free      stops under a safety car, VSC or red flag, first minus second          (2018 on)
    unlucky   stops in the three laps before a neutralisation, first minus second    (2018 on)

and fits   gain = b1 x grid + b2 x stops + b3 x first + b4 x pit time [+ b5 x free + b6 x unlucky]
with no intercept (nobody is "first": swapping the two cars flips every sign and changes nothing). Intervals
come from resampling whole races. Coefficients are in finishing places: b2 is the places gained by making one
more stop than the team-mate, b3 by stopping one lap later, b1 what one grid place is worth (the yardstick).

Reading it. A team chooses who gets which strategy, often the car that is behind, so the estimates are biased
towards "a different strategy is worse"; controlling for the grid and keeping only lead-lap finishers reduces
that, and the second row of every table keeps only pairs that started within three places of each other. A
stop caused by a puncture or a front-wing change shows up as an extra stop. It is a test of whether strategy
moves results among cars that are otherwise alike, not a measure of the best possible strategy; for that the
Stage 3 simulator compares strategy levels on the same race.

Periods: 2011-17 (pit stops from 2011), 2018-21, 2022-now and all of 2011-now.

Run:  python -m src.analysis.strategy
Writes to outputs/strategy_teammates/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import what_decides as wd
from . import race_events as ev

N_BOOT = 500
MIN_PAIRS = 60
NEAR_GRID = 3
SLOW_PIT_S = 8.0            # a stop this much slower than the race median is taken as a repair or a problem
SEED = 20261023
TERMS = [("grid", "Started one place ahead of the team-mate"),
         ("stops", "One more stop than the team-mate"),
         ("first", "First stop one lap later than the team-mate"),
         ("pit_time", "One second more pit-lane time than the team-mate, net of the number of stops"),
         ("free", "One more stop under a safety car, VSC or red flag"),
         ("unlucky", "One more stop just before a neutralisation")]
SPLIT_PERIODS = ["2011-17", "2018-21", "2022-now"]


def build_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per team-mate pair in which both cars finished on the lead lap and the stops are known.
    All difference columns are first car minus second, the first being the one with the smaller driver id."""
    d = rows[(rows["status_class"] == "finished") & rows["n_stops"].notna()].copy()
    sizes = rows.groupby(["race_id", "constructor_id"])["driver_id"].transform("count")
    d = d.merge(rows.assign(_n=sizes)[["race_id", "driver_id", "_n"]], on=["race_id", "driver_id"])
    d = d[d["_n"] == 2]                                         # exactly two cars started for the team
    cols = ["race_id", "season", "period", "constructor_id", "driver_id", "position", "start", "n_stops", "first_lap",
            "pit_excess", "free_stops", "unlucky_stops"]
    d = d[cols].sort_values(["race_id", "constructor_id", "driver_id"])
    g = d.groupby(["race_id", "constructor_id"])
    d = d[g["driver_id"].transform("count") == 2]
    a = d.groupby(["race_id", "constructor_id"], sort=False).head(1).reset_index(drop=True)
    b = d.groupby(["race_id", "constructor_id"], sort=False).tail(1).reset_index(drop=True)
    pairs = a.merge(b, on=["race_id", "constructor_id"], suffixes=("_a", "_b"))
    out = pd.DataFrame({
        "race_id": pairs["race_id"], "season": pairs["season_a"], "period": pairs["period_a"],
        "constructor_id": pairs["constructor_id"], "driver_a": pairs["driver_id_a"], "driver_b": pairs["driver_id_b"],
        "gain": pairs["position_b"] - pairs["position_a"],
        "grid": pairs["start_b"] - pairs["start_a"],
        "stops": pairs["n_stops_a"] - pairs["n_stops_b"],
        "first": pairs["first_lap_a"] - pairs["first_lap_b"],
        "pit_time": pairs["pit_excess_a"] - pairs["pit_excess_b"],
        "free": pairs["free_stops_a"] - pairs["free_stops_b"],
        "unlucky": pairs["unlucky_stops_a"] - pairs["unlucky_stops_b"],
        "worst_pit_excess": pairs[["pit_excess_a", "pit_excess_b"]].max(axis=1)})
    # a car that never stopped has no first stop; inventing one would be dishonest, so those pairs are dropped
    out = out.dropna(subset=["gain", "grid", "stops", "first", "pit_time"]).reset_index(drop=True)
    return out


def usable_terms(pairs: pd.DataFrame) -> list[str]:
    terms = ["grid", "stops", "first", "pit_time"]
    for t in ("free", "unlucky"):
        if pairs[t].notna().mean() >= 0.5 and pairs[t].fillna(0).abs().sum() > 0:
            terms.append(t)
    return terms


def fit_effects(pairs: pd.DataFrame, terms: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame | None:
    """No-intercept least squares of gain on the terms, intervals from resampling whole races."""
    d = pairs.copy()
    for t in terms:
        d[t] = d[t].fillna(0.0)
    if len(d) < MIN_PAIRS or d["race_id"].nunique() < 20:
        return None
    X = d[terms].to_numpy(float)
    y = d["gain"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    codes, uniq = pd.factorize(d["race_id"])
    by_race = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_race), len(by_race))
        idx = np.concatenate([by_race[i] for i in pick])
        try:
            boots.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots) if boots else np.full((1, len(terms)), np.nan)
    lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    return pd.DataFrame({"term": terms, "estimate": beta, "low": lo, "high": hi,
                         "pairs": len(d), "races": int(d["race_id"].nunique())})


def effects_table(pairs: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    out = []
    groups = [(p, pairs[pairs["period"] == p]) for p in SPLIT_PERIODS] + [("2011-now", pairs)]
    for label, sub in groups:
        for variant, vsub in (("all pairs", sub), (f"started within {NEAR_GRID} places", sub[sub["grid"].abs() <= NEAR_GRID]),
                              ("no slow stops", sub[sub["worst_pit_excess"].fillna(0) <= SLOW_PIT_S])):
            terms = usable_terms(vsub) if len(vsub) else []
            res = fit_effects(vsub, terms, n_boot) if terms else None
            if res is None:
                continue
            res.insert(0, "variant", variant)
            res.insert(0, "period", label)
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def split_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Plain counts, no model: when team-mates stopped a different number of times, how often did the one who
    stopped more finish ahead; at equal stops, how often did the earlier stopper. Pairs are restricted to those
    that started within NEAR_GRID places so the grid gap does not decide the answer."""
    out = []
    for label, sub in [(p, pairs[pairs["period"] == p]) for p in SPLIT_PERIODS] + [("2011-now", pairs)]:
        near = sub[sub["grid"].abs() <= NEAR_GRID]
        diff = near[near["stops"] != 0]
        k = int(((diff["stops"] > 0) & (diff["gain"] > 0)).sum() + ((diff["stops"] < 0) & (diff["gain"] < 0)).sum())
        lo, hi = wilson(k, len(diff))
        out.append({"period": label, "comparison": "more stops finished ahead (different stop counts)", "pairs": len(diff),
                    "share": k / len(diff) if len(diff) else np.nan, "low": lo, "high": hi})
        same = near[(near["stops"] == 0) & (near["first"].abs() >= 3)]
        k2 = int(((same["first"] < 0) & (same["gain"] > 0)).sum() + ((same["first"] > 0) & (same["gain"] < 0)).sum())
        lo2, hi2 = wilson(k2, len(same))
        out.append({"period": label, "comparison": "earlier first stop finished ahead (equal stops, 3+ laps apart)",
                    "pairs": len(same), "share": k2 / len(same) if len(same) else np.nan, "low": lo2, "high": hi2})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ output

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
PERIOD_COLOURS = {"2011-17": "#2a78d6", "2018-21": "#eb6834", "2022-now": "#1baf7a"}


def chart(effects: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    e = effects[(effects["variant"] == "all pairs") & effects["period"].isin(SPLIT_PERIODS)]
    terms = [(t, label) for t, label in TERMS if t in set(e["term"])]
    import textwrap
    h = 0.95 * len(terms) + 2.2
    fig, ax = plt.subplots(figsize=(10.5, h), dpi=160)
    fig.subplots_adjust(left=0.33, right=0.97, top=1 - 1.15 / h, bottom=0.8 / h)
    fig.text(0.02, 1 - 0.12 / h, "Does strategy move results between team-mates?", fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.52 / h, "Finishing places gained by the first car, both cars finishing on the lead lap, grid place controlled. "
             "Dots are estimates, lines 95% intervals from resampled races.", fontsize=9, color=TEXT2, va="top")
    ys = np.arange(len(terms))[::-1]
    shown = []
    for k, period in enumerate(SPLIT_PERIODS):
        for (t, _), y in zip(terms, ys):
            r = e[(e["period"] == period) & (e["term"] == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = y + (1 - k) * 0.22
            ax.plot([r["low"], r["high"]], [yy, yy], color=PERIOD_COLOURS[period], lw=2, solid_capstyle="round")
            ax.scatter([r["estimate"]], [yy], color=PERIOD_COLOURS[period], s=34, zorder=3,
                       label=period if period not in shown else None)
            shown.append(period)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([textwrap.fill(label, 34) for _, label in terms])
    ax.set_xlabel("Finishing places gained (positive = the first car finished ahead)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def _fmt(r) -> str:
    return f"{r['estimate']:+.2f} places ({r['low']:+.2f} to {r['high']:+.2f})"


def findings(pairs: pd.DataFrame, effects: pd.DataFrame, splits: pd.DataFrame) -> list[str]:
    if not len(pairs) or not len(effects):
        return ["Not enough team-mate pairs with pit-stop data (need results and pit stops from 2011 on)."]
    L = [f"STRATEGY AMONG TEAM-MATES: {len(pairs)} pairs in {pairs['race_id'].nunique()} races, both cars finishing on the "
         "lead lap, 2011 on"]
    ref = effects[(effects["variant"] == "all pairs") & (effects["term"] == "grid")].set_index("period")
    for label in SPLIT_PERIODS + ["2011-now"]:
        sub = effects[(effects["period"] == label) & (effects["variant"] == "all pairs")]
        if sub.empty:
            continue
        L.append(f"  {label} ({int(sub['pairs'].iloc[0])} pairs, {int(sub['races'].iloc[0])} races):")
        for t, text in TERMS:
            r = sub[sub["term"] == t]
            if len(r):
                L.append(f"    {text}: {_fmt(r.iloc[0])}")
        for variant, text in ((f"started within {NEAR_GRID} places", f"only pairs that started within {NEAR_GRID} places"),
                              ("no slow stops", f"only pairs with no stop more than {SLOW_PIT_S:.0f} s slower than the field's (repairs and problems out)")):
            near = effects[(effects["period"] == label) & (effects["variant"] == variant)]
            if len(near):
                bits = []
                for t in ("stops", "first"):
                    r = near[near["term"] == t]
                    if len(r):
                        bits.append(f"{dict(TERMS)[t].lower()}: {_fmt(r.iloc[0])}")
                L.append(f"    {text} ({int(near['pairs'].iloc[0])} pairs): " + "; ".join(bits))
        if label in ref.index:
            g = ref.loc[label]
            L.append(f"    yardstick: one grid place is worth {g['estimate']:.2f} finishing places here")
    L.append("PLAIN COUNTS (pairs that started within 3 places of each other)")
    for _, r in splits.iterrows():
        if r["pairs"] > 0:
            L.append(f"  {r['period']}, {r['comparison']}: {r['share']:.0%} of {int(r['pairs'])} ({r['low']:.0%}-{r['high']:.0%}); "
                     "50% would mean no effect")
    L.append("READ WITH CARE: teams give the car that is behind a different plan, so these estimates lean towards 'different "
             "strategy is worse'; a stop caused by a puncture or damage counts as a stop; pairs are few in each period; this "
             "tests whether strategy matters among otherwise alike cars, not what the best strategy was.")
    return L


def run(con, out_dir: Path | None = None, n_boot: int = N_BOOT) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "strategy_teammates"
    out.mkdir(parents=True, exist_ok=True)
    results, quali, pits, laps = wd.load_inputs(con)
    rows = wd.build_table(results, quali, pits, laps)
    pairs = build_pairs(rows)
    pairs.to_csv(out / "teammate_pairs.csv", index=False)
    effects = effects_table(pairs, n_boot) if len(pairs) else pd.DataFrame()
    splits = split_table(pairs) if len(pairs) else pd.DataFrame()
    effects.to_csv(out / "strategy_effects.csv", index=False)
    splits.to_csv(out / "strategy_plain_counts.csv", index=False)
    if len(effects):
        chart(effects, out / "strategy_effects.png")
    lines = findings(pairs, effects, splits)
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
