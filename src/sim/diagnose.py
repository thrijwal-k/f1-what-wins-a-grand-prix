"""Stage 3, part 6: where does the simulator's randomness at the front come from, and what would it cost to remove it?

Run:  python -m src.sim.calibrate   (once)   then   python -m src.sim.diagnose
Writes outputs/sim_diagnose/.

Replay shows three linked faults: pole wins 44% against 55% real, the grid's grip on the order dissolves too early when strategies,
safety cars and retirements are sampled (L1), and probabilities are too flat. This module takes them apart. Nothing is refitted.

  1 FACTORS     All eight combinations of strategies / safety cars / retirements known or sampled, scored on pole wins, the
                grid-order grip curve (early and late laps), passes, close running and lap 1. A Shapley split says which
                unknown is responsible for each gap, and a leave-one-out from L1 says what is left when one is given back.
  2 LEVERS      With everything known (L0) each noise and passing setting is turned down (or up) on its own. The table says what
                it does to pole wins and what it costs in grip, passes, close running and lap 1, and the value that would
                close the pole gap if the lever were the only thing changed. A lever that closes the gap only by breaking
                the passing or the grip is not the cause.
  3 MARGIN      Pole wins by how much faster than every other car the pole sitter's pace was. Real against simulated, at
                L0 and with the rest sampled. It says whether the missing wins are the pole sitter being quickest and still
                losing (race noise), or being slower and the real race not punishing it (pace measurement).
  4 STOPS       Stops per car by season, real against the strategies drawn when they are sampled (the holdout draws fewer).
  5 POLE LOSS   How the lead is lost in each combination, against real (needs the database).
"""
from __future__ import annotations

import itertools
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from ..analysis import race_flow
from ..analysis.race_flow import FRACTIONS
from . import ablation as A
from . import calibrate as C
from . import engine as E
from . import replay as R
from .calibrate import RaceTruth, split_of
from .engine import BIG, Params, spearman_masked
from .sample import MIN_POOL, Pool, sample_inputs

S_FACTOR = 60
S_LEVER = 24
S_EVENTS = 6
EARLY = 4                       # the first four points of FRACTIONS (after lap 1 to 30% of the race) are "early"
FACTOR_METRICS = ("pole_win", "early_gap", "late_gap", "abs_gap", "grid_finish_gap", "passes_ratio", "close_ratio", "lap1_ratio")
MARGIN_EDGES = (-math.inf, -0.3, 0.0, 0.2, 0.5, math.inf)
MARGIN_LABELS = ("pole sitter 0.3+ s/lap slower than the best car", "0 to 0.3 s/lap slower", "0 to 0.2 s/lap faster than the rest",
                 "0.2 to 0.5 s/lap faster", "0.5+ s/lap faster")

# name, fields scaled, multipliers tried (1.0 is the fitted setting and is always included)
LEVERS = (
    ("lap-1 noise", ("sigma_start",), (0.5, 0.75)),
    ("lap noise, whole field", ("sigma_lap",), (0.5, 0.75)),
    ("lap noise, top three", ("noise_top",), (0.5, 0.75)),
    ("pace drift", ("drift_sd",), (0.0, 0.5)),
    ("stint effect", ("sigma_stint",), (0.0, 0.5)),
    ("stop-cost spread", ("sigma_pit", "sigma_pit_top"), (0.0, 0.5)),
    ("passing-difficulty spread", ("thr_sd",), (0.0, 0.5)),
    ("passing threshold", ("thr",), (1.5, 2.0)),
    ("grid spacing", ("slot_gap",), (1.5, 2.0)),
)


# ------------------------------------------------------------ measuring one race

def real_side(t: RaceTruth) -> dict:
    """Everything that is the same however the race is simulated."""
    pole = int(np.argmin(t.slot))
    others = np.delete(t.pace, pole)
    rp, rc = C.real_passes(t)
    has_gap = t.gap_ahead is not None
    rclose, rcars = C.real_close(t) if has_gap else (math.nan, math.nan)
    return {"pole": pole, "margin": float(others.min() - t.pace[pole]) if len(others) else math.nan, "grip": C.actual_grip(t),
            "rp": rp, "rc": rc, "rclose": rclose, "rcars": rcars, "has_gap": has_gap, "lap1": C.lap1_move(t),
            "pole_won": bool(t.finish[pole] == 1), "grid_finish": float(np.corrcoef(t.slot, t.finish)[0, 1])}


def measure(t: RaceTruth, res: E.SimResult, real: dict) -> dict:
    """One simulated race (S runs) against the real one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sg = np.nanmean(C.sim_grip(res, t.slot, t.n_laps), axis=0)
        d = sg - real["grip"]
        sp, sc = C.sim_passes(res, t.n_laps)
        row = {"pole_win": float((res.finish[:, real["pole"]] == 1).mean()), "pole_won": real["pole_won"], "margin": real["margin"],
               "early_gap": float(np.nanmean(d[:EARLY])) if np.isfinite(d[:EARLY]).any() else math.nan,
               "late_gap": float(np.nanmean(d[EARLY:])) if np.isfinite(d[EARLY:]).any() else math.nan,
               "abs_gap": float(np.nanmean(np.abs(d))) if np.isfinite(d).any() else math.nan,
               "grid_finish_gap": float(np.nanmean(spearman_masked(np.broadcast_to(t.slot[None].astype(float), res.finish.shape),
                                                                   res.finish.astype(float), np.ones(res.finish.shape, bool)))
                                        - real["grid_finish"]),
               "sim_passes": float(sp.mean()), "sim_car_laps": float(sc.mean()), "real_passes": real["rp"], "real_car_laps": real["rc"],
               "sim_lap1": float(C.sim_lap1_move(res, t.slot).mean()), "real_lap1": real["lap1"],
               "sim_close": math.nan, "sim_close_cars": math.nan, "real_close": math.nan, "real_close_cars": math.nan}
        if real["has_gap"] and res.gap is not None:
            sclose, scars = C.sim_close(res, t.n_laps)
            row.update(sim_close=float(sclose.mean()), sim_close_cars=float(scars.mean()),
                       real_close=real["rclose"], real_close_cars=real["rcars"])
    return row


def aggregate(rows: pd.DataFrame) -> dict:
    """Race rows -> one line of the metrics the tables compare."""
    ok = rows["real_close_cars"].notna()
    l1 = rows.dropna(subset=["real_lap1"])
    real_p = rows["real_passes"].sum() / rows["real_car_laps"].sum() if rows["real_car_laps"].sum() else math.nan
    real_c = rows.loc[ok, "real_close"].sum() / rows.loc[ok, "real_close_cars"].sum() if ok.any() and rows.loc[ok, "real_close_cars"].sum() else math.nan
    sim_p = rows["sim_passes"].sum() / rows["sim_car_laps"].sum() if rows["sim_car_laps"].sum() else math.nan
    sim_c = rows.loc[ok, "sim_close"].sum() / rows.loc[ok, "sim_close_cars"].sum() if ok.any() and rows.loc[ok, "sim_close_cars"].sum() else math.nan
    return {"races": len(rows), "pole_win": float(rows["pole_win"].mean()), "real_pole_win": float(rows["pole_won"].mean()),
            "early_gap": float(rows["early_gap"].mean()), "late_gap": float(rows["late_gap"].mean()), "abs_gap": float(rows["abs_gap"].mean()),
            "grid_finish_gap": float(rows["grid_finish_gap"].mean()),
            "passes_ratio": sim_p / real_p if real_p else math.nan, "close_ratio": sim_c / real_c if real_c else math.nan,
            "lap1_ratio": float(l1["sim_lap1"].mean() / l1["real_lap1"].mean()) if len(l1) and l1["real_lap1"].mean() else math.nan}


# ----------------------------------------------------------------- 1 factors

def factor_rows(truths: list[RaceTruth], par: Params, S: int = S_FACTOR, events: int = S_EVENTS, progress: bool = True):
    rows, cats = [], []
    for i, t in enumerate(truths):
        pool = Pool.before(truths, t)
        if len(pool) < MIN_POOL // 2:
            continue
        real = real_side(t)
        s = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, np.random.default_rng(int(t.race_id) + 7))
        p = par.for_circuit(t.circuit_id)
        for known in itertools.product([False, True], repeat=3):
            res = E.simulate(A.spec_for(t, s, known), p, S=S, seed=int(t.race_id), history=True)
            name = A.combo_name(known)
            rows.append({"race_id": t.race_id, "season": t.season, "split": split_of(t.season), "combo": name,
                         **{f: bool(k) for f, k in zip(A.FACTORS, known)}, "n_known": int(sum(known)), **measure(t, res, real)})
            for s_i in range(min(events, S)):
                laps = R.sim_laps(t, res, s_i)
                r = pd.DataFrame({"driver_id": t.drivers, "position": res.finish[s_i], "grid": t.slot})
                cats.append({"race_id": t.race_id, "season": t.season, "category": race_flow.classify_lead(laps, r), "level": name})
        if progress and (i + 1) % 20 == 0:
            print(f"  factors: {i + 1} of {len(truths)} races", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(cats)


def factor_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = list(A.FACTORS) + ["n_known", "combo"]
    for split, g in list(rows.groupby("split")) + [("all", rows)]:
        for k, d in g.groupby(keys):
            out.append({"split": split, **dict(zip(keys, k)), **aggregate(d)})
    return pd.DataFrame(out).sort_values(["split", "n_known", "combo"]).reset_index(drop=True)


def shapley_of(vals: dict[tuple, float]) -> dict[str, float]:
    """Average change in a metric when each factor becomes known, over every order of adding the three (vals keyed by known-flags)."""
    res = {}
    for j, f in enumerate(A.FACTORS):
        gains = []
        for order in itertools.permutations(range(3)):
            cur = [False, False, False]
            for k in order:
                if k == j:
                    a = tuple(cur)
                    cur[k] = True
                    gains.append(vals[tuple(cur)] - vals[a])
                    break
                cur[k] = True
        res[f] = float(np.mean(gains))
    return res


def factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """For each split and metric: the value with nothing known (L1) and everything known (L0), what each factor adds when
    it alone is given to L1 ("add one") and what is lost when it alone is taken from L0 ("take one away"), and its Shapley share."""
    rows = []
    for split, g in summary.groupby("split"):
        for m in FACTOR_METRICS:
            vals = {tuple(bool(r[f]) for f in A.FACTORS): float(r[m]) for _, r in g.iterrows()}
            if len(vals) < 8 or any(not math.isfinite(v) for v in vals.values()):
                continue
            sh = shapley_of(vals)
            for j, f in enumerate(A.FACTORS):
                only = tuple(k == j for k in range(3))
                allbut = tuple(k != j for k in range(3))
                rows.append({"split": split, "metric": m, "factor": f, "none_known": vals[(False,) * 3], "all_known": vals[(True,) * 3],
                             "add_one_to_L1": vals[only] - vals[(False,) * 3], "take_one_from_L0": vals[allbut] - vals[(True,) * 3],
                             "shapley": sh[f]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2 levers

def scaled(par: Params, fields: tuple[str, ...], mult: float) -> Params:
    """The same settings with these fields multiplied. Passing threshold: the per-circuit thresholds go with it."""
    kw = {}
    for f in fields:
        v = getattr(par, f)
        if v < 0:                    # sigma_pit_top < 0 means "same as sigma_pit"
            continue
        kw[f] = v * mult
    if "thr" in fields:
        kw["circuit_thr"] = {c: v * mult for c, v in par.circuit_thr.items()}
    return replace(par, **kw)


def lever_sweep(truths: list[RaceTruth], par: Params, S: int = S_LEVER, levers=LEVERS, progress: bool = True) -> pd.DataFrame:
    reals = [real_side(t) for t in truths]
    settings = [("fitted settings", "", 1.0, par)] + [(name, ",".join(fields), m, scaled(par, fields, m))
                                                      for name, fields, mults in levers for m in mults]
    out = []
    for k, (name, fields, m, p) in enumerate(settings):
        rows = []
        for t, real in zip(truths, reals):
            res = E.simulate(t.spec(), p.for_circuit(t.circuit_id), S=S, seed=int(t.race_id), history=True)
            rows.append(measure(t, res, real))
        out.append({"lever": name, "fields": fields, "multiplier": m, **aggregate(pd.DataFrame(rows))})
        if progress:
            print(f"  levers: {k + 1} of {len(settings)} settings", flush=True)
    return pd.DataFrame(out)


def lever_reach(sweep: pd.DataFrame) -> pd.DataFrame:
    """For each lever on its own: the multiplier at which the simulated pole-win rate would equal the real one (straight-line
    through the tried points, so only indicative), and the metrics there. `reached` is False when the real rate lies outside the range tried."""
    base = sweep[sweep["lever"] == "fitted settings"].iloc[0]
    target = float(base["real_pole_win"])
    rows = []
    for name, g in sweep[sweep["lever"] != "fitted settings"].groupby("lever", sort=False):
        pts = pd.concat([g, base.to_frame().T.assign(multiplier=1.0)]).astype({"multiplier": float}).sort_values("multiplier")
        x, y = pts["multiplier"].to_numpy(), pts["pole_win"].astype(float).to_numpy()
        slope = np.polyfit(x, y, 1)[0] if len(x) >= 2 else 0.0
        if abs(slope) < 1e-9:
            rows.append({"lever": name, "reached": False, "multiplier_needed": math.nan, "pole_win_range": f"{y.min():.2f}-{y.max():.2f}"})
            continue
        need = float(np.interp(target, y[np.argsort(y)], x[np.argsort(y)], left=math.nan, right=math.nan))
        row = {"lever": name, "reached": bool(math.isfinite(need)), "multiplier_needed": need,
               "pole_win_range": f"{y.min():.2f}-{y.max():.2f}", "slope_per_unit": float(slope)}
        if math.isfinite(need):
            for m in ("early_gap", "late_gap", "abs_gap", "passes_ratio", "close_ratio", "lap1_ratio", "grid_finish_gap"):
                row[m] = float(np.interp(need, x, pts[m].astype(float).to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3 margin

def pole_by_margin(rows: pd.DataFrame) -> pd.DataFrame:
    """Pole wins, real and simulated, by how much faster than every other car the pole sitter's pace was (everything known, and with the rest sampled)."""
    out = []
    for combo in ("everything known", "pace only"):
        d = rows[rows["combo"] == combo].dropna(subset=["margin"])
        b = pd.cut(d["margin"], bins=list(MARGIN_EDGES), labels=MARGIN_LABELS)
        for lab in MARGIN_LABELS:
            g = d[b == lab]
            if len(g):
                out.append({"combo": combo, "pace_margin": lab, "races": len(g), "real_pole_win": float(g["pole_won"].mean()),
                            "simulated_pole_win": float(g["pole_win"].mean())})
    return pd.DataFrame(out)


# ------------------------------------------------------------------- 4 stops

def stops_by_season(truths: list[RaceTruth], S: int = 20, seed: int = 53) -> pd.DataFrame:
    acc: dict = {}
    for i, t in enumerate(truths):
        pool = Pool.before(truths, t)
        if len(pool) < MIN_POOL // 2:
            continue
        _, ns_r, _ = R._first_stops(t.stops, t.slot, t.n_laps, t.retire >= BIG)
        smp = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, np.random.default_rng(seed + i), retirements=False)
        ns_s = np.concatenate([(smp["stops"][s] < BIG).sum(1) for s in range(S)])
        a = acc.setdefault(t.season, {"r": [], "s": [], "races": 0})
        a["r"].append(ns_r)
        a["s"].append(ns_s)
        a["races"] += 1
    rows = []
    for season, a in sorted(acc.items()):
        r, s = np.concatenate(a["r"]), np.concatenate(a["s"])
        rows.append({"season": season, "races": a["races"], "real_stops_per_car": float(r.mean()), "sampled_stops_per_car": float(s.mean()),
                     "real_never_stop": float((r == 0).mean()), "sampled_never_stop": float((s == 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- outputs

def chart(summary: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import AXIS, BLUE, ORANGE, TEXT2, _frame, _plt
    d = summary[summary["split"] == "all"].sort_values("n_known")
    if d.empty:
        return
    plt = _plt()
    fig, ax = _frame(plt, "Which unknown costs the pole sitter his wins?",
                     "Share of races the pole sitter wins in the simulation, by what the simulator is told. The dashed line is the real share.",
                     size=(9.5, 5.2))
    labels = [c.replace("pace + ", "").replace(" + ", "\n+ ") for c in d["combo"]]
    cols = [AXIS if n == 0 else (ORANGE if n == 3 else BLUE) for n in d["n_known"]]
    bars = ax.bar(range(len(d)), d["pole_win"], 0.6, color=cols)
    for b, v in zip(bars, d["pole_win"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=TEXT2)
    ax.axhline(float(d["real_pole_win"].iloc[0]), color=TEXT2, linestyle="--", linewidth=1.2)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, max(0.8, float(d["real_pole_win"].iloc[0]) + 0.1))
    ax.set_ylabel("Pole sitter wins")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def findings(summary: pd.DataFrame, effects: pd.DataFrame, sweep: pd.DataFrame, reach: pd.DataFrame, margin: pd.DataFrame,
             stops: pd.DataFrame, loss: pd.DataFrame) -> list[str]:
    L = []
    for split in ("all", "holdout"):
        d = summary[summary["split"] == split]
        if d.empty:
            continue
        L.append(f"FACTORS ({split}, {int(d['races'].iloc[0])} races; pole wins sim/real, early grip gap, late grip gap, passes sim/real, close sim/real, lap-1 moves sim/real):")
        for _, r in d.sort_values("n_known").iterrows():
            L.append(f"  {r['combo']}: pole {r['pole_win']:.0%}/{r['real_pole_win']:.0%}, grip gap early {r['early_gap']:+.3f} late {r['late_gap']:+.3f}, "
                     f"passes x{r['passes_ratio']:.2f}, close x{r['close_ratio']:.2f}, lap 1 x{r['lap1_ratio']:.2f}")
    for split in ("all", "holdout"):
        for m, label, fmt in (("pole_win", "POLE WINS", "{:+.1%}"), ("early_gap", "EARLY GRIP GAP", "{:+.3f}")):
            g = effects[(effects["split"] == split) & (effects["metric"] == m)]
            if g.empty:
                continue
            tot = g["shapley"].sum()
            L.append(f"{label} ({split}): nothing known {g['none_known'].iloc[0]:.3f}, everything known {g['all_known'].iloc[0]:.3f}; change when each is given "
                     f"(Shapley): " + "; ".join(f"{r['factor']} {fmt.format(r['shapley'])}" for _, r in g.iterrows()) + f" (together {fmt.format(tot)})")
            L.append("  given to L1 alone: " + "; ".join(f"{r['factor']} {fmt.format(r['add_one_to_L1'])}" for _, r in g.iterrows())
                     + ".  Taken from L0 alone: " + "; ".join(f"{r['factor']} {fmt.format(r['take_one_from_L0'])}" for _, r in g.iterrows()))
    if len(sweep):
        b = sweep[sweep["lever"] == "fitted settings"].iloc[0]
        L.append(f"LEVERS at L0 ({int(b['races'])} races): fitted settings give pole wins {b['pole_win']:.0%} against {b['real_pole_win']:.0%} real, "
                 f"passes x{b['passes_ratio']:.2f}, close x{b['close_ratio']:.2f}, grip gap early {b['early_gap']:+.3f} late {b['late_gap']:+.3f}, lap 1 x{b['lap1_ratio']:.2f}.")
        for name, g in sweep[sweep["lever"] != "fitted settings"].groupby("lever", sort=False):
            L.append(f"  {name}: " + "; ".join(f"x{r['multiplier']:g} pole {r['pole_win']:.0%}, passes x{r['passes_ratio']:.2f}, close x{r['close_ratio']:.2f}, |grip gap| {r['abs_gap']:.3f}"
                                               for _, r in g.sort_values("multiplier").iterrows()))
    if len(reach):
        got = reach[reach["reached"]]
        L.append("TO CLOSE THE POLE GAP BY ONE LEVER ALONE (straight-line, indicative): " + (
            "; ".join(f"{r['lever']} to x{r['multiplier_needed']:.2f} (passes x{r['passes_ratio']:.2f}, close x{r['close_ratio']:.2f}, |grip gap| {r['abs_gap']:.3f})"
                      for _, r in got.iterrows()) if len(got) else "none of the levers reaches the real rate inside the range tried") + ".")
        miss = reach[~reach["reached"]]
        if len(miss):
            L.append("  Not reached inside the range tried: " + "; ".join(miss["lever"]) + ".")
    for combo in ("everything known", "pace only"):
        g = margin[margin["combo"] == combo]
        if len(g):
            L.append(f"POLE WINS BY THE POLE SITTER'S PACE MARGIN ({combo}; races, real, simulated): " + "; ".join(
                f"{r['pace_margin']} ({int(r['races'])}) {r['real_pole_win']:.0%}/{r['simulated_pole_win']:.0%}" for _, r in g.iterrows()))
    if len(stops):
        L.append("STOPS PER CAR BY SEASON (real / sampled): " + "; ".join(
            f"{int(r['season'])} {r['real_stops_per_car']:.2f}/{r['sampled_stops_per_car']:.2f}" for _, r in stops.iterrows()))
    if len(loss):
        a = loss[(loss["period"] == "all") & (loss["category"] == race_flow.POLE_WON)]
        L.append("POLE KEPT THE LEAD TO THE END, by combination (real / simulated): " + "; ".join(
            f"{r['level']} {r['real_share_of_races']:.0%}/{r['simulated_share_of_races']:.0%}" for _, r in a.iterrows()))
    return L


def run(con=None, out_dir: Path | None = None, cal_dir: Path | None = None, S: int = S_FACTOR, S_lever: int = S_LEVER,
        events: int = S_EVENTS, progress: bool = True) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sim_diagnose"
    out.mkdir(parents=True, exist_ok=True)
    truths, par = C.load_truths(cal_dir)
    rows, cats = factor_rows(truths, par, S=S, events=events if con is not None else 0, progress=progress)
    if rows.empty:
        lines = ["Too few races for the diagnostic (need earlier races to sample from)."]
        (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    summary = factor_summary(rows)
    effects = factor_effects(summary)
    sweep = lever_sweep(truths, par, S=S_lever, progress=progress)
    reach = lever_reach(sweep)
    margin = pole_by_margin(rows)
    stops = stops_by_season(truths)
    loss = pd.DataFrame()
    if con is not None and len(cats):
        _, real_takeovers = R.actual_targets(con, {t.race_id for t in truths})
        loss = R.takeover_compare(cats, real_takeovers)
    for name, t in [("factor_races", rows), ("factor_summary", summary), ("factor_effects", effects), ("lever_sweep", sweep),
                    ("lever_reach", reach), ("pole_by_pace_margin", margin), ("stops_by_season", stops), ("lead_loss_by_combo", loss)]:
        t.to_csv(out / f"{name}.csv", index=False)
    chart(summary, out / "pole_by_unknown.png")
    lines = findings(summary, effects, sweep, reach, margin, stops, loss)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    path = config.db_path()
    con = duckdb.connect(str(path), read_only=True) if path.exists() else None
    try:
        lines = run(con)
    finally:
        if con is not None:
            con.close()
    print("\n".join(lines))
    print(f"\nTables and chart written to {config.outputs_dir() / 'sim_diagnose'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
