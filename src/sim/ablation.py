"""Stage 3, part 4: what does not knowing each thing cost? An ablation ladder.

Run:  python -m src.sim.calibrate   (once)   then   python -m src.sim.ablation
Writes outputs/sim_ablation/.

Every dry race is simulated with each car's true pace and the real grid, and for every combination of three things the
simulator either knows what happened or samples it from earlier races:
  strategies      each car's stops and compounds
  safety cars     the neutralised laps
  retirements     who stops and when
Eight combinations: nothing known (the L1 level of replay.py) up to everything known (L0). Scoring each against the real
result shows how much of the error each unknown is responsible for. Each factor's share is a Shapley value: its average gain
in accuracy when added, over every order in which the three could be added.

The pace itself is treated as known throughout (it is measured from the same race), so this ladder says how far a perfect pace
forecast could get with the rest unknown. It is the second half of the question "what wins a Grand Prix": once the cars' pace is
fixed, how much is left to strategy, safety cars and reliability.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import calibrate as C
from . import engine as E
from . import replay as R
from .calibrate import RaceTruth, split_of
from .engine import BIG, Params
from .sample import MIN_POOL, Pool, sample_inputs

FACTORS = ("strategies", "safety cars", "retirements")
S_ABLATE = 100
METRICS = {"mean_rank_error": "lower", "spearman_expected": "higher", "brier_top3": "lower", "brier_top10": "lower",
           "winner_logscore": "lower", "spearman_finishers": "higher"}


def combo_name(known: tuple[bool, bool, bool]) -> str:
    got = [f for f, k in zip(FACTORS, known) if k]
    if not got:
        return "pace only"
    if len(got) == 3:
        return "everything known"
    return "pace + " + " + ".join(got)


def spec_for(t: RaceTruth, s: dict, known: tuple[bool, bool, bool]) -> E.RaceSpec:
    strat, sc, ret = known
    comp, stops, age = (t.comp[None], t.stops[None], t.start_age[None]) if strat else (s["comp"], s["stops"], s["start_age"])
    retire = t.retire[None] if ret else s["retire"]
    neutral = t.neutral[None] if sc else s["neutral"]
    return E.RaceSpec(n_laps=t.n_laps, slot=t.slot, pace=t.pace, comp=comp, stops=stops, start_age=age, retire=retire,
                      neutral=neutral, off=t.off, deg=t.deg, fuel=t.fuel, pit_loss=t.pit_loss, cover=not strat, warm=t.warm_s)


def spearman_finishers(finish: np.ndarray, t: RaceTruth) -> float:
    """Rank correlation of the expected order and the real order among the cars that really finished."""
    m = t.retire >= BIG
    if m.sum() < 4:
        return math.nan
    exp = finish.mean(0)[m]
    a = np.argsort(np.argsort(exp, kind="stable"), kind="stable")
    b = np.argsort(np.argsort(t.finish[m], kind="stable"), kind="stable")
    return float(np.corrcoef(a, b)[0, 1])


def ablate(truths: list[RaceTruth], par: Params, S: int = S_ABLATE, progress: bool = True) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(truths):
        pool = Pool.before(truths, t)
        if len(pool) < MIN_POOL // 2:
            continue
        s = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, np.random.default_rng(int(t.race_id) + 7))
        base = R.grid_baseline(pool, t.slot, t.finish)
        p = par.for_circuit(t.circuit_id)
        for known in itertools.product([False, True], repeat=3):
            res = E.simulate(spec_for(t, s, known), p, S=S, seed=int(t.race_id), history=False)
            m = R.finish_metrics(res.finish, t.finish)
            rows.append({"race_id": t.race_id, "season": t.season, "split": split_of(t.season), "combo": combo_name(known),
                         **{f: bool(k) for f, k in zip(FACTORS, known)}, "n_known": int(sum(known)), **m, **base,
                         "spearman_finishers": spearman_finishers(res.finish, t)})
        if progress and (i + 1) % 20 == 0:
            print(f"  ablated {i + 1} of {len(truths)} races", flush=True)
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    keys = list(FACTORS) + ["n_known", "combo"]
    cols = list(METRICS) + ["base_mean_rank_error", "base_spearman", "base_brier_top3", "base_brier_top10", "base_winner_logscore"]
    out = []
    for split, g in list(d.groupby("split")) + [("all", d)]:
        t = g.groupby(keys)[cols].mean().reset_index()
        t.insert(0, "split", split)
        t.insert(1, "races", g["race_id"].nunique())
        out.append(t)
    return pd.concat(out, ignore_index=True).sort_values(["split", "n_known", "combo"]).reset_index(drop=True)


def shapley(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Each factor's average improvement in `metric` when it becomes known, over all orders of adding the three."""
    rows = []
    for split, g in summary.groupby("split"):
        val = {tuple(bool(r[f]) for f in FACTORS): float(r[metric]) for _, r in g.iterrows()}
        sign = 1.0 if METRICS[metric] == "higher" else -1.0
        for j, f in enumerate(FACTORS):
            gains = []
            for order in itertools.permutations(range(3)):
                before = tuple(False for _ in FACTORS)
                cur = list(before)
                for k in order:
                    if k == j:
                        a = tuple(cur)
                        cur[k] = True
                        gains.append(sign * (val[tuple(cur)] - val[a]))
                        break
                    cur[k] = True
            rows.append({"split": split, "metric": metric, "factor": f, "gain": float(np.mean(gains))})
    return pd.DataFrame(rows)


def chart(summary: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import AXIS, BLUE, ORANGE, TEXT2, _frame, _plt
    d = summary[summary["split"] == "all"]
    if d.empty:
        return
    plt = _plt()
    ladder = [("Grid slot only", float(d["base_mean_rank_error"].iloc[0]), AXIS),
              ("True pace,\nrest sampled", None, BLUE),
              ("+ who retires", None, BLUE), ("+ safety cars", None, BLUE), ("+ strategies\n(everything known)", None, ORANGE)]
    keys = [(False, False, False), (False, False, True), (False, True, True), (True, True, True)]
    vals = [ladder[0][1]]
    for k in keys:
        r = d[(d["strategies"] == k[0]) & (d["safety cars"] == k[1]) & (d["retirements"] == k[2])]
        vals.append(float(r["mean_rank_error"].iloc[0]) if len(r) else math.nan)
    fig, ax = _frame(plt, "What each unknown costs, once the pace is known",
                     "Average places wrong per car (expected finishing order against the real one), all dry races. "
                     "Each bar adds one more thing the simulator is told.", size=(9.5, 5.0))
    bars = ax.bar(range(len(vals)), vals, 0.6, color=[x[2] for x in ladder])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=9, color=TEXT2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([x[0] for x in ladder], fontsize=9)
    ax.set_ylabel("Mean places wrong")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def findings(summary: pd.DataFrame, shap: pd.DataFrame) -> list[str]:
    L = []
    for split in ("all", "holdout"):
        d = summary[summary["split"] == split]
        if d.empty:
            continue
        L.append(f"{split.upper()} ({int(d['races'].iloc[0])} races), mean places wrong per car / podium Brier / correlation of expected order:")
        L.append(f"  grid slot only: {d['base_mean_rank_error'].iloc[0]:.2f} / {d['base_brier_top3'].iloc[0]:.3f} / {d['base_spearman'].iloc[0]:.2f}")
        for _, r in d.sort_values("n_known").iterrows():
            L.append(f"  {r['combo']}: {r['mean_rank_error']:.2f} / {r['brier_top3']:.3f} / {r['spearman_expected']:.2f}"
                     f" (among cars that really finished {r['spearman_finishers']:.2f})")
    for split in ("all", "holdout"):
        g = shap[(shap["split"] == split) & (shap["metric"] == "mean_rank_error")]
        if g.empty:
            continue
        tot = g["gain"].sum()
        L.append(f"SHARE OF THE PLACES LOST WITHOUT KNOWING EACH THING ({split}, Shapley over adding order; total {tot:.2f} places): " + "; ".join(
            f"{r['factor']} {r['gain']:.2f} ({r['gain'] / tot:.0%})" for _, r in g.sort_values("gain", ascending=False).iterrows()))
    L.append("Pace is treated as known throughout, so these are the costs of not knowing the rest even with a perfect pace forecast.")
    return L


def run(out_dir: Path | None = None, cal_dir: Path | None = None, S: int = S_ABLATE, progress: bool = True) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sim_ablation"
    out.mkdir(parents=True, exist_ok=True)
    truths, par = C.load_truths(cal_dir)
    races = ablate(truths, par, S=S, progress=progress)
    if races.empty:
        lines = ["Too few races for the ablation (need earlier races to sample from)."]
        (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    summary = summarise(races)
    shap = pd.concat([shapley(summary, m) for m in METRICS], ignore_index=True)
    races.to_csv(out / "ablation_races.csv", index=False)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    shap.to_csv(out / "ablation_shapley.csv", index=False)
    chart(summary, out / "ablation_ladder.png")
    lines = findings(summary, shap)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    lines = run()
    print("\n".join(lines))
    print(f"\nTables and chart written to {config.outputs_dir() / 'sim_ablation'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
