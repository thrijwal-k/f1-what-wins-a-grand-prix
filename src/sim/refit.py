"""Stage 3, part 7 (v2.8): put pace drift, lap noise, the passing threshold and the tyre warm-up on the evidence that exposed them.

Run:  python -m src.sim.calibrate   (once)   then   python -m src.sim.refit   then replay, ablation, forecast, diagnose.
Needs only outputs/sim_calibration/truth.pkl and params.json (no database). Writes outputs/sim_refit/ and replaces
outputs/sim_calibration/params.json with the re-fitted settings (the old file is kept as params_before_refit.json, and a second run
starts from it, so running refit twice gives the same answer).

Why. The v2.7 diagnosis (pole wins 46% against 55% real with everything known) was taken apart on the real races:
  * the simulated pole sitter loses the lead about as often as the real one, but regains it far less often;
  * switching the tyre warm-up off, deciding passes on persistent pace, or giving the simulator the real strategies, safety cars and
    retirements did not move pole wins;
  * removing the independent per-car pace drift and re-tuning the passing threshold to keep the pass rate did (46% -> 54%);
  * the gap between two cars changes over ten clean laps by a robust sd of 3.4 s in real races, 6.4 s in the simulator, 4.5 s with
    no drift at all. The old drift check scaled drift down without re-fitting the threshold, so it favoured full drift;
  * after a stop, a car on new tyres closes on one on old tyres by about 0.8 s a lap in real races; the warm-up term cancelled that.

What this does.
  1 WARM-UP    The warm-up amplitude is scaled by 1, 0.5 or 0 and the one that reproduces the real closing rate after a first stop is kept.
  2 GRID       Drift (x1 .. x0) by lap noise (x1 .. x0.65). For each pair the passing threshold (all circuits together) is searched until
               the pass rate on the fitting races equals the real one. The pair is scored on what the passing fit could not see:
               how far the gap between two cars wanders over ten laps, the share of cars running within a second of the one ahead,
               and the grip of the grid on the running order. Pole wins are reported for every pair but are not fitted.
  3 CHECKS     Old against new against real: the ten-lap gap wander, the swap rate by pace advantage (does the faster car pass?),
               and how often the pole sitter loses and regains the lead and why.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import calibrate as C
from . import engine as E
from .engine import BIG, Params

W = 10                           # laps over which the gap between two cars is followed
MAX_RANK = 8                     # pairs from P1-P2 down to P7-P8
DRIFT_GRID = (1.0, 0.5, 0.25, 0.0)
NOISE_GRID = (1.0, 0.8, 0.65)
WARM_GRID = (1.0, 0.5, 0.0)
WANDER_TOL = 0.25                # a 25% gap in the ten-lap wander is one unit of loss
CLOSE_TOL = 0.10                 # a 10% gap in the close-running share
GRIP_TOL = 0.02                  # a 0.02 gap in the grid's grip on the order
PASS_TOL = 0.15
CLOSING_TOL = 0.25               # s a lap
THR_RANGE = (0.15, 1.6)          # search range of the common factor on every circuit's passing threshold
PACE_BINS = (-9.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 9.0)
SPLITS = ("train", "holdout", "earlier era")


# ------------------------------------------------------------------ gaps behind the leader

def behind_leader(pos: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """Seconds behind the leader at the end of each lap, from the order and the gap to the car directly ahead. pos, gap (..., L, n);
    NaN where a car is not running or a gap in the chain between it and the leader is unknown."""
    pos = np.asarray(pos, float)
    gap = np.asarray(gap, float)
    run = np.isfinite(pos) & (pos > 0)
    order = np.argsort(np.where(run, pos, 1e9), axis=-1, kind="stable")
    gs = np.take_along_axis(gap, order, -1).copy()
    rs = np.take_along_axis(run, order, -1)
    gs[..., 0] = 0.0
    gs = np.where(rs, gs, np.nan)
    cs = np.cumsum(gs, axis=-1)
    out = np.full_like(cs, np.nan)
    np.put_along_axis(out, order, cs, -1)
    return np.where(run, out, np.nan)


def real_bl(t: C.RaceTruth) -> np.ndarray:
    return behind_leader(t.pos_actual, t.gap_ahead)                                  # (L, n)


def sim_bl(res: E.SimResult) -> np.ndarray:
    pos = np.transpose(res.pos, (0, 2, 1)).astype(float)                             # (S, L, n)
    gap = np.transpose(res.gap, (0, 2, 1)).astype(float)
    return behind_leader(np.where(pos > 0, pos, np.nan), gap)


def _stop_free(t: C.RaceTruth, c: int, lo: int, hi: int) -> bool:
    st = t.stops[c]
    return not ((st >= lo) & (st <= hi) & (st < BIG)).any() and t.retire[c] > hi


def wander_windows(t: C.RaceTruth, w: int = W, maxrank: int = MAX_RANK) -> dict | None:
    """Pairs of cars (a directly ahead of b at the end of lap k-1) that neither stop nor retire from lap k-1 to k+w+1, with no safety car
    or VSC in that time: the gap between them changes only through pace, noise and traffic. Returns arrays a, b, k and the real change in
    the gap (b's time behind the leader minus a's) from lap k-1 to lap k+w-1, or None without gaps."""
    if t.gap_ahead is None:
        return None
    L = t.n_laps
    rb = real_bl(t)
    A, B, K = [], [], []
    for k in range(3, L - w):
        if (t.neutral[k - 1:k + w + 2] != 0).any():
            continue
        row = t.pos_actual[k - 1]
        for r in range(1, maxrank):
            ia, ib = np.where(row == r)[0], np.where(row == r + 1)[0]
            if len(ia) != 1 or len(ib) != 1:
                continue
            a, b = int(ia[0]), int(ib[0])
            if _stop_free(t, a, k - 1, k + w + 1) and _stop_free(t, b, k - 1, k + w + 1):
                A.append(a); B.append(b); K.append(k)
    if not A:
        return None
    a, b, k = np.array(A), np.array(B), np.array(K)
    real = (rb[k + w - 1, b] - rb[k + w - 1, a]) - (rb[k - 1, b] - rb[k - 1, a])
    ok = np.isfinite(real)
    if not ok.any():
        return None
    return {"a": a[ok], "b": b[ok], "k": k[ok], "real": real[ok]}


def sim_wander(sb: np.ndarray, win: dict, w: int = W) -> np.ndarray:
    """(S, windows) change in the same gaps in the simulations; NaN where a chain is broken."""
    a, b, k = win["a"], win["b"], win["k"]
    return (sb[:, k + w - 1, b] - sb[:, k + w - 1, a]) - (sb[:, k - 1, b] - sb[:, k - 1, a])


def robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else math.nan


# ------------------------------------------------------------------ the closing rate after a first stop

def closing_events(t: C.RaceTruth, min_gap_laps: int = 5, max_gap_laps: int = 10, settle: int = 3, maxrank: int = MAX_RANK) -> dict | None:
    """Pairs (a directly ahead of b just before the earlier of their next stops) whose second stop comes between `min_gap_laps` and
    `max_gap_laps` laps after the first, on green laps, with no other stop by either car until `settle` laps after the second. In the laps between the stops one car is on new tyres and the other on old. Returns a, b, first-stop lap and
    sign (+1 when a stops first, -1 when b does) so that sign * (change in b's time behind a) is what the first stopper gains."""
    if t.gap_ahead is None:
        return None
    L = t.n_laps
    A, B, F, SG = [], [], [], []
    for p in range(3, L - settle - min_gap_laps):
        row = t.pos_actual[p - 2]
        for r in range(1, maxrank):
            ia, ib = np.where(row == r)[0], np.where(row == r + 1)[0]
            if len(ia) != 1 or len(ib) != 1:
                continue
            a, b = int(ia[0]), int(ib[0])
            sa = t.stops[a][(t.stops[a] >= p) & (t.stops[a] < BIG)]
            sb_ = t.stops[b][(t.stops[b] >= p) & (t.stops[b] < BIG)]
            if not len(sa) or not len(sb_):
                continue
            pa, pb = int(sa.min()), int(sb_.min())
            first, last = min(pa, pb), max(pa, pb)
            if first != p or not (min_gap_laps <= last - first <= max_gap_laps) or last + settle > L:
                continue
            if (t.neutral[max(first - 2, 1):last + settle + 1] != 0).any():
                continue
            if t.retire[a] <= last + settle or t.retire[b] <= last + settle:
                continue
            other = False
            for c, own in ((a, pa), (b, pb)):
                st = t.stops[c]
                if ((st > first) & (st <= last + settle) & (st != own) & (st < BIG)).any():
                    other = True
            if other:
                continue
            A.append(a); B.append(b); F.append(first); SG.append(1 if pa < pb else -1)
    if not A:
        return None
    return {"a": np.array(A), "b": np.array(B), "f": np.array(F), "sign": np.array(SG)}


def closing_gain(bl: np.ndarray, ev: dict, offsets=(2, 3, 4)) -> np.ndarray:
    """Mean over the offsets of the first stopper's gain per lap on the other car; bl (..., L, n); returns (..., events)."""
    a, b, f, sg = ev["a"], ev["b"], ev["f"], ev["sign"]
    g = lambda lap: bl[..., lap - 1, b] - bl[..., lap - 1, a]
    steps = [g(f + o) - g(f + o - 1) for o in offsets]
    return sg * np.nanmean(np.stack(steps), axis=0)


# ------------------------------------------------------------------ one pass through the races

def _acc() -> dict:
    return {"p": 0.0, "c": 0.0, "cl": 0.0, "cars": 0.0, "pole": [], "wander": [], "grip": [], "closing_first_a": [], "closing_first_b": []}


def evaluate(truths: list[C.RaceTruth], par: Params, S: int, seed: int, wins: dict, evs: dict) -> dict:
    """One simulation per race, everything measured on it. Returns {split: accumulator} plus 'all'."""
    acc = {s: _acc() for s in SPLITS + ("all",)}
    for t in truths:
        res = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + int(t.race_id))
        parts = [acc[C.split_of(t.season)], acc["all"]]
        pole = int(np.argmin(t.slot))
        win = (res.finish[:, pole] == 1).mean()
        grip = np.nanmean(C.sim_grip(res, t.slot, t.n_laps), axis=0) - C.actual_grip(t)
        grip = float(np.nanmean(np.abs(grip))) if np.isfinite(grip).any() else math.nan
        sp = sc = cl = cars = 0.0
        sb = None
        if t.gap_ahead is not None:
            p, c = C.sim_passes(res, t.n_laps)
            sp, sc = float(np.mean(p)), float(np.mean(c))
            q, k = C.sim_close(res, t.n_laps)
            cl, cars = float(np.mean(q)), float(np.mean(k))
            sb = sim_bl(res)
        for a in parts:
            a["p"] += sp; a["c"] += sc; a["cl"] += cl; a["cars"] += cars
            a["pole"].append(win)
            a["grip"].append(grip)
        if sb is not None and wins.get(t.race_id) is not None:
            d = sim_wander(sb, wins[t.race_id]).ravel()
            for a in parts:
                a["wander"].append(d)
        ev = evs.get(t.race_id)
        if sb is not None and ev is not None:
            g = closing_gain(sb, ev)                                         # (S, events)
            for a in parts:
                a["closing_first_a"].append(g[:, ev["sign"] > 0].ravel())
                a["closing_first_b"].append(g[:, ev["sign"] < 0].ravel())
    return acc


def summarise(a: dict) -> dict:
    w = np.concatenate(a["wander"]) if a["wander"] else np.array([])
    w = w[np.isfinite(w)]
    ca = np.concatenate(a["closing_first_a"]) if a["closing_first_a"] else np.array([])
    cb = np.concatenate(a["closing_first_b"]) if a["closing_first_b"] else np.array([])
    return {"races": len(a["pole"]), "pass_rate": a["p"] / a["c"] if a["c"] else math.nan, "close_share": a["cl"] / a["cars"] if a["cars"] else math.nan,
            "wander_robust_sd": robust_sd(w), "wander_sd": float(np.std(w)) if len(w) else math.nan, "wander_mean": float(np.mean(w)) if len(w) else math.nan,
            "grip_gap": float(np.nanmean(a["grip"])) if len(a["grip"]) and np.isfinite(a["grip"]).any() else math.nan,
            "pole_wins": float(np.mean(a["pole"])) if a["pole"] else math.nan,
            "closing_a_first": float(np.nanmean(ca)) if np.isfinite(ca).any() else math.nan,
            "closing_b_first": float(np.nanmean(cb)) if np.isfinite(cb).any() else math.nan}


def real_reference(truths: list[C.RaceTruth], wins: dict, evs: dict) -> dict:
    """The same numbers from the real races, per split."""
    out = {}
    for s in SPLITS + ("all",):
        ts = [t for t in truths if s == "all" or C.split_of(t.season) == s]
        p = c = cl = cars = 0.0
        wd, ca, cb = [], [], []
        for t in ts:
            if t.gap_ahead is None:
                continue
            a, b = C.real_passes(t)
            p += a; c += b
            a, b = C.real_close(t)
            cl += a; cars += b
            if wins.get(t.race_id) is not None:
                wd.append(wins[t.race_id]["real"])
            ev = evs.get(t.race_id)
            if ev is not None:
                g = closing_gain(real_bl(t), ev)
                ca.append(g[ev["sign"] > 0]); cb.append(g[ev["sign"] < 0])
        w = np.concatenate(wd) if wd else np.array([])
        ca = np.concatenate(ca) if ca else np.array([])
        cb = np.concatenate(cb) if cb else np.array([])
        out[s] = {"races": len(ts), "pass_rate": p / c if c else math.nan, "close_share": cl / cars if cars else math.nan,
                  "wander_robust_sd": robust_sd(w), "wander_sd": float(np.std(w)) if len(w) else math.nan,
                  "wander_mean": float(np.mean(w)) if len(w) else math.nan, "grip_gap": 0.0,
                  "pole_wins": float(np.mean([t.finish[int(np.argmin(t.slot))] == 1 for t in ts])) if ts else math.nan,
                  "closing_a_first": float(np.nanmean(ca)) if np.isfinite(ca).any() else math.nan,
                  "closing_b_first": float(np.nanmean(cb)) if np.isfinite(cb).any() else math.nan}
    return out


# ------------------------------------------------------------------ settings

def scaled(par: Params, drift: float = 1.0, noise: float = 1.0, thr: float = 1.0, warm: float | None = None) -> Params:
    """The settings with drift, lap noise and every circuit's passing threshold multiplied (the circuit differences are kept)."""
    q = replace(par, drift_sd=par.drift_sd * drift, sigma_lap=par.sigma_lap * noise, thr=par.thr * thr,
                circuit_thr={k: float(v) * thr for k, v in par.circuit_thr.items()})
    return q if warm is None else replace(q, warm_scale=float(warm))


def pass_rate_of(truths: list[C.RaceTruth], par: Params, S: int, seed: int) -> float:
    p = c = 0.0
    for t in truths:
        res = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + int(t.race_id))
        a, b = C.sim_passes(res, t.n_laps)
        p += float(np.mean(a)); c += float(np.mean(b))
    return p / c if c else math.nan


def match_threshold(truths: list[C.RaceTruth], par: Params, target: float, S: int, seed: int, iters: int = 7) -> float:
    """The factor on every passing threshold at which the simulated pass rate on these races equals the target (bisection on the log:
    a higher threshold means fewer passes)."""
    lo, hi = math.log(THR_RANGE[0]), math.log(THR_RANGE[1])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pass_rate_of(truths, scaled(par, thr=math.exp(mid)), S, seed) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def loss_of(sim: dict, real: dict) -> dict:
    """Loss on one split. The pass rate is matched by construction, so it carries little weight; the three terms that decide are the
    ten-lap gap wander, the close-running share and the grid's grip."""
    wl = (math.log(sim["wander_robust_sd"] / real["wander_robust_sd"]) / WANDER_TOL) ** 2 if sim["wander_robust_sd"] > 0 and real["wander_robust_sd"] > 0 else math.nan
    cl = ((sim["close_share"] / real["close_share"] - 1.0) / CLOSE_TOL) ** 2 if real["close_share"] else math.nan
    gl = (sim["grip_gap"] / GRIP_TOL) ** 2 if np.isfinite(sim["grip_gap"]) else 0.0
    pl = ((sim["pass_rate"] / real["pass_rate"] - 1.0) / PASS_TOL) ** 2 if real["pass_rate"] else math.nan
    return {"wander_term": wl, "close_term": cl, "grip_term": gl, "pass_term": pl, "loss": float(np.nansum([wl, cl, gl, pl]))}


def closing_loss(sim: dict, real: dict) -> float:
    return float(np.nansum([((sim[k] - real[k]) / CLOSING_TOL) ** 2 for k in ("closing_a_first", "closing_b_first")]))


# ------------------------------------------------------------------ the fit

def fit(truths: list[C.RaceTruth], base: Params, S: int = 16, S_eval: int = 20, seed: int = 0, drift_grid=DRIFT_GRID, noise_grid=NOISE_GRID,
        warm_grid=WARM_GRID, progress: bool = True) -> dict:
    wins = {t.race_id: wander_windows(t) for t in truths}
    evs = {t.race_id: closing_events(t) for t in truths}
    real = real_reference(truths, wins, evs)
    train = [t for t in truths if C.split_of(t.season) == "train" and t.gap_ahead is not None] or [t for t in truths if t.gap_ahead is not None]
    target = real["train"]["pass_rate"] if np.isfinite(real["train"]["pass_rate"]) else real["all"]["pass_rate"]

    def say(msg):
        if progress:
            print(msg, flush=True)

    # 1 warm-up
    warm_rows = []
    for w in warm_grid:
        m = summarise(evaluate(truths, scaled(base, warm=w), S_eval, seed, wins, evs)["all"])
        warm_rows.append({"warm_scale": w, "closing_a_first": m["closing_a_first"], "closing_b_first": m["closing_b_first"],
                          "real_a_first": real["all"]["closing_a_first"], "real_b_first": real["all"]["closing_b_first"],
                          "loss": closing_loss(m, real["all"]), "pole_wins": m["pole_wins"]})
        say(f"  warm-up x{w}: closing {m['closing_a_first']:+.2f} / {m['closing_b_first']:+.2f} s a lap (real {real['all']['closing_a_first']:+.2f} / {real['all']['closing_b_first']:+.2f})")
    warm_tab = pd.DataFrame(warm_rows)
    warm = float(warm_tab.loc[warm_tab["loss"].idxmin(), "warm_scale"]) if warm_tab["loss"].notna().any() else 1.0
    base = replace(base, warm_scale=warm)

    # 2 grid
    rows = []
    cands = [("v2.7 as fitted", replace(base, warm_scale=1.0), 1.0, 1.0, 1.0, 1.0)]
    for d in drift_grid:
        for g in noise_grid:
            cands.append(("matched passing", None, d, g, None, warm))
    for label, par, d, g, thr, wm in cands:
        if par is None:
            thr = match_threshold(train, scaled(base, drift=d, noise=g), target, S, seed)
            par = scaled(base, drift=d, noise=g, thr=thr)
        acc = evaluate(truths, par, S_eval, seed + 1, wins, evs)
        for s in SPLITS + ("all",):
            m = summarise(acc[s])
            row = {"setting": label, "drift_factor": d, "noise_factor": g, "threshold_factor": thr, "warm_scale": par.warm_scale, "split": s, **m,
                   **{"real_" + k: v for k, v in real[s].items() if k != "races"}}
            if s == "train":
                row.update(loss_of(m, real["train"]))
            rows.append(row)
        say(f"  drift x{d} noise x{g} threshold x{thr:.2f}: pole wins {summarise(acc['all'])['pole_wins']:.1%}")
    table = pd.DataFrame(rows)
    tr = table[(table["split"] == "train") & (table["setting"] == "matched passing")]
    best = tr.loc[tr["loss"].idxmin()]
    chosen = scaled(base, drift=float(best["drift_factor"]), noise=float(best["noise_factor"]), thr=float(best["threshold_factor"]))
    table["chosen"] = (table["setting"] == "matched passing") & (table["drift_factor"] == best["drift_factor"]) & (table["noise_factor"] == best["noise_factor"])
    edge = []
    if best["drift_factor"] in (min(drift_grid), max(drift_grid)) and len(drift_grid) > 1:
        edge.append("drift")
    if best["noise_factor"] in (min(noise_grid), max(noise_grid)) and len(noise_grid) > 1:
        edge.append("lap noise")
    return {"params": chosen, "table": table, "warm_table": warm_tab, "real": real, "wins": wins, "evs": evs, "warm_scale": warm,
            "best": best, "edge": edge}


# ------------------------------------------------------------------ checks

def wander_check(truths, wins, evs, params: dict[str, Params], S: int, seed: int) -> pd.DataFrame:
    real = real_reference(truths, wins, evs)
    rows = []
    for s in SPLITS + ("all",):
        r = real[s]
        rows.append({"split": s, "setting": "real", "robust_sd_s": r["wander_robust_sd"], "sd_s": r["wander_sd"], "mean_s": r["wander_mean"]})
    for name, par in params.items():
        acc = evaluate(truths, par, S, seed, wins, evs)
        for s in SPLITS + ("all",):
            m = summarise(acc[s])
            rows.append({"split": s, "setting": name, "robust_sd_s": m["wander_robust_sd"], "sd_s": m["wander_sd"], "mean_s": m["wander_mean"]})
    return pd.DataFrame(rows)


def _pair_rows(pos, gap, pit, nv, pace, L) -> list[tuple]:
    out = []
    for k in range(3, L):
        lap = k + 1
        if nv[lap] != 0 or nv[lap - 1] != 0 or nv[lap - 2] != 0:
            continue
        prev, cur = pos[k - 1], pos[k]
        run = np.isfinite(prev) & (prev > 0) & np.isfinite(cur) & (cur > 0)
        idx = np.where(run)[0]
        order = idx[np.argsort(prev[idx])]
        for j in range(len(order) - 1):
            a, b = order[j], order[j + 1]
            if prev[b] - prev[a] != 1:
                continue
            if pit[k - 1, a] or pit[k - 1, b] or pit[k, a] or pit[k, b] or (k + 1 < L and (pit[k + 1, a] or pit[k + 1, b])):
                continue
            out.append((pace[a] - pace[b], gap[k - 1, b], float(cur[b] < cur[a])))
    return out


def _real_pit(t: C.RaceTruth) -> np.ndarray:
    pit = np.zeros((t.n_laps, t.n), bool)
    for i in range(t.n):
        for s in t.stops[i]:
            if s < BIG and 1 <= s <= t.n_laps:
                pit[int(s) - 1, i] = True
    return pit


def pass_direction(truths, params: dict[str, Params], S: int, seed: int) -> pd.DataFrame:
    """Swap rate per lap of adjacent running cars (no stop or neutral lap involved) by how much faster the car behind is."""
    data = {"real": []}
    for name in params:
        data[name] = []
    for t in truths:
        if t.gap_ahead is None:
            continue
        P = np.where(np.isfinite(t.pos_actual), t.pos_actual, 0.0)
        data["real"] += [(*r, 1.0) for r in _pair_rows(P, t.gap_ahead, _real_pit(t), t.neutral, t.pace, t.n_laps)]
        for name, par in params.items():
            res = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + int(t.race_id))
            for s in range(S):
                data[name] += [(*r, 1.0 / S) for r in _pair_rows(res.pos[s].T.astype(float), res.gap[s].T.astype(float), res.pit[s].T, t.neutral, t.pace, t.n_laps)]
    rows = []
    for name, d in data.items():
        a = np.array(d, float).reshape(-1, 4)
        if not len(a):
            continue
        a = a[np.isfinite(a[:, 0])]
        dp, sw, w = a[:, 0], a[:, 2], a[:, 3]
        for lo, hi in zip(PACE_BINS[:-1], PACE_BINS[1:]):
            m = (dp > lo) & (dp <= hi)
            if m.any():
                rows.append({"setting": name, "pace_advantage_of_car_behind_s": f"({lo:g}, {hi:g}]", "pairs": float(w[m].sum()), "swap_rate": float(np.average(sw[m], weights=w[m]))})
        swaps = sw > 0
        if swaps.any():
            rows.append({"setting": name, "pace_advantage_of_car_behind_s": "share of swaps made by the faster car", "pairs": float(w[swaps].sum()),
                         "swap_rate": float(np.average(dp[swaps] > 0, weights=w[swaps]))})
            rows.append({"setting": name, "pace_advantage_of_car_behind_s": "share of swaps made by a car 0.3+ s a lap slower", "pairs": float(w[swaps].sum()),
                         "swap_rate": float(np.average(dp[swaps] < -0.3, weights=w[swaps]))})
    return pd.DataFrame(rows)


def _lead_events(pos, pit, nv, pole, ret, L) -> list[tuple]:
    out = []
    lead = (pos == 1)
    for k in range(1, L):
        a, b = bool(lead[k - 1, pole]), bool(lead[k, pole])
        if a == b:
            continue
        lap = k + 1
        new = np.where(pos[k] == 1)[0] if not b else np.array([pole])
        old = np.where(pos[k - 1] == 1)[0] if not a else np.array([pole])
        who = [pole] + [int(x) for x in new] + [int(x) for x in old]
        linked = any(pit[j, w] for w in who for j in (k - 1, k) if 0 <= j < L)
        if nv[lap] > 0 or nv[lap - 1] > 0:
            cause = "safety car or VSC"
        elif ret[pole] <= lap and not b:
            cause = "retired"
        elif linked:
            cause = "stop"
        else:
            cause = "on-track pass"
        out.append((lap / L, "lose" if a else "regain", cause))
    return out


def lead_changes(truths, params: dict[str, Params], S: int, seed: int) -> pd.DataFrame:
    """How often per race the pole sitter loses the lead and gets it back, in the first and second half, and why."""
    rows = []
    n = len(truths)
    acc = {"real": []}
    for name in params:
        acc[name] = []
    for t in truths:
        pole = int(np.argmin(t.slot))
        P = np.where(np.isfinite(t.pos_actual), t.pos_actual, 0.0)
        acc["real"] += [(*e, 1.0) for e in _lead_events(P, _real_pit(t), t.neutral, pole, t.retire, t.n_laps)]
        for name, par in params.items():
            res = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + int(t.race_id))
            for s in range(S):
                acc[name] += [(*e, 1.0 / S) for e in _lead_events(res.pos[s].T.astype(float), res.pit[s].T, t.neutral, pole, t.retire, t.n_laps)]
    for name, ev in acc.items():
        df = pd.DataFrame(ev, columns=["frac", "kind", "cause", "w"])
        if df.empty:
            continue
        df["half"] = np.where(df["frac"] <= 0.5, "first half", "second half")
        g = df.groupby(["half", "kind", "cause"])["w"].sum() / max(n, 1)
        for (half, kind, cause), v in g.items():
            rows.append({"setting": name, "half": half, "kind": kind, "cause": cause, "per_race": float(v)})
        for kind in ("lose", "regain"):
            rows.append({"setting": name, "half": "whole race", "kind": kind, "cause": "all", "per_race": float(df.loc[df["kind"] == kind, "w"].sum() / max(n, 1))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ outputs

def chart(table: pd.DataFrame, real_pole: float, path: Path) -> None:
    from ..analysis.grid import SERIES, TEXT2, _frame, _plt
    d = table[(table["split"] == "all") & (table["setting"] == "matched passing")]
    if d.empty:
        return
    plt = _plt()
    fig, ax = _frame(plt, "Does less pace drift put the pole sitter back in front?",
                     "Share of races the pole sitter wins with everything known, at the real pass rate, by how much independent drift each car has. "
                     "Dashed line: real.", size=(9.5, 5.4))
    for c, (g, e) in zip(SERIES, d.groupby("noise_factor", sort=False)):
        e = e.sort_values("drift_factor")
        ax.plot(e["drift_factor"], e["pole_wins"], color=c, lw=2, marker="o", ms=5, label=f"lap noise x{g:g}")
    ch = d[d["chosen"]]
    if len(ch):
        ax.scatter(ch["drift_factor"], ch["pole_wins"], s=140, facecolors="none", edgecolors=TEXT2, linewidths=1.6, zorder=5)
    ax.axhline(real_pole, color=TEXT2, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Pace drift as a share of the measured size")
    ax.set_ylabel("Pole sitter wins")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def findings(res: dict, chk_wander: pd.DataFrame, direction: pd.DataFrame, lead: pd.DataFrame, old: Params, new: Params) -> list[str]:
    L = []
    wt, tab, real = res["warm_table"], res["table"], res["real"]
    L.append("WARM-UP AFTER A STOP (laps 2-4 after the first of two stops, the car on new tyres gains this much a lap on the car on old tyres, "
             f"real {real['all']['closing_a_first']:+.2f} when the car ahead stops first, {real['all']['closing_b_first']:+.2f} when the car behind does; "
             "simulated by warm-up scale): " + "; ".join(f"x{r.warm_scale:g} {r.closing_a_first:+.2f}/{r.closing_b_first:+.2f}" for r in wt.itertuples())
             + f". Kept: x{res['warm_scale']:g}.")
    best = res["best"]
    L.append(f"REFIT: drift x{best['drift_factor']:g} ({new.drift_sd:.2f} s), lap noise x{best['noise_factor']:g} ({new.sigma_lap:.2f} s), every passing threshold x{best['threshold_factor']:.2f} "
             f"(general {new.thr:.2f}), chosen on the fitting races for the ten-lap gap wander, close running and grip, with the pass rate held at the real one (loss {best['loss']:.1f}).")
    if res["edge"]:
        L.append("  WARNING: settings at the edge of their search range: " + ", ".join(res["edge"]) + " - the best value may lie outside it")
    for s in ("train", "holdout", "earlier era", "all"):
        o = tab[(tab["split"] == s) & (tab["setting"] == "v2.7 as fitted")]
        n = tab[(tab["split"] == s) & tab["chosen"]]
        if o.empty or n.empty:
            continue
        o, n = o.iloc[0], n.iloc[0]
        L.append(f"  {s} ({int(n['races'])} races), old -> new (real): pole wins {o['pole_wins']:.1%} -> {n['pole_wins']:.1%} ({n['real_pole_wins']:.1%}); "
                 f"pass rate {o['pass_rate']:.4f} -> {n['pass_rate']:.4f} ({n['real_pass_rate']:.4f}); close share {o['close_share']:.1%} -> {n['close_share']:.1%} ({n['real_close_share']:.1%}); "
                 f"ten-lap gap wander (robust sd) {o['wander_robust_sd']:.2f} -> {n['wander_robust_sd']:.2f} s ({n['real_wander_robust_sd']:.2f}); grip gap {o['grip_gap']:.3f} -> {n['grip_gap']:.3f}")
    a = tab[(tab["split"] == "all") & (tab["setting"] == "matched passing")]
    L.append("POLE WINS AT THE REAL PASS RATE, by drift x lap noise (all races): " + "; ".join(
        f"drift x{r.drift_factor:g} noise x{r.noise_factor:g}: {r.pole_wins:.1%}" for r in a.sort_values(["noise_factor", "drift_factor"], ascending=[False, False]).itertuples()))
    if not direction.empty:
        for name in direction["setting"].unique():
            d = direction[direction["setting"] == name]
            fw = d[d["pace_advantage_of_car_behind_s"] == "share of swaps made by the faster car"]["swap_rate"]
            sl = d[d["pace_advantage_of_car_behind_s"] == "share of swaps made by a car 0.3+ s a lap slower"]["swap_rate"]
            if len(fw) and len(sl):
                L.append(f"PASS DIRECTION, {name}: {fw.iloc[0]:.0%} of swaps are made by the faster car, {sl.iloc[0]:.1%} by a car 0.3+ s a lap slower")
    if not lead.empty:
        for name in lead["setting"].unique():
            d = lead[(lead["setting"] == name) & (lead["half"] == "whole race")]
            lo = d.loc[d["kind"] == "lose", "per_race"]
            rg = d.loc[d["kind"] == "regain", "per_race"]
            if len(lo) and len(rg):
                L.append(f"POLE SITTER AND THE LEAD, {name}: loses it {lo.iloc[0]:.2f} times a race, regains it {rg.iloc[0]:.2f}")
    w = chk_wander[chk_wander["split"] == "all"]
    L.append("TEN-LAP GAP WANDER, robust sd of the change in the gap between adjacent cars (all races): " + "; ".join(
        f"{r.setting} {r.robust_sd_s:.2f} s" for r in w.itertuples()))
    return L


def run(out_dir: Path | None = None, cal_dir: Path | None = None, S: int = 16, S_eval: int = 20, S_check: int = 10, seed: int = 0,
        drift_grid=DRIFT_GRID, noise_grid=NOISE_GRID, warm_grid=WARM_GRID, progress: bool = True) -> list[str]:
    cal = Path(cal_dir) if cal_dir else config.outputs_dir() / "sim_calibration"
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sim_refit"
    out.mkdir(parents=True, exist_ok=True)
    backup = cal / "params_before_refit.json"
    cur = cal / "params.json"
    if not backup.exists():
        shutil.copyfile(cur, backup)
    truths, _ = C.load_truths(cal)
    meta = json.loads(backup.read_text(encoding="utf-8"))
    base = Params(**meta["params"])
    if progress:
        print(f"Re-fitting on {len(truths)} races ...", flush=True)
    res = fit(truths, base, S=S, S_eval=S_eval, seed=seed, drift_grid=drift_grid, noise_grid=noise_grid, warm_grid=warm_grid, progress=progress)
    new = res["params"]
    old = replace(base, warm_scale=1.0)
    sets = {"v2.7": old, "v2.8": new}
    cw = wander_check(truths, res["wins"], res["evs"], sets, S_check, seed + 5)
    if progress:
        print("Checking the direction of passes and the lead ...", flush=True)
    direction = pass_direction(truths, sets, max(4, S_check // 2), seed + 6)
    lead = lead_changes(truths, sets, S_check, seed + 7)
    res["table"].to_csv(out / "refit_grid.csv", index=False)
    res["warm_table"].to_csv(out / "warm_up_check.csv", index=False)
    cw.to_csv(out / "wander_check.csv", index=False)
    direction.to_csv(out / "pass_direction.csv", index=False)
    lead.to_csv(out / "lead_changes.csv", index=False)
    chart(res["table"], res["real"]["all"]["pole_wins"], out / "refit_pole_wins.png")
    meta_new = {**meta, "params": asdict(new),
                "refit": {"drift_factor": float(res["best"]["drift_factor"]), "noise_factor": float(res["best"]["noise_factor"]),
                          "threshold_factor": float(res["best"]["threshold_factor"]), "warm_scale": float(res["warm_scale"])}}
    cur.write_text(json.dumps(meta_new, indent=2), encoding="utf-8")
    lines = findings(res, cw, direction, lead, old, new)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    cal = config.outputs_dir() / "sim_calibration"
    if not (cal / "truth.pkl").exists():
        print(f"No calibration in {cal}. Run  python -m src.sim.calibrate  first.")
        return 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines = run()
    print("\n".join(lines))
    print(f"\nTables written to {config.outputs_dir() / 'sim_refit'}; the re-fitted settings are in {cal / 'params.json'} (the old ones in params_before_refit.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
