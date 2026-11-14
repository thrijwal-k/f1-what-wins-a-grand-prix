"""Stage 3, part 3: forecasts made before the race, from earlier races only, and scored afterwards.

Run:  python -m src.sim.calibrate   (once)   then   python -m src.sim.forecast
Writes outputs/sim_forecast/.

For each dry race, in date order, with at least MIN_HISTORY earlier races:
  1. Pace estimate. Each car's pace is the weighted average of its team's pace in recent races (weight halves every
     HALF_LIFE races, and by SEASON_DECAY at each season boundary: the cars change) plus the driver's shrunk gap to the
     team-mates. A new team takes its drivers' own history. A car with no history is given a slow default.
  2. Four ways to turn that into the pace the simulator uses:
       history             the estimate above
       grid                the average pace of a car starting from that grid slot (what the grid alone says)
       history+grid        a regression of real race pace on both, fitted on earlier races
       history+quali gap   the same regression plus the car's first-session qualifying time gap (a gap in seconds says more
                           than a slot number)
     Two of these are also scored recalibrated: a two-parameter logistic correction of the probabilities, fitted on the model's
     own forecasts of earlier races (nothing from the race being forecast), with the win probabilities rescaled to add to 1.
  3. Uncertainty. Each simulation draws a team shock and a driver shock for pace, sized from how wrong the same model
     was in earlier races. Strategies, safety cars and retirements are sampled from earlier races (sample.py).
  4. Score the probabilities of winning, finishing on the podium and in the top ten against the real result, next to
     (a) a model that knows only the grid slot and (b) the same simulation given each car's true pace (the ceiling).

Nothing from the race being forecast, or later, is used. The 2025-on races were not used to choose any setting.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import calibrate as C
from . import engine as E
from . import replay as R
from .calibrate import RaceTruth, split_of
from .engine import Params
from .sample import MIN_POOL, Pool, sample_inputs

HALF_LIFE = 5.0          # races
SEASON_DECAY = 0.5       # weight multiplier per season boundary
KAPPA = 3.0              # pseudo-races of zero gap that shrink a driver's gap to the team-mates
MIN_HISTORY = 20
MIN_FIT_RACES = 8
SLOW_DEFAULT_Q = 0.75
PRACTICE = "History + grid + practice"
VARIANTS = ("History", "Grid", "History + grid", "History + quali gap", PRACTICE)
RECAL_BASES = ("History + grid", "History + quali gap", PRACTICE)
RECAL_SUFFIX = ", recalibrated"
MIN_RECAL_CARS = 600     # earlier forecast car-races needed before a recalibration is fitted (about 30 races)
ORACLE = "Oracle pace (L1)"
BASELINE = "Grid slot only"
DEFAULT_SIGMA = (0.30, 0.30)


@dataclass
class RaceCard:
    race_id: int
    season: int
    circuit_id: str
    n_laps: int
    drivers: list
    constructors: list
    slot: np.ndarray
    quali_gap: np.ndarray | None = None      # first-session qualifying gap in percent, known before the race
    practice_long: np.ndarray | None = None  # practice long-run pace (s/lap vs the field's middle), known before qualifying
    practice_best: np.ndarray | None = None  # best practice lap, percent slower than the fastest

    @property
    def n(self) -> int:
        return len(self.drivers)

    @property
    def has_practice(self) -> bool:
        return self.practice_long is not None


def _arr(x):
    return None if x is None else np.asarray(x, float)


def card_of(t: RaceTruth) -> RaceCard:
    return RaceCard(t.race_id, t.season, t.circuit_id, t.n_laps, list(t.drivers), list(t.constructors), t.slot,
                    _arr(getattr(t, "quali_gap", None)), _arr(getattr(t, "practice_long", None)),
                    _arr(getattr(t, "practice_best", None)))


def _zeros_if_none(x, n: int) -> np.ndarray:
    return np.zeros(n) if x is None else np.asarray(x, float)


def _qgap(card_or_truth) -> np.ndarray:
    q = getattr(card_or_truth, "quali_gap", None)
    return np.zeros(len(card_or_truth.drivers)) if q is None else np.asarray(q, float)


# ------------------------------------------------------------------- pace

def history_pace(hist: list[RaceTruth], card: RaceCard) -> np.ndarray:
    """Pre-race pace estimate for each car on the card, from earlier races only."""
    n = card.n
    team_acc, team_w, drv_dev, drv_w, drv_tot, drv_tw = {}, {}, {}, {}, {}, {}
    k_total = len(hist)
    for rank, t in enumerate(hist):
        age = k_total - 1 - rank
        w = 0.5 ** (age / HALF_LIFE) * SEASON_DECAY ** max(0, card.season - t.season)
        cons = np.array(t.constructors)
        for c in set(cons):
            m = (cons == c) & t.pace_known
            if m.any():
                mean = float(t.pace[m].mean())
                team_acc[c] = team_acc.get(c, 0.0) + w * mean
                team_w[c] = team_w.get(c, 0.0) + w
                for i in np.flatnonzero(m):
                    d = t.drivers[i]
                    drv_dev[d] = drv_dev.get(d, 0.0) + w * (t.pace[i] - mean)
                    drv_w[d] = drv_w.get(d, 0.0) + w
        for i in np.flatnonzero(t.pace_known):
            d = t.drivers[i]
            drv_tot[d] = drv_tot.get(d, 0.0) + w * t.pace[i]
            drv_tw[d] = drv_tw.get(d, 0.0) + w
    form = {c: team_acc[c] / team_w[c] for c in team_acc}
    slow = float(np.quantile(list(form.values()), SLOW_DEFAULT_Q)) if form else 0.0
    pred = np.empty(n)
    for i, (d, c) in enumerate(zip(card.drivers, card.constructors)):
        dev = drv_dev.get(d, 0.0) / (drv_w.get(d, 0.0) + KAPPA)
        if c in form and team_w[c] >= 0.5:
            pred[i] = form[c] + dev
        elif d in drv_tot:
            pred[i] = drv_tot[d] / drv_tw[d]               # new team: the driver's own history
        else:
            pred[i] = slow
    # a new team whose drivers have history: nothing more to do; its cars are already on their own averages
    return pred


def slot_profile(hist: list[RaceTruth], card: RaceCard) -> np.ndarray:
    """Average pace of the car starting from each slot in earlier races (same era when there are enough), smoothed."""
    same = [t for t in hist if C.era_of(t.season) == C.era_of(card.season)]
    use = same if len(same) >= MIN_POOL else hist
    s = np.zeros(32)
    c = np.zeros(32)
    for t in use:
        m = t.pace_known
        np.add.at(s, t.slot[m], t.pace[m])
        np.add.at(c, t.slot[m], 1.0)
    kern = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    prof = np.convolve(s, kern, "same") / np.maximum(np.convolve(c, kern, "same"), 1e-9)
    return prof[card.slot]


@dataclass
class Record:
    pred: np.ndarray
    prof: np.ndarray
    oracle: np.ndarray
    known: np.ndarray
    cons: np.ndarray
    qgap: np.ndarray | None = None
    plong: np.ndarray | None = None
    pbest: np.ndarray | None = None

    @property
    def has_practice(self) -> bool:
        return self.plong is not None


def feature_matrix(pred, prof, qgap, plong, pbest) -> np.ndarray:
    """Columns: constant, history, grid profile, qualifying gap, practice long run, practice best lap."""
    n = len(pred)
    return np.column_stack([np.ones(n), pred, prof, _zeros_if_none(qgap, n), _zeros_if_none(plong, n), _zeros_if_none(pbest, n)])


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_blend(records: list[Record], variant: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Coefficients (const, history, grid, quali gap, practice long run, practice best lap) for the variant and the
    team/driver pace-shock sizes, from earlier races. The practice variant is fitted on weekends that have practice laps."""
    fixed = {"History": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), "Grid": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
    use = {"History + grid": [0, 1, 2], "History + quali gap": [0, 1, 2, 3], PRACTICE: [0, 1, 2, 4, 5]}
    if variant == PRACTICE:
        records = [r for r in records if r.has_practice]
    X, y, cons = [], [], []
    for r in records:
        m = r.known
        F = feature_matrix(r.pred, r.prof, r.qgap, r.plong, r.pbest)
        X.append(F[m])
        y.append(r.oracle[m])
        cons.append(r.cons[m])
    if not X or len(records) < MIN_FIT_RACES:
        coef = fixed.get(variant, np.array([0.0, 0.7, 0.3, 0.0, 0.0, 0.0]))
        return coef, DEFAULT_SIGMA
    Xa, ya = np.vstack(X), np.concatenate(y)
    if variant in fixed:
        coef = fixed[variant]
    else:
        cols = use[variant]
        coef = np.zeros(6)
        coef[cols] = _ols(Xa[:, cols], ya)
    team_means, within = [], []
    for x, yy, c in zip(X, y, cons):
        resid = yy - x @ coef
        for team in set(c):
            r = resid[c == team]
            if len(r) == 2:
                team_means.append(r.mean())
                within.append(r - r.mean())
            elif len(r) == 1:
                team_means.append(r[0])
    sd_dev_var = 2.0 * float(np.var(np.concatenate(within))) if within else DEFAULT_SIGMA[1] ** 2
    team_var = max(float(np.var(team_means)) - sd_dev_var / 2.0, 0.01 ** 2) if len(team_means) > 3 else DEFAULT_SIGMA[0] ** 2
    return coef, (math.sqrt(team_var), math.sqrt(sd_dev_var))


def fit_recal(p: np.ndarray, y: np.ndarray, ridge: float = 2.0, iters: int = 30) -> tuple[float, float]:
    """Logistic recalibration p' = sigmoid(a + b logit p), by penalised Newton steps toward (a, b) = (0, 1)."""
    x = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    prior = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (mu - y) + ridge * (w - prior)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def apply_recal(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    x = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    return 1.0 / (1.0 + np.exp(-(ab[0] + ab[1] * x)))


def pace_samples(mu: np.ndarray, cons: np.ndarray, sig: tuple[float, float], S: int, rng: np.random.Generator) -> np.ndarray:
    """(S, n): the estimate plus a shock shared by the team and one for each driver."""
    teams = sorted(set(cons))
    idx = np.array([teams.index(c) for c in cons])
    shock_t = rng.normal(0.0, sig[0], (S, len(teams)))[:, idx]
    shock_d = rng.normal(0.0, sig[1], (S, len(mu)))
    return mu[None, :] + shock_t + shock_d


# ----------------------------------------------------------------- forecast

def probabilities(finish: np.ndarray) -> dict:
    return {"p_win": (finish == 1).mean(0), "p_top3": (finish <= 3).mean(0), "p_top10": (finish <= 10).mean(0),
            "mean_finish": finish.mean(0)}


def forecast_card(hist: list[RaceTruth], card: RaceCard, par: Params, records: list[Record], S: int = 300, seed: int = 0,
                  pool: Pool | None = None, tyre: dict | None = None) -> dict[str, E.SimResult]:
    """Simulate a race that has not happened: one SimResult per variant, sharing the sampled strategies and safety cars.
    `tyre` holds the circuit's wear, compound offsets, fuel trend and pit loss (known before the race); by default the last race's."""
    pred = history_pace(hist, card)
    prof = slot_profile(hist, card)
    qg = _qgap(card)
    cons = np.array(card.constructors)
    pool = pool or Pool.before(hist, card)
    tyre = tyre or {"off": hist[-1].off, "deg": hist[-1].deg, "fuel": hist[-1].fuel, "pit_loss": hist[-1].pit_loss, "warm": hist[-1].warm_s}
    rng = np.random.default_rng(seed + 13)
    s = sample_inputs(pool, card.n, card.n_laps, card.circuit_id, card.slot, S, rng)
    out = {}
    feats = feature_matrix(pred, prof, qg, card.practice_long, card.practice_best)
    for variant in VARIANTS:
        # a weekend without practice laps gets the history + grid blend in the practice variant's place
        coef, sig = fit_blend(records, variant if (variant != PRACTICE or card.has_practice) else "History + grid")
        mu = feats @ coef
        pace = pace_samples(mu, cons, sig, S, np.random.default_rng(seed + 29))
        spec = E.RaceSpec(n_laps=card.n_laps, slot=card.slot, pace=pace, comp=s["comp"], stops=s["stops"],
                          start_age=s["start_age"], retire=s["retire"], neutral=s["neutral"], off=tyre["off"], deg=tyre["deg"],
                          fuel=tyre["fuel"], pit_loss=tyre["pit_loss"], cover=True, warm=tyre.get("warm"))
        out[variant] = E.simulate(spec, par.for_circuit(card.circuit_id), S=S, seed=seed, history=False)
    return out


def _recal_metrics(p: dict, actual: np.ndarray) -> dict:
    w = int(np.argmin(actual))
    return {"winner_logscore": float(-np.log(max(p["p_win"][w], 1e-3))), "winner_top1": bool(np.argmax(p["p_win"]) == w),
            "winner_prob": float(p["p_win"][w]),
            "brier_top3": float(np.mean((p["p_top3"] - (actual <= 3)) ** 2)),
            "brier_top10": float(np.mean((p["p_top10"] - (actual <= 10)) ** 2))}


def proxy_tyre(hist: list[RaceTruth], circuit_id: str) -> dict:
    """The tyre inputs a forecast can know before the race: those of the last earlier race at the same circuit, or of the last race.
    (walk_forward's default hands each race its own values, which a live forecast cannot have.)"""
    same = [t for t in hist if t.circuit_id == circuit_id]
    t = same[-1] if same else hist[-1]
    return {"off": t.off, "deg": t.deg, "fuel": t.fuel, "pit_loss": t.pit_loss, "warm": t.warm_s}


def walk_forward(truths: list[RaceTruth], par: Params, S: int = 300, min_history: int = MIN_HISTORY, progress: bool = True,
                 tyre_mode: str = "own", state: dict | None = None):
    """Forecast every race from the earlier ones. tyre_mode "own" gives each race its own tyre inputs (the original protocol),
    "proxy" gives it what a live forecast would have (proxy_tyre). If `state` is a dict, it is filled with the final `records` and
    `past` (each base model's own earlier forecasts), which a live forecast needs for its blend and recalibration."""
    if tyre_mode not in ("own", "proxy"):
        raise ValueError("tyre_mode must be 'own' or 'proxy'")
    truths = sorted(truths, key=lambda t: t.race_id)
    records: list[Record] = []
    past: dict[str, dict[str, list]] = {b: {"p_win": [], "p_top3": [], "p_top10": [], "y_win": [], "y_top3": [], "y_top10": []}
                                       for b in RECAL_BASES}          # each base model's own forecasts of earlier races
    rows, probs = [], []
    for idx, t in enumerate(truths):
        hist = truths[:idx]
        if len(hist) < min_history:
            continue
        card = card_of(t)
        pool = Pool.before(truths, t)
        base = R.grid_baseline(pool, t.slot, t.finish)
        gp = R.grid_probs(pool, t.slot)
        pole = int(np.argmin(t.slot))
        common = {"race_id": t.race_id, "season": t.season, "split": split_of(t.season), "circuit_id": t.circuit_id,
                  "pole_won": bool(t.finish[pole] == 1), "actual_grid_finish": float(np.corrcoef(t.slot, t.finish)[0, 1])}
        tyre = (proxy_tyre(hist, t.circuit_id) if tyre_mode == "proxy"
                else {"off": t.off, "deg": t.deg, "fuel": t.fuel, "pit_loss": t.pit_loss, "warm": t.warm_s})
        sims = forecast_card(hist, card, par, records, S=S, seed=int(t.race_id), pool=pool, tyre=tyre)
        sims[ORACLE] = R.run_l1(t, pool, par, S, int(t.race_id))
        now = {}
        for name, res in sims.items():
            m = R.finish_metrics(res.finish, t.finish)
            rows.append({**common, "level": name, **m, **base, "sim_pole_win": float((res.finish[:, pole] == 1).mean()),
                         "sim_grid_finish": math.nan})
            p = probabilities(res.finish)
            now[name] = p
            for i, d in enumerate(t.drivers):
                probs.append({"race_id": t.race_id, "season": t.season, "model": name, "driver_id": d,
                              "constructor_id": t.constructors[i], "slot": int(t.slot[i]), "p_win": float(p["p_win"][i]),
                              "p_top3": float(p["p_top3"][i]), "p_top10": float(p["p_top10"][i]),
                              "mean_finish": float(p["mean_finish"][i]), "finish": int(t.finish[i])})
        # recalibrated versions, from this model's own forecasts of earlier races
        for b in RECAL_BASES:
            hp = past[b]
            if len(hp["y_win"]) < MIN_RECAL_CARS:
                continue
            q = {}
            for key, col in (("p_win", "win"), ("p_top3", "top3"), ("p_top10", "top10")):
                ab = fit_recal(np.array(hp[key]), np.array(hp[f"y_{col}"], float))
                q[key] = apply_recal(now[b][key], ab)
            q["p_win"] = q["p_win"] / q["p_win"].sum()
            name = b + RECAL_SUFFIX
            m = R.finish_metrics(sims[b].finish, t.finish)
            m.update(_recal_metrics(q, t.finish))
            rows.append({**common, "level": name, **m, **base, "sim_pole_win": float((sims[b].finish[:, pole] == 1).mean()),
                         "sim_grid_finish": math.nan})
            for i, d in enumerate(t.drivers):
                probs.append({"race_id": t.race_id, "season": t.season, "model": name, "driver_id": d,
                              "constructor_id": t.constructors[i], "slot": int(t.slot[i]), "p_win": float(q["p_win"][i]),
                              "p_top3": float(q["p_top3"][i]), "p_top10": float(q["p_top10"][i]),
                              "mean_finish": float(now[b]["mean_finish"][i]), "finish": int(t.finish[i])})
        for b in RECAL_BASES:
            hp = past[b]
            hp["p_win"] += list(now[b]["p_win"]); hp["p_top3"] += list(now[b]["p_top3"]); hp["p_top10"] += list(now[b]["p_top10"])
            hp["y_win"] += list((t.finish == 1).astype(float)); hp["y_top3"] += list((t.finish <= 3).astype(float))
            hp["y_top10"] += list((t.finish <= 10).astype(float))
        for i, d in enumerate(t.drivers):
            probs.append({"race_id": t.race_id, "season": t.season, "model": BASELINE, "driver_id": d,
                          "constructor_id": t.constructors[i], "slot": int(t.slot[i]), "p_win": float(gp[1][i]),
                          "p_top3": float(gp[3][i]), "p_top10": float(gp[10][i]), "mean_finish": math.nan, "finish": int(t.finish[i])})
        pred = history_pace(hist, card)
        prof = slot_profile(hist, card)
        records.append(Record(pred, prof, t.pace, t.pace_known, np.array(t.constructors), _qgap(t),
                              _arr(getattr(t, "practice_long", None)), _arr(getattr(t, "practice_best", None))))
        if progress and (len(records)) % 20 == 0:
            print(f"  forecast {len(records)} races", flush=True)
    if state is not None:
        state.update({"records": records, "past": past})
    return pd.DataFrame(rows), pd.DataFrame(probs)


# ------------------------------------------------------------------ tables

def skill_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Brier skill against the grid-only benchmark (1 = perfect, 0 = no better than the grid, below 0 = worse)."""
    d = summary.copy()
    d["podium_skill"] = 1 - d["brier_top3"] / d["grid_only_brier_top3"]
    d["top10_skill"] = 1 - d["brier_top10"] / d["grid_only_brier_top10"]
    d["winner_logscore_gain"] = d["grid_only_winner_logscore"] - d["winner_logscore"]
    return d


def season_table(races: pd.DataFrame, models=("History + grid", "History + quali gap", PRACTICE, "History + grid, recalibrated",
                                                 "History + quali gap, recalibrated", PRACTICE + RECAL_SUFFIX)) -> pd.DataFrame:
    """Skill against the grid-only benchmark season by season, so a pooled number cannot hide a change of era."""
    d = races[races["level"].isin(models)].copy()
    if d.empty:
        return pd.DataFrame()
    d["split"] = d["season"].astype(int).astype(str)
    out = skill_table(R.summarise(d))
    out = out[out["split"] != "all"].rename(columns={"split": "season"})
    out["season"] = out["season"].astype(int)
    return out.sort_values(["level", "season"]).reset_index(drop=True)


def practice_signal(truths: list[RaceTruth], min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """How much practice pace tells about race pace, season by season, over the races that have practice laps.
    corr_practice: practice long-run pace against the car's race pace (both against the field's middle, within races).
    corr_history: the history-based estimate against race pace. corr_added: practice long-run pace against what history
    misses (race pace minus the best linear use of history, fitted within each race)."""
    rows = []
    truths = sorted(truths, key=lambda t: t.race_id)
    for idx, t in enumerate(truths):
        if idx < min_history or getattr(t, "practice_long", None) is None:
            continue
        card = card_of(t)
        pred = history_pace(truths[:idx], card)
        m = t.pace_known
        if m.sum() < 8:
            continue
        y = t.pace[m] - np.median(t.pace[m])
        pl = np.asarray(t.practice_long)[m]
        ph = pred[m] - np.median(pred[m])
        beta = np.polyfit(ph, y, 1) if np.ptp(ph) > 0 else np.array([0.0, 0.0])
        resid = y - np.polyval(beta, ph)
        c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else math.nan
        rows.append({"race_id": t.race_id, "season": t.season, "cars": int(m.sum()), "corr_practice": c(pl, y),
                     "corr_history": c(ph, y), "corr_added": c(pl, resid)})
    if not rows:
        return pd.DataFrame(columns=["season", "races", "corr_practice", "corr_history", "corr_added"])
    d = pd.DataFrame(rows)
    out = d.groupby("season").agg(races=("race_id", "size"), corr_practice=("corr_practice", "mean"),
                                  corr_history=("corr_history", "mean"), corr_added=("corr_added", "mean")).reset_index()
    allrow = pd.DataFrame([{"season": 0, "races": len(d), "corr_practice": d["corr_practice"].mean(),
                            "corr_history": d["corr_history"].mean(), "corr_added": d["corr_added"].mean()}])
    return pd.concat([out, allrow], ignore_index=True)


def reliability(probs: pd.DataFrame, bins=(0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0001)) -> pd.DataFrame:
    rows = []
    for model, d in probs.groupby("model"):
        for col, k in (("p_win", 1), ("p_top3", 3), ("p_top10", 10)):
            y = (d["finish"] <= k).astype(float)
            cut = pd.cut(d[col], bins=list(bins), include_lowest=True, right=False)
            g = pd.DataFrame({"p": d[col], "y": y, "bin": cut}).groupby("bin", observed=True)
            for b, x in g:
                rows.append({"model": model, "outcome": {1: "win", 3: "podium", 10: "top ten"}[k], "bin": str(b),
                             "cars": len(x), "mean_forecast": float(x["p"].mean()), "observed": float(x["y"].mean())})
    return pd.DataFrame(rows)


def log_loss_by_model(probs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in probs.groupby("model"):
        for col, k in (("p_win", 1), ("p_top3", 3), ("p_top10", 10)):
            p = d[col].clip(1e-3, 1 - 1e-3)
            y = (d["finish"] <= k).astype(float)
            rows.append({"model": model, "outcome": {1: "win", 3: "podium", 10: "top ten"}[k], "cars": len(d),
                         "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                         "brier": float(((p - y) ** 2).mean())})
    return pd.DataFrame(rows)


def chart_skill(summary: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import BLUE, ORANGE, AXIS, TEXT2, _frame, _plt
    d = summary[summary["split"] == "all"]
    if d.empty:
        return
    plt = _plt()
    order = [BASELINE] + list(VARIANTS) + [b + RECAL_SUFFIX for b in RECAL_BASES] + [ORACLE]
    fig, ax = _frame(plt, "Do pre-race forecasts beat knowing only the grid?",
                     "Podium forecast error (Brier score, lower is better) on every race forecast from earlier races only. "
                     "The last bar is the same simulator given each car's true pace.", size=(11.5, 5.4))
    vals, labels = [], []
    for name in order:
        if name == BASELINE:
            v = float(d["grid_only_brier_top3"].mean())      # note: over every race with a forecast
        else:
            x = d[d["level"] == name]
            v = float(x["brier_top3"].iloc[0]) if len(x) else math.nan
        vals.append(v)
        labels.append(name)
    from ..analysis.grid import SERIES
    colours = [AXIS if n == BASELINE else TEXT2 if n == ORACLE else SERIES[2] if n.endswith(RECAL_SUFFIX) else
               SERIES[3] if "practice" in n else ORANGE if "quali" in n else BLUE for n in labels]
    bars = ax.bar(range(len(vals)), vals, 0.6, color=colours)
    for b, v in zip(bars, vals):
        if not math.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, color=TEXT2)
    ax.set_xticks(range(len(vals)))
    import textwrap
    ax.set_xticklabels(["\n".join(textwrap.wrap(l.replace(", recalibrated", " recalibrated").replace("History + ", "Hist. + "), 12))
                        for l in labels], fontsize=8)
    ax.set_ylabel("Podium Brier score")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def chart_calibration(rel: pd.DataFrame, path: Path, model: str = "History + grid") -> None:
    from ..analysis.grid import BLUE, ORANGE, AXIS, TEXT2, _frame, _plt
    d = rel[rel["model"] == model]
    if d.empty:
        return
    plt = _plt()
    fig, ax = _frame(plt, "When it says 20%, does it happen 20% of the time?",
                     f"Forecast probability against how often it happened, {model.lower()} forecasts. "
                     "Dot size is the number of car-races. On the diagonal is calibrated.", size=(7.4, 6.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls="--")
    for outcome, colour in (("win", ORANGE), ("podium", BLUE), ("top ten", TEXT2)):
        x = d[d["outcome"] == outcome]
        ax.scatter(x["mean_forecast"], x["observed"], s=np.sqrt(x["cars"]) * 9, color=colour, alpha=0.85, label=outcome,
                   edgecolor="white", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def findings(summary: pd.DataFrame, ll: pd.DataFrame, common: pd.DataFrame | None = None, ll_common: pd.DataFrame | None = None,
             seasons: pd.DataFrame | None = None, signal: pd.DataFrame | None = None) -> list[str]:
    L = []
    if signal is not None and len(signal):
        a = signal[signal["season"] == 0]
        if len(a):
            r = a.iloc[0]
            L.append(f"PRACTICE SIGNAL ({int(r['races'])} races with practice laps): practice long-run pace correlates {r['corr_practice']:.2f} "
                     f"with the car's race pace (history-based estimate {r['corr_history']:.2f}); against what history misses, "
                     f"{r['corr_added']:.2f}. By season: " + "; ".join(
                         f"{int(x['season'])} {x['corr_practice']:.2f}/{x['corr_history']:.2f}/{x['corr_added']:.2f}"
                         for _, x in signal[signal["season"] != 0].iterrows()) + " (practice / history / added)")
        else:
            L.append("PRACTICE SIGNAL: no race with practice laps")
    sk = skill_table(summary)
    blocks = [("", sk, split) for split in ("holdout", "train", "earlier era", "all")]
    if common is not None and len(common):
        blocks += [("SAME RACES FOR EVERY MODEL, ", skill_table(common), split) for split in ("holdout", "all")]
    for tag, sk, split in blocks:
        d = sk[sk["split"] == split]
        if d.empty:
            continue
        n = int(d["races"].max())
        L.append(f"{tag}{split.upper()} ({n} races forecast from earlier races only)")
        g = d.iloc[0]
        L.append(f"  grid slot only: winner called {_pct(g['grid_only_winner_called'])}, winner log score {g['grid_only_winner_logscore']:.2f}, "
                 f"podium Brier {g['grid_only_brier_top3']:.3f}, top-ten Brier {g['grid_only_brier_top10']:.3f}")
        for name in list(VARIANTS) + [b + RECAL_SUFFIX for b in RECAL_BASES] + [ORACLE]:
            r = d[d["level"] == name]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(f"  {name}: winner called {_pct(r['winner_called'])}, winner log score {r['winner_logscore']:.2f} "
                     f"({r['winner_logscore_gain']:+.2f} against grid only), podium Brier {r['brier_top3']:.3f} "
                     f"(skill {r['podium_skill']:+.0%}), top-ten Brier {r['brier_top10']:.3f} (skill {r['top10_skill']:+.0%}), "
                     f"finishing-order correlation {r['finish_order_correlation']:.2f} (grid only {r['grid_only_correlation']:.2f})")
    L.append("  Skill = 1 - Brier / grid-only Brier: positive means better than knowing only the grid. The oracle row is a ceiling "
             "(true pace, sampled everything else), not a forecast.")
    if len(ll):
        for outcome in ("win", "podium", "top ten"):
            x = ll[ll["outcome"] == outcome].sort_values("log_loss")
            L.append(f"LOG LOSS, {outcome} (all forecasts, lower is better): " + "; ".join(
                f"{r['model']} {r['log_loss']:.3f}" for _, r in x.iterrows()))
    if ll_common is not None and len(ll_common):
        for outcome in ("win", "podium", "top ten"):
            x = ll_common[ll_common["outcome"] == outcome].sort_values("log_loss")
            L.append(f"LOG LOSS, {outcome}, SAME RACES FOR EVERY MODEL ({int(x['cars'].iloc[0]) // 20} races or so): " + "; ".join(
                f"{r['model']} {r['log_loss']:.3f}" for _, r in x.iterrows()))
    if seasons is not None and len(seasons):
        for name in sorted(seasons["level"].unique(), key=lambda n: (n.endswith(RECAL_SUFFIX), n)):
            x = seasons[seasons["level"] == name]
            L.append(f"BY SEASON, {name} (skill against the grid-only benchmark): " + "; ".join(
                f"{int(r['season'])} ({int(r['races'])} races) podium {r['podium_skill']:+.0%}, top ten {r['top10_skill']:+.0%}, "
                f"winner log score {r['winner_logscore']:.2f} vs {r['grid_only_winner_logscore']:.2f}" for _, r in x.iterrows()))
    return L


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


# --------------------------------------------------------------------- run

def run(out_dir: Path | None = None, cal_dir: Path | None = None, S: int = 300, progress: bool = True,
        tyre_mode: str = "own") -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / ("sim_forecast" if tyre_mode == "own" else "sim_forecast_proxy")
    out.mkdir(parents=True, exist_ok=True)
    truths, par = C.load_truths(cal_dir)
    races, probs = walk_forward(truths, par, S=S, progress=progress, tyre_mode=tyre_mode)
    if races.empty:
        lines = [f"Too few races for a walk-forward forecast (need more than {MIN_HISTORY} usable races)."]
        (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    summary = R.summarise(races)
    ids = set(races.loc[races["level"].str.endswith(RECAL_SUFFIX), "race_id"])
    common = R.summarise(races[races["race_id"].isin(ids)]) if ids else pd.DataFrame()
    rel = reliability(probs)
    ll = log_loss_by_model(probs)
    ll_common = log_loss_by_model(probs[probs["race_id"].isin(ids)]) if ids else pd.DataFrame()
    seasons = season_table(races)
    seasons.to_csv(out / "forecast_by_season.csv", index=False)
    signal = practice_signal(truths)
    signal.to_csv(out / "practice_signal.csv", index=False)
    races.to_csv(out / "forecast_races.csv", index=False)
    summary.to_csv(out / "forecast_summary.csv", index=False)
    common.to_csv(out / "forecast_summary_same_races.csv", index=False)
    probs.to_csv(out / "forecast_probabilities.csv", index=False)
    rel.to_csv(out / "calibration.csv", index=False)
    ll.to_csv(out / "log_loss.csv", index=False)
    ll_common.to_csv(out / "log_loss_same_races.csv", index=False)
    chart_skill(common if len(common) else summary, out / "forecast_skill.png")
    chart_calibration(rel, out / "forecast_calibration.png")
    lines = findings(summary, ll, common, ll_common, seasons, signal)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Walk-forward forecasts of every earlier race, scored against the real result.")
    ap.add_argument("--proxy-tyre", action="store_true",
                    help="give each race the tyre inputs a live forecast would have (last race at the same circuit) instead of its own; "
                         "writes outputs/sim_forecast_proxy")
    args = ap.parse_args(argv)
    mode = "proxy" if args.proxy_tyre else "own"
    lines = run(tyre_mode=mode)
    print("\n".join(lines))
    print(f"\nTables and charts written to {config.outputs_dir() / ('sim_forecast' if mode == 'own' else 'sim_forecast_proxy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
