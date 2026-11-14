"""Stage 3, part 2: replay real races through the simulator and see how close it gets.

Run:  python -m src.sim.calibrate   (once)   then   python -m src.sim.replay
Writes outputs/sim_replay/.

Two levels, each run on every dry, no-red-flag race since 2018:
  L0  everything as it happened: each car's pace, strategy, retirements, safety-car laps. Only traffic, passing and noise are
      the simulator's. This tests the mechanics.
  L1  pace and grid as they were, but strategies, safety cars and retirements are sampled from earlier races. This asks how
      much of a result was set by pace and grid, and how much by what the simulator has to guess.
The grid-only benchmark (the chance of winning, finishing on the podium or in the top ten from the grid slot alone, learnt
from earlier races) shows what pace adds.

Validation against stage 2: the grip-decay curve, how often pole wins, how pole is lost (start, pit-stop cycle, safety car,
retirement, on-track pass) and the undercut advantage by gap between the stops, all computed on simulated and real races with the
same code. It also splits the pass-rate and close-running comparison by running position and compares real stop timing with the sampled strategies.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from ..analysis import race_flow
from ..analysis.race_flow import FRACTIONS
from . import calibrate as C
from . import engine as E
from .calibrate import RaceTruth, split_of
from .engine import BIG, Params, spearman_masked
from .sample import MIN_POOL, Pool, sample_inputs

S_RACE = 200            # simulations per race
S_EVENTS = 24           # of which this many are turned into lap tables for the undercut and lead-change checks
S_POSITION = 40         # simulations per race for the by-position pass and close-running check
LEVELS = ("L0 everything as it happened", "L1 pace known, rest sampled")


# --------------------------------------------------------------- scoring

def finish_metrics(finish: np.ndarray, actual: np.ndarray) -> dict:
    """finish (S, n) simulated positions, actual (n,) real positions."""
    S, n = finish.shape
    act = np.broadcast_to(actual.astype(float), finish.shape)
    rho = float(np.nanmean(spearman_masked(finish.astype(float), act, np.ones(finish.shape, bool))))
    exp_rank = np.argsort(np.argsort(finish.mean(0), kind="stable"), kind="stable") + 1
    w = int(np.argmin(actual))
    win_p = (finish == 1).mean(0)
    p_w = ((finish[:, w] == 1).sum() + 0.5) / (S + 1)
    p3, p10 = (finish <= 3).mean(0), (finish <= 10).mean(0)
    rho_exp = float(np.corrcoef(exp_rank, actual)[0, 1]) if n > 2 else math.nan
    return {"spearman": rho, "spearman_expected": rho_exp, "mean_rank_error": float(np.abs(exp_rank - actual).mean()),
            "winner_logscore": float(-np.log(p_w)), "winner_top1": bool(np.argmax(win_p) == w),
            "winner_prob": float(win_p[w]),
            "brier_top3": float(np.mean((p3 - (actual <= 3)) ** 2)), "brier_top10": float(np.mean((p10 - (actual <= 10)) ** 2))}


def grid_probs(pool: Pool, slot: np.ndarray) -> dict[int, np.ndarray]:
    """P(finish <= k | grid slot) for k = 1, 3, 10 from earlier races, smoothed over neighbouring slots and scaled so the
    probabilities of a race add up to k."""
    kern = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    probs = {}
    for k in (1, 3, 10):
        cnt, tot = np.zeros(30), np.zeros(30)
        for t in pool.truths:
            np.add.at(cnt, t.slot, (t.finish <= k).astype(float))
            np.add.at(tot, t.slot, 1.0)
        sm = lambda a: np.convolve(a, kern, mode="same")
        p = (sm(cnt) + 0.2 * k / 20) / (sm(tot) + 0.2)
        pc = p[slot]
        pc = np.clip(pc * k / pc.sum(), 1e-3, 0.99)
        probs[k] = pc
    return probs


def grid_baseline(pool: Pool, slot: np.ndarray, actual: np.ndarray) -> dict:
    """The same scores for a model that knows only the grid slot."""
    probs = grid_probs(pool, slot)
    w = int(np.argmin(actual))
    return {"base_winner_logscore": float(-np.log(probs[1][w])), "base_winner_top1": bool(np.argmax(probs[1]) == w),
            "base_brier_top3": float(np.mean((probs[3] - (actual <= 3)) ** 2)),
            "base_brier_top10": float(np.mean((probs[10] - (actual <= 10)) ** 2)),
            "base_spearman": float(np.corrcoef(np.argsort(np.argsort(slot)), actual)[0, 1]),
            "base_mean_rank_error": float(np.abs(np.argsort(np.argsort(slot)) + 1 - actual).mean())}


# ---------------------------------------------------- simulated lap tables

def sim_laps(t: RaceTruth, res: E.SimResult, s: int) -> pd.DataFrame:
    """One simulation as a laps table the stage 2 functions can read."""
    n, L = t.n, t.n_laps
    pos = res.pos[s]
    lap = np.broadcast_to(np.arange(1, L + 1)[None, :], (n, L))
    car = np.broadcast_to(np.arange(n)[:, None], (n, L))
    pit_in = res.pit[s]
    pit_out = np.zeros_like(pit_in)
    pit_out[:, 1:] = pit_in[:, :-1]
    status = np.array(["1", "6", "4"])[res.neutral[s][1:L + 1]]
    ok = pos > 0
    df = pd.DataFrame({"driver_id": np.array(t.drivers)[car[ok]], "lap_number": lap[ok], "position": pos[ok].astype(float),
                       "pit_in_s": np.where(pit_in[ok], 1.0, np.nan), "pit_out_s": np.where(pit_out[ok], 1.0, np.nan),
                       "track_status": np.broadcast_to(status[None, :], (n, L))[ok], "season": t.season})
    return df


def sim_events(t: RaceTruth, res: E.SimResult, k: int) -> tuple[list[dict], list[str]]:
    """Undercut events and lead-change categories from the first k simulations."""
    events, cats = [], []
    for s in range(min(k, res.finish.shape[0])):
        laps = sim_laps(t, res, s)
        events += [{"race_id": t.race_id, "season": t.season, **e} for e in race_flow.undercut_events(laps)]
        r = pd.DataFrame({"driver_id": t.drivers, "position": res.finish[s], "grid": t.slot})
        cats.append(race_flow.classify_lead(laps, r))
    return events, cats


# ------------------------------------------------------------------ levels

def run_l0(t: RaceTruth, par: Params, S: int, seed: int) -> E.SimResult:
    return E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed)


def run_l1(t: RaceTruth, pool: Pool, par: Params, S: int, seed: int) -> E.SimResult:
    rng = np.random.default_rng(seed + 7)
    s = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, rng)
    spec = E.RaceSpec(n_laps=t.n_laps, slot=t.slot, pace=t.pace, comp=s["comp"], stops=s["stops"], start_age=s["start_age"],
                      retire=s["retire"], neutral=s["neutral"], off=t.off, deg=t.deg, fuel=t.fuel, pit_loss=t.pit_loss, cover=True, warm=t.warm_s)
    return E.simulate(spec, par.for_circuit(t.circuit_id), S=S, seed=seed)


def replay(truths: list[RaceTruth], par: Params, S: int = S_RACE, s_events: int = S_EVENTS, progress: bool = True):
    rows, grip_rows, events, cats = [], [], [], []
    for i, t in enumerate(truths):
        pool = Pool.before(truths, t)
        seed = int(t.race_id)
        base = grid_baseline(pool, t.slot, t.finish) if len(pool) >= 5 else {}
        act_grip = C.actual_grip(t)
        for level in LEVELS:
            if level.startswith("L1") and len(pool) < MIN_POOL // 2:
                continue
            res = run_l0(t, par, S, seed) if level.startswith("L0") else run_l1(t, pool, par, S, seed)
            m = finish_metrics(res.finish, t.finish)
            pole = int(np.argmin(t.slot))
            sp, sc = C.sim_passes(res, t.n_laps)
            rp, rc = C.real_passes(t)
            if t.gap_ahead is not None and res.gap is not None:
                sclose, scars = C.sim_close(res, t.n_laps)
                rclose, rcars = C.real_close(t)
            else:
                sclose = scars = np.array([math.nan])
                rclose = rcars = math.nan
            rows.append({"race_id": t.race_id, "season": t.season, "split": split_of(t.season), "circuit_id": t.circuit_id,
                         "level": level, **m, **base, "sim_passes": float(sp.mean()), "sim_car_laps": float(sc.mean()),
                         "real_passes": rp, "real_car_laps": rc,
                         "sim_close": float(sclose.mean()), "sim_close_cars": float(scars.mean()),
                         "real_close": rclose, "real_close_cars": rcars,
                         "sim_lap1_move": float(C.sim_lap1_move(res, t.slot).mean()), "real_lap1_move": C.lap1_move(t),
                         "sim_pole_win": float((res.finish[:, pole] == 1).mean()), "pole_won": bool(t.finish[pole] == 1),
                         "sim_grid_finish": float(np.nanmean(spearman_masked(
                             np.broadcast_to(t.slot[None].astype(float), res.finish.shape), res.finish.astype(float),
                             np.ones(res.finish.shape, bool)))),
                         "actual_grid_finish": float(np.corrcoef(t.slot, t.finish)[0, 1])})
            sg = np.nanmean(C.sim_grip(res, t.slot, t.n_laps), axis=0)
            for f, a, s in zip(FRACTIONS, act_grip, sg):
                grip_rows.append({"race_id": t.race_id, "season": t.season, "split": split_of(t.season), "level": level,
                                  "fraction": f, "actual": a, "simulated": s})
            if s_events:
                ev, ct = sim_events(t, res, s_events)
                events += [{**e, "level": level} for e in ev]
                cats += [{"race_id": t.race_id, "season": t.season, "category": c, "level": level} for c in ct]
        if progress and (i + 1) % 20 == 0:
            print(f"  replayed {i + 1} of {len(truths)} races", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(grip_rows), pd.DataFrame(events), pd.DataFrame(cats)


# ------------------------------------------------------------------ tables

def summarise(races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, d0 in races.groupby("level"):
        for split, d in list(d0.groupby("split")) + [("all", d0)]:
            rows.append({"level": level, "split": split, "races": len(d),
                         "finish_order_correlation": d["spearman_expected"].mean(),
                         "single_run_correlation": d["spearman"].mean(),
                         "grid_only_rank_error": d["base_mean_rank_error"].mean() if "base_mean_rank_error" in d else math.nan,
                         "grid_only_correlation": d["base_spearman"].mean() if "base_spearman" in d else math.nan,
                         "mean_rank_error": d["mean_rank_error"].mean(),
                         "winner_called": d["winner_top1"].mean(),
                         "grid_only_winner_called": d["base_winner_top1"].mean() if "base_winner_top1" in d else math.nan,
                         "winner_logscore": d["winner_logscore"].mean(),
                         "grid_only_winner_logscore": d["base_winner_logscore"].mean() if "base_winner_logscore" in d else math.nan,
                         "brier_top3": d["brier_top3"].mean(),
                         "grid_only_brier_top3": d["base_brier_top3"].mean() if "base_brier_top3" in d else math.nan,
                         "brier_top10": d["brier_top10"].mean(),
                         "grid_only_brier_top10": d["base_brier_top10"].mean() if "base_brier_top10" in d else math.nan,
                         "sim_pole_win_rate": d["sim_pole_win"].mean(), "actual_pole_win_rate": d["pole_won"].mean()})
    return pd.DataFrame(rows)


def pass_table(races: pd.DataFrame) -> pd.DataFrame:
    """On-track passes per car-lap, real against simulated (everything known), by split and overall."""
    d0 = races[races["level"] == LEVELS[0]]
    rows = []
    for split, d in list(d0.groupby("split")) + [("all", d0)]:
        dc = d[d["real_close_cars"].notna()] if "real_close_cars" in d.columns else d.iloc[0:0]
        l1 = d.dropna(subset=["real_lap1_move"]) if "real_lap1_move" in d.columns else d.iloc[0:0]
        rows.append({"split": split, "races": len(d), "real_passes_per_car_lap": d["real_passes"].sum() / d["real_car_laps"].sum(),
                     "simulated_passes_per_car_lap": d["sim_passes"].sum() / d["sim_car_laps"].sum(),
                     "real_close_share": dc["real_close"].sum() / dc["real_close_cars"].sum() if len(dc) and dc["real_close_cars"].sum() else math.nan,
                     "simulated_close_share": dc["sim_close"].sum() / dc["sim_close_cars"].sum() if len(dc) and dc["sim_close_cars"].sum() else math.nan,
                     "real_lap1_places_moved": float(l1["real_lap1_move"].mean()) if len(l1) else math.nan,
                     "simulated_lap1_places_moved": float(l1["sim_lap1_move"].mean()) if len(l1) else math.nan})
    return pd.DataFrame(rows)


def circuit_table(races: pd.DataFrame, min_races: int = 3, level: str | None = None) -> pd.DataFrame:
    """Per circuit, the real and simulated correlation of grid and finishing order. With no `level`, both levels, stacked."""
    if level is None:
        parts = [circuit_table(races, min_races, lv).assign(level=lv) for lv in LEVELS if (races["level"] == lv).any()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    d = races[races["level"] == level]
    g = d.groupby("circuit_id")
    out = pd.DataFrame({"races": g.size(), "actual_grid_finish": g["actual_grid_finish"].mean(),
                        "simulated_grid_finish": g["sim_grid_finish"].mean()}).reset_index()
    out = out[out["races"] >= min_races]
    out["gap"] = out["simulated_grid_finish"] - out["actual_grid_finish"]
    return out.sort_values("gap").reset_index(drop=True)


def grip_curves(grip: pd.DataFrame) -> pd.DataFrame:
    g = grip.dropna().groupby(["level", "split", "fraction"])[["actual", "simulated"]].mean().reset_index()
    return g


def actual_targets(con, truth_ids: set[int]):
    """Stage 2 measures recomputed on exactly the races the simulator was run on."""
    laps, flags, results = C.load_inputs(con)
    laps = laps[laps["race_id"].isin(truth_ids)]
    flags = flags[flags["race_id"].isin(truth_ids)]
    events = race_flow.undercut_table(laps, flags)
    takeovers = race_flow.lead_takeovers(laps, results, flags)
    return events, takeovers


def takeover_compare(sim_cats: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """How the pole sitter lost the lead, real against simulated, per level when the simulated table has a `level` column."""
    if "level" in sim_cats.columns:
        parts = [takeover_compare(g.drop(columns="level"), actual).assign(level=lv) for lv, g in sim_cats.groupby("level")]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows = []
    for period in ("2018-21", "2022-now", "all"):
        def part(d):
            if period == "all":
                return d
            return d[d["season"] <= 2021] if period == "2018-21" else d[d["season"] >= 2022]
        a, s = part(actual), part(sim_cats)
        if a.empty or s.empty:
            continue
        for cat in [race_flow.POLE_WON] + race_flow.CATEGORIES:
            lost_a, lost_s = a[a["category"] != race_flow.POLE_WON], s[s["category"] != race_flow.POLE_WON]
            rows.append({"period": period, "category": cat,
                         "real_share_of_races": float((a["category"] == cat).mean()),
                         "simulated_share_of_races": float((s["category"] == cat).mean()),
                         "real_share_of_pole_losses": float((lost_a["category"] == cat).mean()) if cat != race_flow.POLE_WON and len(lost_a) else math.nan,
                         "simulated_share_of_pole_losses": float((lost_s["category"] == cat).mean()) if cat != race_flow.POLE_WON and len(lost_s) else math.nan})
    return pd.DataFrame(rows)


def undercut_compare(sim_events_df: pd.DataFrame, real_events: pd.DataFrame, reps: int = 500) -> pd.DataFrame:
    if "level" in sim_events_df.columns:
        parts = [undercut_compare(g.drop(columns="level"), real_events, reps).assign(level=lv) for lv, g in sim_events_df.groupby("level")]
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    real = race_flow.undercut_summary(real_events, reps=reps)
    sim = race_flow.undercut_summary(sim_events_df, reps=reps) if len(sim_events_df) else pd.DataFrame()
    if real.empty or sim.empty:
        return pd.DataFrame()
    keep = ["period", "scope", "chaser_ahead_if_first", "chaser_ahead_if_second", "undercut_advantage",
            "advantage_ci_low", "advantage_ci_high"]
    m = real[keep].merge(sim[keep], on=["period", "scope"], suffixes=("_real", "_simulated"))
    return m


# ---------------------------------------------------- where the errors are

STOP_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _first_stops(stops: np.ndarray, slot: np.ndarray, n_laps: int, include: np.ndarray):
    """stops (n, K). The first stop of each included car as a share of the race, the number of stops of each, and the gaps between the
    first stops (as a share of the race) of cars that started in neighbouring grid slots."""
    first = stops[:, 0].astype(float)
    frac = np.where(first < BIG, first / n_laps, np.nan)
    ns = (stops < BIG).sum(1)
    o = np.argsort(slot, kind="stable")
    f_o, inc_o = frac[o], include[o]
    both = np.isfinite(f_o[:-1]) & np.isfinite(f_o[1:]) & inc_o[:-1] & inc_o[1:]
    return frac[include], ns[include], np.abs(f_o[1:] - f_o[:-1])[both]


def stop_timing_check(truths: list[RaceTruth], S: int = 20, seed: int = 41) -> pd.DataFrame:
    """Real stops against the strategies the simulator draws when it samples them (level L1; the same earlier races it would draw from).
    Per split: stops per car, the share of cars that never stop, when the first stop comes (as a share of the race, quantiles) and how far
    apart the first stops of neighbours on the grid are. Cars that retired are left out of the real side; sampled strategies come
    before retirements are applied."""
    acc: dict = {}
    for i, t in enumerate(truths):
        pool = Pool.before(truths, t)
        if len(pool) < MIN_POOL // 2:
            continue
        real = _first_stops(t.stops, t.slot, t.n_laps, t.retire >= BIG)
        smp = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, np.random.default_rng(seed + i), retirements=False)
        sims = [_first_stops(smp["stops"][s], t.slot, t.n_laps, np.ones(t.n, bool)) for s in range(S)]
        samp = tuple(np.concatenate([x[k] for x in sims]) for k in range(3))
        for key in (split_of(t.season), "all"):
            a = acc.setdefault(key, {"real": [[], [], []], "sampled": [[], [], []]})
            for k in range(3):
                a["real"][k].append(real[k])
                a["sampled"][k].append(samp[k])
    rows = []
    for key, a in acc.items():
        stats = {}
        for who in ("real", "sampled"):
            frac, ns, diff = (np.concatenate(x) for x in a[who])
            f = frac[np.isfinite(frac)]
            stats[who] = {"stops per car": float(ns.mean()), "share of cars that never stop": float((ns == 0).mean()),
                          **{f"first stop, {int(q * 100)}th percentile (share of race)": float(np.quantile(f, q)) for q in STOP_QUANTILES},
                          "gap between neighbours' first stops (share of race)": float(diff.mean()) if len(diff) else math.nan}
        for name in stats["real"]:
            rows.append({"split": key, "statistic": name, "real": stats["real"][name], "sampled": stats["sampled"][name]})
    return pd.DataFrame(rows)


def position_check(truths: list[RaceTruth], par: Params, S: int = S_POSITION) -> pd.DataFrame:
    return C.position_check(truths, par, S=S)


def lap1_check(truths: list[RaceTruth], par: Params, S: int = S_POSITION) -> pd.DataFrame:
    """Places moved on lap 1 by grid-slot band, real against simulated (everything known), by split: whether the start shuffles the
    front of the grid as little as real starts do and the middle as much."""
    rows = []
    for key in ("train", "holdout", "earlier era", "all"):
        ts = truths if key == "all" else [t for t in truths if split_of(t.season) == key]
        if not ts:
            continue
        real, sim = C.real_lap1_bands(ts), C.sim_lap1_bands(ts, par, S=S)
        for j, (lo, hi) in enumerate(C.LAP1_BANDS):
            rows.append({"split": key, "slots": f"{lo}-{hi}" if hi < 99 else f"{lo}+",
                         "real_places_moved": real[j, 0] / real[j, 1] if real[j, 1] else math.nan,
                         "simulated_places_moved": sim[j, 0] / sim[j, 1] if sim[j, 1] else math.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts

def chart_levels(summary: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import BLUE, ORANGE, AXIS, TEXT, TEXT2, _frame, _plt
    d = summary[summary["split"].isin(["train", "holdout", "earlier era"])]
    if d.empty:
        return
    plt = _plt()
    fig, ax = _frame(plt, "How much of a finishing order does each kind of knowledge explain?",
                     "Rank correlation of simulated and real finishing order, mean over races. "
                     "Grid only = what the starting grid alone predicts.", size=(9.5, 5.2))
    splits = [s for s in ("earlier era", "train", "holdout") if s in set(d["split"])]
    x = np.arange(len(splits))
    w = 0.26
    series = [("Grid only", "grid_only_correlation", AXIS, LEVELS[1]),
              ("Pace known, rest sampled (L1)", "finish_order_correlation", BLUE, LEVELS[1]),
              ("Everything known (L0)", "finish_order_correlation", ORANGE, LEVELS[0])]
    for i, (label, col, colour, level) in enumerate(series):
        vals = [float(d[(d["split"] == s) & (d["level"] == level)][col].mean()) for s in splits]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.9, color=colour, label=label)
        for b, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=TEXT2)
    labels = {"earlier era": "2018-21 (earlier era)", "train": "2022-24 (settings fitted)", "holdout": "2025 on (held out)"}
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in splits])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path)
    plt.close(fig)


def chart_grip(curves: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import BLUE, ORANGE, AXIS, TEXT2, _frame, _plt
    d = curves[curves["split"].isin(["holdout"])]
    if d.empty:
        d = curves[curves["split"] == "train"]
    if d.empty:
        return
    plt = _plt()
    fig, ax = _frame(plt, "Does the grid dissolve at the real pace? (held-out races)",
                     "Rank correlation of grid and running order through the race. Real races against the simulator "
                     "with everything known (L0) and with strategies and safety cars sampled (L1).", size=(9.5, 5.2))
    real = d[d["level"] == LEVELS[0]]
    ax.plot(real["fraction"], real["actual"], color=TEXT2, lw=2.4, marker="o", ms=4, label="Real races")
    for level, colour, label in ((LEVELS[0], ORANGE, "Simulated, everything known (L0)"),
                                 (LEVELS[1], BLUE, "Simulated, rest sampled (L1)")):
        x = d[d["level"] == level]
        if len(x):
            ax.plot(x["fraction"], x["simulated"], color=colour, lw=1.9, ls="--", marker="s", ms=3.5, label=label)
    ax.set_xlabel("Share of the race distance")
    ax.set_ylabel("Grid vs running order")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def chart_undercut(cmp: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import AXIS, BLUE, SERIES, TEXT2, _frame, _plt
    scopes = ["stops 1-2 laps apart", "stops 3+ laps apart", "gap-standardised"]
    d = cmp[(cmp["period"] == "2022-now") & cmp["scope"].isin(scopes)]
    if d.empty:
        d = cmp[(cmp["period"] == "all") & cmp["scope"].isin(scopes)]
    if d.empty:
        return
    levels = [lv for lv in LEVELS if "level" in d.columns and (d["level"] == lv).any()] or [None]
    plt = _plt()
    fig, ax = _frame(plt, "Does the simulator reproduce the undercut?",
                     "Chance the chasing car ends ahead when it stops first, minus when it stops second, in points. "
                     "Real races with 95% range; simulated with real stops (L0) and sampled stops (L1).", size=(9.5, 5.0))
    colours = {LEVELS[0]: SERIES[2], LEVELS[1]: BLUE, None: BLUE}
    names = {LEVELS[0]: "Simulated, real stops (L0)", LEVELS[1]: "Simulated, sampled stops (L1)", None: "Simulated"}
    base = d if levels == [None] else d[d["level"] == levels[0]]
    y = np.arange(len(base))[::-1]
    for yi, (_, r) in zip(y, base.iterrows()):
        ax.plot([100 * r["advantage_ci_low_real"], 100 * r["advantage_ci_high_real"]], [yi + 0.18, yi + 0.18], color=AXIS, lw=2.4)
        ax.scatter([100 * r["undercut_advantage_real"]], [yi + 0.18], s=60, color=TEXT2, zorder=3, label="Real" if yi == y[0] else None)
    for k, lv in enumerate(levels):
        part = d if lv is None else d[d["level"] == lv]
        for yi, (_, r) in zip(y, part.iterrows()):
            ax.scatter([100 * r["undercut_advantage_simulated"]], [yi - 0.05 - 0.17 * k], s=60, color=colours[lv], zorder=3,
                       label=names[lv] if yi == y[0] else None)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(list(base["scope"]))
    ax.set_xlabel("Undercut advantage (percentage points)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.27)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- findings

def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def findings(summary: pd.DataFrame, circuits: pd.DataFrame, takeover: pd.DataFrame, undercut: pd.DataFrame,
             grip: pd.DataFrame, passes: pd.DataFrame | None = None, positions: pd.DataFrame | None = None,
             timing: pd.DataFrame | None = None, lap1: pd.DataFrame | None = None) -> list[str]:
    passes = passes if passes is not None else pd.DataFrame()
    L = []
    for level in LEVELS:
        d = summary[summary["level"] == level]
        if d.empty:
            continue
        L.append(f"{level.upper()}")
        for _, r in d.iterrows():
            L.append(f"  {r['split']} ({int(r['races'])} races): finishing-order correlation of the expected order {r['finish_order_correlation']:.2f}"
                     f" (grid only {r['grid_only_correlation']:.2f}; one simulated race {r['single_run_correlation']:.2f}); winner called {_pct(r['winner_called'])}"
                     f" (grid only {_pct(r['grid_only_winner_called'])}); winner log score {r['winner_logscore']:.2f}"
                     f" (grid only {r['grid_only_winner_logscore']:.2f}); podium Brier {r['brier_top3']:.3f}"
                     f" (grid only {r['grid_only_brier_top3']:.3f}); top-ten Brier {r['brier_top10']:.3f}"
                     f" (grid only {r['grid_only_brier_top10']:.3f}); pole wins {_pct(r['sim_pole_win_rate'])} simulated"
                     f" against {_pct(r['actual_pole_win_rate'])} real")
    L.append("  Lower is better for log score and Brier. The grid-only model is the benchmark to beat.")
    if len(passes):
        L.append("ON-TRACK PASSES per car-lap (real vs simulated, everything known): " + "; ".join(
            f"{r['split']} {r['real_passes_per_car_lap']:.4f} vs {r['simulated_passes_per_car_lap']:.4f}" for _, r in passes.iterrows()))
    if len(passes) and "real_close_share" in passes.columns and passes["real_close_share"].notna().any():
        L.append("CAR-LAPS WITHIN 1 s OF THE CAR AHEAD (real vs simulated, everything known): " + "; ".join(
            f"{r['split']} {r['real_close_share']:.1%} vs {r['simulated_close_share']:.1%}"
            for _, r in passes.iterrows() if np.isfinite(r["real_close_share"])))
    if len(passes) and "real_lap1_places_moved" in passes.columns and passes["real_lap1_places_moved"].notna().any():
        L.append("PLACES MOVED ON LAP 1 per car (real vs simulated): " + "; ".join(
            f"{r['split']} {r['real_lap1_places_moved']:.2f} vs {r['simulated_lap1_places_moved']:.2f}"
            for _, r in passes.iterrows() if np.isfinite(r["real_lap1_places_moved"])))
    if positions is not None and len(positions):
        for key in ("all", "holdout"):
            d = positions[positions["split"] == key]
            if d.empty:
                continue
            L.append(f"PASSES BY RUNNING POSITION per car-lap ({key}; real vs simulated, everything known): " + "; ".join(
                f"{r['position']} {r['real_passes_per_car_lap']:.4f} vs {r['simulated_passes_per_car_lap']:.4f}" for _, r in d.iterrows()))
            if d["real_close_share"].notna().any():
                L.append(f"CAR-LAPS WITHIN 1 s OF THE CAR AHEAD BY RUNNING POSITION ({key}): " + "; ".join(
                    f"{r['position']} {r['real_close_share']:.1%} vs {r['simulated_close_share']:.1%}" for _, r in d.iterrows()
                    if np.isfinite(r["real_close_share"])))
    if timing is not None and len(timing):
        d = timing[timing["split"] == "all"]
        if len(d):
            L.append("STOP TIMING (all races, real vs sampled strategies): " + "; ".join(
                f"{r['statistic']} {r['real']:.2f} vs {r['sampled']:.2f}" for _, r in d.iterrows()))
    if lap1 is not None and len(lap1):
        for split in ("all", "holdout"):
            d = lap1[lap1["split"] == split]
            if len(d):
                L.append(f"LAP 1 PLACES MOVED BY GRID SLOT ({split}; real vs simulated, everything known): " + "; ".join(
                    f"{r['slots']} {r['real_places_moved']:.2f} vs {r['simulated_places_moved']:.2f}" for _, r in d.iterrows()))
    if len(grip):
        d = grip[(grip["split"] == "holdout")]
        if d.empty:
            d = grip
        for level in LEVELS:
            x = d[d["level"] == level]
            if len(x):
                gap = (x["simulated"] - x["actual"]).abs()
                at = {f: x[x["fraction"] == f] for f in (0.0, 0.5, 0.95)}
                L.append(f"GRIP DECAY, {level.split(' ')[0]}: mean absolute gap to the real curve {gap.mean():.3f}; "
                         + ", ".join(f"at {int(f * 100) if f < 1 else f}% real {float(v['actual'].iloc[0]):.2f} simulated {float(v['simulated'].iloc[0]):.2f}"
                                     for f, v in at.items() if len(v)))
    if len(takeover):
        tk = takeover if "level" in takeover.columns else takeover.assign(level="")
        for lv, g in tk.groupby("level", sort=False):
            tag = lv.split(" ")[0] if lv else ""
            t = g[(g["period"] == "all") & (g["category"] != race_flow.POLE_WON)]
            p = g[(g["period"] == "all") & (g["category"] == race_flow.POLE_WON)]
            if len(p):
                L.append(f"HOW POLE IS LOST{', ' + tag if tag else ''} (all races): pole sitter wins {_pct(p['real_share_of_races'].iloc[0])} real, "
                         f"{_pct(p['simulated_share_of_races'].iloc[0])} simulated. Share of pole losses, real vs simulated: "
                         + "; ".join(f"{r['category'].lower()} {_pct(r['real_share_of_pole_losses'])} vs {_pct(r['simulated_share_of_pole_losses'])}"
                                     for _, r in t.iterrows()))
    if len(undercut):
        uc = undercut if "level" in undercut.columns else undercut.assign(level="")
        for lv, g in uc.groupby("level", sort=False):
            tag = lv.split(" ")[0] if lv else ""
            for per in ("2022-now", "all"):
                u = g[(g["period"] == per) & g["scope"].isin(["stops 1-2 laps apart", "stops 3+ laps apart", "gap-standardised"])]
                if len(u):
                    L.append(f"UNDERCUT{', ' + tag + (' (real stops)' if tag == 'L0' else ' (sampled stops)') if tag else ''} ({per}), real vs simulated advantage in points: " + "; ".join(
                        f"{r['scope']} {100 * r['undercut_advantage_real']:+.1f} ({100 * r['advantage_ci_low_real']:+.1f} to "
                        f"{100 * r['advantage_ci_high_real']:+.1f}) vs {100 * r['undercut_advantage_simulated']:+.1f}"
                        for _, r in u.iterrows()))
                    break
    if len(circuits):
        cc = circuits if "level" in circuits.columns else circuits.assign(level="")
        for lv, c in cc.groupby("level", sort=False):
            c = c.sort_values("gap").reset_index(drop=True)
            if len(c) < 6:
                continue
            tag = lv.split(" ")[0] if lv else ""
            rho = float(np.corrcoef(c["actual_grid_finish"], c["simulated_grid_finish"])[0, 1])
            lo, hi = c.iloc[:3], c.iloc[-3:]
            L.append(f"CIRCUITS{', ' + tag if tag else ''} ({len(c)} with 3+ races): across circuits the simulated and real grid-to-finish correlation agree at "
                     f"{rho:.2f}; spread across circuits (sd) real {c['actual_grid_finish'].std():.2f}, simulated "
                     f"{c['simulated_grid_finish'].std():.2f}; mean gap {c['gap'].mean():+.3f}. Simulator too predictable at "
                     + ", ".join(hi["circuit_id"]) + "; too chaotic at " + ", ".join(lo["circuit_id"]) + ".")
    return L


# --------------------------------------------------------------------- run

def run(con, out_dir: Path | None = None, cal_dir: Path | None = None, S: int = S_RACE, s_events: int = S_EVENTS,
        progress: bool = True) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sim_replay"
    out.mkdir(parents=True, exist_ok=True)
    truths, par = C.load_truths(cal_dir)
    races, grip, events, cats = replay(truths, par, S=S, s_events=s_events, progress=progress)
    summary = summarise(races)
    curves = grip_curves(grip)
    circuits = circuit_table(races)
    passes = pass_table(races)
    positions = position_check(truths, par)
    timing = stop_timing_check(truths)
    lap1 = lap1_check(truths, par)
    real_events, real_takeovers = actual_targets(con, {t.race_id for t in truths})
    takeover = takeover_compare(cats, real_takeovers) if len(cats) else pd.DataFrame()
    undercut = undercut_compare(events, real_events) if len(events) else pd.DataFrame()
    for name, t in [("replay_races", races), ("replay_summary", summary), ("grip_curves", curves), ("circuit_check", circuits),
                    ("lead_change_check", takeover), ("undercut_check", undercut), ("pass_check", passes),
                    ("position_check", positions), ("stop_timing_check", timing), ("lap1_check", lap1)]:
        t.to_csv(out / f"{name}.csv", index=False)
    chart_levels(summary, out / "replay_levels.png")
    chart_grip(curves, out / "replay_grip.png")
    if len(undercut):
        chart_undercut(undercut, out / "replay_undercut.png")
    lines = findings(summary, circuits, takeover, undercut, curves, passes, positions, timing, lap1)
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'sim_replay'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
