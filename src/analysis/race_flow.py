"""Stage 2, part 4b: what happens between the grid and the flag (FastF1 laps, 2018 on).

Run:  python -m src.analysis.race_flow
Writes tables (CSV), charts (PNG) and findings.txt to outputs/race_flow/.

Questions
  1. Grip decay: how fast does the grid order dissolve? Rank correlation between grid and running
     order after lap 1, 10%, 20% ... of the distance, among cars still running.
  2. How are races lost from pole? For every race the pole sitter did not win, when did the eventual
     winner take the lead for good, and what was going on: the start, a pit-stop cycle, a safety car or
     red flag, the leader retiring, or a plain pass on track?
  3. Undercut: when two cars are adjacent in the running order and one stops first, how often does the
     chaser end up in front, compared with when the leader stops first? Reported by how far apart the two
     stops were (stops 1-2 laps apart are the true undercut; stops 3+ laps apart mostly measure who was
     on the better strategy), plus a gap-standardised figure. Intervals resample whole races, because
     pairs from one race are not independent. (Observational: who stops first is a choice, so this is
     a rate, not a clean causal effect.)
  4. Neutralisations and rain: how pole win rate and grid grip change with a safety car, VSC or red flag,
     and in the wet.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import grid
from .grid import MIN_STARTERS, BLUE, ORANGE, SERIES, TEXT, TEXT2, AXIS, _frame, _plt, wilson
from .tyres import FIRST_SEASON, NEUTRAL_CODES, load_laps, load_rain_share, race_flags

FRACTIONS = tuple(round(0.1 * i, 1) for i in range(10)) + (0.95,)   # 0.0 means after lap 1; the last lap itself
# is not used: its positions are missing for many cars, so it would be a different, lead-lap-only sample
UNDERCUT_WINDOW = 10
CATEGORIES = ["Took the lead at the start", "Pit-stop cycle", "Safety car or red flag", "Leader retired",
              "On-track pass", "Unclear"]
POLE_WON = "Pole sitter won"


def _wide(g: pd.DataFrame, column: str) -> pd.DataFrame:
    """Lap by driver table of one column; a repeated (driver, lap) row, if any, is ignored."""
    g = g.drop_duplicates(["driver_id", "lap_number"])
    return g.pivot(index="lap_number", columns="driver_id", values=column)


def group_of(season: int, wet: bool) -> str:
    if wet:
        return "Wet races"
    return "Dry, 2018-21" if season <= 2021 else "Dry, 2022-now"


def _is_neutral(status) -> bool:
    return any(ch in NEUTRAL_CODES for ch in str(status or ""))


# ---------------------------------------------------------- 1. grip decay

def grip_decay(laps: pd.DataFrame, results: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    wet = flags.set_index("race_id")["wet"]
    grid_by_race = {rid: g.drop_duplicates("driver_id").set_index("driver_id")["grid"]
                    for rid, g in results.groupby("race_id")}
    rows = []
    for race_id, g in laps.groupby("race_id"):
        if race_id not in grid_by_race:
            continue
        pos = _wide(g, "position")
        total = int(pos.index.max())
        gr = grid_by_race[race_id]
        slot = gr.where(gr > 0, gr.max() + 1)
        season = int(g["season"].iloc[0])
        for f in FRACTIONS:
            lap = max(1, int(round(f * total)))
            if lap not in pos.index:
                continue
            row = pos.loc[lap].dropna()
            common = row.index.intersection(slot.index)
            if len(common) < MIN_STARTERS:
                continue
            rho = slot[common].corr(row[common], method="spearman")
            rows.append({"race_id": race_id, "season": season, "fraction": f, "lap": lap,
                         "n_running": len(common), "spearman": rho,
                         "group": group_of(season, bool(wet.get(race_id, False)))})
    return pd.DataFrame(rows)


def grip_summary(decay: pd.DataFrame) -> pd.DataFrame:
    if decay.empty:
        return pd.DataFrame()
    g = decay.groupby(["group", "fraction"])
    return g.agg(races=("race_id", "nunique"), mean_spearman=("spearman", "mean"),
                 mean_running=("n_running", "mean")).reset_index()


# ------------------------------------------------------ 2. lead takeover

def classify_lead(race_laps: pd.DataFrame, res: pd.DataFrame) -> str:
    """Why did the winner end up in front of the pole sitter? One category per race."""
    win = res[res["position"] == 1]
    if win.empty:
        return "Unclear"
    winner = win["driver_id"].iloc[0]
    pole = res.loc[res["grid"] == 1, "driver_id"]
    if len(pole) and pole.iloc[0] == winner:
        return POLE_WON
    pos = _wide(race_laps, "position")
    if winner not in pos.columns:
        return "Unclear"
    is_leader = pos.eq(1)
    leader = is_leader.idxmax(axis=1).where(is_leader.any(axis=1))
    last = int(leader.index.max())
    if leader.loc[last] != winner:
        return "Unclear"            # for example a post-race penalty changed the order
    t = last
    while (t - 1) in leader.index and leader.loc[t - 1] == winner:
        t -= 1
    if t == int(leader.index.min()):
        return "Took the lead at the start"
    prev = leader.get(t - 1)
    if prev is None or pd.isna(prev):
        return "Unclear"
    window = race_laps[race_laps["driver_id"].isin([winner, prev]) & race_laps["lap_number"].between(t - 1, t + 1)]
    prev_last = int(race_laps.loc[race_laps["driver_id"] == prev, "lap_number"].max())
    if prev_last <= t:
        return "Leader retired"
    if window["track_status"].map(_is_neutral).any():
        return "Safety car or red flag"
    if window["pit_in_s"].notna().any() or window["pit_out_s"].notna().any():
        return "Pit-stop cycle"
    return "On-track pass"


def lead_takeovers(laps: pd.DataFrame, results: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    wet = flags.set_index("race_id")["wet"]
    res_by = {rid: g for rid, g in results.groupby("race_id")}
    rows = []
    for race_id, g in laps.groupby("race_id"):
        if race_id not in res_by:
            continue
        season = int(g["season"].iloc[0])
        rows.append({"race_id": race_id, "season": season, "wet": bool(wet.get(race_id, False)),
                     "category": classify_lead(g, res_by[race_id])})
    return pd.DataFrame(rows)


def takeover_summary(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    t = t.assign(period=np.where(t["season"] <= 2021, "2018-21", "2022-now"))
    rows = []
    for period, d in list(t.groupby("period")) + [("all", t)]:
        n = len(d)
        pole = int((d["category"] == POLE_WON).sum())
        lost = d[d["category"] != POLE_WON]
        for cat in [POLE_WON] + CATEGORIES:
            k = int((d["category"] == cat).sum())
            rows.append({"period": period, "category": cat, "races": k, "share_of_all_races": k / n,
                         "share_of_pole_losses": (k / len(lost)) if cat != POLE_WON and len(lost) else math.nan,
                         "total_races": n, "pole_win_rate": pole / n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- 3. undercut

def undercut_events(race_laps: pd.DataFrame, window: int = UNDERCUT_WINDOW) -> list[dict]:
    pos = _wide(race_laps, "position")
    total = int(pos.index.max())
    pit_laps = {d: sorted(int(x) for x in g.loc[g["pit_in_s"].notna(), "lap_number"] if x < total)
                for d, g in race_laps.groupby("driver_id")}
    neutral = set(int(l) for l in race_laps.loc[race_laps["track_status"].map(_is_neutral), "lap_number"])
    events = []
    for d, plist in pit_laps.items():
        for p in plist:
            if (p - 1) not in pos.index or pd.isna(pos.at[p - 1, d]):
                continue
            rank = pos.at[p - 1, d]
            for role, other_rank in (("chaser_first", rank - 1), ("leader_first", rank + 1)):
                if other_rank < 1:
                    continue
                match = pos.loc[p - 1][pos.loc[p - 1] == other_rank].index
                if len(match) != 1:
                    continue
                o = match[0]
                o_pits = pit_laps.get(o, [])
                if any(p - 3 <= x <= p for x in o_pits):
                    continue
                later = [x for x in o_pits if p < x <= p + window]
                if not later:
                    continue
                p2 = later[0]
                q = p2 + 2
                if q > total or q not in pos.index or pd.isna(pos.at[q, d]) or pd.isna(pos.at[q, o]):
                    continue
                if any(l in neutral for l in range(p - 1, q + 1)):
                    continue
                if any(p < x <= q for x in plist) or any(p2 < x <= q for x in o_pits):
                    continue
                chaser, leader = (d, o) if role == "chaser_first" else (o, d)
                events.append({"role": role, "lap": p, "gap_laps": p2 - p,
                               "chaser_ahead": bool(pos.at[q, chaser] < pos.at[q, leader])})
    return events


def undercut_table(laps: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    wet = flags.set_index("race_id")["wet"]
    rows = []
    for race_id, g in laps.groupby("race_id"):
        if bool(wet.get(race_id, False)):
            continue
        season = int(g["season"].iloc[0])
        for e in undercut_events(g):
            rows.append({"race_id": race_id, "season": season, **e})
    return pd.DataFrame(rows)


SCOPES = ("all stops", "stops 1-2 laps apart", "stops 3+ laps apart", "gap-standardised")
BOOT_REPS = 2000


def _count_tensor(events: pd.DataFrame, window: int = UNDERCUT_WINDOW):
    """races x gap x [n_chaser_first, k_chaser_first_ahead, n_leader_first, k_leader_first_ahead]."""
    key = events["race_id"] if "race_id" in events.columns else pd.Series(np.arange(len(events)), index=events.index)
    codes, _ = pd.factorize(key)
    out = np.zeros((codes.max() + 1, window, 4))
    gap = np.clip(events["gap_laps"].to_numpy(dtype=int) if "gap_laps" in events.columns
                  else np.full(len(events), 1), 1, window) - 1
    first = (events["role"] == "chaser_first").to_numpy()
    ahead = events["chaser_ahead"].to_numpy(dtype=float)
    base = np.where(first, 0, 2)
    np.add.at(out, (codes, gap, base), 1.0)
    np.add.at(out, (codes, gap, base + 1), ahead)
    return out


def _advantage(S: np.ndarray, scope: str) -> np.ndarray:
    """S: (..., gaps, 4) summed over races. Returns the chaser-first minus leader-first rate."""
    with np.errstate(all="ignore"):
        if scope == "gap-standardised":
            pa = S[..., 1] / S[..., 0]
            pb = S[..., 3] / S[..., 2]
            w = np.where((S[..., 0] > 0) & (S[..., 2] > 0), S[..., 0] + S[..., 2], 0.0)
            d = np.where(w > 0, pa - pb, 0.0)
            return (w * d).sum(-1) / w.sum(-1)
        lo, hi = {"all stops": (0, None), "stops 1-2 laps apart": (0, 2), "stops 3+ laps apart": (2, None)}[scope]
        T = S[..., lo:hi, :].sum(-2)
        return T[..., 1] / T[..., 0] - T[..., 3] / T[..., 2]


def _boot(tensor: np.ndarray, scope: str, reps: int, rng) -> np.ndarray:
    R = tensor.shape[0]
    out = []
    for start in range(0, reps, 250):
        m = min(250, reps - start)
        idx = rng.integers(0, R, size=(m, R))
        out.append(_advantage(tensor[idx].sum(1), scope))
    return np.concatenate(out)


def undercut_summary(events: pd.DataFrame, reps: int = BOOT_REPS, seed: int = 11) -> pd.DataFrame:
    """Undercut advantage by how far apart the two stops were, with race-clustered intervals."""
    if events.empty:
        return pd.DataFrame()
    events = events.assign(period=np.where(events["season"] <= 2021, "2018-21", "2022-now"))
    rng = np.random.default_rng(seed)
    rows, tensors, boots = [], {}, {}
    for period, d in [("2018-21", events[events["period"] == "2018-21"]),
                      ("2022-now", events[events["period"] == "2022-now"]), ("all", events)]:
        if d.empty:
            continue
        T = _count_tensor(d)
        tensors[period] = T
        total = T.sum(0)
        for scope in SCOPES:
            lo, hi = {"all stops": (0, None), "stops 1-2 laps apart": (0, 2), "stops 3+ laps apart": (2, None),
                      "gap-standardised": (0, None)}[scope]
            part = total[lo:hi].sum(0)
            if part[0] == 0 or part[2] == 0:
                continue
            b = _boot(T, scope, reps, rng)
            boots[(period, scope)] = b
            adv = float(_advantage(total, scope))
            rows.append({"period": period, "scope": scope, "chaser_stops_first": int(part[0]),
                         "chaser_ahead_if_first": part[1] / part[0], "leader_stops_first": int(part[2]),
                         "chaser_ahead_if_second": part[3] / part[2], "undercut_advantage": adv,
                         "advantage_ci_low": float(np.nanpercentile(b, 2.5)),
                         "advantage_ci_high": float(np.nanpercentile(b, 97.5)), "p_value": math.nan})
    if "2018-21" in tensors and "2022-now" in tensors:
        for scope in SCOPES:
            ba, bb = boots.get(("2018-21", scope)), boots.get(("2022-now", scope))
            if ba is None or bb is None:
                continue
            diff = bb - ba
            est = float(_advantage(tensors["2022-now"].sum(0), scope) - _advantage(tensors["2018-21"].sum(0), scope))
            p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            rows.append({"period": "change (2022-now minus 2018-21)", "scope": scope,
                         "chaser_stops_first": math.nan, "chaser_ahead_if_first": math.nan,
                         "leader_stops_first": math.nan, "chaser_ahead_if_second": math.nan,
                         "undercut_advantage": est, "advantage_ci_low": float(np.nanpercentile(diff, 2.5)),
                         "advantage_ci_high": float(np.nanpercentile(diff, 97.5)), "p_value": float(p)})
    return pd.DataFrame(rows)


# ------------------------------------- shifts between periods (tests)

def takeover_shift_tests(raw: pd.DataFrame, reps: int = 5000, seed: int = 5) -> pd.DataFrame:
    """Did the way pole is lost change between 2018-21 and 2022-now? Exact tests, no multiple-test fix:
    read the p-values as a set, expecting about one in twenty to be small by luck."""
    from scipy.stats import fisher_exact
    if raw.empty:
        return pd.DataFrame()
    t = raw.assign(late=raw["season"] > 2021)
    rows = []

    def two_by_two(label, flag):
        a, b = t[~t["late"]], t[t["late"]]
        ka, kb = int(flag[~t["late"]].sum()), int(flag[t["late"]].sum())
        if len(a) == 0 or len(b) == 0:
            return
        p = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
        rows.append({"test": label, "share_2018_21": ka / len(a), "share_2022_now": kb / len(b),
                     "races_2018_21": len(a), "races_2022_now": len(b), "p_value": float(p)})

    two_by_two("Pole sitter won", t["category"] == POLE_WON)
    for cat in CATEGORIES:
        two_by_two(f"Share of all races: {cat}", t["category"] == cat)
    lost = t[t["category"] != POLE_WON]
    if lost["late"].nunique() == 2:
        tab = pd.crosstab(lost["category"], lost["late"]).reindex(CATEGORIES, fill_value=0)
        obs = _chi2(tab.to_numpy(dtype=float))
        rng = np.random.default_rng(seed)
        cats, late = lost["category"].to_numpy(), lost["late"].to_numpy()
        hits = 0
        for _ in range(reps):
            sh = rng.permutation(late)
            tb = pd.crosstab(pd.Categorical(cats, CATEGORIES), sh).to_numpy(dtype=float)
            hits += _chi2(tb) >= obs - 1e-12
        rows.append({"test": "Mix of pole losses (all categories together)", "share_2018_21": math.nan,
                     "share_2022_now": math.nan, "races_2018_21": int((~lost["late"]).sum()),
                     "races_2022_now": int(lost["late"].sum()), "p_value": (hits + 1) / (reps + 1)})
    return pd.DataFrame(rows)


def _chi2(tab: np.ndarray) -> float:
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    with np.errstate(all="ignore"):
        return float(np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan)))


def grip_recovery(decay: pd.DataFrame, reps: int = BOOT_REPS, seed: int = 17) -> pd.DataFrame:
    """Paired within-race changes in grid grip: the early dip (lap 1 to 40%) and the late recovery
    (40% to 95%), per group, plus the dry 2022-now minus 2018-21 gap at three points. Race-level bootstrap."""
    if decay.empty:
        return pd.DataFrame()
    w = decay.pivot_table(index=["race_id", "group"], columns="fraction", values="spearman").reset_index()
    need = [0.0, 0.4, 0.95]
    if not all(c in w.columns for c in need):
        return pd.DataFrame()
    w = w.dropna(subset=need)
    rng = np.random.default_rng(seed)

    def boot_mean(x):
        idx = rng.integers(0, len(x), size=(reps, len(x)))
        m = x[idx].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    rows = []
    for g in ("Dry, 2018-21", "Dry, 2022-now", "Wet races"):
        d = w[w["group"] == g]
        if len(d) < 3:
            continue
        for label, x in (("dip: lap 1 to 40% of distance", (d[0.4] - d[0.0]).to_numpy()),
                         ("recovery: 40% to 95% of distance", (d[0.95] - d[0.4]).to_numpy())):
            lo, hi = boot_mean(x)
            rows.append({"comparison": g, "measure": label, "races": len(d), "estimate": float(x.mean()),
                         "ci_low": lo, "ci_high": hi, "p_value": math.nan})
    a, b = w[w["group"] == "Dry, 2018-21"], w[w["group"] == "Dry, 2022-now"]
    if len(a) >= 3 and len(b) >= 3:
        for f in need:
            xa, xb = a[f].to_numpy(), b[f].to_numpy()
            ia = rng.integers(0, len(xa), size=(reps, len(xa)))
            ib = rng.integers(0, len(xb), size=(reps, len(xb)))
            diff = xb[ib].mean(1) - xa[ia].mean(1)
            p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            rows.append({"comparison": "Dry 2022-now minus 2018-21", "measure": f"grid correlation at {f:.0%} of distance",
                         "races": len(a) + len(b), "estimate": float(xb.mean() - xa.mean()),
                         "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5)),
                         "p_value": float(p)})
    return pd.DataFrame(rows)


# --------------------------------------------- 4. neutralisation and rain

BORDERLINE_LABEL = "Wet, of which borderline (under 100 wet-tyre laps)"


def condition_of(neutral: bool, wet: bool) -> str:
    if wet:
        return "Wet"
    return "Dry, with safety car, VSC or red flag" if neutral else "Dry, green throughout"


def conditions_table(results: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    from . import grid_checks
    df = grid.prepare(results)
    rel = grid_checks.race_reliability(df)
    f = flags.set_index("race_id")
    rel["condition"] = [condition_of(bool(f.at[r, "neutral"]), bool(f.at[r, "wet"])) if r in f.index else "Unknown"
                        for r in rel["race_id"]]
    border = f["wet_borderline"] if "wet_borderline" in f.columns else pd.Series(False, index=f.index)
    rel["borderline"] = [bool(border.get(r, False)) for r in rel["race_id"]]
    # a race with no lap data has no condition; it is left out rather than shown as its own row
    groups = [(c, d) for c, d in rel[rel["has_pole"] & (rel["condition"] != "Unknown")].groupby("condition")]
    sub = rel[rel["has_pole"] & rel["borderline"] & (rel["condition"] == "Wet")]
    if len(sub):
        groups.append((BORDERLINE_LABEL, sub))
    rows = []
    for cond, d in groups:
        n, won = len(d), int(d["pole_won"].sum())
        lo, hi = wilson(won, n)
        fin = int(d["pole_finished"].sum())
        rows.append({"condition": cond, "races": n, "pole_win_rate": won / n, "pole_ci_low": lo, "pole_ci_high": hi,
                     "pole_finish_rate": fin / n, "win_given_finish": won / fin if fin else math.nan,
                     "spearman_all_starters": d["spearman_all"].mean(),
                     "spearman_finishers_only": d["spearman_finishers"].mean()})
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


# ---------------------------------------------------------------- charts

def chart_grip(summary: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    order = ["Dry, 2018-21", "Dry, 2022-now", "Wet races"]
    groups = [g for g in order if g in set(summary["group"])]
    ref = summary[summary["group"] == (groups[0] if not (summary["group"] == "Dry, 2022-now").any() else "Dry, 2022-now")]
    ref = ref.sort_values("fraction")
    if len(ref) >= 2:
        low = ref.loc[ref["mean_spearman"].idxmin()]
        sub = (f"Cars still running, {ref['group'].iloc[0]}: {ref['mean_spearman'].iloc[0]:.2f} after lap 1, "
               f"low of {low['mean_spearman']:.2f} at {low['fraction']:.0%} of the distance, {ref['mean_spearman'].iloc[-1]:.2f} near the end (95% of the distance). "
               "Late rise not clear since 2022.")
    else:
        sub = "Rank correlation between grid order and running order among cars still running."
    fig, ax = _frame(plt, "How fast the grid order dissolves during a race", sub)
    for i, g in enumerate(groups):
        d = summary[summary["group"] == g].sort_values("fraction")
        ax.plot(d["fraction"], d["mean_spearman"], color=SERIES[i], lw=2.2, marker="o", ms=4,
                label=f"{g} ({int(d['races'].max())} races)", zorder=3)
    ax.set_xlabel("Share of race distance (0% = end of lap 1)")
    ax.set_ylabel("Grid vs running order correlation")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(min(0.2, float(summary["mean_spearman"].min()) - 0.05), 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)


def chart_takeover(summary: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    periods = [p for p in ("2018-21", "2022-now") if p in set(summary["period"])]
    pole = summary[(summary["category"] == POLE_WON) & (summary["period"] == "all")]
    pole_txt = f"The pole sitter won {pole['share_of_all_races'].iloc[0]:.0%} of races. " if len(pole) else ""
    fig, ax = _frame(plt, "How races are lost from pole",
                     pole_txt + "Bars split the rest by how the winner took the lead for good. "
                     "Period differences are within chance.")
    y = np.arange(len(CATEGORIES))
    h = 0.36
    for i, p in enumerate(periods):
        d = summary[summary["period"] == p].set_index("category")
        vals = [d.at[c, "share_of_pole_losses"] if c in d.index else 0.0 for c in CATEGORIES]
        ns = [int(d.at[c, "races"]) if c in d.index else 0 for c in CATEGORIES]
        pos = y + (i - (len(periods) - 1) / 2) * (h + 0.04)
        ax.barh(pos, vals, height=h, color=SERIES[i], label=p, zorder=3)
        for yi, v, n in zip(pos, vals, ns):
            ax.text(v + 0.008, yi, f"{v:.0%} ({n})", va="center", fontsize=8.5, color=TEXT)
    fig.subplots_adjust(left=0.27)
    ax.set_yticks(y)
    ax.set_yticklabels(CATEGORIES)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, max(0.5, float(summary["share_of_pole_losses"].max(skipna=True)) + 0.15))
    ax.set_xlabel("Share of races the pole sitter did not win (number of races in brackets)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------- findings

def _p(p) -> str:
    """p-value for text: never print a rounded zero."""
    return "< 0.005" if p < 0.005 else f"= {p:.2f}"


def _sig(p) -> str:
    return "not significant" if (p is None or math.isnan(p) or p >= 0.05) else "significant"


def findings(summary, takeovers, undercut, conditions, recovery=None, shifts=None) -> list[str]:
    L: list[str] = []
    if len(summary):
        L.append("GRIP DECAY (grid vs running order, cars still running)")
        for g in ("Dry, 2018-21", "Dry, 2022-now", "Wet races"):
            d = summary[summary["group"] == g].sort_values("fraction")
            if len(d) >= 2:
                mid = d.iloc[(d["fraction"] - 0.5).abs().argmin()]
                L.append(f"  {g} ({int(d['races'].max())} races): {d['mean_spearman'].iloc[0]:.2f} after lap 1, "
                         f"{mid['mean_spearman']:.2f} at half distance, {d['mean_spearman'].iloc[-1]:.2f} near the end (95%)")
    if recovery is not None and len(recovery):
        L.append("  Does grip come back late in the race? (paired within each race; 95% interval)")
        for _, r in recovery[recovery["p_value"].isna()].iterrows():
            L.append(f"    {r['comparison']}, {r['measure']}: {r['estimate']:+.3f} ({r['ci_low']:+.3f} to {r['ci_high']:+.3f})")
        for _, r in recovery[recovery["p_value"].notna()].iterrows():
            L.append(f"    {r['comparison']}, {r['measure']}: {r['estimate']:+.3f} "
                     f"({r['ci_low']:+.3f} to {r['ci_high']:+.3f}), p {_p(r['p_value'])}, {_sig(r['p_value'])}")
        L.append("  An interval that includes zero means no evidence of recovery or decay in that group, not evidence of none.")
    if len(takeovers):
        allp = takeovers[takeovers["period"] == "all"]
        L.append(f"HOW THE WINNER BEAT THE POLE SITTER ({int(allp['total_races'].iloc[0])} races since {FIRST_SEASON}; "
                 f"pole sitter won {allp['pole_win_rate'].iloc[0]:.0%})")
        for _, r in allp[allp["category"] != POLE_WON].sort_values("races", ascending=False).iterrows():
            n = int(r['races'])
            L.append(f"  {r['category']}: {n} race{'s' if n != 1 else ''} ({r['share_of_pole_losses']:.0%} of pole losses)")
    if shifts is not None and len(shifts):
        small = shifts[shifts["p_value"] < 0.05]
        L.append(f"  Did the mix change between 2018-21 and 2022-now? {len(small)} of {len(shifts)} tests have p < 0.05 "
                 f"(about {len(shifts) * 0.05:.1f} expected by luck). Any period difference in the shares is not established.")
        for _, r in small.iterrows():
            L.append(f"    {r['test']}: p = {r['p_value']:.3f}")
    if len(undercut):
        L.append("UNDERCUT (cars adjacent in running order, green flag, other car stops within 10 laps; "
                 "intervals resample whole races)")
        for _, r in undercut.iterrows():
            scope = r.get("scope", "all stops")
            if str(r["period"]).startswith("change"):
                L.append(f"  {r['period']}, {scope}: {r['undercut_advantage']:+.0%} "
                         f"({r['advantage_ci_low']:+.0%} to {r['advantage_ci_high']:+.0%}), p {_p(r['p_value'])}")
                continue
            L.append(f"  {r['period']}, {scope}: chaser ends ahead {r['chaser_ahead_if_first']:.0%} when it stops first "
                     f"({int(r['chaser_stops_first'])} pairs) against {r['chaser_ahead_if_second']:.0%} when the leader stops first "
                     f"({int(r['leader_stops_first'])}); advantage {r['undercut_advantage']:+.0%} "
                     f"({r['advantage_ci_low']:+.0%} to {r['advantage_ci_high']:+.0%})")
        L.append("  Stops 1-2 laps apart is the real undercut. Stops 3+ laps apart mostly reflect who was on the better "
                 "strategy, so the 'all stops' figure overstates the undercut. The gap-standardised row compares like with like.")
    if len(conditions):
        L.append("CONDITIONS (races since 2018)")
        for _, r in conditions.iterrows():
            L.append(f"  {r['condition']}: {int(r['races'])} races; pole won {r['pole_win_rate']:.0%} "
                     f"({r['pole_ci_low']:.0%} to {r['pole_ci_high']:.0%}); grid correlation "
                     f"{r['spearman_all_starters']:.2f} all starters, {r['spearman_finishers_only']:.2f} finishers only")
    return L


def lap_coverage(results: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Races in the results since 2018 with no lap data: these drop out of every lap-based table."""
    have = set(laps["race_id"].unique())
    races = results[["race_id", "season"]].drop_duplicates()
    missing = races[~races["race_id"].isin(have)]
    return missing.sort_values("race_id").reset_index(drop=True)


def run(con, out_dir: Path | None = None) -> list[str]:
    from . import grid_checks
    out = Path(out_dir) if out_dir else config.outputs_dir() / "race_flow"
    out.mkdir(parents=True, exist_ok=True)

    laps = load_laps(con)
    laps = laps[laps["season"] >= FIRST_SEASON]
    flags = race_flags(laps, load_rain_share(con))
    results = grid_checks.load(con)
    results = results[results["season"] >= FIRST_SEASON]

    decay = grip_decay(laps, results, flags)
    grip = grip_summary(decay)
    takeovers_raw = lead_takeovers(laps, results, flags)
    takeovers = takeover_summary(takeovers_raw)
    events = undercut_table(laps, flags)
    undercut = undercut_summary(events)
    conditions = conditions_table(results, flags)
    recovery = grip_recovery(decay)
    shifts = takeover_shift_tests(takeovers_raw)

    for name, table in [("grip_decay_by_race", decay), ("grip_decay", grip), ("grip_recovery", recovery),
                        ("lead_takeover_shift_tests", shifts), ("lead_takeover_by_race", takeovers_raw),
                        ("lead_takeover", takeovers), ("undercut_events", events), ("undercut", undercut),
                        ("conditions", conditions)]:
        table.to_csv(out / f"{name}.csv", index=False)

    if len(grip):
        chart_grip(grip, out / "grip_decay.png")
    if len(takeovers):
        chart_takeover(takeovers, out / "how_pole_is_lost.png")

    lines = findings(grip, takeovers, undercut, conditions, recovery, shifts)
    missing = lap_coverage(results, laps)
    missing.to_csv(out / "races_without_laps.csv", index=False)
    total = results["race_id"].nunique()
    lines.insert(0, f"LAP-DATA COVERAGE: {total - len(missing)} of {total} races since {FIRST_SEASON} have lap data"
                    + (f"; without: {', '.join(str(int(r)) for r in missing['race_id'])} (they drop out of the lap-based tables below, "
                       "so pole-win rates here can differ slightly from the grid tables)" if len(missing) else ""))
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'race_flow'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
