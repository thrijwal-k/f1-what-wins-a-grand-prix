"""Stage 2, part 5: has what wins a Grand Prix changed, and do rule changes move it?

Run:  python -m src.analysis.trends
Writes tables (CSV), charts (PNG) and findings.txt to outputs/trends_and_rules/.

What it does
  1. Six season-level measures of "what decides a race" for every season since 1950:
       pole sitter wins, grid-vs-finish correlation, share of cars classified (reliability),
       top team's share of wins (dominance), share of races won from 4th or lower, and
       how many different drivers won (variety).
  2. A look at the big rule changes. For each one, the change in each measure between the
     seasons just before and just after, compared with the same change at years when no
     rule changed ("placebo breaks"). A rule change only counts as a real shift if it is
     unusually large next to the ordinary year-to-year drift.

Honest limits (also written into findings.txt)
  - Rule changes are not randomly assigned. Several coincide with one team getting ahead
    (Mercedes in 2014, Red Bull in 2022), so a shift can be the rules or the team. The test
    can say "unusual", not "caused by".
  - About thirty comparisons are made, so one or two will look unusual by luck. The table
    gives false-discovery-adjusted q-values; use those, not the raw p-values.
  - The latest season is still running, so a rule change in it is shown but not tested.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import grid, grid_checks
from .grid import AXIS, BLUE, GRID, ORANGE, SURFACE, TEXT, TEXT2, _plt

# (first season under the new rules, short label)
RULES = [
    (1994, "1994: driver aids banned, refuelling back"),
    (2009, "2009-11: slicks and new aero, refuelling ban, DRS, Pirelli"),
    (2014, "2014: hybrid turbo power units"),
    (2017, "2017: wider, faster cars"),
    (2022, "2022: ground-effect cars"),
    (2026, "2026: new power units and active aero"),
]
WINDOWS = (2, 3, 4)
MAIN_WINDOW = 3
MIN_RACES_SEASON = 5
UPSET_SLOT = 4
Q_ALPHA = 0.10
MODERN_FROM = 1975          # second benchmark: placebo breaks whose 'before' window starts here or later
P_RANK_MAX = 0.05
Z_MODERN_MIN = 2.0

METRICS = [
    ("pole_win_rate", "Pole sitter wins", "share of races"),
    ("grid_corr", "Grid order vs finishing order", "rank correlation"),
    ("finish_rate", "Cars classified at the flag", "share of starters"),
    ("top_team_win_share", "Top team's share of wins", "share of races"),
    ("upset_rate", f"Winner started {UPSET_SLOT}th or lower", "share of races"),
    ("winner_variety", "Different race winners", "winners per race"),
]
METRIC_KEYS = [m[0] for m in METRICS]


# ------------------------------------------------------------------ season measures

def race_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per race: grid.race_summary plus the share classified and who won."""
    races = grid.race_summary(df)
    win = df[df["position"] == 1].drop_duplicates("race_id").set_index("race_id")
    fin = df.groupby("race_id")["classified"].mean().rename("finish_rate")
    out = races.merge(fin, left_on="race_id", right_index=True, how="left")
    out["winner_team"] = out["race_id"].map(win["constructor_id"]) if "constructor_id" in win else None
    out["winner_driver"] = out["race_id"].map(win["driver_id"])
    return out


def season_table(races: pd.DataFrame, min_races: int = MIN_RACES_SEASON) -> pd.DataFrame:
    rows = []
    for season, g in races.groupby("season"):
        if len(g) < min_races:
            continue
        pole = g[g["has_pole"]]
        wins_by_team = g["winner_team"].value_counts() if g["winner_team"].notna().any() else pd.Series(dtype=float)
        grid_known = g["winner_grid"].dropna()
        rows.append({
            "season": int(season), "races": len(g),
            "pole_win_rate": float(pole["pole_won"].mean()) if len(pole) else np.nan,
            "grid_corr": float(g["spearman"].mean()),
            "finish_rate": float(g["finish_rate"].mean()),
            "top_team_win_share": float(wins_by_team.iloc[0] / len(g)) if len(wins_by_team) else np.nan,
            "upset_rate": float((grid_known >= UPSET_SLOT).mean()) if len(grid_known) else np.nan,
            "winner_variety": float(g["winner_driver"].nunique() / len(g)),
        })
    cols = ["season", "races"] + METRIC_KEYS
    return pd.DataFrame(rows, columns=cols).sort_values("season").reset_index(drop=True)


# ------------------------------------------------------------------ rule-change test

def _window_mean(s: pd.Series, first: int, last: int) -> tuple[float, int]:
    vals = s.reindex(range(first, last + 1)).dropna()
    return (float(vals.mean()) if len(vals) else np.nan), len(vals)


def _placebo_breaks(seasons: pd.Index, k: int, rule_years: list[int]) -> list[int]:
    have = set(int(s) for s in seasons)
    out = []
    for s in sorted(have):
        pre, post = range(s - k, s), range(s, s + k)
        if not all(y in have for y in list(pre) + list(post)):
            continue
        if any(s - k <= q <= s + k - 1 for q in rule_years):  # a real rule change inside the windows
            continue
        out.append(s)
    return out


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-adjusted p-values; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order] * len(vals) / (np.arange(len(vals)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(vals)
    adj[order] = np.minimum(ranked, 1.0)
    q[ok] = adj
    return q


def _normal_p(z: float) -> float:
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2)))


def rule_change_tests(seasons: pd.DataFrame, rules=RULES, windows=WINDOWS, main_window: int = MAIN_WINDOW,
                      metrics=METRIC_KEYS, modern_from: int = MODERN_FROM) -> pd.DataFrame:
    """Change across each rule change against the same change at years with no rule change.

    Three yardsticks, all from the placebo changes:
      p_value   rank-based: share of placebo changes at least as large (coarse; cannot go below 1/(n+1)).
      z, p_z    the change in units of the placebo spread, and the normal tail of that (smooth, so the
                false-discovery adjustment has something to work with); q_value adjusts p_z.
      z_modern  the same z using only placebo breaks from `modern_from` on, because 1950s-60s reliability
                swings would otherwise make every modern change look small.
    A change is `unusual` only when it clears all three.
    """
    s = seasons.set_index("season")
    rule_years = [r for r, _ in rules]
    last = int(s.index.max())
    rows = []
    for k in windows:
        placebo = _placebo_breaks(s.index, k, rule_years)
        for metric in metrics:
            series = s[metric]
            diffs, firsts = [], []
            for b in placebo:
                a, _ = _window_mean(series, b - k, b - 1)
                z, _ = _window_mean(series, b, b + k - 1)
                if not (np.isnan(a) or np.isnan(z)):
                    diffs.append(z - a)
                    firsts.append(b - k)
            diffs, firsts = np.array(diffs), np.array(firsts)
            sd = float(diffs.std(ddof=1)) if len(diffs) > 2 else np.nan
            modern = diffs[firsts >= modern_from]
            sd_m = float(modern.std(ddof=1)) if len(modern) > 4 else np.nan
            for year, label in rules:
                pre, n_pre = _window_mean(series, year - k, year - 1)
                post, n_post = _window_mean(series, year, year + k - 1)
                if n_pre < k or n_post == 0 or np.isnan(pre) or np.isnan(post):
                    continue
                complete = n_post == k and (year + k - 1) <= last
                change = post - pre
                ok = complete and len(diffs) > 2 and sd and sd > 0
                p_rank = (1 + float((np.abs(diffs) >= abs(change)).sum())) / (1 + len(diffs)) if ok else np.nan
                z_score = change / sd if ok else np.nan
                z_mod = change / sd_m if (ok and sd_m and sd_m > 0) else np.nan
                rows.append({"rule_year": year, "rule": label, "metric": metric, "window": k,
                             "pre_mean": pre, "post_mean": post, "change": change,
                             "post_seasons": n_post, "tested": bool(ok),
                             "placebo_breaks": len(diffs), "placebo_breaks_modern": len(modern),
                             "placebo_sd": sd, "placebo_sd_modern": sd_m,
                             "z": z_score, "z_modern": z_mod,
                             "p_value": p_rank, "p_z": _normal_p(z_score) if ok else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = np.nan
    main = out["window"] == main_window
    out.loc[main, "q_value"] = bh_qvalues(out.loc[main, "p_z"].to_numpy())
    out["unusual"] = ((out["q_value"] <= Q_ALPHA) & (out["p_value"] <= P_RANK_MAX)
                      & (out["z_modern"].abs() >= Z_MODERN_MIN)).fillna(False)
    return out.sort_values(["window", "rule_year", "metric"]).reset_index(drop=True)


def stability(tests: pd.DataFrame, main_window: int = MAIN_WINDOW) -> pd.DataFrame:
    """For each (rule, metric) flagged at the main window: do the other window lengths agree on direction and unusualness?"""
    main = tests[(tests["window"] == main_window) & tests["tested"]]
    rows = []
    for _, r in main.iterrows():
        other = tests[(tests["rule_year"] == r["rule_year"]) & (tests["metric"] == r["metric"])
                      & (tests["window"] != main_window) & tests["tested"]]
        same_sign = int((np.sign(other["change"]) == np.sign(r["change"])).sum())
        small_p = int((other["p_z"] < 0.10).sum())
        rows.append({"rule_year": r["rule_year"], "metric": r["metric"], "other_windows": len(other),
                     "same_direction": same_sign, "also_unusual": small_p})
    return pd.DataFrame(rows)


TREND_PRE = 8


def _trend_deviation(series: pd.Series, year: int, k: int, pre_n: int = TREND_PRE):
    """Mean of the k seasons from `year` minus where a straight line fitted to the `pre_n` seasons before would put them."""
    pre = series.reindex(range(year - pre_n, year)).dropna()
    post = series.reindex(range(year, year + k)).dropna()
    if len(pre) < max(6, pre_n - 2) or len(post) < k:
        return np.nan
    slope, intercept = np.polyfit(pre.index.to_numpy(dtype=float), pre.to_numpy(dtype=float), 1)
    pred = intercept + slope * post.index.to_numpy(dtype=float)
    return float(post.to_numpy().mean() - pred.mean())


def trend_adjusted_tests(seasons: pd.DataFrame, rules=RULES, window: int = MAIN_WINDOW, metrics=METRIC_KEYS,
                         modern_from: int = MODERN_FROM) -> pd.DataFrame:
    """Robustness check: the main test compares two flat means, so a measure that was already trending
    looks like it jumped. Here the after-period is compared with the continuation of the pre-trend,
    against the same quantity at years with no rule change (modern years only)."""
    s = seasons.set_index("season")
    rule_years = [r for r, _ in rules]
    last = int(s.index.max())
    placebo = [b for b in _placebo_breaks(s.index, window, rule_years) if b >= modern_from]
    rows = []
    for metric in metrics:
        series = s[metric]
        devs = np.array([d for d in (_trend_deviation(series, b, window) for b in placebo) if not np.isnan(d)])
        if len(devs) < 5:
            continue
        sd = float(devs.std(ddof=1))
        for year, label in rules:
            if year + window - 1 > last:
                continue
            dev = _trend_deviation(series, year, window)
            if np.isnan(dev) or sd <= 0:
                continue
            z = dev / sd
            rows.append({"rule_year": year, "rule": label, "metric": metric, "window": window,
                         "deviation_from_trend": dev, "placebo_breaks": len(devs), "z": z, "p_z": _normal_p(z),
                         "p_value": (1 + float((np.abs(devs) >= abs(dev)).sum())) / (1 + len(devs))})
    return pd.DataFrame(rows)


def nearest_misses(tests: pd.DataFrame, n: int = 3, main_window: int = MAIN_WINDOW) -> pd.DataFrame:
    """The n comparisons that came closest to being flagged (smallest adjusted q) among those that were not."""
    m = tests[(tests["window"] == main_window) & tests["tested"] & ~tests["unusual"]]
    return m.sort_values(["q_value", "p_z"]).head(n).reset_index(drop=True)


# ------------------------------------------------------------------ charts

def chart_trends(seasons: pd.DataFrame, rules, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=160, sharex=True)
    fig.subplots_adjust(top=0.82, left=0.07, right=0.98, bottom=0.08, hspace=0.30, wspace=0.28)
    fig.text(0.07, 0.965, "What decides a race, season by season", fontsize=13.5, fontweight="bold", color=TEXT,
             ha="left", va="top")
    fig.text(0.07, 0.915, "One line per measure, one point per season. Dashed lines mark the first season under new rules. "
             "The current season is still running.", fontsize=9.5, color=TEXT2, ha="left", va="top")
    for ax, (key, label, unit) in zip(axes.ravel(), METRICS):
        ax.plot(seasons["season"], seasons[key], color=BLUE, lw=1.6, marker="o", ms=2.8, zorder=3)
        for year, _ in rules:
            ax.axvline(year - 0.5, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_title(label, fontsize=10.5, color=TEXT, loc="left", fontweight="bold")
        ax.set_ylabel(unit, fontsize=8.5)
        ax.tick_params(labelsize=8.5)
        lo, hi = float(seasons[key].min()), float(seasons[key].max())
        pad = 0.08 * (hi - lo if hi > lo else 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Season", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def chart_rule_effects(tests: pd.DataFrame, path: Path, main_window: int = MAIN_WINDOW) -> None:
    plt = _plt()
    t = tests[(tests["window"] == main_window) & tests["tested"]].copy()
    if t.empty:
        return
    rules = sorted(t["rule_year"].unique())
    labels = {y: l for y, l in RULES}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=160, sharey=True)
    fig.subplots_adjust(top=0.79, left=0.30, right=0.98, bottom=0.09, hspace=0.32, wspace=0.12)
    fig.text(0.02, 0.965, "Did a rule change move the result more than a normal year does?", fontsize=13.5,
             fontweight="bold", color=TEXT, ha="left", va="top")
    sub = (f"Change from the {main_window} seasons before to the {main_window} after, in units of the usual change at years with "
           "no rule change. Inside the shaded band is ordinary drift. Orange = still unusual after allowing for the many "
           "comparisons and when set against modern years alone.")
    fig.text(0.02, 0.915, textwrap.fill(sub, 150), fontsize=9.5, color=TEXT2, ha="left", va="top", linespacing=1.45)
    ypos = {y: i for i, y in enumerate(reversed(rules))}
    for ax, (key, label, _) in zip(axes.ravel(), METRICS):
        d = t[t["metric"] == key]
        ax.axvspan(-2, 2, color=GRID, alpha=0.7, zorder=0)
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for _, r in d.iterrows():
            flagged = bool(r["unusual"])
            ax.scatter(r["z"], ypos[r["rule_year"]], s=46, color=ORANGE if flagged else BLUE, zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
        ax.set_title(label, fontsize=10.5, color=TEXT, loc="left", fontweight="bold")
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([labels.get(y, str(y)) for y in ypos], fontsize=8.5)
        ax.tick_params(labelsize=8.5)
        lim = max(3.0, float(np.nanmax(np.abs(d["z"]))) + 0.5) if len(d) else 3.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(rules) - 0.4)
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Change, in usual-year units", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ reporting

def _fmt(metric: str, v: float) -> str:
    if metric == "grid_corr":
        return f"{v:.2f}"
    if metric == "winner_variety":
        return f"{v:.2f}"
    return f"{v:.0%}"


def _fmt_change(metric: str, v: float) -> str:
    if metric in ("grid_corr", "winner_variety"):
        return f"{v:+.2f}"
    return f"{v * 100:+.0f} points"


def findings(seasons: pd.DataFrame, tests: pd.DataFrame, stab: pd.DataFrame, main_window: int = MAIN_WINDOW,
             trend_tests: pd.DataFrame | None = None) -> list[str]:
    L: list[str] = []
    labels = {k: l for k, l, _ in METRICS}
    L.append(f"SEASONS: {len(seasons)} seasons, {int(seasons['season'].min())} to {int(seasons['season'].max())}")
    for first, last, name in grid.ERAS:
        e = seasons[(seasons["season"] >= first) & (seasons["season"] <= last)]
        if e.empty:
            continue
        L.append(f"  {name}: pole wins {e['pole_win_rate'].mean():.0%}, grid correlation {e['grid_corr'].mean():.2f}, "
                 f"{e['finish_rate'].mean():.0%} of cars classified, top team wins {e['top_team_win_share'].mean():.0%} of races, "
                 f"winner starts 4th or lower in {e['upset_rate'].mean():.0%}, {e['winner_variety'].mean():.2f} different winners per race")
    main = tests[(tests["window"] == main_window) & tests["tested"]]
    n_tests = len(main)
    L.append(f"RULE CHANGES ({main_window} seasons before against {main_window} after; {n_tests} comparisons, each set against the "
             "same change at years with no rule change)")
    for year, label in RULES:
        d = tests[(tests["window"] == main_window) & (tests["rule_year"] == year)]
        if d.empty:
            continue
        if not bool(d["tested"].any()):
            first = d.iloc[0]
            L.append(f"  {label}: not tested, only {int(first['post_seasons'])} season(s) under the new rules so far")
            continue
        flagged = d[d["unusual"]].sort_values("q_value")
        if flagged.empty:
            L.append(f"  {label}: nothing moved by more than a normal year does")
            continue
        for _, r in flagged.iterrows():
            L.append(f"  {label}: {labels[r['metric']].lower()} went from {_fmt(r['metric'], r['pre_mean'])} to "
                     f"{_fmt(r['metric'], r['post_mean'])} ({_fmt_change(r['metric'], r['change'])}); unusual even allowing for "
                     f"{n_tests} comparisons (adjusted q = {r['q_value']:.3f}, rank p = {r['p_value']:.3f}, "
                     f"{r['z']:+.1f} usual-year units, {r['z_modern']:+.1f} against 1975-on years only)")
    unadjusted = int((main["p_z"] < 0.05).sum())
    L.append(f"  {unadjusted} of {n_tests} comparisons pass p < 0.05 on their own; about {0.05 * n_tests:.1f} would by luck alone")
    miss = nearest_misses(tests, 3, main_window)
    if len(miss):
        L.append("  closest to being flagged (none passed): " + "; ".join(
            f"{int(r['rule_year'])} {labels[r['metric']].lower()} {_fmt_change(r['metric'], r['change'])} "
            f"(q = {r['q_value']:.2f}, {r['z_modern']:+.1f} modern-year units)" for _, r in miss.iterrows()))
    if trend_tests is not None and len(trend_tests):
        big = trend_tests[(trend_tests["p_z"] < 0.05) & (trend_tests["p_value"] <= 0.10)]
        L.append(f"  trend-adjusted check (after-period against the continuation of the 8-season pre-trend): "
                 f"{len(big)} of {len(trend_tests)} comparisons look unusual before any allowance for multiple testing"
                 + ("" if big.empty else "; largest: " + "; ".join(
                     f"{int(r['rule_year'])} {labels[r['metric']].lower()} ({r['z']:+.1f} usual-year units)"
                     for _, r in big.reindex(big["z"].abs().sort_values(ascending=False).index).head(3).iterrows())))
    if len(stab):
        flagged = main[main["unusual"]]
        keep = stab.merge(flagged[["rule_year", "metric"]], on=["rule_year", "metric"])
        for _, r in keep.iterrows():
            L.append(f"  check, {int(r['rule_year'])} {labels[r['metric']].lower()}: with other window lengths the change has the same "
                     f"direction in {int(r['same_direction'])} of {int(r['other_windows'])} and is also unusual (p < 0.10) in "
                     f"{int(r['also_unusual'])}")
    if n_tests:
        sd = main.groupby("metric")["placebo_sd"].first()
        parts = [f"{labels[k].lower()} {_fmt_change(k, 2 * float(sd[k])).replace('+', '+/-')}" for k in METRIC_KEYS if k in sd.index]
        L.append("WHAT THE TEST COULD SEE: a rule change has to move a measure by about this much (two usual-year units) to stand "
                 "out; smaller shifts cannot be told from ordinary season-to-season noise. " + "; ".join(parts))
    L.append("READ WITH CARE: rule changes are not randomly assigned and several coincide with one team pulling ahead, so an "
             "unusual shift means 'something changed then', not 'the rules did it'. Compare the top-team share in the same year.")
    return L


# ------------------------------------------------------------------ run

def run(con, out_dir: Path | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "trends_and_rules"
    out.mkdir(parents=True, exist_ok=True)
    raw = grid_checks.load(con)
    df = grid_checks.prepare_checks(raw, drop_indy=True)
    races = race_table(df)
    seasons = season_table(races)
    tests = rule_change_tests(seasons)
    stab = stability(tests)
    trend_tests = trend_adjusted_tests(seasons)
    for name, table in [("season_measures", seasons), ("rule_change_tests", tests), ("rule_change_stability", stab),
                        ("rule_change_trend_adjusted", trend_tests)]:
        table.to_csv(out / f"{name}.csv", index=False)
    if len(seasons) >= 2:
        chart_trends(seasons, RULES, out / "what_decides_a_race_by_season.png")
    if len(tests):
        chart_rule_effects(tests, out / "rule_change_effects.png")
    lines = findings(seasons, tests, stab, trend_tests=trend_tests)
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'trends_and_rules'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
