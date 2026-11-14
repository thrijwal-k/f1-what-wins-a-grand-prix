"""Stage 2, part 1b: can the grid-vs-result findings be trusted?

Run:  python -m src.analysis.grid_checks
Writes tables (CSV), two charts (PNG) and findings.txt to outputs/grid_vs_result_checks/.

Checks
  A. Coverage: how much of each era is missing a grid slot, how big the fields are, how many cars retire.
  B. Indianapolis 500: the 1950-60 races are in the championship, but are they distorting 1950-69?
  C. Reliability: P(pole wins) = P(pole car finishes) x P(wins | finishes). Which factor moves over time?
     Also the grid-vs-finish correlation using only cars that reached the flag.
  D. Circuits: do circuits really differ in how much the grid decides, or is it noise?
     A permutation test (outcomes shuffled between circuits inside each season, so a dominant
     year cannot fake a circuit effect) and empirical-Bayes shrinkage of every circuit's rate.
  E. Dominance: is "the grid decides" partly "one team was untouchable that year"?
  F. Pole by grid vs pole by qualifying: how often grid penalties change who starts first.
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
from .grid import (AXIS, BLUE, ERA_LABELS, MIN_STARTERS, ORANGE, SERIES, SURFACE, TEXT, TEXT2, _frame, _plt,
                   prepare, race_summary, wilson)

N_PERM = 2000
SINCE = 2000
MIN_RACES = 8
SEED = 20240601


# ------------------------------------------------------------------ data

def load(con) -> pd.DataFrame:
    """Every result row (Indianapolis 500 included, flagged), with the car and the qualifying position."""
    has_q = bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'qualifying'").fetchone()[0])
    q_join = ("LEFT JOIN (SELECT race_id, driver_id, min(position) AS q_pos FROM qualifying "
              "GROUP BY race_id, driver_id) q ON q.race_id = r.race_id AND q.driver_id = r.driver_id") if has_q else ""
    q_col = "q.q_pos" if has_q else "NULL::INTEGER"
    df = con.execute(f"""
        SELECT r.race_id, r.season, r.round, ra.circuit_id,
               coalesce(c.circuit_name, ra.circuit_id) AS circuit_name,
               r.driver_id, r.constructor_id, r.grid, r.position,
               coalesce(r.classified, FALSE) AS classified, {q_col} AS q_pos
        FROM results r
        JOIN races ra ON ra.race_id = r.race_id
        LEFT JOIN circuits c ON c.circuit_id = ra.circuit_id
        {q_join}
        WHERE r.position IS NOT NULL
    """).df()
    df["is_indy"] = (df["circuit_id"] == "indianapolis") & (df["season"] <= 1960)
    return df


def prepare_checks(raw: pd.DataFrame, drop_indy: bool = True) -> pd.DataFrame:
    d = raw[~raw["is_indy"]] if drop_indy else raw
    return prepare(d)


# --------------------------------------------------------------- A. coverage

def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """df: prepared rows including Indianapolis, so the Indy share per era is visible."""
    rows = []
    for era in ERA_LABELS:
        d = df[df["era"] == era]
        if d.empty:
            continue
        rows.append({
            "era": era, "rows": len(d), "races": d["race_id"].nunique(),
            "share_grid_missing_or_zero": float((d["grid"] <= 0).mean()),
            "mean_field_size": float(d.groupby("race_id").size().mean()),
            "indy_rows": int(d["is_indy"].sum()),
            "share_not_classified": float((~d["classified"].astype(bool)).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- B. Indy

def indy_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, drop in [("with Indianapolis 500", False), ("without", True)]:
        df = prepare_checks(raw, drop_indy=drop)
        if df.empty:
            continue
        races = race_summary(df)
        back = df.assign(slot=df["grid_slot"].clip(upper=grid.MAX_SLOT))
        for era in ERA_LABELS:
            r = races[races["era"] == era]
            if r.empty or era not in ("1950-69", "1970-89"):
                continue
            pole_races = int(r["has_pole"].sum())
            low, high = wilson(int(r["pole_won"].sum()), pole_races)
            back_era = back[(back["era"] == era) & (back["slot"] >= grid.MAX_SLOT)]
            rows.append({
                "era": era, "version": label, "races": len(r), "pole_races": pole_races,
                "pole_win_rate": float(r["pole_won"].sum() / pole_races) if pole_races else math.nan,
                "pole_ci_low": low, "pole_ci_high": high,
                "mean_spearman": float(r["spearman"].mean()),
                "mean_finish_from_back": float(back_era["position"].mean()) if len(back_era) else math.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- C. reliability

def _finisher_spearman(g: pd.DataFrame) -> float:
    f = g[g["classified"].astype(bool)]
    if len(f) < MIN_STARTERS:
        return math.nan
    return f["grid_slot"].corr(f["position"], method="spearman")


def race_reliability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in df.groupby("race_id", sort=True):
        pole = g[g["grid"] == 1]
        first = g.iloc[0]
        row = {"race_id": race_id, "season": int(first["season"]), "era": first["era"],
               "has_pole": len(pole) > 0, "pole_finished": False, "pole_won": False,
               "spearman_all": g["grid_slot"].corr(g["position"], method="spearman"),
               "spearman_finishers": _finisher_spearman(g)}
        if len(pole):
            row["pole_finished"] = bool(pole["classified"].astype(bool).any())
            row["pole_won"] = bool((pole["position"] == 1).any())
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for era in ERA_LABELS:
        r = rel[(rel["era"] == era) & rel["has_pole"]]
        if r.empty:
            continue
        n = len(r)
        fin = int(r["pole_finished"].sum())
        won = int(r["pole_won"].sum())
        f_lo, f_hi = wilson(fin, n)
        g_lo, g_hi = wilson(won, fin)
        rows.append({
            "era": era, "pole_races": n,
            "pole_finish_rate": fin / n, "finish_ci_low": f_lo, "finish_ci_high": f_hi,
            "win_given_finish": won / fin if fin else math.nan, "wgf_ci_low": g_lo, "wgf_ci_high": g_hi,
            "pole_win_rate": won / n,
            "spearman_all_starters": float(r["spearman_all"].mean()),
            "spearman_finishers_only": float(r["spearman_finishers"].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------- D. circuits

def _between_stat(values: np.ndarray, groups: np.ndarray, sizes: np.ndarray) -> float:
    means = np.bincount(groups, weights=values) / sizes
    overall = values.mean()
    return float((sizes * (means - overall) ** 2).sum())


def circuit_permutation(values: np.ndarray, groups: np.ndarray, strata: np.ndarray,
                        n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Do groups differ more than chance allows? Values are shuffled between groups inside each stratum."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    sizes = np.bincount(groups).astype(float)
    sst = float(((values - values.mean()) ** 2).sum())
    observed = _between_stat(values, groups, sizes)
    blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    blocks = [b for b in blocks if len(b) > 1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = values.copy()
        for b in blocks:
            shuffled[b] = values[rng.permutation(b)]
        null[i] = _between_stat(shuffled, groups, sizes)
    return {"p_value": (1 + int((null >= observed).sum())) / (n_perm + 1),
            "share_explained": observed / sst if sst > 0 else math.nan,
            "share_by_chance": float(null.mean()) / sst if sst > 0 else math.nan}


def shrink_rates(wins: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float]:
    """Empirical-Bayes (beta-binomial, moment estimates): pull small samples towards the overall rate.
    Returns the shrunk rates and the prior strength in races (large = circuits barely differ)."""
    wins, n = np.asarray(wins, float), np.asarray(n, float)
    pbar = wins.sum() / n.sum()
    p = wins / n
    between = max(0.0, float(np.var(p, ddof=1) - (pbar * (1 - pbar) / n).mean())) if len(p) > 1 else 0.0
    if between < 1e-9:
        strength = 1e9
    else:
        strength = max(pbar * (1 - pbar) / between - 1, 1e-6)
    return (wins + pbar * strength) / (n + strength), float(strength)


def shrink_means(means: np.ndarray, n: np.ndarray, within_var: float) -> tuple[np.ndarray, np.ndarray]:
    """Normal-normal shrinkage of circuit means. Returns shrunk means and each circuit's reliability (0 to 1)."""
    means, n = np.asarray(means, float), np.asarray(n, float)
    mu = float(np.average(means, weights=n))
    sampling = within_var / n
    tau2 = max(0.0, float(np.var(means, ddof=1) - sampling.mean())) if len(means) > 1 else 0.0
    rel = tau2 / (tau2 + sampling) if tau2 > 0 else np.zeros_like(means)
    return mu + rel * (means - mu), rel


def circuit_heterogeneity(races: pd.DataFrame, since: int = SINCE, min_races: int = MIN_RACES,
                          n_perm: int = N_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-circuit table with raw and shrunk values, test summary with one row per measure)."""
    r = races[(races["season"] >= since) & races["has_pole"]]
    counts = r.groupby("circuit_id")["race_id"].size()
    keep = counts[counts >= min_races].index
    r = r[r["circuit_id"].isin(keep)].copy()
    if r["circuit_id"].nunique() < 3:
        return pd.DataFrame(), pd.DataFrame()
    codes, uniques = pd.factorize(r["circuit_id"])
    names = r.groupby("circuit_id")["circuit_name"].agg(lambda s: s.iloc[-1])

    g = r.groupby("circuit_id")
    table = pd.DataFrame({"circuit_id": uniques})
    table["circuit_name"] = table["circuit_id"].map(names)
    table["n_races"] = table["circuit_id"].map(g.size())
    table["pole_wins"] = table["circuit_id"].map(g["pole_won"].sum()).astype(int)
    table["pole_rate_raw"] = table["pole_wins"] / table["n_races"]
    table["pole_rate_shrunk"], strength = shrink_rates(table["pole_wins"].to_numpy(), table["n_races"].to_numpy())
    table["spearman_raw"] = table["circuit_id"].map(g["spearman"].mean())
    sp = r["spearman"].to_numpy()
    resid = sp - table["spearman_raw"].to_numpy()[codes]
    within_var = float((resid ** 2).sum() / max(len(r) - len(table), 1))
    table["spearman_shrunk"], table["spearman_reliability"] = shrink_means(
        table["spearman_raw"].to_numpy(), table["n_races"].to_numpy(), within_var)

    tests = []
    for label, vals in [("pole won", r["pole_won"].astype(float).to_numpy()), ("grid-vs-finish correlation", sp)]:
        res = circuit_permutation(vals, codes, r["season"].to_numpy(), n_perm=n_perm)
        tests.append({"measure": label, "circuits": len(table), "races": len(r), **res})
    tests = pd.DataFrame(tests)
    tests["pole_prior_strength_races"] = strength
    return table.sort_values("pole_rate_shrunk", ascending=False).reset_index(drop=True), tests


# ---------------------------------------------------------- E. dominance

def season_dominance(df: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Per season: the share of races won by the most successful constructor, next to the grid measures."""
    win = df[df["position"] == 1].groupby(["season", "constructor_id"]).size().rename("wins").reset_index()
    top = win.groupby("season")["wins"].max()
    out = grid.summarise(races, "season").set_index("season")
    out["top_team_win_share"] = (top / out["n_races"]).reindex(out.index)
    return out.reset_index()[["season", "n_races", "top_team_win_share", "pole_win_rate", "mean_spearman"]]


def dominance_correlations(seasons: pd.DataFrame, starts=(1950, 1990, 2010)) -> pd.DataFrame:
    rows = []
    for start in starts:
        s = seasons[seasons["season"] >= start].dropna(subset=["top_team_win_share"])
        if len(s) < 5:
            continue
        rows.append({"from_season": start, "seasons": len(s),
                     "rank_corr_with_pole_win_rate": s["top_team_win_share"].corr(s["pole_win_rate"], method="spearman"),
                     "rank_corr_with_grid_correlation": s["top_team_win_share"].corr(s["mean_spearman"], method="spearman")})
    return pd.DataFrame(rows)


# ------------------------------------------- F. pole by grid vs by qualifying

def pole_definitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in df.groupby("race_id", sort=True):
        if not g["q_pos"].notna().any():
            continue
        grid1 = g[g["grid"] == 1]
        q1 = g[g["q_pos"] == 1]
        if grid1.empty or q1.empty:
            continue
        rows.append({"race_id": race_id, "era": g.iloc[0]["era"],
                     "same": grid1.iloc[0]["driver_id"] == q1.iloc[0]["driver_id"],
                     "grid_pole_won": bool(grid1.iloc[0]["position"] == 1),
                     "quali_p1_won": bool(q1.iloc[0]["position"] == 1)})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    out = []
    for era in ERA_LABELS:
        e = d[d["era"] == era]
        if e.empty:
            continue
        diff = e[~e["same"]]
        out.append({"era": era, "races": len(e), "grid_pole_differs_share": float((~e["same"]).mean()),
                    "win_rate_grid_pole": float(e["grid_pole_won"].mean()),
                    "win_rate_quali_p1": float(e["quali_p1_won"].mean()),
                    "races_differing": len(diff),
                    "differing_win_rate_grid_pole": float(diff["grid_pole_won"].mean()) if len(diff) else math.nan,
                    "differing_win_rate_quali_p1": float(diff["quali_p1_won"].mean()) if len(diff) else math.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- charts

def chart_reliability(rel: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = _frame(plt, "Why pole does not always win: the car has to finish first",
                     "Per era: how often the pole car reaches the flag, how often it wins when it does, and the product "
                     "(the pole win rate). The last bar is always the first two multiplied.")
    fig.subplots_adjust(bottom=0.2)
    x = np.arange(len(rel))
    w = 0.26
    series = [("pole_finish_rate", "Pole car finishes", SERIES[0]),
              ("win_given_finish", "Wins, given it finishes", SERIES[1]),
              ("pole_win_rate", "Pole wins (product)", SERIES[2])]
    for i, (col, label, colour) in enumerate(series):
        pos = x + (i - 1) * (w + 0.02)
        ax.bar(pos, rel[col], width=w, color=colour, label=label, zorder=3)
        for xi, v in zip(pos, rel[col]):
            ax.text(xi, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(rel["era"])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share of races with a pole sitter")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", fontsize=9, labelcolor=TEXT, ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.savefig(path)
    plt.close(fig)


def chart_shrinkage(table: pd.DataFrame, path: Path, since: int = SINCE) -> None:
    plt = _plt()
    t = table.sort_values("pole_rate_shrunk", ascending=True).reset_index(drop=True)
    height = max(3.6, 0.3 * len(t) + 2.2)
    fig, ax = _frame(plt, "Circuit pole-win rates once small samples are pulled in",
                     f"Hollow dot: raw share of races won from pole since {since}. Filled dot: after empirical-Bayes "
                     "shrinkage towards the all-circuit rate. A long line means the raw rate rests on few races.",
                     size=(10.5, height))
    fig.subplots_adjust(top=1 - 1.5 / height, left=0.40, right=0.96, bottom=0.9 / height)
    y = np.arange(len(t))
    ax.hlines(y, t["pole_rate_raw"], t["pole_rate_shrunk"], color=AXIS, lw=1.6, zorder=2)
    ax.scatter(t["pole_rate_raw"], y, s=34, facecolor=SURFACE, edgecolor=TEXT2, lw=1.2, zorder=3, label="Raw")
    colours = [ORANGE if c in grid.HIGHLIGHT_CIRCUITS else BLUE for c in t["circuit_id"]]
    ax.scatter(t["pole_rate_shrunk"], y, s=40, color=colours, zorder=4, label="Shrunk")
    ax.axvline(float(t["pole_wins"].sum() / t["n_races"].sum()), color=AXIS, ls="--", lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={k})" for n, k in zip(t["circuit_name"], t["n_races"])], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Races won from pole (dashed line: all circuits together)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT)
    fig.savefig(path)
    plt.close(fig)



# --------------------------------------------------------------- findings

def findings(cov, indy, rel_table, tests, circ, dom_corr, poles) -> list[str]:
    L: list[str] = []
    L.append("COVERAGE")
    for _, r in cov.iterrows():
        L.append(f"  {r['era']}: {int(r['races'])} races, field {r['mean_field_size']:.1f} cars; "
                 f"{r['share_grid_missing_or_zero']:.0%} of rows without a normal grid slot; "
                 f"{r['share_not_classified']:.0%} not classified; {int(r['indy_rows'])} Indianapolis 500 rows")
    if len(indy):
        L.append("INDIANAPOLIS 500 (1950-60) IN OR OUT")
        for _, r in indy.iterrows():
            L.append(f"  {r['era']} {r['version']}: pole won {r['pole_win_rate']:.0%} of {int(r['pole_races'])} races "
                     f"({r['pole_ci_low']:.0%} to {r['pole_ci_high']:.0%}); correlation {r['mean_spearman']:.2f}; "
                     f"from 20th or lower finish {r['mean_finish_from_back']:.1f}")
    L.append("RELIABILITY: P(pole wins) = P(pole car finishes) x P(wins | finishes)")
    for _, r in rel_table.iterrows():
        L.append(f"  {r['era']}: finishes {r['pole_finish_rate']:.0%} x wins-if-finishes {r['win_given_finish']:.0%} "
                 f"= {r['pole_win_rate']:.0%} (n={int(r['pole_races'])}); grid-vs-finish correlation "
                 f"{r['spearman_all_starters']:.2f} all starters, {r['spearman_finishers_only']:.2f} finishers only")
    if len(tests):
        L.append(f"CIRCUITS SINCE {SINCE} (circuits with at least {MIN_RACES} races)")
        for _, r in tests.iterrows():
            verdict = "circuits differ beyond chance" if r["p_value"] < 0.05 else "no clear circuit effect"
            L.append(f"  {r['measure']}: circuit explains {r['share_explained']:.1%} of race-to-race variation, "
                     f"chance alone would give {r['share_by_chance']:.1%}; permutation p = {r['p_value']:.3f} ({verdict}); "
                     f"{int(r['circuits'])} circuits, {int(r['races'])} races")
        for cid in grid.HIGHLIGHT_CIRCUITS:
            row = circ[circ["circuit_id"] == cid]
            if len(row):
                r = row.iloc[0]
                L.append(f"  {r['circuit_name']}: pole won {r['pole_rate_raw']:.0%} raw, {r['pole_rate_shrunk']:.0%} shrunk "
                         f"(n={int(r['n_races'])}); correlation {r['spearman_raw']:.2f} raw, {r['spearman_shrunk']:.2f} shrunk")
    if len(dom_corr):
        L.append("TEAM DOMINANCE vs HOW MUCH THE GRID DECIDES (rank correlation across seasons; 1 = move together)")
        for _, r in dom_corr.iterrows():
            L.append(f"  from {int(r['from_season'])} ({int(r['seasons'])} seasons): top-team win share vs pole win rate "
                     f"{r['rank_corr_with_pole_win_rate']:+.2f}, vs grid correlation {r['rank_corr_with_grid_correlation']:+.2f}")
    if len(poles):
        L.append("POLE BY GRID vs POLE BY QUALIFYING (races where both are known)")
        for _, r in poles.iterrows():
            L.append(f"  {r['era']}: grid pole differs from qualifying P1 in {r['grid_pole_differs_share']:.0%} of "
                     f"{int(r['races'])} races; win rate {r['win_rate_grid_pole']:.0%} (grid) vs {r['win_rate_quali_p1']:.0%} (qualifying)")
    return L


def run(con, out_dir: Path | None = None, n_perm: int = N_PERM) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "grid_vs_result_checks"
    out.mkdir(parents=True, exist_ok=True)

    raw = load(con)
    with_indy = prepare_checks(raw, drop_indy=False)
    df = with_indy[~with_indy["is_indy"]].copy()
    races = race_summary(df)

    cov = coverage(with_indy)
    indy = indy_sensitivity(raw)
    rel = race_reliability(df)
    rel_table = reliability_table(rel)
    circ, tests = circuit_heterogeneity(races, n_perm=n_perm)
    seasons = season_dominance(df, races)
    dom_corr = dominance_correlations(seasons)
    poles = pole_definitions(df)

    for name, table in [("coverage_by_era", cov), ("indy_sensitivity", indy), ("reliability_by_era", rel_table),
                        ("circuit_shrinkage", circ), ("circuit_tests", tests), ("season_dominance", seasons),
                        ("dominance_correlations", dom_corr), ("pole_by_grid_vs_qualifying", poles)]:
        table.to_csv(out / f"{name}.csv", index=False)

    if len(rel_table):
        chart_reliability(rel_table, out / "reliability_decomposition.png")
    if len(circ):
        chart_shrinkage(circ, out / "circuit_shrinkage.png")

    lines = findings(cov, indy, rel_table, tests, circ, dom_corr, poles)
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'grid_vs_result_checks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
