"""Stage 2, part 8: driver ratings from race pace (FastF1 laps, 2018 on), next to qualifying pace.

The ratings in ratings.py measure one lap in qualifying. Winning a Grand Prix also needs pace over a
race distance, tyre care and racecraft, so this module builds the same kind of team-mate rating from
race laps and compares the two.

Race pace of a driver in a race:
  - use clean laps only: green-flag, timed accurately, not an in-lap or out-lap, not lap 1, within
    107% of the driver's own median;
  - compare each lap with the median of all drivers' clean laps on the same lap number, so fuel
    burn, track evolution and the pace of the day cancel;
  - the driver's pace is the median of ln(lap time / field median) over his clean laps (at least
    MIN_LAPS of them). Lower = faster.
Team-mate gap in a race = difference between the two drivers' paces (positive when the first is
faster). Only dry races and teams with exactly two cars are used. The gaps go into the same Bayesian
team-mate model as ratings.py, so the result is one rating per driver in percent of a lap per
race, with an interval.

Limits. The lap-by-lap comparison is not corrected for tyre age or fuel, so a driver on a different
strategy from his team-mate (older tyres, a late stop) is credited or penalised for it; the median over
many laps reduces but does not remove this. Drivers who run in traffic are slowed by it. One rating per
driver for 2018 on, so improvement and decline are not captured. The qualifying rating is fitted on the
same seasons so the two are comparable.

Run:  python -m src.analysis.race_pace
Writes to outputs/race_pace_ratings/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import ratings
from .grid import SERIES, TEXT, TEXT2, _frame, _plt
from .tyres import FIRST_SEASON, load_laps, load_rain_share, normalise_compounds, race_flags

MIN_LAPS = 15
SLOW_LIMIT = 1.07
MAX_ABS_GAP = 0.05          # drop race team-mate gaps above 5%: damage, a problem or a different race
MIN_OBS = 30                # comparisons needed for a driver to appear in rankings and comparisons


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    d = laps[laps["lap_time_s"].notna() & laps["is_accurate"].fillna(False).astype(bool)
             & laps["pit_in_s"].isna() & laps["pit_out_s"].isna()
             & (laps["track_status"].fillna("").astype(str) == "1") & (laps["lap_number"] > 1)].copy()
    med = d.groupby(["race_id", "driver_id"])["lap_time_s"].transform("median")
    return d[d["lap_time_s"] <= SLOW_LIMIT * med]


def race_pace(laps: pd.DataFrame, min_laps: int = MIN_LAPS) -> pd.DataFrame:
    """One row per driver per race: median ln(lap time / field median on the same lap), from clean laps."""
    d = clean_laps(laps)
    field = d.groupby(["race_id", "lap_number"])["lap_time_s"].transform("median")
    n_field = d.groupby(["race_id", "lap_number"])["lap_time_s"].transform("size")
    d = d.assign(rel=np.log(d["lap_time_s"] / field))[n_field >= 8]
    g = d.groupby(["race_id", "driver_id"])["rel"]
    out = pd.DataFrame({"pace": g.median(), "laps": g.size()}).reset_index()
    return out[out["laps"] >= min_laps].reset_index(drop=True)


def build_pace_obs(pace: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team-mate pair per race: y = pace_b - pace_a = ln(time_b / time_a), positive when a is faster."""
    d = pace.merge(teams, on=["race_id", "driver_id"], how="inner")
    d = ratings._two_car_teams(d)
    m = ratings._pairs(d, ["race_id", "constructor_id"])
    y = m["pace_b"] - m["pace_a"]
    m = m.assign(y=y)[y.abs() <= MAX_ABS_GAP]
    return pd.DataFrame({"race_id": m["race_id"], "season": (m["race_id"] // 100).astype(int),
                         "constructor_id": m["constructor_id"], "a": m["driver_id_a"], "b": m["driver_id_b"],
                         "y": m["y"]}).reset_index(drop=True)


def compare(race_t: pd.DataFrame, quali_t: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Per driver: race rating, qualifying rating (same seasons), and the difference with its interval."""
    r = race_t[race_t["n_obs"] >= min_obs][["driver_id", "name", "rating", "sd", "n_obs"]]
    q = quali_t[quali_t["n_obs"] >= min_obs][["driver_id", "rating", "sd", "n_obs"]]
    m = r.merge(q, on="driver_id", suffixes=("_race", "_quali"))
    m["race_minus_quali"] = m["rating_race"] - m["rating_quali"]
    m["diff_sd"] = np.sqrt(m["sd_race"] ** 2 + m["sd_quali"] ** 2)
    m["z"] = m["race_minus_quali"] / m["diff_sd"]
    return m.sort_values("race_minus_quali", ascending=False).reset_index(drop=True)


def noise_ceiling(cmp_: pd.DataFrame, spread: float, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Rank correlation expected if race and qualifying measured exactly the same skill, given each driver's
    own rating error. Draws a true skill per driver (normal, SD `spread` in percent of a lap), adds each
    rating's own error, and returns the 5th, 50th and 95th percentile of the rank correlation. If the
    observed correlation sits well below this, the two ratings differ by more than noise (real skill
    difference, or tyre age, strategy and traffic in the race laps)."""
    rng = np.random.default_rng(seed)
    n = len(cmp_)
    sr = cmp_["sd_race"].to_numpy(dtype=float)
    sq = cmp_["sd_quali"].to_numpy(dtype=float)
    out = np.empty(draws)
    for i in range(draws):
        t = rng.normal(0.0, spread, n)
        a = pd.Series(t + rng.normal(0.0, sr))
        b = pd.Series(t + rng.normal(0.0, sq))
        out[i] = a.rank().corr(b.rank())
    lo, mid, hi = np.nanpercentile(out, [5, 50, 95])
    return float(lo), float(mid), float(hi)


def chart_compare(cmp_: pd.DataFrame, path: Path, corr: float, ceiling: tuple[float, float, float] | None = None) -> None:
    plt = _plt()
    fig, ax = _frame(plt, "Does the qualifying rating match the race-pace rating?",
                     f"Each dot is a driver with at least {MIN_OBS} comparisons in both. Rating = percent of a lap faster than the "
                     f"average team-mate, 2018 on. Rank correlation {corr:.2f}"
                     + (f" (noise alone would give about {ceiling[1]:.2f}, range {ceiling[0]:.2f} to {ceiling[2]:.2f})" if ceiling else "")
                     + ". Dots above the diagonal are faster in the race than in qualifying.",
                     size=(8, 6.4))
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.11)
    lim = float(np.ceil(max(cmp_[["rating_race", "rating_quali"]].abs().max()) * 10) / 10) + 0.1
    ax.plot([-lim, lim], [-lim, lim], color="#c9c8c2", lw=1.2, zorder=1)
    ax.axhline(0, color="#c9c8c2", lw=0.8)
    ax.axvline(0, color="#c9c8c2", lw=0.8)
    ax.errorbar(cmp_["rating_quali"], cmp_["rating_race"], xerr=1.96 * cmp_["sd_quali"], yerr=1.96 * cmp_["sd_race"],
                fmt="none", ecolor="#e6e5e1", lw=1, zorder=2)
    ax.scatter(cmp_["rating_quali"], cmp_["rating_race"], s=32, color=SERIES[0], zorder=3)
    biggest = cmp_.reindex(cmp_["race_minus_quali"].abs().sort_values(ascending=False).index).head(8)
    best = cmp_.loc[[cmp_["rating_race"].idxmax()]]          # the top-rated driver is always named
    for _, r in pd.concat([biggest, best]).drop_duplicates("driver_id").iterrows():
        ax.annotate(r["name"].split()[-1], (r["rating_quali"], r["rating_race"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Qualifying rating (% of a lap)")
    ax.set_ylabel("Race-pace rating (% of a lap)")
    fig.savefig(path)
    plt.close(fig)


def findings(obs: pd.DataFrame, race_t: pd.DataFrame, noise: float, prior: float, cmp_: pd.DataFrame,
             corr: float, n_races: int, ceiling: tuple[float, float, float] | None = None) -> list[str]:
    L = [f"RACE PACE: {len(obs)} team-mate comparisons in {n_races} dry races since {FIRST_SEASON}, {len(race_t)} drivers; "
         f"typical gap noise {noise * 100:.2f}% of a lap, spread of true skill {prior * 100:.2f}%"]
    shown = race_t[race_t["n_obs"] >= MIN_OBS].head(10)
    L.append(f"  (top ten among drivers with at least {MIN_OBS} comparisons; the chart shows those with at least "
             f"{ratings.MIN_OBS_CHART})")
    for i, (_, r) in enumerate(shown.iterrows(), start=1):
        L.append(f"  {i}. {r['name']}: {r['rating']:+.2f}% ({r['ci_low']:+.2f} to {r['ci_high']:+.2f}), "
                 f"{int(r['n_obs'])} comparisons")
    if len(cmp_) >= 5:
        L.append(f"RACE vs QUALIFYING (same seasons, {len(cmp_)} drivers with at least {MIN_OBS} comparisons in both): "
                 f"rank correlation of the two ratings {corr:.2f}")
        if ceiling is not None:
            verdict = ("below" if corr < ceiling[0] else "at the low end of" if corr < ceiling[1] else "within")
            L.append(f"  if both ratings measured the same skill, rating error alone would give about {ceiling[1]:.2f} "
                     f"(5% to 95%: {ceiling[0]:.2f} to {ceiling[2]:.2f}); the observed {corr:.2f} is {verdict} that range, so "
                     f"{'race and qualifying skill differ, or tyre age, strategy and traffic matter' if corr < ceiling[0] else 'a difference between race and qualifying skill is possible but not established'}")
        flagged = cmp_[cmp_["z"].abs() >= 2]
        if len(flagged):
            for _, r in flagged.iterrows():
                L.append(f"  {r['name']}: race {r['rating_race']:+.2f}% against qualifying {r['rating_quali']:+.2f}% "
                         f"(difference {r['race_minus_quali']:+.2f}%, {r['z']:+.1f} standard errors)")
        else:
            L.append("  no driver's race rating differs from his qualifying rating by two standard errors")
        L.append(f"  {int((cmp_['z'].abs() >= 2).sum())} of {len(cmp_)} differ by two standard errors "
                 f"(about {0.05 * len(cmp_):.1f} expected by chance)")
    L.append("READ WITH CARE: lap times are not corrected for tyre age, so a different strategy from the team-mate moves a driver's "
             "rating; the intervals say how much, the order inside the top group is not settled.")
    return L


def load_inputs(con):
    laps = load_laps(con)
    laps = normalise_compounds(laps[laps["season"] >= FIRST_SEASON])
    teams = con.execute("SELECT race_id, driver_id, constructor_id FROM results").df().drop_duplicates(["race_id", "driver_id"])
    names = con.execute("""SELECT driver_id, trim(coalesce(given_name, '') || ' ' || coalesce(family_name, '')) AS name
                           FROM drivers""").df()
    quali = con.execute("""SELECT q.race_id, q.driver_id, q.constructor_id, q.position, q.q1, q.q2, q.q3
                           FROM qualifying q JOIN races ra USING (race_id) WHERE ra.season >= ?""", [FIRST_SEASON]).df()
    return laps, teams, names, quali


def run(con, out_dir: Path | None = None) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "race_pace_ratings"
    out.mkdir(parents=True, exist_ok=True)
    laps, teams, names, quali = load_inputs(con)
    flags = race_flags(laps, load_rain_share(con))
    dry = set(flags.loc[~flags["wet"], "race_id"])
    pace = race_pace(laps[laps["race_id"].isin(dry)])
    obs = build_pace_obs(pace, teams)
    if len(obs) < 50:
        lines = ["Not enough team-mate race-pace comparisons (need lap data from the Stage 1 FastF1 step)."]
        (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    fit, noise, prior = ratings.fit_ratings(obs)
    race_t = ratings.tidy(fit, names, 100.0)
    race_t.to_csv(out / "ratings_race_pace_2018_on.csv", index=False)
    pace.to_csv(out / "race_pace_by_driver_race.csv", index=False)
    ratings.chart_ratings(race_t, out / "ratings_race_pace_2018_on.png",
                          "Best drivers once the car is taken out: race pace since 2018",
                          "Percent of a lap faster than the average team-mate, race laps", "2018 to now")

    q_obs = ratings.build_time_obs(quali)
    cmp_ = pd.DataFrame()
    corr = float("nan")
    ceiling = None
    if len(q_obs) >= 50:
        qfit, _, _ = ratings.fit_ratings(q_obs)
        quali_t = ratings.tidy(qfit, names, 100.0)
        quali_t.to_csv(out / "ratings_qualifying_same_seasons.csv", index=False)
        cmp_ = compare(race_t, quali_t)
        if len(cmp_) >= 5:
            corr = float(cmp_["rating_race"].rank().corr(cmp_["rating_quali"].rank()))
            cmp_.to_csv(out / "race_vs_qualifying_ratings.csv", index=False)
            ceiling = noise_ceiling(cmp_, prior * 100.0)
            chart_compare(cmp_, out / "race_vs_qualifying_ratings.png", corr, ceiling)
    lines = findings(obs, race_t, noise, prior, cmp_, corr, int(pace["race_id"].nunique()), ceiling)
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
