"""Stage 2, deep dive 3: hardening the car-vs-driver split.

car_driver.py gives one number per era. This module asks how far that number can be trusted, in three ways.

  1 stress tests   The earlier recovery check invents data from the same model it then fits, so it can only show
                   that the design can identify the shares, never what happens when the world is different.
                   Here outcomes are invented from the REAL drivers, constructors and seasons of each era under
                   five worlds the model does not assume, ranked within each race and turned into rankit scores
                   exactly as the real pipeline does, and the fitted shares are compared with the shares built in:
                     well specified          the model's own world (ranking and scoring only)
                     driver form drifts      each driver's skill changes from season to season (counts as driver)
                     driver-car fit          some drivers suit some cars (counts as neither: "everything else")
                     in-season development   a car gets better or worse through the season (counts as car)
                     reliability varies      some cars retire far more often (the expected loss counts as car)
                     random retirements      12% of runs lose 2.5 units to bad luck (counts as everything else)
  2 what is hiding   An extended model adds three things the basic model folds into other parts: a driver-by-season
                   effect (form that changes from season to season, against skill that lasts), a driver-by-constructor
                   effect (a driver who suits one team's cars), and a car-development slope (a car that gets better
                   or worse through the season). Team-mate differences within a season identify the first, drivers who
                   change teams the second, the order of races inside a season the third. The sizes found in the data are
                   then used to re-run the stress tests with worlds of realistic size ("data-sized"), next to the
                   deliberately strong ones ("extreme").
  3 intervals      Refits on resampled races (whole races, within each season) give an interval for the shares that
                   reflects how many races the era has (the spread of the refits, centred on the estimate, because
                   resampling repeats races and shifts the refits up); the season-subset spread in car_driver.py reflects which
                   seasons are included. The stress tests give an allowance for a wrong model: the range of
                   (estimate - truth) across the worlds above is added to the interval, once for the extreme worlds
                   and once for the data-sized ones.

Run:  python -m src.analysis.car_driver_robust            (about 15 minutes)
      python -m src.analysis.car_driver_robust --quick    (a few minutes, fewer repeats)
Writes to outputs/car_vs_driver_robust/.

Limits. Invented worlds are only a few of the ways reality can differ. The extended model's two driver parts are
separated only by team-mates within a season; for eras with few seasons per driver the split is soft.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import car_driver as cd
from .grid import ERA_LABELS

BASE = {"team": 0.25, "car": 0.25, "driver": 0.12}      # shares of the latent variance; noise is the rest (38%)
SCENARIOS = ["well specified", "driver form drifts", "driver-car fit", "in-season development",
             "reliability varies", "random retirements"]
N_REP_STRESS = 3
N_BOOT = 60
BOOT_ITER = 30
SEED = 20261024
MIN_ROWS = 400


def _factor(values: pd.Series) -> np.ndarray:
    return pd.factorize(values)[0]


# ------------------------------------------------------------------ 1 stress tests

def simulate(design: pd.DataFrame, scenario: str, rng: np.random.Generator, base: dict | None = None, as_ranks: bool = True,
             sizes: dict | None = None):
    """Outcomes for the design's driver-races under a scenario. Returns (score, truth) with truth the shares
    of latent variance that belong to the constructor and to the driver."""
    base = base or BASE
    sizes = sizes or {}
    n = len(design)
    drv, car, team = _factor(design["driver_id"]), _factor(design["car"]), _factor(design["constructor_id"])
    sd = {k: np.sqrt(v) for k, v in base.items()}
    sd_noise = np.sqrt(1 - sum(base.values()))
    comp_team = rng.normal(0, sd["team"], team.max() + 1)[team]
    comp_car = rng.normal(0, sd["car"], car.max() + 1)[car]
    comp_drv = rng.normal(0, sd["driver"], drv.max() + 1)[drv]
    comp_other = rng.normal(0, sd_noise, n)
    car_extra = np.zeros(n)
    drv_extra = np.zeros(n)
    if scenario == "driver form drifts":
        key = _factor(design["driver_id"].astype(str) + "|" + design["season"].astype(str))
        drv_extra = rng.normal(0, np.sqrt(sizes.get("drift", base["driver"])), key.max() + 1)[key]
    elif scenario == "driver-car fit":
        key = _factor(design["driver_id"].astype(str) + "|" + design["constructor_id"].astype(str))
        comp_other = comp_other + rng.normal(0, np.sqrt(sizes.get("fit", base["driver"])), key.max() + 1)[key]
    elif scenario == "in-season development":
        rank_in_season = design.groupby("season")["race_id"].transform(lambda s: pd.factorize(s, sort=True)[0])
        span = design.groupby("season")["race_id"].transform(lambda s: max(s.nunique() - 1, 1))
        t = (rank_in_season / span).to_numpy(float) * 2 - 1                  # -1 at the first race, +1 at the last
        slope = rng.normal(0, np.sqrt(3 * sizes.get("dev", base["car"] / 3)), car.max() + 1)[car]
        car_extra = slope * t
    elif scenario == "reliability varies":
        p_car = np.clip(rng.normal(0.12, 0.08, car.max() + 1), 0.01, 0.4)[car]
        hit = rng.random(n) < p_car
        car_extra = -2.5 * p_car
        comp_other = comp_other - 2.5 * (hit - p_car)
    elif scenario == "random retirements":
        comp_other = comp_other - 2.5 * (rng.random(n) < 0.12)
    latent = comp_team + comp_car + comp_drv + car_extra + drv_extra + comp_other
    var = float(np.var(latent))
    truth = {"car": float(np.var(comp_team + comp_car + car_extra)) / var, "driver": float(np.var(comp_drv + drv_extra)) / var}
    if not as_ranks:                       # diagnostic: the latent values themselves, centred within each race
        return (latent - pd.Series(latent, index=design.index).groupby(design["race_id"]).transform("mean").to_numpy()), truth
    # exactly the pipeline's order: position within each race, then the rankit score
    pos = pd.Series(latent, index=design.index).groupby(design["race_id"]).rank(ascending=False, method="first")
    score = cd.rankit(pos, design["race_id"]).to_numpy(float)
    return score, truth


def stress_era(era_rows: pd.DataFrame, era: str, n_rep: int = N_REP_STRESS, seed: int = SEED,
               sizes: dict | None = None, scenarios: list | None = None) -> pd.DataFrame:
    """sizes=None gives the deliberately strong worlds ("extreme"); sizes from the data give "data-sized" worlds."""
    design = era_rows[["race_id", "season", "driver_id", "constructor_id", "car"]].drop_duplicates(["race_id", "driver_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = []
    for scenario in (scenarios or SCENARIOS):
        est, tru = [], []
        for _ in range(n_rep):
            score, truth = simulate(design, scenario, rng, sizes=sizes)
            fit = cd.fit_era(design.assign(score=score), "score")
            est.append((fit["car"], fit["driver"]))
            tru.append((truth["car"], truth["driver"]))
        est, tru = np.mean(est, axis=0), np.mean(tru, axis=0)
        out.append({"era": era, "world": "extreme" if sizes is None else "data-sized", "scenario": scenario, "reps": n_rep,
                    "truth_car": tru[0], "est_car": est[0],
                    "bias_car": est[0] - tru[0], "truth_driver": tru[1], "est_driver": est[1],
                    "bias_driver": est[1] - tru[1]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 2 what is hiding

def _season_position(d: pd.DataFrame) -> np.ndarray:
    """-1 at the first race of a season, +1 at the last."""
    code = d.groupby("season")["race_id"].transform(lambda x: pd.factorize(x, sort=True)[0]).to_numpy(float)
    span = d.groupby("season")["race_id"].transform(lambda x: max(x.nunique() - 1, 1)).to_numpy(float)
    return code / span * 2 - 1


def fit_extended(era_rows: pd.DataFrame, column: str) -> dict:
    """Shares of the total for: car (season and persistent), car development through the season, persistent driver skill,
    driver form by season, driver-by-constructor fit, and everything else."""
    d = era_rows[era_rows[column].notna()]
    driver, nd = cd._codes(d["driver_id"])
    car, nc = cd._codes(d["car"])
    team, nt = cd._codes(d["constructor_id"])
    ds, nds = cd._codes(d["driver_id"].astype(str) + "|" + d["season"].astype(str))
    dc, ndc = cd._codes(d["driver_id"].astype(str) + "|" + d["constructor_id"].astype(str))
    t = _season_position(d)
    v = cd.fit_components(d[column].to_numpy(float), driver, car, nd, nc, team, nt,
                          extra=[("dseason", ds, nds), ("dxc", dc, ndc), ("cslope", car, nc, t)])
    dev = v["cslope"] * float(np.mean(t ** 2))
    total = v["car"] + v["team"] + v["driver"] + v["dseason"] + v["dxc"] + dev + v["other"]
    return {"car": (v["car"] + v["team"]) / total, "car_development": dev / total,
            "driver_persistent": v["driver"] / total, "driver_form": v["dseason"] / total,
            "driver_car_fit": v["dxc"] / total, "other": v["other"] / total,
            "driver_total": (v["driver"] + v["dseason"]) / total}


def form_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for era in ERA_LABELS:
        era_rows = rows[rows["era"] == era]
        if len(era_rows) < MIN_ROWS:
            continue
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            base = cd.fit_era(era_rows, column)
            ext = fit_extended(era_rows, column)
            out.append({"era": era, "outcome": outcome, "driver_races": int(era_rows[column].notna().sum()),
                        "base_car": base["car"], "base_driver": base["driver"], "base_other": base["other"],
                        **{f"ext_{k}": v for k, v in ext.items()}})
    return pd.DataFrame(out)


def data_sizes(form: pd.DataFrame, era: str) -> dict:
    """Sizes of the three hidden effects found in the data for an era (race outcome), as shares of the total, for
    the data-sized stress worlds. A floor of 0.5% keeps a world from vanishing."""
    r = form[(form["era"] == era) & (form["outcome"] == "race")]
    if r.empty:
        return {}
    r = r.iloc[0]
    return {"drift": max(float(r["ext_driver_form"]), 0.005), "fit": max(float(r["ext_driver_car_fit"]), 0.005),
            "dev": max(float(r["ext_car_development"]), 0.005)}


DATA_SCENARIOS = ["driver form drifts", "driver-car fit", "in-season development"]


def race_bootstrap(era_rows: pd.DataFrame, column: str, n_rep: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Refits on resampled races (whole races, within each season), warm-started from the point fit."""
    d = era_rows[era_rows[column].notna()].reset_index(drop=True)
    driver, nd = cd._codes(d["driver_id"])
    car, nc = cd._codes(d["car"])
    team, nt = cd._codes(d["constructor_id"])
    y = d[column].to_numpy(float)
    point = cd.fit_components(y, driver, car, nd, nc, team, nt)
    races = {s: [np.flatnonzero((d["race_id"] == r).to_numpy()) for r in g["race_id"].unique()]
             for s, g in d.groupby("season")}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        idx = np.concatenate([r[i] for r in races.values() for i in rng.integers(0, len(r), len(r))])
        v = cd.fit_components(y[idx], driver[idx], car[idx], nd, nc, team[idx], nt, n_iter=BOOT_ITER, init=point)
        out.append(cd.shares(v))
    return pd.DataFrame(out)[["car", "driver", "other"]]


def _widen(low: float, high: float, bias: pd.Series) -> tuple[float, float]:
    """Truth is about estimate minus bias: a model that overstates by b puts the truth b lower, so the interval is
    widened downwards by the largest overstatement and upwards by the largest understatement."""
    if not len(bias):
        return low, high
    return max(0.0, low - max(0.0, float(bias.max()))), min(1.0, high - min(0.0, float(bias.min())))


def intervals_table(rows: pd.DataFrame, stress: pd.DataFrame, n_boot: int = N_BOOT,
                    stress_data: pd.DataFrame | None = None) -> pd.DataFrame:
    out = []
    for era in ERA_LABELS:
        era_rows = rows[rows["era"] == era]
        if len(era_rows) < MIN_ROWS:
            continue
        allow = stress[stress["era"] == era]
        base_row = allow[allow["scenario"] == "well specified"]
        allow_data = pd.concat([base_row, stress_data[stress_data["era"] == era]]) if stress_data is not None and len(stress_data) else None
        for outcome, column in (("qualifying", "start_score"), ("race", "finish_score")):
            point = cd.fit_era(era_rows, column)
            boots = race_bootstrap(era_rows, column, n_boot)
            rec = {"era": era, "outcome": outcome, "driver_races": int(era_rows[column].notna().sum())}
            for part in ("car", "driver"):
                rec[f"{part}_share"] = point[part]
                # Resampled races contain repeats of the same race, which makes the refits look more consistent and
                # pushes the driver share up, so the percentile interval sits off the estimate. Its spread is kept and
                # centred on the estimate; the shift is reported as boot_shift.
                med = float(boots[part].median())
                rec[f"{part}_boot_shift"] = med - point[part]
                low = max(0.0, point[part] - (med - float(boots[part].quantile(0.025))))
                high = min(1.0, point[part] + (float(boots[part].quantile(0.975)) - med))
                rec[f"{part}_boot_low"], rec[f"{part}_boot_high"] = low, high
                rec[f"{part}_allowed_low"], rec[f"{part}_allowed_high"] = _widen(low, high, allow[f"bias_{part}"])
                if allow_data is not None:
                    rec[f"{part}_data_low"], rec[f"{part}_data_high"] = _widen(low, high, allow_data[f"bias_{part}"])
                else:
                    rec[f"{part}_data_low"], rec[f"{part}_data_high"] = low, high
            out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ output

SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"


def chart_intervals(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False})
    eras = [e for e in ERA_LABELS if e in set(table["era"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.62 * len(eras) + 3.0), dpi=160, sharey=True)
    h = 0.62 * len(eras) + 3.0
    fig.subplots_adjust(top=1 - 1.45 / h, left=0.11, right=0.975, bottom=0.85 / h, wspace=0.14)
    fig.text(0.11, 1 - 0.12 / h, "How sure is the driver share?", fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.11, 1 - 0.5 / h, textwrap.fill(
        "Dot: estimate. Orange: 95% interval from resampling races. Dark grey: widened by how far the model was off in invented "
        "worlds whose hidden effects are the size the data show. Light grey: widened for deliberately strong worlds (drivers "
        "drifting, driver-car fit, car development, unreliable cars, random retirements).", 135), fontsize=9.2, color=TEXT2, va="top", linespacing=1.45)
    y = np.arange(len(eras))[::-1]
    for ax, outcome, title in zip(axes, ("qualifying", "race"), ("Qualifying position", "Race finishing position")):
        t = table[table["outcome"] == outcome].set_index("era").reindex(eras)
        ax.hlines(y, t["driver_allowed_low"], t["driver_allowed_high"], color="#c9c8c2", lw=2)
        ax.hlines(y, t["driver_data_low"], t["driver_data_high"], color="#8a8984", lw=3.2)
        ax.hlines(y, t["driver_boot_low"], t["driver_boot_high"], color="#eb6834", lw=5, capstyle="round")
        ax.scatter(t["driver_share"], y, color=TEXT, s=26, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
        ax.set_xlim(0, max(0.4, float(np.nanmax(t["driver_allowed_high"])) + 0.03))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(eras)
    axes[0].set_xlabel("Driver share of the variation")
    axes[1].set_xlabel("Driver share of the variation")
    fig.savefig(path)
    plt.close(fig)


def _p(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def findings(stress: pd.DataFrame, form: pd.DataFrame, intervals: pd.DataFrame, stress_data: pd.DataFrame | None = None) -> list[str]:
    L = []

    def block(frame: pd.DataFrame, title: str) -> None:
        L.append(title)
        for era in ERA_LABELS:
            sub = frame[frame["era"] == era]
            if sub.empty:
                continue
            L.append(f"  {era}:")
            for _, r in sub.iterrows():
                L.append(f"    {r['scenario']}: driver truth {_p(r['truth_driver'])}, estimate {_p(r['est_driver'])} "
                         f"(bias {r['bias_driver'] * 100:+.0f}); car truth {_p(r['truth_car'])}, estimate {_p(r['est_car'])} "
                         f"(bias {r['bias_car'] * 100:+.0f})")

    if len(stress):
        block(stress, "STRESS TESTS, DELIBERATELY STRONG WORLDS: outcomes invented on the real drivers, constructors and seasons of "
                      "each era, ranked and scored as in the pipeline; estimate against the share built in (bias = estimate minus "
                      "truth, in points of share). The well-specified world shows the cost of ranking alone: the car share comes "
                      "out about 3 points low.")
    if stress_data is not None and len(stress_data):
        block(stress_data, "STRESS TESTS, DATA-SIZED WORLDS: the same, with each hidden effect (driver form, driver-car fit, "
                           "in-season car development) set to the size the extended model finds in that era's real race results")
    if len(form):
        L.append("WHAT THE EXTENDED MODEL FINDS (race results; qualifying alongside): the driver share split into skill that lasts "
                 "and form that changes by season, plus driver-car fit and in-season car development")
        for _, r in form.iterrows():
            L.append(f"  {r['era']} {r['outcome']}: car {_p(r['ext_car'])}, car development in-season {_p(r['ext_car_development'])}, "
                     f"driver skill {_p(r['ext_driver_persistent'])}, driver form {_p(r['ext_driver_form'])}, driver-car fit "
                     f"{_p(r['ext_driver_car_fit'])} (driver total {_p(r['ext_driver_total'])}; the basic model said "
                     f"{_p(r['base_driver'])}), everything else {_p(r['ext_other'])}")
    if len(intervals):
        L.append("INTERVALS FOR THE DRIVER AND CAR SHARES: estimate; 95% interval from resampled races; widened by the data-sized "
                 "worlds; widened by the deliberately strong worlds")
        for _, r in intervals.iterrows():
            L.append(f"  {r['era']} {r['outcome']}: driver {_p(r['driver_share'])} (races {_p(r['driver_boot_low'])}-{_p(r['driver_boot_high'])}; "
                     f"data-sized {_p(r['driver_data_low'])}-{_p(r['driver_data_high'])}; strong {_p(r['driver_allowed_low'])}-{_p(r['driver_allowed_high'])}); "
                     f"car {_p(r['car_share'])} (races {_p(r['car_boot_low'])}-{_p(r['car_boot_high'])}; "
                     f"data-sized {_p(r['car_data_low'])}-{_p(r['car_data_high'])}; strong {_p(r['car_allowed_low'])}-{_p(r['car_allowed_high'])})")
    L.append("READ WITH CARE: the invented worlds are a few of the ways reality can differ, so the allowances are guides, not bounds; "
             "the form, fit and development sizes rest on team-mate differences and are soft where drivers or cars have few seasons.")
    return L


def run(con, out_dir: Path | None = None, quick: bool = False) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "car_vs_driver_robust"
    out.mkdir(parents=True, exist_ok=True)
    results, quali = cd.load_inputs(con)
    rows = cd.build_rows(results, quali)
    n_rep, n_boot = (1, 12) if quick else (N_REP_STRESS, N_BOOT)
    eras = [e for e in ERA_LABELS if (rows["era"] == e).sum() >= MIN_ROWS]
    stress = pd.concat([stress_era(rows[rows["era"] == e], e, n_rep) for e in eras] or [pd.DataFrame()], ignore_index=True)
    stress.to_csv(out / "stress_tests.csv", index=False)
    form = form_table(rows)
    form.to_csv(out / "driver_form_vs_skill.csv", index=False)
    parts = []
    for e in eras:
        sizes = data_sizes(form, e)
        if sizes:
            parts.append(stress_era(rows[rows["era"] == e], e, n_rep, sizes=sizes, scenarios=DATA_SCENARIOS))
    stress_data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    stress_data.to_csv(out / "stress_tests_data_sized.csv", index=False)
    intervals = intervals_table(rows, stress, n_boot, stress_data) if len(stress) else pd.DataFrame()
    intervals.to_csv(out / "driver_share_intervals.csv", index=False)
    if len(intervals):
        chart_intervals(intervals, out / "driver_share_intervals.png")
    lines = findings(stress, form, intervals, stress_data)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    db = config.db_path()
    if not Path(db).exists():
        sys.exit(f"No database at {db}. Run stage 1 first: python -m src.run_all")
    con = duckdb.connect(str(db), read_only=True)
    try:
        print("\n".join(run(con, quick="--quick" in sys.argv)))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
