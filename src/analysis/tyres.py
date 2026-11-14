"""Stage 2, part 4a: tyres, pit stops and strategy mix (FastF1 laps, 2018 on).

Run:  python -m src.analysis.tyres
Writes tables (CSV), charts (PNG) and findings.txt to outputs/tyres_and_pits/.

What it measures
  - Tyre degradation per compound, race by race: lap time on clean green-flag laps explained by a
    driver effect (pace of that driver in that race), a lap-number trend (fuel burn and track
    evolution) and, for each compound, an effect of tyre age. The lap-number trend is a built-in sanity
    check: it should come out negative (cars get quicker as fuel burns off).
  - Pit loss per circuit: in-lap plus out-lap time, minus two laps at that driver's own green-flag
    pace around the stop. This is the number a race simulator needs.
  - Strategy mix: stops per car, and whether winners stop less than the cars around them.

Laps in the wet, under yellow, safety car or VSC, in the pits, on the first lap, or more than 7% slower
than the race median are left out of the tyre model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from .grid import AXIS, BLUE, SERIES, TEXT, TEXT2, _frame, _plt

DRY = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
NEUTRAL_CODES = "4567"          # safety car, red flag, VSC, VSC ending
WET_LAPS_MIN = 30               # a race with this many laps on wet-weather tyres counts as wet
RAIN_SHARE_MIN = 0.25           # ... or with rain in this share of weather samples
BORDERLINE_WET_LAPS = 100       # a wet race with fewer wet-tyre laps than this (a wet start, or rain late) is called borderline
NEUTRAL_SHARE = 0.25            # share of cars that must record a safety-car code on a lap for it to count
RAW_COMPOUND_ORDER = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"]
OLD_NAMES = {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SUPERHARD"}
MIN_LAPS_NAMED = 100
SLOW_CUT = 1.07
MIN_LAPS_RACE = 250
MIN_DRIVERS = 8
MIN_LAPS_COMPOUND = 60
MIN_AGE_SPREAD = 5.0            # a compound needs laps over a spread of tyre ages, or its slope is not identified
REF_AGE = 10                    # pace gaps between compounds are compared at this tyre age
PACE_GAP_TRUST = 0.65           # share of races in which hard must be slower than medium for the pace gaps to be trusted
PIT_LOSS_RANGE = (8.0, 45.0)
MIN_STOPS_CIRCUIT = 30
FIRST_SEASON = 2018


# ------------------------------------------------------------------ data

def _has(con, name: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                            [name]).fetchone()[0])


def load_laps(con) -> pd.DataFrame:
    has_end = bool(con.execute("SELECT count(*) FROM information_schema.columns WHERE table_name = 'laps' "
                               "AND column_name = 'lap_end_s'").fetchone()[0])
    end = "l.lap_end_s" if has_end else "CAST(NULL AS DOUBLE) AS lap_end_s"
    return con.execute(f"""
        SELECT l.race_id, ra.season, ra.round, ra.circuit_id,
               coalesce(c.circuit_name, ra.circuit_id) AS circuit_name,
               l.driver_id, l.lap_number, l.lap_time_s, l.stint, l.compound, l.tyre_life,
               l.pit_in_s, l.pit_out_s, l.track_status, l.position, l.is_accurate, {end}
        FROM laps l
        JOIN races ra ON ra.race_id = l.race_id
        LEFT JOIN circuits c ON c.circuit_id = ra.circuit_id
        WHERE l.driver_id IS NOT NULL
    """).df()


def load_rain_share(con) -> pd.Series | None:
    if not _has(con, "weather"):
        return None
    df = con.execute("SELECT race_id, avg(CASE WHEN rainfall THEN 1.0 ELSE 0.0 END) AS share "
                     "FROM weather GROUP BY race_id").df()
    return df.set_index("race_id")["share"]


def normalise_compounds(laps: pd.DataFrame) -> pd.DataFrame:
    """2018 used names such as ULTRASOFT; from 2019 the three compounds at a race are called SOFT, MEDIUM, HARD.
    In a race with old names, the three dry compounds actually used are relabelled softest -> SOFT, middle -> MEDIUM,
    hardest -> HARD. A race that does not show exactly three is relabelled UNKNOWN (left out of the tyre model)."""
    laps = laps.copy()
    comp = laps["compound"].astype("object")
    for race_id, idx in laps.groupby("race_id").groups.items():
        names = comp.loc[idx]
        if not names.isin(OLD_NAMES).any():
            continue
        counts = names[names.isin(RAW_COMPOUND_ORDER)].value_counts()
        used = [c for c in RAW_COMPOUND_ORDER if counts.get(c, 0) >= MIN_LAPS_NAMED]
        mapping = dict(zip(used, DRY)) if len(used) == 3 else {}
        comp.loc[idx] = names.map(lambda c: mapping.get(c, "UNKNOWN") if c in RAW_COMPOUND_ORDER else c)
    laps["compound"] = comp
    return laps


def race_flags(laps: pd.DataFrame, rain_share: pd.Series | None = None) -> pd.DataFrame:
    """One row per race: laps under safety car, VSC and red flag, and whether the race was wet.

    A lap counts as neutralised only when at least NEUTRAL_SHARE of the cars recorded the code on it, so one
    car's stray status does not flag a whole race."""
    status = laps["track_status"].fillna("").astype(str)
    f = pd.DataFrame({"race_id": laps["race_id"], "lap_number": laps["lap_number"],
                      "sc": status.str.contains("4"), "vsc": status.str.contains("6"),
                      "red": status.str.contains("5"), "wet_lap": laps["compound"].isin(WET_COMPOUNDS)})
    per_lap = f.groupby(["race_id", "lap_number"])[["sc", "vsc", "red"]].mean() >= NEUTRAL_SHARE
    laps_by_code = per_lap.groupby("race_id").sum().rename(columns={"sc": "sc_laps", "vsc": "vsc_laps", "red": "red_laps"})
    out = laps_by_code.join(f.groupby("race_id")["wet_lap"].sum().rename("wet_laps")).reset_index()
    out["neutral"] = (out["sc_laps"] + out["vsc_laps"] + out["red_laps"]) > 0
    out["red"] = out["red_laps"] > 0
    out["rain_share"] = out["race_id"].map(rain_share).fillna(0.0) if rain_share is not None else 0.0
    out["wet"] = (out["wet_laps"] >= WET_LAPS_MIN) | (out["rain_share"] >= RAIN_SHARE_MIN)
    out["wet_borderline"] = out["wet"] & (out["wet_laps"] < BORDERLINE_WET_LAPS)
    return out[["race_id", "neutral", "red", "sc_laps", "vsc_laps", "red_laps", "wet_laps", "rain_share", "wet", "wet_borderline"]]


def status_diagnostics(flags: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Per season: how many races had a safety car, VSC, red flag. A season far out of line points to a data problem."""
    season = laps.drop_duplicates("race_id").set_index("race_id")["season"]
    f = flags.assign(season=flags["race_id"].map(season))
    return f.groupby("season").agg(races=("race_id", "size"), share_safety_car=("sc_laps", lambda x: (x > 0).mean()),
                                   share_vsc=("vsc_laps", lambda x: (x > 0).mean()),
                                   share_red_flag=("red_laps", lambda x: (x > 0).mean()),
                                   share_any_neutralisation=("neutral", "mean"),
                                   median_sc_laps_when_present=("sc_laps", lambda x: x[x > 0].median()),
                                   share_wet=("wet", "mean")).reset_index()


# ------------------------------------------------------------ tyre model

def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps that say something about tyres: green flag, on track, dry compound, accurate timing."""
    ok = (laps["is_accurate"].fillna(False).astype(bool) & laps["lap_time_s"].notna()
          & laps["compound"].isin(DRY) & laps["pit_in_s"].isna() & laps["pit_out_s"].isna()
          & (laps["track_status"].fillna("").astype(str) == "1") & (laps["lap_number"] > 1)
          & laps["tyre_life"].notna())
    return laps[ok]


def fit_race(d: pd.DataFrame, detail: bool = False):
    """Degradation per compound for one race. d = clean laps of that race.
    With detail=True returns (row, extra): extra holds each driver's pace effect (s, relative), the compound offsets
    (relative to the reference compound at age 0) and the wear slopes, which the simulator uses."""
    d = d[d["lap_time_s"] <= SLOW_CUT * d["lap_time_s"].median()]
    comps = []
    for c in DRY:
        age = d.loc[d["compound"] == c, "tyre_life"]
        if len(age) >= MIN_LAPS_COMPOUND and (age.quantile(0.9) - age.quantile(0.1)) >= MIN_AGE_SPREAD:
            comps.append(c)
    d = d[d["compound"].isin(comps)]
    if not comps or len(d) < MIN_LAPS_RACE or d["driver_id"].nunique() < MIN_DRIVERS:
        return None
    ref = "MEDIUM" if "MEDIUM" in comps else max(comps, key=lambda c: int((d["compound"] == c).sum()))
    codes, uniques = pd.factorize(d["driver_id"])
    lap_codes, lap_values = pd.factorize(d["lap_number"])
    nd, nl = len(uniques), len(lap_values)
    # a separate effect for every lap number absorbs everything the whole field shares on that lap:
    # fuel burn, track evolution, weather. What is left to explain is who is on which tyre, and how old.
    cols = [np.eye(nd)[codes], np.eye(nl)[lap_codes][:, 1:]]
    others = [c for c in comps if c != ref]
    for c in others:
        cols.append((d["compound"] == c).to_numpy(float)[:, None])
    for c in comps:
        cols.append(((d["compound"] == c).to_numpy(float) * d["tyre_life"].to_numpy(float))[:, None])
    X = np.hstack(cols)
    y = d["lap_time_s"].to_numpy(float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((y - X @ beta) ** 2)))
    lap_effect = np.concatenate([[0.0], beta[nd:nd + nl - 1]])
    order = np.argsort(lap_values)
    laps_sorted, effect_sorted = lap_values[order].astype(float), lap_effect[order]
    weights = np.sqrt(np.bincount(lap_codes, minlength=nl)[order])
    keep = laps_sorted >= 3
    trend = float(np.polyfit(laps_sorted[keep], effect_sorted[keep], 1, w=weights[keep])[0]) if keep.sum() >= 5 else math.nan
    base = nd + nl - 1
    offsets = {c: float(beta[base + i]) for i, c in enumerate(others)}
    offsets[ref] = 0.0
    deg = {c: float(beta[base + len(others) + i]) for i, c in enumerate(comps)}
    row = {"n_laps": len(d), "n_drivers": nd, "lap_trend_s_per_lap": trend, "rmse_s": rmse, "reference": ref}
    for c in DRY:
        row[f"deg_{c}"] = deg.get(c, math.nan)
    for c in ("SOFT", "HARD"):
        if ref == "MEDIUM" and c in deg:
            row[f"gap_{c}_vs_MEDIUM"] = (offsets[c] + deg[c] * REF_AGE) - deg["MEDIUM"] * REF_AGE
        else:
            row[f"gap_{c}_vs_MEDIUM"] = math.nan
    if detail:
        eff = pd.Series(beta[:nd], index=uniques)
        return row, {"driver_effect": eff - float(eff.median()), "offsets": offsets, "deg": deg, "reference": ref}
    return row


def tyre_by_race(laps: pd.DataFrame, rain_wet: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for race_id, d in clean_laps(laps).groupby("race_id"):
        if rain_wet is not None and bool(rain_wet.get(race_id, False)):
            continue
        fit = fit_race(d)
        if fit is None:
            continue
        first = d.iloc[0]
        rows.append({"race_id": race_id, "season": int(first["season"]), "circuit_id": first["circuit_id"],
                     "circuit_name": first["circuit_name"], **fit})
    return pd.DataFrame(rows)


def season_tyre_summary(per_race: pd.DataFrame) -> pd.DataFrame:
    if per_race.empty:
        return pd.DataFrame()
    g = per_race.groupby("season")
    out = g.agg(races=("race_id", "size"), lap_trend=("lap_trend_s_per_lap", "median"),
                gap_soft=("gap_SOFT_vs_MEDIUM", "median"), gap_hard=("gap_HARD_vs_MEDIUM", "median")).reset_index()
    for c in DRY:
        col = f"deg_{c}"
        out[f"{col}_median"] = out["season"].map(g[col].median())
        out[f"{col}_q25"] = out["season"].map(g[col].quantile(0.25))
        out[f"{col}_q75"] = out["season"].map(g[col].quantile(0.75))
        out[f"{col}_races"] = out["season"].map(g[col].count())
    return out


def circuit_tyre_summary(per_race: pd.DataFrame, min_races: int = 3) -> pd.DataFrame:
    """Median wear per compound at each circuit. A compound with fewer than min_races fits there is left blank."""
    if per_race.empty:
        return pd.DataFrame()
    g = per_race.groupby("circuit_id")
    out = g.agg(circuit_name=("circuit_name", "last"), races=("race_id", "size"),
                lap_trend=("lap_trend_s_per_lap", "median")).reset_index()
    for c, name in zip(DRY, ("soft", "medium", "hard")):
        med = out["circuit_id"].map(g[f"deg_{c}"].median())
        n = out["circuit_id"].map(g[f"deg_{c}"].count())
        out[f"deg_{name}"] = med.where(n >= min_races)
        out[f"deg_{name}_races"] = n
    out = out[out["races"] >= min_races]
    return out.sort_values("deg_medium", ascending=False).reset_index(drop=True)


# -------------------------------------------------- do the estimates mean anything?

SEED = 20241012
STEP_SEASON = 2022              # new cars and 18-inch tyres


def era_step_test(per_race: pd.DataFrame, col: str, split: int = STEP_SEASON, n_perm: int = 3000) -> dict | None:
    """Is wear different from `split` on? Era labels are shuffled between races of the same circuit, so a change in
    which circuits were raced cannot create a step."""
    d = per_race.dropna(subset=[col])
    x = d[col].to_numpy(float)
    post = (d["season"] >= split).to_numpy()
    if post.sum() < 5 or (~post).sum() < 5:
        return None
    obs = float(np.median(x[post]) - np.median(x[~post]))
    codes = d["circuit_id"].to_numpy()
    blocks = [b for b in (np.flatnonzero(codes == c) for c in np.unique(codes)) if len(b) > 1]
    rng = np.random.default_rng(SEED)
    null = np.empty(n_perm)
    for i in range(n_perm):
        lab = post.copy()
        for b in blocks:
            lab[b] = rng.permutation(post[b])
        null[i] = np.median(x[lab]) - np.median(x[~lab])
    return {"pre_median": float(np.median(x[~post])), "post_median": float(np.median(x[post])), "difference": obs,
            "races_pre": int((~post).sum()), "races_post": int(post.sum()),
            "p_value": (1 + int((np.abs(null) >= abs(obs)).sum())) / (n_perm + 1)}


def split_half_reliability(per_race: pd.DataFrame, col: str, n_rep: int = 300, min_races: int = 4,
                           adjust_season: bool = False) -> dict | None:
    """If circuits really differ in wear, two random halves of each circuit's races should rank the circuits alike.

    adjust_season: subtract each season's median first, so a circuit raced mostly after the 2022 step is not
    ranked as harsh just for when it was visited."""
    if adjust_season:
        per_race = per_race.assign(**{col: per_race[col] - per_race.groupby("season")[col].transform("median")})
    groups = [g[col].dropna().to_numpy(float) for _, g in per_race.groupby("circuit_id")]
    groups = [a for a in groups if len(a) >= min_races]
    if len(groups) < 5:
        return None
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_rep):
        a_half, b_half = [], []
        for arr in groups:
            idx = rng.permutation(len(arr))
            h = len(arr) // 2
            a_half.append(np.median(arr[idx[:h]]))
            b_half.append(np.median(arr[idx[h:2 * h]]))
        vals.append(pd.Series(a_half).corr(pd.Series(b_half), method="spearman"))
    vals = np.array(vals, dtype=float)
    return {"circuits": len(groups), "split_half_corr": float(np.nanmean(vals)),
            "split_half_low": float(np.nanpercentile(vals, 5)), "split_half_high": float(np.nanpercentile(vals, 95))}


def tyre_checks(per_race: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, name in zip(DRY, ("soft", "medium", "hard")):
        step = era_step_test(per_race, f"deg_{c}") or {}
        rel = split_half_reliability(per_race, f"deg_{c}") or {}
        adj = split_half_reliability(per_race, f"deg_{c}", adjust_season=True) or {}
        if adj:
            rel = {**rel, "split_half_corr_season_adjusted": adj["split_half_corr"]}
        if step or rel:
            rows.append({"compound": name, **step, **rel})
    return pd.DataFrame(rows)


def _normal_shrink(means: np.ndarray, n: np.ndarray, within_var: float, fit_mask: np.ndarray | None = None):
    """Normal-normal shrinkage towards the weighted mean. The between-circuit variance is estimated on the
    circuits selected by fit_mask (default all). Returns shrunk values and each circuit's reliability (0 to 1)."""
    means, n = np.asarray(means, float), np.asarray(n, float)
    fit = np.ones(len(means), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    mu = float(np.average(means[fit], weights=n[fit]))
    sampling = within_var / n
    tau2 = max(0.0, float(np.var(means[fit], ddof=1) - sampling[fit].mean())) if fit.sum() > 1 else 0.0
    rel = tau2 / (tau2 + sampling) if tau2 > 0 else np.zeros_like(means)
    return mu + rel * (means - mu), rel


def circuit_wear_shrunk(per_race: pd.DataFrame, min_races_fit: int = 2) -> pd.DataFrame:
    """Circuit wear offsets for the simulator: each race's wear minus its season's median, averaged by circuit and
    pulled towards zero in proportion to how little data the circuit has. The extremes of the raw ranking are mostly
    noise (the split-half check), and shrinkage stops the simulator from trusting them."""
    rows = []
    for c, name in zip(DRY, ("soft", "medium", "hard")):
        col = f"deg_{c}"
        d = per_race.dropna(subset=[col]).copy()
        if d.empty:
            continue
        d["adj"] = d[col] - d.groupby("season")[col].transform("median")
        g = d.groupby("circuit_id")
        agg = pd.DataFrame({"circuit_name": g["circuit_name"].last(), "races": g.size(), "raw_offset": g["adj"].mean()}).reset_index()
        dof = len(d) - d["circuit_id"].nunique()
        if dof <= 0 or len(agg) < 3:
            continue
        within = float(((d["adj"] - g["adj"].transform("mean")) ** 2).sum() / dof)
        shrunk, rel = _normal_shrink(agg["raw_offset"], agg["races"], within, (agg["races"] >= min_races_fit).to_numpy())
        agg["shrunk_offset"] = shrunk
        agg["reliability"] = rel
        agg["compound"] = name
        agg["typical_wear_s_per_lap"] = float(d[col].median()) + agg["shrunk_offset"]
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[["compound", "circuit_id", "circuit_name", "races", "raw_offset",
                                                "shrunk_offset", "reliability", "typical_wear_s_per_lap"]]


def wear_level_by_era(per_race: pd.DataFrame, split: int = STEP_SEASON) -> pd.DataFrame:
    rows = []
    for c, name in zip(DRY, ("soft", "medium", "hard")):
        col = f"deg_{c}"
        for label, mask in ((f"2018-{split - 1}", per_race["season"] < split), (f"{split}-now", per_race["season"] >= split)):
            x = per_race.loc[mask, col].dropna()
            if len(x):
                rows.append({"compound": name, "era": label, "races": len(x), "median_wear": float(x.median()),
                             "q25": float(x.quantile(.25)), "q75": float(x.quantile(.75))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- pit loss

def pit_loss(laps: pd.DataFrame) -> pd.DataFrame:
    """Time lost per green-flag stop: in-lap + out-lap minus two laps at the driver's own pace around the stop."""
    rows = []
    for (race_id, driver), g in laps.sort_values("lap_number").groupby(["race_id", "driver_id"]):
        lap = g["lap_number"].to_numpy()
        t = g["lap_time_s"].to_numpy(float)
        pin, pout = g["pit_in_s"].notna().to_numpy(), g["pit_out_s"].notna().to_numpy()
        status = g["track_status"].fillna("").astype(str).to_numpy()
        neutral = np.array([any(ch in NEUTRAL_CODES for ch in s) for s in status], dtype=bool)
        green = (status == "1") & ~pin & ~pout & g["is_accurate"].fillna(False).astype(bool).to_numpy() & ~np.isnan(t)
        n = len(g)
        for i in np.flatnonzero(pin):
            if i + 1 >= n or not pout[i + 1] or lap[i + 1] != lap[i] + 1:
                continue
            if np.isnan(t[i]) or np.isnan(t[i + 1]) or neutral[i] or neutral[i + 1]:
                continue
            before = t[max(0, i - 6):i][green[max(0, i - 6):i]]
            after = t[i + 2:i + 8][green[i + 2:i + 8]]
            refs = np.concatenate([before, after])
            if len(refs) < 4:
                continue
            loss = float(t[i] + t[i + 1] - 2 * np.median(refs))
            if PIT_LOSS_RANGE[0] <= loss <= PIT_LOSS_RANGE[1]:
                first = g.iloc[0]
                pos_arr = g["position"].to_numpy(float) if "position" in g.columns else np.full(n, np.nan)
                rows.append({"race_id": race_id, "season": int(first["season"]), "circuit_id": first["circuit_id"],
                             "circuit_name": first["circuit_name"], "driver_id": driver,
                             "in_lap": int(lap[i]), "loss_s": loss,
                             "position": float(pos_arr[i - 1]) if i >= 1 else float(pos_arr[i])})
    return pd.DataFrame(rows)


def split_wet_stops(stops: pd.DataFrame, wet: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dry stops, wet stops). Wet-race stops are kept out of the circuit table: a pit lane costs
    more time in the wet and a circuit seen only in the wet (Istanbul, Hockenheim) would otherwise
    be reported with a wet-weather number."""
    if stops.empty:
        return stops, stops
    is_wet = stops["race_id"].map(wet).fillna(False).astype(bool)
    return stops[~is_wet].reset_index(drop=True), stops[is_wet].reset_index(drop=True)


def dry_vs_wet_table(dry: pd.DataFrame, wet_stops: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, d in (("dry", dry), ("wet", wet_stops)):
        if len(d):
            rows.append({"conditions": label, "stops": len(d), "races": int(d["race_id"].nunique()),
                         "median_loss_s": float(d["loss_s"].median()), "q25_loss_s": float(d["loss_s"].quantile(.25)),
                         "q75_loss_s": float(d["loss_s"].quantile(.75))})
    return pd.DataFrame(rows)


def pit_loss_by_circuit(stops: pd.DataFrame, min_stops: int = MIN_STOPS_CIRCUIT) -> pd.DataFrame:
    if stops.empty:
        return pd.DataFrame()
    g = stops.groupby("circuit_id")["loss_s"]
    out = pd.DataFrame({"circuit_name": stops.groupby("circuit_id")["circuit_name"].last(),
                        "stops": g.size(), "races": stops.groupby("circuit_id")["race_id"].nunique(),
                        "median_loss_s": g.median(), "q25_loss_s": g.quantile(0.25),
                        "q75_loss_s": g.quantile(0.75)}).reset_index()
    race_med = stops.groupby(["circuit_id", "race_id"])["loss_s"].median().reset_index()
    rg = race_med.groupby("circuit_id")["loss_s"]
    means, n = rg.mean(), rg.size()
    dof = len(race_med) - race_med["circuit_id"].nunique()
    if dof > 0 and len(means) > 2:
        within = float(((race_med["loss_s"] - rg.transform("mean")) ** 2).sum() / dof)
        shrunk, rel = _normal_shrink(means.to_numpy(), n.to_numpy(), within, (n >= 2).to_numpy())
        out["loss_shrunk_s"] = out["circuit_id"].map(pd.Series(shrunk, index=means.index))
        out["weight_on_circuit"] = out["circuit_id"].map(pd.Series(rel, index=means.index))
    return out[out["stops"] >= min_stops].sort_values("median_loss_s").reset_index(drop=True)


# ----------------------------------------------------------- strategy mix

def _stops_per_driver(laps: pd.DataFrame, results: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    ok = flags.loc[~flags["wet"] & ~flags["red"], "race_id"]
    l = laps[laps["race_id"].isin(ok)]
    stops = (l.groupby(["race_id", "driver_id"])["stint"].nunique() - 1).rename("stops").reset_index()
    res = results[["race_id", "driver_id", "season", "position", "classified"]].drop_duplicates(["race_id", "driver_id"])
    return stops.merge(res, on=["race_id", "driver_id"], how="inner")


def stop_mix(laps, results, flags) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stops per car by season, winner's stops against the median of the top ten finishers)."""
    s = _stops_per_driver(laps, results, flags)
    fin = s[s["classified"].astype(bool)]
    if fin.empty:
        return pd.DataFrame(), pd.DataFrame()
    mix = fin.groupby("season").agg(races=("race_id", "nunique"), finishers=("stops", "size"),
                                    mean_stops=("stops", "mean"),
                                    share_0_stops=("stops", lambda x: (x == 0).mean()),
                                    share_1_stop=("stops", lambda x: (x == 1).mean()),
                                    share_2_stops=("stops", lambda x: (x == 2).mean()),
                                    share_3plus_stops=("stops", lambda x: (x >= 3).mean())).reset_index()
    rows = []
    for race_id, g in fin.groupby("race_id"):
        win = g[g["position"] == 1]
        top = g[g["position"] <= 10]
        if win.empty or len(top) < 5:
            continue
        w, med = float(win["stops"].iloc[0]), float(top["stops"].median())
        rows.append({"race_id": race_id, "season": int(g["season"].iloc[0]), "winner_stops": w,
                     "top10_median_stops": med, "fewer": w < med, "same": w == med, "more": w > med})
    wf = pd.DataFrame(rows)
    if wf.empty:
        return mix, wf
    by = wf.groupby("season").agg(races=("race_id", "size"), winner_fewer_share=("fewer", "mean"),
                                  winner_same_share=("same", "mean"), winner_more_share=("more", "mean")).reset_index()
    total = pd.DataFrame([{"season": 0, "races": len(wf), "winner_fewer_share": wf["fewer"].mean(),
                           "winner_same_share": wf["same"].mean(), "winner_more_share": wf["more"].mean()}])
    return mix, pd.concat([by, total], ignore_index=True)   # season 0 = all seasons together


# ---------------------------------------------------------------- charts

MIN_RACES_POINT = 6


def spread_labels(ends: list[tuple[float, float, str]], ylim: tuple[float, float]) -> list[tuple[float, float, str]]:
    """Line-end labels (y, x, text) pushed apart vertically so that none overlap. Labels sharing a
    last x are separated by at least 4% of the axis height; order is kept."""
    if not ends:
        return []
    gap = 0.04 * (ylim[1] - ylim[0])
    order = sorted(range(len(ends)), key=lambda i: ends[i][0])
    ys = [ends[i][0] for i in order]
    for k in range(1, len(ys)):
        if abs(ends[order[k]][1] - ends[order[k - 1]][1]) < 0.5 and ys[k] - ys[k - 1] < gap:
            ys[k] = ys[k - 1] + gap
    mid_shift = (sum(ys) - sum(ends[i][0] for i in order)) / len(ys)
    out = [None] * len(ends)
    for pos, i in enumerate(order):
        out[i] = (ys[pos] - mid_shift, ends[i][1], ends[i][2])
    return out


def chart_degradation(summary: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    s = summary.sort_values("season")
    fig, ax = _frame(plt, "How fast each tyre loses pace",
                     "Seconds lost per lap of tyre age: median across dry races, shaded = middle half of races. "
                     f"A season is drawn for a compound only when at least {MIN_RACES_POINT} races support it. "
                     "A compound is seen only where teams chose it.")
    labels = {"SOFT": "Soft", "MEDIUM": "Medium", "HARD": "Hard"}
    x = s["season"].to_numpy()
    ends = []
    for i, c in enumerate(DRY):
        ok = (s[f"deg_{c}_races"] >= MIN_RACES_POINT).to_numpy()
        if not ok.any():
            continue
        med, lo, hi = (np.where(ok, s[f"deg_{c}_{k}"].to_numpy(float), np.nan) for k in ("median", "q25", "q75"))
        ax.fill_between(x, lo, hi, color=SERIES[i], alpha=0.12, lw=0, zorder=2)
        ax.plot(x, med, color=SERIES[i], lw=2.2, marker="o", ms=4, label=labels[c], zorder=3)
        last = np.flatnonzero(ok)[-1]
        ends.append((float(med[last]), float(x[last]), labels[c]))
    for y, xx, text in spread_labels(ends, ax.get_ylim()):
        ax.text(xx + 0.15, y, text, color=TEXT, va="center", fontsize=9)
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.set_xlabel("Season")
    ax.set_ylabel("Pace lost per lap of tyre age (s)")
    ax.set_xlim(s["season"].min() - 0.3, s["season"].max() + 1.2)
    ax.set_xticks(s["season"])
    fig.subplots_adjust(bottom=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="upper center", fontsize=9, labelcolor=TEXT, ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(path)
    plt.close(fig)


def chart_pit_loss(table: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    t = table.sort_values("median_loss_s", ascending=True).reset_index(drop=True)
    height = max(3.6, 0.3 * len(t) + 2.2)
    fig, ax = _frame(plt, "How much time a pit stop costs, circuit by circuit",
                     "In-lap plus out-lap, against two laps at the driver's own pace, dry-race green-flag stops only. "
                     "Dot = median, line = middle half of stops.", size=(10.5, height))
    fig.subplots_adjust(top=1 - 1.5 / height, left=0.40, right=0.96, bottom=0.9 / height)
    y = np.arange(len(t))
    ax.hlines(y, t["q25_loss_s"], t["q75_loss_s"], color=AXIS, lw=2.4, zorder=2)
    ax.scatter(t["median_loss_s"], y, s=40, color=BLUE, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={k})" for n, k in zip(t["circuit_name"], t["stops"])], fontsize=8.5)
    ax.set_xlabel("Time lost per stop (s)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------- findings

def findings(per_race, season, circuit, stops_by_circuit, mix, winner_stops, checks=None, wear_shrunk=None,
             dry_wet=None) -> list[str]:
    L: list[str] = []
    if len(per_race):
        L.append(f"TYRE MODEL: fitted in {len(per_race)} dry races, {int(season['season'].min())} to {int(season['season'].max())}")
        L.append(f"  lap-by-lap field trend (fuel burn plus track evolution): median "
                 f"{per_race['lap_trend_s_per_lap'].median():+.3f} s per lap; it should be negative")
        both = per_race.dropna(subset=["deg_SOFT", "deg_HARD"])
        if len(both):
            L.append(f"  SANITY: soft wears faster than hard in {(both['deg_SOFT'] > both['deg_HARD']).mean():.0%} of "
                     f"{len(both)} races that have both, and a hard tyre is slower than a medium at age {REF_AGE} in "
                     f"{(per_race['gap_HARD_vs_MEDIUM'].dropna() > 0).mean():.0%} of races. Near 50% would mean the model "
                     "is not separating compounds")
        for c, name in zip(DRY, ("soft", "medium", "hard")):
            x = per_race[f"deg_{c}"].dropna()
            if len(x):
                L.append(f"  {name}: loses {x.median():+.3f} s per lap of age (middle half of races {x.quantile(.25):+.3f} to "
                         f"{x.quantile(.75):+.3f}, {len(x)} races)")
        for col, name in (("gap_SOFT_vs_MEDIUM", "soft"), ("gap_HARD_vs_MEDIUM", "hard")):
            x = per_race[col].dropna()
            if len(x):
                L.append(f"  at {REF_AGE} laps of age a {name} tyre is {x.median():+.2f} s per lap against a medium "
                         f"(negative = faster)")
    if len(per_race):
        hard_slower = (per_race["gap_HARD_vs_MEDIUM"].dropna() > 0).mean() if per_race["gap_HARD_vs_MEDIUM"].notna().any() else math.nan
        if hard_slower < PACE_GAP_TRUST:
            L.append(f"  WARNING: a hard tyre is slower than a medium in only {hard_slower:.0%} of races (below {PACE_GAP_TRUST:.0%}), "
                     "so the compound pace gaps are not separated and should not be used. The wear slopes are the usable output")
    if checks is not None and len(checks):
        L.append("DO THE ESTIMATES MEAN ANYTHING?")
        for _, r in checks.iterrows():
            if pd.notna(r.get("difference")):
                L.append(f"  {r['compound']}: wear from {STEP_SEASON} on is {r['difference']:+.3f} s per lap against before "
                         f"({r['pre_median']:.3f} to {r['post_median']:.3f}; {int(r['races_pre'])} and {int(r['races_post'])} races; "
                         f"permutation inside circuits p = {r['p_value']:.4f})")
        for _, r in checks.iterrows():
            if pd.notna(r.get("split_half_corr")):
                L.append(f"  {r['compound']}: circuits ranked by wear from two random halves of their races agree with rank "
                         f"correlation {r['split_half_corr']:.2f} (90% range {r['split_half_low']:.2f} to {r['split_half_high']:.2f}, "
                         f"{int(r['circuits'])} circuits); near 0 would mean circuit rankings are noise"
                         + (f"; {r['split_half_corr_season_adjusted']:.2f} once each season's level is removed first"
                            if pd.notna(r.get("split_half_corr_season_adjusted")) else ""))
    if wear_shrunk is not None and len(wear_shrunk):
        m = wear_shrunk[(wear_shrunk["compound"] == "medium") & (wear_shrunk["races"] >= 3)].sort_values("shrunk_offset")
        if len(m) >= 2:
            hi, lo = m.iloc[-1], m.iloc[0]
            L.append(f"  medium wear by circuit, after removing season level and shrinking small samples (circuits with 3+ races): "
                     f"hardest {hi['circuit_name']} ({hi['shrunk_offset']:+.3f} s/lap against a typical circuit, raw {hi['raw_offset']:+.3f}, "
                     f"{int(hi['races'])} races); gentlest {lo['circuit_name']} ({lo['shrunk_offset']:+.3f}, raw {lo['raw_offset']:+.3f}, "
                     f"{int(lo['races'])} races). Raw extremes are mostly noise: treat these as groups, not a ranking")
    elif len(circuit) >= 2:
        hi, lo = circuit.iloc[0], circuit.iloc[-1]
        L.append(f"  hardest on medium tyres: {hi['circuit_name']} ({hi['deg_medium']:+.3f} s/lap); "
                 f"gentlest: {lo['circuit_name']} ({lo['deg_medium']:+.3f})")
    if len(stops_by_circuit):
        L.append(f"PIT LOSS (dry-race green-flag stops, {int(stops_by_circuit['stops'].sum())} stops, {len(stops_by_circuit)} circuits)")
        if dry_wet is not None and len(dry_wet) == 2:
            d, w = dry_wet.set_index("conditions").loc["dry"], dry_wet.set_index("conditions").loc["wet"]
            L.append(f"  wet-race stops are kept out (median {w['median_loss_s']:.1f} s in {int(w['stops'])} wet stops against "
                     f"{d['median_loss_s']:.1f} s in the dry); see pit_loss_wet_by_circuit.csv")
        lo, hi = stops_by_circuit.iloc[0], stops_by_circuit.iloc[-1]
        L.append(f"  median across circuits {stops_by_circuit['median_loss_s'].median():.1f} s; cheapest "
                 f"{lo['circuit_name']} ({lo['median_loss_s']:.1f} s), dearest {hi['circuit_name']} ({hi['median_loss_s']:.1f} s)")
        if "loss_shrunk_s" in stops_by_circuit:
            thin = stops_by_circuit[stops_by_circuit["races"] < 3]
            if len(thin):
                L.append("  circuits with under 3 races, raw median then shrunk towards the average: " + "; ".join(
                    f"{r['circuit_name']} {r['median_loss_s']:.1f} to {r['loss_shrunk_s']:.1f} s ({int(r['races'])} races)"
                    for _, r in thin.iterrows()))
    if len(mix):
        L.append("STOPS PER CAR (classified finishers, dry races without a red flag)")
        for _, r in mix.iterrows():
            L.append(f"  {int(r['season'])}: {r['mean_stops']:.2f} stops on average ({r['share_1_stop']:.0%} one-stop, "
                     f"{r['share_2_stops']:.0%} two-stop, {r['share_3plus_stops']:.0%} three or more; {int(r['races'])} races)")
    if len(winner_stops):
        t = winner_stops[winner_stops["season"] == 0].iloc[0]
        L.append(f"WINNER'S STOPS vs the median of the top ten ({int(t['races'])} races): fewer {t['winner_fewer_share']:.0%}, "
                 f"same {t['winner_same_share']:.0%}, more {t['winner_more_share']:.0%}")
    return L


def run(con, out_dir: Path | None = None) -> list[str]:
    from . import grid_checks
    out = Path(out_dir) if out_dir else config.outputs_dir() / "tyres_and_pits"
    out.mkdir(parents=True, exist_ok=True)

    laps = load_laps(con)
    laps = normalise_compounds(laps[laps["season"] >= FIRST_SEASON])
    rain = load_rain_share(con)
    flags = race_flags(laps, rain)
    wet = flags.set_index("race_id")["wet"]
    results = grid_checks.load(con)
    results = results[results["season"] >= FIRST_SEASON]

    per_race = tyre_by_race(laps, wet)
    season = season_tyre_summary(per_race)
    circuit = circuit_tyre_summary(per_race)
    all_stops = pit_loss(laps)
    stops, wet_stops = split_wet_stops(all_stops, wet)
    by_circuit = pit_loss_by_circuit(stops)
    wet_by_circuit = pit_loss_by_circuit(wet_stops, min_stops=10)
    dry_wet = dry_vs_wet_table(stops, wet_stops)
    mix, winner = stop_mix(laps, results, flags)
    checks = tyre_checks(per_race)
    wear_shrunk = circuit_wear_shrunk(per_race)
    wear_era = wear_level_by_era(per_race)

    for name, table in [("circuit_wear_shrunk", wear_shrunk), ("wear_level_by_era", wear_era), ("tyre_by_race", per_race), ("tyre_by_season", season), ("tyre_by_circuit", circuit),
                        ("pit_loss_stops", stops), ("pit_loss_by_circuit", by_circuit),
                        ("pit_loss_wet_stops", wet_stops), ("pit_loss_wet_by_circuit", wet_by_circuit),
                        ("pit_loss_dry_vs_wet", dry_wet),
                        ("stops_per_car_by_season", mix), ("winner_stops_vs_field", winner), ("race_flags", flags), ("tyre_checks", checks),
                        ("status_by_season", status_diagnostics(flags, laps))]:
        table.to_csv(out / f"{name}.csv", index=False)

    if len(season):
        chart_degradation(season, out / "tyre_degradation_by_season.png")
    if len(by_circuit):
        chart_pit_loss(by_circuit, out / "pit_loss_by_circuit.png")

    lines = findings(per_race, season, circuit, by_circuit, mix, winner, checks, wear_shrunk, dry_wet)
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
    print(f"\nTables and charts written to {config.outputs_dir() / 'tyres_and_pits'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
