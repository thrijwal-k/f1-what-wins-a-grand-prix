"""What the Friday practice sessions say about each car's race pace, known before qualifying.

Two numbers per car and weekend, from the practice laps stage 1 stores:

  long-run pace    seconds a lap slower than the field's middle, from the stints of at least five clean consecutive laps
                   (the race-simulation runs). Within a session a model with a separate effect for every driver, every
                   tyre compound used by drivers who also used another, one shared wear slope and a track-evolution
                   step for every 15 minutes gives the driver effects; sessions are averaged, weighted by laps.
  best-lap gap     percent a car's best clean lap is slower than the session's fastest, best across sessions.

Teams run different fuel loads and engine modes in practice, so neither is the race pace; the forecast blend learns
how much each is worth from earlier weekends. A car with no usable laps takes its team-mate's value (long-run pace) or
the slow end of the field (best lap); a weekend with no practice laps at all has no practice feature.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DRY = ("SOFT", "MEDIUM", "HARD")
MIN_RUN = 5                  # clean laps in a stint to count as a long run
MIN_DRIVER_LAPS = 5          # long-run laps a driver needs in a session to get an effect
COOL_DOWN = 1.05             # laps slower than this share of the stint median are cool-down or traffic laps
BUCKET_S = 900.0             # track evolution is allowed to step every 15 minutes
MIN_BUCKET_LAPS = 10
BUCKET_RIDGE = 40.0           # shrinkage of the evolution steps, in laps' worth of evidence
MIN_COMPOUND_LAPS = 15
WEIGHT_CAP = 30              # a session counts for at most this many laps of a driver's effect
SLOW_Q = 0.9


def load_practice(con) -> pd.DataFrame:
    """All stored practice laps with a driver id; empty when the table is missing or empty."""
    cols = ["race_id", "session", "driver_id", "lap_number", "lap_time_s", "stint", "compound", "tyre_life",
            "pit_in_s", "pit_out_s", "track_status", "session_time_s"]
    has = bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'practice_laps'").fetchone()[0])
    if not has:
        return pd.DataFrame(columns=cols)
    return con.execute(f"SELECT {', '.join(cols)} FROM practice_laps WHERE driver_id IS NOT NULL").df()


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    ok = (d["lap_time_s"].notna() & d["compound"].isin(DRY) & d["pit_in_s"].isna() & d["pit_out_s"].isna()
          & (d["track_status"].fillna("").astype(str) == "1") & d["tyre_life"].notna())
    return d[ok]


def long_runs(d: pd.DataFrame) -> pd.DataFrame:
    """The laps of every stint with at least MIN_RUN clean laps left once cool-down and traffic laps are dropped."""
    d = _clean(d)
    parts = []
    for _, g in d.groupby(["driver_id", "stint"]):
        g = g[g["lap_time_s"] <= COOL_DOWN * g["lap_time_s"].median()]
        if len(g) >= MIN_RUN:
            parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]


def session_effects(d: pd.DataFrame) -> pd.DataFrame:
    """One practice session's laps -> a frame (driver_id, effect_s, laps): each driver's long-run pace relative to the median."""
    lr = long_runs(d)
    if lr.empty:
        return pd.DataFrame(columns=["driver_id", "effect_s", "laps"])
    counts = lr.groupby("driver_id").size()
    lr = lr[lr["driver_id"].isin(counts[counts >= MIN_DRIVER_LAPS].index)]
    if lr["driver_id"].nunique() < 3:
        return pd.DataFrame(columns=["driver_id", "effect_s", "laps"])
    # compounds that can be compared: used by enough laps, and linked through drivers who ran more than one
    comp_laps = lr.groupby("compound").size()
    comps = [c for c in DRY if comp_laps.get(c, 0) >= MIN_COMPOUND_LAPS]
    per_driver = lr[lr["compound"].isin(comps)].groupby("driver_id")["compound"].nunique()
    if len(comps) >= 2 and (per_driver >= 2).sum() >= 2:
        lr = lr[lr["compound"].isin(comps)]
    else:
        main = comp_laps.idxmax()
        comps = [main]
        lr = lr[lr["compound"] == main]
    drivers = sorted(lr["driver_id"].unique())
    cols = [(lr["driver_id"] == d_).to_numpy(float)[:, None] for d_ in drivers]
    for c in comps[1:]:
        cols.append((lr["compound"] == c).to_numpy(float)[:, None])
    cols.append(lr["tyre_life"].to_numpy(float)[:, None])
    n_fixed = len(cols)
    if lr["session_time_s"].notna().all() and len(lr):
        b = ((lr["session_time_s"] - lr["session_time_s"].min()) // BUCKET_S).astype(int).to_numpy()
        for k in sorted(set(b))[1:]:
            if (b == k).sum() >= MIN_BUCKET_LAPS:
                cols.append((b == k).astype(float)[:, None])
    X = np.hstack(cols)
    y = lr["lap_time_s"].to_numpy(float)
    # the track-evolution steps are shrunk toward zero, so that when cars happen to run in separate blocks of the session
    # the steps cannot soak up the drivers' differences
    n_ridge = X.shape[1] - n_fixed
    if n_ridge > 0:
        pen = np.zeros((n_ridge, X.shape[1]))
        pen[np.arange(n_ridge), n_fixed + np.arange(n_ridge)] = math.sqrt(BUCKET_RIDGE)
        X = np.vstack([X, pen])
        y = np.concatenate([y, np.zeros(n_ridge)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    eff = beta[:len(drivers)]
    eff = eff - np.median(eff)
    return pd.DataFrame({"driver_id": drivers, "effect_s": eff,
                         "laps": [int((lr["driver_id"] == d_).sum()) for d_ in drivers]})


def best_gaps(d: pd.DataFrame) -> pd.Series:
    """Each driver's best clean lap in percent over the session's fastest, the smallest across sessions."""
    out: dict[str, float] = {}
    for _, s in _clean(d).groupby("session"):
        best = s.groupby("driver_id")["lap_time_s"].min()
        gap = 100.0 * (best / best.min() - 1.0)
        for drv, v in gap.items():
            out[drv] = min(out.get(drv, math.inf), float(v))
    return pd.Series(out, dtype=float)


def race_features(d: pd.DataFrame) -> dict:
    """One weekend's practice laps -> {driver_id: (long_run_s, long_run_laps, best_gap_pct)}, NaN where there is nothing."""
    eff = []
    for s, g in d.groupby("session"):
        e = session_effects(g)
        if len(e):
            eff.append(e)
    long_s, long_n = {}, {}
    if eff:
        e = pd.concat(eff, ignore_index=True)
        e["w"] = e["laps"].clip(upper=WEIGHT_CAP)
        for drv, g in e.groupby("driver_id"):
            long_s[drv] = float((g["effect_s"] * g["w"]).sum() / g["w"].sum())
            long_n[drv] = int(g["laps"].sum())
        mid = float(np.median(list(long_s.values())))
        long_s = {k: v - mid for k, v in long_s.items()}
    best = best_gaps(d)
    out = {}
    for drv in set(long_s) | set(best.index):
        out[drv] = (long_s.get(drv, math.nan), long_n.get(drv, 0), float(best.get(drv, math.nan)))
    return out


def features(prac: pd.DataFrame) -> dict[int, dict]:
    """race_id -> race_features, for every weekend that has practice laps."""
    out = {}
    if prac is None or prac.empty:
        return out
    for race_id, g in prac.groupby("race_id"):
        f = race_features(g)
        if f:
            out[int(race_id)] = f
    return out


def fill(feat: dict | None, drivers: list, constructors: list):
    """Arrays for one race's cars: (long_run_s, best_gap_pct, long_run_laps), with the fill rules in the module note.
    Returns (None, None, None) when the weekend has no practice features."""
    if not feat:
        return None, None, None
    n = len(drivers)
    lng = np.array([feat.get(d, (math.nan, 0, math.nan))[0] for d in drivers], float)
    cnt = np.array([feat.get(d, (math.nan, 0, math.nan))[1] for d in drivers], float)
    bst = np.array([feat.get(d, (math.nan, 0, math.nan))[2] for d in drivers], float)
    if not np.isfinite(lng).any() and not np.isfinite(bst).any():
        return None, None, None
    cons = np.array(constructors)

    def _fill(v, fallback):
        v = v.copy()
        for i in np.flatnonzero(~np.isfinite(v)):
            mates = [j for j in np.flatnonzero(np.isfinite(v)) if cons[j] == cons[i]]
            v[i] = float(np.mean(v[mates])) if mates else fallback
        return v

    lng = _fill(lng, 0.0)
    slow = float(np.nanquantile(bst[np.isfinite(bst)], SLOW_Q)) if np.isfinite(bst).any() else 0.0
    bst = _fill(bst, slow)
    return lng, bst, cnt
