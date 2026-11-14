"""Stage 3, part 1: turn the real races into simulator inputs, and fit the few settings the data cannot give directly.

Run:  python -m src.sim.calibrate
Writes outputs/sim_calibration/ (tables, findings.txt, params.json, truth.pkl).

For every dry race with usable lap data (2018 on, no red flag) this builds a `RaceTruth`: the cars, grid, each car's pace
(from the same per-race regression as stage 2: lap time on driver, lap number, compound and tyre age), what it really did
(stops, compounds, retirements, safety-car laps) and how it really finished. A truth can be handed to the simulator
with everything as it happened (level 0) or with parts replaced by samples (levels 1 and 2, in replay.py and forecast.py).

Settings that come from the data: tyre wear by compound (era level plus the shrunk circuit offset from stage 2), compound
offsets (era medians), the fuel/track trend, pit loss by circuit (shrunk), the safety-car stop rate and its cheaper stop.
Settings that are fitted, on 2022-24 races only, by coordinate search: how hard it is to pass (`thr`), the lap-1 noise
(`sigma_start`), the grid spacing (`slot_gap`) and how much of the measured lap noise is kept. The target is two measured things:
the grip-decay curve (rank correlation of grid and running order through the race) and the number of on-track passes per
car-lap. Then each circuit gets its own passing threshold, set so the simulated pass rate matches that circuit's measured
pass rate in 2018-24 races (shrunk toward the general value when a circuit has few races). The spread of what a stop costs
is measured from real stops. Dirty air (the lap time a car loses right behind another) is measured from real laps and its multiple is searched with the passing settings. 2025 on is held out, and nothing about pole wins or how pole is lost is fitted
(replay.py reports those as checks). Since version 2.5 the undercut is fitted on the training years (the share of a stop paid on the out-lap,
against the undercut advantage at close and wide stop gaps); replay.py checks it again on every year. Since version 2.7 the tyre warm-up (the extra time on the first laps of a new set) is measured from real laps by era, and the lap-1 fit
also picks a chance and cost of a bad start from lap-1 positions.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from ..analysis import race_flow, tyres
from ..analysis.race_flow import FRACTIONS
from . import engine as E
from . import practice as P
from .engine import BIG, Params, spearman_masked

FIT_SEASONS = (2022, 2024)         # the races the passing and start settings are fitted on
ERA_SPLIT = 2022
MIN_CLEAN_LAPS = 8                 # fewer clean laps than this and a car's pace is taken from its team-mate
MAX_STINTS = 5
DEFAULT_PIT_LOSS = 22.0


def split_of(season: int) -> str:
    if FIT_SEASONS[0] <= season <= FIT_SEASONS[1]:
        return "train"
    return "holdout" if season > FIT_SEASONS[1] else "earlier era"


def era_of(season: int) -> str:
    return "2018-21" if season < ERA_SPLIT else "2022-now"


# ------------------------------------------------------------------ truth

@dataclass
class RaceTruth:
    race_id: int
    season: int
    circuit_id: str
    n_laps: int
    drivers: list
    constructors: list
    slot: np.ndarray          # (n,) unique grid slots 1..n
    pace: np.ndarray          # (n,) s/lap vs the field, lower is faster
    comp: np.ndarray          # (n, K+1)
    stops: np.ndarray         # (n, K)
    start_age: np.ndarray     # (n, K+1)
    retire: np.ndarray        # (n,) lap not completed, BIG if finished
    neutral: np.ndarray       # (L+1,) 0/1/2
    finish: np.ndarray        # (n,) real finishing position 1..n
    pos_actual: np.ndarray    # (L, n) real running position after each lap, NaN if unknown
    off: np.ndarray
    deg: np.ndarray
    fuel: float
    pit_loss: float
    pace_known: np.ndarray    # (n,) True where the pace came from the car's own laps
    quali_gap: np.ndarray | None = None   # (n,) first-qualifying-session time gap to the fastest, in percent; filled where missing
    practice_long: np.ndarray | None = None   # (n,) practice long-run pace vs the field's middle, s/lap; None without practice laps
    practice_best: np.ndarray | None = None   # (n,) best practice lap, percent slower than the session's fastest
    practice_laps: np.ndarray | None = None   # (n,) long-run laps behind each car's practice pace
    gap_ahead: np.ndarray | None = None   # (L, n) real seconds to the car directly ahead at the end of each lap, NaN where unknown
    warm: float = math.nan    # the era's tyre warm-up (s on the first lap after the out-lap); NaN = not measured (the simulator then uses Params.warm_amp)

    @property
    def n(self) -> int:
        return len(self.drivers)

    def spec(self) -> E.RaceSpec:
        return E.RaceSpec(n_laps=self.n_laps, slot=self.slot, pace=self.pace, comp=self.comp[None], stops=self.stops[None],
                          start_age=self.start_age[None], retire=self.retire[None], neutral=self.neutral[None],
                          off=self.off, deg=self.deg, fuel=self.fuel, pit_loss=self.pit_loss, warm=self.warm_s)

    @property
    def warm_s(self) -> float | None:
        return float(self.warm) if np.isfinite(self.warm) else None


def load_inputs(con):
    laps = tyres.load_laps(con)
    laps = tyres.normalise_compounds(laps[laps["season"] >= tyres.FIRST_SEASON])
    flags = tyres.race_flags(laps, tyres.load_rain_share(con))
    results = con.execute("""
        SELECT r.race_id, ra.season, ra.round, ra.circuit_id, r.driver_id, r.constructor_id, r.grid, r.position,
               coalesce(r.classified, FALSE) AS classified
        FROM results r JOIN races ra ON ra.race_id = r.race_id
        WHERE ra.season >= ? AND r.position IS NOT NULL""", [tyres.FIRST_SEASON]).df()
    return laps, flags, results


def load_quali(con):
    """race_id, driver_id, q1 in seconds (NaN when no time) from the qualifying table; empty when there is none."""
    from ..build_db import parse_time_str
    has = bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'qualifying'").fetchone()[0])
    if not has:
        return pd.DataFrame(columns=["race_id", "driver_id", "q1_s"])
    q = con.execute("SELECT race_id, driver_id, q1 FROM qualifying").df()
    q["q1_s"] = q["q1"].map(parse_time_str).astype(float)
    return q[["race_id", "driver_id", "q1_s"]]


def quali_gaps(quali: pd.DataFrame | None, race_id: int, drivers: list) -> np.ndarray:
    """Each driver's first-session qualifying time as a percentage above the fastest. Missing times take the 90th percentile of
    the others' gaps (a car that set no time was slow or had a problem); all zeros when the race has no qualifying data."""
    n = len(drivers)
    if quali is None or len(quali) == 0:
        return np.zeros(n)
    q = quali[quali["race_id"] == race_id].drop_duplicates("driver_id").set_index("driver_id")["q1_s"]
    t = np.array([float(q.get(d, math.nan)) for d in drivers])
    t = np.where(t > 0, t, math.nan)
    if np.isfinite(t).sum() < 5:
        return np.zeros(n)
    gap = (t / np.nanmin(t) - 1.0) * 100.0
    gap = np.where(gap > 15.0, math.nan, gap)                   # a time more than 15% off is not a pace measure
    fill = float(np.nanquantile(gap, 0.9))
    return np.where(np.isfinite(gap), gap, fill)


def neutral_laps(laps: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per race a vector indexed by lap number: 0 green, 1 VSC, 2 safety car (a lap counts when a quarter of the cars carry it)."""
    st = laps["track_status"].fillna("").astype(str)
    d = pd.DataFrame({"race_id": laps["race_id"], "lap": laps["lap_number"], "sc": st.str.contains("4"),
                      "vsc": st.str.contains("6|7")})
    share = d.groupby(["race_id", "lap"])[["sc", "vsc"]].mean()
    code = np.where(share["sc"] >= tyres.NEUTRAL_SHARE, 2, np.where(share["vsc"] >= tyres.NEUTRAL_SHARE, 1, 0))
    share = share.assign(code=code).reset_index()
    out = {}
    for race_id, g in share.groupby("race_id"):
        v = np.zeros(int(g["lap"].max()) + 1, int)
        v[g["lap"].to_numpy(int)] = g["code"].to_numpy()
        out[int(race_id)] = v
    return out


def era_tyres(per_race: pd.DataFrame, extras: dict[int, dict]) -> pd.DataFrame:
    """Era medians: compound offset (relative to MEDIUM at age 0), wear per lap by compound, and the fuel/track trend."""
    rows = []
    for race_id, ex in extras.items():
        r = per_race[per_race["race_id"] == race_id].iloc[0]
        off = ex["offsets"]
        if ex["reference"] != "MEDIUM":
            continue                              # offsets of a race not referenced to MEDIUM are not comparable
        rows.append({"race_id": race_id, "era": era_of(int(r["season"])),
                     "off_SOFT": off.get("SOFT", math.nan), "off_HARD": off.get("HARD", math.nan),
                     "deg_SOFT": ex["deg"].get("SOFT", math.nan), "deg_MEDIUM": ex["deg"].get("MEDIUM", math.nan),
                     "deg_HARD": ex["deg"].get("HARD", math.nan), "fuel": r["lap_trend_s_per_lap"]})
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame()
    out = d.groupby("era").median(numeric_only=True).drop(columns="race_id").reset_index()
    out["races"] = out["era"].map(d.groupby("era").size())
    return out


def stint_table(g: pd.DataFrame) -> list[tuple[int, int, int]]:
    """One driver's laps -> [(compound code, start age, last lap of the stint), ...]. The last stint's last lap is its final lap.
    A missing stint number takes the one before it (or after it, at the start); always returns at least one stint."""
    g = g.dropna(subset=["lap_number"]).sort_values("lap_number").copy()
    if g.empty:
        return [(1, 0, 1)]
    g["stint"] = g["stint"].ffill().bfill().fillna(1)
    out = []
    for _, s in g.groupby("stint", sort=True):
        comp = s["compound"].dropna()
        name = comp.mode().iloc[0] if len(comp) else "MEDIUM"
        code = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}.get(name, 1)
        life = s["tyre_life"].dropna()
        age = max(0, int(life.iloc[0]) - 1) if len(life) else 0
        out.append((code, age, int(s["lap_number"].max())))
    return out


def real_gaps(g: pd.DataFrame, drivers: list, n_laps: int) -> np.ndarray | None:
    """(L, n) seconds between each car and the car directly ahead of it (by running position) as they cross the line at the end of
    each lap, from the race-clock times of the line crossings. None when the race has too few of those times."""
    if "lap_end_s" not in g.columns or g["lap_end_s"].notna().mean() < 0.5:
        return None
    d = g.drop_duplicates(["driver_id", "lap_number"])
    end = d.pivot(index="lap_number", columns="driver_id", values="lap_end_s").reindex(range(1, n_laps + 1))
    pos = d.pivot(index="lap_number", columns="driver_id", values="position").reindex(range(1, n_laps + 1))
    out = np.full((n_laps, len(drivers)), np.nan)
    col = {dr: i for i, dr in enumerate(drivers)}
    for li in range(n_laps):
        e, p = end.iloc[li].dropna(), pos.iloc[li].dropna()
        ids = [x for x in p.index if x in e.index and x in col]
        if len(ids) < 2:
            continue
        ids.sort(key=lambda x: p[x])
        for a, b in zip(ids[:-1], ids[1:]):
            if p[b] - p[a] == 1.0:
                gap = float(e[b] - e[a])
                if gap > 0:
                    out[li, col[b]] = gap
    return out if np.isfinite(out).sum() > 5 * n_laps else None


def build_truths(laps, flags, results, per_race, extras, circuit_wear, pit_by_circuit, era_t, quali=None, practice=None) -> tuple[list[RaceTruth], pd.DataFrame]:
    """Dry races without a red flag, with a usable tyre fit and a full set of dry compounds."""
    neutral = neutral_laps(laps)
    wet = flags.set_index("race_id")["wet"]
    red = flags.set_index("race_id")["red"]
    era_by = era_t.set_index("era") if len(era_t) else pd.DataFrame()
    shrunk = {}
    if len(circuit_wear):
        for _, r in circuit_wear.iterrows():
            shrunk[(r["compound"], r["circuit_id"])] = float(r["shrunk_offset"])
    pit_med = {}
    if len(pit_by_circuit):
        col = "loss_shrunk_s" if "loss_shrunk_s" in pit_by_circuit.columns else "median_loss_s"
        pit_med = dict(zip(pit_by_circuit["circuit_id"], pit_by_circuit[col]))
    global_pit = float(np.median(list(pit_med.values()))) if pit_med else DEFAULT_PIT_LOSS
    clean = tyres.clean_laps(laps)
    clean_n = clean.groupby(["race_id", "driver_id"]).size()
    res_by = {rid: g for rid, g in results.groupby("race_id")}
    truths, skipped = [], []
    for race_id, g in laps.groupby("race_id"):
        why = None
        if bool(wet.get(race_id, False)):
            why = "wet"
        elif bool(red.get(race_id, False)):
            why = "red flag"
        elif race_id not in extras:
            why = "no tyre fit"
        elif race_id not in res_by:
            why = "no results"
        elif g["compound"].isin(["UNKNOWN"]).any():
            why = "unknown compound"
        if why:
            skipped.append({"race_id": race_id, "reason": why})
            continue
        res = res_by[race_id].drop_duplicates("driver_id")
        season = int(g["season"].iloc[0])
        circuit = g["circuit_id"].iloc[0]
        n_laps = int(g["lap_number"].max())
        drivers = list(res["driver_id"])
        n = len(drivers)
        if n < 10:
            skipped.append({"race_id": race_id, "reason": "too few cars"})
            continue
        # grid slots, unique: grid 0 (pit lane) goes last; ties are broken by the finishing order
        grid = res["grid"].to_numpy(float)
        grid = np.where(grid > 0, grid, np.nanmax(grid) + 1)
        slot = np.argsort(np.argsort(grid + 1e-3 * res["position"].to_numpy(float), kind="stable"), kind="stable") + 1
        finish = np.argsort(np.argsort(res["position"].to_numpy(float), kind="stable"), kind="stable") + 1
        # pace: each car's own effect from the race fit; thin or missing ones from the team-mate, else the slow end
        ex = extras[race_id]
        eff = ex["driver_effect"]
        counts = np.array([int(clean_n.get((race_id, d), 0)) for d in drivers])
        pace = np.array([float(eff.get(d, math.nan)) for d in drivers])
        known = np.isfinite(pace) & (counts >= MIN_CLEAN_LAPS)
        cons = res["constructor_id"].to_numpy()
        fallback = float(np.nanquantile(pace[known], 0.75)) if known.any() else 0.0
        for i in np.flatnonzero(~known):
            mates = [j for j in np.flatnonzero(known) if cons[j] == cons[i]]
            pace[i] = float(np.mean(pace[mates])) if mates else fallback
        # what the cars did
        by_driver = {d: x for d, x in g.groupby("driver_id")}
        stints, ret, done = [], np.full(n, BIG), np.zeros(n, int)
        for i, d in enumerate(drivers):
            x = by_driver.get(d)
            if x is None or x["lap_number"].isna().all():
                stints.append([(1, 0, 1)])
                ret[i], done[i] = 1, 0
                continue
            st = stint_table(x)[:MAX_STINTS]
            stints.append(st)
            done[i] = int(x["lap_number"].max())
            if (not bool(res["classified"].iloc[i])) and done[i] < n_laps - 1:
                ret[i] = done[i] + 1
        K = max(1, max(len(s) for s in stints) - 1)
        comp = np.full((n, K + 1), 1, int)
        stops = np.full((n, K), BIG, int)
        sage = np.zeros((n, K + 1))
        for i, s in enumerate(stints):
            for j, (cc, age, last) in enumerate(s):
                comp[i, j], sage[i, j] = cc, age
                if j < len(s) - 1:
                    stops[i, j] = last
            comp[i, len(s):] = s[-1][0]
        # tyres: era level plus the shrunk circuit offset; offsets and fuel are era medians
        era = era_of(season)
        if era not in era_by.index:
            skipped.append({"race_id": race_id, "reason": "no era tyre medians"})
            continue
        e = era_by.loc[era]
        deg = np.array([max(0.005, float(e[f"deg_{c}"]) + shrunk.get((c.lower(), circuit), 0.0)) for c in E.COMPOUNDS])
        off = np.array([float(e["off_SOFT"]), 0.0, float(e["off_HARD"])])
        pos_w = g.drop_duplicates(["driver_id", "lap_number"]).pivot(index="lap_number", columns="driver_id", values="position")
        pos_actual = np.full((n_laps, n), np.nan)
        for i, d in enumerate(drivers):
            if d in pos_w.columns:
                ser = pos_w[d].reindex(range(1, n_laps + 1))
                pos_actual[:, i] = ser.to_numpy(float)
        nv = neutral.get(race_id, np.zeros(n_laps + 1, int))
        nv = np.pad(nv, (0, max(0, n_laps + 1 - len(nv))))[:n_laps + 1]
        truths.append(RaceTruth(race_id=int(race_id), season=season, circuit_id=circuit, n_laps=n_laps, drivers=drivers,
                                constructors=list(cons), slot=slot, pace=pace, comp=comp, stops=stops, start_age=sage,
                                retire=ret, neutral=nv, finish=finish, pos_actual=pos_actual, off=off, deg=deg,
                                fuel=float(e["fuel"]), pit_loss=float(pit_med.get(circuit, global_pit)),
                                pace_known=known, quali_gap=quali_gaps(quali, int(race_id), drivers),
                                gap_ahead=real_gaps(g, drivers, n_laps),
                                **dict(zip(("practice_long", "practice_best", "practice_laps"),
                                           P.fill((practice or {}).get(int(race_id)), drivers, list(cons))))))
    return truths, pd.DataFrame(skipped)


# ------------------------------------------------------- measured: noise by position, cover stops

NOISE_BUCKETS = (("top 3", 1, 3), ("4 to 10", 4, 10), ("11 and below", 11, 99))
MIN_STINT_LAPS = 8


def lap_noise_by_position(laps: pd.DataFrame, wet_races=()) -> pd.DataFrame:
    """Lap-to-lap noise of clean laps by running position. Each lap's residual is its distance from the stint's straight-line trend
    minus what the whole field did on that lap; the spread is robust (1.4826 x median absolute deviation). One row per bucket."""
    d = tyres.clean_laps(laps)
    d = d[~d["race_id"].isin(set(wet_races)) & d["position"].notna()]
    d = d[d["lap_time_s"] <= tyres.SLOW_CUT * d.groupby("race_id")["lap_time_s"].transform("median")]
    parts = []
    for _, g in d.groupby(["race_id", "driver_id", "stint"]):
        if len(g) < MIN_STINT_LAPS:
            continue
        x, y = g["lap_number"].to_numpy(float), g["lap_time_s"].to_numpy(float)
        res = y - np.polyval(np.polyfit(x, y, 1), x)
        parts.append(pd.DataFrame({"race_id": g["race_id"].to_numpy(), "season": g["season"].to_numpy(), "lap": x,
                                   "position": g["position"].to_numpy(float), "res": res}))
    if not parts:
        return pd.DataFrame(columns=["bucket", "laps", "noise_sd_s", "multiplier"])
    r = pd.concat(parts, ignore_index=True)
    field = r.groupby(["race_id", "lap"])["res"].transform("median")
    cnt = r.groupby(["race_id", "lap"])["res"].transform("size")
    r = r[cnt >= 6]
    r["res"] = r["res"] - field[cnt >= 6]
    rows = []
    for name, lo, hi in NOISE_BUCKETS:
        z = r[(r["position"] >= lo) & (r["position"] <= hi)]["res"].to_numpy()
        rows.append({"bucket": name, "laps": len(z),
                     "noise_sd_s": float(1.4826 * np.median(np.abs(z - np.median(z)))) if len(z) else math.nan})
    out = pd.DataFrame(rows)
    ok = out["noise_sd_s"].notna() & (out["laps"] > 0)
    if ok.sum() == len(out):
        mean = float((out["laps"] * out["noise_sd_s"]).sum() / out["laps"].sum())
        out["multiplier"] = (out["noise_sd_s"] / mean).clip(0.5, 1.5)
    else:
        out["multiplier"] = 1.0
    r["bucket"] = pd.cut(r["position"], [0, 3, 10, 99], labels=[b[0] for b in NOISE_BUCKETS])
    by = r.groupby(["season", "bucket"], observed=True)["res"].agg(
        laps="size", noise_sd_s=lambda z: 1.4826 * np.median(np.abs(z - np.median(z)))).reset_index()
    out.attrs["by_season"] = by
    return out


def add_gap_prev(laps: pd.DataFrame) -> pd.DataFrame:
    """The laps table with a `gap_prev` column: seconds between each car and the car directly ahead of it as they crossed the line at the
    end of the previous lap (NaN for a leader, when either time is missing, or when the car ahead was not the next position)."""
    out = laps.drop(columns="gap_prev", errors="ignore")
    if "lap_end_s" not in laps.columns or laps["lap_end_s"].notna().mean() < 0.5:
        return out.assign(gap_prev=np.nan)
    d = laps[["race_id", "driver_id", "lap_number", "position", "lap_end_s"]].dropna(subset=["position", "lap_end_s", "lap_number"]).copy()
    d = d.sort_values(["race_id", "lap_number", "position"])
    grp = d.groupby(["race_id", "lap_number"])
    ahead_pos, ahead_end = grp["position"].shift(1), grp["lap_end_s"].shift(1)
    gap = (d["lap_end_s"] - ahead_end).where((d["position"] - ahead_pos) == 1)
    d["gap_prev"] = gap.where(gap > 0)
    d["lap_number"] = d["lap_number"] + 1                      # the gap at the end of lap L is the gap at the start of lap L + 1
    d = d[["race_id", "driver_id", "lap_number", "gap_prev"]].drop_duplicates(["race_id", "driver_id", "lap_number"])
    return out.merge(d, on=["race_id", "driver_id", "lap_number"], how="left")


DA_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)        # gap to the car ahead at the start of a lap (s); 3 s and more is the clean-air reference
DA_RANGES = (1.5, 2.0, 3.0, 4.0)
MIN_DA_LAPS = 4000
DA_BOOT = 100


def dirty_air(laps: pd.DataFrame, wet_races=(), seed: int = 0) -> dict:
    """How much a car loses when it starts a lap close behind another. Clean laps (green, no stop) are compared within the same stint:
    each stint's straight-line trend and a constant are taken out of both the lap times and the gap-bin indicators (so only differences
    between laps of the same stint count), the field's movement on that lap is removed, and the pooled loss in each gap bin is read off
    against laps that started 3 s or more behind the car ahead. Intervals resample races. A straight line `loss x (1 - gap / range)` is then
    fitted to the bin losses for each range in DA_RANGES and the best is kept. Returns {da_loss, da_range, note, table, laps}.
    The loss is probably a little too small: a car that has just had a fast lap is also closer to the car ahead, which pushes the
    other way, so the settings later search upward from this value as well as down."""
    out = {"da_loss": 0.0, "da_range": Params().da_range, "note": "not measured", "table": pd.DataFrame(), "laps": 0}
    if "gap_prev" not in laps.columns:
        out["note"] = "no line-crossing times"
        return out
    d = tyres.clean_laps(laps)
    d = d[~d["race_id"].isin(set(wet_races)) & d["position"].notna() & d["gap_prev"].notna()]
    if d.empty:
        out["note"] = "no line-crossing times"
        return out
    d = d[d["lap_time_s"] <= tyres.SLOW_CUT * d.groupby("race_id")["lap_time_s"].transform("median")]
    edges = list(DA_BINS) + [np.inf]
    K = len(DA_BINS) - 1                                         # dirty bins; the last bin (3 s and more) is the reference
    parts = []
    for (rid, dr, st), g in d.groupby(["race_id", "driver_id", "stint"]):
        if len(g) < MIN_STINT_LAPS:
            continue
        x, y, gp = g["lap_number"].to_numpy(float), g["lap_time_s"].to_numpy(float), g["gap_prev"].to_numpy(float)
        b = np.digitize(gp, edges[1:-1])                         # 0..K
        D = (b[:, None] == np.arange(K)[None, :]).astype(float)
        if D.sum() == 0:
            continue
        X0 = np.column_stack([np.ones_like(x), x])
        proj = X0 @ np.linalg.lstsq(X0, np.column_stack([y, D]), rcond=None)[0]
        res = np.column_stack([y, D]) - proj
        parts.append((rid, x, res[:, 0], res[:, 1:], gp))
    if not parts:
        out["note"] = "too few stints"
        return out
    rid = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    lap = np.concatenate([p[1] for p in parts])
    yr = np.concatenate([p[2] for p in parts])
    Dr = np.vstack([p[3] for p in parts])
    gp = np.concatenate([p[4] for p in parts])
    f = pd.DataFrame({"rid": rid, "lap": lap, "y": yr})
    cnt = f.groupby(["rid", "lap"])["y"].transform("size").to_numpy()
    keep = cnt >= 6
    f, Dr, gp, rid = f[keep].reset_index(drop=True), Dr[keep], gp[keep], rid[keep]
    f["y"] = f["y"] - f.groupby(["rid", "lap"])["y"].transform("median")
    sd = 1.4826 * float(np.median(np.abs(f["y"] - f["y"].median())))
    y = f["y"].clip(-4 * sd, 4 * sd).to_numpy()
    out["laps"] = int(len(y))
    if len(y) < MIN_DA_LAPS:
        out["note"] = "too few laps"
        return out
    races = np.unique(rid)
    XtX = {r: Dr[rid == r].T @ Dr[rid == r] for r in races}
    Xty = {r: Dr[rid == r].T @ y[rid == r] for r in races}

    def solve(rs):
        A = sum(XtX[r] for r in rs)
        c = sum(Xty[r] for r in rs)
        return np.linalg.solve(A + 1e-9 * np.eye(K), c)

    beta = solve(races)
    rng = np.random.default_rng(seed)
    boots = np.array([solve(rng.choice(races, size=len(races))) for _ in range(DA_BOOT)])
    lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    b_all = np.digitize(gp, edges[1:-1])
    rows, centres, weights = [], [], []
    for k in range(K):
        m = b_all == k
        centres.append(float(np.mean(gp[m])) if m.any() else math.nan)
        weights.append(float(m.sum()))
        rows.append({"gap_from_s": edges[k], "gap_to_s": edges[k + 1], "laps": int(m.sum()), "mean_gap_s": centres[-1],
                     "loss_s": float(beta[k]), "low_s": float(lo[k]), "high_s": float(hi[k])})
    tab = pd.DataFrame(rows)
    c, w, bk = np.array(centres), np.array(weights), np.array(beta)
    ok = np.isfinite(c) & (w > 0)
    best = None
    for R in DA_RANGES:
        shape = np.clip(1.0 - c[ok] / R, 0.0, 1.0)
        denom = float((w[ok] * shape * shape).sum())
        L = float((w[ok] * shape * bk[ok]).sum() / denom) if denom > 0 else 0.0
        sse = float((w[ok] * (bk[ok] - L * shape) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, R, L)
    _, R, L = best
    out["table"] = tab
    if L < 0.02:
        out["note"] = "no dirty-air loss found"
        return out
    out.update(da_loss=float(L), da_range=float(R), note="measured")
    return out


DEP_LAGS = (0, 1, 2, 3, 5, 8, 12)
DEP_RHO = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98)
MIN_DEP_PAIRS = 3000


def lap_dependence(laps: pd.DataFrame, wet_races=(), da: dict | None = None) -> dict:
    """How long a good or bad spell lasts. Clean laps' residuals around each stint's straight-line trend (field movement removed) are
    multiplied in pairs 0, 1, 2, 3, 5, 8 and 12 laps apart. A straight-line fit absorbs part of any slow wander, so the products are not
    the covariances themselves: for a model of drift (variance s2, carrying over a share rho of itself each lap) plus white noise (s_w2),
    the products the fit would leave are computed stint by stint through its hat matrix, and (s2, s_w2) are found by non-negative least
    squares for each rho on a grid, keeping the rho that fits best. Returns the numbers and the drift settings (drift_sd 0 when the
    data show no lasting spells). cov_lag1 and cov_lag5 are the raw mean products, before that correction.
    With `da` (the result of dirty_air) each lap time first has its dirty-air loss taken off, so traffic that lasts for several laps
    is not counted as a drift of the car's own pace. `capped` says the drift variance hit its ceiling of 0.7 of the total."""
    from scipy.optimize import nnls
    out = {"stints": 0, "pairs_lag1": 0, "pairs_lag5": 0, "variance": math.nan, "cov_lag1": math.nan, "cov_lag5": math.nan,
           "drift_sd": 0.0, "drift_rho": Params().drift_rho, "white_sd": math.nan, "note": "not measured", "capped": False,
           "dirty_air_removed": bool(da and da.get("da_loss", 0) > 0 and "gap_prev" in laps.columns)}
    d = tyres.clean_laps(laps)
    d = d[~d["race_id"].isin(set(wet_races)) & d["position"].notna()]
    if d.empty:
        return out
    d = d[d["lap_time_s"] <= tyres.SLOW_CUT * d.groupby("race_id")["lap_time_s"].transform("median")]
    parts = []
    for (rid, dr, st), g in d.groupby(["race_id", "driver_id", "stint"]):
        if len(g) < MIN_STINT_LAPS:
            continue
        x, y = g["lap_number"].to_numpy(float), g["lap_time_s"].to_numpy(float)
        if out["dirty_air_removed"]:
            gp = np.nan_to_num(g["gap_prev"].to_numpy(float), nan=np.inf)
            y = y - da["da_loss"] * np.clip(1.0 - gp / da["da_range"], 0.0, 1.0)
        parts.append(pd.DataFrame({"race_id": rid, "driver_id": dr, "stint": st, "lap": x, "res": y - np.polyval(np.polyfit(x, y, 1), x)}))
    if not parts:
        return out
    r = pd.concat(parts, ignore_index=True)
    cnt = r.groupby(["race_id", "lap"])["res"].transform("size")
    r = r[cnt >= 6].copy()
    r["res"] = r["res"] - r.groupby(["race_id", "lap"])["res"].transform("median")
    sd = 1.4826 * float(np.median(np.abs(r["res"] - r["res"].median())))
    r["res"] = r["res"].clip(-4 * sd, 4 * sd)
    stints = []
    for _, g in r.groupby(["race_id", "driver_id", "stint"]):
        g = g.sort_values("lap")
        if len(g) < MIN_STINT_LAPS:
            continue
        x = g["lap"].to_numpy(float)
        xc = x - x.mean()
        sxx = float((xc ** 2).sum())
        if sxx > 0:
            stints.append((x, g["res"].to_numpy(float), np.eye(len(x)) - (1.0 / len(x) + np.outer(xc, xc) / sxx)))
    if not stints:
        return out
    masks = [[(x[None, :] - x[:, None]) == h for h in DEP_LAGS] for x, _, _ in stints]
    obs = np.zeros(len(DEP_LAGS))
    pairs = np.zeros(len(DEP_LAGS))
    for (x, e, M), mk in zip(stints, masks):
        pe = np.outer(e, e)
        for j, m in enumerate(mk):
            obs[j] += float((pe * m).sum()) * (2.0 if DEP_LAGS[j] > 0 else 1.0)       # a pair counts both ways round
            pairs[j] += float(m.sum()) * (2.0 if DEP_LAGS[j] > 0 else 1.0)
    out.update(stints=len(stints), pairs_lag1=int(pairs[DEP_LAGS.index(1)] / 2), pairs_lag5=int(pairs[DEP_LAGS.index(5)] / 2),
               cov_lag1=float(obs[DEP_LAGS.index(1)] / pairs[DEP_LAGS.index(1)]), cov_lag5=float(obs[DEP_LAGS.index(5)] / pairs[DEP_LAGS.index(5)]))
    if pairs.min() < MIN_DEP_PAIRS:
        out["note"] = "too few pairs of laps"
        return out
    B = np.zeros(len(DEP_LAGS))
    for (x, e, M), mk in zip(stints, masks):
        for j, m in enumerate(mk):
            B[j] += float((M * m).sum()) * (2.0 if DEP_LAGS[j] > 0 else 1.0)
    best = None
    for rho in DEP_RHO:
        A = np.zeros(len(DEP_LAGS))
        for (x, e, M), mk in zip(stints, masks):
            P = M @ (rho ** np.abs(x[None, :] - x[:, None])) @ M
            for j, m in enumerate(mk):
                A[j] += float((P * m).sum()) * (2.0 if DEP_LAGS[j] > 0 else 1.0)
        X = np.column_stack([A / pairs, B / pairs])
        coef, resid = nnls(X, obs / pairs)
        if best is None or resid < best[0]:
            best = (resid, rho, coef)
    _, rho, (s2, sw2) = best
    var = s2 + sw2
    out.update(variance=float(var), white_sd=math.sqrt(max(sw2, 0.0)))
    if s2 <= 0 or s2 < 0.03 * var:
        out["note"] = "no lasting spells found"
        return out
    out["capped"] = bool(s2 > 0.7 * var)
    s2 = min(s2, 0.7 * var)
    out.update(drift_sd=math.sqrt(s2), drift_rho=float(rho), white_sd=math.sqrt(max(var - s2, 0.0)), note="measured")
    return out


def cover_rates(truths: list[RaceTruth], window: int = Params().sc_pit_window, min_age: int = Params().sc_min_age) -> dict:
    """Does the car directly ahead of a car that stops on a green lap stop on the next lap? Among cars whose own stop is due soon
    (same eligibility as for safety cars), the share that stop next lap when the car right behind them has just stopped, against
    the share when it has not. cover_prob = (exposed - unexposed) / (1 - unexposed), floored at zero."""
    ex_n = ex_k = un_n = un_k = 0
    for t in truths:
        nv, L, n = t.neutral, t.n_laps, t.n
        stop_laps = [set(int(v) for v in t.stops[i] if v < BIG) for i in range(n)]
        for lap in range(3, L - 1):
            if nv[lap - 1] or nv[lap] or nv[lap + 1]:
                continue
            before = t.pos_actual[lap - 2]
            stopper_pos = {int(before[i]) for i in range(n) if lap in stop_laps[i] and np.isfinite(before[i])}
            for y in range(n):
                if t.retire[y] <= lap + 1 or lap in stop_laps[y] or not np.isfinite(before[y]):
                    continue
                past = [v for v in stop_laps[y] if v < lap]
                future = [v for v in stop_laps[y] if v > lap]
                if not future:
                    continue
                age = lap - (max(past) if past else 0) + t.start_age[y, len(past)]
                nxt = min(future)
                if age < min_age or nxt - lap > window:
                    continue
                hit = int(nxt == lap + 1)
                if int(before[y]) + 1 in stopper_pos:
                    ex_n += 1; ex_k += hit
                else:
                    un_n += 1; un_k += hit
    pe = ex_k / ex_n if ex_n else math.nan
    pu = un_k / un_n if un_n else math.nan
    prob = max(0.0, (pe - pu) / (1.0 - pu)) if ex_n >= 30 and un_n >= 30 and np.isfinite(pe) and np.isfinite(pu) else 0.0
    return {"exposed": ex_n, "exposed_stopped": ex_k, "unexposed": un_n, "unexposed_stopped": un_k,
            "exposed_rate": pe, "unexposed_rate": pu, "cover_prob": float(prob)}


# ------------------------------------------------------------ measured settings

def neutral_stop_rates(truths: list[RaceTruth], window: int = Params().sc_pit_window, min_age: int = Params().sc_min_age) -> dict:
    """Of the cars that had a planned stop within `window` laps and tyres at least `min_age` laps old when a neutralisation
    began, the share that came in on its first or second lap. Safety car and VSC separately."""
    out = {}
    for code, name in ((2, "sc"), (1, "vsc")):
        elig = came = 0
        for t in truths:
            nv = t.neutral
            starts = [l for l in range(1, t.n_laps + 1) if nv[l] == code and nv[l - 1] == 0]
            for l in starts:
                for i in range(t.n):
                    if t.retire[i] <= l:
                        continue
                    stops = t.stops[i][t.stops[i] < BIG]
                    past = stops[stops < l]
                    age = l - (past.max() if len(past) else 0) + t.start_age[i, len(past)]
                    # stops made on laps l or l+1 count as coming in under the neutralisation; the next planned stop is the first at or after l
                    future = stops[stops >= l]
                    if len(future) == 0 or age < min_age:
                        continue
                    nxt = future.min()
                    if nxt - l > window and nxt not in (l, l + 1):
                        continue
                    elig += 1
                    came += int(nxt in (l, l + 1))
        out[name] = {"eligible": elig, "came_in": came, "rate": came / elig if elig else math.nan}
    return out


def neutral_pit_cost(laps: pd.DataFrame, truths: list[RaceTruth]) -> pd.DataFrame:
    """Time lost on a stop under a neutralisation as a share of the circuit's green-flag loss.
    In-lap plus out-lap minus the field's median on those two laps (cars not stopping)."""
    by = {t.race_id: t for t in truths}
    rows = []
    for race_id, g in laps[laps["race_id"].isin(by)].groupby("race_id"):
        t = by[race_id]
        g = g.drop_duplicates(["driver_id", "lap_number"])
        tm = g.pivot(index="lap_number", columns="driver_id", values="lap_time_s")
        pin = g.pivot(index="lap_number", columns="driver_id", values="pit_in_s").notna()
        pout = g.pivot(index="lap_number", columns="driver_id", values="pit_out_s").notna()
        free = tm.where(~pin & ~pout)
        ref = free.median(axis=1)
        for d in tm.columns:
            for lap in pin.index[pin[d]]:
                if lap + 1 not in tm.index:
                    continue
                code = max(t.neutral[lap] if lap < len(t.neutral) else 0, t.neutral[lap + 1] if lap + 1 < len(t.neutral) else 0)
                if code == 0:
                    continue
                a, b = tm.at[lap, d], tm.at[lap + 1, d]
                if np.isnan(a) or np.isnan(b) or np.isnan(ref.get(lap, np.nan)) or np.isnan(ref.get(lap + 1, np.nan)):
                    continue
                loss = a + b - ref[lap] - ref[lap + 1]
                rows.append({"race_id": race_id, "driver_id": d, "lap": int(lap), "kind": "sc" if code == 2 else "vsc",
                             "loss_s": float(loss), "green_loss_s": t.pit_loss, "share": float(loss / t.pit_loss)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fitting

def lap_of(fraction: float, n_laps: int) -> int:
    return max(1, int(round(fraction * n_laps)))


def actual_grip(t: RaceTruth) -> np.ndarray:
    """Real rank correlation of grid and running order at each of FRACTIONS (NaN where fewer than 3 cars have a position)."""
    out = []
    for f in FRACTIONS:
        lap = lap_of(f, t.n_laps)
        p = t.pos_actual[lap - 1]
        m = np.isfinite(p)
        out.append(float(spearman_masked(t.slot[None].astype(float), np.where(m, p, 0.0)[None], m[None])[0]) if m.sum() >= 8 else math.nan)
    return np.array(out)


def sim_grip(res: E.SimResult, slot: np.ndarray, n_laps: int) -> np.ndarray:
    """(S, len(FRACTIONS)) simulated rank correlation of grid and running order among cars still running."""
    out = []
    for f in FRACTIONS:
        lap = lap_of(f, n_laps)
        p = res.pos[:, :, lap - 1].astype(float)
        m = p > 0
        out.append(spearman_masked(np.broadcast_to(slot[None].astype(float), p.shape), p, m))
    return np.stack(out, axis=1)


def pass_counts(pos: np.ndarray, pit: np.ndarray, green: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """On-track passes, counted the same way for real and simulated races.
    pos (S, L, n) running position after each lap (0 or NaN = not running), pit (S, L, n) True on a stop's in-lap,
    green (S, L) True on a lap without a safety car or VSC. For each pair of consecutive laps (from lap 2 on), among the cars that
    ran both laps on green, did not stop within a lap of them: the number of pairs that swapped order. Returns (passes, car-laps) per S."""
    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    S, L, n = pos.shape
    pit = np.asarray(pit, bool)
    ext = pit.copy()
    ext[:, 1:, :] |= pit[:, :-1, :]
    ext[:, :-1, :] |= pit[:, 1:, :]
    passes, cars = np.zeros(S), np.zeros(S)
    for m in range(1, L - 1):
        ok = (pos[:, m] > 0) & (pos[:, m + 1] > 0) & ~ext[:, m] & green[:, m, None] & green[:, m + 1, None]
        a = np.sign(pos[:, m, :, None] - pos[:, m, None, :])
        b = np.sign(pos[:, m + 1, :, None] - pos[:, m + 1, None, :])
        pair = ok[:, :, None] & ok[:, None, :]
        passes += (((a * b) < 0) & pair).sum((1, 2)) / 2.0
        cars += ok.sum(1)
    return passes, cars


def real_pit(t: RaceTruth) -> np.ndarray:
    """(1, L, n) True on the in-lap of each real stop."""
    pit = np.zeros((1, t.n_laps, t.n), bool)
    for i in range(t.n):
        for lap in t.stops[i][t.stops[i] < BIG]:
            if 1 <= lap <= t.n_laps:
                pit[0, int(lap) - 1, i] = True
    return pit


def real_passes(t: RaceTruth) -> tuple[float, float]:
    pit = real_pit(t)
    green = (t.neutral[1:t.n_laps + 1] == 0)[None, :]
    p, c = pass_counts(t.pos_actual[None], pit, green)
    return float(p[0]), float(c[0])


CLOSE_S = 1.0          # a car this close behind another is "in its dirty air" for the close-running measure


def close_counts(gap: np.ndarray, pit: np.ndarray, green: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counted the same way for real and simulated races. gap (S, L, n) seconds to the car ahead at each lap's end (NaN for the
    leader or an unknown), pit (S, L, n) True on a stop's in-lap, green (S, L). Among the cars with a gap, on a green lap that follows
    a green lap, from lap 2 on, not within a lap of their own stop: how many were within CLOSE_S of the car ahead, and how many cars."""
    gap = np.asarray(gap, float)
    S, L, n = gap.shape
    ext = np.asarray(pit, bool).copy()
    ext[:, 1:, :] |= pit[:, :-1, :]
    ext[:, :-1, :] |= pit[:, 1:, :]
    g2 = np.asarray(green, bool)
    ok = np.isfinite(gap) & ~ext
    ok[:, 0, :] = False
    ok[:, 1:, :] &= (g2[:, 1:] & g2[:, :-1])[:, :, None]
    ok[:, 0, :] = False
    close = (ok & (gap < CLOSE_S)).sum((1, 2)).astype(float)
    return close, ok.sum((1, 2)).astype(float)


GAP_BOUNDS = (0.5, 2.0, 3.5)     # seconds: the gap-to-the-car-ahead distribution is compared at these points as well as at CLOSE_S


def gap_profile_counts(gap: np.ndarray, pit: np.ndarray, green: np.ndarray, bounds=GAP_BOUNDS) -> tuple[np.ndarray, np.ndarray]:
    """Same cars and laps as close_counts; for each bound, how many were closer than it. Returns (counts (S, len(bounds)), cars (S,))."""
    gap = np.asarray(gap, float)
    S, L, n = gap.shape
    ext = np.asarray(pit, bool).copy()
    ext[:, 1:, :] |= pit[:, :-1, :]
    ext[:, :-1, :] |= pit[:, 1:, :]
    g2 = np.asarray(green, bool)
    ok = np.isfinite(gap) & ~ext
    ok[:, 0, :] = False
    ok[:, 1:, :] &= (g2[:, 1:] & g2[:, :-1])[:, :, None]
    ok[:, 0, :] = False
    cnt = np.stack([(ok & (gap < b)).sum((1, 2)) for b in bounds], axis=1).astype(float)
    return cnt, ok.sum((1, 2)).astype(float)


def real_gap_profile(truths: list[RaceTruth]) -> np.ndarray:
    """Share of the car-laps counted by close_counts that were within each of GAP_BOUNDS of the car ahead (NaN when no race has gaps)."""
    c = np.zeros(len(GAP_BOUNDS))
    k = 0.0
    for t in truths:
        if t.gap_ahead is None:
            continue
        pit = real_pit(t)
        green = (t.neutral[1:t.n_laps + 1] == 0)[None, :]
        a, b = gap_profile_counts(t.gap_ahead[None], pit, green)
        c += a[0]
        k += float(b[0])
    return c / k if k else np.full(len(GAP_BOUNDS), math.nan)


def sim_gap_profile(res: E.SimResult, n_laps: int) -> tuple[np.ndarray, np.ndarray]:
    return gap_profile_counts(res.gap.transpose(0, 2, 1), res.pit.transpose(0, 2, 1), (res.neutral[:, 1:n_laps + 1] == 0))


def real_close(t: RaceTruth) -> tuple[float, float]:
    if t.gap_ahead is None:
        return 0.0, 0.0
    pit = real_pit(t)
    green = (t.neutral[1:t.n_laps + 1] == 0)[None, :]
    c, k = close_counts(t.gap_ahead[None], pit, green)
    return float(c[0]), float(k[0])


def real_close_share(truths: list[RaceTruth]) -> float:
    c = k = 0.0
    for t in truths:
        a, b = real_close(t)
        c += a
        k += b
    return c / k if k else math.nan


def sim_close(res: E.SimResult, n_laps: int) -> tuple[np.ndarray, np.ndarray]:
    return close_counts(res.gap.transpose(0, 2, 1), res.pit.transpose(0, 2, 1), (res.neutral[:, 1:n_laps + 1] == 0))


def lap1_move(t: RaceTruth) -> float:
    """Mean places a car gains or loses on lap 1 (absolute), among cars with a lap-1 position. NaN when unknown."""
    p = t.pos_actual[0]
    m = np.isfinite(p) & (p > 0)
    return float(np.mean(np.abs(p[m] - t.slot[m]))) if m.sum() >= 8 else math.nan


def sim_lap1_move(res: E.SimResult, slot: np.ndarray) -> np.ndarray:
    p = res.pos[:, :, 0].astype(float)
    m = p > 0
    with np.errstate(invalid="ignore"):
        return (np.abs(p - slot[None]) * m).sum(1) / np.maximum(m.sum(1), 1)


def sim_passes(res: E.SimResult, n_laps: int) -> tuple[np.ndarray, np.ndarray]:
    return pass_counts(res.pos.transpose(0, 2, 1), res.pit.transpose(0, 2, 1), (res.neutral[:, 1:n_laps + 1] == 0))


BANDS = (("P1", 1, 1), ("P2-3", 2, 3), ("P4-10", 4, 10), ("P11 and below", 11, 99))   # running positions the checks are split by


def _ext_pit(pit: np.ndarray) -> np.ndarray:
    ext = np.asarray(pit, bool).copy()
    ext[:, 1:, :] |= pit[:, :-1, :]
    ext[:, :-1, :] |= pit[:, 1:, :]
    return ext


def pass_counts_by_band(pos: np.ndarray, pit: np.ndarray, green: np.ndarray, bands=BANDS) -> tuple[np.ndarray, np.ndarray]:
    """pass_counts split by where in the order the swap happened: a swap counts in the band of the better-placed of the two cars (before
    the swap), car-laps in the band of the car's own position. Same arguments as pass_counts. Returns (passes (S, B), car-laps (S, B))."""
    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    S, L, n = pos.shape
    ext = _ext_pit(pit)
    P, K = np.zeros((S, len(bands))), np.zeros((S, len(bands)))
    for m in range(1, L - 1):
        ok = (pos[:, m] > 0) & (pos[:, m + 1] > 0) & ~ext[:, m] & green[:, m, None] & green[:, m + 1, None]
        a = np.sign(pos[:, m, :, None] - pos[:, m, None, :])
        b = np.sign(pos[:, m + 1, :, None] - pos[:, m + 1, None, :])
        swap = ((a * b) < 0) & ok[:, :, None] & ok[:, None, :]
        front = np.minimum(pos[:, m, :, None], pos[:, m, None, :])
        for j, (_, lo, hi) in enumerate(bands):
            P[:, j] += (swap & (front >= lo) & (front <= hi)).sum((1, 2)) / 2.0
            K[:, j] += (ok & (pos[:, m] >= lo) & (pos[:, m] <= hi)).sum(1)
    return P, K


def close_counts_by_band(gap: np.ndarray, pos: np.ndarray, pit: np.ndarray, green: np.ndarray, bands=BANDS) -> tuple[np.ndarray, np.ndarray]:
    """close_counts split by the running position of the following car. gap and pos are (S, L, n). Returns (close (S, B), cars (S, B))."""
    gap = np.asarray(gap, float)
    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    ext = _ext_pit(pit)
    g2 = np.asarray(green, bool)
    ok = np.isfinite(gap) & ~ext
    ok[:, 0, :] = False
    ok[:, 1:, :] &= (g2[:, 1:] & g2[:, :-1])[:, :, None]
    close = np.stack([(ok & (gap < CLOSE_S) & (pos >= lo) & (pos <= hi)).sum((1, 2)) for _, lo, hi in bands], axis=1).astype(float)
    cars = np.stack([(ok & (pos >= lo) & (pos <= hi)).sum((1, 2)) for _, lo, hi in bands], axis=1).astype(float)
    return close, cars


def position_check(truths: list[RaceTruth], par: Params, S: int = 40, seed: int = 31) -> pd.DataFrame:
    """Real against simulated (everything known) on-track passes per car-lap and share of car-laps within CLOSE_S of the car ahead, by
    running position, by split. This is where the simulator's overall pass rate and bunching errors show up: at the front, in the middle or at the back."""
    acc: dict = {}
    for i, t in enumerate(truths):
        res = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + i)
        green = (t.neutral[1:t.n_laps + 1] == 0)[None, :]
        rp, rk = pass_counts_by_band(t.pos_actual[None], real_pit(t), green)
        sp, sk = pass_counts_by_band(res.pos.transpose(0, 2, 1), res.pit.transpose(0, 2, 1), (res.neutral[:, 1:t.n_laps + 1] == 0))
        parts = [np.stack([rp[0], rk[0], sp.mean(0), sk.mean(0)], axis=1)]
        if t.gap_ahead is not None and res.gap is not None:
            rc, rn = close_counts_by_band(t.gap_ahead[None], t.pos_actual[None], real_pit(t), green)
            sc, sn = close_counts_by_band(res.gap.transpose(0, 2, 1), res.pos.transpose(0, 2, 1), res.pit.transpose(0, 2, 1),
                                          (res.neutral[:, 1:t.n_laps + 1] == 0))
            parts.append(np.stack([rc[0], rn[0], sc.mean(0), sn.mean(0)], axis=1))
        else:
            parts.append(np.zeros((len(BANDS), 4)))
        for key in (split_of(t.season), "all"):
            acc[key] = acc.get(key, 0) + np.hstack(parts)
    rows = []
    for key, a in acc.items():
        for j, (name, _, _) in enumerate(BANDS):
            rp_, rk_, sp_, sk_, rc_, rn_, sc_, sn_ = a[j]
            rows.append({"split": key, "position": name,
                         "real_passes_per_car_lap": rp_ / rk_ if rk_ else math.nan,
                         "simulated_passes_per_car_lap": sp_ / sk_ if sk_ else math.nan,
                         "real_close_share": rc_ / rn_ if rn_ else math.nan,
                         "simulated_close_share": sc_ / sn_ if sn_ else math.nan,
                         "real_car_laps": rk_, "real_passes": rp_})
    return pd.DataFrame(rows)


def evaluate(truths: list[RaceTruth], par: Params, S: int, seed: int, bands: bool = False) -> dict:
    """Simulate every truth as it happened. Returns the grip table, pooled simulated pass rate and per-race simulated rates
    (and, with bands=True, the pooled passes and car-laps by running-position band)."""
    rows, sp, sc, per = [], 0.0, 0.0, {}
    bp = np.zeros((len(BANDS), 2))
    cl = ck = 0.0
    pc, pk = np.zeros(len(GAP_BOUNDS)), 0.0
    for i, t in enumerate(truths):
        r = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + i)
        sim = np.nanmean(sim_grip(r, t.slot, t.n_laps), axis=0)
        for f, a, s_ in zip(FRACTIONS, actual_grip(t), sim):
            rows.append({"race_id": t.race_id, "season": t.season, "fraction": f, "actual": a, "simulated": s_})
        p, c = sim_passes(r, t.n_laps)
        sp += float(p.mean())
        sc += float(c.mean())
        per[t.race_id] = (float(p.mean()), float(c.mean()))
        if bands:
            pb, kb = pass_counts_by_band(r.pos.transpose(0, 2, 1), r.pit.transpose(0, 2, 1), (r.neutral[:, 1:t.n_laps + 1] == 0))
            bp += np.stack([pb.mean(0), kb.mean(0)], axis=1)
        if t.gap_ahead is not None and r.gap is not None:
            a, b = sim_close(r, t.n_laps)
            cl += float(a.mean())
            ck += float(b.mean())
            a2, b2 = sim_gap_profile(r, t.n_laps)
            pc += a2.mean(0)
            pk += float(b2.mean())
    return {"grip": pd.DataFrame(rows), "pass_rate": sp / sc if sc else math.nan, "per_race": per, "band_passes": bp,
            "close_share": cl / ck if ck else math.nan, "gap_profile": pc / pk if pk else np.full(len(GAP_BOUNDS), math.nan)}


def grip_fit_table(truths: list[RaceTruth], par: Params, S: int, seed: int, sigma_lap=None) -> pd.DataFrame:
    return evaluate(truths, par, S, seed)["grip"]


def grip_loss(table: pd.DataFrame) -> float:
    m = table.dropna().groupby("fraction")[["actual", "simulated"]].mean()
    return float(((m["actual"] - m["simulated"]) ** 2).mean())


def real_pass_rate(truths: list[RaceTruth]) -> float:
    p = c = 0.0
    for t in truths:
        a, b = real_passes(t)
        p += a
        c += b
    return p / c if c else math.nan


GRIP_TOL = 0.02        # a gap of this size in the mean grip curve counts as one unit of loss
PASS_TOL = 0.15        # a log-ratio of this size in the pass rate counts as one unit of loss
CLOSE_TOL = 0.10       # ... and in the share of car-laps spent within CLOSE_S of the car ahead
THR_GRID = (0.4, 0.7, 1.0, 1.4, 1.9, 2.6, 3.5)
START_GRID = (0.0, 0.25, 0.5, 0.8, 1.2)         # used only when the races have no lap-1 positions to measure
SLOT_GRID = (0.2, 0.3, 0.45, 0.65, 0.9)
NOISE_GRID = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)   # shares of the measured lap noise that the simulator keeps
GAP_GRID = (0.3, 0.45, 0.6, 0.8, 1.0)           # closest a follower gets to the car ahead without passing (s)
DA_MULT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)         # multiples of the measured dirty-air loss that the fit tries (0 = none)
DRIFT_SHARE = (0.0, 0.5, 0.75, 1.0)             # shares of the measured drift that the fit tries (the variance given up goes to white lap noise)
START_FRONT_GRID = (1.0, 0.6, 0.35)             # lap-1 noise of the pole sitter as a share of the field's (rises to 1 by slot 11)
START_BAD_GRID = ((0.0, 0.0), (0.03, 2.0), (0.06, 2.0), (0.10, 2.0), (0.03, 4.0), (0.06, 4.0))   # (chance of a bad start, mean time it costs in s)
KEEP_TOL = 0.02        # a difference of this size in the share of pole sitters still leading after lap 1 counts as one unit of loss
TAIL_TOL = 0.25        # a log-ratio of this size in the share of front-six cars that move 3+ places on lap 1 counts as one unit
LAP1_BANDS = ((1, 2), (3, 6), (7, 12), (13, 99))   # grid slots the lap-1 movement is split by
BAND_TOL = 0.30        # a log-ratio of this size in the pass rate of a running-position band counts as one unit of loss
OUTLAP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)       # shares of a stop's cost paid on the out-lap that the undercut fit tries
CIRCUIT_MULT = (0.5, 0.7, 1.0, 1.4, 2.0)
THR_RANGE = (0.2, 4.5)
START_RANGE = (0.0, 3.0)


def real_lap1_move(truths: list[RaceTruth]) -> float:
    v = [x for x in (lap1_move(t) for t in truths) if np.isfinite(x)]
    return float(np.mean(v)) if v else math.nan


def fit_start(truths: list[RaceTruth], par: Params, target: float, S: int = 40, seed: int = 3, iters: int = 11) -> float:
    """The lap-1 noise (sigma_start) at which simulated cars change places on lap 1 as often as real ones (bisection; lap 1 has no
    passing rule, so only the first lap is simulated). The result is clipped to START_RANGE."""
    ts = [t for t in truths if np.isfinite(lap1_move(t))]
    if not ts or not np.isfinite(target):
        return par.sigma_start

    def move(sig):
        q = replace(par, sigma_start=float(sig))
        tot = 0.0
        for i, t in enumerate(ts):
            sp = replace(t.spec(), n_laps=1, neutral=t.neutral[None, :2])
            r = E.simulate(sp, q, S=S, seed=seed + i)
            tot += float(sim_lap1_move(r, t.slot).mean())
        return tot / len(ts)

    lo, hi = START_RANGE
    if move(hi) < target:
        return hi
    if move(lo) > target:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if move(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def real_lap1_bands(truths: list[RaceTruth]) -> np.ndarray:
    """(bands, 2): places moved on lap 1 (summed) and cars, by grid-slot band, pooled over races with lap-1 positions."""
    out = np.zeros((len(LAP1_BANDS), 2))
    for t in truths:
        p = t.pos_actual[0]
        m = np.isfinite(p) & (p > 0)
        if m.sum() < 8:
            continue
        mv = np.abs(np.where(m, p, 0.0) - t.slot)
        for j, (lo, hi) in enumerate(LAP1_BANDS):
            b = m & (t.slot >= lo) & (t.slot <= hi)
            out[j] += (float(mv[b].sum()), float(b.sum()))
    return out


def sim_lap1_bands(truths: list[RaceTruth], par: Params, S: int = 40, seed: int = 3) -> np.ndarray:
    """The same from one simulated lap per race (mean over simulations), for the races that have lap-1 positions."""
    out = np.zeros((len(LAP1_BANDS), 2))
    for i, t in enumerate(truths):
        if not np.isfinite(lap1_move(t)):
            continue
        r = E.simulate(replace(t.spec(), n_laps=1, neutral=t.neutral[None, :2]), par, S=S, seed=seed + i)
        p = r.pos[:, :, 0].astype(float)
        alive = p > 0
        mv = np.abs(p - t.slot[None]) * alive
        for j, (lo, hi) in enumerate(LAP1_BANDS):
            b = (t.slot >= lo) & (t.slot <= hi)
            out[j] += (float(mv[:, b].sum(1).mean()), float(alive[:, b].sum(1).mean()))
    return out


def real_lap1_tail(truths: list[RaceTruth]) -> dict:
    """Lap-1 outcomes that a Gaussian start gets wrong: how often the pole sitter is still first after lap 1, and how often a car from the
    front six (slots 1-6) changes places by 3 or more. Pooled over races with lap-1 positions. Only positions on lap 1 are used."""
    kept = n_pole = big = n_front = 0
    for t in truths:
        p = t.pos_actual[0]
        m = np.isfinite(p) & (p > 0)
        if m.sum() < 8:
            continue
        pole = int(np.argmin(t.slot))
        if m[pole]:
            kept += int(p[pole] == 1)
            n_pole += 1
        f = m & (t.slot <= 6)
        big += int((np.abs(p[f] - t.slot[f]) >= 3).sum())
        n_front += int(f.sum())
    return {"kept": kept, "pole": n_pole, "big": big, "front": n_front}


def sim_lap1_tail(truths: list[RaceTruth], par: Params, S: int = 40, seed: int = 3) -> dict:
    out = {"kept": 0.0, "pole": 0, "big": 0.0, "front": 0}
    for i, t in enumerate(truths):
        if not np.isfinite(lap1_move(t)):
            continue
        r = E.simulate(replace(t.spec(), n_laps=1, neutral=t.neutral[None, :2]), par, S=S, seed=seed + i)
        p = r.pos[:, :, 0].astype(float)
        pole = int(np.argmin(t.slot))
        alive = p[:, pole] > 0
        if alive.any():
            out["kept"] += float((p[alive, pole] == 1).mean())
            out["pole"] += 1
        f = (t.slot <= 6)
        a = p[:, f] > 0
        out["big"] += float(((np.abs(p[:, f] - t.slot[f][None]) >= 3) & a).sum(1).mean())
        out["front"] += int(f.sum())
    return out


def fit_start_shape(truths: list[RaceTruth], par: Params, target: float, S: int = 40, seed: int = 3,
                    fronts=START_FRONT_GRID, bads=START_BAD_GRID) -> tuple[float, float, float, float, pd.DataFrame]:
    """Lap-1 noise (sigma_start), how much smaller it is at the front (start_front) and the chance and cost of a bad start
    (bad_start_p, bad_start_loss). For each combination sigma_start is set so the field as a whole changes places as much as real lap 1s do;
    the combination whose places moved by grid-slot band (LAP1_BANDS), whose share of pole sitters still leading after lap 1, and whose share
    of front-six cars moving 3+ places are closest to the real ones is kept. A bad start gives the mostly-clean, sometimes-chaotic shape
    that Gaussian noise cannot. Only lap-1 positions are used (nothing about who wins the race). Returns (sigma_start, start_front,
    bad_start_p, bad_start_loss, table)."""
    real = real_lap1_bands(truths)
    rt = real_lap1_tail(truths)
    real_keep = rt["kept"] / rt["pole"] if rt["pole"] else math.nan
    real_big = rt["big"] / rt["front"] if rt["front"] else math.nan
    rows = []
    for f in fronts:
        for bp, bl in bads:
            q = replace(par, start_front=float(f), bad_start_p=float(bp), bad_start_loss=float(bl) if bp > 0 else par.bad_start_loss)
            sig = fit_start(truths, q, target, S=S, seed=seed)
            q = replace(q, sigma_start=sig)
            sim = sim_lap1_bands(truths, q, S=S, seed=seed)
            st = sim_lap1_tail(truths, q, S=S, seed=seed)
            sim_keep = st["kept"] / st["pole"] if st["pole"] else math.nan
            sim_big = st["big"] / st["front"] if st["front"] else math.nan
            err = []
            for j, (name_lo, name_hi) in enumerate(LAP1_BANDS):
                rv = real[j, 0] / real[j, 1] if real[j, 1] else math.nan
                sv = sim[j, 0] / sim[j, 1] if sim[j, 1] else math.nan
                ok = np.isfinite(rv) and np.isfinite(sv) and rv > 0 and sv > 0
                err.append(math.log(sv / rv) if ok else math.nan)
                rows.append({"start_front": float(f), "bad_start_p": float(bp), "bad_start_loss": float(bl), "sigma_start": float(sig),
                             "slots": f"{name_lo}-{name_hi}" if name_hi < 99 else f"{name_lo}+",
                             "real_places_moved": rv, "simulated_places_moved": sv,
                             "real_pole_keeps_lead": real_keep, "simulated_pole_keeps_lead": sim_keep,
                             "real_front_big_moves": real_big, "simulated_front_big_moves": sim_big})
            e = np.array(err, float)
            score = float(np.nanmean((e / 0.15) ** 2)) if np.isfinite(e).any() else math.inf
            if np.isfinite(real_keep) and np.isfinite(sim_keep):
                score += ((sim_keep - real_keep) / KEEP_TOL) ** 2
            if np.isfinite(real_big) and real_big > 0:
                score += (math.log(max(sim_big, 1e-3) / real_big) / TAIL_TOL) ** 2 if np.isfinite(sim_big) else 0.0
            for r in rows[-len(LAP1_BANDS):]:
                r["start_loss"] = score
    tab = pd.DataFrame(rows)
    if tab.empty or not np.isfinite(tab["start_loss"]).any():
        return par.sigma_start, par.start_front, par.bad_start_p, par.bad_start_loss, tab
    best = tab.loc[tab["start_loss"].idxmin()]
    return (float(best["sigma_start"]), float(best["start_front"]), float(best["bad_start_p"]),
            float(best["bad_start_loss"]) if best["bad_start_p"] > 0 else par.bad_start_loss, tab)


def real_pass_bands(truths: list[RaceTruth]) -> np.ndarray:
    """(bands, 2): real on-track passes and car-laps by running-position band (BANDS), pooled."""
    out = np.zeros((len(BANDS), 2))
    for t in truths:
        green = (t.neutral[1:t.n_laps + 1] == 0)[None, :]
        p, k = pass_counts_by_band(t.pos_actual[None], real_pit(t), green)
        out += np.stack([p[0], k[0]], axis=1)
    return out


def fit_loss(truths, par, real_rate, S, seed, real_close_=math.nan, real_profile=None, real_bands=None) -> tuple[float, float, float, float, float]:
    """Loss = grip^2 + pass^2 + close^2, where the close term is the gap-to-the-car-ahead distribution: the share within CLOSE_S
    and, when `real_profile` is given, the shares within each of GAP_BOUNDS (the mean of their squared log-ratios, in CLOSE_TOL units).
    When `real_bands` is given (real_pass_bands), the pass term also holds the mean squared log-ratio of the pass rate in each running-position
    band (P1, P2-3, P4-10, P11 and below) in BAND_TOL units, so passes cannot be right overall by being too many at the back and too few at the front."""
    ev = evaluate(truths, par, S, seed, bands=real_bands is not None)
    g = math.sqrt(grip_loss(ev["grip"])) / GRIP_TOL
    pr = ev["pass_rate"]
    q = abs(math.log(max(pr, 1e-6) / max(real_rate, 1e-6))) / PASS_TOL if np.isfinite(pr) and np.isfinite(real_rate) else 0.0
    cs = ev["close_share"]
    c = (abs(math.log(max(cs, 1e-6) / max(real_close_, 1e-6))) / CLOSE_TOL
         if np.isfinite(cs) and np.isfinite(real_close_) else 0.0)
    c2 = c * c
    if real_bands is not None:
        sb = ev["band_passes"]
        rr = np.where(real_bands[:, 1] > 0, real_bands[:, 0] / np.maximum(real_bands[:, 1], 1e-9), np.nan)
        ss = np.where(sb[:, 1] > 0, sb[:, 0] / np.maximum(sb[:, 1], 1e-9), np.nan)
        okb = np.isfinite(rr) & np.isfinite(ss) & (rr > 0) & (ss > 0)
        if okb.any():
            q = math.sqrt(q * q + float(np.mean((np.log(ss[okb] / rr[okb]) / BAND_TOL) ** 2)))
    if real_profile is not None:
        sp_, rp = np.asarray(ev["gap_profile"], float), np.asarray(real_profile, float)
        ok = np.isfinite(sp_) & np.isfinite(rp) & (sp_ > 0) & (rp > 0)
        if ok.any():
            c2 += float(np.mean((np.log(sp_[ok] / rp[ok]) / CLOSE_TOL) ** 2))
    return g * g + q * q + c2, g * g, q * q, pr, c2, cs


def fit_passing(truths: list[RaceTruth], base: Params, S: int = 40, seed: int = 1, thr_grid=THR_GRID, start_grid=START_GRID,
                slot_grid=SLOT_GRID, noise_grid=NOISE_GRID, gap_grid=GAP_GRID, da_grid=DA_MULT, drift_grid=DRIFT_SHARE,
                front_grid=START_FRONT_GRID, bad_grid=START_BAD_GRID, rounds: int = 2) -> tuple[Params, pd.DataFrame, list[str]]:
    """Coordinate search: sweep one setting at a time over its grid with the others at their best so far; a second round
    revisits the passing threshold and the closest-follow gap. Three targets: the grip curve, the on-track pass rate, and the
    share of car-laps within CLOSE_S of the car ahead (when the races have line-crossing times; without them the closest-follow gap
    stays as it is). When `base` has a measured dirty-air loss, the multiple of it that the simulator uses is searched too (`da_grid`,
    0 meaning none), so the data can switch it off. The lap-1 noise is not searched: it is set to the value at which lap 1 shuffles the field as much as real
    lap 1s do (bisection), for each start spacing tried, together with how much quieter the front of the grid is (`front_grid`, chosen
    on the places moved by grid-slot band); without lap-1 positions it is searched on `start_grid` instead. When `base` has a measured
    drift, the share of it kept is searched too (`drift_grid`; the variance given up goes to white lap noise, so the total per lap is unchanged).
    The pass term of the loss includes the pass rate by running-position band.
    Returns (settings, every evaluation, settings at the edge of their grid)."""
    real_rate = real_pass_rate(truths)
    real_cl = real_close_share(truths)
    real_prof = real_gap_profile(truths)
    real_bands = real_pass_bands(truths)
    has_close = bool(np.isfinite(real_cl))
    target1 = real_lap1_move(truths)
    has_lap1 = bool(np.isfinite(target1))
    sigma_lap0 = base.sigma_lap
    gap_grid = tuple(gap_grid) if has_close else (base.min_gap,)
    has_da = bool(has_close and base.da_loss > 0)
    has_drift = bool(base.drift_sd > 0)
    cur = {"thr": float(thr_grid[len(thr_grid) // 2]), "min_gap": float(base.min_gap if not has_close else gap_grid[len(gap_grid) // 2]),
           "sigma_start": float(start_grid[len(start_grid) // 2]), "slot_gap": float(slot_grid[min(1, len(slot_grid) - 1)]), "noise": float(noise_grid[-1]),
           "da": 1.0 if has_da else 0.0, "drift": 1.0 if has_drift else 0.0}
    grids = {"thr": thr_grid, "min_gap": gap_grid, "slot_gap": slot_grid, "noise": noise_grid}
    if has_da:
        grids["da"] = tuple(da_grid)
    if has_drift:
        grids["drift"] = tuple(drift_grid)
    if not has_lap1:
        grids["sigma_start"] = start_grid
    rows, memo, start_memo, shape_tabs = [], {}, {}, []

    def start_of(c):
        if not has_lap1:
            return c["sigma_start"]
        key = round(c["slot_gap"], 6)
        if key not in start_memo:
            sig, front, bp, bl, tab = fit_start_shape(truths, replace(base, slot_gap=c["slot_gap"]), target1, S=S, seed=seed + 100,
                                                      fronts=front_grid, bads=bad_grid)
            start_memo[key] = (sig, front, bp, bl)
            shape_tabs.append(tab.assign(slot_gap=c["slot_gap"]))
        return start_memo[key][0]

    def front_of(c):
        if not has_lap1:
            return base.start_front
        start_of(c)
        return start_memo[round(c["slot_gap"], 6)][1]

    def bad_of(c):
        if not has_lap1:
            return base.bad_start_p, base.bad_start_loss
        start_of(c)
        return start_memo[round(c["slot_gap"], 6)][2:]

    def par_of(c):
        m = c["drift"] if has_drift else 1.0
        white = math.sqrt((sigma_lap0 * c["noise"]) ** 2 + (1.0 - m * m) * base.drift_sd ** 2)
        return Params(**{**asdict(base), "thr": c["thr"], "min_gap": c["min_gap"], "sigma_start": start_of(c), "start_front": front_of(c),
                         "bad_start_p": bad_of(c)[0], "bad_start_loss": bad_of(c)[1], "slot_gap": c["slot_gap"], "sigma_lap": white, "drift_sd": base.drift_sd * m,
                         "da_loss": base.da_loss * c["da"] if has_da else base.da_loss})

    def score(c):
        key = tuple(round(c[k], 6) for k in sorted(c))
        if key not in memo:
            memo[key] = fit_loss(truths, par_of(c), real_rate, S, seed, real_cl, real_prof if has_close else None, real_bands)
        return memo[key]

    da_step = (["da"] if has_da else []) + (["drift"] if has_drift else [])
    order = ["thr", "min_gap"] + da_step + ["slot_gap", "noise"] + (["thr", "min_gap"] + da_step + ["noise"]) * (rounds - 1)
    if not has_lap1:
        order = (["thr", "min_gap"] + da_step + ["sigma_start", "slot_gap", "noise"]
                 + (["thr", "min_gap"] + da_step + ["sigma_start", "noise"]) * (rounds - 1))
    for rnd, name in enumerate(order):
        if len(grids[name]) < 2 and name == "min_gap":
            continue
        best_v, best_l = cur[name], math.inf
        for v in grids[name]:
            c = {**cur, name: float(v)}
            loss, g, q, pr, cterm, cs = score(c)
            rows.append({"step": rnd + 1, "setting": name, "value": float(v), **{f"at_{k}": x for k, x in c.items()},
                         "at_sigma_start_used": start_of(c), "loss": loss, "grip_term": g, "pass_term": q, "close_term": cterm,
                         "sim_pass_rate": pr, "real_pass_rate": real_rate, "sim_close_share": cs, "real_close_share": real_cl})
            if loss < best_l:
                best_v, best_l = float(v), loss
        cur[name] = best_v
    edge = [k for k, g in grids.items() if len(g) > 1 and k != "drift" and (cur[k] == max(g) or (cur[k] == min(g) and k != "da"))]
    fit_par = par_of(cur)
    fit_grid = pd.DataFrame(rows)
    fit_grid.attrs["start_shape"] = pd.concat(shape_tabs, ignore_index=True) if shape_tabs else pd.DataFrame()
    fit_grid.attrs["drift_share"] = float(cur["drift"]) if has_drift else math.nan
    return fit_par, fit_grid, edge


def circuit_passing(truths: list[RaceTruth], par: Params, S: int = 30, seed: int = 5, prior_races: float = 2.0,
                    mults=CIRCUIT_MULT) -> tuple[dict, pd.DataFrame]:
    """A passing threshold per circuit. For each circuit, the pass rate of its real races (the races given) against the simulated
    rate at several multiples of the general threshold; the multiple that matches (interpolated on the log scale) is shrunk toward 1
    with weight races / (races + prior_races)."""
    by = {}
    for t in truths:
        by.setdefault(t.circuit_id, []).append(t)
    out, rows = {}, []
    for circuit, ts in sorted(by.items()):
        rp = real_pass_rate(ts)
        rates = []
        for m in mults:
            ev = evaluate(ts, Params(**{**asdict(par), "thr": float(np.clip(par.thr * m, *THR_RANGE)), "circuit_thr": {}}), S, seed)
            rates.append(ev["pass_rate"])
        rates = np.array(rates, float)
        ok = np.isfinite(rates) & (rates > 0)
        if ok.sum() >= 2 and np.isfinite(rp) and rp > 0:
            lr, lm = np.log(rates[ok]), np.log(np.array(mults)[ok])
            order = np.argsort(lr)
            m_hat = float(np.exp(np.interp(np.log(rp), lr[order], lm[order])))
        else:
            m_hat = 1.0
        w = len(ts) / (len(ts) + prior_races)
        m = float(np.exp(w * np.log(m_hat)))
        thr_c = float(np.clip(par.thr * m, *THR_RANGE))
        out[circuit] = thr_c
        rows.append({"circuit_id": circuit, "races": len(ts), "real_pass_rate": rp,
                     "simulated_at_general": float(rates[list(mults).index(1.0)]) if 1.0 in mults else math.nan,
                     "multiple_matching": m_hat, "weight": w, "multiple_used": m, "thr_used": thr_c})
    return out, pd.DataFrame(rows)


DISPERSION_GRID = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)


def pass_dispersion(truths: list[RaceTruth], par: Params, S: int = 40, seed: int = 7) -> float:
    """How much more the real races differ from the simulator's expectation than the simulator's own spread allows: the sum over races
    of (real passes - simulated mean)^2 divided by the sum of the simulated variances. 1 means the simulator is as variable as the
    real races; above 1, real races differ more from each other (in how easy passing was) than the simulator lets them."""
    num = den = 0.0
    for i, t in enumerate(truths):
        a, _ = real_passes(t)
        r = E.simulate(t.spec(), par.for_circuit(t.circuit_id), S=S, seed=seed + i, history=True)
        p, _ = sim_passes(r, t.n_laps)
        num += (a - float(p.mean())) ** 2
        den += float(p.var(ddof=1))
    return num / den if den > 0 else math.nan


def fit_pass_difficulty(truths: list[RaceTruth], par: Params, grid=DISPERSION_GRID, S: int = 40, seed: int = 7) -> tuple[float, pd.DataFrame]:
    """The race-to-race spread of passing difficulty (`thr_sd`) at which the dispersion above is closest to 1; 0 when the simulator is
    already as variable as the real races. An upper bound on the real spread, because anything else the simulator misses in a race
    (a wrong pace, say) counts here as difficulty."""
    rows = [{"thr_sd": float(v), "dispersion": pass_dispersion(truths, replace(par, thr_sd=float(v)), S, seed)} for v in grid]
    tab = pd.DataFrame(rows)
    ok = tab[np.isfinite(tab["dispersion"])]
    if ok.empty or ok["dispersion"].iloc[0] <= 1.0:
        return 0.0, tab
    best = ok.iloc[int(np.argmin(np.abs(np.log(ok["dispersion"].to_numpy()))))]
    return float(best["thr_sd"]), tab


def fit_outlap(truths: list[RaceTruth], par: Params, real_events: pd.DataFrame, grid=OUTLAP_GRID, S: int = 12, seed: int = 41) -> tuple[float, pd.DataFrame]:
    """The share of a stop's cost paid on the out-lap (`outlap_share`) at which the simulator's undercut matches the real one, using
    the training races as they happened (real stops and safety cars). The targets are the undercut advantage (chaser stopping first minus
    chaser stopping second, in points) for stops 1-2 laps apart and for stops 3+ laps apart, each against its real value in units of 6 points; both are
    scored so the fit cannot buy the close-stops case with the wide-stops one. Nothing about pole wins or how pole is lost is used.
    Returns (share, one row per share tried)."""
    from . import replay as R           # replay imports this module, so it is imported here and not at the top
    if real_events is None or len(real_events) < 40:
        return par.outlap_share, pd.DataFrame()
    real = race_flow.undercut_summary(real_events, reps=20)
    real = real[real["period"] == "all"].set_index("scope")
    if not {"stops 1-2 laps apart", "stops 3+ laps apart"} <= set(real.index):
        return par.outlap_share, pd.DataFrame()
    rows = []
    for share in grid:
        q = replace(par, outlap_share=float(share))
        ev = []
        for i, t in enumerate(truths):
            res = E.simulate(t.spec(), q.for_circuit(t.circuit_id), S=S, seed=seed + i)
            e, _ = R.sim_events(t, res, S)
            ev += e
        sim = race_flow.undercut_summary(pd.DataFrame(ev), reps=20) if ev else pd.DataFrame()
        if sim.empty:
            continue
        sim = sim[sim["period"] == "all"].set_index("scope")
        row = {"outlap_share": float(share), "events": len(ev)}
        loss = 0.0
        for scope, tag in (("stops 1-2 laps apart", "close"), ("stops 3+ laps apart", "wide"), ("gap-standardised", "std")):
            if scope in sim.index and scope in real.index:
                a, r = float(sim.at[scope, "undercut_advantage"]), float(real.at[scope, "undercut_advantage"])
                row[f"simulated_{tag}"], row[f"real_{tag}"] = a, r
                if tag != "std":
                    loss += ((a - r) / 0.06) ** 2
        row["loss"] = loss
        rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        return par.outlap_share, tab
    return float(tab.loc[tab["loss"].idxmin(), "outlap_share"]), tab


def drift_check(truths: list[RaceTruth], par: Params, mults=(0.0, 0.5, 1.0), S: int = 40, seed: int = 21) -> pd.DataFrame:
    """Does the lap-time drift earn its place? The fitted settings with the drift scaled down (the lost variance moved to white lap noise,
    so each lap's total spread is unchanged), scored on the same three targets as the fit. Nothing is re-fitted: a drift that is
    really standing in for something else shows as a loss that does not rise much when it is removed."""
    if par.drift_sd <= 0:
        return pd.DataFrame()
    real_rate, real_cl, real_prof = real_pass_rate(truths), real_close_share(truths), real_gap_profile(truths)
    prof = real_prof if np.isfinite(real_cl) else None
    bands = real_pass_bands(truths)
    rows = []
    for m in mults:
        q = replace(par, drift_sd=m * par.drift_sd, sigma_lap=math.sqrt(par.sigma_lap ** 2 + (1.0 - m * m) * par.drift_sd ** 2))
        loss, g2, q2, pr, c2, cs = fit_loss(truths, q, real_rate, S, seed, real_cl, prof, bands)
        rows.append({"drift_share": m, "drift_sd": q.drift_sd, "white_sd": q.sigma_lap, "loss": loss, "grip_term": g2, "pass_term": q2,
                     "close_term": c2, "sim_pass_rate": pr, "real_pass_rate": real_rate, "sim_close_share": cs, "real_close_share": real_cl})
    return pd.DataFrame(rows)


def stop_spread(stops: pd.DataFrame, sigma_lap: float, top: bool | None = None) -> float:
    """Spread of what a stop costs beyond the simulator's own lap noise (which hits the in-lap and the out-lap):
    robust sd of stop loss around its race median, minus 2 x sigma_lap^2, as a standard deviation.
    top=True keeps only stops made from the top three running positions, top=False the rest (needs a `position` column)."""
    if stops is None or len(stops) < 30:
        return 0.0
    d = stops.copy()
    d["dev"] = d["loss_s"] - d.groupby("race_id")["loss_s"].transform("median")
    d = d[d.groupby("race_id")["loss_s"].transform("size") >= 5]
    if top is not None and "position" in d.columns:
        pos = d["position"]
        d = d[(pos <= 3) & pos.notna()] if top else d[~((pos <= 3) & pos.notna())]
    if len(d) < 30:
        return 0.0
    mad_sd = 1.4826 * float(np.median(np.abs(d["dev"])))
    return float(math.sqrt(max(mad_sd ** 2 - 2.0 * sigma_lap ** 2, 0.0)))


def pit_spreads(stops: pd.DataFrame, par: Params) -> dict:
    """sigma_pit for everyone and for the top three, each after the lap noise those cars have in the simulator (par's fitted one).
    The top-three value falls back to the general one where there are too few top-three stops."""
    allv = stop_spread(stops, par.sigma_lap)
    if stops is None or "position" not in stops.columns or int((stops["position"] <= 3).sum()) < 30:
        return {"sigma_pit": allv, "sigma_pit_top": -1.0}
    rest = stop_spread(stops, par.sigma_lap * par.noise_rest, top=False) or allv
    top = stop_spread(stops, par.sigma_lap * par.noise_top, top=True)
    return {"sigma_pit": rest, "sigma_pit_top": top}


# --------------------------------------------------------------------- run

WARM_LAPS = (2, 3, 4, 5, 6, 7, 8)       # stint laps measured (1 is the out-lap, which the stop's cost already holds)
WARM_REF = (12, 24)                     # stint laps taken as the warmed-up reference
WARM_DECAYS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_WARM_RACES = 6
MIN_WARM_AMP = 0.03


def stint_lap_of(laps: pd.DataFrame) -> pd.Series:
    """Laps since the stint began, the out-lap being 1 (the first lap of the stint's own lap numbers)."""
    first = laps.groupby(["race_id", "driver_id", "stint"])["lap_number"].transform("min")
    return laps["lap_number"] - first + 1            # NaN where the stint or lap number is missing


def _race_residuals(d: pd.DataFrame) -> pd.Series | None:
    """The per-race tyre regression of stage 2 (driver, lap number, compound, compound x tyre age), fitted WITHOUT the laps being measured
    (stint laps 2-8 after a stop), and its residuals for all of one race's clean laps. Fitting without them keeps the wear slope from absorbing
    part of the warm-up."""
    d = d[d["lap_time_s"] <= tyres.SLOW_CUT * d["lap_time_s"].median()]
    comps = [c for c in tyres.DRY if (d["compound"] == c).sum() >= tyres.MIN_LAPS_COMPOUND
             and (lambda a: a.quantile(0.9) - a.quantile(0.1))(d.loc[d["compound"] == c, "tyre_life"]) >= tyres.MIN_AGE_SPREAD]
    d = d[d["compound"].isin(comps)]
    if not comps or len(d) < tyres.MIN_LAPS_RACE or d["driver_id"].nunique() < tyres.MIN_DRIVERS:
        return None
    ref = "MEDIUM" if "MEDIUM" in comps else max(comps, key=lambda c: int((d["compound"] == c).sum()))
    codes, uniques = pd.factorize(d["driver_id"])
    lap_codes, lap_values = pd.factorize(d["lap_number"])
    nd, nl = len(uniques), len(lap_values)
    cols = [np.eye(nd)[codes], np.eye(nl)[lap_codes][:, 1:]]
    cols += [(d["compound"] == c).to_numpy(float)[:, None] for c in comps if c != ref]
    cols += [((d["compound"] == c).to_numpy(float) * d["tyre_life"].to_numpy(float))[:, None] for c in comps]
    X = np.hstack(cols)
    y = d["lap_time_s"].to_numpy(float)
    hold = ((d["stint"] > d["first_stint"]) & d["stint_lap"].isin(WARM_LAPS)).to_numpy()
    if (~hold).sum() < tyres.MIN_LAPS_RACE:
        return None
    beta = np.linalg.lstsq(X[~hold], y[~hold], rcond=None)[0]
    return pd.Series(y - X @ beta, index=d.index)


def warm_up(laps: pd.DataFrame, wet_races=()) -> dict:
    """How slow the first laps on a new set are. For every dry race the stage 2 tyre regression is fitted on clean laps and its residuals
    are grouped by stint lap (the out-lap is lap 1 and is not a clean lap; the first stint of the race is left out). The excess at stint lap
    j is the mean residual there minus the mean at stint laps 12-24, averaged over races. A decay is chosen for all years (the share of the
    excess still there one lap later), then an amplitude per era (2018-21 and 2022 on). `amp` is the excess on stint lap 2. An amplitude below
    0.03 s is called not measurable and set to 0. Returns {'table', 'decay', 'amp': {era: s}, 'pooled', 'note'}."""
    out = {"table": pd.DataFrame(), "decay": Params().warm_decay, "amp": {}, "pooled": 0.0, "note": "not measured"}
    d0 = tyres.clean_laps(laps)
    d0 = d0[~d0["race_id"].isin(set(wet_races))].copy()
    sl = stint_lap_of(laps).reindex(d0.index)
    d0["first_stint"] = laps.groupby(["race_id", "driver_id"])["stint"].transform("min").reindex(d0.index)
    d0["stint_lap"] = sl.reindex(d0.index)
    d0 = d0[d0["stint_lap"].notna() & d0["first_stint"].notna()]
    rows = []
    for rid, d in d0.groupby("race_id"):
        res = _race_residuals(d)
        if res is None:
            continue
        d = d.loc[res.index].assign(res=res)
        d = d[d["stint"] > d["first_stint"]]
        ref = d[(d["stint_lap"] >= WARM_REF[0]) & (d["stint_lap"] <= WARM_REF[1])]["res"]
        if len(ref) < 20:
            continue
        base = float(ref.mean())
        for j in WARM_LAPS:
            z = d[d["stint_lap"] == j]["res"]
            if len(z) >= 3:
                rows.append({"race_id": rid, "season": int(d["season"].iloc[0]), "stint_lap": j, "laps": len(z), "excess": float(z.mean()) - base})
    if not rows:
        return out
    r = pd.DataFrame(rows)
    r["era"] = r["season"].map(era_of)
    tabs = []
    for era, g in list(r.groupby("era")) + [("all", r)]:
        for j, h in g.groupby("stint_lap"):
            w = h["laps"].to_numpy(float)
            m = float(np.average(h["excess"], weights=w))
            se = float(np.sqrt(np.cov(h["excess"], aweights=w)) / math.sqrt(len(h))) if len(h) > 2 else math.nan
            tabs.append({"era": era, "stint_lap": int(j), "races": int(h["race_id"].nunique()), "laps": int(w.sum()), "excess_s": m, "se_s": se})
    tab = pd.DataFrame(tabs)

    def fit(sub: pd.DataFrame, decay: float | None = None) -> tuple[float, float, float]:
        j = sub["stint_lap"].to_numpy(float)
        m, w = sub["excess_s"].to_numpy(float), sub["laps"].to_numpy(float)
        best = None
        for dec in (WARM_DECAYS if decay is None else (decay,)):
            x = dec ** (j - 2.0)
            amp = max(float((w * m * x).sum() / (w * x * x).sum()), 0.0)
            loss = float((w * (m - amp * x) ** 2).sum())
            if best is None or loss < best[0]:
                best = (loss, dec, amp)
        return best[1], best[2], best[0]

    allt = tab[tab["era"] == "all"]
    if allt["races"].max() < MIN_WARM_RACES:
        out["table"], out["note"] = tab, "too few races"
        return out
    decay, pooled, _ = fit(allt)
    out.update(table=tab, decay=float(decay), pooled=float(pooled) if pooled >= MIN_WARM_AMP else 0.0)
    for era in ("2018-21", "2022-now"):
        g = tab[tab["era"] == era]
        if g["races"].max() >= MIN_WARM_RACES:
            _, a, _ = fit(g, decay)
            out["amp"][era] = float(a) if a >= MIN_WARM_AMP else 0.0
    out["note"] = "measured" if out["pooled"] > 0 else "no measurable warm-up"
    return out


def warm_for(wu: dict, season: int) -> float:
    return float(wu["amp"].get(era_of(season), wu["pooled"]))


def build_inputs(con) -> dict:
    """Everything the calibration measures from the data before any simulator setting is fitted: the laps, the per-race tyre
    fits, the race histories (RaceTruth) and the tyre warm-up. The live forecast calls this alone, so that new races can enter the
    history without the simulator's settings being refitted."""
    laps, flags, results = load_inputs(con)
    laps = add_gap_prev(laps)
    quali = load_quali(con)
    wet = flags.set_index("race_id")["wet"]
    per_race_rows, extras = [], {}
    for race_id, d in tyres.clean_laps(laps).groupby("race_id"):
        if bool(wet.get(race_id, False)):
            continue
        out = tyres.fit_race(d, detail=True)
        if out is None:
            continue
        row, ex = out
        first = d.iloc[0]
        per_race_rows.append({"race_id": race_id, "season": int(first["season"]), "circuit_id": first["circuit_id"],
                              "circuit_name": first["circuit_name"], **row})
        extras[int(race_id)] = ex
    per_race = pd.DataFrame(per_race_rows)
    era_t = era_tyres(per_race, extras)
    circuit_wear = tyres.circuit_wear_shrunk(per_race)
    stops = tyres.pit_loss(laps)
    pit_by_circuit = tyres.pit_loss_by_circuit(stops, min_stops=1)
    prac = P.features(P.load_practice(con))
    truths, skipped = build_truths(laps, flags, results, per_race, extras, circuit_wear, pit_by_circuit, era_t, quali=quali,
                                   practice=prac)
    wu = warm_up(laps, list(wet[wet].index))
    for t in truths:
        t.warm = warm_for(wu, t.season)
    return {"laps": laps, "flags": flags, "results": results, "wet": wet, "per_race": per_race, "extras": extras,
            "era_t": era_t, "circuit_wear": circuit_wear, "stops": stops, "truths": truths, "skipped": skipped, "wu": wu}


def calibrate(con, S: int = 40, thr_grid=THR_GRID, start_grid=START_GRID):
    b = build_inputs(con)
    laps, flags, wet, per_race, era_t = b["laps"], b["flags"], b["wet"], b["per_race"], b["era_t"]
    circuit_wear, stops, truths, skipped, wu = b["circuit_wear"], b["stops"], b["truths"], b["skipped"], b["wu"]
    base = Params(warm_amp=wu["pooled"], warm_decay=wu["decay"])
    train = [t for t in truths if split_of(t.season) == "train"]
    sigma_lap = float(per_race["rmse_s"].median() * 0.9) if len(per_race) else base.sigma_lap
    base = Params(**{**asdict(base), "sigma_lap": sigma_lap})
    rates = neutral_stop_rates(truths)
    cost = neutral_pit_cost(laps, truths)
    if len(cost):
        shares = cost[(cost["share"] > 0) & (cost["share"] < 1.3)]
        sc_cost = shares[shares["kind"] == "sc"]["share"]
        vsc_cost = shares[shares["kind"] == "vsc"]["share"]
    else:
        sc_cost = vsc_cost = pd.Series(dtype=float)
    upd = {}
    if len(sc_cost) >= 10:
        upd["sc_pit_factor"] = float(sc_cost.median())
    if len(vsc_cost) >= 10:
        upd["vsc_pit_factor"] = float(vsc_cost.median())
    if rates["sc"]["eligible"] >= 15:
        upd["sc_pit_prob"] = float(rates["sc"]["rate"])
    if rates["vsc"]["eligible"] >= 15:
        upd["vsc_pit_prob"] = float(rates["vsc"]["rate"])
    base = Params(**{**asdict(base), **upd})
    wet_ids = list(wet[wet].index)
    noise_pos = lap_noise_by_position(laps, wet_ids)
    if len(noise_pos) and (noise_pos["laps"] > 500).all():
        m = dict(zip(noise_pos["bucket"], noise_pos["multiplier"]))
        base = Params(**{**asdict(base), "noise_top": float(m["top 3"]), "noise_mid": float(m["4 to 10"]),
                         "noise_rest": float(m["11 and below"])})
    da = dirty_air(laps, wet_ids)
    dep_raw = lap_dependence(laps, wet_ids)
    dep = lap_dependence(laps, wet_ids, da=da) if da["da_loss"] > 0 else dep_raw
    if da["da_loss"] > 0:
        base = Params(**{**asdict(base), "da_loss": da["da_loss"], "da_range": da["da_range"]})
    if dep["drift_sd"] > 0:
        white = math.sqrt(max(base.sigma_lap ** 2 - dep["drift_sd"] ** 2, (0.3 * base.sigma_lap) ** 2))
        base = Params(**{**asdict(base), "drift_sd": dep["drift_sd"], "drift_rho": dep["drift_rho"], "sigma_lap": white})
        sigma_lap = white
    cover = cover_rates(truths)
    base = Params(**{**asdict(base), "cover_prob": cover["cover_prob"]})
    base = Params(**{**asdict(base), **pit_spreads(stops, base)})
    edge, circ_tab, disp_tab, drift_tab = [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    outlap_tab, start_tab, drift_share = pd.DataFrame(), pd.DataFrame(), math.nan
    if len(train) >= 5:
        par, grid_tab, edge = fit_passing(train, base, S=S, thr_grid=thr_grid, start_grid=start_grid)
        # the stop spread is measured again with the lap noise the fit settled on (it was first set with the measured one)
        par = Params(**{**asdict(par), **pit_spreads(stops, par)})
        known = [t for t in truths if t.season <= FIT_SEASONS[1]]
        cthr, circ_tab = circuit_passing(known, par, S=max(20, S // 2))
        par = Params(**{**asdict(par), "circuit_thr": cthr})
        thr_sd, disp_tab = fit_pass_difficulty(train, par, S=max(20, S))
        par = Params(**{**asdict(par), "thr_sd": thr_sd})
        tr_ids = {t.race_id for t in train}
        real_ev = race_flow.undercut_table(laps[laps["race_id"].isin(tr_ids)], flags[flags["race_id"].isin(tr_ids)])
        share, outlap_tab = fit_outlap(train, par, real_ev)
        par = Params(**{**asdict(par), "outlap_share": share})
        drift_tab = drift_check(train, par, S=max(20, S // 2))
        start_tab = grid_tab.attrs.get("start_shape", pd.DataFrame())
        drift_share = grid_tab.attrs.get("drift_share", math.nan)
    else:
        par, grid_tab = base, pd.DataFrame()
    return {"truths": truths, "skipped": skipped, "params": par, "base": base, "fit_grid": grid_tab, "era_tyres": era_t,
            "per_race": per_race, "rates": rates, "pit_cost": cost, "sigma_lap": sigma_lap, "circuit_wear": circuit_wear,
            "edge": edge, "circuit_passing": circ_tab, "lap_dependence": dep, "lap_dependence_raw": dep_raw, "dirty_air": da,
            "drift_check": drift_tab, "outlap_fit": outlap_tab, "start_shape": start_tab, "drift_share": drift_share, "warm_up": wu,
            "pass_difficulty": disp_tab, "noise_position": noise_pos, "cover": cover, "real_pass_rate": real_pass_rate(train) if train else math.nan,
            "real_close_share": real_close_share(train) if train else math.nan,
            "real_lap1_move": real_lap1_move(train) if train else math.nan}


def truth_summary(truths: list[RaceTruth]) -> pd.DataFrame:
    rows = []
    for t in truths:
        n_stops = (t.stops < BIG).sum(1)
        rows.append({"race_id": t.race_id, "season": t.season, "circuit_id": t.circuit_id, "split": split_of(t.season),
                     "cars": t.n, "laps": t.n_laps, "neutral_laps": int((t.neutral > 0).sum()),
                     "retirements": int((t.retire < BIG).sum()), "mean_stops": float(n_stops.mean()),
                     "cars_with_own_pace": int(t.pace_known.sum()), "pit_loss_s": t.pit_loss,
                     "pace_spread_s": float(np.std(t.pace)),
                     "real_pass_rate": (lambda pc: pc[0] / pc[1] if pc[1] else math.nan)(real_passes(t))})
    return pd.DataFrame(rows)


def findings(c: dict) -> list[str]:
    t = truth_summary(c["truths"])
    L = []
    L.append(f"RACES: {len(c['truths'])} dry races without a red flag are usable "
             f"({int((t['split'] == 'train').sum())} fit 2022-24, {int((t['split'] == 'holdout').sum())} held out from 2025, "
             f"{int((t['split'] == 'earlier era').sum())} from 2018-21).")
    sk = c["skipped"]
    if len(sk):
        L.append("  left out: " + ", ".join(f"{k} {v}" for k, v in sk["reason"].value_counts().items()))
    if len(t):
        L.append(f"  cars with their own pace measured: {t['cars_with_own_pace'].sum()} of {t['cars'].sum()}; "
                 f"pace spread within a race (sd) median {t['pace_spread_s'].median():.2f} s")
    L.append("TYRES (era medians): " + "; ".join(
        f"{r['era']}: wear soft {r['deg_SOFT']:.3f}, medium {r['deg_MEDIUM']:.3f}, hard {r['deg_HARD']:.3f} s/lap, "
        f"soft {r['off_SOFT']:+.2f} and hard {r['off_HARD']:+.2f} s against medium at age 0, trend {r['fuel']:+.3f} s/lap"
        for _, r in c["era_tyres"].iterrows()))
    if len(t):
        by = t.groupby("season")["pace_spread_s"].median()
        L.append("PACE SPREAD within a race (sd of car pace, s), median by season: " + ", ".join(f"{int(k)} {v:.2f}" for k, v in by.items()))
    for _, r in c["era_tyres"].iterrows():
        if not (r["deg_SOFT"] >= r["deg_MEDIUM"] >= r["deg_HARD"]):
            L.append(f"  CHECK: {r['era']} wear is not ordered soft >= medium >= hard ({r['deg_SOFT']:.3f}, {r['deg_MEDIUM']:.3f}, "
                     f"{r['deg_HARD']:.3f}); the era medians mix compound ranges that differ by weekend")
    r = c["rates"]
    L.append(f"STOPS UNDER NEUTRALISATION: safety car, {r['sc']['came_in']} of {r['sc']['eligible']} eligible cars came in "
             f"({r['sc']['rate']:.0%}); VSC, {r['vsc']['came_in']} of {r['vsc']['eligible']} ({r['vsc']['rate']:.0%})")
    cost = c["pit_cost"]
    if len(cost):
        for kind in ("sc", "vsc"):
            s = cost[(cost["kind"] == kind) & (cost["share"] > 0) & (cost["share"] < 1.3)]["share"]
            if len(s):
                L.append(f"  a stop under a {'safety car' if kind == 'sc' else 'VSC'} costs {s.median():.0%} of a green-flag stop "
                         f"(median of {len(s)} stops, middle half {s.quantile(.25):.0%} to {s.quantile(.75):.0%})")
    p, b = c["params"], c["base"]
    L.append(f"SETTINGS USED: lap noise {p.sigma_lap:.2f} s, safety-car stop probability {p.sc_pit_prob:.2f}, VSC {p.vsc_pit_prob:.2f}, "
             f"safety-car stop cost {p.sc_pit_factor:.2f}, VSC {p.vsc_pit_factor:.2f}, dirty-air loss {p.da_loss:.2f} s over {p.da_range:.1f} s")
    if len(c["fit_grid"]):
        g = c["fit_grid"].sort_values("loss")
        L.append(f"FITTED on 2022-24 (grip decay, on-track pass rate, share of car-laps within {CLOSE_S:.0f} s of the car ahead): passing threshold "
                 f"{p.thr:.2f} s/lap, closest follow {p.min_gap:.2f} s, grid spacing {p.slot_gap:.2f} s, lap noise {p.sigma_lap:.2f} s "
                 f"(loss {g['loss'].iloc[0]:.2f}; one unit is a 0.02 gap in grip, a 15% gap in passes or a 10% gap in the close share)")
        last = g.iloc[0]
        L.append(f"  on-track passes per car-lap, 2022-24: real {last['real_pass_rate']:.4f}, simulated {last['sim_pass_rate']:.4f}")
        if np.isfinite(last.get("real_close_share", math.nan)):
            L.append(f"  car-laps within {CLOSE_S:.0f} s of the car ahead (green, away from stops), 2022-24: real {last['real_close_share']:.1%}, "
                     f"simulated {last['sim_close_share']:.1%}")
        else:
            L.append("  no race-clock times for the line crossings, so the close-running target was not used (re-run stage 1 for them)")
        if np.isfinite(c.get("real_lap1_move", math.nan)):
            L.append(f"  LAP 1 is measured, not fitted: cars end lap 1 on average {c['real_lap1_move']:.2f} places from their grid slot; "
                     f"lap-1 noise {p.sigma_start:.2f} s reproduces that, with the pole sitter's noise {p.start_front:.2f} of the field's "
                     f"(rising to the field's by slot {int(1 + p.start_ramp)}; chosen on places moved by grid-slot band)")
            ss = c.get("start_shape")
            if ss is not None and len(ss):
                sel = (ss["start_front"] == p.start_front) & ((ss["bad_start_p"] == p.bad_start_p) if "bad_start_p" in ss else True)
                best = ss[sel].drop_duplicates("slots")
                if len(best):
                    L.append("  places moved on lap 1 by grid slot (real vs simulated): " + "; ".join(
                        f"{r['slots']} {r['real_places_moved']:.2f} vs {r['simulated_places_moved']:.2f}" for _, r in best.iterrows()))
                    if "real_pole_keeps_lead" in best:
                        r0 = best.iloc[0]
                        L.append(f"  bad starts: chance {p.bad_start_p:.0%} of losing {p.bad_start_loss:.1f} s on average (0% = none kept); the pole sitter "
                                 f"is still first after lap 1 in {r0['simulated_pole_keeps_lead']:.1%} of simulations against {r0['real_pole_keeps_lead']:.1%} real; "
                                 f"front-six cars move 3+ places on {r0['simulated_front_big_moves']:.1%} against {r0['real_front_big_moves']:.1%} real")
        else:
            L.append(f"  no lap-1 positions, so lap-1 noise was searched with the grip curve: {p.sigma_start:.2f} s")
        if c.get("edge"):
            L.append("  WARNING: settings at the edge of their search range: " + ", ".join(c["edge"]) +
                     " - the best value may lie outside it")
        else:
            L.append("  no setting sits at the edge of its search range")
    wu = c.get("warm_up")
    if wu is not None and len(wu["table"]):
        t_all = wu["table"][wu["table"]["era"] == "all"]
        L.append(f"TYRE WARM-UP ({wu['note']}): excess lap time on stint laps 2 to {int(t_all['stint_lap'].max())} over stint laps {WARM_REF[0]}-{WARM_REF[1]}, "
                 "all years: " + "; ".join(f"lap {int(r['stint_lap'])} {r['excess_s']:+.3f} s (se {r['se_s']:.3f})" for _, r in t_all.iterrows()))
        L.append(f"  fitted: {wu['decay']:.2f} of it carried to the next lap; amplitude on stint lap 2 " + ", ".join(
            f"{k} {v:.2f} s" for k, v in wu["amp"].items()) + f" (all years {wu['pooled']:.2f} s). The simulator charges it on laps after a stop.")
    ol = c.get("outlap_fit")
    if ol is not None and len(ol):
        L.append("UNDERCUT FIT (2022-24 as it happened; advantage in points, stops 1-2 laps apart / 3+ laps apart; real "
                 f"{100 * ol['real_close'].iloc[0]:+.1f} / {100 * ol['real_wide'].iloc[0]:+.1f}): " + "; ".join(
                     f"{r['outlap_share']:.0%} of the stop paid on the out-lap {100 * r['simulated_close']:+.1f} / {100 * r['simulated_wide']:+.1f}"
                     for _, r in ol.iterrows()) + f"; kept {p.outlap_share:.0%}")
        if p.outlap_share >= max(OUTLAP_GRID):
            L.append("  WARNING: out-lap share at the top of its search range")
        if ol["simulated_close"].max() - ol["simulated_close"].min() < 0.03:
            L.append("  NOTE: the out-lap split barely moves the undercut, so the close-stops excess (if any) comes from something else")
    ds = c.get("drift_share")
    if ds is not None and np.isfinite(ds):
        L.append(f"  drift share kept by the fit: {ds:.0%} of the measured drift (the rest moved to white lap noise)")
    da = c.get("dirty_air")
    if da is not None and da.get("laps"):
        tab = da["table"]
        if len(tab):
            L.append(f"DIRTY AIR (from {da['laps']} clean laps, same stint, against laps that started 3 s or more behind the car ahead): " + "; ".join(
                f"{r['gap_from_s']:.1f}-{r['gap_to_s']:.1f} s behind {r['loss_s']:+.2f} s ({r['low_s']:+.2f} to {r['high_s']:+.2f}, {int(r['laps'])} laps)"
                for _, r in tab.iterrows()))
        if da["da_loss"] > 0:
            L.append(f"  straight-line fit: {da['da_loss']:.2f} s lost when right behind a car, falling to nothing at {da['da_range']:.1f} s; "
                     f"the simulator uses {p.da_loss:.2f} s (the fit tried 0 to 3 times the measured loss; {p.da_loss / da['da_loss']:.1f} times kept)")
        else:
            L.append(f"  {da['note']}; no dirty air in the simulator")
    elif da is not None:
        L.append(f"DIRTY AIR: {da['note']}; none in the simulator")
    dep = c.get("lap_dependence")
    dep_raw = c.get("lap_dependence_raw")
    if dep and dep["stints"]:
        if dep["drift_sd"] > 0:
            L.append(f"LASTING SPELLS (from {dep['stints']} clean stints): lap-time residuals {dep['pairs_lag1']} pairs 1 lap apart have a raw mean product "
                     f"{dep['cov_lag1']:.3f} s2 and {dep['pairs_lag5']} pairs 5 laps apart {dep['cov_lag5']:.3f} s2 (before allowing for what a "
                     f"straight-line fit absorbs); the fitted variance is {dep['variance']:.3f} s2, made of a pace drift of sd {dep['drift_sd']:.2f} s that keeps {dep['drift_rho']:.2f} of itself from lap to "
                     f"lap, on top of {p.sigma_lap:.2f} s of lap noise")
            if dep.get("capped"):
                L.append("  NOTE: the drift variance hit its ceiling (0.7 of the total), so the data want at least that much lasting variation; "
                         "treat the size as a floor, not a measurement")
            if dep.get("dirty_air_removed") and dep_raw and dep_raw["drift_sd"] > 0:
                L.append(f"  before taking dirty air off the lap times the drift was sd {dep_raw['drift_sd']:.2f} s (keeping {dep_raw['drift_rho']:.2f}); "
                         f"after, {dep['drift_sd']:.2f} s")
        else:
            L.append(f"LASTING SPELLS: {dep['note']} (variance {dep['variance']:.3f}, covariance 1 lap apart {dep['cov_lag1']:.3f}, "
                     f"5 laps apart {dep['cov_lag5']:.3f}); no drift in the simulator")
    dc = c.get("drift_check")
    if dc is not None and len(dc):
        L.append("DRIFT CHECK (fitted settings, drift scaled down, total lap spread kept, not re-fitted; loss in the units of the fit): " + "; ".join(
            f"{r['drift_share']:.0%} of the drift: loss {r['loss']:.2f}, passes {r['sim_pass_rate']:.4f}, close share {r['sim_close_share']:.1%}"
            for _, r in dc.iterrows()) + f" (real: passes {dc['real_pass_rate'].iloc[0]:.4f}, close share {dc['real_close_share'].iloc[0]:.1%})")
    pdf = c.get("pass_difficulty")
    if pdf is not None and len(pdf):
        L.append(f"PASSING DIFFICULTY FROM RACE TO RACE: real passes per race differ from the simulator's expectation "
                 f"{pdf['dispersion'].iloc[0]:.1f} times as much as its own variance allows with a fixed difficulty; spread of difficulty used "
                 f"(log scale, an upper bound) {p.thr_sd:.2f}" + (f", which brings that ratio to {float(pdf.loc[pdf['thr_sd'] == p.thr_sd, 'dispersion'].iloc[0]):.1f}"
                                                                   if (pdf['thr_sd'] == p.thr_sd).any() else ""))
    top = f"; from the top three {p.sigma_pit_top:.2f} s" if p.sigma_pit_top >= 0 else ""
    L.append(f"MEASURED stop-cost spread beyond lap noise (redone with the fitted lap noise): {p.sigma_pit:.2f} s{top}")
    npz = c.get("noise_position")
    if npz is not None and len(npz):
        L.append("LAP NOISE BY RUNNING POSITION (clean laps, robust sd around the stint trend, field effect removed): " + "; ".join(
            f"{r['bucket']} {r['noise_sd_s']:.2f} s ({int(r['laps'])} laps, x{r['multiplier']:.2f})" for _, r in npz.iterrows())
            + f". Simulator multipliers: top 3 {p.noise_top:.2f}, 4-10 {p.noise_mid:.2f}, rest {p.noise_rest:.2f}")
    cv = c.get("cover")
    if cv:
        L.append(f"COVER STOPS: a car with its own stop due soon stopped on the next lap in {cv['exposed_stopped']} of {cv['exposed']} "
                 f"cases when the car right behind it had just stopped ({cv['exposed_rate']:.1%}) against {cv['unexposed_stopped']} of "
                 f"{cv['unexposed']} otherwise ({cv['unexposed_rate']:.1%}); cover probability used in sampled races {p.cover_prob:.2f}")
    ct = c.get("circuit_passing")
    if ct is not None and len(ct):
        hard, easy = ct.sort_values("thr_used").iloc[-1], ct.sort_values("thr_used").iloc[0]
        L.append(f"CIRCUIT PASSING (from {int(ct['races'].sum())} races in 2018-24, shrunk): thresholds run from {easy['thr_used']:.2f} s/lap "
                 f"({easy['circuit_id']}, easiest to pass) to {hard['thr_used']:.2f} ({hard['circuit_id']}, hardest)")
    return L


def chart_grip(table: pd.DataFrame, path: Path) -> None:
    from ..analysis.grid import BLUE, ORANGE, AXIS, TEXT2, _frame, _plt
    plt = _plt()
    fig, ax = _frame(plt, "Does the simulator dissolve the grid like the real races?",
                     "Rank correlation of grid and running order through the race, mean over dry races. "
                     "Simulated with the real pace, strategies, safety cars and retirements.", size=(9.5, 5.4))
    for split, colour, ls in (("train", BLUE, "-"), ("holdout", ORANGE, "-")):
        d = table[table["split"] == split].dropna().groupby("fraction")[["actual", "simulated"]].mean()
        if d.empty:
            continue
        ax.plot(d.index, d["actual"], color=colour, lw=2.2, marker="o", ms=4, label=f"{split}: real")
        ax.plot(d.index, d["simulated"], color=colour, lw=1.8, ls="--", marker="s", ms=3.5, label=f"{split}: simulated")
    ax.set_xlabel("Share of the race distance")
    ax.set_ylabel("Grid vs running order")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def run(con, out_dir: Path | None = None, S: int = 40) -> list[str]:
    out = Path(out_dir) if out_dir else config.outputs_dir() / "sim_calibration"
    out.mkdir(parents=True, exist_ok=True)
    c = calibrate(con, S=S)
    truths = c["truths"]
    with open(out / "truth.pkl", "wb") as f:
        pickle.dump([asdict(t) for t in truths], f)        # plain dicts: a pickle of the class breaks when run with -m
    (out / "params.json").write_text(json.dumps({"params": asdict(c["params"]), "sigma_lap": c["sigma_lap"]}, indent=2), encoding="utf-8")
    truth_summary(truths).to_csv(out / "race_truth_summary.csv", index=False)
    c["era_tyres"].to_csv(out / "era_tyres.csv", index=False)
    c["fit_grid"].to_csv(out / "passing_fit_grid.csv", index=False)
    c["skipped"].to_csv(out / "races_left_out.csv", index=False)
    c["pit_cost"].to_csv(out / "neutral_pit_cost.csv", index=False)
    c["circuit_passing"].to_csv(out / "circuit_passing.csv", index=False)
    c["noise_position"].to_csv(out / "noise_by_position.csv", index=False)
    pd.DataFrame([c["lap_dependence"]]).to_csv(out / "lap_dependence.csv", index=False)
    pd.DataFrame([c["lap_dependence_raw"]]).to_csv(out / "lap_dependence_before_dirty_air.csv", index=False)
    c["dirty_air"]["table"].to_csv(out / "dirty_air.csv", index=False)
    c["drift_check"].to_csv(out / "drift_check.csv", index=False)
    c["outlap_fit"].to_csv(out / "undercut_fit.csv", index=False)
    c["start_shape"].to_csv(out / "start_shape.csv", index=False)
    c["warm_up"]["table"].to_csv(out / "warm_up.csv", index=False)
    c["pass_difficulty"].to_csv(out / "pass_difficulty.csv", index=False)
    if "by_season" in c["noise_position"].attrs:
        c["noise_position"].attrs["by_season"].to_csv(out / "noise_by_position_season.csv", index=False)
    pd.DataFrame([c["cover"]]).to_csv(out / "cover_stops.csv", index=False)
    if len(truths):
        tab = grip_fit_table(truths, c["params"], S=S, seed=11)
        tab["split"] = tab["season"].map(split_of)
        tab.to_csv(out / "grip_fit.csv", index=False)
        chart_grip(tab, out / "grip_fit.png")
    lines = findings(c)
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def load_truths(out_dir: Path | None = None) -> tuple[list[RaceTruth], Params]:
    d = Path(out_dir) if out_dir else config.outputs_dir() / "sim_calibration"
    if not (d / "truth.pkl").exists() or not (d / "params.json").exists():
        raise FileNotFoundError(f"Run  python -m src.sim.calibrate  first (no calibration in {d}).")
    with open(d / "truth.pkl", "rb") as f:
        truths = [RaceTruth(**x) for x in pickle.load(f)]
    meta = json.loads((d / "params.json").read_text(encoding="utf-8"))
    return truths, Params(**meta["params"])


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
    print(f"\nTables written to {config.outputs_dir() / 'sim_calibration'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
