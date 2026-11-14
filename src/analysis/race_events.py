"""Shared building blocks for the deep-dive analyses (what_decides.py, strategy.py).

What happened to each car in each race, taken from the Jolpica results status, the Jolpica pit stops
(2011 on) and, where the FastF1 laps exist (2018 on), the safety-car laps.

Status classes
    finished      classified on the lead lap ("Finished")
    lapped        classified, one or more laps down ("+1 Lap", "+2 Laps", "Lapped")
    incident      retired after contact or an off ("Collision", "Accident", "Spun off", "Damage" ...)
    mechanical    retired with a failure, or for any reason not recognised as an incident
                  (engine, gearbox, hydraulics, "Retired" with no detail ...)
    disqualified  disqualified or excluded: a ruling, not a performance, so these rows are dropped
    not_started   did not start, did not qualify, withdrew, injured, ill: dropped

The split between "incident" and "mechanical" is read from a free-text field, so it is approximate; a bare "Retired" is kept as "unspecified".
A puncture or a wheel failure is counted as mechanical although contact often causes it.

Strategy features (per driver-race, from pit stops)
    n_stops       number of stops
    first_stop    lap of the first stop as a fraction of the race distance
    pit_time      total pit-lane time, seconds (clipped at 60 s per stop so a repair does not dominate)
    pit_excess    pit_time minus the car's stops times the race's median seconds per stop: how much slower or
                  quicker the pit work was than the field's, with the effect of the number of stops removed
Safety-car features (need FastF1 laps)
    free_stops    stops made on a lap neutralised by a safety car, VSC or red flag
    unlucky_stops stops made in the three laps before a neutralisation began (the field then gains
                  a cheap stop the car just missed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL_CODES = "4567"          # safety car, red flag, VSC, VSC ending (the FastF1 track-status codes)
NEUTRAL_SHARE = 0.25            # share of cars that must record a neutral code on a lap for it to count
UNLUCKY_WINDOW = 3              # laps before a neutralisation in which a stop counts as unlucky
PIT_CLIP_S = 60.0

_FINISHED = ("finished", "lapped")
_NOT_STARTED = ("did not start", "did not qualify", "did not prequalify", "withdrew", "illness", "injured", "injury",
                "unwell", "sick", "fatal", "eye ", "physical", "not classified", "safety concerns", "excluded from",
                "driver unwell", "death", "stalled on grid")
_DISQUALIFIED = ("disqualified", "excluded", "black flag")
_INCIDENT = ("collision", "accident", "spun off", "spin", "crash", "damage", "debris", "contact", "hit ", "off track")


def classify_status(status) -> str:
    """One of finished, lapped, incident, mechanical, unspecified, disqualified, not_started.

    "unspecified" is a bare "Retired" (or a blank): the data does not say why the car stopped, and in recent seasons
    it is used for crashes as well as failures, so it is kept apart rather than guessed."""
    s = str(status or "").strip().lower()
    if s in ("", "nan", "none", "retired"):
        return "unspecified"
    if s == "finished":
        return "finished"
    if s.startswith("+") or s == "lapped":
        return "lapped"
    if any(k in s for k in _DISQUALIFIED):
        return "disqualified"
    if any(k in s for k in _NOT_STARTED):
        return "not_started"
    if any(k in s for k in _INCIDENT):
        return "incident"
    return "mechanical"


def add_status(results: pd.DataFrame) -> pd.DataFrame:
    """Add status_class, finished (bool) and completed (share of the winner's laps) to a results table."""
    d = results.copy()
    d["status_class"] = d["status"].map(classify_status)
    d["finished"] = d["status_class"].isin(_FINISHED)
    laps = pd.to_numeric(d["laps"], errors="coerce")
    top = laps.groupby(d["race_id"]).transform("max")
    d["completed"] = (laps / top).clip(0, 1).fillna(0.0)
    return d


# ----------------------------------------------------------------- pit-stop features

def stop_features(results: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Per driver-race strategy features. A car with no recorded stop gets n_stops = 0 (and no first_stop)."""
    laps = pd.to_numeric(results["laps"], errors="coerce")
    race_laps = laps.groupby(results["race_id"]).transform("max").rename("race_laps")
    base = results[["race_id", "driver_id"]].assign(race_laps=race_laps.to_numpy())
    base = base.drop_duplicates(["race_id", "driver_id"])
    if pit_stops is None or not len(pit_stops):
        return base.assign(n_stops=np.nan, first_lap=np.nan, first_stop=np.nan, pit_time=np.nan, pit_excess=np.nan)[
            ["race_id", "driver_id", "n_stops", "first_lap", "first_stop", "pit_time", "pit_excess"]]
    ps = pit_stops[["race_id", "driver_id", "stop", "lap", "duration_s"]].copy()
    ps["duration_s"] = pd.to_numeric(ps["duration_s"], errors="coerce").clip(upper=PIT_CLIP_S)
    g = ps.groupby(["race_id", "driver_id"]).agg(n_stops=("stop", "count"), first_lap=("lap", "min"),
                                                 pit_time=("duration_s", "sum")).reset_index()
    out = base.merge(g, on=["race_id", "driver_id"], how="left")
    covered = out.groupby("race_id")["n_stops"].transform(lambda s: s.notna().any())
    out["n_stops"] = out["n_stops"].fillna(0).where(covered)
    out["pit_time"] = out["pit_time"].fillna(0.0).where(covered)
    out["first_stop"] = (out["first_lap"] / out["race_laps"]).where(covered)
    per_stop = (out["pit_time"] / out["n_stops"].where(out["n_stops"] > 0))
    med = per_stop.groupby(out["race_id"]).transform("median")
    out["pit_excess"] = (out["pit_time"] - out["n_stops"] * med).where(covered)
    return out[["race_id", "driver_id", "n_stops", "first_lap", "first_stop", "pit_time", "pit_excess"]]


def neutral_laps(laps: pd.DataFrame) -> dict[int, list[int]]:
    """Per race, the lap numbers on which a safety car, VSC or red flag was out (at least 25% of the cars say so)."""
    if laps is None or not len(laps) or "track_status" not in laps.columns:
        return {}
    d = laps[["race_id", "lap_number", "track_status"]].copy()
    d["neutral"] = d["track_status"].astype(str).map(lambda s: any(c in s for c in NEUTRAL_CODES))
    share = d.groupby(["race_id", "lap_number"])["neutral"].mean()
    flagged = share[share >= NEUTRAL_SHARE].reset_index()
    return {int(r): sorted(int(x) for x in g["lap_number"]) for r, g in flagged.groupby("race_id")}


def safety_car_features(results: pd.DataFrame, pit_stops: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """free_stops and unlucky_stops for every driver-race of the races that have lap data (zeros for cars that
    made no such stop); races without lap data are absent, so a merge leaves them missing."""
    cols = ["race_id", "driver_id", "free_stops", "unlucky_stops"]
    windows = neutral_laps(laps)
    if laps is None or not len(laps) or pit_stops is None:
        return pd.DataFrame(columns=cols)
    have_laps = set(laps["race_id"].unique())
    base = results.loc[results["race_id"].isin(have_laps), ["race_id", "driver_id"]].drop_duplicates()
    counts: dict[tuple, list[int]] = {}
    for race_id, g in pit_stops[pit_stops["race_id"].isin(have_laps)].groupby("race_id"):
        neutral = set(windows.get(int(race_id), []))
        starts = sorted(l for l in neutral if (l - 1) not in neutral)
        for driver_id, gd in g.groupby("driver_id"):
            lap_list = [int(l) for l in gd["lap"]]
            free = sum(1 for l in lap_list if l in neutral)
            unlucky = sum(1 for l in lap_list if l not in neutral and any(0 < st - l <= UNLUCKY_WINDOW for st in starts))
            counts[(race_id, driver_id)] = [free, unlucky]
    vals = [counts.get((r, d), [0, 0]) for r, d in zip(base["race_id"], base["driver_id"])]
    base = base.assign(free_stops=[v[0] for v in vals], unlucky_stops=[v[1] for v in vals])
    return base[cols]


def deviations(rows: pd.DataFrame, cols: list[str], by: str = "race_id") -> pd.DataFrame:
    """Each column minus the race median among cars that finished (so a retirement's truncated stops do not
    set the benchmark). Cars that did not finish get 0: their strategy cannot be judged."""
    d = rows.copy()
    for c in cols:
        ref = d.loc[d["finished"], c].groupby(d.loc[d["finished"], by]).median()
        d[f"d_{c}"] = (d[c] - d[by].map(ref)).where(d["finished"]).fillna(0.0)
    return d
