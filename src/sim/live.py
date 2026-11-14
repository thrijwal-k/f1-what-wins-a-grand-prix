"""Stage 4: forecasts made before a race, kept in a log nobody edits, and scored afterwards.

    python -m src.sim.live next                    which race is next, and whether its qualifying is in the database
    python -m src.sim.live forecast [--race ID]    make and log the forecast (after qualifying, before the race)
    python -m src.sim.live score                   score every logged forecast whose race has a result
    python -m src.sim.live verify                  check that no logged forecast has been changed

What a forecast is. The frozen version 2.8 simulator (settings in outputs/sim_calibration/params.json, never refitted here) is
run for the race from the races before it only. Pace is the history-based estimate plus the first-session qualifying gap
("History + quali gap"), recalibrated; the benchmark is the model that knows only the grid slot. Every driver gets a probability of
winning, finishing on the podium and finishing in the top ten.

Rules that keep it honest
  - A race that already has results, or whose start time has passed, cannot be forecast.
  - Each forecast is one file in forecasts/ named by race and UTC time. It holds the probabilities, a fingerprint (SHA-256) of the
    code and settings, a fingerprint of the inputs, and the fingerprint of the previous forecast file, so changing or removing an
    old one breaks the chain (`verify`). The date that matters is the date the file was committed to a public repository.
  - If a race has several forecasts, the last one made before the start is the one scored.
  - Wet races are forecast like any other but scored apart: the simulator is calibrated on dry races only.

Known limits. The qualifying order stands in for the starting grid, so penalties announced after qualifying are not in it. The tyre
inputs are those of the last race at the same circuit (the earlier backtest gave each race its own; python -m src.sim.forecast
--proxy-tyre measures the difference). Practice laps are not used: they added nothing in the backtest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import config
from . import calibrate as C
from . import forecast as F
from . import replay as R
from .engine import Params
from .sample import Pool

SCHEMA = 1
PRIMARY = "History + quali gap, recalibrated"
BASELINE = F.BASELINE
LOGGED = (PRIMARY, "History + quali gap", "History + grid, recalibrated", "History + grid", BASELINE)
RECAL = {"History + grid, recalibrated": "History + grid", PRIMARY: "History + quali gap"}
MIN_QUALI_CARS = 10
HISTORY_SIMS = 300       # simulations per earlier race when learning the blend and the recalibration (as in the backtest)
LIVE_SIMS = 1000         # simulations for the race being forecast
INDEX = "log.csv"
INDEX_COLUMNS = ["race_id", "created_utc", "file", "sha256", "prev_sha256"]
GENESIS = "genesis"


class ForecastError(RuntimeError):
    """The forecast must not be made (race already run, qualifying missing, too little history ...)."""


# ------------------------------------------------------------------ hashes

def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def code_fingerprint(cal_dir: Path | None = None) -> str:
    """One hash over every simulator source file and the frozen settings, so a forecast shows which code and settings made it."""
    cal = Path(cal_dir) if cal_dir else config.outputs_dir() / "sim_calibration"
    h = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    params = cal / "params.json"
    h.update(params.read_bytes() if params.exists() else b"")
    return h.hexdigest()


def load_params(cal_dir: Path | None = None) -> Params:
    cal = Path(cal_dir) if cal_dir else config.outputs_dir() / "sim_calibration"
    path = cal / "params.json"
    if not path.exists():
        raise ForecastError(f"No frozen settings at {path}. Run  python -m src.sim.calibrate  and  python -m src.sim.refit  first.")
    return Params(**json.loads(path.read_text(encoding="utf-8"))["params"])


# ------------------------------------------------------------------ the race

def _utc(ts) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def parse_utc(text: str) -> datetime:
    return _utc(datetime.fromisoformat(str(text)))


def race_info(con, race_id: int) -> dict:
    row = con.execute("SELECT race_id, season, round, race_name, circuit_id, race_date, race_time FROM races WHERE race_id = ?",
                      [race_id]).fetchone()
    if row is None:
        raise ForecastError(f"Race {race_id} is not in the database.")
    keys = ["race_id", "season", "round", "race_name", "circuit_id", "race_date", "race_time"]
    return dict(zip(keys, row))


def race_start(info: dict) -> datetime:
    """The scheduled start in UTC; midnight of the race date when the time is not in the data (so a forecast on race day is late)."""
    d = pd.Timestamp(info["race_date"]).to_pydatetime()
    t = str(info.get("race_time") or "").strip().rstrip("Z")
    if t and t.lower() not in ("none", "nan"):
        try:
            h, m, *_ = [int(x) for x in t.split(":")]
            d = d.replace(hour=h, minute=m)
        except ValueError:
            pass
    return d.replace(tzinfo=timezone.utc)


def next_race(con, now: datetime | None = None) -> dict | None:
    """The earliest race without results that has not started."""
    now = _utc(now or datetime.now(timezone.utc))
    rows = con.execute("""
        SELECT r.race_id FROM races r
        WHERE NOT EXISTS (SELECT 1 FROM results x WHERE x.race_id = r.race_id) AND r.race_date >= ?
        ORDER BY r.race_id LIMIT 1""", [(now.date() - __import__("datetime").timedelta(days=1)).isoformat()]).fetchall()
    if not rows:
        return None
    info = race_info(con, rows[0][0])
    n_q = con.execute("SELECT count(*) FROM qualifying WHERE race_id = ?", [info["race_id"]]).fetchone()[0]
    return {**info, "start_utc": race_start(info).isoformat(timespec="minutes"), "qualifying_rows": int(n_q),
            "ready": n_q >= MIN_QUALI_CARS and race_start(info) > now}


def check_open(con, race_id: int, now: datetime) -> dict:
    info = race_info(con, race_id)
    if con.execute("SELECT count(*) FROM results WHERE race_id = ?", [race_id]).fetchone()[0]:
        raise ForecastError(f"Race {race_id} already has results in the database; a forecast now would not be a forecast.")
    start = race_start(info)
    if now >= start:
        raise ForecastError(f"Race {race_id} starts {start:%Y-%m-%d %H:%M} UTC, which has passed.")
    n_q = con.execute("SELECT count(*) FROM qualifying WHERE race_id = ?", [race_id]).fetchone()[0]
    if n_q < MIN_QUALI_CARS:
        raise ForecastError(f"Qualifying for race {race_id} is not in the database yet ({n_q} rows). "
                            "Update the data after qualifying:  python -m src.run_all --from-season "
                            f"{info['season']} --fastf1-from {info['season']} --skip-practice")
    return info


def card_for(con, info: dict, truths: list) -> F.RaceCard:
    q = con.execute("SELECT driver_id, constructor_id, position FROM qualifying WHERE race_id = ?", [info["race_id"]]).df()
    q = q.dropna(subset=["position"]).drop_duplicates("driver_id").sort_values("position")
    drivers, constructors = q["driver_id"].tolist(), q["constructor_id"].tolist()
    slot = np.arange(1, len(drivers) + 1)
    quali = C.load_quali(con)
    gap = C.quali_gaps(quali, info["race_id"], drivers)
    same = [t for t in truths if t.circuit_id == info["circuit_id"]]
    n_laps = int(same[-1].n_laps) if same else int(np.median([t.n_laps for t in truths[-20:]]))
    return F.RaceCard(info["race_id"], int(info["season"]), info["circuit_id"], n_laps, drivers, constructors, slot, gap)


# ------------------------------------------------------------------ the forecast

def model_probabilities(sims: dict, card: F.RaceCard, pool: Pool, state: dict) -> dict[str, dict[str, np.ndarray]]:
    base = {name: F.probabilities(res.finish) for name, res in sims.items()}
    out = {name: {k: base[name][k] for k in ("p_win", "p_top3", "p_top10")} for name in ("History + quali gap", "History + grid")}
    for name, source in RECAL.items():
        hp = state["past"][source]
        if len(hp["y_win"]) < F.MIN_RECAL_CARS:
            raise ForecastError(f"Only {len(hp['y_win'])} earlier forecast car-races to recalibrate on; {F.MIN_RECAL_CARS} needed.")
        q = {}
        for key, col in (("p_win", "win"), ("p_top3", "top3"), ("p_top10", "top10")):
            ab = F.fit_recal(np.array(hp[key]), np.array(hp[f"y_{col}"], float))
            q[key] = F.apply_recal(base[source][key], ab)
        q["p_win"] = q["p_win"] / q["p_win"].sum()
        out[name] = q
    gp = R.grid_probs(pool, card.slot)
    out[BASELINE] = {"p_win": gp[1], "p_top3": gp[3], "p_top10": gp[10]}
    return out


def build_record(info: dict, card: F.RaceCard, probs: dict, hist: list, now: datetime, sims: int, history_sims: int,
                 code_sha: str, prev_sha: str) -> dict:
    drivers = []
    for i, d in enumerate(card.drivers):
        drivers.append({"driver_id": d, "constructor_id": card.constructors[i], "slot": int(card.slot[i]),
                        "quali_gap_pct": round(float(card.quali_gap[i]), 4),
                        "models": {m: {k: round(float(probs[m][k][i]), 5) for k in ("p_win", "p_top3", "p_top10")} for m in LOGGED}})
    inputs = {"race_id": card.race_id, "circuit_id": card.circuit_id, "n_laps": card.n_laps, "drivers": card.drivers,
              "constructors": card.constructors, "slot": [int(x) for x in card.slot],
              "quali_gap_pct": [round(float(x), 4) for x in card.quali_gap], "history_through": int(hist[-1].race_id),
              "n_history": len(hist)}
    return {"schema": SCHEMA, "race_id": int(info["race_id"]), "season": int(info["season"]), "round": int(info["round"]),
            "race_name": info["race_name"], "circuit_id": info["circuit_id"],
            "race_start_utc": race_start(info).isoformat(timespec="minutes"),
            "created_utc": now.isoformat(timespec="seconds"), "primary_model": PRIMARY, "baseline_model": BASELINE,
            "sims": sims, "history_sims": history_sims, "assumes_dry": True,
            "grid_source": "qualifying position (penalties after qualifying not included)",
            "tyre_source": "last earlier race at the same circuit, else the last race",
            "history_through_race_id": inputs["history_through"], "n_history_races": len(hist),
            "code_sha256": code_sha, "inputs_sha256": sha256_of(inputs), "prev_sha256": prev_sha,
            "drivers": drivers}


# ------------------------------------------------------------------ the log

def read_index(folder: Path) -> pd.DataFrame:
    path = folder / INDEX
    if not path.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(path, dtype={"sha256": str, "prev_sha256": str, "file": str, "created_utc": str})


def last_sha(folder: Path) -> str:
    idx = read_index(folder)
    return GENESIS if idx.empty else str(idx.sort_values("created_utc").iloc[-1]["sha256"])


def write_forecast(record: dict, folder: Path) -> Path:
    """Add the file's own hash, write it (never over an existing file) and append it to the index."""
    folder.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in record.items() if k != "sha256"}
    full = {**body, "sha256": sha256_of(body)}
    stamp = datetime.fromisoformat(full["created_utc"]).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{full['race_id']}_{stamp}.json"
    if path.exists():
        raise ForecastError(f"{path.name} already exists; forecasts are never overwritten.")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(full, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    idx = path.parent / INDEX
    row = f"{full['race_id']},{full['created_utc']},{path.name},{full['sha256']},{full['prev_sha256']}\n"
    if not idx.exists():
        idx.write_text(",".join(INDEX_COLUMNS) + "\n", encoding="utf-8")
    with open(idx, "a", encoding="utf-8", newline="") as f:
        f.write(row)
    return path


def load_forecasts(folder: Path) -> list[dict]:
    out = []
    for _, row in read_index(folder).sort_values("created_utc").iterrows():
        out.append(json.loads((folder / row["file"]).read_text(encoding="utf-8")))
    return out


def verify(folder: Path) -> list[str]:
    """Problems found in the log: a missing or edited file, a broken chain, a file not in the index."""
    problems: list[str] = []
    idx = read_index(folder)
    prev = GENESIS
    for _, row in idx.sort_values("created_utc").iterrows():
        path = folder / row["file"]
        if not path.exists():
            problems.append(f"{row['file']}: listed in {INDEX} but the file is missing")
            prev = row["sha256"]
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        body = {k: v for k, v in rec.items() if k != "sha256"}
        if sha256_of(body) != rec.get("sha256"):
            problems.append(f"{row['file']}: the contents do not match the hash written in the file (edited)")
        if rec.get("sha256") != row["sha256"]:
            problems.append(f"{row['file']}: hash differs from the one in {INDEX}")
        if rec.get("prev_sha256") != prev or row["prev_sha256"] != prev:
            problems.append(f"{row['file']}: does not follow the previous forecast (a forecast was removed or reordered)")
        prev = row["sha256"]
    listed = set(idx["file"]) if len(idx) else set()
    for path in sorted(folder.glob("*.json")):
        if path.name not in listed:
            problems.append(f"{path.name}: not listed in {INDEX}")
    return problems


# ------------------------------------------------------------------ make one

def make_forecast(con, race_id: int | None = None, sims: int = LIVE_SIMS, history_sims: int = HISTORY_SIMS,
                  now: datetime | None = None, folder: Path | None = None, cal_dir: Path | None = None,
                  truths: list | None = None, progress: bool = True, min_history: int | None = None) -> Path:
    now = _utc(now or datetime.now(timezone.utc)).replace(microsecond=0)
    folder = Path(folder) if folder else config.forecasts_dir()
    if race_id is None:
        nxt = next_race(con, now)
        if nxt is None:
            raise ForecastError("No upcoming race without results in the database.")
        race_id = int(nxt["race_id"])
    info = check_open(con, race_id, now)
    problems = verify(folder)
    if problems:
        raise ForecastError("The forecast log does not verify, so nothing is added to it:\n  " + "\n  ".join(problems))
    par = load_params(cal_dir)
    if truths is None:
        if progress:
            print("Rebuilding the race histories from the database (a few minutes) ...", flush=True)
        truths = C.build_inputs(con)["truths"]
    hist = sorted([t for t in truths if t.race_id < race_id], key=lambda t: t.race_id)
    need = F.MIN_HISTORY if min_history is None else min_history
    if len(hist) < need:
        raise ForecastError(f"Only {len(hist)} earlier races with usable laps; {need} needed.")
    card = card_for(con, info, hist)
    if progress:
        print(f"Learning the blend and recalibration from {len(hist)} earlier races ...", flush=True)
    state: dict = {}
    F.walk_forward(hist, par, S=history_sims, progress=progress, tyre_mode="proxy", state=state,
                       min_history=need)
    if progress:
        print(f"Forecasting {info['race_name']} ({info['season']} round {info['round']}) ...", flush=True)
    pool = Pool.before(hist, card)
    sim = F.forecast_card(hist, card, par, state["records"], S=sims, seed=race_id, pool=pool, tyre=F.proxy_tyre(hist, card.circuit_id))
    probs = model_probabilities(sim, card, pool, state)
    record = build_record(info, card, probs, hist, now, sims, history_sims, code_fingerprint(cal_dir), last_sha(folder))
    return write_forecast(record, folder)


# ------------------------------------------------------------------ scoring

def _results(con, race_id: int) -> dict[str, int]:
    rows = con.execute("SELECT driver_id, position FROM results WHERE race_id = ? AND position IS NOT NULL", [race_id]).fetchall()
    return {d: int(p) for d, p in rows}


def _wet(con, race_id: int):
    has = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'race_summary'").fetchone()[0]
    if not has:
        return None
    row = con.execute("SELECT wet FROM race_summary WHERE race_id = ?", [race_id]).fetchone()
    return None if row is None or row[0] is None else bool(row[0])


def scored_forecasts(con, folder: Path) -> list[dict]:
    """For each race with results, the last forecast made before the start."""
    chosen: dict[int, dict] = {}
    for rec in load_forecasts(folder):
        if parse_utc(rec["created_utc"]) >= parse_utc(rec["race_start_utc"]):
            continue                                              # made after the start: never scored
        chosen[rec["race_id"]] = rec                              # later ones replace earlier ones
    return [r for r in chosen.values() if _results(con, r["race_id"])]


def score_one(rec: dict, finish: dict[str, int]) -> list[dict]:
    drivers = [d["driver_id"] for d in rec["drivers"]]
    pos = np.array([finish.get(d, 99) for d in drivers])
    winner = [d for d, p in zip(drivers, pos) if p == 1]
    rows = []
    for model in LOGGED:
        pw = np.array([d["models"][model]["p_win"] for d in rec["drivers"]])
        p3 = np.array([d["models"][model]["p_top3"] for d in rec["drivers"]])
        p10 = np.array([d["models"][model]["p_top10"] for d in rec["drivers"]])
        pw_winner = float(pw[pos == 1][0]) if winner else 0.0
        rows.append({"race_id": rec["race_id"], "season": rec["season"], "model": model, "created_utc": rec["created_utc"],
                     "winner": winner[0] if winner else None, "p_winner": pw_winner,
                     "winner_logscore": float(-math.log(max(pw_winner, 1e-3))),
                     "winner_called": bool(winner and int(np.argmax(pw)) == drivers.index(winner[0])),
                     "brier_top3": float(np.mean((p3 - (pos <= 3)) ** 2)), "brier_top10": float(np.mean((p10 - (pos <= 10)) ** 2)),
                     "cars": len(drivers), "cars_missing_from_results": int(sum(d not in finish for d in drivers))})
    return rows


def score(con, folder: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(folder) if folder else config.forecasts_dir()
    rows = []
    for rec in scored_forecasts(con, folder):
        wet = _wet(con, rec["race_id"])
        for r in score_one(rec, _results(con, rec["race_id"])):
            rows.append({**r, "wet": wet})
    races = pd.DataFrame(rows)
    if races.empty:
        return races, pd.DataFrame()
    dry = races[races["wet"] != True]                                  # noqa: E712  (unknown counts with dry, and is shown)
    summ = dry.groupby("model").agg(races=("race_id", "nunique"), winner_logscore=("winner_logscore", "mean"),
                                    winner_called=("winner_called", "mean"), brier_top3=("brier_top3", "mean"),
                                    brier_top10=("brier_top10", "mean")).reset_index()
    b = summ[summ["model"] == BASELINE].iloc[0]
    summ["podium_skill"] = 1 - summ["brier_top3"] / b["brier_top3"]
    summ["top10_skill"] = 1 - summ["brier_top10"] / b["brier_top10"]
    summ["winner_logscore_gain"] = b["winner_logscore"] - summ["winner_logscore"]
    summ["order"] = summ["model"].map({m: i for i, m in enumerate(LOGGED)})
    return races, summ.sort_values("order").drop(columns="order").reset_index(drop=True)


def write_scores(con, folder: Path | None = None) -> list[str]:
    folder = Path(folder) if folder else config.forecasts_dir()
    races, summ = score(con, folder)
    if races.empty:
        return ["No logged forecast has a result yet."]
    races.to_csv(folder / "scores.csv", index=False)
    summ.to_csv(folder / "scores_summary.csv", index=False)
    n_wet = int(races.loc[races["wet"] == True, "race_id"].nunique())               # noqa: E712
    n_unknown = int(races.loc[races["wet"].isna(), "race_id"].nunique())
    lines = [f"Scored {races['race_id'].nunique()} logged forecast(s); dry or unknown-weather races in the summary, {n_wet} wet race(s) apart.",
             f"({n_unknown} race(s) without lap data yet counted as dry.)" if n_unknown else ""]
    for _, r in summ.iterrows():
        lines.append(f"  {r['model']}: winner log score {r['winner_logscore']:.2f} (gain on the grid-only model "
                     f"{r['winner_logscore_gain']:+.2f}), winner called {r['winner_called']:.0%}, podium Brier {r['brier_top3']:.3f} "
                     f"(skill {r['podium_skill']:+.0%}), top-ten Brier {r['brier_top10']:.3f} (skill {r['top10_skill']:+.0%})")
    lines.append("Few races mean a lot of noise: the backtest needed about 95 races before the ranking of models was stable.")
    return [x for x in lines if x]


# ------------------------------------------------------------------ command line

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next")
    f = sub.add_parser("forecast")
    f.add_argument("--race", type=int, default=None, help="race id such as 202618; default the next race")
    f.add_argument("--sims", type=int, default=LIVE_SIMS)
    sub.add_parser("score")
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    folder = config.forecasts_dir()
    if args.cmd == "verify":
        problems = verify(folder)
        print("\n".join(problems) if problems else f"Forecast log verifies ({len(read_index(folder))} forecast(s)).")
        return 1 if problems else 0
    path = config.db_path()
    if not path.exists():
        print(f"No database at {path}. Run stage 1 first.")
        return 1
    con = duckdb.connect(str(path), read_only=True)
    try:
        if args.cmd == "next":
            n = next_race(con)
            if n is None:
                print("No upcoming race without results in the database.")
                return 0
            print(f"Next race: {n['race_name']} ({n['season']} round {n['round']}, race id {n['race_id']}), "
                  f"starts {n['start_utc']} UTC. Qualifying rows in the database: {n['qualifying_rows']}. "
                  + ("Ready to forecast." if n["ready"] else "Not ready: update the data after qualifying."))
            return 0
        if args.cmd == "score":
            print("\n".join(write_scores(con, folder)))
            print(f"\nTables written to {folder}")
            return 0
        try:
            out = make_forecast(con, args.race, sims=args.sims, folder=folder)
        except ForecastError as e:
            print(f"Not forecast: {e}")
            return 1
        print(f"\nLogged {out}\nNow commit and push the forecasts folder: this is what makes the date checkable.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
