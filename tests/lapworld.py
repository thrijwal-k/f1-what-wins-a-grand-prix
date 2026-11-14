"""A synthetic season of races with known tyre wear, fuel burn, pit loss, safety cars, retirements and rain.

Lap times come from: driver pace + fuel trend + compound offset + compound wear x tyre age + noise.
Positions are the order of cumulative race time, so lap-by-lap positions are consistent with the lap times.
"""
import duckdb
import numpy as np
import pandas as pd

FUEL = -0.05
WEAR = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.03}
OFFSET = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.5}
IN_EXTRA, OUT_EXTRA = 5.0, 15.0     # true pit loss is about IN_EXTRA + OUT_EXTRA = 20 s (plus tyre effects)
CIRCUITS = ["alpha", "beta", "gamma"]


def world(seed=1, seasons=(2018, 2019, 2020), races=6, n_drivers=14, n_laps=45, sc_races=(3, 4), wet_race=6,
          retire_prob=0.1, confound=False):
    """confound=True adds what makes real tyre data hard: softs are used early and hards late, the track
    improves non-linearly, and cars at the back of the grid lose time in traffic early on."""
    rng = np.random.default_rng(seed)
    laps_rows, res_rows, race_rows, weather_rows, practice_rows = [], [], [], [], []
    prng = np.random.default_rng(seed + 991)          # practice has its own stream so the races above do not change
    for season in seasons:
        for rnd in range(1, races + 1):
            race_id = season * 100 + rnd
            circuit = CIRCUITS[rnd % len(CIRCUITS)]
            race_rows.append({"race_id": race_id, "season": season, "round": rnd, "circuit_id": circuit})
            wet = rnd == wet_race
            sc = rnd in sc_races
            pace = rng.normal(0, 0.5, n_drivers)
            grid_order = np.argsort(pace + rng.normal(0, 0.35, n_drivers))   # grid mostly follows pace
            grid = np.empty(n_drivers, int)
            grid[grid_order] = np.arange(1, n_drivers + 1)
            cum = {d: 0.25 * grid[d] for d in range(n_drivers)}
            retire_at = {d: (int(rng.integers(5, n_laps - 4)) if rng.random() < retire_prob else None)
                         for d in range(n_drivers)}
            per_lap = {}
            for d in range(n_drivers):
                n_stops = int(rng.integers(1, 3))
                if confound:
                    first = int(rng.integers(8, 18))
                    stop_laps = [first] if n_stops == 1 else [first, int(rng.integers(26, n_laps - 8))]
                    comps = (["SOFT", "HARD"] if rng.random() < 0.5 else ["SOFT", "MEDIUM"]) if n_stops == 1 \
                        else ["SOFT", "MEDIUM", "HARD"]
                else:
                    stop_laps = sorted(rng.choice(np.arange(12, n_laps - 8), size=n_stops, replace=False))
                    comps = list(rng.choice(["SOFT", "MEDIUM", "HARD"], size=n_stops + 1))
                    if len(set(comps)) == 1:
                        comps[-1] = "HARD" if comps[0] != "HARD" else "MEDIUM"
                if wet:
                    comps = ["INTERMEDIATE"] * (n_stops + 1)
                stint, age, t_cum = 1, 0, cum[d]
                for lap in range(1, n_laps + 1):
                    if retire_at[d] is not None and lap > retire_at[d]:
                        break
                    age += 1
                    c = comps[stint - 1]
                    base = 90.0 + pace[d] + FUEL * lap + OFFSET.get(c, 0) + WEAR.get(c, 0) * age
                    base += rng.normal(0, 0.08)
                    if confound:
                        base += -0.5 * (1 - np.exp(-lap / 12)) + 0.8 * np.exp(-lap / 10) * grid[d] / n_drivers
                    pit_in = lap in stop_laps
                    pit_out = (lap - 1) in stop_laps
                    status = "1"
                    if sc and 20 <= lap <= 22:
                        base += 25.0
                        status = "4"
                    if pit_in:
                        base += IN_EXTRA
                    if pit_out:
                        base += OUT_EXTRA
                    t_cum += base
                    row = {"race_id": race_id, "abbreviation": f"D{d}", "driver_id": f"d{d}", "lap_number": lap,
                           "lap_time_s": base, "stint": stint, "compound": c, "tyre_life": float(age),
                           "fresh_tyre": True, "pit_in_s": t_cum if pit_in else np.nan,
                           "pit_out_s": t_cum - base + OUT_EXTRA if pit_out else np.nan, "track_status": status,
                           "position": np.nan, "is_accurate": not (pit_in or pit_out or status != "1"),
                           "lap_end_s": t_cum}
                    per_lap.setdefault(lap, []).append((d, t_cum, row))
                    if pit_in:
                        stint += 1
                        age = 0
                cum[d] = t_cum
            for lap, items in per_lap.items():
                for rank, (d, _, row) in enumerate(sorted(items, key=lambda x: x[1]), start=1):
                    row["position"] = rank
                    laps_rows.append(row)
            done = {d: max(l for l, its in per_lap.items() if any(i[0] == d for i in its)) for d in range(n_drivers)}
            order = sorted(range(n_drivers), key=lambda d: (-done[d], cum[d]))
            for pos, d in enumerate(order, start=1):
                res_rows.append({"race_id": race_id, "season": season, "round": rnd, "driver_id": f"d{d}",
                                 "constructor_id": f"c{d // 2}", "grid": int(grid[d]), "position": pos,
                                 "classified": done[d] >= 0.9 * n_laps})
            for session in ("FP2", "FP3"):
                for d in range(n_drivers):
                    clock = float(prng.uniform(0, 3000))              # cars go out at different times
                    fuel_load = prng.normal(0, 0.15)                 # teams run different fuel loads
                    for stint_no, comp in enumerate(["MEDIUM", "HARD"], start=1):
                        for age in range(1, int(prng.integers(6, 10)) + 1):
                            t = 90.0 + pace[d] + fuel_load + OFFSET[comp] + WEAR[comp] * age + prng.normal(0, 0.12)
                            clock += t
                            practice_rows.append({"race_id": race_id, "session": session, "abbreviation": f"D{d}",
                                                  "driver_id": f"d{d}", "lap_number": len(practice_rows) + 1, "lap_time_s": t,
                                                  "stint": stint_no, "compound": comp, "tyre_life": float(age), "fresh_tyre": True,
                                                  "pit_in_s": np.nan, "pit_out_s": np.nan, "track_status": "1",
                                                  "is_accurate": True, "session_time_s": clock})
            for k in range(6):
                weather_rows.append({"race_id": race_id, "time_s": 600.0 * k, "rainfall": bool(wet)})
    circuits = pd.DataFrame({"circuit_id": CIRCUITS, "circuit_name": [c.title() for c in CIRCUITS]})
    return {"laps": pd.DataFrame(laps_rows), "results": pd.DataFrame(res_rows), "races": pd.DataFrame(race_rows),
            "circuits": circuits, "weather": pd.DataFrame(weather_rows),
            "practice_laps": pd.DataFrame(practice_rows)}


def build_con(path, w):
    con = duckdb.connect(str(path))
    for name, df in w.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    return con
