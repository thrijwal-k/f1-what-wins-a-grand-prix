"""A synthetic championship with known causes, for what_decides and strategy.

Race performance of a car = car effect + driver skill + noise + STRATEGY_EFFECT x (extra stop) + FREE_BONUS x (stop under a
safety car); mechanical and accident retirements put a car at the back. The grid is car + driver + its own noise. Every
table has the shape of the real one (results, qualifying, pit_stops, laps).
"""
import numpy as np
import pandas as pd


def world(seed=1, seasons=(2018, 2019, 2020, 2021), races=12, teams=8, pool=24, n_laps=40, sd_car=1.0, sd_driver=0.4,
          sd_noise=0.6, p_mech=0.08, p_inc=0.04, strategy_effect=0.0, free_bonus=0.0, sc_prob=0.4, with_pits=True,
          with_laps=True):
    rng = np.random.default_rng(seed)
    skill = {f"d{i}": rng.normal(0, sd_driver) for i in range(pool)}
    res, qual, pits, laps = [], [], [], []
    for season in seasons:
        car = rng.normal(0, sd_car, teams)
        active = list(rng.choice(list(skill), size=2 * teams, replace=False))
        for rnd in range(1, races + 1):
            race_id = season * 100 + rnd
            sc_lap = int(rng.integers(10, 30)) if rng.random() < sc_prob else None
            neutral = set(range(sc_lap, sc_lap + 4)) if sc_lap else set()
            recs = []
            for j, d in enumerate(active):
                t = j // 2
                stops = int(rng.integers(1, 3))
                stop_laps = sorted(rng.choice(np.arange(8, n_laps - 6), size=stops, replace=False).tolist())
                free = sum(1 for l in stop_laps if l in neutral)
                perf = car[t] + skill[d] + rng.normal(0, sd_noise) + strategy_effect * (stops - 1.5) + free_bonus * free
                grid_perf = car[t] + skill[d] + rng.normal(0, 0.5)
                u = rng.random()
                status = "Finished"
                done = n_laps
                if u < p_mech:
                    status, done = str(rng.choice(["Engine", "Gearbox", "Hydraulics"])), int(rng.integers(2, n_laps - 1))
                elif u < p_mech + p_inc:
                    status, done = str(rng.choice(["Collision", "Accident", "Spun off"])), int(rng.integers(2, n_laps - 1))
                recs.append(dict(driver=d, constructor=f"c{t}", perf=perf, grid_perf=grid_perf, status=status, done=done,
                                 stops=stops, stop_laps=stop_laps))
            df = pd.DataFrame(recs)
            df["rank_key"] = df["perf"] - 100 * (df["done"] < n_laps) + 0.001 * df["done"]
            df["position"] = df["rank_key"].rank(ascending=False, method="first").astype(int)
            df["grid"] = df["grid_perf"].rank(ascending=False, method="first").astype(int)
            for _, r in df.iterrows():
                lapped = r["status"] == "Finished" and r["position"] > len(df) - 2
                res.append(dict(race_id=race_id, season=season, constructor_id=r["constructor"], driver_id=r["driver"],
                                grid=r["grid"], position=r["position"],
                                laps=r["done"] - (1 if lapped else 0),
                                status=("+1 Lap" if lapped else r["status"])))
                qual.append(dict(race_id=race_id, driver_id=r["driver"], position=r["grid"]))
                if with_pits:
                    for k, l in enumerate(r["stop_laps"], start=1):
                        if l <= r["done"]:
                            pits.append(dict(race_id=race_id, driver_id=r["driver"], stop=k, lap=l,
                                             duration_s=float(rng.normal(22, 1))))
                if with_laps:
                    for lap in range(1, int(r["done"]) + 1):
                        laps.append(dict(race_id=race_id, lap_number=lap, track_status="4" if lap in neutral else "1"))
    return (pd.DataFrame(res), pd.DataFrame(qual), pd.DataFrame(pits, columns=["race_id", "driver_id", "stop", "lap", "duration_s"]),
            pd.DataFrame(laps, columns=["race_id", "lap_number", "track_status"]))
