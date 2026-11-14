"""Stage 3: what the simulator invents when it is not told what happened.

A `Pool` is a set of past races (always earlier than the race being simulated, so nothing leaks from the future). From it:
  strategies   each simulation takes a past race's stops and compounds (a clean one, with few safety-car laps, from the same
               circuit when there are some, and counted for more the more recent its season) and gives each car the strategy of
               the car that started in the same place. A car whose template has no stop (it retired early) takes a neighbour's, and the
               stop laps get extra spread so that neighbours' first stops are as far apart as in the pool's real races
  safety cars  each simulation takes a past race's neutralised periods (start lap scaled to the race length, duration kept)
  retirements  each car retires with the pool's retirement rate for cars starting in its part of the grid (front, middle, back),
               on a lap drawn from the pool's retirement laps

Everything is drawn independently per simulation, so the spread across simulations is the uncertainty.
"""
from __future__ import annotations

import numpy as np

from .calibrate import RaceTruth, era_of
from .engine import BIG

K_MAX = 4
MIN_POOL = 12
CIRCUIT_WEIGHT = 3.0
CLEAN_NEUTRAL_LAPS = 3
RECENCY_TAU = 1.5              # seasons: a race this many seasons older than the target counts e^-1 as much
RET_BANDS = ((1, 3), (4, 10), (11, 99))     # grid slots the retirement rate is measured in
RET_PRIOR = 40.0               # car-races of the overall rate mixed into each band's rate


class Pool:
    def __init__(self, truths: list[RaceTruth], season: int | None = None):
        self.truths = list(truths)
        self.season = season            # season of the race being simulated, for the recency weights; None = no recency weighting
        self._jit: float | None = None

    @classmethod
    def before(cls, truths: list[RaceTruth], target: RaceTruth, min_n: int = MIN_POOL) -> "Pool":
        """Earlier races of the same era; if there are too few, all earlier races."""
        earlier = [t for t in truths if t.race_id < target.race_id]
        same = [t for t in earlier if era_of(t.season) == era_of(target.season)]
        return cls(same if len(same) >= min_n else earlier, season=target.season)

    def __len__(self) -> int:
        return len(self.truths)

    def _weights(self, items: list[RaceTruth], circuit: str) -> np.ndarray:
        w = np.array([CIRCUIT_WEIGHT if t.circuit_id == circuit else 1.0 for t in items])
        if self.season is not None:
            w = w * np.exp(-np.maximum(self.season - np.array([t.season for t in items], float), 0.0) / RECENCY_TAU)
        return w / w.sum()

    # ------------------------------------------------------------- pieces

    def retire_stats(self) -> tuple[float, np.ndarray]:
        n = sum(t.n for t in self.truths)
        out = sum(int((t.retire < BIG).sum()) for t in self.truths)
        fr = np.concatenate([t.retire[t.retire < BIG] / t.n_laps for t in self.truths] or [np.array([0.5])])
        return (out / n if n else 0.0), (fr if len(fr) else np.array([0.5]))

    def retire_rates(self) -> list[float]:
        """Retirement rate of the cars that started in each of RET_BANDS, each pulled toward the overall rate (RET_PRIOR car-races)."""
        overall, _ = self.retire_stats()
        out = []
        for lo, hi in RET_BANDS:
            n = k = 0
            for t in self.truths:
                m = (t.slot >= lo) & (t.slot <= hi)
                n += int(m.sum())
                k += int(((t.retire < BIG) & m).sum())
            out.append((k + RET_PRIOR * overall) / (n + RET_PRIOR))
        return out

    def stop_mix(self) -> float:
        n = sum(t.n for t in self.truths)
        return float(sum(int((t.stops < BIG).sum()) for t in self.truths) / n) if n else float("nan")


def _episodes(neutral: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of the same non-zero code: (start lap, length, code)."""
    eps, start, code = [], None, 0
    for lap in range(1, len(neutral)):
        c = int(neutral[lap])
        if c != code:
            if code:
                eps.append((start, lap - start, code))
            start, code = lap, c
    if code:
        eps.append((start, len(neutral) - start, code))
    return eps


def first_stop_gap(stops: np.ndarray, slot: np.ndarray, n_laps: int) -> float:
    """Mean distance (share of the race) between the first stops of cars that started in neighbouring grid slots, among pairs that both stop.
    stops is (n, K) of in-laps with BIG for none. NaN when no pair qualifies."""
    o = np.argsort(slot, kind="stable")
    first = np.where(stops[o, 0] < BIG, stops[o, 0] / n_laps, np.nan)
    d = np.abs(np.diff(first))
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else float("nan")


JITTER_MAX = 0.08


def _stop_jitter(pool: Pool, n_races: int = 15, S: int = 6) -> float:
    """The extra spread (sd, share of the race, per car) to give sampled stop laps so that neighbours' first stops are as far apart as in
    the pool's own races. Copying a whole past race by grid slot keeps neighbours' strategies too alike; this measures by how much
    (moment matching on the mean distance between neighbours' first stops) and does not look at the race being simulated."""
    if pool._jit is not None:
        return pool._jit
    ts = pool.truths[-n_races:]
    rng = np.random.default_rng(12345)
    real, samp = [], []
    for t in ts:
        g = first_stop_gap(t.stops, t.slot, t.n_laps)
        sm = sample_inputs(pool, t.n, t.n_laps, t.circuit_id, t.slot, S, rng, retirements=False, stop_jitter=0.0)
        gs = [first_stop_gap(sm["stops"][i], t.slot, t.n_laps) for i in range(S)]
        gs = [x for x in gs if np.isfinite(x)]
        if np.isfinite(g) and gs:
            real.append(g)
            samp.append(float(np.mean(gs)))
    if not real or np.mean(samp) <= 0 or np.mean(real) <= np.mean(samp):
        pool._jit = 0.0
        return 0.0
    sd_d = np.mean(samp) * np.sqrt(np.pi / 2.0)             # sd of the difference of two neighbours' first stops (half-normal relation)
    extra = (sd_d * np.mean(real) / np.mean(samp)) ** 2 - sd_d ** 2
    pool._jit = float(min(np.sqrt(max(extra, 0.0) / 2.0), JITTER_MAX))
    return pool._jit


def sample_inputs(pool: Pool, n_cars: int, n_laps: int, circuit: str, slot: np.ndarray, S: int, rng: np.random.Generator,
                  retirements: bool = True, stop_jitter: float | None = None) -> dict:
    """Arrays with a leading axis of S for a RaceSpec: comp, stops, start_age, retire, neutral. `stop_jitter` is the extra sd of stop laps
    (share of the race); None = the pool's own (see _stop_jitter), 0 = none."""
    ts = pool.truths
    if not ts:
        raise ValueError("empty pool")
    clean = [t for t in ts if int((t.neutral > 0).sum()) <= CLEAN_NEUTRAL_LAPS]
    clean = clean if len(clean) >= 3 else ts
    w_all, w_clean = pool._weights(ts, circuit), pool._weights(clean, circuit)
    p_ret, fr = pool.retire_stats()
    jit_sd = _stop_jitter(pool) if stop_jitter is None else float(stop_jitter)

    comp = np.ones((S, n_cars, K_MAX + 1), int)
    stops = np.full((S, n_cars, K_MAX), BIG, int)
    sage = np.zeros((S, n_cars, K_MAX + 1))
    neutral = np.zeros((S, n_laps + 1), int)
    retire = np.full((S, n_cars), BIG, int)

    slot_rank = np.argsort(np.argsort(slot, kind="stable"), kind="stable")            # 0..n-1 by grid position
    pick_s = rng.choice(len(clean), size=S, p=w_clean)
    pick_n = rng.choice(len(ts), size=S, p=w_all)
    for s in range(S):
        t = clean[pick_s[s]]
        order_t = np.argsort(t.slot, kind="stable")                                   # template car at each slot
        jitter = rng.integers(-1, 2, size=n_cars)
        j = np.clip(np.round((slot_rank + 0.5) * t.n / n_cars - 0.5).astype(int) + jitter, 0, t.n - 1)
        fin_j = np.flatnonzero(t.retire[order_t] >= BIG)             # a template car that retired never finished its plan: use the nearest finisher's
        if 0 < len(fin_j) < t.n:
            bad = t.retire[order_t[j]] < BIG
            if bad.any():
                j = np.where(bad, fin_j[np.argmin(np.abs(fin_j[None, :] - j[:, None]), axis=1)], j)
        src = order_t[j]
        k = t.comp.shape[1]
        kk = min(k, K_MAX + 1)
        comp[s, :, :kk] = t.comp[src, :kk]
        comp[s, :, kk:] = t.comp[src, kk - 1:kk]
        sage[s, :, :kk] = t.start_age[src, :kk]
        st = t.stops[src, :min(k - 1, K_MAX)]
        scaled = np.where(st < BIG, np.clip(np.round(st * n_laps / t.n_laps), 2, n_laps - 2), BIG)
        stops[s, :, :scaled.shape[1]] = scaled
        nostop = stops[s, :, 0] >= BIG                      # a template car that never stopped (it retired early): take the nearest stopper's strategy
        if nostop.any() and (~nostop).any():
            donors = np.flatnonzero(~nostop)
            for i in np.flatnonzero(nostop):
                d = donors[np.argmin(np.abs(slot_rank[donors] - slot_rank[i]))]
                comp[s, i], sage[s, i], stops[s, i] = comp[s, d], sage[s, d], stops[s, d]
        if jit_sd > 0:
            sh = np.round(rng.normal(0.0, jit_sd * n_laps, size=(n_cars, K_MAX))).astype(int)
            stops[s] = np.where(stops[s] < BIG, np.clip(stops[s] + sh, 2, n_laps - 2), BIG)
        # an episode-based copy of a past race's neutralised laps
        t2 = ts[pick_n[s]]
        for start, length, code in _episodes(t2.neutral):
            a = int(np.clip(round(start * n_laps / t2.n_laps), 2, n_laps))
            b = min(n_laps, a + length - 1)
            neutral[s, a:b + 1] = np.maximum(neutral[s, a:b + 1], code)
    for j in range(1, K_MAX):                              # strictly increasing stop laps after rescaling
        prev = stops[:, :, j - 1]
        stops[:, :, j] = np.where(stops[:, :, j] < BIG, np.minimum(np.maximum(stops[:, :, j], prev + 1), n_laps - 1), BIG)
        stops[:, :, j] = np.where(prev >= BIG, BIG, stops[:, :, j])
    if retirements:
        rates = pool.retire_rates()
        p_car = np.array([rates[next(b for b, (lo, hi) in enumerate(RET_BANDS) if lo <= r + 1 <= hi)] for r in slot_rank])
        out = rng.random((S, n_cars)) < p_car[None, :]
        laps = np.maximum(1, np.round(rng.choice(fr, size=(S, n_cars)) * n_laps).astype(int))
        retire = np.where(out, laps, BIG)
    return {"comp": comp, "stops": stops, "start_age": sage, "retire": retire, "neutral": neutral}
