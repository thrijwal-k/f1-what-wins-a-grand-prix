"""Stage 3: a lap-by-lap Monte Carlo race simulator.

Many races at once (the first array axis is the simulation), one lap at a time.

What happens each lap, for every car still running
  free lap time = pace + fuel/track trend x lap + compound offset + wear x tyre age (+ a small curve)
                  + a stint effect + lap noise (+ the pit-stop time loss on a stop lap)
  On a neutralised lap every car runs the same lap time. Under a safety car the field also closes up.
  A car cannot finish a lap within `min_gap` seconds of the car it was following unless it passes. It passes
  with a probability that rises with how much faster it is than that car on the lap (centre `thr`).
  Optionally each car's pace also wanders (`drift_sd`, `drift_rho`: a mean-reverting drift that lasts several laps, so a good or
  bad spell is not forgotten at once) and each simulated race gets its own passing difficulty (`thr_sd`).
  Optionally a car that starts a lap close behind another loses time in its dirty air (`da_loss`, `da_range`), which both widens
  the gaps cars settle at and makes a pass need a bigger pace advantage.
  Lap 1 has no such rule: the start is the grid spacing plus noise (`sigma_start`, smaller at the front: `start_front`) and each car's pace.
  A new set is slow while it warms up (`warm_amp`, `warm_decay`, from the second lap on it: the out-lap is the stop's cost), and any car can
  make a bad start (`bad_start_p`, `bad_start_loss`): most starts are clean and a few lose a lot, which Gaussian noise cannot give.
  Part of a green-lap stop's cost can be paid on the out-lap (`outlap_share`): the total is the same, but the car on fresh tyres is not
  seen as fast on the lap it rejoins, which is when it would pass the cars it comes out behind.
  Retired cars stop. The finishing order is laps completed, then time.

Times are relative: only differences between cars matter, so a lap is measured as seconds over a common lap.
Known simplifications (checked against real races in replay.py): no lapping or blue flags, no tyre-saving
or fuel-saving, no weather, no penalties, pit-lane queues ignored, and drivers do not react to each other
except through safety-car stops.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SOFT, MEDIUM, HARD = 0, 1, 2
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
BIG = 10_000          # "never": a stop lap or a retirement lap that does not occur


@dataclass
class Params:
    thr: float = 0.6            # s/lap faster than the car ahead for a 50% chance to pass
    thr_scale: float = 0.25     # how sharply the pass probability rises around thr
    min_gap: float = 0.6        # closest a following car gets without passing (s)
    sc_gap: float = 1.0         # spacing the field closes up to under a safety car (s)
    sc_close: float = 0.8       # share of the excess gap removed per safety-car lap
    slot_gap: float = 0.25      # time per grid slot at the start (s)
    sigma_start: float = 0.6    # lap-1 noise (s), used in place of the lap noise on lap 1
    sigma_lap: float = 0.35     # lap-to-lap noise (s)
    sigma_stint: float = 0.15   # a constant effect for each stint (s)
    sc_pit_factor: float = 0.55  # share of the normal pit loss paid under a safety car
    vsc_pit_factor: float = 0.7
    sc_pit_prob: float = 0.6    # chance a car whose stop is near comes in on the first safety-car lap
    vsc_pit_prob: float = 0.2
    sc_pit_window: int = 12     # a planned stop within this many laps can be brought forward
    sc_min_age: int = 4         # ... if the tyres are at least this old
    sc_age_factor: float = 0.5  # tyres wear this much per neutralised lap
    depth: int = 2              # how many cars ahead can hold a car up on a lap
    noise_top: float = 1.0      # lap-noise multipliers by running position (top 3, 4-10, the rest), measured from real laps
    noise_mid: float = 1.0
    noise_rest: float = 1.0
    cover_prob: float = 0.0     # chance the car directly ahead of a car that stops on a green lap stops on the next lap (measured)
    sigma_pit: float = 0.0      # spread of what a stop costs, beyond lap noise (s); measured from real stops
    sigma_pit_top: float = -1.0  # the same for cars running in the top three when they stop; negative = same as sigma_pit
    drift_sd: float = 0.0       # persistent pace drift: stationary sd (s) of a per-car pace wander that lasts several laps (measured)
    drift_rho: float = 0.9      # share of the drift that carries over from one lap to the next (measured)
    thr_sd: float = 0.0         # race-to-race spread (log scale) of how hard passing is: each simulated race gets thr x exp(thr_sd x z)
    da_loss: float = 0.0        # dirty air: seconds a car loses on a lap when it started the lap right behind another car (0 = none)
    da_range: float = 2.0       # ... the loss falls in a straight line to nothing at this gap (s) to the car ahead
    start_front: float = 1.0    # lap-1 noise of the pole sitter as a share of `sigma_start`; it rises in a straight line to 1 at slot 1 + start_ramp
    start_ramp: float = 10.0    # ... slots over which it rises (1.0 for start_front = no difference between slots)
    outlap_share: float = 0.0   # share of a green-lap stop's time loss paid on the out-lap instead of the in-lap (the total is unchanged)
    warm_amp: float = 0.0       # tyre warm-up: seconds lost on the first lap after the out-lap of a new set (a race's own value in RaceSpec.warm wins)
    warm_decay: float = 0.6     # ... the share of it that is still lost one lap later
    warm_scale: float = 1.0     # multiplies whichever warm-up is in force (a race's own value or warm_amp); 0 switches the term off
    bad_start_p: float = 0.0    # chance that a car makes a bad start (lap 1 only), independent of its slot
    bad_start_loss: float = 2.0  # ... and the mean of the extra time (s) it then loses (exponential)
    circuit_thr: dict = field(default_factory=dict)   # circuit_id -> passing threshold where it differs from `thr`

    def for_circuit(self, circuit_id: str) -> "Params":
        """The same settings with this circuit's passing threshold (the general one if the circuit has none)."""
        thr = self.circuit_thr.get(circuit_id)
        return self if thr is None else replace(self, thr=float(thr))


@dataclass
class RaceSpec:
    """One race. Arrays with a leading axis of 1 are shared by every simulation; a leading axis of S differs by sim."""
    n_laps: int
    slot: np.ndarray            # (n,) grid slot of each car, 1..n, unique
    pace: np.ndarray            # (n,) or (S, n) seconds per lap slower than the field's reference (lower = faster)
    comp: np.ndarray            # (S|1, n, K+1) compound code of each stint
    stops: np.ndarray           # (S|1, n, K) in-lap number of each stop, BIG if none
    start_age: np.ndarray       # (S|1, n, K+1) laps already on the tyres when the stint starts
    retire: np.ndarray          # (S|1, n) lap the car does not complete, BIG if it finishes
    neutral: np.ndarray         # (S|1, n_laps+1) 0 green, 1 VSC, 2 safety car; column = lap number, column 0 unused
    off: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.0, 0.4]))     # compound offset at age 0
    deg: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.06, 0.04]))   # wear per lap of tyre age
    deg2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fuel: float = -0.056        # trend per lap (fuel burn and track evolution)
    pit_loss: float = 22.0
    cover: bool = False         # True when the stops are sampled, so the cover rule applies; real stops already contain their covers
    warm: float | None = None   # this race's tyre warm-up (s, first lap after the out-lap); None = Params.warm_amp


@dataclass
class SimResult:
    finish: np.ndarray          # (S, n) finishing position of each car, 1 = winner
    done: np.ndarray            # (S, n) laps completed
    pos: np.ndarray | None      # (S, n, L) running position after each lap, 0 once retired
    pit: np.ndarray | None      # (S, n, L) True on a lap the car came in
    neutral: np.ndarray         # (S, L+1)
    gap: np.ndarray | None = None   # (S, n, L) seconds to the car directly ahead at the end of each lap, NaN for the leader and retired cars


def _bs(a: np.ndarray, S: int) -> np.ndarray:
    a = np.asarray(a)
    if a.shape[0] == S:
        return a
    if a.shape[0] != 1:
        raise ValueError(f"leading axis {a.shape[0]} does not match {S} simulations")
    return np.broadcast_to(a, (S,) + a.shape[1:])


def simulate(spec: RaceSpec, par: Params | None = None, S: int = 200, seed: int = 0, history: bool = True) -> SimResult:
    par = par or Params()
    rng = np.random.default_rng(seed)
    pace_in = np.asarray(spec.pace, float)
    n, L = pace_in.shape[-1], int(spec.n_laps)
    comp = _bs(spec.comp, S).astype(int)
    stops = _bs(spec.stops, S).astype(int)
    sage = _bs(spec.start_age, S).astype(float)
    ret = _bs(spec.retire, S).astype(int)
    neu = _bs(spec.neutral, S).astype(int)
    K = stops.shape[2]
    rows = np.arange(S)[:, None]
    cols = np.arange(n)[None, :]
    pace = pace_in[None, :] if pace_in.ndim == 1 else pace_in          # (n,) shared, or (S, n) different in every simulation
    slot = np.asarray(spec.slot, float)[None, :]
    off, deg, deg2 = (np.asarray(x, float) for x in (spec.off, spec.deg, spec.deg2))

    T = np.repeat(par.slot_gap * (slot - 1.0), S, axis=0)
    k = np.zeros((S, n), int)
    age = sage[:, :, 0].copy()
    alive = np.ones((S, n), bool)
    done = np.zeros((S, n), int)
    jit = rng.normal(0.0, par.sigma_stint, (S, n, K + 1))
    use_drift = par.drift_sd > 0
    drift = rng.normal(0.0, par.drift_sd, (S, n)) if use_drift else None
    rho = float(np.clip(par.drift_rho, 0.0, 0.999))
    thr_s = par.thr * np.exp(par.thr_sd * rng.normal(0.0, 1.0, S)) if par.thr_sd > 0 else np.full(S, par.thr)
    pos_hist = np.zeros((S, n, L), np.int16) if history else None
    pit_hist = np.zeros((S, n, L), bool) if history else None
    gap_hist = np.full((S, n, L), np.nan, np.float32) if history else None

    use_cover = bool(spec.cover) and par.cover_prob > 0
    warm_amp = (par.warm_amp if spec.warm is None or not np.isfinite(spec.warm) else float(spec.warm)) * par.warm_scale
    since = np.full((S, n), 1000.0)   # laps since the car's last in-lap (the out-lap is 1); a car that has not stopped is far from any warm-up
    cover_force = np.zeros((S, n), bool)
    carry = np.zeros((S, n))        # the part of last lap's stop cost that is paid on this (out-)lap
    for lap in range(1, L + 1):
        nz = neu[:, lap]
        nzc = nz[:, None]
        alive_lap = alive & (ret > lap)
        age = age + np.where(nzc > 0, par.sc_age_factor, 1.0)
        since = since + 1.0
        c = comp[rows, cols, k]
        jj = jit[rows, cols, k]
        sig = par.sigma_start if lap == 1 else par.sigma_lap
        if lap == 1 and par.start_front != 1.0:
            sig = sig * (par.start_front + (1.0 - par.start_front) * np.clip((slot - 1.0) / max(par.start_ramp, 1e-9), 0.0, 1.0))
        if lap > 1 and (par.noise_top, par.noise_mid, par.noise_rest) != (1.0, 1.0, 1.0):
            rank0 = np.argsort(np.argsort(np.where(alive, T, np.inf), axis=1, kind="stable"), axis=1, kind="stable")
            sig = sig * np.where(rank0 < 3, par.noise_top, np.where(rank0 < 10, par.noise_mid, par.noise_rest))
        free = (pace + spec.fuel * lap + off[c] + deg[c] * age + deg2[c] * age * age + jj
                + rng.normal(0.0, 1.0, (S, n)) * sig)
        if lap == 1 and par.bad_start_p > 0:
            free = free + (rng.random((S, n)) < par.bad_start_p) * rng.exponential(par.bad_start_loss, (S, n))
        if warm_amp > 0:
            free = free + warm_amp * np.power(par.warm_decay, np.maximum(since - 2.0, 0.0)) * (since >= 2.0)
        if use_drift:
            drift = rho * drift + np.sqrt(1.0 - rho * rho) * par.drift_sd * rng.normal(0.0, 1.0, (S, n))
            free = free + drift
        free = np.where(nzc > 0, 0.0, free)

        nxt = np.where(k >= K, BIG, stops[rows, cols, np.minimum(k, K - 1)])
        planned = nxt <= lap            # a stop whose lap has passed (tied stops) happens as soon as it can
        react = np.zeros((S, n), bool)
        if (nz > 0).any():
            prev = neu[:, lap - 1] if lap >= 1 else np.zeros(S, int)
            first = (nzc > 0) & (prev[:, None] == 0)
            prob = np.where(nzc == 2, par.sc_pit_prob, par.vsc_pit_prob)
            eligible = (k < K) & (nxt > lap) & (nxt - lap <= par.sc_pit_window) & (age >= par.sc_min_age)
            react = first & eligible & (rng.random((S, n)) < prob)
        if use_cover:
            planned = planned | (cover_force & (k < K) & (nzc == 0))
        pit = alive_lap & (planned | react)
        cover_force = np.zeros((S, n), bool)
        if use_cover:
            # the car directly ahead of a car that stops on a green lap may stop on the next lap to cover the undercut
            srow = np.arange(S)
            ordS = np.argsort(np.where(alive_lap, T, np.inf), axis=1, kind="stable")
            elig_y = (k < K) & (nxt > lap) & (nxt - lap <= par.sc_pit_window) & (age >= par.sc_min_age)
            for r in range(1, n):
                x, y = ordS[:, r], ordS[:, r - 1]
                trig = ((nz == 0) & pit[srow, x] & alive_lap[srow, y] & ~pit[srow, y] & elig_y[srow, y]
                        & (rng.random(S) < par.cover_prob))
                cover_force[srow[trig], y[trig]] = True
        if par.da_loss > 0 and lap > 1:
            # dirty air: a car that starts the lap close behind another is slowed, more the closer it is (not on a stop lap or
            # under a neutralisation). The gap is the one at the end of the last lap, so the slowdown is part of the lap time that
            # the passing rule below compares.
            Ts0 = np.where(alive_lap, T, np.inf)
            o0 = np.argsort(Ts0, axis=1, kind="stable")
            t0 = np.take_along_axis(Ts0, o0, 1)
            with np.errstate(invalid="ignore"):
                g0 = np.diff(t0, axis=1)
            gp = np.full((S, n), np.inf)
            gp[:, 1:] = np.where(np.isfinite(g0), g0, np.inf)
            gprev = np.empty((S, n))
            np.put_along_axis(gprev, o0, gp, 1)
            pen = par.da_loss * np.clip(1.0 - gprev / par.da_range, 0.0, 1.0)
            free = free + np.where((nzc == 0) & ~pit, pen, 0.0)
        factor = np.where(nzc == 2, par.sc_pit_factor, np.where(nzc == 1, par.vsc_pit_factor, 1.0))
        stop_cost = pit * spec.pit_loss * factor
        free = free + stop_cost * (1.0 - par.outlap_share) + np.where(nzc == 0, carry, 0.0)
        if par.sigma_pit > 0 or par.sigma_pit_top > 0:
            sp = np.full((S, n), par.sigma_pit)
            if par.sigma_pit_top >= 0:
                rk = np.argsort(np.argsort(np.where(alive, T, np.inf), axis=1, kind="stable"), axis=1, kind="stable")
                sp = np.where(rk < 3, par.sigma_pit_top, sp)
            free = free + pit * factor * rng.normal(0.0, 1.0, (S, n)) * sp
        carry = stop_cost * par.outlap_share if par.outlap_share > 0 else carry * 0.0
        Tn = np.where(alive_lap, T + free, np.inf)

        if lap == 1:
            newT = Tn
        else:
            ordT = np.argsort(np.where(alive & (ret > lap), T, np.inf), axis=1, kind="stable")
            Tn_o = np.take_along_axis(Tn, ordT, 1)
            f_o = np.take_along_axis(free, ordT, 1)
            N_o = np.empty_like(Tn_o)
            no_pass = nz > 0
            for r in range(n):
                tn = Tn_o[:, r].copy()
                for m in range(1, par.depth + 1):
                    if r - m < 0:
                        break
                    tj = N_o[:, r - m]
                    d = f_o[:, r - m] - f_o[:, r]
                    p = 1.0 / (1.0 + np.exp(-(d - thr_s) / par.thr_scale))
                    win = (rng.random(S) < p) & ~no_pass
                    contest = np.isfinite(tn) & np.isfinite(tj) & (tn < tj + par.min_gap)
                    tn = np.where(contest & ~win, tj + par.min_gap, tn)
                N_o[:, r] = tn
            newT = np.empty_like(Tn)
            np.put_along_axis(newT, ordT, N_o, 1)

        sc = nz == 2
        if sc.any():
            ordN = np.argsort(newT, axis=1, kind="stable")
            Ns = np.take_along_axis(newT, ordN, 1)
            with np.errstate(invalid="ignore"):
                for r in range(1, n):
                    target = Ns[:, r - 1] + par.sc_gap
                    excess = Ns[:, r] - target
                    close = sc & np.isfinite(Ns[:, r]) & (excess > 0)
                    Ns[:, r] = np.where(close, Ns[:, r] - par.sc_close * excess, Ns[:, r])
            newT = np.empty_like(Ns)
            np.put_along_axis(newT, ordN, Ns, 1)

        T = np.where(alive_lap, newT, T)
        alive = alive_lap
        done = done + alive
        k_new = k + pit
        age = np.where(pit, sage[rows, cols, np.minimum(k_new, K)], age)
        since = np.where(pit, 0.0, since)
        k = k_new
        if history:
            rank = np.argsort(np.argsort(np.where(alive, T, np.inf), axis=1, kind="stable"), axis=1, kind="stable") + 1
            pos_hist[:, :, lap - 1] = np.where(alive, rank, 0)
            pit_hist[:, :, lap - 1] = pit
            if n > 1:
                Ts = np.where(alive, T, np.inf)
                o = np.argsort(Ts, axis=1, kind="stable")
                tso = np.take_along_axis(Ts, o, 1)
                with np.errstate(invalid="ignore"):
                    g = np.diff(tso, axis=1)
                g = np.where(np.isfinite(g), g, np.nan)
                gg = np.full((S, n), np.nan)
                gg[:, 1:] = g
                gl = np.full((S, n), np.nan)
                np.put_along_axis(gl, o, gg, 1)
                gap_hist[:, :, lap - 1] = np.where(alive, gl, np.nan)

    key = -done.astype(float) * 1e7 + np.where(np.isfinite(T), T, 1e6)
    finish = np.argsort(np.argsort(key, axis=1, kind="stable"), axis=1, kind="stable") + 1
    return SimResult(finish=finish, done=done, pos=pos_hist, pit=pit_hist, neutral=neu, gap=gap_hist)


# ------------------------------------------------------------------ helpers

def make_spec(n_laps: int, slot, pace, strategies, *, retire=None, neutral=None, **kw) -> RaceSpec:
    """Build a RaceSpec from simple python objects. strategies[i] = list of (compound_code, start_age) stints with the
    in-lap of each stop between them: [(MEDIUM, 0, 20), (HARD, 0)] means medium, stop on lap 20, then hard."""
    n = len(pace)
    K = max(len(s) for s in strategies) - 1
    K = max(K, 1)
    comp = np.full((1, n, K + 1), MEDIUM, int)
    stops = np.full((1, n, K), BIG, int)
    sage = np.zeros((1, n, K + 1))
    for i, s in enumerate(strategies):
        for j, stint in enumerate(s):
            comp[0, i, j] = stint[0]
            sage[0, i, j] = stint[1]
            if j < len(s) - 1:
                stops[0, i, j] = stint[2]
        for j in range(len(s), K + 1):
            comp[0, i, j] = s[-1][0]
    ret = np.full((1, n), BIG, int) if retire is None else np.asarray(retire, int).reshape(1, n)
    neu = np.zeros((1, n_laps + 1), int) if neutral is None else np.asarray(neutral, int).reshape(1, n_laps + 1)
    return RaceSpec(n_laps=n_laps, slot=np.asarray(slot), pace=np.asarray(pace, float), comp=comp, stops=stops,
                    start_age=sage, retire=ret, neutral=neu, **kw)


def spearman_masked(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rank correlation along the last axis among the entries where mask is True. NaN with fewer than 3."""
    def rk(a):
        big = np.where(mask, a, np.inf)
        return np.argsort(np.argsort(big, axis=-1, kind="stable"), axis=-1, kind="stable").astype(float)
    rx, ry = rk(x), rk(y)
    m = mask.sum(-1).astype(float)
    with np.errstate(all="ignore"):
        mx = (rx * mask).sum(-1) / m
        my = (ry * mask).sum(-1) / m
        cx, cy = (rx - mx[..., None]) * mask, (ry - my[..., None]) * mask
        r = (cx * cy).sum(-1) / np.sqrt((cx ** 2).sum(-1) * (cy ** 2).sum(-1))
    return np.where(m >= 3, r, np.nan)
