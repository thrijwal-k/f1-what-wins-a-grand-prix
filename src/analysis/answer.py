"""Stage 2, deep dive 4: the unified answer, assembled from the other outputs so every number comes from one place.

Reads what the deep-dive modules wrote and puts the pieces side by side for the most recent periods:

  what_decides/decomposition_by_period.csv       shares of the race result: car, driver, qualifying form, strategy,
                                                  safety-car timing, mechanical failures, accidents, unexplained
  car_vs_driver_robust/driver_share_intervals.csv  the driver share with its interval and the stress-test allowance
  strategy_teammates/strategy_effects.csv         places per stop, per grid place, among team-mates
  who_wins/win_sources_by_era.csv                 how often the pole sitter, or the quickest team, wins

Missing inputs are skipped and said so. Nothing is recomputed here.

Run:  python -m src.analysis.answer     (after what_decides, strategy, car_driver_robust and winners)
Writes to outputs/stage2_answer/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FACTORS = [("Car (team and season)", "car"), ("Driver", "driver"), ("Qualifying form", "Qualifying form"),
           ("Strategy on the day", "Strategy"), ("Safety-car timing", "Safety-car timing"),
           ("Mechanical failures", "Mechanical retirements"), ("Accidents", "Accidents"),
           ("Retirements (cause not split)", "Retirements"), ("Unexplained", "unexplained")]
PERIODS_SHOWN = ["2018-21", "2022-now"]


def _read(root: Path, folder: str, name: str) -> pd.DataFrame | None:
    path = root / folder / name
    return pd.read_csv(path) if path.exists() else None


def _col(period_row: pd.Series, key: str, which: str):
    name = {"car": "car", "driver": "driver", "unexplained": "unexplained"}.get(key, key)
    suffix = {"share": "_share", "low": "_low", "high": "_high"}[which]
    return period_row.get(f"{name}{suffix}", np.nan)


def build_table(decomp: pd.DataFrame, intervals: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for period in PERIODS_SHOWN:
        sub = decomp[decomp["period"] == period]
        if sub.empty:
            continue
        r = sub.iloc[0]
        for label, key in FACTORS:
            rec = {"period": period, "factor": label, "share": _col(r, key, "share"), "low": _col(r, key, "low"),
                   "high": _col(r, key, "high"), "allowed_low": np.nan, "allowed_high": np.nan,
                   "data_low": np.nan, "data_high": np.nan}
            if key == "driver" and intervals is not None and period == "2022-now":
                iv = intervals[(intervals["era"] == period) & (intervals["outcome"] == "race")]
                if len(iv):
                    rec["allowed_low"], rec["allowed_high"] = iv["driver_allowed_low"].iloc[0], iv["driver_allowed_high"].iloc[0]
                    if "driver_data_low" in iv:
                        rec["data_low"], rec["data_high"] = iv["driver_data_low"].iloc[0], iv["driver_data_high"].iloc[0]
            if pd.notna(rec["share"]):          # a block that does not exist in this period leaves no row
                rows.append(rec)
    return pd.DataFrame(rows)


def _p(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def lines_from(decomp, intervals, strategy, winners) -> list[str]:
    if decomp is None or decomp.empty:
        return ["The decomposition has not been run (python -m src.analysis.what_decides)."]
    table = build_table(decomp, intervals)
    L = ["THE ANSWER SO FAR: what decides the finishing order of a Grand Prix (shares of the variation, ranges in brackets)"]
    for period in PERIODS_SHOWN:
        sub = table[table["period"] == period]
        if sub.empty:
            continue
        bits = []
        for _, r in sub.iterrows():
            if pd.isna(r["share"]):
                continue
            rng = f" ({_p(r['low'])}-{_p(r['high'])})" if pd.notna(r["low"]) else ""
            bits.append(f"{r['factor'].lower()} {_p(r['share'])}{rng}")
        L.append(f"  {period}: " + ", ".join(bits))
        allow = sub[sub["factor"] == "Driver"]
        if len(allow) and pd.notna(allow["allowed_low"].iloc[0]):
            data = (f"{_p(allow['data_low'].iloc[0])}-{_p(allow['data_high'].iloc[0])} with hidden effects the size the data show, "
                    if pd.notna(allow.get("data_low", pd.Series([np.nan])).iloc[0]) else "")
            L.append(f"    driver share allowing for a wrong model: {data}{_p(allow['allowed_low'].iloc[0])}-"
                     f"{_p(allow['allowed_high'].iloc[0])} with deliberately strong ones")
    if strategy is not None and len(strategy):
        s = strategy[(strategy["variant"] == "all pairs") & (strategy["period"] == "2011-now")].set_index("term")
        if {"stops", "grid"} <= set(s.index):
            L.append(f"STRATEGY AMONG TEAM-MATES (2011 on, {int(s['pairs'].iloc[0])} pairs): one more stop than the team-mate "
                     f"{s.loc['stops', 'estimate']:+.2f} places ({s.loc['stops', 'low']:+.2f} to {s.loc['stops', 'high']:+.2f}); "
                     f"one grid place {s.loc['grid', 'estimate']:+.2f} places ({s.loc['grid', 'low']:+.2f} to {s.loc['grid', 'high']:+.2f})")
    if winners is not None and len(winners):
        w = winners.iloc[-1]
        L.append(f"WINNERS ({w['era']}, {int(w['races'])} races): the pole sitter won {w['Pole sitter_share']:.0%}, the quickest team "
                 f"{w['quickest_team_share']:.0%}, a team further back than the second quickest {w['Further back_share']:.0%}")
    L.append("READ WITH CARE: the shares describe the whole field, not only the winner; the car share contains reliability and "
             "habitual strategy; blocks are associations, not proof of cause; every range is a guide, not a bound.")
    return L


def run(root: Path | None = None, out_dir: Path | None = None) -> list[str]:
    root = Path(root) if root else config.outputs_dir()
    out = Path(out_dir) if out_dir else root / "stage2_answer"
    out.mkdir(parents=True, exist_ok=True)
    decomp = _read(root, "what_decides", "decomposition_by_period.csv")
    intervals = _read(root, "car_vs_driver_robust", "driver_share_intervals.csv")
    strategy = _read(root, "strategy_teammates", "strategy_effects.csv")
    winners = _read(root, "who_wins", "win_sources_by_era.csv")
    if decomp is not None and len(decomp):
        build_table(decomp, intervals).to_csv(out / "answer_table.csv", index=False)
    lines = lines_from(decomp, intervals, strategy, winners)
    missing = [n for n, d in (("what_decides", decomp), ("car_vs_driver_robust", intervals), ("strategy_teammates", strategy),
                              ("who_wins", winners)) if d is None]
    if missing:
        lines.append("MISSING INPUTS (run these first): " + ", ".join(missing))
    (out / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    print("\n".join(run()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
