"""One command for everything: download, build, check.

    python -m src.run_all                      full update (first run takes hours)
    python -m src.run_all --from-season 2024   small trial run
    python -m src.run_all --inventory          what do the sources actually return?
    python -m src.run_all --skip-jolpica       fetch only what is missing from FastF1 (e.g. practice laps), rebuild, check
    python -m src.run_all --skip-jolpica --skip-fastf1   rebuild + check only
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import build_db, checks, config, derive, fastf1_loader, inventory, jolpica


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", action="store_true", help="show what each source returns, then stop")
    parser.add_argument("--from-season", type=int, default=config.FIRST_SEASON, help="first Jolpica season")
    parser.add_argument("--fastf1-from", type=int, default=config.FASTF1_FIRST_SEASON, help="first FastF1 season")
    parser.add_argument("--skip-jolpica", action="store_true")
    parser.add_argument("--skip-fastf1", action="store_true")
    parser.add_argument("--skip-practice", action="store_true", help="do not download practice-session laps")
    parser.add_argument("--skip-laptimes", action="store_true", help="do not download race line-crossing times")
    parser.add_argument("--practice-sessions", default="FP1,FP2,FP3",
                        help="which practice sessions to download, e.g. FP2 alone is about a third of the calls")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-derive", action="store_true", help="do not build race_summary, driver_race, team_race")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.inventory:
        inventory.main()
        return 0
    if not args.skip_jolpica:
        jolpica.download_all(from_season=args.from_season, force=args.force)
    if not args.skip_fastf1:
        fastf1_loader.download(from_season=args.fastf1_from, force=args.force)
    if not (args.skip_fastf1 or args.skip_practice):
        fastf1_loader.download_practice(from_season=args.fastf1_from, force=args.force,
                                        sessions=tuple(x.strip() for x in args.practice_sessions.split(",") if x.strip()))
    if not (args.skip_fastf1 or args.skip_laptimes):
        fastf1_loader.download_lap_times(from_season=args.fastf1_from, force=args.force)
    if not args.skip_build:
        build_db.build(log_fn=logging.getLogger("build").info)
        if not args.skip_derive:
            derive.build(log_fn=logging.getLogger("derive").info)
    if not args.skip_checks:
        return checks.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
