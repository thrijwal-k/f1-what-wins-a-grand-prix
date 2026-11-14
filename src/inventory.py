"""Quick inventory: what does each source actually return, season by season?

Run this first (about 30 requests). It confirms the coverage dates the plan
assumes before the big download starts.
"""
from __future__ import annotations

from . import config
from .jolpica import Client

SEASONS = [1950, 1980, 1996, 2003, 2011, 2018, 2024, 2025]


def total(client: Client, path: str) -> int:
    return int(client.get(path, {"limit": 1})["MRData"]["total"])


def main() -> None:
    client = Client()
    print("Jolpica-F1 row counts (0 means the source has nothing for that part)\n")
    print(f"{'season':>6} {'races':>6} {'results':>8} {'qualifying':>11} {'pit stops (rd 1)':>17}")
    for season in SEASONS:
        races = total(client, f"{season}.json")
        results = total(client, f"{season}/results.json")
        quali = total(client, f"{season}/qualifying.json")
        pits = total(client, f"{season}/1/pitstops.json")
        print(f"{season:>6} {races:>6} {results:>8} {quali:>11} {pits:>17}")

    print("\nFastF1 race calendars")
    try:
        import fastf1
        for year in (config.FASTF1_FIRST_SEASON, config.current_season()):
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            print(f"{year}: {len(schedule)} rounds on the calendar")
    except Exception as exc:  # noqa: BLE001
        print(f"FastF1 check failed: {exc}")


if __name__ == "__main__":
    main()
