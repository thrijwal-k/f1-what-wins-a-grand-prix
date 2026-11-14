# Stage 2: what the history says about what wins a Grand Prix

The question for the whole project is: **what actually wins a Grand Prix, the driver, the car, qualifying or strategy?** Stage 2 asks it of the historical record. This note summarises what the analysis found, how sure we are, and what it hands to stage 3 (the race strategy simulator). Every number below comes from a `findings.txt` or CSV written by the modules listed at the end, and every module has tests.

**Scope.** Results and qualifying for every season from 1950 to 2026 (Jolpica); lap, tyre and weather detail from 2018 (FastF1). Sections 1-6 analyse the Sunday Grand Prix; sprint results (2021 on, Jolpica) are analysed separately in section 7, and championship points are not used. The lap data is the race session only. Race counts differ slightly between analyses (190 races since 2018 for grid results in the latest data pull, one fewer where lap data is needed) because one race has no usable lap data.

## The short answer

1. **The car matters far more than the driver, and the driver's share has been small for thirty years.** In 2010-21 the car explains about 36% of the variation in race results and the driver about 3%; in qualifying it is 67% against 6%. Since 2022 the driver share looks larger (8% in races, 16% in qualifying), but that is the era where the method is least able to separate drivers from cars.
2. **Qualifying gives a real advantage that fades quickly.** The pole sitter has won 57% of races since 2022 (48% to 66%), up from 32% in the 1970s-80s. Most of that rise is reliability: the pole car now finishes 93% of the time, against about 67% in the 1970s. The grid advantage is mostly spent by the end of the first stint.
3. **Strategy decides who is in front when the order is reshuffled, not whether the order is reshuffled.** Winners usually made the same number of stops as the cars around them (86% of races), so stop count alone says little. But when the pole sitter does not win, the pit-stop cycle is the single biggest way the lead is lost (47% of those races), and a genuine undercut (the two cars stop one or two laps apart) is worth roughly 5 points of the chance of ending ahead, an effect of at most about 10 points whose interval only just clears zero (+0.2 to +9.9); the large headline figure comes from stops further apart, where it mostly reflects who was on the better strategy.
4. **Circuits barely change how often pole wins; they do change how tightly the grid predicts the finish.**
5. **A sprint strips out most of what happens between the grid and the flag, and the car's share rises when it does.** In sprints the car explains 41% of finishing position against 28% on Sundays since 2021, and the driver 7% in both; sprints keep a little more of their grid order than the Sunday race (a lean, not a clear result). Adding sprints to the car/driver model changes nothing.
6. **No single rule change shows a clear step**, but the test could only have seen large shifts. The long slow trends (reliability, a more decisive grid, fewer different winners) are the real story.

## 1. Qualifying and the grid

| Era | Pole won | Grid-vs-finish correlation | Pole car finishes | Wins if it finishes |
| --- | --- | --- | --- | --- |
| 1950-69 | 39% of 184 races (32% to 46%) | 0.38 | 68% | 60% |
| 1970-89 | 32% of 300 (27% to 38%) | 0.32 | 67% | 48% |
| 1990-2009 | 46% of 336 (41% to 52%) | 0.45 | 78% | 60% |
| 2010-21 | 51% of 237 (45% to 57%) | 0.62 | 92% | 56% |
| 2022-now | 57% of 108 (48% to 66%) | 0.63 | 93% | 62% |

- **Reliability explains most of the rise.** The pole win rate is the chance the pole car finishes times the chance it wins if it does. The second factor moves little between eras (48% to 62%); the first goes from about two thirds to over 90%. Among cars that finish, the grid-vs-finish correlation rose far less than for all starters: from 0.66-0.67 before 1990 to 0.74-0.77 since, against 0.32-0.40 to 0.62-0.63 when retirements are included.
- **Circuits.** Since 2000, across 26 circuits with at least eight races, circuit explains 6.4% of race-to-race variation in whether pole wins, about what chance alone would give (permutation p = 0.22). For how tightly grid order predicts the finish it explains 12.1% against 6.1% by chance (p = 0.001), so circuits do differ there. Monaco and Monza look different on raw counts (62% and 59%) but shrink to 53% each once small samples are accounted for.
- **Team dominance.** Across all 77 seasons, seasons with a dominant team also have more decisive grids (rank correlation +0.45 with pole win rate). Since 2010 that link with grid correlation disappears (-0.13).
- **Checks.** Leaving out the Indianapolis 500 (1950-60) changes the 1950-69 pole win rate by a point. Pole by grid and pole by qualifying differ in 1% to 5% of recent races and win at the same rate. In 2021-22 six sprint weekends set the Sunday grid by the sprint result; removing them moves the pole win rate since 2018 from 53.7% to 54.3% (see section 7).

## 2. Driver or car

Two complementary methods.

**Team-mate ratings.** Two drivers in the same team drive nearly the same car, so the gap between them is mostly the driver; drivers who change teams link the comparisons across careers. For 1994 onward, using qualifying lap-time gaps: typical noise is 1.17% of a lap, the spread of true driver skill is 0.60% of a lap, and the top-rated drivers are Verstappen (+0.95%, interval +0.74 to +1.17), Russell (+0.91%), Norris (+0.90%), Michael Schumacher (+0.83%) and Rosberg (+0.74%). Intervals overlap heavily within the top ten, so this is a tier list, not a ranking.

**Variance shares.** A mixed model splits the variation in each result into the car (the constructor, shared by its drivers), the driver, and everything else (luck, reliability on the day, strategy, incidents).

| Era | Qualifying: car / driver / other | Race: car / driver / other |
| --- | --- | --- |
| 1970-89 | 49% / 24% / 26% | 12% / 4% / 84% |
| 1990-2009 | 64% / 10% / 26% | 20% / 3% / 77% |
| 2010-21 | 67% / 6% / 27% | 36% / 3% / 61% |
| 2022-now | 40% / 16% / 44% | 27% / 8% / 66% |

- The car's share rises as results become more reliable and less random, and the "other" share of race results falls from 84% to 61%.
- Rolling ten-season windows show the driver's share of qualifying falling from about 30% in the 1960s-70s to a low of 4% (2010-2019), then rising to about 13-14% in the latest windows.

**Why to be careful.** These checks were run, and they set the limits of the claim.
- *Driver share is a lower bound.* Drivers are only separated from cars through team changes and team-mates.
- *Placebo.* Shuffling driver labels inside each constructor-season brings the driver share to about 0% in every era, so the model is not inventing driver skill.
- *Only two-car teams.* Dropping customer and privateer cars changes the shares by a point or two from 1990 on.
- *Rank against lap time.* The same drivers and races give different driver shares by lap-time gap than by rank: 14% against 10% (1990-2009), 3% against 6% (2010-21), 12% against 16% (2022-now). The car share is 63%, 78% and 38% by time. The rise in driver share since 2010-21 survives the change of yardstick (3% to 12%), but its size depends on it.
- *Recovery on the real design.* On invented outcomes with a known 20% driver share the model returns 20-22% in every era (2022-now: 21%); with a true 8% it returns 6-8% (2022-now: 6%, the largest miss). That check invents data from the model's own world; the stress tests in `car_driver_robust.py` use worlds it does not assume. The "stability ranges" in the output are spreads over subsets of seasons, not confidence intervals; race-resampled intervals and a stress-test allowance are in `car_vs_driver_robust/`.
- *Correction (October 2026).* An earlier version of this file said the model recovered only 16% of a true 20% in 2022-now with the car share at 59%. That described an older model; the current one does not show it.

## 3. Strategy and tyres

**Pit stops.** The time lost per green-flag stop (in-lap plus out-lap against two laps at the driver's own pace; dry races only, 3,617 stops, 27 circuits) has a median of 22.4 s across circuits, from 18.5 s at Spa to 28.1 s at Imola. Wet-race stops are kept out (527 stops, median 25.2 s against 22.5 s dry). Algarve rests on two races; `pit_loss_by_circuit.csv` includes a shrunk value (`loss_shrunk_s`) for exactly this. Circuit ranking is stable (split-half agreement 0.81). 148 stops (3.6%) exceed 35 s and probably include repairs or penalties; medians are robust to them. Pit loss drifts upward by about 0.1 s per season. Stops per car rose from 1.36 (2018) to 1.95 (2022) and have settled near 1.7. Winners made the same number of stops as the median of the top ten in 86% of 148 races (fewer 3%, more 10%).

**Tyre wear.** Wear per lap of tyre age: soft 0.050 s, medium 0.049 s, hard 0.036 s (medians over dry races). Wear rose from 2022 (medium +0.030 s per lap, p = 0.0003; hard +0.012, p = 0.014), but the medium figure is a spike in 2022 (0.070) that settles back to about 0.05, not a permanent step. Circuit-by-circuit wear is noisy: ranking circuits from two random halves of their races agrees at only 0.38 to 0.44. Part of that is the 2022 step (a circuit visited mostly after 2022 looks harsh); subtracting each season's median first lifts the agreement to 0.59 (soft), 0.49 (medium) and 0.51 (hard), better but still moderate. Wear also moves against the fuel and track-evolution trend (correlation about -0.2 to -0.4), and a circuit's wear in consecutive visits agrees only at 0.35 (medium) to 0.42 (hard). So the hardest and gentlest circuits in the raw table are mostly noise: the simulator should use `circuit_wear_shrunk.csv` (season-adjusted and pulled toward the overall mean in proportion to how little data a circuit has) and `wear_level_by_era.csv` for the era level.

**What did not work.** The compound pace gaps (soft against medium against hard) are not separated: a hard tyre is slower than a medium in only 57% of races, and the season estimates swing by several tenths of a second. They are flagged in the output and should not be used. The simulator calibrates compound pace against real stint lengths instead.

**How races are reshuffled.**
- *Grip decay.* The grid-vs-running-order correlation is about 0.9 after lap 1 and falls to roughly 0.75 by 30-40% of the distance in dry races. Whether it recovers late (the pit cycle scrambles the order temporarily, then it settles) is only established for 2018-21: the paired change from 40% to 95% of the distance is +0.07 (interval +0.05 to +0.10). For dry races since 2022 it is +0.02 (-0.00 to +0.04), indistinguishable from none, and for wet races -0.07 (-0.16 to +0.01), also not distinguishable from none. The difference between the two dry periods is small and not significant: +0.03 at lap 1 (p = 0.08), +0.04 at 40% (p = 0.09), -0.02 at 95% (p = 0.42).
- *How pole is lost.* Of the 87 races since 2018 where the pole sitter did not win: pit-stop cycle 47%, on-track pass 25%, safety car or red flag 10%, took the lead at the start 9%, leader retired 6%, unclear 2%. Between 2018-21 and 2022-now the pit cycle's share of pole losses went from 54% to 41%, on-track passes from 20% to 30%, and leader retirements from 10% to 2%, but none of these shifts, nor the mix as a whole, nor the pole win rate itself, is statistically distinguishable from chance (tests in `lead_takeover_shift_tests.csv`). Treat the 2018-now split as the best estimate and do not read a period story into it.
- *Undercut.* Pairs adjacent in the running order under green flags where the other car stops within ten laps. The pooled figure (the chaser ends ahead 28% of the time when it stops first against 15% when the leader does, +13 points) is not an undercut effect: it mostly comes from stops three or more laps apart (+21.5 points, interval +16.3 to +26.7), where the later stopper is on a different strategy. Where the stops are one or two laps apart, which is a true undercut, the advantage is +4.9 points (+0.2 to +9.9), borderline, and +6.4 (+0.4 to +12.6) since 2022 against +2.0 (-7.2 to +10.2) before, a change that is not significant. Comparing like with like (standardising on the stop gap) gives +15.0 (+11.2 to +18.7) overall, +8.8 (+3.5 to +14.0) in 2018-21 and +18.9 (+13.5 to +23.7) since 2022; the change between periods is +10.1 (+2.7 to +17.4, p = 0.008). Almost all of the rise since 2022 sits in stops three or more laps apart (+27 against +12), i.e. strategy differences rather than the undercut itself. Intervals resample whole races, because pairs from the same race are not independent (the earlier intervals were too narrow). Still observational: teams choose who stops first.

**Conditions.** Pole won 52% of dry races with no neutralisation (40 races), 56% with a safety car, VSC or red flag (123) and 44% in the wet (25; one race without lap data is excluded; some of these have under 100 wet-tyre laps and are reported separately as borderline in `conditions.csv`). The intervals overlap heavily (37% to 67%, 47% to 65%, 27% to 63%), so the data cannot separate them, although the grid correlation does differ (0.73, 0.60, 0.57).

## 4. Trends and rule changes

Six season-level measures from 1950: pole sitter wins, grid-vs-finish correlation, share of cars classified, the top team's share of wins, winners starting 4th or lower, and different winners per race.

- **Slow trends.** Cars classified rose from about 50% to about 87-90%; the winner starting 4th or lower fell from 34% (1970-89) to 11-12%; different winners per race fell from 0.47 to 0.24.
- **Rule changes.** For each of 1994, 2009-11 (slicks, new aero, refuelling ban, DRS, Pirelli), 2014 (hybrids), 2017 (wider cars) and 2022 (ground effect), the change from the three seasons before to the three after was compared with the same change at years when no rule changed, with a false-discovery adjustment across 30 comparisons. Nothing cleared the bar. The closest were the top team's share of wins in 2014 (+31 points, adjusted q = 0.38), which is Mercedes' run, and the reverse in 2017. A second check compares the after-period with the continuation of the 8-season pre-trend, so a measure already trending does not look like a jump. It has no multiple-testing adjustment, and 3 of 30 comparisons look unusual (about 1.5 would by luck): the pole win rate at 1994 (58% to 33%, z = -2.5; q = 0.64 in the main test), the share of cars classified after 2014 (z = -2.4, plausibly the new power units' early reliability) and the top team's share in 2017 (z = -2.1). 2014 top-team share is the only case that is also unusual in both other window lengths. None is established; they are the places to look first.
- **What the test could see.** A rule change had to move pole win rate by about 28 points, top-team share by about 25 points or the share of cars classified by about 10 points to stand out. Smaller effects are invisible, so this is "no evidence of a rule-change effect", not "no effect". The 2026 rules are shown but not tested while the season runs.

## 7. Sprints

Run `python -m src.analysis.sprints` (outputs/sprints) after `python -m src.run_all --skip-fastf1` has loaded the sprint results. 30 sprints, 2021 to 2026 (3, 3, 6, 6, 6, 6 a year). Sprint results are 2021 onward only, so every comparison here uses 2021-26.

**What a sprint looks like.** The sprint's own pole sitter won 63% of sprints (46% to 78%; 19 of 30), against 54% for Sunday poles since 2018. Grid-to-finish correlation averages 0.77 over all starters and 0.84 for finishers; 93% of cars are classified. By year the pole win rate bounces between 33% and 83% on three to six sprints each, which is noise at this size.

**Sprint against Sunday, same weekend (24 weekends with independent grids).** A sprint keeps more of its grid order than the Sunday race does:

| Measure | Sprint | Sunday | Difference (95%) | p | q |
| --- | --- | --- | --- | --- | --- |
| Grid vs finishing order, all starters | 0.77 | 0.63 | +0.14 (+0.01 to +0.26) | 0.022 | 0.064 |
| Grid vs finishing order, finishers only | 0.84 | 0.73 | +0.11 (+0.01 to +0.21) | 0.032 | 0.064 |
| vs Sunday running order at 30% of distance | 0.85 | 0.75 | +0.10 (+0.00 to +0.20) | 0.048 | 0.072 |
| vs Sunday running order at 40% of distance | 0.85 | 0.79 | +0.07 (-0.01 to +0.14) | 0.086 | 0.103 |
| Pole sitter wins | 0.67 | 0.46 | +0.21 (+0.00 to +0.42) | 0.104 | 0.104 |
| Share of cars classified | 0.92 | 0.85 | +0.07 (+0.02 to +0.12) | 0.014 | 0.064 |

All six differences point the same way, but with six measures compared none clears q < 0.05 (the measures overlap, so q is cautious; the unadjusted p-values for four of them are below 0.05). The honest reading is a lean, not a result: a sprint keeps a bit more order than a Grand Prix does at the same share of distance, which would fit stops and tyre strategy doing some of the scrambling on Sunday, but the gap shrinks as the Sunday comparison point moves from the end of the race to 30-40% of it, and 24 weekends cannot settle it. More cars finishing a short race is expected from distance alone.

**Does a sprint weekend change the Sunday race?** (30 sprint weekends; other Sundays at the same circuit and season; 12 to 14 circuits have both kinds.) Seven of eight measures show nothing: grid vs finishing order (+0.02, p = 0.67), grid vs running order at 40% (+0.01), cars classified, stops per car (+0.17, cannot detect below about 0.5), pit loss, medium-tyre wear, safety car share (+0.18, p = 0.06). One does: the pole sitter won 46% of Sundays on sprint weekends against 56% on others (raw difference -10 points); the circuit and season adjusted estimate is -0.33 (-0.61 to -0.04), p = 0.006, q = 0.048. Treat it as a lead to watch, not a finding: q is only just under 0.05 after eight tests; the adjusted effect is three times the raw gap and larger than the smallest effect these races could detect (0.40), the pattern of an estimate inflated by a small sample; and grid correlation, the more sensitive measure of the same thing, does not move. A possible reason (less practice before qualifying on a sprint weekend) is a guess this data cannot test.

**Sprints as extra data for car and driver (2021 on).** The split of variation in finishing position:

| Data | Car | Driver | Everything else |
| --- | --- | --- | --- |
| Sunday races (2,652 driver-races) | 28% (24-32) | 7% (6-11) | 65% |
| Sprints (612) | 41% (32-48) | 7% (5-8) | 52% |
| Both (3,264) | 29% (24-33) | 7% (6-11) | 63% |

In a sprint the car explains more (the intervals do not overlap) and the driver the same. The 13 points that move from "everything else" to the car are what a third of the distance, no required stop and fewer retirements take out: strategy, incidents and reliability together, not strategy alone. Adding the sprints to the Sunday data changes the split by about a point, so the part 3 result does not depend on leaving them out. For qualifying the picture is the same within wide ranges (Sunday car 43%, driver 16%; sprint grids car 42%, driver 13%); the sprint figure uses the sprint's own starting grid.

**Team-mates.** The team-mate ahead in the sprint is ahead on Sunday 58% of the time (52% to 63%, 306 cases); the team-mate ahead in one Sunday is ahead in the next 59% (56% to 62%, 943 cases). At season level the sprint gap and the Sunday gap correlate 0.46 across 62 pairs, against 0.48 for an equal number of other Sundays (difference -0.01, -0.32 to +0.27). A sprint says as much about two team-mates as another Sunday does, and a single race says little: a team-mate order barely repeats more than half the time, in line with the small driver share.

**Limits.** Few sprint weekends, so wide intervals; "not significant" means no evidence at this sample size, not no difference. Which circuits get a sprint is a choice, so the weekend comparison is observational. The 2026 season is partial and the newest 2026 round has no lap data (the run skipped FastF1), so lap-based measures skip it.

**The six 2021-22 sprint-grid races** are now taken from the sprint results table and match the hand-written list exactly. Their grid is the sprint result, with penalties applied (in Italy 2021 Verstappen was on pole after Bottas's penalty); the grid pole won 2 of 6 and the qualifying P1 driver 3 of 6, and the two were different drivers in four. Leaving them out moves the pole win rate since 2018 from 53.7% (190 races) to 54.3% (184), and the grid correlation from 0.623 to 0.626: they do not change a headline.

## What stage 3 takes from this

| Use in the simulator | Source |
| --- | --- |
| Time lost per stop, by circuit (use the shrunk column for thin circuits) | `pit_loss_by_circuit.csv` |
| Tyre wear per lap by compound: era level, plus a shrunk circuit offset | `wear_level_by_era.csv`, `circuit_wear_shrunk.csv` |
| Compound pace offsets: calibrate inside the simulator against real stint lengths, not the regression gaps | `tyre_checks.csv` |
| Fuel and track-evolution trend: about -0.056 s per lap | `tyre_by_race.csv` |
| Stop-count mix by season | `stops_per_car_by_season.csv` |
| Safety-car, VSC and red-flag frequency, and wet-race share | `race_flags.csv`, `status_by_season.csv` |
| Retirement rate (cars classified about 87-90% recently) | `season_measures.csv` |
| Validation targets: grip decay curve, route by which pole is lost, undercut rates, pole win rate | `grip_decay.csv`, `lead_takeover.csv`, `undercut.csv`, `conditions.csv` |

A simulator that cannot roughly reproduce the undercut advantage by stop gap (about +5 points for stops 1-2 laps apart, about +21 for 3+ apart), the grip-decay curve and the 47% pit-cycle share is not ready to be used for strategy advice. Use the gap-banded rows in `undercut.csv`, not the pooled rate.

## Review additions (October 2026)

After the full re-run was reviewed against its CSVs, these were fixed or added. The one-page result is `docs/stage2_summary.md`.

| Change | Why | Where |
|---|---|---|
| Pit loss uses dry races only; wet stops get their own table and a dry-vs-wet comparison | wet stops (527 of 4,189, median 25.2 s against 22.5 s) were in the circuit table; Istanbul and Hockenheim were wet-only | `tyres.py`, `pit_loss_wet_by_circuit.csv`, `pit_loss_dry_vs_wet.csv` |
| Tyre chart labels no longer overlap; subtitle says a compound is only seen where teams chose it | 2026 labels collided; selection effect was unstated | `tyres.py` |
| Lap-data coverage line and `races_without_laps.csv` | 2018 Italian GP has no lap data, so lap-based pole rates cover 189 of 190 races | `race_flow.py` |
| Spreads in grid places next to the shares (corrected after the re-run) | meant to separate 'cars bunched up' from 'drivers spread more'; it cannot, because ranks always spread evenly so the table is the shares in other units. Labelled as such; the lap-time table is the real-units check | `car_driver.py`, `*_sd_places` columns |
| Qualifying split by lap-time gap, shown under the chart | a place is worth very different time in different fields; rank gives 16% driver share for 2022-now and 6% for 2010-21, time gives 12% and 3% | `variance_shares_qualifying_by_laptime.csv` |
| Where winners come from, by qualifying | the variance split is about the whole field, not winners | `winners.py`, `outputs/who_wins/` |
| Race-pace team-mate ratings, 2018 on, against qualifying ratings | the ratings measured one lap in qualifying only | `race_pace.py`, `outputs/race_pace_ratings/` |

## Limits, in one place

- Observational history, not experiments. Team strength, driver quality and strategy are entangled; the analyses separate them statistically and say where that breaks down.
- The 2022-now era is the hardest to read (few team changes, a short window), and it is the era that matters most for a forecast.
- The compound pace gaps and circuit-by-circuit tyre wear are too noisy to use directly; use the shrunk circuit values.
- Period differences in how pole is lost, and the late-race grip recovery since 2022, are not statistically established.
- The undercut and "how pole is lost" results describe when and how the lead changed, not what caused it.
- The rule-change test has low power and cannot separate rules from team strength.
- Safety-car frequency is derived from lap status codes (a lap counts when a quarter of the cars carry the code); it looks plausible but has not been checked against an independent count, so check it before it drives the simulator.
- One race (202003, flagged wet on 56 wet-compound laps but only 1.5% rain share) may be misclassified. It is one of 25 wet races and does not change conclusions.

## Reproduce

```
python -m pytest -q                              # all tests
python -m src.analysis.grid                      # outputs/grid_vs_result
python -m src.analysis.grid_checks               # outputs/grid_vs_result_checks
python -m src.analysis.ratings                   # outputs/driver_vs_car
python -m src.analysis.car_driver                # outputs/car_vs_driver
python -m src.analysis.car_driver_checks         # outputs/car_vs_driver_checks
python -m src.analysis.tyres                     # outputs/tyres_and_pits
python -m src.analysis.race_flow                 # outputs/race_flow
python -m src.analysis.trends                    # outputs/trends_and_rules
python -m src.analysis.sprint_check              # outputs/sprint_check
python -m src.analysis.sprints                   # outputs/sprints
python -m src.analysis.winners                   # outputs/who_wins
python -m src.analysis.race_pace                 # outputs/race_pace_ratings
```
