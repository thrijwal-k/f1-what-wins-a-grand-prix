# Stage 3: the race simulator

One lap-by-lap Monte Carlo simulator, used three ways. The question it serves is the project's: how much of a result is the car, the driver, the grid, the strategy, or luck?

## What is simulated

Many races at once (200-300 per real race), one lap at a time, for every car still running:

- **Lap time** = the car's pace + a trend per lap (fuel burn and track evolution, about -0.056 s) + compound offset + tyre wear x tyre age + a constant effect for each stint + lap noise. A stop adds the circuit's pit loss plus a random spread (`sigma_pit`, measured from real stops).
- **Passing.** A car cannot finish a lap within `min_gap` (0.6 s) of the car it was following unless it passes. It passes with a probability that rises with how much faster it is on that lap, centred on `thr`, which is fitted overall and set per circuit (below).
- **Start.** Grid spacing (`slot_gap`, fitted) plus lap-1 noise (`sigma_start`, fitted; it replaces the ordinary lap noise on lap 1). The passing rule is off on lap 1.
- **Neutralisations.** Under a safety car every car runs the same lap time and the field closes up; under a VSC the gaps freeze. Nobody passes. Cars with a stop due soon may come in on the first neutralised lap (rate measured from the data), paying a reduced pit loss (measured).
- **Retirements.** A car stops on its retirement lap. The finishing order is laps completed, then time.

Known simplifications: no lapping or blue flags, no tyre or fuel saving, no weather (wet races are left out), no penalties, no pit-lane queues, and drivers do not react to each other except through safety-car stops. The replay shows how much each costs.

## Where each setting comes from

| Setting | Source |
| --- | --- |
| Each car's pace | per-race regression of lap time on driver, lap number, compound and tyre age (stage 2's tyre fit, keeping the driver effects) |
| Tyre wear by compound | era median plus the shrunk circuit offset (stage 2) |
| Compound offsets, fuel/track trend | era medians |
| Pit loss | circuit value, shrunk towards the overall level (stage 2) |
| Safety-car stop rate and its cheaper stop | measured on the races (`calibrate.py`) |
| Passing threshold, lap-1 noise, grid spacing, share of lap noise kept | fitted on 2022-24 by coordinate search, against two measured things: the grip-decay curve and the number of on-track passes per car-lap (counted the same way on real and simulated races: swaps of order between two green laps among cars that are not stopping) |
| Passing threshold by circuit | each circuit's measured pass rate in 2018-24 races, matched by scaling the general threshold, shrunk toward the general value with weight races / (races + 2) |
| Stop-cost spread | robust spread of real stop losses within a race, less the simulator's own lap noise |
| Lap noise | the tyre fit's residual spread (times the fitted share) |

2025 on is held out from every fitted setting. 2018-21 is a different era and is reported separately. Nothing about how often pole wins, how pole is lost, the undercut or the grid-to-finish correlation by circuit is fitted: those stay as checks. If a setting ends at the edge of its search range, calibrate says so rather than reporting it as found.

## Three levels

| Level | What the simulator is told | What it answers |
| --- | --- | --- |
| L0 | everything: pace, strategies, safety-car laps, retirements | do traffic, passing and noise reproduce a result? |
| L1 | pace and grid; strategies, safety cars and retirements are sampled from earlier races | how much of a result does pace plus grid set, and how much does the simulator have to guess? |
| L2 (forecast) | only what was known before the race: earlier races, and the grid | can it beat knowing only the grid slot? |

Comparing L0, L1, L2 and the grid-only benchmark is how the stage answers "what wins a Grand Prix": the gap between the grid-only benchmark and L1 is what pace adds; the gap between L1 and L0 is what strategy, safety cars and retirements add; the gap between L0 and a perfect order is traffic, passing and noise.

## Validation targets (from stage 2)

A simulator that cannot roughly reproduce these is not ready for strategy advice:

1. The grip-decay curve (grid vs running order through the race).
2. How often pole wins, and how it is lost (start, pit-stop cycle, safety car, retirement, on-track pass).
3. The undercut advantage by gap between the stops (about +5 points for 1-2 laps, about +21 for 3+ laps).
4. Circuit differences in how tightly the grid predicts the finish.

`replay.py` computes each on simulated and real races with the same code.

## Ablation ladder

`ablation.py` runs every race with true pace and the real grid, and for each of three things (strategies, safety cars, retirements) either tells the simulator what happened or has it sample from earlier races: eight combinations from "pace only" (L1) to "everything known" (L0). Each factor's share of the error is a Shapley value, its average gain over every order in which the three could be added. It answers: with a perfect pace forecast, what is still unknowable, and which unknown costs most?

## Forecast (L2)

Pace for each car comes from earlier races only: its team's pace in recent races (weight halves every 5 races, halved again at a season boundary) plus the driver's shrunk gap to the team-mates. Four versions are scored: history only, the grid only, a regression of real race pace on history and the grid slot, and the same plus the car's first-session qualifying time gap (a gap in percent says more than a slot number). Each simulation draws a team shock and a driver shock sized from how wrong the same model was before. Probabilities of winning, a podium and the top ten are scored with log loss and the Brier score against (a) the grid-slot-only benchmark and (b) the L1 ceiling, and checked for calibration.

The two regression versions are also scored recalibrated: a two-parameter logistic correction fitted on the model's own forecasts of earlier races (so nothing from the race being forecast), with win probabilities rescaled to add to one. The recalibrated models exist only once about 30 races of forecasts are behind them, so the summary also reports every model on that same set of races.

Scoring note: the finishing-order correlation uses the expected finishing order (mean position over the simulations). The first version averaged the correlation of single simulated races, which understates any forecast with real uncertainty; the single-run figure is still reported as `single_run_correlation`.

Limits: a team that changes its name has no history (its drivers' own history is used instead); a regulation change makes history weak, which is why the first races of 2022 are the hardest; the tyre and pit-loss values come from circuit-level stage 2 tables that include the race being forecast (a small leak for a backtest; a live forecast uses the same tables without it).

## First run on real data (version 1) and what it changed

Version 1 (144 dry races) reproduced the broad dynamics (grip-decay curve within 0.017; finishing-order correlation 0.91 against 0.64 for the grid, though with each car's own pace, so a ceiling) but let cars pass too easily: pole won 44% of simulated races against 55% real (50% against 70% on the holdout), 43% of lost poles went to an on-track pass against 29% real, and the fitted passing threshold and start noise sat at the edge of their search ranges. The undercut for stops 1-2 laps apart came out at +18.0 points against +6.7 real (95% range +0.6 to +12.9), the circuit spread of the grid-to-finish correlation was compressed (0.41-0.63 simulated, 0.35-0.80 real), and forecasts from history alone lost to the grid on the 2025-on holdout (podium Brier skill -26%) while history plus grid was level with it (-6%). The simulator was also underconfident about favourites (at a forecast win chance of 40%, 57% won). Version 2 widens the search, fits against a measured pass rate, sets passing difficulty per circuit, adds the measured stop-cost spread, adds the qualifying gap and the recalibration, and replaces the correlation of single runs by that of the expected order. The version 2 numbers are in `outputs/sim_*/findings.txt`.

## Second run on real data (version 2) and what it changed

Version 2 (144 dry races: 52 for fitting in 2022-24, 33 in the 2025-on holdout, 59 from 2018-21) fitted a passing threshold of 1.40 s/lap, lap-1 noise 0.8 s, grid spacing 0.30 s and a lap noise of 0.49 s (the edge of that search range, so version 2.1 widens it), with a measured stop-cost spread of 1.44 s. On-track passes per car-lap, real against simulated: 0.0397 vs 0.0401 on the fitting years, 0.0364 vs 0.0365 earlier, 0.0349 vs 0.0391 on the holdout. The grip gap fell from 0.017 to 0.011.

- **Replay.** With everything known (L0) the expected finishing order correlates 0.93 with the real one (grid 0.64) and the winner is called 80% of the time (grid 55%). Pole wins 45% of simulated races against 55% real, so the simulator still under-protects pole. With stops, safety cars and retirements sampled from earlier races (L1) the correlation is 0.76 (0.76 against 0.66 for the grid on the holdout; a single simulated race 0.54). How pole is lost, real against simulated: start 9% vs 8%, pit cycle 43% vs 38%, safety car or red flag 9% vs 5%, leader retired 6% vs 9%, on-track pass 29% vs 40%. The undercut for stops 1-2 laps apart was +16.3 points simulated against +6.7 real (still too strong); for 3+ laps apart +30.5 vs +29.5. The across-circuit agreement of the grid-to-finish correlation was 0.66 and the simulated spread across circuits was half the real one (sd 0.05 vs 0.11).
- **Ablation** (138 races, true pace known, 3.39 places wrong from the grid alone, 2.57 with pace only, 1.13 with everything known). Not knowing who retires costs 1.25 of the 1.45 places that the three unknowns cost together (86%), strategies 0.18 (12%), safety cars 0.02 (1%); on the holdout 82%, 14% and 4%. For the winner's log score safety cars matter more (0.08 of 0.26 overall, 0.14 of 0.24 on the holdout).
- **Forecast.** History plus the grid and history plus the qualifying gap were level (winner log score 1.28 vs 1.29, podium Brier 0.063 each), so the qualifying gap adds almost nothing. Recalibration removed the favourite underconfidence; on the 94 races where every model has a forecast, log loss was 0.096 for the win (grid slot 0.114), 0.200 for the podium (0.230) and 0.447 for the top ten (0.504), and Brier skill against the grid was +5% for the podium and +13% for the top ten. On the 2025-on holdout the model is level with the grid (podium -3%, top ten 0%); the skill comes from earlier seasons.

## Version 2.1: checks without confounds, and measured additions

- Undercut and circuit checks run at both L0 (the real stops, safety cars and retirements) and L1; any gap at L0 is the simulator's own, any extra gap at L1 comes from sampled inputs.
- Forecast skill is reported season by season (`forecast_by_season.csv`), so the 2025 vs 2026 difference is visible.
- The lap-noise search range is widened to 0.7-1.5 times the measured value.
- **Lap noise by running position.** Clean laps are measured around each stint's straight-line trend with the field's common movement removed, in three buckets (top 3, 4-10, the rest). The ratios, normalised to average 1 over laps, are written to `noise_by_position.csv` and used as fixed multipliers (`noise_top`, `noise_mid`, `noise_rest`) on the lap noise by current running position. They are measured, not fitted.
- **Cover stops.** When a car stops on a green lap, the car directly ahead, if its own stop is due within 12 laps and its tyres are at least 4 laps old, may stop on the next lap. The probability is measured as (exposed - unexposed) / (1 - unexposed), the share stopping next lap when the car behind has just stopped against when it has not (`cover_stops.csv`). It applies only where stops are sampled (L1, L2, ablation with strategies unknown); with real stops (L0) the covers are already in the data.

### Version 2.1 on real data

Ablation (138 races, true pace known): 3.39 places wrong per car from the grid alone, 2.59 with pace only, 1.15 with everything known (holdout 3.31, 2.71, 1.29). Not knowing who retires costs 1.27 of the 1.44 places the three unknowns cost together (88%; holdout 85%), strategies 0.15 (11%; holdout 12%), safety cars 0.02 (1%; holdout 3%). With the true pace the model's log loss for the win is 0.071 against 0.113 for the grid slot, so a good pace forecast is worth a lot more than any of the three unknowns; the history-based forecast recovers only a small part of that (win log loss 0.097 recalibrated), and its skill against the grid was about zero in 2025-26. That is the reason for adding Friday practice pace below.

## Version 2.2: three more measured targets

The version 2.1 fit had a ridge: the passing threshold and the lap noise could trade off against each other (the lap noise sat at the edge of its range) because the grip curve and the pass rate cannot tell them apart. Version 2.2 adds measurements that can.

- **Time within a second of the car ahead.** Stage 1 now stores the race-clock time at which each car crosses the line at the end of every lap (`laps.lap_end_s`, from FastF1's lap-time pass). From those, the gap to the car directly ahead (by running position) is known at the end of every lap. The share of car-laps within 1.0 s of the car ahead (green laps that follow a green lap, from lap 2, not within a lap of the car's own stop) is computed identically for real and simulated races (`close_counts`). Passing threshold and the closest-follow gap `min_gap` are now searched against three targets: the grip curve, the pass rate and this close share (one loss unit is a 10% gap in the share). A high threshold with low noise gives long trains of cars at the minimum gap; a low threshold with high noise spreads the field out, so the share separates them.
- **Lap 1 is measured, not fitted.** The mean absolute number of places a car is from its grid slot after lap 1 is measured on the fitting races; the lap-1 noise is the value (bisection, lap 1 simulated alone) at which the simulator reproduces it, for each grid spacing tried. Without lap-1 positions the old grid search is used.
- **Stop-cost spread for front-runners.** Stops made from the top three running positions are cleaner than the rest (fewer queues, no traffic in the pit lane), so the spread beyond lap noise is measured separately (`sigma_pit_top`, used in the engine for cars running in the top three at their stop). Both spreads are measured again after the fit with the lap noise the fit settled on (version 2.1 subtracted the measured lap noise before the fit, which was inconsistent). The purpose is the pole under-win: at L0 the simulator lost pole in the pit cycle more often than real races do.

Replay now reports the close share and the places moved on lap 1, real against simulated (`pass_check.csv`).

## Practice pace in the forecast

Stage 1 now also downloads the Friday practice laps (`practice_laps`). `src/sim/practice.py` turns them into two numbers per car and weekend: long-run pace (stints of at least five clean laps; a session model with a separate effect for every driver, linked tyre-compound effects, one shared wear slope and ridge-shrunk 15-minute track-evolution steps; sessions averaged weighted by laps, capped at 30) and best-lap gap in percent. Cars without usable laps take their team-mate's long-run pace or the slow end of the field for best lap. The forecast regression gets both as extra columns (a sixth-column blend: history, grid, qualifying gap, long-run pace, best-lap gap, intercept), is scored as a further variant (recalibrated like the others) and `practice_signal.csv` reports how much the practice columns explain on their own. Teams run different fuel loads and engine modes in practice, so the blend is left to learn how much each is worth.



## Version 2.3: lasting spells, race-to-race passing difficulty, the whole gap distribution, sharpened forecasts

Every addition is off by default (settings of zero) and is switched on only by what `calibrate` measures, so earlier results are reproduced until the calibration is re-run.

- **Pace drift (`drift_sd`, `drift_rho`).** Lap noise in version 2.2 was white: a car's slow lap said nothing about the next. Real cars have spells. `lap_dependence` multiplies clean-lap residuals (around each stint's straight-line trend, field movement removed) in pairs 0, 1, 2, 3, 5, 8 and 12 laps apart. A straight-line fit absorbs part of any slow wander, so the expected products under a drift (variance s2, carrying over a share rho each lap) plus white noise are computed stint by stint through the fit's hat matrix, and s2 and the white variance are found by non-negative least squares for each rho on a grid. On planted data the method recovers the drift size to about 10% and finds nothing in pure noise. When a drift is found, the lap noise used is what remains of the tyre fit's total residual after the drift is taken out, so the total variance is unchanged. Written to `lap_dependence.csv`.
- **Passing difficulty from race to race (`thr_sd`).** Each simulated race draws its passing threshold as thr x exp(thr_sd x z). `pass_dispersion` compares the real passes in each 2022-24 race with the simulator's expectation: the sum of squared differences over the sum of the simulator's own variances (1 means the simulator is as variable as reality). `thr_sd` is the grid value (0 to 0.6) at which that ratio is closest to 1, and 0 if it is already at or below 1. Read it as an upper bound: anything else the simulator misses in a race (a wrong pace estimate, say) is counted as difficulty. Written to `pass_difficulty.csv`.
- **The whole gap distribution.** The fit's close-running term used only the share of car-laps within 1.0 s of the car ahead. It now also compares the shares within 0.5, 2.0 and 3.5 s (the mean of their squared log-ratios, in the same units), so a simulator with the right share of tight laps but too many or too few loose ones is penalised.
- **Sharpened win probabilities.** The recalibration is fitted on pooled car-level probabilities, which ignores that exactly one car wins each race. `fit_sharpen` fits one exponent g, p -> p^g / sum p^g, on the model's own earlier win forecasts by log score (a conditional-logit fit, pulled toward 1), once 20 earlier races exist. "History + grid, sharpened" and "History + grid + practice, sharpened" carry those win probabilities, with podium and top-ten probabilities as recalibrated. The exponent used is in `forecast_races.csv` (`sharpen_exponent`).
- **Stage 2 side.** The qualifying-by-lap-time table now carries stability ranges (40 random 70% subsets of seasons), and the answer table drops the blank rows for blocks that do not exist in a period.

What to look for on the first real run: whether drift is found and how big (a drift of a tenth of a second or more that lasts 10 laps matters for who follows whom); whether `thr_sd` is above zero and whether pole then wins closer to the real 55% and the undercut for stops 1-2 laps apart falls toward +6.7 points; whether the gap shares at 0.5, 2.0 and 3.5 s sit near the real ones; and whether the sharpened win probabilities beat the recalibrated ones on the same races.


## Version 2.4: dirty air, a drift check, where the errors sit

The first real run of version 2.3 left two clear faults with everything known (L0): too many on-track passes (0.0407 per car-lap against 0.0371 real; 0.0413 against 0.0345 in the held-out races), too many car-laps within a second of the car ahead (28.9% against 20.8%), and a pole sitter who wins 42% of simulated races against 55% real because he is passed on track too often (40% of simulated pole losses against 29% real). With sampled strategies (L1) the grid also dissolved too early (grid-versus-running-order correlation 0.778 at 30% of the race against 0.822 real) and circuits looked alike (spread across circuits 0.05 against 0.11). Two further things looked suspicious in the calibration: the lap-time drift (sd 0.46 s) was larger than the white lap noise (0.34 s) and sat at its ceiling of 0.7 of the variance, and the slot spacing sat at the edge of its search grid.

- **Dirty air (`da_loss`, `da_range`).** A car that starts a lap close behind another loses time, in a straight line from `da_loss` seconds right behind to nothing at `da_range` seconds. It is not charged on stop laps or neutralised laps, and it enters the lap time the passing rule compares, so a pass needs a bigger pace advantage and a follower settles at a wider gap than the closest-follow gap. Version 2.3 let a follower close right up to `min_gap` at full speed and sit there, which is how the simulator got both too much bunching and too many passes.
- **Measuring it (`dirty_air`).** `add_gap_prev` gives each lap the gap to the car ahead at the end of the previous lap, from the line-crossing times. For clean laps, each stint's straight line and constant are taken out of the lap times and of the gap-bin indicators (so only differences between laps of the same stint count), the field's movement on the lap is removed, and the pooled loss in each gap bin (0-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3 s) is read against laps that started 3 s or more behind. Intervals resample races. A straight line is fitted over ranges of 1.5, 2, 3 and 4 s. A car that has just had a fast lap is also closer to the car ahead, which hides some of the loss, so the measured value is likely too small; the fit therefore tries 0, 0.5, 1, 1.5, 2 and 3 times it and the data choose (0 switches dirty air off). Written to `dirty_air.csv`.
- **Drift with traffic taken off.** Traffic lasts several laps, so persistent slow laps behind a slower car look like a drift in the car's own pace. `lap_dependence` now takes the measured dirty-air loss off each lap time first, reports both versions (`lap_dependence.csv` and `lap_dependence_before_dirty_air.csv`), and says when the drift sits at its ceiling (then it is a floor, not a measurement).
- **Drift check (`drift_check.csv`).** The fitted settings with the drift scaled to 0, 50% and 100% (the lost variance moved to white noise so each lap's spread is unchanged), scored on the three fit targets without re-fitting. A drift that stands in for something else shows as a loss that barely rises when it is removed.
- **Where the errors sit (replay).** `position_check.csv` splits passes per car-lap and the close share by running position (P1, P2-3, P4-10, P11 and below), real against simulated, so a pole sitter who is passed too easily shows up as an excess at P1. `stop_timing_check.csv` compares real stops with the strategies the sampler draws: stops per car, the share that never stop, when the first stop comes (10th to 90th percentile, as a share of the race) and how far apart neighbours on the grid make their first stops. The sampler already copies a whole earlier race (all its cars' stops, assigned by grid slot), so independence between neighbours was not the cause of the early dissolve; this table says whether it is the timing itself.
- **Search range.** Grid spacing now searches up to 0.9 s (it sat at the 0.65 edge).
- **Forecast.** The sharpened variants are removed: they matched the recalibrated forecasts and added nothing.

What to look for on the next real run: the dirty-air table (how much is lost at 0-0.5, 0.5-1, 1-1.5 s, and whether the multiple the fit keeps is 0, 1 or higher); the drift before and after taking traffic off, and whether the drift check's loss rises when it is removed; whether passes per car-lap and the close share both move toward the real values (0.0371 and 20.8%) and the pole sitter's win rate toward 55%; and in the by-position table whether the excess passes sit at P1 and P2-3. If the L1 early dissolve remains, the stop-timing table shows whether first stops come earlier or are more bunched than real ones.


## Version 2.5: undercut, start noise by slot, a sampler that looks like recent racing

The first real run of version 2.4 fixed the passing and bunching errors with everything known (L0): passes per car-lap 0.0371 real against 0.0376 simulated, car-laps within a second 20.8% against 23.4%, grip curve within 0.007. What was left: the undercut for stops 1-2 laps apart was far too strong (+15.6 points against +6.2 real; stops 3+ laps apart and the gap-standardised measure were fine), which explains most of the pole sitter's excess losses in the pit-stop cycle (28% of races against 19%); the pole sitter lost the lead at the start in 7.0% of races against 4.1% although the field changed places on lap 1 less than real fields (1.47 against 1.62 places per car); passes sat too few at the front (P1 0.0093 against 0.0117, P2-3 0.0177 against 0.0219) and too many at the back (P11 and below 0.0466 against 0.0415). With sampled inputs (L1) the grid dissolved too early in the held-out races (correlation 0.771 against 0.822 at 30% of the race) because the sampled first stops came too early (median 0.29 of the race against 0.35; 75th percentile 0.41 against 0.49), 3-4% of sampled cars never stopped (none do in real races), neighbours' first stops were closer than real ones (0.10 against 0.13 of the race) and the leader retired in 11% of pole losses against 6%.

- **Out-lap share of a stop (`outlap_share`).** A share of a green-lap stop's cost is paid on the out-lap instead of the in-lap, the total unchanged. The car coming out on fresh tyres then is not seen as fast on the lap it rejoins, which is when it would pass the cars it comes out behind. `fit_outlap` picks the share (0, 25, 50, 75, 100%) at which the simulator, run on 2022-24 as it happened with real stops, gets the undercut advantage right for stops 1-2 laps apart and for stops 3+ laps apart (each against its real value in units of 6 points). No pole outcome is used. `undercut_fit.csv` has every share tried; `findings.txt` says when the split barely moves the undercut (then the excess lies elsewhere).
- **Start noise by grid slot (`start_front`, `start_ramp`).** Lap-1 noise of the pole sitter is `start_front` times the field's, rising in a straight line to the field's by slot 11. `fit_start_shape` tries 1.0, 0.7, 0.5, 0.35 and 0.2; for each, the field-wide noise is set so cars change places on lap 1 as much as real ones do, and the share whose places moved by grid-slot band (1-2, 3-6, 7-12, 13+) is closest to the real ones is kept. Only lap-1 positions are used. `start_shape.csv` has the table.
- **Passes by running position in the loss.** The pass term of the fit now also holds the log-ratio of the pass rate in each running-position band (P1, P2-3, P4-10, P11 and below) in units of 0.30, so the fit cannot get the overall rate right by having too many passes at the back and too few at the front. The drift check uses the same loss.
- **Drift share and noise range.** The fit searches the share of the measured drift it keeps (0, 50, 75, 100%; the variance given up goes to white noise, so each lap's spread is unchanged), and lap noise up to 2.0 times the measured value (it sat at 1.5).
- **Sampler, recency and circuit.** A pool race is weighted by e^(-seasons older / 1.5) as well as 3 times for the same circuit, so the held-out seasons are sampled mostly from the latest ones. Sampling never looks at the race being simulated.
- **Sampler, no car without a stop.** A template car with no stop (it retired early) takes the strategy of the nearest car that stopped.
- **Sampler, stop spread.** `_stop_jitter` samples strategies for the pool's most recent 15 races, compares the mean distance between neighbours' first stops with the real one, and gives each stop an independent normal shift of the size that closes the gap (moment matching on the half-normal relation, at most 8% of the race). It is a property of the pool, not of the target race.
- **Sampler, retirement by grid band.** Cars retire at the pool's rate for their grid band (slots 1-3, 4-10, 11+), each pulled toward the overall rate by 40 car-races.

What to look for on the next real run: the undercut for stops 1-2 laps apart in L0 and L1 (target +6.2, real range -0.1 to +12.3) and the pole sitter's win rate with the share of pole losses in the pit-stop cycle; the lap-1 places moved by slot band and the share of races in which pole loses the lead at the start (target 4.1%); passes by running position at the front; the L1 grip curve in the held-out races and the first-stop percentiles in `stop_timing_check.csv`; and whether the leader's retirement share of pole losses falls from 11% toward 6%. If the out-lap split does not move the undercut, the next suspect is the size of the fresh-tyre advantage itself (the compound offsets and wear), which would be fitted against the same two undercut targets.


## Version 2.6: a diagnostic, no new settings

Version 2.5 left one fault in three forms: pole sitters win 44% (L0) and 42-43% (L1) of races against 55% real, the grid's grip dissolves too early once strategies, safety cars and retirements are sampled, and forecast probabilities are too flat (favourites win more often than the forecast says, even with true pace). Two v2.5 ideas (`start_front`, the out-lap share) were neutral. This version changes nothing in the simulator; `diagnose.py` finds where the fault sits.

- **Factors.** All eight combinations of strategies, safety cars and retirements known or sampled, each scored on the pole sitter's win rate, the gap to the real grip curve (early: after lap 1 to 30% of the race; late: the rest), passes, close running and lap 1. Each unknown's share is a Shapley value over adding order; "given to L1 alone" and "taken from L0 alone" show it without averaging.
- **Levers.** At L0 each noise and passing setting is scaled alone (lap-1 noise, lap noise for the field and for the top three, drift, stint effect, stop-cost spread, passing-difficulty spread, passing threshold, grid spacing). `lever_reach.csv` gives the multiplier at which a lever alone would equal the real pole rate (straight line through the points tried, so indicative) and what that does to passes, bunching and the grip curve. A lever that closes the gap only by breaking passes or grip is not the cause.
- **Pace margin.** Pole wins by how much faster than every other car the pole sitter's measured pace was, real against simulated. If the simulator under-wins mainly when the pole sitter is clearly quickest, the cause is noise in the race; if it under-wins when he is not quickest, the cause is the pace measurement or the real race not punishing a slower pole sitter.
- **Stops by season** and **lead loss by combination** (needs the database) complete the picture.

What to look for: which unknown carries the pole gap and the early grip gap; which lever reaches the real pole rate with the smallest damage to passes and grip; where in the pace-margin table the shortfall sits.


## Version 2.7: tyre warm-up, bad starts, whole strategies

The diagnostic (`diagnose.py`, version 2.6) on the real run of version 2.5 showed that the pole sitter's shortfall (45% of simulated races won against 55% real, with everything known) is not the sampled inputs (nothing known 42%, everything known 45%) and not noise (no single noise or passing setting reaches the real rate without breaking passes, bunching or the lap-1 shuffle). By how the lead is lost, it is the pit-stop cycle (27% of races against 19%; since 2022 25% against 15%) and the start (7.2% against 4.1%). The pole sitter under-wins most when he is also the fastest car (77% real, 55% simulated when he is 0 to 0.2 s/lap faster than the rest). With sampled inputs the early loss of the grid's grip came two thirds from sampled strategies, and sampled stops per car sat below real ones in most seasons.

- **Tyre warm-up (`warm_amp`, `warm_decay`, `RaceSpec.warm`).** The engine charged wear from age zero, so a fresh set was at full speed on its first lap. A new set is now slower on the laps after the out-lap by `warm_amp` x `warm_decay`^(lap - 2) seconds (stint lap 2 onward; the out-lap is part of the stop's cost). It is measured, not fitted: `warm_up` fits the stage 2 tyre regression for each dry race without the laps being measured (stint laps 2-8 after a stop, first stint left out, so the wear slope cannot absorb them) and takes the mean residual at each stint lap minus the mean at stint laps 12-24. One decay is chosen for all years, then an amplitude for 2018-21 and for 2022 on; below 0.03 s it is set to 0. Each race carries its era's value (`RaceTruth.warm`), so L0 and the sampled levels use it alike. `warm_up.csv` and `findings.txt` give the excess by stint lap with its standard error. Nothing about the undercut or pole wins is used. If the excess is not measurable, the amplitude is 0 and the engine is unchanged.
- **Bad starts (`bad_start_p`, `bad_start_loss`).** On lap 1 each car has a chance of losing an exponentially distributed extra time, so most starts are clean and a few cost several places. `fit_start_shape` now searches the front-row noise share and the bad-start chance and cost together (the field-wide noise is still set so the whole field shuffles as much as real fields do), scoring the places moved by slot band, the share of pole sitters still leading after lap 1 (units of 0.02) and the share of front-six cars that move 3 or more places (log-ratio in units of 0.25). Only lap-1 positions are used. The front-noise grid is 1.0, 0.6, 0.35. `start_shape.csv` has every combination.
- **Sampler.** A template car that retired never completed its plan, so a sampled car copying it had too few stops. It now takes the plan of the nearest finisher in the template race (by grid rank). Stops per car by season (diagnose) are the check.

What to look for on the next real run: the warm-up table (is the excess on stint lap 2 well above its standard error, and bigger since 2022?); the undercut for stops 1-2 laps apart (real +6.3, version 2.5 +11) and the pit-stop cycle's share of lost leads (19% of races); whether the start losses fall toward 4.1% of races; the pole sitter's win rate; and stops per car by season. If the warm-up is not measurable, the pit-cycle excess is from something else and the next suspect is the pit stop's own time loss and the cover/reaction rules.

## Version 2.8: the pole-sitter shortfall was the drift

Pole wins sat at 46% (simulated) against 55% (real). Tests on the real races: switching the tyre warm-up off restores fresh-tyre closing but not pole wins; persistent-pace passing does not help; removing the independent per-car pace drift and re-matching the passing threshold to the real pass rate lifts pole wins to about 54%. The old drift check turned drift down without re-fitting the threshold, so it looked harmless. A second check agrees: the ten-lap change in the gap between adjacent cars (robust sd) is 3.35 s in real races and 6.4 s in v2.7, because independent drift lets pairs wander apart faster than real cars do.

- **`src/sim/refit.py`** re-fits together a drift factor (1, 0.5, 0.25, 0), a lap-noise factor (1, 0.8, 0.65) and a threshold factor (found for each so the pass rate equals the real one), and a warm-up scale (1, 0.5, 0). Chosen on 2022-24 only by gap wander, close share, grip and, for warm-up, the closing rate after the first stop; 2025+ and 2018-21 are left for checking.
- **Engine:** `Params.warm_scale` multiplies whichever warm-up is in force (0 switches it off). Default 1, so the engine is unchanged unless the refit sets it.
- **Real run (145 races):** drift x0.25, lap noise x0.8, thresholds x0.49, warm-up off (its measured excess was mostly an artefact: closing after a stop was real +0.78/+0.53 s a lap, simulated -0.04/-0.37 with warm-up and +0.68/+0.31 without). Pole wins 46.5% to 53.4% (real 55.2%); pass rate 0.0374 (real 0.0371); wander 6.42 to 4.64 s (real 3.35); faster car makes the pass 69% to 81% (real 86%).
- **Still short:** holdout pole wins 59.3% against 70.6% real; wander still above real; the pole sitter regains the lead 0.55 times a race against 0.76 real. Some of what was measured as drift may be deterministic structure (tyre-wear curvature), which is untested.
- Run `python -m src.sim.refit` after `calibrate`; it backs up `params.json` as `params_before_refit.json` and a second run starts from the backup. Replay, ablation, forecast and diagnose then read the refitted parameters.

## Stage 3 frozen at version 2.8

The simulator, its calibration and the five analysis steps (refit, replay, ablation, forecast, diagnose) are closed. Nothing below is to be tuned further; later stages use the simulator as it stands. Real run on 145 dry races (2018 to 2026), 348 tests passing.

**What it does well**
- Pole sitter wins 53.8% of races with everything known, against 55.2% real (v2.7: 46.5%). Pass rate 0.0374 against 0.0371 real. Grip decay within 0.015 correlation points of the real curve.
- Finishing-order correlation 0.93 with everything known; 0.76 with only the true pace known and the rest sampled; 0.64 for the grid alone.
- Places lost even with a perfect pace forecast: retirements 87%, strategies 13%, safety cars 1% (same ranking in every split).
- Forecasts (125 races, each forecast from earlier races only): history plus the qualifying gap, recalibrated, beats grid-only on winner log score (holdout 0.99 against 1.11) and ties it on podium Brier (0.061 against 0.061). Practice adds nothing (correlation 0.32 with race pace, 0.10 beyond history, -0.03 in 2026). Ceiling with true pace: podium Brier 0.037.

**Known limits (do not read results past these)**
1. Holdout (2025 on) pole wins: 59% simulated against 71% real. 34 races, so about 1.4 standard errors; unresolved either way.
2. The pit-stop cycle costs the pole sitter too often: 28% of races against 19% real (57% of pole losses against 43%). On-track passes at the front are too rare (P1 0.005 against 0.012 per car-lap).
3. Undercut for stops 1 to 2 laps apart: +14 points simulated against +6 real (real range -0.1 to +12.3). Stops 3 or more laps apart agree (+33 against +29). Warm-up is switched off in the fit, which made the 1-2 lap case slightly worse than version 2.7.
4. Lap 1 moves 1.39 places per car against 1.62 real, mostly the front rows (0.63 against 0.90 for grid slots 1 to 2).
5. Ten-lap gap wander between adjacent cars is 4.65 s against 3.35 s real; the pole sitter regains a lost lead 0.55 times a race against 0.76 real.
6. Sampled strategies stop slightly less than real cars (1.60 against 1.69 stops per car; 2021 1.38 against 1.81).
7. Some of the pace drift removed in 2.8 may be fixed structure (tyre-wear curvature) rather than noise; untested.
8. Circuit differences: the simulator is too predictable at Austin, Zandvoort and Las Vegas and too chaotic at Imola, Monaco and Suzuka (everything known).

**Why frozen.** No single dial closes the remaining gaps without breaking passes, bunching or the grip curve (diagnose, lever sweep). None of the headline findings (what wins a race, what is lost to retirements, how far forecasts beat the grid) depends on them.

**How to reproduce.** `calibrate`, `refit`, `replay`, `ablation`, `forecast`, `diagnose` in that order (see the README). Output folders: `sim_calibration`, `sim_refit`, `sim_replay`, `sim_ablation`, `sim_forecast`, `sim_diagnose`.
