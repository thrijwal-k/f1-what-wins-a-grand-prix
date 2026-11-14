# F1 project - stage 1: data foundation

Downloads Formula 1 history (Jolpica-F1, 1950 onward) and lap, tyre and weather
detail (FastF1, 2018 onward), builds one DuckDB database from the untouched raw
files, and runs data checks on it.

```
Jolpica + FastF1  ->  data/raw (untouched)  ->  data/f1.duckdb  ->  checks
```

## One-time setup (Windows PowerShell)

Open PowerShell and run these one at a time.

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
```

Close PowerShell and open a new one so the new programs are found, then go to
the folder you unzipped this project into:

```powershell
cd $HOME\Documents\f1-project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

then run the `Activate.ps1` line again. You should see `(.venv)` at the start of
the prompt. Every new PowerShell window needs the `Activate.ps1` line again.

## Run it

1. Check the tests pass on your machine (takes a few seconds):

   ```powershell
   python -m pytest -q
   ```

2. Inventory: what do the sources actually return? (about 30 requests)

   ```powershell
   python -m src.run_all --inventory
   ```

3. Small trial run: two recent seasons, whole pipeline end to end.

   ```powershell
   python -m src.run_all --from-season 2024 --fastf1-from 2024
   ```

4. The full run, 1950 to now. Jolpica is rate limited, so expect this to take a
   couple of hours, and FastF1 adds more. Leave it running (plug the laptop in
   and stop it sleeping). You can press Ctrl+C at any time and start it again:
   finished seasons and races are skipped, nothing is downloaded twice.

   ```powershell
   python -m src.run_all
   ```

Stage 1 also downloads, for each FastF1 weekend, the Friday practice laps (FP1-FP3 or the sprint-weekend equivalents; used by the Stage 3 forecast for practice long-run pace) and the race-clock time of every lap's finish line crossing (used by Stage 3 to measure how close cars run). Both are resumable and skipped when already downloaded. To add them to an existing database without refetching anything else:

```powershell
python -m src.run_all --skip-jolpica
```

This makes several thousand FastF1 requests (FastF1 allows about 500 an hour, so expect several hours; stop and restart as often as you like). `--practice-sessions FP2` fetches only FP2, `--skip-practice` and `--skip-laptimes` skip either extra.

Afterwards, to rebuild the database and re-check without downloading anything:

```powershell
python -m src.run_all --skip-jolpica --skip-fastf1
```

To look inside the database:

```powershell
python -c "import duckdb; print(duckdb.connect('data/f1.duckdb', read_only=True).sql('SELECT season, count(*) AS races FROM races GROUP BY season ORDER BY season DESC LIMIT 5'))"
```

If you see many "HTTP 429" waits in the log, the API is asking you to slow
down. Set a longer pause between requests and run again:

```powershell
$env:F1_REQUEST_DELAY = "2"
python -m src.run_all
```

## Derived tables (race_summary, driver_race, team_race)

`python -m src.run_all` builds three joined tables after the base tables, so the analysis, the forecast log and any dashboard read the same rows. To rebuild only these from an existing database (a few seconds, no downloads):

```
python -m src.derive
```

| Table | One row per | Holds |
| --- | --- | --- |
| `race_summary` | race | starters, classified, retirements, winner, grid pole and qualifying pole, whether pole won, safety-car, VSC and red-flag laps, wet flag |
| `driver_race` | driver per race | grid, result, status class (finished, lapped, mechanical, incident, unspecified, disqualified, not_started), points, qualifying stage and gap to pole, sprint result, stops and first stop, clean-lap race pace, laps led, tyre sequence |
| `team_race` | constructor per race | cars, best grid and finish, points, retirements by cause, wins, podiums, poles, best and mean qualifying gap, best and mean race pace, mean stops |

A null means not known, never zero: pace and tyre columns are null before 2018, pit-stop columns before 2011. Full rules are at the top of `src/derive.py`. `src.checks` verifies the tables reconcile with `results` (same rows, same points).

## What the checks mean

`python -m src.run_all` ends by printing every check. `FAIL` (an error) means the
data is wrong or incomplete, and the command exits with an error code. `warn`
means worth a look but possibly legitimate, such as shared drives in the 1950s.
Results are also saved in the `data_checks` table, and any disagreement between
Jolpica and FastF1 is logged in the `mismatches` table, never silently fixed.

## Folder layout

| Path | What it holds |
| --- | --- |
| `src/jolpica.py` | throttled, resumable Jolpica downloader |
| `src/fastf1_loader.py` | FastF1 laps, tyres, weather (no car telemetry) |
| `src/build_db.py` | rebuilds `data/f1.duckdb` from raw files |
| `src/derive.py` | builds `race_summary`, `driver_race` and `team_race` from the base tables |
| `src/checks.py` | data quality checks |
| `src/inventory.py` | quick look at what each source returns |
| `src/run_all.py` | the one command that runs it all |
| `tests/` | tests using small synthetic files, no internet needed |
| `.github/workflows/` | `ci.yml` runs the tests on every push; `update.yml` is the weekly update job |
| `data/` | created on first run, not committed to Git |

## Putting it on GitHub

```powershell
git init
git add .
git commit -m "Stage 1: data foundation"
git branch -M main
git remote add origin https://github.com/<your-username>/f1-project.git
git push -u origin main
```

The tests then run automatically on every push. The weekly update job in
`update.yml` stays switched off until you add a repository variable called
`ENABLE_UPDATE` with the value `true`; do that after the first full run works
on your laptop.

## Stage 3: the race simulator

Stage 3 is frozen at version 2.8 (known limits at the end of `docs/stage3_method.md`). Six steps; each needs the calibration. Details of the method are in `docs/stage3_method.md`.

```
python -m src.sim.calibrate     # outputs\sim_calibration  (a few minutes)
python -m src.sim.refit         # outputs\sim_refit  (version 2.8; re-fits drift, lap noise, passing threshold and warm-up together, rewrites params.json; keeps params_before_refit.json)
python -m src.sim.replay        # outputs\sim_replay
python -m src.sim.forecast      # outputs\sim_forecast
python -m src.sim.ablation      # outputs\sim_ablation
python -m src.sim.diagnose      # outputs\sim_diagnose  (version 2.6; a diagnostic, changes nothing)
explorer outputs\sim_calibration
explorer outputs\sim_refit
explorer outputs\sim_replay
explorer outputs\sim_forecast
explorer outputs\sim_ablation
explorer outputs\sim_diagnose
```

| Step | What it does | Main files |
|---|---|---|
| calibrate | turns each dry race into simulator inputs, measures the safety-car stop rate and cost, fits the passing threshold, closest-follow gap, grid spacing and lap noise on 2022-24 against the grip curve, the measured pass rate and the share of car-laps within a second of the car ahead, sets lap-1 noise from the measured lap-1 shuffling, sets a passing threshold per circuit, measures the stop-cost spread (separately for the top three), lap noise by running position, the cover-stop probability, lasting pace spells (drift) and the race-to-race spread of passing difficulty (version 2.3), the lap-time loss in dirty air from real laps, the drift again with that loss taken off, and a check of whether the drift earns its place (version 2.4), then (version 2.5) the share of a stop paid on the out-lap fitted on the undercut, start noise by grid slot from lap-1 movement by slot band, the share of drift kept, and pass rates by running position inside the fit, then (version 2.7) the tyre warm-up measured from real laps by era and a bad-start chance fitted on lap-1 positions | `findings.txt`, `dirty_air.csv`, `drift_check.csv`, `undercut_fit.csv`, `start_shape.csv`, `warm_up.csv`, `lap_dependence_before_dirty_air.csv`, `era_tyres.csv`, `passing_fit_grid.csv`, `circuit_passing.csv`, `noise_by_position.csv`, `noise_by_position_season.csv`, `cover_stops.csv`, `lap_dependence.csv`, `pass_difficulty.csv`, `grip_fit.csv` / `.png`, `neutral_pit_cost.csv`, `race_truth_summary.csv`, `races_left_out.csv`, `params.json` |
| refit | version 2.8. Re-fits the pace drift, lap noise and every passing threshold together (threshold re-matched so the pass rate stays real) and the tyre warm-up scale, choosing on the fitting races (2022-24) by the ten-lap gap wander between adjacent cars, close running and grip; checks pass direction and how the pole sitter loses and regains the lead; the earlier drift check never re-fitted the threshold | `findings.txt`, `refit_grid.csv`, `warm_up_check.csv`, `wander_check.csv`, `pass_direction.csv`, `lead_changes.csv`, `refit_pole_wins.png`, `params.json` (in sim_calibration, replaced) |
| replay | runs every race with everything known (L0) and with pace known but the rest sampled (L1); checks the grip curve, how pole is lost, the undercut and circuit differences against stage 2 (undercut and circuits at both levels), and (version 2.4) where the pass-rate and bunching errors sit by running position and whether the sampled strategies stop at the times real cars do; in version 2.5 the sampler weights recent seasons, gives every car a stop, spreads stop laps like real ones and retires cars by grid band; in version 2.7 a retired template car's plan is replaced by a finisher's | `findings.txt`, `replay_summary.csv`, `replay_races.csv`, `grip_curves.csv`, `lead_change_check.csv`, `undercut_check.csv`, `circuit_check.csv`, `pass_check.csv`, `position_check.csv`, `stop_timing_check.csv`, `replay_levels.png`, `replay_grip.png`, `replay_undercut.png` |
| forecast | pre-race probabilities from earlier races only (history, grid, history + grid, history + qualifying gap, history + practice pace, three of them recalibrated), scored against the grid-only benchmark | `findings.txt`, `forecast_summary.csv`, `forecast_summary_same_races.csv`, `forecast_by_season.csv`, `practice_signal.csv`, `forecast_probabilities.csv`, `calibration.csv`, `log_loss.csv`, `log_loss_same_races.csv`, `forecast_skill.png`, `forecast_calibration.png` |
| ablation | with true pace known, tells the simulator each of strategies, safety cars and retirements or has it sample them, and scores all eight combinations | `findings.txt`, `ablation_summary.csv`, `ablation_shapley.csv`, `ablation_races.csv`, `ablation_ladder.png` |
| diagnose | splits the pole-win gap, the early loss of the grid's grip and the flat probabilities by cause: all eight combinations of known and sampled strategies, safety cars and retirements (Shapley and one-at-a-time), each noise and passing setting turned down alone at L0 with what it costs in passes, bunching and grip, pole wins by the pole sitter's pace margin, stops per car by season, and how the lead is lost in each combination | `findings.txt`, `factor_summary.csv`, `factor_effects.csv`, `lever_sweep.csv`, `lever_reach.csv`, `pole_by_pace_margin.csv`, `stops_by_season.csv`, `lead_loss_by_combo.csv`, `factor_races.csv`, `pole_by_unknown.png` |

`truth.pkl` and `params.json` in `outputs\sim_calibration` are what replay and forecast read; do not delete them between steps.

## Stage 4: public forecasts, logged before the race

Stage 3 is frozen, so this stage only uses it. Each race weekend the frozen simulator forecasts the race from the earlier races
plus the qualifying result, and the forecast goes into a log that cannot be quietly edited. After the race it is scored against
a benchmark that knows only the grid slot.

| Command | What it does |
|---|---|
| `python -m src.sim.live next` | Shows the next race without results, its start time, and whether qualifying is in the database |
| `python -m src.sim.live forecast` | Makes and logs the forecast (about 10 minutes; refuses if the race has results or has started) |
| `python -m src.sim.live score` | Scores every logged forecast whose race now has a result; writes `forecasts/scores.csv` and `scores_summary.csv` |
| `python -m src.sim.live verify` | Checks that no logged forecast was edited, removed or reordered |
| `python -m src.sim.forecast --proxy-tyre` | Re-runs the backtest with the tyre inputs a live forecast would have, to measure how much the backtest flattered it |

**The log.** Each forecast is one JSON file in `forecasts/`, with win, podium and top-ten probabilities for every driver under five
models (the main one is "History + quali gap, recalibrated"; the benchmark is "Grid slot only"). The file records fingerprints of
the code, the frozen settings and the inputs, plus the fingerprint of the previous forecast, so removing or changing an old file
breaks the chain. The date that counts is when the file was pushed to the public repository.

**Weekend routine.** After qualifying: update the data, run `next`, run `forecast`, commit and push `forecasts/`. After the race:
update the data, run `score`, push.

**Limits.** The qualifying order stands in for the grid, so penalties given after qualifying are not included. The simulator is
calibrated on dry races; wet races are logged but scored apart. Tyre inputs come from the last race at the same circuit (the
backtest gave each race its own, which a live forecast cannot have). Practice laps are not used. A handful of races says little:
the backtest needed about 95 races before the ranking of models settled.

## Not covered yet

- Standings tables and car telemetry (not needed for the plan).
- OpenF1 (optional, from 2023, only if live in-race data needs it).
- Where the database lives for the public dashboard (decided in stage 5).

## Stage 2, part 1: how much does the grid decide the race?

Needs the finished stage 1 database. In Anaconda Prompt, in the project folder with
`conda activate f1` done:

```
python -m src.analysis.grid
explorer outputs\grid_vs_result
```

The first command prints the main findings and writes these files to
`outputs\grid_vs_result`; the second opens that folder.

| File | What it shows |
| --- | --- |
| `expected_finish_by_grid.png` | mean finishing position from each grid slot, by era |
| `grid_grip_by_season.png` | how tightly grid order predicts finishing order, season by season |
| `pole_win_rate_by_era.png` | how often pole wins, by era, with 95% intervals |
| `pole_win_rate_by_circuit.png` | the same, circuit by circuit since 2000 (Monaco and Monza highlighted) |
| `*.csv` | the tables behind every chart (era, season, circuit, grid slot, one row per race) |
| `findings.txt` | the printed summary |

Method: one row per finishing position per race (shared drives before 1965 count
once), pit-lane starters count as starting behind the last slot, and races with
fewer than 8 finishers are ignored.

## Stage 2, part 2: driver or car?

Needs the stage 1 database. In Anaconda Prompt, in the project folder with `conda activate f1` done:

```
python -m src.analysis.ratings
explorer outputs\driver_vs_car
```

Team-mates drive the same car, so the gap between them is mostly driver. Each pair of
team-mates in each race is one comparison, and a Bayesian model estimates one rating
per driver from all of them at once, with a 95% interval.

| File | What it shows |
| --- | --- |
| `ratings_time_model_1994_on.png/.csv` | gap in qualifying lap time (percent of a lap) vs the average team-mate, 1994 on |
| `ratings_rank_model_1950_93.png/.csv` | grid places per race vs the average team-mate, 1950 to 1993 only (places mean different amounts of pace in different eras, so modern drivers are judged by the time model) |
| `teammate_head_to_head.csv` | every team-mate pair: comparisons, wins, average gap |
| `findings.txt` | the printed summary |

Method: the model is solved exactly (no sampling), with noise and spread of skill
estimated from the data. Time gaps use the latest qualifying stage both drivers set a
time in, and gaps over 20% are dropped as crashes or mechanical trouble. Pairs with the
same position (shared drives) are dropped. Only teams with exactly two cars in a race
are used (customer and privateer entries are not team-mates), and the Indianapolis 500 is
excluded before 1961 (it counted towards the championship, with one-off entries). The charts show only drivers with at least 40
comparisons. The tests check the model recovers known skills from simulated data.

Note on the 1950-93 grid-place chart: because only two-car teams are used, almost every
1950s race drops out (works teams then ran three or four cars), so that chart is in
practice a 1960s-to-1993 picture. Fangio, Ascari and Moss have too few clean pairs to rate.

## Stage 2, part 3: car, driver or everything else?

In Anaconda Prompt, in the project folder with `conda activate f1` done:

```
python -m src.analysis.car_driver
explorer outputs\car_vs_driver
```

It takes a few minutes (it refits each era 40 times on random subsets of seasons).

| File | What it shows |
| --- | --- |
| `variance_shares_by_era.png` | for each era, how much of qualifying position and of race finishing position is car, driver or everything else |
| `variance_shares_by_era.csv` | the numbers, with a stability range for each share |
| `findings.txt` | the printed summary |

Method: a mixed model gives every constructor-season a car effect, every constructor a
persistent effect for the whole era (without it, a team that stays strong for years is
credited to the drivers who stay with it), and every driver an effect for the era, on a score that does not depend on field size (the normal score of
finishing position within the race). The effects are solved exactly and the variances
learned by EM. The low/high columns are the spread across random 70% subsets of seasons;
a stability range, not a formal confidence interval. The driver share is a lower bound when
drivers rarely change team, because a driver is only separated from a car by moving.

### Stage 2, part 3b: how far can the split be trusted?

```
python -m src.analysis.car_driver_checks
explorer outputs\car_vs_driver_checks
```

Takes several minutes. Four checks on the car-vs-driver model:

| File | What it answers |
| --- | --- |
| `mobility_by_era.csv` | how many drivers changed team and how connected the data is (a driver is only told apart from a car by moving or by having different team-mates) |
| `recovery_by_era.csv` | outcomes invented on each era's real drivers and teams with known shares, then refitted: how far off is the model in each era? |
| `rank_vs_laptime_by_era.csv` | qualifying measured as a rank and as a lap-time gap; do they agree? |
| `rolling_shares.png/.csv` | the same shares over 10-season windows, so dependence on the era boundaries shows up |
| `placebo_by_era.csv` | driver labels shuffled inside each constructor-season: the driver share should fall to about zero |
| `two_car_teams_by_era.csv` | the shares again using only teams that ran exactly two cars (customer and privateer cars removed) |
| `findings.txt` | the printed summary |


### Stage 2, part 1b: how far can the grid findings be trusted?

Rerun part 1 first, because it now leaves out the 1950-60 Indianapolis 500s (a championship round,
but 33 one-off entries in a different kind of race), then the checks:

```
python -m src.analysis.grid
python -m src.analysis.grid_checks
explorer outputs\grid_vs_result_checks
```

| File | What it answers |
| --- | --- |
| `coverage_by_era.csv` | field size, share of rows with no grid slot, share not classified, Indy rows per era |
| `indy_sensitivity.csv` | the early-era numbers with and without the Indianapolis 500 |
| `reliability_by_era.csv`, `reliability_decomposition.png` | pole win rate = P(pole car finishes) x P(wins if it finishes); grid-vs-finish correlation for all starters and for finishers only |
| `circuit_tests.csv`, `circuit_shrinkage.csv/.png` | do circuits really differ? A permutation test shuffles outcomes between circuits inside each season (so a dominant year cannot fake a circuit effect), and empirical-Bayes shrinkage pulls thinly sampled circuits towards the overall rate |
| `season_dominance.csv`, `dominance_correlations.csv` | does the grid look decisive mainly in seasons when one team won almost everything? |
| `pole_by_grid_vs_qualifying.csv` | how often grid penalties change who starts first, and which definition of pole wins more |
| `findings.txt` | the printed summary |


### Stage 2, parts 7 and 8: who wins, and race-pace driver ratings

```
python -m src.analysis.winners
python -m src.analysis.race_pace
explorer outputs\who_wins
explorer outputs\race_pace_ratings
```

`winners` asks where each race winner came from on the grid (pole sitter, his team-mate, the next-quickest team, further back). `race_pace` rates drivers from race laps against their team-mate (2018 on) and compares with the qualifying rating. The October review fixes (dry-only pit loss, absolute spreads, lap-time qualifying split) are in `tyres`, `car_driver` and `race_flow` and are described in `docs/stage2_findings.md`.

### Stage 2, deep dive: opening up "everything else", strategy, and how far to trust the split

```
python -m src.analysis.what_decides
python -m src.analysis.strategy
python -m src.analysis.car_driver_robust            (about 15 minutes; add --quick for a short version)
python -m src.analysis.answer
explorer outputs\what_decides
explorer outputs\strategy_teammates
explorer outputs\car_vs_driver_robust
explorer outputs\stage2_answer
```

`what_decides` splits the race result into car, driver, qualifying form, mechanical failures and accidents (one combined "retirements" block where too many retirements have no stated cause, as in 2022-now), safety-car timing, strategy and unexplained, by period (1950-69, 1970-89, 1990-2010, 2011-17, 2018-21, 2022-now). `strategy` compares team-mates who both finished on the lead lap, with grid place controlled, to estimate the places an extra stop, a later first stop or a safety-car stop is worth. `car_driver_robust` stress-tests the car-vs-driver model on invented worlds it does not assume, splits driver skill from driver form, fits driver-car fit and in-season car development from the data, and gives race-resampled intervals with allowances for a wrong model (hidden effects the size the data show, and deliberately strong ones). It takes several minutes. `answer` puts the pieces side by side from the other outputs; run it last.

### Stage 2, part 4: tyres, pit stops and what happens after the grid

Uses the FastF1 lap data (2018 on). Two modules:

```
python -m src.analysis.tyres
python -m src.analysis.race_flow
explorer outputs\tyres_and_pits
explorer outputs\race_flow
```

`tyres` (outputs/tyres_and_pits):

| File | What it shows |
| --- | --- |
| `tyre_degradation_by_season.png`, `tyre_by_race/season/circuit.csv` | seconds lost per lap of tyre age for soft, medium and hard, from a model with a driver effect, a separate effect for every lap number (fuel burn, track evolution and weather shared by the whole field) and a wear slope per compound. It is checked against simulated races with known wear, including races where soft tyres are used early and the track improves non-linearly. `findings.txt` prints a SANITY line: if soft does not wear faster than hard in clearly more than half the races, the model is not separating compounds on real data |
| `pit_loss_by_circuit.png`, `pit_loss_by_circuit.csv`, `pit_loss_stops.csv` | time lost per green-flag stop (in-lap plus out-lap against two laps at the driver's own pace); the number the stage 3 simulator needs. `loss_shrunk_s` pulls thin circuits toward the overall level |
| `stops_per_car_by_season.csv`, `winner_stops_vs_field.csv` | strategy mix, and whether winners stop less than the cars around them |
| `tyre_checks.csv` | does the tyre model mean anything: within-circuit permutation test for the 2022 step in wear, and split-half circuit reliability (about 0.4 raw, higher once each season's level is removed, so read circuit ranks coarsely). Compound pace gaps are flagged as unreliable; the simulator calibrates them against real stint lengths instead |
| `circuit_wear_shrunk.csv`, `wear_level_by_era.csv` | what the simulator should use for tyre wear: each circuit's wear minus its season's median, pulled toward the overall mean in proportion to how little data it has; and the median wear by compound before and from 2022. Raw circuit extremes are mostly noise |
| `race_flags.csv`, `status_by_season.csv` | (`wet_borderline` marks wet races with under 100 wet-tyre laps) laps under safety car, VSC and red flag per race (a lap counts when a quarter of the cars record the code), whether the race was wet, and a per-season summary to spot a data problem |

2018 used different tyre names (ultrasoft and so on). In 2018 the three compounds used at a race are
relabelled soft, medium, hard by softness; from 2019 FastF1 already does this. A 2018 race that
does not show exactly three compounds is left out of the tyre model.

`race_flow` (outputs/race_flow):

| File | What it shows |
| --- | --- |
| `grip_decay.png/.csv` | rank correlation between grid and running order, after lap 1 and at each 10% of the distance, ending at 95% (the last lap itself is not used: its positions are missing for many cars, so it would be a different, lead-lap-only sample) |
| `how_pole_is_lost.png`, `lead_takeover.csv` | for each race the pole sitter did not win, when the winner took the lead for good: the start, a pit-stop cycle, a safety car or red flag, the leader retiring, or a pass on track (checked in that priority) |
| `undercut.csv` | how often the chaser ends ahead when it stops first, against when the leader stops first, split by how far apart the two stops were (1-2 laps is the real undercut; 3+ laps mostly reflects strategy), a gap-standardised figure, and the change between 2018-21 and 2022-now. Intervals resample whole races. Observational: who stops first is a choice |
| `grip_recovery.csv` | paired early dip and late recovery of grid grip per group, and the dry 2022-now minus 2018-21 gap |
| `lead_takeover_shift_tests.csv` | whether the way pole is lost changed between periods (it did not, statistically) |
| `conditions.csv` | pole win rate and grid correlation in dry races, with a neutralisation, and in the wet (with a borderline-wet subset) |

## Stage 2, part 5: trends and rule changes

```
python -m src.analysis.trends
explorer outputs\trends_and_rules
```

| File | What it shows |
| --- | --- |
| `season_measures.csv`, `what_decides_a_race_by_season.png` | six season-level measures since 1950: pole sitter wins, grid-vs-finish correlation, share of cars classified, top team's share of wins, share of races won from 4th or lower, different winners per race |
| `rule_change_tests.csv`, `rule_change_effects.png` | for each big rule change (1994, 2009-11, 2014, 2017, 2022; 2026 shown but not tested while the season is running), the change from the seasons before to the seasons after, set against the same change at years with no rule change. Windows of 2, 3 and 4 seasons; 3 is the main one |
| `rule_change_stability.csv` | do the other window lengths agree with the main one |
| `rule_change_trend_adjusted.csv` | robustness check: the after-period against the continuation of the 8-season pre-trend, so an already-trending measure does not look like a jump |
| `findings.txt` | the readable summary |

How to read the test: a change counts as `unusual` only if it clears three bars: its rank among the no-rule-change years, a false-discovery adjustment across all thirty comparisons (q-value), and the same z-score when only 1975-on years are the benchmark. Rule changes are not randomly assigned and several coincide with one team pulling ahead (Mercedes 2014, Red Bull 2022), so an unusual shift means "something changed then", not "the rules did it".

## Stage 2, part 6: scope, sprints and standings

```
python -m src.analysis.sprint_check
explorer outputs\sprint_check
```

Scope of parts 1-6: the Sunday Grand Prix only. Sprint results are loaded from stage 1 (2021 on) and analysed in part 7; points are not used, and the lap and tyre data is fetched for the race session only.

On a sprint weekend the Sunday grid is what the results table says it was. In 2021 and 2022 that was the finishing order of the Saturday sprint, so the grid "pole sitter" was the sprint winner, not the fastest qualifier. From 2023 the Sunday grid comes from Friday qualifying as usual. That leaves six races (Britain, Italy, Brazil 2021; Emilia-Romagna, Austria, Brazil 2022) with a different grid definition. `sprint_check` now takes them from the `sprint_results` table (seasons up to 2022), checks they match its hand-written list of six, and shows whether leaving them out changes the headline pole win rate or grid correlation.

Championship standings are not ingested. They are not needed for any stage 2 conclusion; if the dashboard later needs title context, add them then (Jolpica `driverStandings`).

## Stage 2, part 7: sprints

Needs the sprint results in the database first (about six small downloads, no FastF1):

```
python -m src.run_all --skip-fastf1
python -m src.analysis.sprints
python -m src.analysis.sprint_check
explorer outputs\sprints
```

| File in `outputs/sprints` | What it holds |
|---|---|
| `findings.txt` | The written summary |
| `sprint_races.csv`, `sprint_summary.csv` | One row per sprint; totals |
| `sprint_vs_sunday.csv` / `.png` | Each sprint against its own Sunday at the same share of distance |
| `sunday_race_measures.csv`, `sprint_weekend_effect.csv` / `.png` | Do Sundays on sprint weekends differ from other Sundays (circuit and season adjusted)? |
| `car_driver_with_sprints.csv` / `.png` | Car, driver and everything else with and without sprints |
| `teammate_agreement.csv` | Does the team-mate ahead in the sprint stay ahead on Sunday? |

What it found (details and caveats in `docs/stage2_findings.md`, section 7): a sprint keeps slightly more grid order than the same weekend's Sunday race (a lean, not clear after allowing for six comparisons); Sundays on sprint weekends look like other Sundays except for one borderline pole-win dip; in sprints the car explains 41% of finishing position against 28% on Sundays with the driver at 7% in both; and a team-mate order from one sprint is about as repeatable as one from a Sunday (58% against 59%).

The stage 2 write-up, with every headline number and its caveats, is in `docs/stage2_findings.md`.
