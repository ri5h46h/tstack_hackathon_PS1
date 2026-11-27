# Validation against actual bus trips

Data: 13 day(s) (2026-10-20, 2026-10-25, 2026-10-30, 2026-10-31, 2026-11-06, 2026-11-07, 2026-11-09, 2026-11-10, 2026-11-18, 2026-11-19, 2026-11-20, 2026-11-21, 2026-11-23). Held-out by day: each case is scored by a table built without that group. Real time = arrival at stop B minus arrival at stop A on a real bus run.

## Overall

| method | cases | MAE (min) | median abs error (min) | MAPE % | median APE % | bias % | within 20 % |
|---|---|---|---|---|---|---|---|
| Ours (time-of-day table) | 328,129 | 2.50 | 1.20 | 19.0 | 14.4 | +0.3 | 64 % |
| All-day table (no time-of-day) | 328,129 | 2.63 | 1.18 | 19.2 | 14.3 | -3.5 | 63 % |
| GTFS timetable | 314,320 | 3.68 | 2.07 | 30.1 | 24.5 | +5.1 | 42 % |

Timetable is available for 314,320 of 328,129 cases. On those same cases:

| method | MAPE % |
|---|---|
| Ours (time-of-day table) | 18.9 |
| All-day table (no time-of-day) | 19.1 |
| GTFS timetable | 30.1 |

## By trip length

| trip length | cases | ours MAPE % | allday MAPE % | timetable MAPE % |
|---|---|---|---|---|
| 3-4 stops | 134,637 | 21.5 | 21.4 | 34.3 |
| 5-9 stops | 133,966 | 18.1 | 18.5 | 29.3 |
| 10-19 stops | 47,804 | 15.4 | 16.0 | 23.6 |
| 20+ stops | 11,722 | 14.0 | 14.6 | 18.9 |

## By time of day

| time bucket | cases | ours MAPE % | allday MAPE % | timetable MAPE % |
|---|---|---|---|---|
| midday | 119,822 | 20.4 | 18.1 | 30.0 |
| morning | 112,442 | 16.9 | 18.1 | 28.1 |
| night | 57,216 | 18.4 | 19.6 | 32.9 |
| pm_peak | 38,649 | 21.4 | 25.2 | 32.3 |

## By hour

| hour (IST) | cases | ours MAPE % | allday MAPE % | timetable MAPE % |
|---|---|---|---|---|
| 7 | 13,904 | 22.5 | 26.9 | 32.1 |
| 8 | 15,529 | 16.4 | 19.5 | 28.0 |
| 9 | 30,990 | 15.0 | 16.0 | 26.9 |
| 10 | 52,019 | 16.8 | 16.6 | 27.8 |
| 11 | 47,223 | 20.1 | 16.4 | 28.8 |
| 12 | 36,663 | 19.8 | 17.6 | 29.8 |
| 13 | 9,036 | 21.4 | 16.5 | 30.8 |
| 14 | 14,039 | 21.2 | 22.4 | 32.3 |
| 15 | 12,861 | 21.3 | 21.9 | 32.4 |
| 16 | 17,993 | 20.0 | 22.2 | 31.0 |
| 17 | 12,982 | 21.9 | 27.3 | 32.8 |
| 18 | 5,244 | 25.5 | 32.3 | 35.8 |
| 19 | 2,430 | 20.4 | 21.3 | 31.2 |
| 20 | 16,450 | 18.6 | 19.2 | 31.6 |
| 21 | 28,055 | 17.5 | 18.4 | 32.1 |
| 22 | 11,640 | 19.6 | 22.3 | 35.8 |
| 23 | 1,071 | 25.6 | 30.3 | 44.9 |

## Effect of data coverage

- cases where at least 80 % of the stretches had real stop-pair data: 324,627 (MAPE 18.9 %)
- cases with less coverage (speed fallback used): 3,502 (MAPE 28.9 %)

## Notes

- Real time includes time standing at stops and signals, because it runs from arrival at A to arrival at B.
- Stop pairs are sampled at random from real runs (about equal numbers per run), not from planner output.
- With one day of data, groups are sets of buses, so the same road conditions appear in learn and test data. Results on several days are the honest ones.
