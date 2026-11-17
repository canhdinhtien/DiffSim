# realdata-v1 EDA — basis for synthetic driver-tracking time series

Scripts and raw outputs: see [README.md](README.md). Evidence tags like `[02:trips_vs_daily]` mean "file `out/02_relations.json`, key `trips_vs_daily`". `03` = `out/03_tel_totals.json` / `03_tel_hist.json` / `03_tel_daily.parquet`, `04` = `04_states.json`, `05` = `05_patterns.json`, `06` = `06_extra.json`, `01` = `01_profile.json`, `00` = `00_schema.json`, `02b` = `02b_followups.json`.

## 1. Executive summary
1. **The reference data is simulated, not real.** `manifest.json` reads `"label": "MOCK"`, generator `gsm_core.mockgen.realdata v5`, seed 42. Every table's `source` is `MOCK`, except `public_frauds`, which is `INFERRED` `[01]`. Many fields follow exact formulas, so a model trained on them will learn the simulator's rules.
2. **The fleet is cars, not bikes.** `driver_type`, `vehicle_type`, `service_type` and `travel_mode` are all the constant `car`. The models are VinFast EVs: VF5, VF6, VF8 and VF_e34 `[01]`. `driver_bike_stoppoints` keeps "bike" only in its name.
3. **Coverage is complete.** 3,000 drivers appear on all 123 days (2026-07-01 → 2026-10-31) in every daily table, in trips and in telemetry. No day is missing, and every join on driver_id or date succeeds `[02:key_*, drivers_everywhere]`.
4. **The tables reconcile exactly.** Daily orders, rush orders and fees equal sums over trips on 100% of driver-days, and commission is within 1 VND per trip. Each telemetry `trip_id` is found in trips (100%) and falls inside [pickup, complete] (100%) `[02:trips_vs_daily, 02b:commission_diff, 04:tel_trip_link]`.
5. **About half of the shortlist is redundant or derivable.** I removed 34 fields (including all 8 shortlisted fields of `driver_income_daily`, so the whole table goes) and added 9. Section 2 has the reasons.
6. **Telemetry pings every 30–120 s for 95% of intervals** (median about 36 s; some days about 61 s). There are no out-of-order pings, 3,538 duplicate timestamps, and 1.9% of gaps longer than 5 min `[03]`.
7. **The simulator leaves visible artifacts:**
   - pings with speed 0 still move about 361 m between fixes `[04:stationary_jitter]`;
   - heading is exactly the bearing between consecutive fixes;
   - range is exactly battery × constant;
   - all 883 intra-day battery jumps coincide with the odometer going backwards by about 144 km `[06:odo_back_vs_charge]`.
8. **Driver states can be reconstructed** as on_trip / to_pickup / idle / offline from telemetry + trips. Hex dwell exists only in `public_driver_hex_tracking`: in telemetry, consecutive pings never share a res-9 hex `[04:tel_hex_dwell]`.
9. **Recommended MVP:** a generator hierarchy of driver-day plan → trip lifecycle → per-ping trajectory with odometer and battery. Daily aggregates are then computed, not generated.

## 2. Full inventory

| File | Size | Rows | Grain / key | In shortlist | Relevance |
|---|---|---|---|---|---|
| `telemetry/vehicle_telemetry_ping_2026-MM-DD.parquet` ×123 | 3,396 MB | 141,758,183 (722,610–1,372,395/day) | ping; `ping_id` unique | yes | core target |
| `public_driver_hex_tracking.parquet` | 8,053 MB | 141,758,183 | hex event; per-day count = telemetry on 123/123 days `[02:hex_vs_tel_daily_counts]` | yes | dwell / idle source |
| `trips.parquet` | 486 MB | 5,685,443 | `trip_id` unique, 1 driver each `[02:trip_id_dups]` | yes | core target |
| `driver_statistic_daily.parquet` | 6.3 MB | 369,000 | (driver_id, local_date) unique; 3000×123 | yes | constraint |
| `driver_orders_rush_hours.parquet` | 24.7 MB | 369,000 | same | yes | constraint |
| `driver_income_daily.parquet` | 12.8 MB | 369,000 | (driver_id, order_date) | yes | **redundant** |
| `driver_online_hours_sap_id.parquet` | 8.4 MB | 369,000 | (driver_id, local_date) | yes | target (online_time) |
| `driver_bike_stoppoints.parquet` | 2.8 MB | 369,000 | (driver_id, local_date) | yes | target (weak) |
| `driver_profile.parquet` | 0.02 MB | 3,000 | driver_id | yes | conditions |
| `policy_bundle.parquet` | 0.01 MB | 1 | — | **no → ADD** | fare / peak-hour / battery rules |
| `public_frauds.parquet` | 0.36 MB | 6,179 | fraud_id; detected daily at 18:00 | **no → ADD (labels)** | anomaly labels |
| `public_mission_earn_history.parquet` | 23.8 MB | 446,570 | mission payout | no | low: incentive payouts only |
| `public_user_mission_progress.parquet` | 0.08 MB | 6,883 | driver × mission | no | low |
| `public_mission.parquet` | 0.01 MB | 6 | mission | no | low |
| `kpi_driver_platform_calculator_gbq.parquet` | 0.5 MB | 54,000 | driver × ISO week (18 weeks) | no | low: weekly status only |
| `driver_penalization_ATA.parquet` | 0 | 0 | empty, no schema | no | none |
| `manifest.json`, `telemetry/_hw_state.json` | — | — | generator metadata; end-of-period vehicle state per driver | — | context only. `_hw_state.json` holds IDs and coordinates, so it is sensitive |

**Timestamps.** All are ISO strings with `+07:00`: 0 exceptions in trips and telemetry `[02:trips_tz, 03 n_not_07]`. Dates are `YYYY-MM-DD` strings.

**Join paths and cardinalities.**
- `driver_profile` 1:N every table.
- The 5 daily tables are 1:1:1:1:1 on (driver_id, date), dense with 369,000 rows each.
- trips N:1 driver-day: 368,912 driver-days have at least 1 trip `[02:trips_vs_daily]`.
- telemetry N:1 trips via `trip_id`: completed trips have a median of 15 pings, and 99.9% have pings. Cancelled trips have 0 pings `[04:trip_coverage]`.
- hex_tracking ↔ telemetry: same daily counts, but only 26–55% of rows match on (driver_id, timestamp). Matched rows agree on hex 99.95–99.997% `[04:hex_tracking_join_tstz]`.
- H3 resolutions: telemetry and hex_tracking use res 9. Trips `pickup_h3`/`drop_h3` use **res 7**. 97.5–98.1% of pings at pickup time lie inside `pickup_h3` `[04:pickup_h3]`.
- frauds N:1 driver-day.

Per-table schemas, with null %, distinct counts and quantiles for every column, are in `out/01_profile.json`. Sensitive columns are masked there.

## 3. Shortlist audit
Null % and distinct counts come from `[01]`; identities come from `[02]`/`[02b]`/`[06]`. Distinct counts for IDs and dates are HyperLogLog approximations (`approx_distinct`); exact counts are in `[02:key_*]`. "100%" means the identity held on every row checked.

### driver_bike_stoppoints
| Field | Profile | Verdict | Reason |
|---|---|---|---|
| driver_id | 0% null, 3,000 exact | KEEP (id) | join key to all tables |
| local_date | 0% null, 123 days, string | FIX | cast to DATE |
| total_stoppoints_rush_hour | 0–35, median 6 | KEEP (target, low priority) | not derivable: equals `total_order_rush_hour` on only 36.6% of rows, corr 0.81 `[02:daily_identities, 02b:stoppoints_rel]` |
| **total_stoppoints** (new) | 0–80, median 17 | ADD | needed for the `rush ≤ total` constraint (holds 100%). About 1.25 × completed, corr 0.95. Meaning is unclear (open question) |

### driver_income_daily → REMOVE whole table
| Field | Verdict | Reason |
|---|---|---|
| driver_id, order_date | REMOVE | keys of a removed table |
| commission, total_order, total_fee, revenue_not_relate_driver | REMOVE | identical to `driver_orders_rush_hours` on 100% of rows `[02:orders_rush_eq_income]` |
| avg_daily_revenue | REMOVE | = total_fee / total_order on 100% of rows. It is per order, not a daily average (name misleading) |
| total_core_order | REMOVE | ≈ 0.90 × total_order in every vehicle class (0.900–0.901), corr 0.99 `[02b:core_by_class]`. No temporal signal of its own; meaning is an open question |

### driver_online_hours_sap_id
| Field | Profile | Verdict | Reason |
|---|---|---|---|
| local_date | 123 days | KEEP (id) | — |
| schedule_date | — | REMOVE | = local_date on 100% of rows |
| driver_id | — | KEEP (id) | — |
| driver_type | constant `car` | REMOVE | constant |
| online_time | 3.0–26.82 h, median 9.14 | FIX → target | 59 rows exceed 24 h, 5,751 exceed 16 h, and the minimum sits at a 3.0 h floor `[06:online_gt24]`. Corr 0.90 with telemetry active hours, but about 3.9 h larger (median) and 0.9 h larger than the first-to-last-ping span `[05:tel_vs_daily]`. Definition is an open question |
| (not added) full_name, phone_number, sap_profile_id | PII | drop | privacy. `phone_number` has a single distinct value |

### driver_orders_rush_hours
| Field | Verdict | Reason |
|---|---|---|
| driver_id, local_date | KEEP (id) | — |
| total_order | KEEP (constraint) | = count of completed trips on 100% of rows, and = `completed_count` |
| commission | KEEP (constraint) | = Σ trip commission ±1 VND per trip `[02b:commission_diff]` |
| total_fee | KEEP (constraint) | = Σ gross_vnd of completed trips (100%) |
| revenue_not_relate_driver | REMOVE | = total_fee − commission (100%) |
| total_order_normal_hour, commission_normal_hour, total_fee_normal_hour, revenue_not_relate_driver_normal_hour | REMOVE | = total − rush (100%) |
| total_order_rush_hour | KEEP (constraint) | = completed trips with rush_hour (100%) |
| commission_rush_hour, total_fee_rush_hour | KEEP (constraint) | rush split of the money fields (sums hold 100%) |
| revenue_not_relate_driver_rush_hour | REMOVE | = fee − commission |

### driver_profile
| Field | Verdict | Reason |
|---|---|---|
| driver_id | KEEP (id) | 3,000 unique |
| vehicle_model | KEEP (condition) | 4 values (1500/600/540/360). Strong effects: trips/day 8.3–22.0 and fee/day 0.93–3.5 M VND across model × fleet `[05:vehicle_effects]` |
| seat_capacity | REMOVE | determined by model (VF5 = 4, others = 5), 100% `[02:profile_xtab]` |
| **fleet** (new) | ADD (condition) | station_charge 2,308 / depot_charge 692. It changes behaviour: VF5 depot drivers make 8.3 trips/day on 5.1 h online, against 12.3 trips on 8.8 h for station drivers. Battery < 10% occurs only in station_charge fleets `[06:battery_low]` |
| **declared_shift_window** (new) | ADD (condition), FIX | JSON `[start_min, end_min]`. Start hour correlates 0.997 with the observed first-ping hour, end hour 0.98 with the last `[05:shift_vs_actual]`. It drives the bimodal shift structure |
| (dropped) vehicle_class | — | 1:1 with vehicle_model. Kept only as a lookup key for policy rules |
| (dropped) track, tenure_months, tier, home_zone_h3r8 | — | constant (`core_owned`, 12) or 100% null |

### driver_statistic_daily
| Field | Verdict | Reason |
|---|---|---|
| local_date, driver_id | KEEP (id) | — |
| completed_count | KEEP (constraint) | = completed trips (100%) |
| accepted_count | REMOVE | = completed_count + cancelled_count (100%) |
| cancelled_count | FIX (constraint) | equals the trip-derived cancel count on only 90.2% of driver-days, and the driver-cancel count on 61% `[02:trips_vs_daily]`. Its definition is unresolved |
| total_request_calculate_complete | REMOVE | = accepted_count (100%) |
| **total_request_calculate_accept** (new) | ADD (target) | only field carrying the number of offered requests: mean 15.9 vs 15.5 accepted. `acceptance_rate` matches accepted/offered on 76.6% of rows |
| (dropped) rates, ratings | — | `fulfillment_rate`/`cancellation_rate` are only partly derivable (54–58%). Ratings have no tracking relevance |

### public_driver_hex_tracking
| Field | Profile | Verdict | Reason |
|---|---|---|---|
| id | surrogate | REMOVE | no information |
| driver_id | — | KEEP (id) | — |
| init_hex | 0.26% null | REMOVE | = last_hex on 100% of rows |
| current_hex | res 9 | KEEP (target) | only source of hex dwell. In telemetry, consecutive pings change hex 99.997% of the time `[04:tel_hex_dwell]` |
| last_hex | — | REMOVE | previous hex; derivable from the sequence |
| last_seen_at | — | KEEP (target), FIX parse | — |
| entered_current_hex_at | — | KEEP (target) | — |
| stay_duration (actual name `stay_duration_seconds`) | 0–10,574 s; median 0 | FIX → constraint | = last_seen_at − entered_current_hex_at (100%). The shortlist name is wrong |
| created_at | — | REMOVE | ≠ last_seen_at on 47% of rows; ingestion time |
| updated_at | — | REMOVE | = last_seen_at (100%) |
| tracking_status | moving 98.27%, idle 1.73%, offline 2,348 rows | KEEP (target) | idle stays median 1,059 s, offline 6,267 s `[02:hex_status]`. Main stop/offline signal |

Not added: `campaign_id`/`target_hex`/`reached_target`. They are 95% null, with 1 campaign and 61 targets; usable later for repositioning studies.

### trips
| Field | Profile | Verdict | Reason |
|---|---|---|---|
| trip_id | unique | KEEP (id) | links telemetry |
| driver_id | — | KEEP (id) | — |
| service_type | constant `car` | REMOVE | constant |
| status | completed 94.98% / cancelled 5.02% | KEEP (target) | — |
| request_time | — | KEEP (target), FIX parse | — |
| assign_time | — | KEEP, FIX | 0.086% are earlier than request_time `[02:trips_order]` |
| pickup_time, complete_time | null iff cancelled | KEEP (target) | — |
| pickup_h3, drop_h3 | res 7, 922/1,105 cells | KEEP (target), FIX doc | coarser than telemetry (res 9) |
| distance_km | median 4.64 km for completed; 0 for cancelled | KEEP (target) | — |
| duration_seconds | — | REMOVE | = complete − pickup (100%) |
| gross_vnd | median 109k VND | KEEP (target) | ≈ linear in km: slope 18.8k (A) to 27.6k (D) VND/km, corr 0.85–0.86 |
| commission_vnd | — | REMOVE → constraint | = 0.35 × gross (A/B/C) or 0.40 × gross (D_LUXURY), p5–p95 range 0.34999–0.35001 `[02:fare_check]` |
| rush_hour | 33.5% true | REMOVE → constraint | = hour(complete_time) ∈ {6,7,8,16,17,18} on 100% of completed trips `[04:rush_ref_complete]`. Matches `policy_bundle.points.peak_hours` |
| cancel_reason | 3 values | KEEP (target) | — |
| cancelled_by | — | REMOVE | determined by reason: `khach_doi_y` → customer, `su_co_xe`/`khach_khong_den` → driver `[02:trips_cancel]` |
| (dropped) customer_id, vehicle_class, travel_mode | — | PII / = profile class (100%) / constant | — |

### telemetry
| Field | Profile | Verdict | Reason |
|---|---|---|---|
| driver_id | — | KEEP (id) | — |
| occurred_at | — | KEEP, FIX parse | 0 out-of-order rows; 3,538 duplicate (driver, ts) pairs |
| lat, lng | — | KEEP (target) | sensitive |
| current_hex_h3_res9 | — | REMOVE → constraint | = `latlng_to_cell(lat, lng, 9)` on 100% of 600k sampled pings `[04:h3_consistency]` |
| speed_kmh | 0–90 | KEEP, FIX | speed = 0 does not mean stationary (about 361 m moved). Reported vs implied speed corr is 0.95 on trip but 0.39–0.89 when idle `[04:speed_implied_by_state]` |
| heading_deg | 0.002% null | REMOVE → constraint | = bearing between consecutive fixes: median error 0.023°, p90 0.045° on all 123 days `[03]` |
| odometer_km | — | KEEP (target) | monotone across days (100%) `[06:cross_day_continuity]`. 1,196 backward steps (883 at charge events) |
| battery_level_pct | — | KEEP (target) | drain/km matches the policy kWh rates (section 4.3) |
| remaining_range_km | — | REMOVE → constraint | = battery_pct × battery_kwh / kwh_per_100km per model, R² = 1.0 `[06:range_vs_battery_by_model]` |
| seat_occupancy/driver | always true | REMOVE | constant |
| seat_occupancy/any_passenger_seat | — | REMOVE → constraint | = (trip_id IS NOT NULL), 0 mismatches over 141.8 M pings `[03]` |
| **trip_id** (new) | 38.7% null in sample | ADD (target / link) | joins pings to trips (100% found) and enables state reconstruction |
| **gear** (new, optional) | D / P | ADD | parking signal. Every speed-0 ping has gear P, but 26% of P pings have speed > 0 (FIX). `charging_status` is identical (P ⇔ IDLE) and is dropped |
| **door_lock_status** (new, optional) | 13.7% UNLOCKED | ADD | unlocked on every speed-0 ping; a weak stop/boarding signal |
| (dropped) phone_connectivity_lost, trunk_status, charging_status, source_device, vin, ping_id | — | constant / duplicate / PII / surrogate | `phone_connectivity_lost` is always false, so there is **no GPS-loss label**. `charging_status` is never `CHARGING` |

### Added tables
- **policy_bundle** (ADD, constraint source). It holds the peak hours `[6,7,8,16,17,18]` that exactly define `rush_hour`, per-class `battery_kwh` and `kwh_per_100km` (which explain range and drain), and fare parameters. The fare and share parameters do *not* match the observed data: policy says 13,000 VND/km and `driver_share` 0.75, while the observed slope is about 18.8k VND/km and commission is 35% (open question).
- **public_frauds** (ADD, labels). All rows are `abnormal_cancel`. Flagged driver-days average 2.93 cancels and a 0.26 cancel rate, against 0.84 and 0.048 for unflagged days `[02b:frauds_vs_cancels]`. This gives evaluation labels for the cancellation-anomaly detector.

**Count:** 34 shortlisted fields removed: 8 income, 2 online-hours, 6 rush-hours, 1 profile, 2 statistic, 5 hex-tracking, 5 trips and 5 telemetry. The income table goes entirely. 9 added: `total_stoppoints`, `fleet`, `declared_shift_window`, `total_request_calculate_accept`, `telemetry.trip_id`, `gear`, `door_lock_status`, and the `policy_bundle` and `public_frauds` tables.

## 4. Quality and time-series findings

### 4.1 Telemetry sampling and quality (all 141.76 M pings) `[03]`
- **Inter-ping dt (141.39 M intervals):**

  | dt | share |
  |---|---|
  | < 30 s | 0.03% |
  | 30–60 s | 54.3% |
  | 60–120 s | 40.6% |
  | 2–5 min | 3.1% |
  | 5–30 min | 1.4% |
  | 30–60 min | 0.43% |
  | > 1 h | 61,654 intervals (shift breaks) |

  The per-day median dt is 35.4–64.5 s. Days such as 2026-07-15, 08-04, 08-11 and 09-08 have a median of about 61 s, so the sampling regime switches by day.
- **Gaps longer than 60 s** are normal sampling, not loss. Gaps longer than 5 min number 2.66 M and are treated as offline segments.
- **Order and duplicates:** 0 out-of-order rows (file order equals time order); 3,538 duplicate timestamps.
- **GPS jumps:** 14,321 intervals imply more than 150 km/h, and 1,539 move more than 2 km within 60 s (0.01%).
- **Speed:** maximum 90 km/h on every day. 0.12% of pings exceed 80 km/h, and 2.0% are 0.
- **Odometer:** Σ odometer delta is 51.81 M km against Σ haversine 51.95 M km (0.3% apart), but the per-step correlation is only 0.12. The odometer is right in aggregate and noisy per step.
- **Battery:**
  - it never rises gradually; there are 883 intra-day jumps, all of 10 points or more (476 of 30–60 points, 366 of 60+);
  - every intra-day jump coincides with an odometer drop of about 144 km (median), which looks like a simulator reset. **Treat these as artifacts, not real charging**;
  - real charging is overnight: 99.97% of drivers start the next day higher, with a median gain of 57.4 points and a start-of-day battery of p50 92.5% `[06:cross_day_continuity]`.
- **Heading** equals the position bearing (see the audit). **H3** matches 100%.
- **Hard braking and acceleration** (|Δv/Δt| > 3 m/s² with dt ≤ 30 s): 1,141 and 1,350 events. These are nearly undetectable because only 0.03% of intervals are shorter than 30 s.
- **Stops** (runs of speed = 0): 2.79 M runs last under 1 min (single pings). 7,543 last 5–15 min, 4,331 last 15–30 min, 2,179 last 30–60 min, and 559 last longer than 60 min.

### 4.2 Coverage
- 3,000 drivers have telemetry every day, and every telemetry driver is in profile and the daily tables.
- Per driver-day, p5 / p50 / p95 `[05:driverday_dist]`:

  | Measure | p5 | p50 | p95 |
  |---|---|---|---|
  | pings | 165 | 384 | 602 |
  | span (first to last ping) | 3.4 h | 8.0 h | 11.5 h |
  | active time (dt ≤ 5 min) | 2.5 h | 5.5 h | 8.5 h |
  | passenger time | 1.3 h | 3.2 h | 5.7 h |
  | km | 61 | 143 | 226 |

- Trips counted in telemetry equal `completed_count` on 98.9% of driver-days (corr 0.9999). Paid km are 51% of odometer km (median) `[05:tel_trip_km_vs_trips]`.
- **Shifts are bimodal.** The first ping falls in 05–10 h on 228,779 driver-days and in 15–18 h on 140,220 `[03:first_ping_hour]`. This follows `declared_shift_window`.

### 4.3 Seasonality, trend, holidays `[05]`
- **Hour of day:**
  - weekdays peak at 08 h (3,175 trips/day) and 17–18 h (about 3,625);
  - weekends shift later, peaking at 10 h (3,586) and 19 h (3,560).
- **Rush-hour slowdown:** median trip speed is 19.4–20.1 km/h in 06–08 and 16–18 h, against about 25.6 midday and 32–33 at 05 h and 21–23 h.
- **Supply squeeze in the evening peak:** assignment wait jumps to a median of 18.9 s at 18 h and 12.6 s at 19 h, against 3–5 s at other hours.
- **Day of week:** Saturday averages 52.2k trips/day and Tuesday 43.2k. Mean online time is 10.1 h on Saturday against 9.1–9.3 h on weekdays.
- **Trend:** +7.8 trips/day per day, about 2% over the period, which is negligible.
- **Holiday:** around the 2026-09-02 National Day, 09-02 had 53.8k trips and 09-03 had 57.5k (the maximum), against a median of 45.9k. The effect is mild.
- **Anomaly day:** 2026-07-15 had 27,044 trips, a median trip speed of 16.7 km/h, 61 s sampling and 25.7k completed trips. It looks like a scripted disruption (assumption: weather or another event).

### 4.4 Driver states `[04]`, three sample days
| State | Rule | Share of pings (07-15 / 08-02 / 10-20) | Median speed |
|---|---|---|---|
| on_trip | `trip_id` not null; 100% inside [pickup, complete] | 58.5% / 60.8% / 59.5% | 21 / 39 / 39 km/h |
| to_pickup | assign ≤ t < pickup (interval join with trips) | 16.5% / 22.6% / 21.0% | 20 / 36 / 36 km/h |
| idle (online, no job) | the rest of the online pings | 25.0% / 16.6% / 19.5% | 14 / 21 / 21 km/h; only 6–11% at speed 0 |
| offline | dt > 5 min, or outside first/last ping | 1.9% of intervals | — |

How reliable this is:
- `on_trip` is exact.
- `to_pickup` needs the trips join: 3.6–4.4% of these pings are already tagged with the trip_id.
- 7.7k–21.6k pings per day fall into two overlapping trip windows (chained assignments), so the next trip can be assigned before the current one completes.
- Cancelled trips have no pings at all, so a cancel only appears in the timeline as a to_pickup segment.
- "idle" is mostly cruising, not parking. Real stops come from `hex_tracking.tracking_status = idle` (stay median 1,059 s).

### 4.5 Cross-signal relations
- **Speed vs trip phase:** see the table in 4.4.
- **Battery drain vs distance** in %/km, observed against the policy value (`kwh_per_100km / battery_kwh`) `[05:battery_vs_km]`:

  | Model | Observed | Policy |
  |---|---|---|
  | VF5 | 0.381 | 0.389 |
  | VF6 | 0.348 | 0.357 |
  | VF8 | 0.219 | 0.222 |
  | VF_e34 | 0.364 | 0.369 |

  Drain is a deterministic function of km. No speed effect was tested beyond this.
- **Earnings vs online_time:** corr 0.78, about 271k VND and 1.86 trips per online hour `[05:online_vs_earn]`.
- **Cancellations:** flat across hours (4.9–5.1%; 6.1% at 23 h), day of week (5.0%) and area (res-7 cells p10–p90 4.5–5.5%). Space and time carry no cancel signal; cancels concentrate in drivers flagged by `public_frauds`.
- **stay_duration vs hex:** moving rows have a median stay of 0 s (p99 129 s); idle rows have a median of 1,059 s.

### 4.6 Rare events (frequency over the 123 days)
| Event | Count | Source |
|---|---|---|
| Stops of 15 min or longer (consecutive speed 0) | 7,069 | `[03:stop_run_len_min]` |
| Idle hex stays | 2.46 M rows | `[02:hex_status]` |
| Gaps of 30 min or longer (GPS loss / offline) | 672,660 | `[03]` |
| Position jumps above 150 km/h | 14,321 | `[03]` |
| Speeding above 80 km/h | 176,459 pings | `[03]` |
| Hard brake / acceleration | 1,141 / 1,350 | `[03]` |
| Intra-day battery jump (artifact) | 883 | `[06]` |
| Abnormal-cancel fraud flag | 6,179 driver-days | `[01]` |

Route deviation has **no reference data**: there are no planned routes, and `target_hex` covers only 5% of hex rows.

### 4.7 Heterogeneity
- Drivers differ more from each other than from day to day: between-driver variance of daily completed trips is 22.0, against 17.2 within a driver. Pooled ACF of completed trips is 0.57 at lag 1 and 0.60 at lag 7 (weekly) `[05:between_within_var, driver_autocorr]`.
- K-means with k = 4 on per-driver means (no clear elbow `[05:kmeans_inertia]`) recovers model × shift groups `[05:kmeans4_*]`:

  | Group | Drivers | Trips/day | Online | Revenue/trip | Rush share | Mean start |
  |---|---|---|---|---|---|---|
  | VF8 + VF_e34 | 896 | 18.9 | 10.8 h | 148k VND | — | 14 h |
  | VF5 early-morning | 677 | 12.1 | — | — | 0.46 | 05:16 |
  | VF5 afternoon, short shift | 420 | 8.3 | 5.1 h | — | — | 16:43 |
  | VF6 + VF5 day shift | 1,007 | 15.2 | 10.7 h | — | — | 07:16 |

- Since `driver_type` is constant, **vehicle_model, fleet and shift window are the only heterogeneity conditions**.

## 5. Proposed data catalog

### P1. Driver-day plan
1. **What:** one row per driver-day: online flag, shift start/end, online_time, number of completed trips, cancels, offers, stoppoints. **Purpose:** the top-level conditioning for every lower level.
2. **Structure:** driver_id (str), date (date), shift_start_min / shift_end_min (int, minutes), online_time_h (float), n_completed (int), n_cancelled (int), n_offered (int), total_stoppoints (int), total_stoppoints_rush (int). Sequence: 123 days per driver.
3. **Evidence:** `driver_statistic_daily`, `driver_online_hours_sap_id`, `driver_bike_stoppoints`, `driver_profile.declared_shift_window`. Supported.
4. **Depends on:** profile only.
5. **Method:** statistical / generative per driver. Because between-driver variance dominates and lag-7 ACF is 0.60, use a driver embedding plus a day-of-week term: for example a TimeVAE or Diffusion-TS over 123-day sequences, or a hierarchical negative binomial as a baseline.
6. **Conditions:** vehicle_model, fleet, shift window, day of week, holiday flag.
7. **Constraints:** online_time ≤ 24 (the source violates this 59 times); stoppoints_rush ≤ stoppoints; n_completed ≈ 1.86 × online_h.
8. **Evaluation:** marginals per model (KS test), ACF at lags 1 and 7, between/within variance ratio, train-on-synthetic-test-on-real (TSTR) regression of fee from online_time.
9. **Difficulty:** low. **Risk:** the online_time definition is unclear.

### P2. Trip lifecycle events
1. **What:** a trip table per driver-day: request → assign → pickup → complete, or a cancel, with H3 res-7 origin/destination, distance and fare. **Purpose:** ETA, dispatch and earnings simulation.
2. **Structure:** trip_id, driver_id, request_time / assign_time / pickup_time / complete_time (timestamp_tz), status, cancel_reason, pickup_h3 / drop_h3 (res 7), distance_km, gross_vnd. Derived: duration, commission, rush_hour, cancelled_by. About 15 trips per driver-day.
3. **Evidence:** `trips` (5.69 M rows) and `policy_bundle`. Supported.
4. **Depends on:** P1 (n_completed, n_cancelled, shift window).
5. **Method:** a temporal point process for request times (hour × weekend intensity), parametric distributions for the inter-event waits, an origin-destination matrix over the 925 res-7 cells, and a rule-based fare (gross ≈ a + b·km per class). A generative model is unnecessary at this level.
6. **Conditions:** hour, weekend, vehicle_class, pickup cell, rush.
7. **Constraints:**
   - Σ per driver-day = P1 counts, and Σ gross = total_fee (exact in the source);
   - commission = round(rate × gross), with rate 0.35 or 0.40 by class;
   - rush_hour = hour(complete) ∈ peak hours;
   - timestamps ordered;
   - trips of one driver do not overlap, except chained assignment;
   - cancelled trips have no pickup, distance 0 and gross 0.
8. **Evaluation:** distributions of wait / to-pickup / duration by hour; OD-matrix Jensen-Shannon divergence; fare residuals; daily reconciliation must be exact.
9. **Difficulty:** medium. **Risk:** OD flows at res 7 are coarse.

### P3. Telemetry ping trajectories
1. **What:** irregular pings with position, speed, odometer, battery, trip_id and gear. **Purpose:** location tracking, map matching, ETA.
2. **Structure:**
   - ts (timestamp_tz), lat / lng (float, degrees), speed_kmh (float), odometer_km (float), battery_pct (float), trip_id (str or null), gear (D/P), door (LOCKED/UNLOCKED);
   - derived: hex_r9, heading, range_km, passenger_flag;
   - dt between pings drawn from {about 36 s, about 61 s} by day; about 384 pings per driver-day.
3. **Evidence:** telemetry, 141.8 M rows. Supported, with artifacts.
4. **Depends on:** P2 (trip windows, pickup/drop cells) and P1 (shift span).
5. **Method:** hierarchical:
   - a rule-based skeleton (waypoints between cells along a road graph, assumption: needs OSM);
   - a generative model for the speed profile and the dt process per state, using Diffusion-TS or TimeGAN on segments of about 15–40 pings;
   - odometer and battery integrated deterministically.
6. **Conditions:** state, hour, rush, vehicle_model, fleet, battery level.
7. **Constraints:**
   - odometer monotone, with Δodometer ≈ haversine;
   - Δbattery = −Δkm × kWh/100km ÷ battery_kWh;
   - range = battery × k(model);
   - heading = bearing;
   - hex = latlng_to_cell;
   - passenger flag ⇔ trip_id;
   - pings with a trip_id lie in [pickup, complete].
8. **Evaluation:** dt histogram, speed by state, ACF of speed, dynamic time warping (DTW) distance between trajectories, discriminative score, km/day and pings/day distributions.
9. **Difficulty:** high. **Risks:** do not copy the simulator artifacts (speed 0 while moving, odometer resets). The real road network is not in the data.

### P4. Driver state timeline and hex dwell
1. **What:** a segment sequence (offline / idle / to_pickup / on_trip) with start, end and hex, plus hex-dwell events. **Purpose:** state tracking, utilisation analytics.
2. **Structure:** driver_id, state, t_start, t_end, hex_r9, stay_s, tracking_status. About 40–60 segments per driver-day (assumption, from about 15 trips × 3 states).
3. **Evidence:** telemetry ⋈ trips (section 4.4) and `hex_tracking` (dwell and idle/offline). Supported.
4. **Depends on:** P1 and P2. P3 can be produced from it.
5. **Method:** a semi-Markov model whose durations depend on state and hour, fitted to the reconstructed segments. Alternatively, generate it directly as the middle layer between P2 and P3.
6. **Conditions:** hour, rush, model, shift.
7. **Constraints:** segments tile the online span; on_trip segments equal P2 trips; stay = last_seen − entered.
8. **Evaluation:** state-duration distributions, transition matrix, utilisation (passenger hours / active hours, median 3.2 / 5.5).
9. **Difficulty:** medium. **Risk:** the meaning of the hex-tracking idle/offline rows is unclear (open question).

### P5. Daily and rush-hour aggregates
- **What:** daily aggregates derived from P2: orders, fee, commission and their rush splits, cancels.
- **Method:** deterministic computation from P2, not generated. This is the consistency check for the whole hierarchy.
- **Evidence:** the daily tables reconcile with trips at 100%. Supported.
- **Difficulty:** low.

### P6. Behaviour, anomaly and fault injection
- **What:**
  - long stops (≥ 15 min, 7k observed) and idle stays;
  - speeding (> 80 km/h);
  - GPS jumps (0.01%) and duplicate pings (3.5k);
  - gaps / GPS loss (≥ 30 min, 0.47% of intervals);
  - abnormal-cancel drivers (labels from `public_frauds`).
- **Purpose:** robustness testing and anomaly-detector training with ground-truth labels.
- **Evidence:** frequencies in section 4.6. There is **no reference data** for route deviation or for real GPS-loss labels (`phone_connectivity_lost` is always false).
- **Method:** rule-based injection at the observed rates, on top of P3.
- **Difficulty:** low to medium. **Risk:** injected faults are not real faults.

## 6. Final field spec, priorities, roadmap
The field-level spec is in [final_field_spec.yaml](final_field_spec.yaml): roles identifier / condition / target / constraint / drop.

| Rank | Proposal | Value | Feasibility | Notes |
|---|---|---|---|---|
| 1 | P2 trip lifecycle | high | high | exact reconciliation gives a hard check |
| 2 | P1 driver-day plan | high | high | small and fast to fit (369k rows) |
| 3 | P4 state timeline and dwell | high | medium | glue between trips and pings |
| 4 | P3 ping trajectories | very high | medium–low | needs a road network; avoid the artifacts |
| 5 | P5 aggregates | medium | trivial | derived |
| 6 | P6 anomalies | medium | medium | labels only for cancels |

**Roadmap:**
1. **MVP.** Build P1 + P2 for one vehicle model (VF5 station_charge, 1,080 drivers), with derived fields computed by rule. Accept it when P5 reconciles exactly and the hour/weekend distributions match (KS test).
2. Extend to all model × fleet groups, adding holiday and day-of-week effects.
3. Build P4 as a semi-Markov layer, and validate it against the reconstructed states on the 3 sample days and then all days.
4. Build P3 with the rule-based path, add the generative speed and dt model (Diffusion-TS from `src/`), and integrate odometer and battery.
5. Add P6 injection and an evaluation suite (discriminative score, TSTR). `utils/` already has a discriminative score and context-FID.

## 7. Risks, assumptions, open questions
**Risks**
- The "real" data is a simulator (MOCK). Distributions reflect simulator rules, for example a speed cap of exactly 90 and fixed commission rates. A generator fitted here will reproduce the simulator, not Grab/Be reality.
- The artifacts listed in section 1 must be cleaned or excluded before training P3.
- `public_driver_hex_tracking` is 8 GB. Only column-projected scans are feasible.
- Privacy: `_hw_state.json`, `full_name`, `phone_number`, `sap_profile_id`, `driver_name`, `vehicle_license_plate` and coordinates are sensitive. None of them is needed for generation, except coordinates.

**Assumptions**
- Gaps longer than 5 min are offline. The threshold is a choice, not a documented rule.
- The hard-brake threshold of 3 m/s² is a choice.
- 2026-07-15 is a scripted disruption day.
- Intra-day battery jumps are reset artifacts.
- Hex-tracking idle/offline rows are emitted on stops rather than at ping times: no idle row matched a ping timestamp `[04:hex_tracking_join_tstz]`.

**Open questions**
1. ~~Is this mock acceptable as the reference?~~ **Resolved 2026-11-17: yes, use the mock data as the reference.**
2. ~~Bikes or cars?~~ **Resolved 2026-11-17: target the car fleet (VF5/VF6/VF8/VF_e34) as in the data.**
3. **online_time:** what does it measure? It is about 3.9 h above telemetry active time, it has a 3.0 h floor, and 59 values exceed 24 h.
4. **total_stoppoints:** what is a stop point? It is about 1.25 × completed trips, and its rush variant ≠ rush orders.
5. **cancelled_count** matches trip cancels on only 90% of driver-days. Which definition is authoritative (driver vs customer cancels, cancel before vs after accept)?
6. **total_core_order:** what is a "core" order? It is a constant 90% of orders.
7. **Policy vs data:** policy says 13k VND/km and `driver_share` 0.75, but the data shows about 18.8k VND/km and 35–40% commission. Which applies to synthetic fares?
8. **Rush hour:** should the flag be defined by complete_time, as the data does, or by request time?
9. Should P3 snap to a real road network (OSM Hanoi), or is cell-level movement enough?
