"""Keys, join cardinalities, derivability and cross-table consistency (full scans, small projections)."""
from common import P, TEL, con, save
c = con()
R = {}


def q(name, sql):
    df = c.execute(sql).df()
    R[name] = df.to_dict(orient="records")
    print(f"--- {name}\n{df.to_string(max_cols=40)[:3000]}")


DAILY = {"stoppoints": ("driver_bike_stoppoints", "local_date"), "income": ("driver_income_daily", "order_date"),
         "online": ("driver_online_hours_sap_id", "local_date"), "orders_rush": ("driver_orders_rush_hours", "local_date"),
         "stat": ("driver_statistic_daily", "local_date")}
# 1. key uniqueness
for k, (t, d) in DAILY.items():
    q(f"key_{k}", f"SELECT count(*) n, count(DISTINCT (driver_id,{d})) n_key, count(DISTINCT driver_id) drivers, count(DISTINCT {d}) AS n_days FROM {P(t)}")
q("key_profile", f"SELECT count(*) n, count(DISTINCT driver_id) n_drivers FROM {P('driver_profile')}")
q("key_trips", f"SELECT count(*) n, count(DISTINCT trip_id) n_trip, count(DISTINCT driver_id) drivers FROM {P('trips')}")
q("trip_id_dups", f"""SELECT k AS copies, count(*) trip_ids, sum(same_driver::INT) same_driver, sum(same_req::INT) same_request_time
  FROM (SELECT trip_id, count(*) k, count(DISTINCT driver_id)=1 same_driver, count(DISTINCT request_time)=1 same_req FROM {P('trips')} GROUP BY 1) GROUP BY 1 ORDER BY 1""")
q("key_ping", f"SELECT count(*) n, count(DISTINCT ping_id) n_ping FROM read_parquet('{TEL}')")
q("drivers_everywhere", f"""WITH p AS (SELECT DISTINCT driver_id FROM {P('driver_profile')}),
  t AS (SELECT DISTINCT driver_id FROM {P('trips')}), s AS (SELECT DISTINCT driver_id FROM {P('driver_statistic_daily')}),
  m AS (SELECT DISTINCT driver_id FROM read_parquet('{TEL}')), h AS (SELECT DISTINCT driver_id FROM {P('public_driver_hex_tracking')})
  SELECT (SELECT count(*) FROM p) profile, (SELECT count(*) FROM t) trips, (SELECT count(*) FROM s) daily, (SELECT count(*) FROM m) telemetry, (SELECT count(*) FROM h) hex,
   (SELECT count(*) FROM m WHERE driver_id NOT IN (SELECT driver_id FROM p)) tel_not_in_profile,
   (SELECT count(*) FROM t WHERE driver_id NOT IN (SELECT driver_id FROM m)) trips_not_in_tel,
   (SELECT count(*) FROM s WHERE driver_id NOT IN (SELECT driver_id FROM p)) daily_not_in_profile""")
# 2. profile internals
q("profile_xtab", f"SELECT vehicle_model, vehicle_class, seat_capacity, fleet, count(*) n FROM {P('driver_profile')} GROUP BY ALL ORDER BY ALL")
q("shift_window", f"""SELECT min(s) s_min, quantile_cont(s,[.1,.5,.9]) s_q, min(e) e_min, max(e) e_max, quantile_cont(e-s,[.1,.5,.9]) len_q
  FROM (SELECT json_extract(declared_shift_window,'$[0]')::INT s, json_extract(declared_shift_window,'$[1]')::INT e FROM {P('driver_profile')})""")
# 3. daily-table internal identities (fraction of rows where identity holds exactly)
q("daily_identities", f"""SELECT
  avg((o.total_order = o.total_order_normal_hour + o.total_order_rush_hour)::INT) ord_sum,
  avg((o.commission = o.commission_normal_hour + o.commission_rush_hour)::INT) com_sum,
  avg((o.total_fee = o.total_fee_normal_hour + o.total_fee_rush_hour)::INT) fee_sum,
  avg((abs(o.revenue_not_relate_driver - (o.revenue_not_relate_driver_normal_hour + o.revenue_not_relate_driver_rush_hour))<=1)::INT) rnr_sum,
  avg((abs(o.total_fee - o.commission - o.revenue_not_relate_driver)<=1)::INT) fee_eq_com_plus_rnr,
  avg((o.total_order=i.total_order AND o.commission=i.commission AND o.total_fee=i.total_fee AND o.revenue_not_relate_driver=i.revenue_not_relate_driver)::INT) orders_rush_eq_income,
  avg((i.total_order = s.completed_count)::INT) income_order_eq_completed,
  avg((s.accepted_count = s.total_request_calculate_complete)::INT) accepted_eq_req_complete,
  avg((s.accepted_count = s.completed_count + s.cancelled_count)::INT) accepted_eq_compl_plus_canc,
  avg((abs(i.avg_daily_revenue - i.total_fee/nullif(i.total_order,0))<1)::INT) avgrev_eq_fee_per_order,
  avg((abs(i.avg_daily_revenue - i.commission/nullif(i.total_order,0))<1)::INT) avgrev_eq_com_per_order,
  avg((abs(i.avg_daily_revenue - i.revenue_not_relate_driver/nullif(i.total_order,0))<1)::INT) avgrev_eq_rnr_per_order,
  avg((abs(s.fulfillment_rate - s.completed_count::DOUBLE/nullif(s.accepted_count,0))<1e-6)::INT) fulfil_eq_compl_over_acc,
  avg((abs(s.cancellation_rate - s.cancelled_count::DOUBLE/nullif(s.accepted_count,0))<1e-6)::INT) canc_eq_canc_over_acc,
  avg((i.total_core_order <= i.total_order)::INT) core_le_total,
  avg((st.total_stoppoints = s.completed_count)::INT) stop_eq_completed,
  avg((st.total_stoppoints_rush_hour = o.total_order_rush_hour)::INT) stoprush_eq_ordrush,
  avg((st.total_stoppoints_rush_hour <= st.total_stoppoints)::INT) stoprush_le_stop,
  corr(st.total_stoppoints, s.completed_count) corr_stop_completed,
  avg((on_.schedule_date = on_.local_date)::INT) sched_eq_local,
  corr(on_.online_time, s.completed_count) corr_online_completed,
  corr(on_.online_time, i.total_fee) corr_online_fee
  FROM {P('driver_orders_rush_hours')} o
  JOIN {P('driver_income_daily')} i ON i.driver_id=o.driver_id AND i.order_date=o.local_date
  JOIN {P('driver_statistic_daily')} s ON s.driver_id=o.driver_id AND s.local_date=o.local_date
  JOIN {P('driver_bike_stoppoints')} st ON st.driver_id=o.driver_id AND st.local_date=o.local_date
  JOIN {P('driver_online_hours_sap_id')} on_ ON on_.driver_id=o.driver_id AND on_.local_date=o.local_date""")
q("stat_rate_formulas", f"""SELECT
  avg((abs(acceptance_rate - accepted_count::DOUBLE/nullif(total_request_calculate_accept,0))<1e-6)::INT) acc_eq_acc_over_reqacc,
  avg((abs(cancellation_rate - total_request_calculate_cancel::DOUBLE/nullif(total_request_calculate_complete,0))<1e-6)::INT) canc_eq_reqcanc_over_reqcomp,
  avg((abs(fulfillment_rate - completed_count::DOUBLE/nullif(total_request_calculate_complete,0))<1e-6)::INT) ful_eq_comp_over_reqcomp,
  avg((total_request_calculate_cancel = cancelled_count + count_cancel_not_relate_driver)::INT) reqcanc_eq_canc_plus_notrel,
  avg((count_rating_5_star <= total_order_rating AND total_order_rating <= completed_count)::INT) rating_order_ok,
  avg((completed_count=0)::INT) zero_days
  FROM {P('driver_statistic_daily')}""")
q("zero_activity_days", f"""SELECT avg((s.completed_count=0)::INT) frac_zero_completed, avg((on_.online_time<=3.0)::INT) frac_online_at_floor,
  avg((s.completed_count=0 AND on_.online_time<=3.0)::INT) frac_both
  FROM {P('driver_statistic_daily')} s JOIN {P('driver_online_hours_sap_id')} on_ USING (driver_id, local_date)""")
# 4. trips internals
TR = f"""(SELECT *, request_time::TIMESTAMPTZ rq, assign_time::TIMESTAMPTZ asg, pickup_time::TIMESTAMPTZ pk, complete_time::TIMESTAMPTZ cp,
          substr(request_time,1,10) d, substr(request_time,12,2)::INT hr FROM {P('trips')})"""
q("trips_tz", f"SELECT count(*) FILTER (WHERE request_time NOT LIKE '%+07:00') rq_not07, count(*) FILTER (WHERE complete_time NOT LIKE '%+07:00') cp_not07 FROM {P('trips')}")
q("trips_order", f"""SELECT status, count(*) n,
  avg((asg>=rq)::INT) asg_ge_rq, avg((pk>=asg)::INT) pk_ge_asg, avg((cp>=pk)::INT) cp_ge_pk,
  avg((pk IS NULL)::INT) pk_null, avg((abs(epoch(cp)-epoch(pk)-duration_seconds)<=1)::INT) dur_eq_cp_minus_pk,
  quantile_cont(epoch(asg)-epoch(rq),[.05,.5,.95]) wait_assign_s, quantile_cont(epoch(pk)-epoch(asg),[.05,.5,.95]) to_pickup_s,
  quantile_cont(duration_seconds,[.05,.5,.95]) dur_s, quantile_cont(distance_km,[.05,.5,.95]) dist_km,
  avg((distance_km=0)::INT) dist_zero, avg((gross_vnd=0)::INT) gross_zero,
  quantile_cont(distance_km/nullif(duration_seconds,0)*3600,[.05,.5,.95]) avg_speed_kmh
  FROM {TR} GROUP BY 1""")
q("trips_cancel", f"SELECT cancelled_by, cancel_reason, count(*) n, avg((pk IS NULL)::INT) pk_null, avg((distance_km>0)::INT) dist_pos, avg((gross_vnd>0)::INT) gross_pos FROM {TR} WHERE status='cancelled' GROUP BY ALL ORDER BY ALL")
q("rush_vs_policy_hours", f"SELECT rush_hour, hr IN (6,7,8,16,17,18) policy_peak, count(*) n FROM {TR} GROUP BY ALL ORDER BY ALL")
q("rush_hours_by_hour", f"SELECT hr, avg(rush_hour::INT) rush_frac, count(*) n FROM {TR} GROUP BY 1 ORDER BY 1")
q("fare_check", f"""SELECT t.vehicle_class, count(*) n, corr(gross_vnd, distance_km) corr_gross_dist,
  regr_slope(gross_vnd, distance_km) slope_vnd_per_km, regr_intercept(gross_vnd, distance_km) intercept,
  quantile_cont(commission_vnd::DOUBLE/nullif(gross_vnd,0),[.05,.5,.95]) com_share,
  avg((t.vehicle_class=p.vehicle_class)::INT) class_matches_profile
  FROM {TR} t JOIN {P('driver_profile')} p USING (driver_id) WHERE status='completed' GROUP BY 1 ORDER BY 1""")
q("fare_rush", f"SELECT rush_hour, regr_slope(gross_vnd, distance_km) slope, regr_intercept(gross_vnd, distance_km) icpt, median(gross_vnd) med FROM {TR} WHERE status='completed' AND vehicle_class='A_COMPACT' GROUP BY 1")
q("h3_shapes", f"SELECT length(pickup_h3) l, count(*) n FROM {P('trips')} GROUP BY 1 ORDER BY 1")
# 5. trips vs daily aggregates
q("trips_vs_daily", f"""WITH t AS (SELECT driver_id, d, count(*) FILTER (WHERE status='completed') comp, count(*) FILTER (WHERE status='cancelled') canc,
     count(*) FILTER (WHERE status='cancelled' AND cancelled_by='driver') canc_drv,
     count(*) FILTER (WHERE status='completed' AND rush_hour) comp_rush,
     sum(gross_vnd) FILTER (WHERE status='completed') gross, sum(commission_vnd) FILTER (WHERE status='completed') com, sum(gross_vnd) gross_all, sum(commission_vnd) com_all
     FROM {TR} GROUP BY 1,2)
  SELECT count(*) n_driver_days_with_trips,
   avg((t.comp = s.completed_count)::INT) comp_eq, avg((t.canc = s.cancelled_count)::INT) canc_eq, avg((t.canc_drv = s.cancelled_count)::INT) cancdrv_eq,
   avg((t.comp+t.canc = s.accepted_count)::INT) all_eq_accepted, avg((t.comp_rush = o.total_order_rush_hour)::INT) rush_eq,
   avg((t.gross = i.total_fee)::INT) gross_eq_fee, avg((t.com = i.commission)::INT) com_eq_commission,
   avg((t.gross_all = i.total_fee)::INT) grossall_eq_fee, corr(t.gross, i.total_fee) corr_gross_fee, corr(t.com, i.commission) corr_com,
   median(i.total_fee - t.gross) med_fee_minus_gross, median(i.commission - t.com) med_com_diff, avg(t.comp - s.completed_count) mean_comp_diff
  FROM t JOIN {P('driver_statistic_daily')} s ON s.driver_id=t.driver_id AND s.local_date=t.d
  JOIN {P('driver_income_daily')} i ON i.driver_id=t.driver_id AND i.order_date=t.d
  JOIN {P('driver_orders_rush_hours')} o ON o.driver_id=t.driver_id AND o.local_date=t.d""")
# 6. telemetry <-> hex tracking
q("hex_vs_tel_daily_counts", f"""WITH a AS (SELECT substr(occurred_at,1,10) d, count(*) n FROM read_parquet('{TEL}') GROUP BY 1),
  b AS (SELECT substr(created_at,1,10) d, count(*) n FROM {P('public_driver_hex_tracking')} GROUP BY 1)
  SELECT count(*) AS n_days, sum((a.n=b.n)::INT) AS n_days_equal, sum(a.n) tel, sum(b.n) hex FROM a FULL JOIN b USING (d)""")
q("hex_internal", f"""SELECT avg((init_hex IS NOT DISTINCT FROM last_hex)::INT) init_eq_last, avg((current_hex = last_hex)::INT) cur_eq_last,
  avg((abs(epoch(last_seen_at::TIMESTAMPTZ)-epoch(entered_current_hex_at::TIMESTAMPTZ)-stay_duration_seconds)<=1)::INT) stay_eq_seen_minus_entered,
  avg((created_at=last_seen_at)::INT) created_eq_seen, avg((updated_at=last_seen_at)::INT) updated_eq_seen,
  avg((campaign_id IS NOT NULL)::INT) has_campaign, count(DISTINCT campaign_id) campaigns, count(DISTINCT target_hex) targets
  FROM {P('public_driver_hex_tracking')}""")
q("hex_status", f"SELECT tracking_status, count(*) n, quantile_cont(stay_duration_seconds,[.5,.9,.99]) stay_q FROM {P('public_driver_hex_tracking')} GROUP BY 1")
save("02_relations.json", R)
