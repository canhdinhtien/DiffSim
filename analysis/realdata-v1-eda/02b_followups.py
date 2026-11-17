"""Follow-ups on mismatches found by 02_relations.py (commission, cancels, stoppoints, rush window, hex<->ping mapping)."""
from common import P, ROOT, con, save
c = con()
R = {}


def q(name, sql):
    df = c.execute(sql).df()
    R[name] = df.to_dict(orient="records")
    print(f"--- {name}\n{df.to_string(max_cols=40)[:2500]}")


TR = f"""(SELECT *, substr(request_time,1,10) d, substr(request_time,12,2)::INT hr, substr(request_time,15,2)::INT mi,
          substr(assign_time,12,2)::INT ahr, substr(pickup_time,12,2)::INT phr FROM {P('trips')})"""
TD = f"""(SELECT driver_id, d, count(*) FILTER (WHERE status='completed') comp, count(*) FILTER (WHERE status='cancelled') canc,
     count(*) FILTER (WHERE cancelled_by='driver') canc_drv, count(*) FILTER (WHERE cancelled_by='customer') canc_cust,
     sum(commission_vnd) FILTER (WHERE status='completed') com, sum(gross_vnd) FILTER (WHERE status='completed') gross,
     count(*) FILTER (WHERE status='completed' AND vehicle_class<>'D_LUXURY') comp_nonlux
     FROM {TR} GROUP BY 1,2)"""
q("commission_diff", f"""SELECT quantile_cont(i.commission - t.com,[0,.01,.25,.5,.75,.99,1]) diff_q, avg((abs(i.commission-t.com)<=t.comp)::INT) within_1vnd_per_trip
  FROM {TD} t JOIN {P('driver_income_daily')} i ON i.driver_id=t.driver_id AND i.order_date=t.d""")
q("cancel_diff", f"""SELECT s.cancelled_count - t.canc diff, s.count_cancel_not_relate_driver = t.canc_cust cust_match, count(*) n
  FROM {TD} t JOIN {P('driver_statistic_daily')} s ON s.driver_id=t.driver_id AND s.local_date=t.d GROUP BY ALL ORDER BY n DESC LIMIT 8""")
q("cancel_req_fields", f"""SELECT avg((s.total_request_calculate_cancel = t.canc)::INT) reqcanc_eq_trips_canc,
  avg((s.count_cancel_not_relate_driver = t.canc_cust)::INT) notrel_eq_cust, avg((s.cancelled_count = t.canc_drv)::INT) canc_eq_drv,
  avg((s.total_request_calculate_accept - s.accepted_count)) mean_accept_req_minus_acc
  FROM {TD} t JOIN {P('driver_statistic_daily')} s ON s.driver_id=t.driver_id AND s.local_date=t.d""")
q("stoppoints_rel", f"""SELECT avg(st.total_stoppoints - s.completed_count) mean_diff, quantile_cont(st.total_stoppoints - s.completed_count,[.01,.5,.99]) diff_q,
  avg(st.total_stoppoints_rush_hour - o.total_order_rush_hour) mean_rush_diff, corr(st.total_stoppoints_rush_hour, o.total_order_rush_hour) corr_rush,
  regr_slope(st.total_stoppoints, s.completed_count) slope, regr_intercept(st.total_stoppoints, s.completed_count) icpt
  FROM {P('driver_bike_stoppoints')} st JOIN {P('driver_statistic_daily')} s USING (driver_id, local_date)
  JOIN {P('driver_orders_rush_hours')} o USING (driver_id, local_date)""")
q("core_order", f"""SELECT avg(i.total_order - i.total_core_order) mean_noncore, avg((i.total_core_order = t.comp_nonlux)::INT) core_eq_nonlux,
  corr(i.total_core_order, i.total_order) corr
  FROM {TD} t JOIN {P('driver_income_daily')} i ON i.driver_id=t.driver_id AND i.order_date=t.d""")
q("core_by_class", f"""SELECT p.vehicle_class, avg(i.total_order - i.total_core_order) mean_noncore, avg(i.total_core_order::DOUBLE/nullif(i.total_order,0)) core_share
  FROM {P('driver_income_daily')} i JOIN {P('driver_profile')} p USING (driver_id) GROUP BY 1 ORDER BY 1""")
# rush flag: which timestamp defines it?  fraction of trips where rush_hour == (hour of X in policy peak hours)
q("rush_ref_ts", f"""SELECT avg((rush_hour = (hr IN (6,7,8,16,17,18)))::INT) by_request, avg((rush_hour = (ahr IN (6,7,8,16,17,18)))::INT) by_assign,
  avg((rush_hour = (phr IN (6,7,8,16,17,18)))::INT) by_pickup FROM {TR} WHERE status='completed'""")
q("rush_by_minute_edges", f"""SELECT hr, mi//10*10 m10, avg(rush_hour::INT) rush_frac, count(*) n FROM {TR} WHERE hr IN (5,8,15,18) GROUP BY ALL ORDER BY ALL""")
# hex tracking row <-> telemetry ping on one day: join on driver_id + timestamp
day = "2026-07-15"
q("hex_vs_ping_one_day", f"""WITH t AS (SELECT driver_id, occurred_at, current_hex_h3_res9 hx, speed_kmh FROM read_parquet('{ROOT}/telemetry/vehicle_telemetry_ping_{day}.parquet')),
  h AS (SELECT driver_id, last_seen_at, created_at, current_hex, last_hex, tracking_status, stay_duration_seconds FROM {P('public_driver_hex_tracking')} WHERE last_seen_at LIKE '{day}%')
  SELECT (SELECT count(*) FROM t) n_tel, (SELECT count(*) FROM h) n_hex, count(*) n_join_ts, avg((h.current_hex=t.hx)::INT) cur_eq_pinghex,
   avg((h.last_hex=t.hx)::INT) last_eq_pinghex
  FROM h JOIN t ON t.driver_id=h.driver_id AND t.occurred_at=h.last_seen_at""")
q("hex_status_vs_speed", f"""WITH t AS (SELECT driver_id, occurred_at, speed_kmh FROM read_parquet('{ROOT}/telemetry/vehicle_telemetry_ping_{day}.parquet')),
  h AS (SELECT driver_id, last_seen_at, tracking_status FROM {P('public_driver_hex_tracking')} WHERE last_seen_at LIKE '{day}%')
  SELECT tracking_status, count(*) n, avg((speed_kmh=0)::INT) speed0, median(speed_kmh) med_speed FROM h JOIN t ON t.driver_id=h.driver_id AND t.occurred_at=h.last_seen_at GROUP BY 1""")
q("hex_res", f"SELECT length(current_hex) l, count(*) n FROM {P('public_driver_hex_tracking')} WHERE last_seen_at LIKE '{day}%' GROUP BY 1")
q("frauds_vs_cancels", f"""WITH f AS (SELECT driver_id, substr(detected_at,1,10) d, count(*) nf, max(severity) sev FROM {P('public_frauds')} GROUP BY 1,2)
  SELECT f.d IS NOT NULL flagged, count(*) n, avg(s.cancelled_count) mean_canc, avg(s.cancellation_rate) mean_canc_rate
  FROM {P('driver_statistic_daily')} s LEFT JOIN f ON f.driver_id=s.driver_id AND f.d=s.local_date GROUP BY 1""")
q("mission_vs_trips", f"""WITH m AS (SELECT driver_id, substr(order_time,1,10) d, count(*) n FROM {P('public_mission_earn_history')} GROUP BY 1,2)
  SELECT avg(m.n) mean_mission_rows_per_dd, count(*) dd FROM m""")
q("kpi_weekly", f"SELECT status, count(*) n FROM {P('kpi_driver_platform_calculator_gbq')} GROUP BY 1")
save("02b_followups.json", R)
