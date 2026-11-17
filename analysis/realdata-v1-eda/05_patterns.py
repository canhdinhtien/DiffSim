"""Seasonality, trend, cross-signal relations, rare events, driver heterogeneity.
Needs out/03_tel_driverday.parquet (from 03_telemetry.py)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from common import P, OUT, con, save

c = con("8GB")
R = {}


def q(name, sql):
    df = c.execute(sql).df()
    R[name] = df.to_dict(orient="records")
    print(f"--- {name}\n{df.to_string(max_cols=40, max_rows=40)[:3500]}")
    return df


TR = f"""(SELECT *, substr(request_time,1,10)::DATE d, substr(request_time,12,2)::INT hr, pickup_time IS NOT NULL picked FROM {P('trips')})"""
q("trips_by_dow", f"SELECT dayofweek(d) dow, count(DISTINCT d) n_days, count(*)/count(DISTINCT d) trips_per_day, avg((status='cancelled')::INT) cancel_rate FROM {TR} GROUP BY 1 ORDER BY 1")
daily = q("trips_by_day", f"""SELECT d, dayofweek(d) dow, count(*) trips, avg((status='cancelled')::INT) cancel_rate, median(distance_km) med_km,
   median(duration_seconds) med_dur, median(distance_km/nullif(duration_seconds,0)*3600) med_speed FROM {TR} GROUP BY 1 ORDER BY 1""")
d = daily.set_index("d")
R["day_outliers"] = d[(d.trips - d.trips.median()).abs() > 2.5 * d.trips.std()].reset_index().to_dict(orient="records")
R["trips_per_day_stats"] = d.trips.describe().to_dict()
R["trend_slope_trips_per_day"] = float(np.polyfit(np.arange(len(d)), d.trips.values, 1)[0])
print("outliers", R["day_outliers"], "slope", R["trend_slope_trips_per_day"])
q("trips_by_hour", f"""SELECT hr, count(*) n, avg((status='cancelled')::INT) cancel_rate, avg((cancelled_by='driver')::INT) drv_cancel,
   median(distance_km/nullif(duration_seconds,0)*3600) med_speed, median(epoch(pickup_time::TIMESTAMPTZ)-epoch(assign_time::TIMESTAMPTZ)) med_to_pickup_s,
   median(epoch(assign_time::TIMESTAMPTZ)-epoch(request_time::TIMESTAMPTZ)) med_assign_wait_s FROM {TR} GROUP BY 1 ORDER BY 1""")
q("hour_dow_weekend", f"SELECT dayofweek(d) IN (0,6) weekend, hr, count(*)/count(DISTINCT d) per_day FROM {TR} GROUP BY ALL ORDER BY ALL")
q("cancel_by_area", f"""WITH a AS (SELECT pickup_h3, count(*) n, avg((status='cancelled')::INT) cr FROM {TR} GROUP BY 1)
   SELECT count(*) n_cells, quantile_cont(n,[.1,.5,.9]) trips_per_cell_q, quantile_cont(cr,[.1,.5,.9]) cancel_rate_q,
   quantile_cont(cr,[.1,.5,.9]) FILTER (WHERE n>=1000) cancel_rate_q_n1000, sum(n) FILTER (WHERE n >= (SELECT quantile_cont(n,.9) FROM a))/sum(n) top10pct_cells_share FROM a""")
q("vehicle_effects", f"""SELECT p.vehicle_model, p.fleet, count(*) dd, avg(s.completed_count) completed, avg(o.online_time) online_h, avg(i.total_fee) fee,
   avg(s.cancellation_rate) canc_rate, avg(st.total_stoppoints) stoppoints, avg(r.total_order_rush_hour::DOUBLE/nullif(r.total_order,0)) rush_share
   FROM {P('driver_statistic_daily')} s JOIN {P('driver_profile')} p USING (driver_id)
   JOIN {P('driver_online_hours_sap_id')} o USING (driver_id, local_date) JOIN {P('driver_income_daily')} i ON i.driver_id=s.driver_id AND i.order_date=s.local_date
   JOIN {P('driver_bike_stoppoints')} st USING (driver_id, local_date) JOIN {P('driver_orders_rush_hours')} r USING (driver_id, local_date)
   GROUP BY ALL ORDER BY ALL""")
q("online_vs_earn", f"""SELECT corr(o.online_time, i.total_fee) corr_online_fee, regr_slope(i.total_fee, o.online_time) vnd_per_online_h,
   corr(o.online_time, s.completed_count) corr_online_completed, regr_slope(s.completed_count, o.online_time) trips_per_online_h
   FROM {P('driver_online_hours_sap_id')} o JOIN {P('driver_income_daily')} i ON i.driver_id=o.driver_id AND i.order_date=o.local_date
   JOIN {P('driver_statistic_daily')} s ON s.driver_id=o.driver_id AND s.local_date=o.local_date""")
q("online_by_dow", f"SELECT dayofweek(local_date::DATE) dow, avg(online_time) online_h, avg(completed_count) completed FROM {P('driver_online_hours_sap_id')} o JOIN {P('driver_statistic_daily')} s USING (driver_id, local_date) GROUP BY 1 ORDER BY 1")
q("driver_autocorr", f"""WITH x AS (SELECT driver_id, local_date, completed_count y, lag(completed_count) OVER (PARTITION BY driver_id ORDER BY local_date) y1,
     lag(completed_count,7) OVER (PARTITION BY driver_id ORDER BY local_date) y7 FROM {P('driver_statistic_daily')})
   SELECT corr(y,y1) acf1_pooled, corr(y,y7) acf7_pooled FROM x""")
q("between_within_var", f"""WITH m AS (SELECT driver_id, avg(completed_count) mu, var_samp(completed_count) v FROM {P('driver_statistic_daily')} GROUP BY 1)
   SELECT var_samp(mu) between_var, avg(v) within_var FROM m""")
# telemetry driver-day joined with daily tables
c.execute(f"CREATE TEMP TABLE dd AS SELECT * FROM read_parquet('{OUT}/03_tel_driverday.parquet'.replace('\\\\','/'))")
q("tel_vs_daily", f"""SELECT count(*) n, corr(dd.active_h, o.online_time) corr_active_online, median(dd.active_h - o.online_time) med_active_minus_online,
   median(dd.span_h - o.online_time) med_span_minus_online, corr(dd.n_trips_tel, s.completed_count) corr_trips, avg((dd.n_trips_tel = s.completed_count)::INT) trips_eq,
   median(dd.n_trips_tel - s.completed_count) med_trip_diff,
   corr(dd.pax_h, s.completed_count) corr_paxh_completed, corr(dd.odo_km, i.total_fee) corr_km_fee
   FROM dd JOIN {P('driver_online_hours_sap_id')} o ON o.driver_id=dd.driver_id AND o.local_date=dd.day
   JOIN {P('driver_statistic_daily')} s ON s.driver_id=dd.driver_id AND s.local_date=dd.day
   JOIN {P('driver_income_daily')} i ON i.driver_id=dd.driver_id AND i.order_date=dd.day""")
q("tel_trip_km_vs_trips", f"""WITH t AS (SELECT driver_id, substr(request_time,1,10) d, sum(distance_km) km FROM {P('trips')} WHERE status='completed' GROUP BY 1,2)
   SELECT corr(dd.odo_km, t.km) corr_odo_tripkm, median(t.km/nullif(dd.odo_km,0)) med_paid_km_share FROM dd JOIN t ON t.driver_id=dd.driver_id AND t.d=dd.day""")
q("battery_vs_km", f"""SELECT p.vehicle_model, count(*) n, median(dd.bat_used/nullif(dd.odo_km,0)) pct_per_km, median(dd.odo_km) km, median(dd.n_charge_steps) charge_steps
   FROM dd JOIN {P('driver_profile')} p USING (driver_id) WHERE dd.odo_km > 10 GROUP BY 1 ORDER BY 1""")
q("shift_vs_actual", f"""SELECT corr(json_extract(p.declared_shift_window,'$[0]')::INT/60.0, dd.first_hr) corr_start, corr(json_extract(p.declared_shift_window,'$[1]')::INT/60.0, dd.last_hr) corr_end,
   median(dd.first_hr - json_extract(p.declared_shift_window,'$[0]')::INT/60.0) med_start_diff_h
   FROM dd JOIN {P('driver_profile')} p USING (driver_id)""")
q("driverday_dist", """SELECT quantile_cont(n_pings,[.05,.5,.95]) pings, quantile_cont(span_h,[.05,.5,.95]) span_h, quantile_cont(active_h,[.05,.5,.95]) active_h,
   quantile_cont(moving_h,[.05,.5,.95]) moving_h, quantile_cont(pax_h,[.05,.5,.95]) pax_h, quantile_cont(odo_km,[.05,.5,.95]) odo_km,
   quantile_cont(n_gaps60,[.05,.5,.95]) gaps60, quantile_cont(bat_used,[.05,.5,.95]) bat_used, avg((n_charge_steps>0)::INT) frac_days_with_charge FROM dd""")
q("drivers_per_day", "SELECT min(n) mn, max(n) mx, median(n) med FROM (SELECT day, count(*) n FROM dd GROUP BY 1)")
# driver clustering (per-driver means)
feat = c.execute(f"""SELECT s.driver_id, avg(s.completed_count) completed, avg(o.online_time) online_h, avg(s.cancellation_rate) canc,
   avg(r.total_order_rush_hour::DOUBLE/nullif(r.total_order,0)) rush_share, avg(i.avg_daily_revenue) rev_per_trip, avg(dd.first_hr) first_hr, avg(dd.mean_speed) speed
   FROM {P('driver_statistic_daily')} s JOIN {P('driver_online_hours_sap_id')} o USING (driver_id, local_date)
   JOIN {P('driver_orders_rush_hours')} r USING (driver_id, local_date) JOIN {P('driver_income_daily')} i ON i.driver_id=s.driver_id AND i.order_date=s.local_date
   JOIN dd ON dd.driver_id=s.driver_id AND dd.day=s.local_date GROUP BY 1""").df()
X = StandardScaler().fit_transform(feat.drop(columns="driver_id").fillna(0))
inertia = {k: float(KMeans(k, n_init=5, random_state=0).fit(X).inertia_) for k in range(2, 8)}
km = KMeans(4, n_init=10, random_state=0).fit(X)
feat["cluster"] = km.labels_
R["kmeans_inertia"] = inertia
R["kmeans4_profiles"] = feat.drop(columns="driver_id").groupby("cluster").mean().round(3).reset_index().to_dict(orient="records")
R["kmeans4_sizes"] = feat.cluster.value_counts().sort_index().to_dict()
prof = c.execute(f"SELECT driver_id, vehicle_model, fleet FROM {P('driver_profile')}").df()
R["kmeans4_x_model"] = pd.crosstab(feat.merge(prof).cluster, feat.merge(prof).vehicle_model).to_dict()
print(inertia, R["kmeans4_sizes"]); print(feat.drop(columns="driver_id").groupby("cluster").mean().round(2).to_string()); print(R["kmeans4_x_model"])
save("05_patterns.json", R)
