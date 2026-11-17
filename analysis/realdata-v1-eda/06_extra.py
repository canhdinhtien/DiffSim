"""Extra field-level checks used by verdicts (online_time > 24h, range vs battery per model, gear/door semantics, odometer per driver across days)."""
from common import P, ROOT, con, save
c = con("6GB"); R = {}
def q(name, sql):
    df = c.execute(sql).df(); R[name] = df.to_dict(orient="records"); print(f"--- {name}\n{df.to_string()[:2000]}")
q("online_gt24", f"SELECT count(*) FILTER (WHERE online_time>24) n_gt24, count(*) FILTER (WHERE online_time>16) n_gt16, count(*) FILTER (WHERE online_time=3.0) n_eq3, count(*) n FROM {P('driver_online_hours_sap_id')}")
q("hub_depot_per_driver", f"SELECT max(nh) max_hubs_per_driver, max(nd) max_depots_per_driver FROM (SELECT driver_id, count(DISTINCT hub_id) nh, count(DISTINCT depot_id) nd FROM {P('driver_online_hours_sap_id')} GROUP BY 1)")
day = "2026-08-02"
T = f"(SELECT t.*, p.vehicle_model, p.fleet FROM read_parquet('{ROOT}/telemetry/vehicle_telemetry_ping_{day}.parquet') t JOIN {P('driver_profile')} p USING (driver_id))"
q("range_vs_battery_by_model", f"SELECT vehicle_model, corr(remaining_range_km, battery_level_pct) r, regr_slope(remaining_range_km, battery_level_pct) km_per_pct, regr_r2(remaining_range_km, battery_level_pct) r2, quantile_cont(remaining_range_km/nullif(battery_level_pct,0),[.05,.5,.95]) ratio_q FROM {T} GROUP BY 1 ORDER BY 1")
q("gear_door_semantics", f"SELECT gear, charging_status, door_lock_status, speed_kmh=0 stopped, trip_id IS NOT NULL on_trip, count(*) n FROM {T} GROUP BY ALL ORDER BY n DESC")
q("battery_low", f"SELECT vehicle_model, fleet, avg((battery_level_pct<10)::INT) lt10, avg((battery_level_pct=0)::INT) eq0, min(battery_level_pct) mn FROM {T} GROUP BY ALL ORDER BY ALL")
q("vin_per_driver", f"SELECT max(n) max_vins_per_driver_day, avg((n>1)::INT) frac_multi FROM (SELECT driver_id, count(DISTINCT vin) n FROM {T} GROUP BY 1)")
# odometer / battery continuity across days (end of day d vs start of d+1)
q("cross_day_continuity", f"""WITH a AS (SELECT driver_id, substr(occurred_at,1,10) d, arg_max(odometer_km, occurred_at) odo_end, arg_min(odometer_km, occurred_at) odo_start,
     arg_max(battery_level_pct, occurred_at) bat_end, arg_min(battery_level_pct, occurred_at) bat_start
     FROM read_parquet(['{ROOT}/telemetry/vehicle_telemetry_ping_2026-08-01.parquet','{ROOT}/telemetry/vehicle_telemetry_ping_2026-08-02.parquet']) GROUP BY 1,2)
  SELECT avg((b.odo_start >= a.odo_end - 1e-6)::INT) odo_continuous, median(b.odo_start - a.odo_end) odo_gap_km,
         median(b.bat_start - a.bat_end) overnight_bat_gain, avg((b.bat_start > a.bat_end + 1)::INT) frac_overnight_charge, quantile_cont(b.bat_start,[.1,.5,.9]) start_bat_q
  FROM a JOIN a b ON b.driver_id=a.driver_id AND a.d='2026-08-01' AND b.d='2026-08-02'""")
save("06_extra.json", R)
# odometer decreases vs battery jumps: same pings?
R2 = c.execute(f"""WITH w AS (SELECT odometer_km - lag(odometer_km) OVER w dodo, battery_level_pct - lag(battery_level_pct) OVER w dbat,
      epoch(occurred_at::TIMESTAMPTZ) - lag(epoch(occurred_at::TIMESTAMPTZ)) OVER w dt, substr(occurred_at,12,2)::INT hr
    FROM read_parquet('{ROOT}/telemetry/*.parquet') WINDOW w AS (PARTITION BY driver_id, substr(occurred_at,1,10) ORDER BY occurred_at))
  SELECT count(*) FILTER (WHERE dodo < -1e-6) odo_back, count(*) FILTER (WHERE dbat > 0.5) bat_up, count(*) FILTER (WHERE dodo < -1e-6 AND dbat > 0.5) n_both,
    median(dodo) FILTER (WHERE dodo < -1e-6) med_odo_back_km, median(dt) FILTER (WHERE dbat > 0.5) med_dt_at_charge_s FROM w""").df()
R["odo_back_vs_charge"] = R2.to_dict(orient="records"); print(R2.to_string())
save("06_extra.json", R)
