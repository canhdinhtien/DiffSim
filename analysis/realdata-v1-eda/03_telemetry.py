"""Telemetry quality + time-series structure, full data, one daily file at a time (bounded memory).
Writes out/03_tel_daily.parquet (per-day aggregates), out/03_tel_driverday.parquet (per driver-day), out/03_tel_hist.json."""
import glob, os, json
import pandas as pd
from common import ROOT, OUT, con, save

c = con("8GB")
files = sorted(glob.glob(f"{ROOT}/telemetry/vehicle_telemetry_ping_*.parquet"))[:int(os.environ.get("TEL_LIMIT", "999"))]
DT_BINS = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60, 120, 300, 600, 1800, 3600, 1e9]
day_rows, dd_frames, hist = [], [], {}


def add_hist(key, df):
    for r in df.itertuples(index=False):
        hist.setdefault(key, {}).setdefault(str(r[0]), 0)
        hist[key][str(r[0])] += int(r[1])


CASE_DT = "CASE " + " ".join(f"WHEN dt < {b} THEN '{a}-{b}'" for a, b in zip(DT_BINS[:-1], DT_BINS[1:])) + " END"
for f in files:
    day = os.path.basename(f)[-18:-8]
    fp = f.replace(os.sep, "/")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE t AS
      SELECT file_row_number rn, driver_id, trip_id, occurred_at, epoch(occurred_at::TIMESTAMPTZ) ts, substr(occurred_at,12,2)::INT hr,
             occurred_at LIKE '%+07:00' tz07, lat, lng, speed_kmh, heading_deg, gear, odometer_km, battery_level_pct bat, remaining_range_km rng,
             charging_status, door_lock_status, phone_connectivity_lost lost, seat_occupancy.any_passenger_seat pax, seat_occupancy.driver drv
      FROM read_parquet('{fp}', file_row_number=true)""")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE w AS
      SELECT *, ts - lag(ts) OVER w dt,
        lag(rn) OVER w prev_rn,
        2*6371000*asin(sqrt(pow(sin(radians(lat-lag(lat) OVER w)/2),2) + cos(radians(lat))*cos(radians(lag(lat) OVER w))*pow(sin(radians(lng-lag(lng) OVER w)/2),2))) dist_m,
        odometer_km - lag(odometer_km) OVER w dodo, bat - lag(bat) OVER w dbat, speed_kmh - lag(speed_kmh) OVER w dspd,
        degrees(atan2(sin(radians(lng-lag(lng) OVER w))*cos(radians(lat)),
                      cos(radians(lag(lat) OVER w))*sin(radians(lat)) - sin(radians(lag(lat) OVER w))*cos(radians(lat))*cos(radians(lng-lag(lng) OVER w)))) brg,
        lag(trip_id) OVER w prev_trip, lag(pax) OVER w prev_pax
      FROM t WINDOW w AS (PARTITION BY driver_id ORDER BY ts, rn)""")
    r = c.execute(f"""SELECT '{day}' AS day, count(*) n, count(DISTINCT driver_id) drivers, sum((NOT tz07)::INT) n_not_07,
      sum((rn < prev_rn)::INT) n_out_of_file_order, sum((dt=0)::INT) n_dup_ts,
      sum((dt>60)::INT) n_gap60, sum((dt>300)::INT) n_gap300, sum((dt>1800)::INT) n_gap1800,
      median(dt) dt_med, quantile_cont(dt,0.9) dt_p90,
      sum((dt>0 AND dist_m/dt*3.6 > 150)::INT) n_jump150, sum((dt>0 AND dt<=60 AND dist_m > 2000)::INT) n_jump2km,
      sum((speed_kmh>80)::INT) n_speed80, sum((speed_kmh=0)::INT) n_speed0, max(speed_kmh) speed_max,
      sum((dodo < -1e-6)::INT) n_odo_back, sum(CASE WHEN dt<=300 THEN dodo END) odo_km_sum, sum(CASE WHEN dt<=300 THEN dist_m END)/1000 hav_km_sum,
      corr(CASE WHEN dt<=300 AND dt>0 THEN dodo END, CASE WHEN dt<=300 AND dt>0 THEN dist_m END) corr_odo_hav,
      corr(CASE WHEN dt BETWEEN 1 AND 30 THEN speed_kmh END, CASE WHEN dt BETWEEN 1 AND 30 THEN dist_m/dt*3.6 END) corr_speed_implied,
      sum((dbat > 0.5)::INT) n_bat_up, sum(CASE WHEN dbat>0.5 THEN dbat END) bat_up_sum, sum((dbat < -5)::INT) n_bat_drop5,
      sum(CASE WHEN dt<=300 AND dbat<=0 THEN -dbat END) bat_used_pct, sum(CASE WHEN dt<=300 AND dbat<=0 THEN dodo END) bat_used_km,
      corr(rng, bat) corr_range_bat, regr_slope(rng, bat) range_per_pct,
      sum((dt BETWEEN 1 AND 30 AND dspd/dt/3.6 < -3)::INT) n_hard_brake, sum((dt BETWEEN 1 AND 30 AND dspd/dt/3.6 > 3)::INT) n_hard_acc,
      median(CASE WHEN dist_m>30 AND speed_kmh>10 THEN abs(((heading_deg - (brg+360)%360 + 540) % 360) - 180) END) heading_err_med,
      quantile_cont(CASE WHEN dist_m>30 AND speed_kmh>10 THEN abs(((heading_deg - (brg+360)%360 + 540) % 360) - 180) END, 0.9) heading_err_p90,
      sum((heading_deg IS NULL)::INT) n_heading_null,
      sum((gear='P')::INT) n_gear_p, sum((gear='P' AND charging_status='IDLE')::INT) n_p_idle, sum((gear='P' AND speed_kmh=0)::INT) n_p_speed0,
      sum((speed_kmh=0 AND gear='D')::INT) n_d_speed0,
      sum((lost)::INT) n_lost, sum((NOT drv)::INT) n_no_driver,
      sum(((trip_id IS NOT NULL) <> pax)::INT) n_trip_vs_pax_mismatch, sum((door_lock_status='UNLOCKED')::INT) n_unlocked,
      sum((door_lock_status='UNLOCKED' AND speed_kmh=0)::INT) n_unlocked_speed0,
      sum((trip_id IS NOT NULL AND prev_trip IS NULL)::INT) n_trip_starts
      FROM w""").df()
    day_rows.append(r)
    add_hist("dt_s", c.execute(f"SELECT {CASE_DT} b, count(*) FROM w WHERE dt IS NOT NULL GROUP BY 1").df())
    add_hist("gap_len_min_gt60s", c.execute("SELECT CASE WHEN dt<300 THEN '1-5' WHEN dt<900 THEN '5-15' WHEN dt<1800 THEN '15-30' WHEN dt<3600 THEN '30-60' WHEN dt<4*3600 THEN '60-240' ELSE '240+' END, count(*) FROM w WHERE dt>60 GROUP BY 1").df())
    add_hist("speed_10kmh", c.execute("SELECT (floor(speed_kmh/10)*10)::INT, count(*) FROM w GROUP BY 1").df())
    add_hist("pings_by_hour", c.execute("SELECT hr, count(*) FROM w GROUP BY 1").df())
    add_hist("speed_sum_by_hour", c.execute("SELECT hr, sum(speed_kmh)::BIGINT FROM w GROUP BY 1").df())
    add_hist("moving_pings_by_hour", c.execute("SELECT hr, sum((speed_kmh>0)::INT) FROM w GROUP BY 1").df())
    add_hist("pax_pings_by_hour", c.execute("SELECT hr, sum(pax::INT) FROM w GROUP BY 1").df())
    add_hist("battery_10pct", c.execute("SELECT (floor(bat/10)*10)::INT, count(*) FROM w GROUP BY 1").df())
    add_hist("bat_jump_size", c.execute("SELECT CASE WHEN dbat<2 THEN '0.5-2' WHEN dbat<10 THEN '2-10' WHEN dbat<30 THEN '10-30' WHEN dbat<60 THEN '30-60' ELSE '60+' END, count(*) FROM w WHERE dbat>0.5 GROUP BY 1").df())
    add_hist("bat_jump_hour", c.execute("SELECT hr, count(*) FROM w WHERE dbat>0.5 GROUP BY 1").df())
    add_hist("first_ping_hour", c.execute("SELECT hr, count(*) FROM (SELECT driver_id, arg_min(hr, ts) hr FROM w GROUP BY 1) GROUP BY 1").df())
    add_hist("last_ping_hour", c.execute("SELECT hr, count(*) FROM (SELECT driver_id, arg_max(hr, ts) hr FROM w GROUP BY 1) GROUP BY 1").df())
    add_hist("stop_run_len_min", c.execute("""WITH s0 AS (SELECT driver_id, ts, rn, speed_kmh=0 z, CASE WHEN (speed_kmh=0) IS DISTINCT FROM (lag(speed_kmh) OVER w2 = 0) THEN 1 ELSE 0 END chg
          FROM w WINDOW w2 AS (PARTITION BY driver_id ORDER BY ts, rn)),
        s AS (SELECT *, sum(chg) OVER (PARTITION BY driver_id ORDER BY ts, rn) grp FROM s0),
        runs AS (SELECT driver_id, grp, max(ts)-min(ts) len FROM s WHERE z GROUP BY 1,2)
        SELECT CASE WHEN len<60 THEN '<1' WHEN len<300 THEN '1-5' WHEN len<900 THEN '5-15' WHEN len<1800 THEN '15-30' WHEN len<3600 THEN '30-60' ELSE '60+' END, count(*) FROM runs GROUP BY 1""").df())
    dd = c.execute(f"""SELECT '{day}' AS day, driver_id, count(*) n_pings, (max(ts)-min(ts))/3600 span_h,
        sum(CASE WHEN dt<=300 THEN dt ELSE 0 END)/3600 active_h, sum(CASE WHEN dt<=300 AND speed_kmh>0 THEN dt ELSE 0 END)/3600 moving_h,
        sum(CASE WHEN dt<=300 AND pax THEN dt ELSE 0 END)/3600 pax_h, max(odometer_km)-min(odometer_km) odo_km,
        sum(CASE WHEN dt<=300 AND dbat<=0 THEN -dbat ELSE 0 END) bat_used, sum((dbat>0.5)::INT) n_charge_steps, avg(speed_kmh) mean_speed,
        count(DISTINCT trip_id) n_trips_tel, sum((dt>60)::INT) n_gaps60, min(hr) first_hr, max(hr) last_hr
        FROM w GROUP BY 2""").df()
    dd_frames.append(dd)
    print(day, int(r.n[0]), flush=True)

daily = pd.concat(day_rows)
daily.to_parquet(f"{OUT}/03_tel_daily.parquet")
pd.concat(dd_frames).to_parquet(f"{OUT}/03_tel_driverday.parquet")
save("03_tel_hist.json", hist)
# additive counters are summed; per-day statistics (medians, corr, max, slopes) are summarised across days
nonadd = [k for k in daily.columns if k.startswith(("dt_", "corr_", "heading_err", "range_per", "speed_max"))]
tot = {"sum": daily.drop(columns=["day"] + nonadd).sum(numeric_only=True).to_dict(),
       "per_day_stats": daily[nonadd].describe().to_dict()}
save("03_tel_totals.json", tot)
print(tot)
