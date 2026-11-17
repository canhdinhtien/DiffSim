"""Driver-state reconstruction (telemetry x trips), H3 consistency, hex dwell, rush-flag reference timestamp.
Uses 3 sample days (weekday / weekend / late) to bound cost."""
import os
import h3
import pandas as pd
from common import ROOT, P, con, save

c = con("8GB")
DAYS = os.environ.get("STATE_DAYS", "2026-07-15,2026-08-02,2026-10-20").split(",")
R = {}


def q(name, sql):
    df = c.execute(sql).df()
    R.setdefault(name, []).extend(df.to_dict(orient="records"))
    print(f"--- {name}\n{df.to_string(max_cols=40)[:2500]}")
    return df


q("rush_ref_complete", f"""SELECT avg((rush_hour = (substr(complete_time,12,2)::INT IN (6,7,8,16,17,18)))::INT) by_complete,
   avg((rush_hour = ((epoch(pickup_time::TIMESTAMPTZ)+epoch(complete_time::TIMESTAMPTZ))/2 % 86400 // 3600 + 7) % 24 IN (6,7,8,16,17,18))::INT) by_midtrip
   FROM {P('trips')} WHERE status='completed'""")
for day in DAYS:
    tel = f"{ROOT}/telemetry/vehicle_telemetry_ping_{day}.parquet"
    c.execute(f"""CREATE OR REPLACE TEMP TABLE t AS SELECT driver_id, trip_id, occurred_at::TIMESTAMPTZ ts, file_row_number rn, lat, lng, speed_kmh,
        current_hex_h3_res9 hx, seat_occupancy.any_passenger_seat pax, gear, door_lock_status door FROM read_parquet('{tel}', file_row_number=true)""")
    c.execute(f"""CREATE OR REPLACE TEMP TABLE tr AS SELECT trip_id, driver_id, status, request_time::TIMESTAMPTZ rq, assign_time::TIMESTAMPTZ asg,
        pickup_time::TIMESTAMPTZ pk, complete_time::TIMESTAMPTZ cp, pickup_h3, drop_h3, distance_km FROM {P('trips')}
        WHERE request_time >= '{day}' AND request_time < '{day}T23:59:59.999999+07:00'""")
    q("tel_trip_link", f"""SELECT '{day}' AS day, count(*) FILTER (WHERE t.trip_id IS NOT NULL) pings_with_trip,
        avg((tr.trip_id IS NOT NULL)::INT) FILTER (WHERE t.trip_id IS NOT NULL) trip_found,
        avg((tr.status='completed')::INT) FILTER (WHERE t.trip_id IS NOT NULL) trip_completed,
        avg((t.ts BETWEEN tr.pk AND tr.cp)::INT) FILTER (WHERE tr.trip_id IS NOT NULL) in_pickup_complete,
        avg((t.ts BETWEEN tr.asg AND tr.pk)::INT) FILTER (WHERE tr.trip_id IS NOT NULL) in_assign_pickup
        FROM t LEFT JOIN tr ON tr.trip_id=t.trip_id""")
    q("trip_coverage", f"""SELECT '{day}' AS day, status, count(*) trips, avg((n>0)::INT) has_pings, median(n) med_pings, median(n/nullif(epoch(cp)-epoch(pk),0)*60) pings_per_min
        FROM (SELECT tr.*, (SELECT count(*) FROM t WHERE t.trip_id=tr.trip_id) n FROM tr) GROUP BY 2""")
    # interval join: state of each ping from trips windows
    c.execute("""CREATE OR REPLACE TEMP TABLE st AS SELECT t.*, CASE WHEN tr2.pk IS NOT NULL AND t.ts BETWEEN tr2.pk AND tr2.cp THEN 'on_trip'
                  WHEN tr2.asg IS NOT NULL AND t.ts >= tr2.asg AND (tr2.pk IS NULL OR t.ts < tr2.pk) THEN 'to_pickup' ELSE 'idle' END state
        FROM t LEFT JOIN tr tr2 ON tr2.driver_id=t.driver_id AND t.ts >= tr2.asg AND t.ts <= coalesce(tr2.cp, tr2.asg + INTERVAL 15 MINUTE)""")
    q("state_mix", f"""SELECT '{day}' AS day, state, count(*) n, avg(pax::INT) pax_rate, avg((trip_id IS NOT NULL)::INT) tripid_rate, median(speed_kmh) med_speed,
        avg((speed_kmh=0)::INT) speed0, avg((gear='P')::INT) gear_p, avg((door='UNLOCKED')::INT) unlocked FROM st GROUP BY 2 ORDER BY 2""")
    q("multi_match", f"SELECT '{day}' AS day, count(*) - count(DISTINCT (driver_id, ts, rn)) dup_rows_from_overlap FROM st")
    # hex dwell from telemetry runs
    q("tel_hex_dwell", f"""WITH a AS (SELECT driver_id, ts, hx, CASE WHEN hx IS DISTINCT FROM lag(hx) OVER w THEN 1 ELSE 0 END chg FROM t WINDOW w AS (PARTITION BY driver_id ORDER BY ts, rn)),
        b AS (SELECT *, sum(chg) OVER (PARTITION BY driver_id ORDER BY ts) g FROM a),
        r AS (SELECT driver_id, g, epoch(max(ts))-epoch(min(ts)) dwell, count(*) n FROM b GROUP BY 1,2)
        SELECT '{day}' AS day, count(*) runs, quantile_cont(dwell,[.5,.9,.99]) dwell_q, quantile_cont(n,[.5,.9,.99]) pings_q, avg((n=1)::INT) single_ping_runs FROM r""")
    q("hex_tracking_join_tstz", f"""WITH h AS (SELECT driver_id, last_seen_at::TIMESTAMPTZ ts, current_hex, stay_duration_seconds sd, tracking_status s
            FROM {P('public_driver_hex_tracking')} WHERE last_seen_at LIKE '{day}%')
        SELECT '{day}' AS day, (SELECT count(*) FROM h) n_hex, count(*) n_join, avg((h.current_hex=t.hx)::INT) hex_eq, avg((s='idle')::INT) idle_rate,
           avg((s='idle' AND t.speed_kmh=0)::INT)/nullif(avg((s='idle')::INT),0) idle_speed0
        FROM h JOIN t ON t.driver_id=h.driver_id AND t.ts=h.ts""")
    q("stationary_jitter", f"""WITH w AS (SELECT speed_kmh, lag(speed_kmh) OVER w ps, hx = lag(hx) OVER w same_hex,
          2*6371000*asin(sqrt(pow(sin(radians(lat-lag(lat) OVER w)/2),2) + cos(radians(lat))*cos(radians(lag(lat) OVER w))*pow(sin(radians(lng-lag(lng) OVER w)/2),2))) d
          FROM t WINDOW w AS (PARTITION BY driver_id ORDER BY ts, rn))
        SELECT '{day}' AS day, count(*) n_both_speed0, quantile_cont(d,[.1,.5,.9]) dist_m_q, avg(same_hex::INT) same_hex_rate FROM w WHERE speed_kmh=0 AND ps=0""")
    # reported vs implied speed by state (to explain negative correlation)
    q("speed_implied_by_state", f"""WITH w AS (SELECT state, speed_kmh, epoch(ts)-epoch(lag(ts) OVER w) dt,
          2*6371000*asin(sqrt(pow(sin(radians(lat-lag(lat) OVER w)/2),2) + cos(radians(lat))*cos(radians(lag(lat) OVER w))*pow(sin(radians(lng-lag(lng) OVER w)/2),2))) d
          FROM st WINDOW w AS (PARTITION BY driver_id ORDER BY ts, rn))
        SELECT '{day}' AS day, state, median(speed_kmh) rep_med, median(d/dt*3.6) impl_med, corr(speed_kmh, d/dt*3.6) corr, quantile_cont(dt,[.1,.5,.9]) dt_q
        FROM w WHERE dt BETWEEN 1 AND 300 GROUP BY 2 ORDER BY 2""")
    # H3 consistency (python h3 on a 200k sample)
    s = c.execute("SELECT lat, lng, hx FROM t USING SAMPLE 200000 ROWS (reservoir, 1)").df()
    s["calc"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(s.lat, s.lng)]
    s["res"] = [h3.get_resolution(x) for x in s.hx]
    s["k"] = [h3.grid_distance(a, b) if a != b else 0 for a, b in zip(s.hx, s.calc)]
    R.setdefault("h3_consistency", []).append({"day": day, "n": len(s), "res_values": s.res.value_counts().to_dict(),
        "exact_match": float((s.calc == s.hx).mean()), "neighbor_k1": float((s.k == 1).mean()), "k_gt1": float((s.k > 1).mean())})
    print(R["h3_consistency"][-1])
    # pickup_h3 resolution + match to telemetry hex at pickup
    pk = c.execute("""SELECT tr.pickup_h3, tr.drop_h3, arg_min(t.hx, abs(epoch(t.ts)-epoch(tr.pk))) hx_at_pk, min(abs(epoch(t.ts)-epoch(tr.pk))) gap
        FROM tr JOIN t ON t.driver_id=tr.driver_id AND t.ts BETWEEN tr.pk - INTERVAL 2 MINUTE AND tr.pk + INTERVAL 2 MINUTE
        WHERE tr.status='completed' GROUP BY 1,2, tr.trip_id""").df()
    rp = pk.pickup_h3.map(h3.get_resolution)
    par = [h3.cell_to_parent(x, r) == p for x, r, p in zip(pk.hx_at_pk, rp, pk.pickup_h3)]
    R.setdefault("pickup_h3", []).append({"day": day, "n": len(pk), "pickup_res": rp.value_counts().to_dict(),
        "drop_res": pk.drop_h3.map(h3.get_resolution).value_counts().to_dict(), "ping_hex_in_pickup_cell": float(pd.Series(par).mean()),
        "median_gap_s": float(pk.gap.median())})
    print(R["pickup_h3"][-1])
save("04_states.json", R)
