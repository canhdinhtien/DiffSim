"""Per-column profile of every table (duckdb SUMMARIZE + top values for low-cardinality, non-sensitive columns).
Sensitive columns (ids, names, coordinates, hexes) are masked: only type/null/distinct/length pattern reported.
hex_tracking and telemetry are profiled on a deterministic subset (see SAMPLE_*) — full-scan checks live in 02/03."""
import re, glob, os
from common import ROOT, OUT, TEL, P, con, save
SENS = re.compile(r"(^id$|_id$|^ping_id|vin|name|phone|email|^tel$|sap|plate|^lat$|^lng$|hex|h3|evidence|hex_history|customer)", re.I)
c = con()
def pattern(col, src):
    # shape of values, e.g. 'd-9999' -> 'a-9' ; never real values
    r = c.execute(f"""SELECT regexp_replace(regexp_replace(CAST("{col}" AS VARCHAR),'[0-9]+','9','g'),'[A-Za-z]+','a','g') p, count(*) n
                      FROM {src} WHERE "{col}" IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 3""").fetchall()
    return r
TABLES = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{ROOT}/*.parquet"))]
# hex_tracking: 10 evenly spaced row groups via file_row_number filter is not pushdown-friendly; use TABLESAMPLE on system blocks
SRC = {t: P(t) for t in TABLES}
SRC["public_driver_hex_tracking"] = f"(SELECT * FROM {P('public_driver_hex_tracking')} TABLESAMPLE 2% (system, 7))"
days = sorted(glob.glob(f"{ROOT}/telemetry/*.parquet"))[::12]  # 11 days spread over the range
tel_list = "[" + ",".join(f"'{d.replace(os.sep,'/')}'" for d in days) + "]"
SRC["telemetry_sample"] = f"(SELECT * EXCLUDE (seat_occupancy), seat_occupancy.driver AS seat_driver, seat_occupancy.any_passenger_seat AS seat_any_passenger FROM read_parquet({tel_list}))"
res = {}
for t, src in SRC.items():
    if t == "driver_penalization_ATA":
        res[t] = "EMPTY (0 rows, no schema)"; continue
    df = c.execute(f"SUMMARIZE SELECT * FROM {src}").df()
    cols = {}
    n = int(df["count"].iloc[0]) if len(df) else 0
    for _, r in df.iterrows():
        col = r.column_name; s = bool(SENS.search(col))
        d = {"type": r.column_type, "null_pct": float(r.null_percentage), "approx_distinct": int(r.approx_unique)}
        if s:
            d["masked"] = True; d["value_shape_top3"] = pattern(col, src)
        else:
            for k in ("min", "max", "avg", "std", "q25", "q50", "q75"):
                v = r[k]
                if v is not None and str(v) != "nan": d[k] = str(v)[:80]
            if r.approx_unique <= 40 and not r.column_type.startswith("STRUCT"):
                d["top"] = c.execute(f"""SELECT CAST("{col}" AS VARCHAR) v, count(*) n FROM {src} GROUP BY 1 ORDER BY 2 DESC LIMIT 12""").fetchall()
            elif r.column_type == "VARCHAR":
                d["len_q"] = c.execute(f"""SELECT min(length("{col}")), median(length("{col}")), max(length("{col}")) FROM {src}""").fetchone()
        cols[col] = d
    res[t] = {"rows_profiled": n, "columns": cols}
    print("done", t, n)
save("01_profile.json", res)
