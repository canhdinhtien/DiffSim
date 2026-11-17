"""Shared helpers. READ-ONLY access to realdata-v1."""
import os, json, duckdb
ROOT = os.environ.get("REALDATA_ROOT", r"D:\Projects\DiffSim\data\mock\realdata-v1").replace("\\", "/")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
TEL = f"{ROOT}/telemetry/vehicle_telemetry_ping_*.parquet"

def P(name):
    return f"read_parquet('{ROOT}/{name}.parquet')"

def con(mem="10GB"):
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{mem}'; SET threads=12; SET preserve_insertion_order=false;")
    c.execute(f"SET temp_directory='{OUT}/_duck_tmp'")
    return c

def save(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str, ensure_ascii=False)
