"""Inventory: files, sizes, row counts, schemas (parquet metadata only)."""
import glob, os
import pyarrow.parquet as pq
from common import ROOT, save
inv = []
for f in sorted(glob.glob(f"{ROOT}/*.parquet")) + sorted(glob.glob(f"{ROOT}/telemetry/*.parquet"))[:1]:
    m = pq.ParquetFile(f)
    inv.append({"file": os.path.relpath(f, ROOT), "size_mb": round(os.path.getsize(f)/1e6, 2),
                "rows": m.metadata.num_rows, "row_groups": m.metadata.num_row_groups,
                "schema": {n: str(t) for n, t in zip(m.schema_arrow.names, m.schema_arrow.types)}})
tel = sorted(glob.glob(f"{ROOT}/telemetry/*.parquet"))
tel_rows = [pq.ParquetFile(f).metadata.num_rows for f in tel]
inv.append({"telemetry_files": len(tel), "first": os.path.basename(tel[0]), "last": os.path.basename(tel[-1]),
            "total_rows": sum(tel_rows), "rows_min": min(tel_rows), "rows_max": max(tel_rows),
            "total_size_mb": round(sum(os.path.getsize(f) for f in tel)/1e6, 1)})
save("00_schema.json", inv)
for x in inv: print(x)
