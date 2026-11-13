"""Building footprints and road centre lines from Overture Maps (free, no account).

Both are stored as GeoParquet-like tables in WGS84 with a WKB `geometry` column.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BUILDING_COLS = ["id", "geometry", "height", "num_floors", "class"]
SEGMENT_COLS = ["id", "geometry", "subtype", "class"]


def _read(kind: str, bbox, cols: list[str]) -> pa.Table:
    from overturemaps import record_batch_reader

    reader = record_batch_reader(kind, bbox=tuple(bbox))
    if reader is None:
        raise RuntimeError(f"Overture returned nothing for {kind} in {bbox}")
    table = reader.read_all()
    keep = [c for c in cols if c in table.column_names]
    return table.select(keep)


def fetch_buildings(bbox, path: Path) -> int:
    table = _read("building", bbox, BUILDING_COLS)
    pq.write_table(table, path)
    return table.num_rows


def fetch_roads(bbox, path: Path) -> int:
    table = _read("segment", bbox, SEGMENT_COLS)
    df = table.to_pandas()
    df = df[df["subtype"] == "road"].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return len(df)


def load(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()
