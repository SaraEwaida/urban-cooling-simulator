"""Vector layers (WGS84, WKB) burned onto the scene grid."""
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio.features import rasterize

from .grid import Grid


def to_grid_crs(geoms: np.ndarray, crs: str) -> np.ndarray:
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def fn(xy):
        x, y = t.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, fn)


def buildings(df: pd.DataFrame, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Building id raster (0 = none, k = row k-1 of df) and the Overture height per pixel (NaN if unknown)."""
    if len(df) == 0:
        return np.zeros(grid.shape, "int32"), np.full(grid.shape, np.nan, "float32")
    geoms = to_grid_crs(shapely.from_wkb(df["geometry"].to_numpy()), grid.crs)
    ids = rasterize(
        ((g, i + 1) for i, g in enumerate(geoms) if g is not None and not g.is_empty),
        out_shape=grid.shape, transform=grid.transform, fill=0, dtype="int32",
    )
    h = np.full(len(df) + 1, np.nan, dtype="float32")
    height = pd.to_numeric(df.get("height"), errors="coerce") if "height" in df else None
    floors = pd.to_numeric(df.get("num_floors"), errors="coerce") if "num_floors" in df else None
    if height is not None:
        h[1:] = height.to_numpy(dtype="float32")
    if floors is not None:
        fill = np.isnan(h[1:]) & ~np.isnan(floors.to_numpy(dtype="float32"))
        h[1:][fill] = floors.to_numpy(dtype="float32")[fill] * 3.5
    return ids, h[ids]


def roads(df: pd.DataFrame, grid: Grid, widths: dict) -> np.ndarray:
    """Boolean road surface: centre lines buffered to a width set by the road class."""
    if len(df) == 0:
        return np.zeros(grid.shape, bool)
    geoms = to_grid_crs(shapely.from_wkb(df["geometry"].to_numpy()), grid.crs)
    cls = df["class"].fillna("unknown").to_numpy() if "class" in df else np.full(len(df), "unknown")
    w = np.array([widths.get(c, widths["unknown"]) for c in cls], dtype=float)
    polys = shapely.buffer(geoms, w / 2.0, cap_style="flat")
    mask = rasterize(
        ((p, 1) for p in polys if p is not None and not p.is_empty),
        out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8",
    )
    return mask.astype(bool)
