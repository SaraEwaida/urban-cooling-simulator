"""Rasters onto the scene grid, all free and anonymous:
NAIP imagery, 3DEP lidar and Copernicus DEM from Microsoft Planetary Computer, and the
Meta/WRI 1 m global canopy height map from the AWS Open Data registry.

Every source is warped straight onto the scene grid with GDAL (WarpedVRT), so only the
needed part of each cloud-optimised GeoTIFF is downloaded.
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .grid import Grid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def warp_to_grid(href: str, grid: Grid, bands, resampling: Resampling, nodata) -> np.ndarray:
    """Read `bands` of a raster onto the grid; pixels outside the source are NaN."""
    with rasterio.open(href) as src:
        src_nodata = src.nodata if src.nodata is not None else nodata
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.nx,
            height=grid.ny,
            resampling=resampling,
            src_nodata=src_nodata,
            nodata=src_nodata,
        ) as vrt:
            arr = vrt.read(list(bands), masked=True).astype("float32")
    return arr.filled(np.nan)


def mosaic(items, asset: str, grid: Grid, bands, resampling, nodata) -> tuple[np.ndarray, list[str]]:
    """Fill the grid from items in order (newest first); returns the stack and the item ids used."""
    out = np.full((len(bands), *grid.shape), np.nan, dtype="float32")
    used = []
    for item in items:
        missing = np.isnan(out[0])
        if not missing.any():
            break
        arr = warp_to_grid(item.assets[asset].href, grid, bands, resampling, nodata)
        take = missing & ~np.isnan(arr[0])
        if take.any():
            out[:, take] = arr[:, take]
            used.append(item.id)
    return out, used


def _date(item) -> str:
    d = item.datetime or item.properties.get("start_datetime") or ""
    return str(d)[:10]


def _items(collection: str, bbox, datetime=None):
    search = _client().search(collections=[collection], bbox=bbox, datetime=datetime)
    items = list(search.items())
    items.sort(key=_date, reverse=True)
    return items


def fetch_naip(grid: Grid, years: list[int]) -> tuple[np.ndarray, dict]:
    """NAIP R, G, B, NIR (digital numbers 0-255) averaged from 0.6 m to the grid resolution."""
    bbox = grid.bbox_wgs84()
    for year in years:
        items = _items("naip", bbox, f"{year}-01-01/{year}-12-31")
        if not items:
            continue
        arr, used = mosaic(items, "image", grid, [1, 2, 3, 4], Resampling.average, 0)
        cover = float(np.mean(~np.isnan(arr[0])))
        if cover > 0.99:
            dates = sorted({_date(it) for it in items if it.id in used})
            return arr, {"year": year, "items": used, "dates": dates, "coverage": cover}
    raise RuntimeError(f"No NAIP flight in {years} fully covers the scene")


def fetch_lidar(grid: Grid, collection: str) -> tuple[np.ndarray, dict]:
    """A single-band 3DEP lidar product (e.g. height above ground, terrain model) on the grid."""
    items = _items(collection, grid.bbox_wgs84())
    if not items:
        return np.full(grid.shape, np.nan, dtype="float32"), {"items": [], "coverage": 0.0}
    arr, used = mosaic(items, "data", grid, [1], Resampling.bilinear, -9999)
    meta = {
        "items": used,
        "dates": sorted({_date(it) for it in items if it.id in used}),
        "coverage": float(np.mean(~np.isnan(arr[0]))),
    }
    return arr[0], meta


CHM_ROOT = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"


def chm_tiles(bbox, cache_path) -> list[str]:
    """Names of the canopy-height tiles intersecting a lon/lat box (tile index cached locally)."""
    import json
    from pathlib import Path

    import requests
    import shapely

    cache_path = Path(cache_path)
    if not cache_path.exists():
        r = requests.get(f"{CHM_ROOT}/tiles.geojson", timeout=300)
        r.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(r.content)
    feats = json.loads(cache_path.read_text(encoding="utf-8"))["features"]
    box = shapely.box(*bbox)
    return [f["properties"]["tile"] for f in feats if shapely.geometry.shape(f["geometry"]).intersects(box)]


def fetch_canopy_height(grid: Grid, cache_path) -> tuple[np.ndarray, dict]:
    """Meta/WRI canopy height (m, about 1 m, from Maxar imagery) averaged onto the grid."""
    tiles = chm_tiles(grid.bbox_wgs84(), cache_path)
    out = np.full(grid.shape, np.nan, dtype="float32")
    for t in tiles:
        arr = warp_to_grid(f"{CHM_ROOT}/chm/{t}.tif", grid, [1], Resampling.average, -1)[0]
        take = np.isnan(out) & ~np.isnan(arr)
        out[take] = arr[take]
    return out, {"tiles": tiles, "coverage": float(np.mean(~np.isnan(out)))}
