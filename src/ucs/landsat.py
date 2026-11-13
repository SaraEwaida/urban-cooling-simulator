"""Landsat 8/9 Collection 2 Level-2 surface temperature from Planetary Computer.

ST_B10 is delivered on a 30 m grid but measured by TIRS at 100 m, so comparisons are made
on blocks of at least 100 m. Pixels flagged by CFMask (fill, dilated cloud, cirrus, cloud,
cloud shadow) are dropped.
"""
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from .grid import Grid
from .imagery import _client, warp_to_grid

ST_SCALE, ST_OFFSET = 0.00341802, 149.0
BAD_BITS = (0, 1, 2, 3, 4)  # fill, dilated cloud, cirrus, cloud, cloud shadow


def _asset(item, *keys):
    for k in keys:
        if k in item.assets:
            return item.assets[k].href
    raise KeyError(f"none of {keys} in {item.id}")


def candidates(grid: Grid, window: str, max_cloud: float):
    search = _client().search(collections=["landsat-c2-l2"], bbox=grid.bbox_wgs84(), datetime=window,
                              query={"eo:cloud_cover": {"lt": max_cloud},
                                     "platform": {"in": ["landsat-8", "landsat-9"]}})
    return list(search.items())


def read(item, grid: Grid, read_res: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Surface temperature (C) and a clear-sky mask on a `read_res` grid with the scene's extent."""
    fine = Grid(grid.x0, grid.y1, read_res, int(grid.nx * grid.res / read_res), int(grid.ny * grid.res / read_res),
                grid.crs)
    dn = warp_to_grid(_asset(item, "lwir11", "ST_B10"), fine, [1], Resampling.nearest, 0)[0]
    qa = warp_to_grid(_asset(item, "qa_pixel", "QA_PIXEL"), fine, [1], Resampling.nearest, 1)[0]
    lst = dn * ST_SCALE + ST_OFFSET - 273.15
    qa_i = np.nan_to_num(qa, nan=1).astype("uint16")
    bad = np.zeros(qa_i.shape, bool)
    for b in BAD_BITS:
        bad |= ((qa_i >> b) & 1).astype(bool)
    clear = ~bad & ~np.isnan(dn)
    return np.where(clear, lst, np.nan).astype("float32"), clear


def clear_scenes(grid: Grid, cfg: dict, n: int = 1, verbose=True):
    """The `n` clear scenes closest to the target date, at least `min_gap_days` apart.

    Returns a list of (item, lst at read_res, clear fraction), closest first.
    """
    lc = cfg["landsat"]
    target = pd.Timestamp(cfg["sun"]["date"], tz="UTC")
    items = candidates(grid, lc["window"], lc["max_scene_cloud"])
    items.sort(key=lambda it: abs(it.datetime - target))
    found = []
    for it in items:
        if any(abs((it.datetime - f[0].datetime).total_seconds()) < lc.get("min_gap_days", 0) * 86400 for f in found):
            continue
        lst, clear = read(it, grid, lc["read_res_m"])
        frac = float(clear.mean())
        if verbose:
            print(f"  {it.id} {it.datetime:%Y-%m-%d %H:%M} UTC: clear over scene {frac:.0%}")
        if frac >= lc["min_clear"]:
            found.append((it, lst, frac))
            if len(found) == n:
                return found
    raise RuntimeError(f"Only {len(found)} of {n} Landsat scenes in {lc['window']} are "
                       f"{lc['min_clear']:.0%} clear over the scene")
