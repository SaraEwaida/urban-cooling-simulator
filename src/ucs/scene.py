"""Milestone 1: fetch the inputs and build the scene (classes, heights, materials) as Zarr."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from . import classify as C
from . import geohash, overture, rasterize
from .config import out_path
from .grid import Grid


def fetch(cfg: dict, verbose: bool = True) -> dict:
    """Download everything the scene needs (network). Writes vectors as parquet and rasters as .npz."""
    from . import imagery

    grid = Grid.from_config(cfg)
    bbox = grid.bbox_wgs84()
    log = print if verbose else (lambda *a: None)
    log(f"Scene {grid.nx}x{grid.ny} px at {grid.res} m, bbox {tuple(round(v, 4) for v in bbox)}")

    n = overture.fetch_buildings(bbox, out_path(cfg, "buildings"))
    log(f"Overture buildings: {n}")
    n = overture.fetch_roads(bbox, out_path(cfg, "roads"))
    log(f"Overture road segments: {n}")

    naip, naip_meta = imagery.fetch_naip(grid, cfg["sources"]["naip_years"])
    log(f"NAIP {naip_meta['year']}: {len(naip_meta['items'])} tile(s), {naip_meta['dates']}")
    hag, hag_meta = imagery.fetch_lidar(grid, cfg["sources"]["lidar_collection"])
    log(f"Lidar height above ground: coverage {hag_meta['coverage']:.0%}, {hag_meta['items']}")
    dtm, dtm_meta = imagery.fetch_lidar(grid, "3dep-lidar-dtm")
    log(f"Lidar terrain model: coverage {dtm_meta['coverage']:.0%}")

    meta = {"naip": naip_meta, "hag": hag_meta, "dtm": dtm_meta,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    np.savez_compressed(out_path(cfg, "rasters"), naip=naip, hag=hag, dtm=dtm,
                        meta=np.array(json.dumps(meta)))
    return meta


def build(cfg: dict, naip=None, hag=None, dtm=None, buildings_df=None, roads_df=None,
          meta: dict | None = None, write: bool = True) -> xr.Dataset:
    """Classify the surface and assemble the scene. Inputs default to the files written by fetch()."""
    grid = Grid.from_config(cfg)
    if naip is None:
        z = np.load(out_path(cfg, "rasters"))
        naip, hag, dtm = z["naip"], z["hag"], z["dtm"]
        meta = json.loads(str(z["meta"]))
    if buildings_df is None:
        buildings_df = overture.load(out_path(cfg, "buildings"))
        roads_df = overture.load(out_path(cfg, "roads"))
    meta = meta or {}

    bid, ov_h = rasterize.buildings(buildings_df, grid)
    road = rasterize.roads(roads_df, grid, cfg["classify"]["road_width_m"])
    classes, notes = C.classify(naip, hag, bid, road, cfg["classify"])
    b_h = C.building_heights(bid, hag, ov_h, cfg["default_building_height_m"])
    t_h = C.canopy_height(classes, hag)
    mats = C.material_maps(classes, cfg["materials"])

    x, y = grid.coords()
    dims = ("y", "x")
    ds = xr.Dataset(
        {
            "naip": (("band", "y", "x"), naip.astype("float32")),
            "ndvi": (dims, C.ndvi(naip)),
            "surface_class": (dims, classes),
            "building_id": (dims, bid),
            "building_height": (dims, b_h),
            "canopy_height": (dims, t_h),
            "height_above_ground": (dims, hag.astype("float32")),
            "terrain": (dims, dtm.astype("float32")),
            "road": (dims, road),
            **{k: (dims, v) for k, v in mats.items()},
        },
        coords={"x": x, "y": y, "band": ["red", "green", "blue", "nir"]},
        attrs={
            "name": cfg["name"],
            "crs": grid.crs,
            "res_m": grid.res,
            "transform": list(grid.transform)[:6],
            "classes": json.dumps(C.CODE),
            "sources": json.dumps(meta),
            "lidar_fraction": notes["lidar_fraction"],
        },
    )
    ds["surface_class"].attrs["flag_meanings"] = " ".join(C.CLASSES)
    ds["surface_class"].attrs["flag_values"] = list(C.CODE.values())
    for v in ("building_height", "canopy_height", "height_above_ground", "terrain"):
        ds[v].attrs["units"] = "m"

    if write:
        ds.to_zarr(out_path(cfg, "scene"), mode="w", zarr_format=2)
        table = geohash_table(ds, grid, cfg["geohash_precision"])
        table.to_csv(out_path(cfg, "geohash_table"), index=False, float_format="%.4f")
        summary = summarize(ds, table)
        with open(out_path(cfg, "scene_summary"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return ds


def geohash_table(ds: xr.Dataset, grid: Grid, precision: int) -> pd.DataFrame:
    """One row per geohash cell: class fractions, tree and building cover, heights, mean albedo."""
    lon, lat = grid.lonlat()
    gh = geohash.encode(lon.ravel(), lat.ravel(), precision)
    cls = ds["surface_class"].values.ravel()
    df = pd.DataFrame({"geohash": gh, "cls": cls,
                       "albedo": ds["albedo"].values.ravel(),
                       "bh": ds["building_height"].values.ravel(),
                       "ch": ds["canopy_height"].values.ravel()})
    df = df[df["cls"] > 0]
    g = df.groupby("geohash")
    out = pd.DataFrame({"pixels": g.size()})
    for name, code in C.CODE.items():
        out[f"frac_{name}"] = g["cls"].apply(lambda s, c=code: float((s == c).mean()))
    out["mean_albedo"] = g["albedo"].mean()
    out["mean_building_height_m"] = g["bh"].mean()
    out["mean_canopy_height_m"] = g["ch"].mean()
    return out.reset_index()


def summarize(ds: xr.Dataset, table: pd.DataFrame) -> dict:
    cls = ds["surface_class"].values
    valid = cls > 0
    frac = {name: round(float((cls[valid] == code).mean()), 4) for name, code in C.CODE.items()}
    bh = ds["building_height"].values
    return {
        "name": ds.attrs["name"],
        "grid": {"crs": ds.attrs["crs"], "res_m": ds.attrs["res_m"],
                 "shape": list(cls.shape)},
        "valid_fraction": round(float(valid.mean()), 4),
        "lidar_fraction": round(float(ds.attrs["lidar_fraction"]), 4),
        "class_fractions": frac,
        "buildings": int(len(np.unique(ds["building_id"].values)) - 1),
        "building_height_m": {"median": round(float(np.nanmedian(bh)), 2) if np.isfinite(bh).any() else None,
                              "p95": round(float(np.nanpercentile(bh, 95)), 2) if np.isfinite(bh).any() else None},
        "mean_albedo": round(float(np.nanmean(ds["albedo"].values)), 4),
        "geohash_cells": int(len(table)),
        "sources": json.loads(ds.attrs["sources"]),
    }
