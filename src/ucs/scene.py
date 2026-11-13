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
    n = overture.fetch_water(bbox, out_path(cfg, "water"))
    log(f"Overture water polygons: {n}")

    naip, naip_meta = imagery.fetch_naip(grid, cfg["sources"]["naip_years"])
    log(f"NAIP {naip_meta['year']}: {len(naip_meta['items'])} tile(s), {naip_meta['dates']}")
    hag, hag_meta = imagery.fetch_lidar(grid, cfg["sources"]["lidar_collection"])
    log(f"Lidar height above ground: coverage {hag_meta['coverage']:.0%}, {hag_meta['items']}")
    chm = np.full(grid.shape, np.nan, dtype="float32")
    chm_meta = {"tiles": [], "coverage": 0.0}
    if hag_meta["coverage"] < 0.5:
        chm, chm_meta = imagery.fetch_canopy_height(grid, out_path(cfg, "chm_index"))
        log(f"Meta/WRI canopy height (lidar fallback): coverage {chm_meta['coverage']:.0%}, tiles {chm_meta['tiles']}")
    dtm, dtm_meta = imagery.fetch_lidar(grid, "3dep-lidar-dtm")
    if dtm_meta["coverage"] < 0.5:
        dtm, dtm_meta = imagery.fetch_lidar(grid, "cop-dem-glo-30")
        dtm_meta["source"] = "cop-dem-glo-30"
    log(f"Terrain ({dtm_meta.get('source', '3dep-lidar-dtm')}): coverage {dtm_meta['coverage']:.0%}")

    meta = {"naip": naip_meta, "hag": hag_meta, "chm": chm_meta, "dtm": dtm_meta,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    np.savez_compressed(out_path(cfg, "rasters"), naip=naip, hag=hag, chm=chm, dtm=dtm,
                        meta=np.array(json.dumps(meta)))
    return meta


def build(cfg: dict, naip=None, hag=None, dtm=None, chm=None, buildings_df=None, roads_df=None, water_df=None,
          meta: dict | None = None, write: bool = True) -> xr.Dataset:
    """Classify the surface and assemble the scene. Inputs default to the files written by fetch()."""
    grid = Grid.from_config(cfg)
    if naip is None:
        z = np.load(out_path(cfg, "rasters"))
        naip, hag, dtm = z["naip"], z["hag"], z["dtm"]
        chm = z["chm"] if "chm" in z else None
        meta = json.loads(str(z["meta"]))
    if buildings_df is None:
        buildings_df = overture.load(out_path(cfg, "buildings"))
        roads_df = overture.load(out_path(cfg, "roads"))
        water_df = overture.load(out_path(cfg, "water"))
    meta = meta or {}

    if chm is None:
        chm = np.full(grid.shape, np.nan, dtype="float32")
    # Vegetation height: lidar where it exists, else the canopy height map. Roofs use lidar only,
    # because the canopy model is not trained on buildings.
    veg_h = np.where(np.isnan(hag), chm, hag).astype("float32")
    canopy_source = ("3DEP lidar" if np.mean(~np.isnan(hag)) > 0.5
                     else "Meta/WRI canopy height map" if np.mean(~np.isnan(chm)) > 0.5 else "none")

    bid, ov_h = rasterize.buildings(buildings_df, grid)
    road = rasterize.roads(roads_df, grid, cfg["classify"]["road_width_m"])
    water = rasterize.polygons(water_df, grid) if water_df is not None else None
    classes, notes = C.classify(naip, veg_h, bid, road, cfg["classify"], water)
    b_h = C.building_heights(bid, hag, ov_h, cfg["default_building_height_m"])
    t_h = C.canopy_height(classes, veg_h, cfg["classify"]["tree_min_height_m"],
                           cfg["default_tree_height_m"])
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
            "canopy_height_map": (dims, chm.astype("float32")),
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
            "lidar_fraction": float(np.mean(~np.isnan(hag))),
            "canopy_source": canopy_source,
        },
    )
    ds["surface_class"].attrs["flag_meanings"] = " ".join(C.CLASSES)
    ds["surface_class"].attrs["flag_values"] = list(C.CODE.values())
    for v in ("building_height", "canopy_height", "height_above_ground", "canopy_height_map", "terrain"):
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
        "canopy_source": ds.attrs["canopy_source"],
        "class_fractions": frac,
        "buildings": int(len(np.unique(ds["building_id"].values)) - 1),
        "building_height_m": {"median": round(float(np.nanmedian(bh)), 2) if np.isfinite(bh).any() else None,
                              "p95": round(float(np.nanpercentile(bh, 95)), 2) if np.isfinite(bh).any() else None},
        "mean_albedo": round(float(np.nanmean(ds["albedo"].values)), 4),
        "geohash_cells": int(len(table)),
        "sources": json.loads(ds.attrs["sources"]),
    }
