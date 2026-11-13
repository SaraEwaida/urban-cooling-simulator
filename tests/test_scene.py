import json

import numpy as np
import xarray as xr

from ucs import classify as C
from ucs import geohash, plots, scene
from ucs.grid import Grid


def test_geohash_known_values():
    # Reference values from the original geohash.org examples
    assert geohash.encode([10.40744], [57.64911], 11)[0] == "u4pruydqqvj"
    assert geohash.encode([-5.6], [42.6], 5)[0] == "ezs42"
    lon0, lat0, lon1, lat1 = geohash.decode_bounds("ezs42")
    assert lon0 <= -5.6 <= lon1 and lat0 <= 42.6 <= lat1


def test_grid_is_snapped_and_square(cfg):
    g = Grid.from_config(cfg)
    assert g.shape == (100, 100)
    assert g.x0 % g.res == 0 and g.y1 % g.res == 0
    lon, lat = g.lonlat()
    assert abs(lon.mean() - cfg["center"][0]) < 0.001 and abs(lat.mean() - cfg["center"][1]) < 0.001


def test_classes_on_synthetic_scene(cfg, synthetic):
    ds = scene.build(cfg, **synthetic, write=False)
    cls = ds["surface_class"].values
    assert cls[20, 20] == C.CODE["tree"]          # tall vegetation
    assert cls[5, 40] == C.CODE["grass"]          # short vegetation
    assert cls[20, 90] == C.CODE["bare_soil"]     # bright, bare
    assert cls[50, 70] == C.CODE["roof"]          # building footprint
    assert cls[90, 80] == C.CODE["paved"]         # road buffer over bare ground
    assert cls[90, 30] == C.CODE["grass"]         # verge inside the buffer stays vegetation
    assert cls[75, 75] == C.CODE["water"]
    assert cls[10, 65] == C.CODE["tree"]          # dark vegetation with no height
    assert ds["canopy_height"].values[10, 65] == cfg["default_tree_height_m"]
    assert cls[90, 10] == C.CODE["water"]         # mapped pond, even though it looks green
    assert (cls > 0).all()


def test_building_height_prefers_lidar_then_overture(cfg, synthetic):
    ds = scene.build(cfg, **synthetic, write=False)
    assert np.nanmedian(ds["building_height"].values) == 7.0  # lidar median inside the footprint
    no_lidar = dict(synthetic, hag=np.full_like(synthetic["hag"], np.nan))
    ds2 = scene.build(cfg, **no_lidar, write=False)
    assert np.nanmedian(ds2["building_height"].values) == 7.0  # 2 floors x 3.5 m from Overture


def test_materials_follow_classes(cfg, synthetic):
    ds = scene.build(cfg, **synthetic, write=False)
    cls = ds["surface_class"].values
    alb = ds["albedo"].values
    for name, code in C.CODE.items():
        if (cls == code).any():
            assert np.allclose(alb[cls == code], cfg["materials"][name]["albedo"])


def test_written_outputs(cfg, synthetic):
    scene.build(cfg, **synthetic, write=True)
    ds = xr.open_zarr(cfg["paths"]["scene"])
    assert ds["surface_class"].shape == (100, 100)
    assert json.loads(ds.attrs["classes"])["tree"] == C.CODE["tree"]
    summary = json.load(open(cfg["paths"]["scene_summary"]))
    assert abs(sum(summary["class_fractions"].values()) - 1) < 1e-3
    import pandas as pd
    table = pd.read_csv(cfg["paths"]["geohash_table"])
    assert table["pixels"].sum() == 100 * 100
    fr = table[[c for c in table.columns if c.startswith("frac_")]].sum(axis=1)
    assert np.allclose(fr, 1)
    path = plots.scene_figure(ds.load(), cfg, cfg["paths"]["scene_figure"], table)
    assert open(path, "rb").read(4) == b"\x89PNG"


def test_canopy_height_map_replaces_missing_lidar(cfg, synthetic):
    chm = synthetic["hag"].copy()
    chm[40:60, 60:80] = 12.0  # the canopy model sees the roof as tall; it must not set roof height
    no_lidar = dict(synthetic, hag=np.full_like(synthetic["hag"], np.nan), chm=chm)
    ds = scene.build(cfg, **no_lidar, write=False)
    cls = ds["surface_class"].values
    assert cls[20, 20] == C.CODE["tree"] and cls[5, 40] == C.CODE["grass"]
    assert np.nanmedian(ds["canopy_height"].values) == 8.0
    assert np.nanmedian(ds["building_height"].values) == 7.0  # Overture floors, not the canopy map
    assert ds.attrs["canopy_source"] == "Meta/WRI canopy height map"
