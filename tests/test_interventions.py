import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from helpers_forcing import synthetic_forcing  # noqa: E402

from ucs import classify as C  # noqa: E402
from ucs import interventions as I  # noqa: E402
from ucs import plots, scene, shade, sun  # noqa: E402


def test_morphology():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    assert I.erode(m, 1).sum() == 16 and I.dilate(m, 1).sum() == 64
    p = np.zeros((11, 11), bool)
    p[5, 5] = True
    assert I.dilate(p, 2, disk=True).sum() == 13


def test_sail_shades_the_ground_under_it_and_beside_it():
    hb = np.zeros((40, 40), "float32")
    sail = np.zeros_like(hb)
    sail[10:20, 10:20] = 4.0
    s = shade.sunlit_fraction(hb, hb, 45.0, 180.0, 2.0, sail_h=sail, sail_transmissivity=0.1)
    assert s[15, 15] == np.float32(0.1)        # under the sail
    assert s[8, 15] == np.float32(0.1)         # 4 m north of a 4 m sail at a 45 degree sun
    assert s[5, 15] == 1.0 and s[25, 15] == 1.0


def _setup(cfg, synthetic):
    cfg["heat"]["spinup_days"] = 1
    scene.build(cfg, **synthetic, write=True)
    sun.run(cfg, verbose=False)
    synthetic_forcing(days=3).to_csv(cfg["paths"]["forcing"], index=False)
    return xr.open_zarr(cfg["paths"]["scene"]).load()


def test_tree_sites_follow_the_road_edge(cfg, synthetic):
    sc = _setup(cfg, synthetic)
    road = sc["road"].values.astype(bool)
    crowns, n = I.tree_sites(road, sc["surface_class"].values, 2.0, cfg["interventions"]["street_trees"])
    assert n > 0 and crowns.any()
    near = I.dilate(road, 3)
    assert (crowns <= near).all()                       # crowns stay within a crown radius of the road
    assert not (crowns & (sc["surface_class"].values == C.CODE["roof"])).any()


def test_interventions_cool_and_report(cfg, synthetic):
    sc = _setup(cfg, synthetic)
    cfg["interventions"]["scenarios"] = ["cool_roofs", "street_trees", "cool_pavement", "combined"]
    table, deltas, hours, _ = I.run(cfg, verbose=False)
    t = table.set_index("scenario")
    i15 = hours.index(15)
    roof = sc["surface_class"].values == C.CODE["roof"]
    assert deltas["cool_roofs"][i15][roof].mean() < -5           # reflective roofs run much cooler (synthetic roof albedo 0.46 -> 0.65)
    assert abs(deltas["cool_roofs"][i15][~roof]).max() < 1e-4     # nothing else changes: pixels are independent
    assert t.loc["street_trees", "d_street_15h"] < 0
    assert t.loc["combined", "d_scene_15h"] < min(t.loc[n, "d_scene_15h"] for n in ("cool_roofs", "street_trees"))
    gh = pd.read_csv(cfg["paths"]["geohash_interventions"])
    assert "d_cool_roofs_15h" in gh.columns and (gh["d_cool_roofs_15h"] <= 1e-6).all()
    path = plots.interventions_figure(table, deltas, hours, cfg, cfg["paths"]["interventions_figure"])
    assert Path(path).read_bytes()[:4] == b"\x89PNG"
