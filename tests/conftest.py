import copy

import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Transformer

from ucs.config import load_config
from ucs.grid import Grid


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config("configs/encanto.yaml"))
    c["size_m"] = 200  # 100 x 100 px
    c["paths"] = {k: str(tmp_path / v.split("/")[-1]) for k, v in c["paths"].items()}
    return c


def _wgs84(geom, crs):
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return shapely.transform(geom, lambda xy: np.column_stack(t.transform(xy[:, 0], xy[:, 1])))


@pytest.fixture
def synthetic(cfg):
    """A 200 m scene: west half grass with a tree clump, east half bare, a building, a road, a pond."""
    g = Grid.from_config(cfg)
    ny, nx = g.shape
    naip = np.zeros((4, ny, nx), "float32")
    naip[:] = np.array([180, 160, 140, 190], "float32")[:, None, None]  # bright bare soil, NDVI ~0.03
    naip[:, :, : nx // 2] = np.array([80, 110, 70, 200], "float32")[:, None, None]  # grass NDVI ~0.43
    naip[:, 70:80, 70:80] = np.array([30, 40, 50, 25], "float32")[:, None, None]  # dark water, NDVI < 0
    naip[:, 5:15, 60:70] = np.array([40, 60, 35, 160], "float32")[:, None, None]  # dark green: canopy, no height
    naip[:, 85:95, 5:15] = np.array([60, 90, 70, 110], "float32")[:, None, None]  # green-tinted pond
    hag = np.zeros((ny, nx), "float32")
    hag[10:30, 10:30] = 8.0  # tree clump
    hag[40:60, 60:80] = 7.0  # building roof
    dtm = np.full((ny, nx), 330.0, "float32")

    x0, y0, x1, y1 = g.bounds
    bldg = shapely.box(x0 + 120, y1 - 120, x0 + 160, y1 - 80)  # rows 40-60, cols 60-80
    road = shapely.LineString([(x0, y1 - 180), (x1, y1 - 180)])  # horizontal at row 90
    b = pd.DataFrame({"id": ["b1"], "geometry": [shapely.to_wkb(_wgs84(bldg, g.crs))],
                      "height": [np.nan], "num_floors": [2]})
    r = pd.DataFrame({"id": ["r1"], "geometry": [shapely.to_wkb(_wgs84(road, g.crs))],
                      "subtype": ["road"], "class": ["residential"]})
    pond = shapely.box(x0 + 10, y1 - 190, x0 + 30, y1 - 170)  # rows 85-95, cols 5-15
    w = pd.DataFrame({"id": ["w1"], "geometry": [shapely.to_wkb(_wgs84(pond, g.crs))],
                      "subtype": ["lake"], "class": ["pond"]})
    return dict(naip=naip, hag=hag, dtm=dtm, buildings_df=b, roads_df=r, water_df=w,
                meta={"naip": {"year": 2023}})
