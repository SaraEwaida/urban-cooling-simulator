import numpy as np
import pandas as pd
import xarray as xr

from ucs import scene, shade, solar, sun


def test_solstice_noon_elevation_and_azimuth():
    # Max elevation at the June solstice is 90 - latitude + 23.44 (obliquity); the sun is due south then
    t = pd.date_range("2023-06-21 17:00", "2023-06-21 22:00", freq="1min", tz="UTC")
    e, a = solar.sun_position(t, 33.477, -112.09)
    i = int(np.argmax(e))
    assert abs(e[i] - (90 - 33.477 + 23.44)) < 0.1
    assert abs(a[i] - 180) < 1.0
    # Arizona solar noon is about 19:28 UTC (12:28 MST) at this longitude
    assert t[i].hour == 19 and 20 <= t[i].minute <= 35


def test_morning_sun_is_in_the_east_and_night_is_dark():
    e, a = solar.sun_position(["2023-07-15 15:00Z", "2023-07-15 09:00Z"], 33.477, -112.09)
    assert 0 < e[0] < 45 and 60 < a[0] < 110       # 08:00 MST, low in the east-north-east
    assert e[1] < 0                                # 02:00 MST
    hb = np.zeros((10, 10))
    assert shade.sunlit_fraction(hb, hb, e[1], a[1], 2.0).max() == 0


def test_building_shadow_length():
    # A 10 m block with the sun due south at 45 degrees casts a 10 m (5 pixel) shadow to the north
    hb = np.zeros((60, 60), "float32")
    hb[30:40, 20:40] = 10.0
    s = shade.sunlit_fraction(hb, np.zeros_like(hb), 45.0, 180.0, 2.0)
    assert (s[30:40, 20:40] == 1).all()            # the roof itself is sunlit
    assert (s[25:30, 22:38] == 0).all()            # rows 25-29: within 10 m north of the block
    assert (s[22:24, 22:38] == 1).all()            # beyond the shadow
    assert (s[40:, :] == 1).all()                  # nothing south of it is shaded


def test_tree_shade_is_partial_and_building_wins():
    hb = np.zeros((40, 40), "float32")
    ht = np.zeros((40, 40), "float32")
    ht[20:24, 20:24] = 6.0
    s = shade.sunlit_fraction(hb, ht, 45.0, 180.0, 2.0, tree_transmissivity=0.25)
    assert s[18, 21] == 0.25
    hb[20:24, 10:14] = 6.0
    s = shade.sunlit_fraction(hb, ht, 45.0, 180.0, 2.0)
    assert s[18, 11] == 0.0


def test_sky_view_factor_open_ground_and_street_canyon():
    flat = np.zeros((50, 50), "float32")
    assert np.allclose(shade.sky_view_factor(flat, 2.0, 16, 40), 1.0)
    # Infinite canyon, walls 10 m high, 10 m apart: SVF at the centre = cos(atan(2H/W)) = 1/sqrt(5) (Oke 1981)
    h = np.zeros((200, 101), "float32")
    h[:, :45] = 10.0
    h[:, 50:] = 10.0  # street = columns 45-49 (10 m), centre column 47
    svf = shade.sky_view_factor(h, 2.0, 64, 150)
    assert abs(svf[100, 47] - 1 / np.sqrt(5)) < 0.01
    assert abs(svf[100, 20] - 1.0) < 1e-6         # flat roof top sees the whole sky


def test_sun_run_writes_outputs(cfg, synthetic):
    scene.build(cfg, **synthetic, write=True)
    ds = sun.run(cfg, verbose=False)
    loaded = xr.open_zarr(cfg["paths"]["sun"])
    assert loaded["sunlit"].shape == (len(cfg["sun"]["hours_local"]), 100, 100)
    assert float(ds["sunlit_hours"].max()) <= len(cfg["sun"]["hours_local"])
    # The 7 m building casts a shadow in the afternoon
    i = cfg["sun"]["hours_local"].index(15)
    assert ds["sunlit"].values[i].min() == 0
    table = pd.read_csv(cfg["paths"]["geohash_sun"])
    assert {"mean_svf", "mean_sunlit_hours", "mean_shade_09h", "mean_shade_15h"} <= set(table.columns)
    from ucs import plots
    path = plots.sun_figure(ds, cfg, cfg["paths"]["sun_figure"])
    assert open(path, "rb").read(4) == b"\x89PNG"
