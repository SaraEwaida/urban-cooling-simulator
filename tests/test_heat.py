import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from helpers_forcing import synthetic_forcing  # noqa: E402

from ucs import classify as C  # noqa: E402
from ucs import forcing as F  # noqa: E402
from ucs import heat, plots, scene, sun  # noqa: E402
from ucs.energy import SIGMA, SurfaceModel  # noqa: E402


def _const(ta=35.0, direct=0.0, diffuse=0.0, wind=2.0):
    return {"temperature_2m": ta, "dew_point_2m": 5.0, "surface_pressure": 965.0, "wind_speed_10m": wind,
            "cloud_cover": 0.0, "direct_radiation": direct, "diffuse_radiation": diffuse}


def _model(cfg, names, svf=1.0, albedo=None):
    cls = np.array([[C.CODE[n] for n in names]], "uint8")
    alb = np.array([[albedo if albedo is not None else cfg["materials"][n]["albedo"] for n in names]], "float32")
    em = np.array([[cfg["materials"][n]["emissivity"] for n in names]], "float32")
    ev = np.array([[cfg["materials"][n]["evap"] for n in names]], "float32")
    return SurfaceModel(alb, em, ev, np.full(cls.shape, svf, "float32"), cls, C.CODE, cfg["thermal"],
                        cfg["heat"]["wind_a"], cfg["heat"]["wind_b"], cfg["heat"]["env_albedo"])


def _run_day(cfg, model, sunlit, days=3):
    at = F.interpolator(synthetic_forcing(days=days))
    start = pd.Timestamp("2023-07-13 07:00", tz="UTC")
    model.init(at(start.timestamp())["temperature_2m"], 3.0)
    rec, t = {}, start
    while t < start + pd.Timedelta(days=days):
        f = at(t.timestamp())
        if t >= start + pd.Timedelta(days=days - 1) and t.minute == 0:
            rec[(t - pd.Timedelta(hours=7)).hour] = model.ts[0].copy()
        model.step(f, sunlit, 300)
        t += pd.Timedelta(seconds=300)
    return rec


def test_dry_surface_without_sun_relaxes_to_air_temperature(cfg):
    # Surrounded by walls at air temperature (SVF 0), no sun, no evaporation: the only equilibrium is Ts = Ta
    m = _model(cfg, ["paved", "roof"], svf=0.0)
    m.init(20.0)
    m.K[:] = 0  # no deep-ground or interior exchange
    for _ in range(2000):
        m.step(_const(ta=35.0), np.zeros((1, 2), "float32"), 600)
    assert np.allclose(m.ts, 35.0, atol=0.05)


def test_step_is_stable_with_long_time_steps(cfg):
    m = _model(cfg, C.CLASSES)
    m.init(35.0)
    for _ in range(50):
        m.step(_const(direct=900, diffuse=100), np.ones((1, 6), "float32"), 3600)
    assert np.isfinite(m.ts).all() and m.ts.max() < 120


def test_afternoon_ordering_of_surfaces(cfg):
    names = ["roof", "paved", "bare_soil", "grass", "tree", "water"]
    rec = _run_day(cfg, _model(cfg, names), np.ones((1, 6), "float32"))
    ts = dict(zip(names, rec[14]))
    air = 43.0
    assert ts["roof"] > ts["paved"] > ts["bare_soil"] > air      # dry, sunlit surfaces run far above air
    assert ts["paved"] - air > 15
    assert ts["grass"] < air and ts["tree"] < air + 2           # transpiration keeps vegetation near or below air
    assert ts["water"] < ts["grass"] + 2                          # large heat capacity, evaporating
    assert abs(ts["tree"] - air) < 5


def test_shade_and_albedo_cool_pavement(cfg):
    sunny = _run_day(cfg, _model(cfg, ["paved"]), np.ones((1, 1), "float32"))
    shaded = _run_day(cfg, _model(cfg, ["paved"]), np.zeros((1, 1), "float32"))
    bright = _run_day(cfg, _model(cfg, ["paved"], albedo=0.35), np.ones((1, 1), "float32"))
    assert sunny[14][0] - shaded[14][0] > 15
    assert sunny[14][0] - bright[14][0] > 5


def test_vegetation_stops_transpiring_at_night(cfg):
    m = _model(cfg, ["grass", "tree"])
    m.init(30.0)
    night, _ = m.fluxes(_const(ta=32.0), np.zeros((1, 2), "float32"), m.ts)
    day, _ = m.fluxes(_const(ta=32.0, direct=800, diffuse=100), np.ones((1, 2), "float32"), m.ts)
    assert (night["le"] < 0.05 * day["le"]).all()


def test_forcing_parse_and_radiation_is_centred_on_the_half_hour():
    payload = {"hourly": {"time": ["2023-07-15T18:00", "2023-07-15T19:00"],
                          **{v: [0.0, 0.0] for v in F.VARS}}}
    payload["hourly"]["direct_radiation"] = [0.0, 600.0]   # mean over 18:00-19:00 is 600
    payload["hourly"]["temperature_2m"] = [30.0, 40.0]
    df = F.parse(payload)
    at = F.interpolator(df)
    t = pd.Timestamp("2023-07-15 18:30", tz="UTC").timestamp()
    assert at(t)["direct_radiation"] == 600.0
    assert at(t)["temperature_2m"] == 35.0


def test_heat_run_end_to_end(cfg, synthetic):
    cfg["heat"]["spinup_days"] = 1
    scene.build(cfg, **synthetic, write=True)
    sun.run(cfg, verbose=False)
    synthetic_forcing(days=3).to_csv(cfg["paths"]["forcing"], index=False)
    ds, by_class, summary = heat.run(cfg, verbose=False)
    assert ds["surface_temperature"].shape == (24, 100, 100)
    assert list(pd.DatetimeIndex(ds.time.values).hour) == list(range(24))
    assert summary["by_class"]["roof"]["ts_15h"] > summary["by_class"]["grass"]["ts_15h"]
    assert xr.open_zarr(cfg["paths"]["heat"])["surface_temperature_max"].shape == (100, 100)
    table = pd.read_csv(cfg["paths"]["geohash_heat"])
    assert {"mean_ts_11h", "mean_ts_15h", "mean_ts_05h", "mean_ts_max"} <= set(table.columns)
    path = plots.heat_figure(ds, by_class, cfg, cfg["paths"]["heat_figure"])
    assert Path(path).read_bytes()[:4] == b"\x89PNG"


def test_heat_uses_current_config_materials(cfg, synthetic):
    cfg["heat"]["spinup_days"] = 1
    cfg["roof_albedo"]["from_naip"] = False
    scene.build(cfg, **synthetic, write=True)
    sun.run(cfg, verbose=False)
    synthetic_forcing(days=3).to_csv(cfg["paths"]["forcing"], index=False)
    _, _, base = heat.run(cfg, verbose=False)
    cfg["materials"]["roof"]["albedo"] = 0.65  # a cool roof, changed after the scene was built
    _, _, cool = heat.run(cfg, verbose=False)
    assert base["by_class"]["roof"]["ts_15h"] - cool["by_class"]["roof"]["ts_15h"] > 10


def test_roof_albedo_follows_roof_brightness(cfg):
    naip = np.zeros((4, 4, 4), "float32")
    ids = np.zeros((4, 4), "int32")
    ids[:2, :2], ids[2:, 2:] = 1, 2
    naip[:3, :2, :2] = 60    # dark shingle roof
    naip[:3, 2:, 2:] = 220   # white roof
    naip[:3, 0, 0] = 255     # one bright pixel (a skylight) must not move the median much
    alb = C.roof_albedo(naip, ids, cfg["roof_albedo"])
    p = cfg["roof_albedo"]
    assert np.isnan(alb[0, 3]) and np.allclose(alb[:2, :2], alb[0, 1])
    assert p["dark_albedo"] <= alb[0, 1] < 0.15 < 0.55 < alb[3, 3] <= p["bright_albedo"]
