import numpy as np

from ucs import classify as C
from ucs import landsat, validate


def test_block_mean_and_dominant_class():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    assert np.allclose(validate.block_mean(a, 2), [[2.5, 4.5], [10.5, 12.5]])
    a[0, 0] = np.nan
    assert np.isclose(validate.block_mean(a, 2)[0, 0], (1 + 4 + 5) / 3)
    cls = np.full((4, 4), C.CODE["paved"], "uint8")
    cls[:2, :2] = C.CODE["tree"]
    dom, share = validate.dominant_class(cls, 2)
    assert dom[0, 0] == "tree" and dom[1, 1] == "paved" and share.min() == 1


def test_compare_recovers_a_known_bias():
    rng = np.random.default_rng(1)
    truth_100m = rng.uniform(35, 60, (4, 4))
    sim = np.kron(truth_100m, np.ones((50, 50))) + 2.0          # 2 m grid, +2 C bias
    lst = np.kron(truth_100m, np.ones((10, 10)))                # 10 m grid
    lst[:10, :10] = np.nan                                      # one cloudy block
    cls = np.full(sim.shape, C.CODE["paved"], "uint8")
    res, table = validate.compare(sim, lst, cls, 2.0, 10.0, [100, 200])
    m = res["100m"]["all"]
    assert m["n"] == 15 and abs(m["bias"] - 2) < 1e-6 and m["rmse_bias_removed"] < 1e-6 and m["r"] > 0.999
    assert len(table) == 15 and res["200m"]["all"]["n"] == 3   # the 200 m block with the cloud is dropped


def test_gaussian_blur_keeps_the_mean_and_spreads_a_point():
    a = np.zeros((101, 101), "float32")
    a[50, 50] = 1.0
    b = validate.gaussian_blur(a, 5.0)
    assert abs(b.sum() - 1.0) < 1e-4 and b[50, 50] < 0.01
    assert abs(b[50, 55] / b[50, 50] - np.exp(-0.5)) < 0.01      # one sigma away
    c = np.full((40, 40), 30.0, "float32")
    assert np.allclose(validate.gaussian_blur(c, 8.0), 30.0, atol=1e-4)  # no edge darkening


def test_qa_bits_mask_clouds():
    # bit 3 cloud, bit 4 shadow, bit 1 dilated cloud; 21824 is a typical clear Landsat 8/9 value
    qa = np.array([21824, 21824 | (1 << 3), 21824 | (1 << 4), 21824 | (1 << 1)], "uint16")
    bad = np.zeros(qa.shape, bool)
    for b in landsat.BAD_BITS:
        bad |= ((qa >> b) & 1).astype(bool)
    assert bad.tolist() == [False, True, True, True]
    assert abs(44947 * landsat.ST_SCALE + landsat.ST_OFFSET - 302.6) < 0.05  # USGS worked example


def test_validate_run_end_to_end(cfg, synthetic, monkeypatch):
    """Landsat replaced by a fake scene: the run must simulate the overpass date and time and score it."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent))
    from helpers_forcing import synthetic_forcing

    from ucs import plots, scene

    cfg["heat"]["spinup_days"] = 1
    scene.build(cfg, **synthetic, write=True)
    overpass = pd.Timestamp("2023-07-15 17:50", tz="UTC")  # 10:50 MST
    fake_lst = np.full((20, 20), 50.0, "float32")            # 200 m scene on a 10 m grid
    item = SimpleNamespace(id="LC09_fake", datetime=overpass.to_pydatetime(), properties={"platform": "landsat-9"})
    monkeypatch.setattr(landsat, "clear_scenes", lambda grid, cfg, n=1, verbose=True: [(item, fake_lst, 1.0)])
    synthetic_forcing(days=3).to_csv(cfg["paths"]["forcing"].replace(".csv", "_20230715.csv"), index=False)

    summary, sim, lst, table = validate.run(cfg, verbose=False)
    assert summary["overpass_local"] == "2023-07-15 10:50"
    assert sim.shape == (100, 100) and summary["metrics"]["100m"]["all"]["n"] == 4
    assert abs(summary["metrics"]["100m"]["all"]["bias"] - (table["sim"].mean() - 50)) < 1e-3
    path = plots.validation_figure(summary, sim, lst, table, cfg, cfg["paths"]["validation_figure"])
    assert Path(path).read_bytes()[:4] == b"\x89PNG"


def test_calibration_recovers_a_known_grass_resistance(cfg, synthetic, monkeypatch):
    """Landsat replaced by the model's own output with grass r_s_min = 350; the search must find 350."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    import xarray as xr

    sys.path.insert(0, str(Path(__file__).parent))
    from helpers_forcing import synthetic_forcing

    from ucs import calibrate, scene
    from ucs.validate import Case, block_mean

    cfg["heat"]["spinup_days"] = 1
    cfg["landsat"]["psf_sigma_m"] = 0
    cfg["calibration"] = {"grass_r_s_min": [60, 200, 350, 600], "tree_r_s_min": [150]}
    scene.build(cfg, **synthetic, write=True)
    sc = xr.open_zarr(cfg["paths"]["scene"]).load()
    t = pd.Timestamp("2023-07-15 17:50", tz="UTC")
    synthetic_forcing(days=3).to_csv(cfg["paths"]["forcing"].replace(".csv", "_20230715.csv"), index=False)
    item = SimpleNamespace(id="fake", datetime=t.to_pydatetime(), properties={"platform": "landsat-9"})
    probe = Case(cfg, item, np.zeros((20, 20), "float32"), 1.0, sc)
    truth = probe.simulate({"thermal.grass.r_s_min": 350})
    lst = np.kron(block_mean(truth, 5), np.ones((1, 1))).astype("float32")  # 10 m "Landsat"
    case = Case(cfg, item, lst, 1.0, sc)
    best, table, _, _ = calibrate.search(case, cfg, verbose=False)
    assert best["thermal.grass.r_s_min"] == 350
    assert table.loc[table["value"] == 350, "rmse"].iloc[0] < 1e-3
