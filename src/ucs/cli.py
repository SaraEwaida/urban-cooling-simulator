import argparse
import json

from . import calibrate, heat, plots, scene, sun, validate
from .config import load_config, out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="ucs", description="Urban cooling simulator")
    p.add_argument("command", choices=["fetch", "scene", "sun", "heat", "validate", "calibrate", "all"],
                   help="fetch: download inputs; scene: classify, write Zarr, table and figure; "
                        "sun: hourly shade and sky view factor; heat: surface energy balance; "
                        "validate: compare with Landsat; calibrate: tune vegetation on one Landsat date "
                        "and test on another; all: scene to validation")
    p.add_argument("--config", default="configs/encanto.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.command in ("fetch", "all"):
        scene.fetch(cfg)
    if a.command in ("scene", "all"):
        ds = scene.build(cfg)
        fig = plots.scene_figure(ds, cfg, out_path(cfg, "scene_figure"))
        with open(cfg["paths"]["scene_summary"], encoding="utf-8") as f:
            s = json.load(f)
        print(json.dumps({k: s[k] for k in ("valid_fraction", "lidar_fraction", "canopy_source", "class_fractions",
                                             "buildings", "building_height_m", "geohash_cells")}, indent=2))
        print(f"Scene: {cfg['paths']['scene']}\nFigure: {fig}")
    if a.command in ("sun", "all"):
        print(f"Sun and shade on {cfg['sun']['date']}:")
        ds = sun.run(cfg)
        fig = plots.sun_figure(ds, cfg, out_path(cfg, "sun_figure"))
        with open(cfg["paths"]["sun_summary"], encoding="utf-8") as f:
            s = json.load(f)
        print(json.dumps({"by_class": s["by_class"], "scene_mean": s["scene_mean"]}, indent=2))
        print(f"Sun: {cfg['paths']['sun']}\nFigure: {fig}")
    if a.command in ("heat", "all"):
        print(f"Surface energy balance for {cfg['sun']['date']} (after {cfg['heat']['spinup_days']} spin-up days):")
        ds, by_class, s = heat.run(cfg)
        fig = plots.heat_figure(ds, by_class, cfg, out_path(cfg, "heat_figure"))
        print(json.dumps({k: s[k] for k in ("air_temperature", "by_class", "scene_mean")}, indent=2))
        print(f"Heat: {cfg['paths']['heat']}\nFigure: {fig}")
    if a.command in ("validate", "all"):
        s, sim, lst, table = validate.run(cfg)
        fig = plots.validation_figure(s, sim, lst, table, cfg, out_path(cfg, "validation_figure"))
        print(json.dumps({k: s[k] for k in ("landsat_item", "overpass_local", "clear_share",
                                             "landsat_scene_mean", "sim_scene_mean")}, indent=2))
        for r, res in s["metrics"].items():
            print(f"{r} blocks:")
            for k, m in res.items():
                print(f"  {k:10s} n={m['n']:4d}  bias {m['bias']:+6.2f}  MAE {m['mae']:5.2f}  RMSE {m['rmse']:5.2f}"
                      f"  RMSE bias removed {m['rmse_bias_removed']:5.2f}  r {m['r']:5.2f}")
        print(f"Figure: {fig}")
    if a.command == "calibrate":
        out, sims = calibrate.run(cfg)
        case, tuned, table, summ = sims["test"]
        fig = plots.validation_figure(summ, tuned, case.lst, table, cfg, out_path(cfg, "test_figure"),
                                      " - held-out date, tuned")
        print("Tuned:", json.dumps(out["tuned"]))
        for label in ("calibration", "test"):
            for k in ("baseline", "tuned"):
                m = out[f"{label}_scene"][k]["metrics"]["100m"]
                by = ", ".join(f"{c} {m[c]['bias']:+.1f}" for c in ("paved", "grass", "roof", "tree") if c in m)
                print(f"  {label:11s} {k:8s}: MAE {m['all']['mae']:.2f}  RMSE {m['all']['rmse']:.2f}  "
                      f"bias {m['all']['bias']:+.2f}  RMSE bias removed {m['all']['rmse_bias_removed']:.2f}  "
                      f"r {m['all']['r']:.2f}  | class bias: {by}")
        print(f"Figure (held-out date): {fig}")


if __name__ == "__main__":
    main()
