import numpy as np
import pandas as pd

from ucs import solar


def synthetic_forcing(date="2023-07-15", days=3, t_min=31.0, t_max=44.0, wind=3.0, dew=8.0):
    """A clear Phoenix summer day repeated: sinusoidal air temperature (min 05:00, max 16:00 MST)."""
    t = pd.date_range(pd.Timestamp(date) - pd.Timedelta(days=days - 1), periods=24 * (days + 1), freq="h", tz="UTC") \
        + pd.Timedelta(hours=7)
    local_h = ((t - pd.Timedelta(hours=7)).hour).to_numpy()
    phase = np.where((local_h >= 5) & (local_h < 16), (local_h - 5) / 11 * np.pi, np.pi + (np.mod(local_h - 16, 24)) / 13 * np.pi)
    ta = t_min + (t_max - t_min) * (1 - np.cos(phase)) / 2
    el, _ = solar.sun_position(t - pd.Timedelta(minutes=30), 33.477, -112.09)  # radiation = preceding-hour mean
    s = np.clip(np.sin(np.radians(el)), 0, None)
    direct = 950 * s ** 1.15
    diffuse = 110 * s
    return pd.DataFrame({"time": t, "temperature_2m": ta, "dew_point_2m": dew, "surface_pressure": 965.0,
                         "wind_speed_10m": wind, "cloud_cover": 0.0, "shortwave_radiation": direct + diffuse,
                         "direct_radiation": direct, "diffuse_radiation": diffuse})
