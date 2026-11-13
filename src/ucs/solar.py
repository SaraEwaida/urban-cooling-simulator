"""Sun position from the NOAA solar calculator equations (accuracy about 0.01 degrees for 1800-2100)."""
import numpy as np
import pandas as pd


def sun_position(times_utc, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Solar elevation and azimuth (degrees; azimuth clockwise from north) for UTC times.

    Elevation includes the standard atmospheric refraction correction.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times_utc, utc=True))
    jd = t.to_julian_date().to_numpy()
    jc = (jd - 2451545.0) / 36525.0

    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)  # mean longitude
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)  # mean anomaly
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)  # orbit eccentricity
    mr = np.radians(m)
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * mr) * (0.019993 - 0.000101 * jc) + np.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))
    decl = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(app_long))))

    y = np.tan(np.radians(eps / 2)) ** 2
    l0r = np.radians(l0)
    eq_time = 4 * np.degrees(
        y * np.sin(2 * l0r) - 2 * e * np.sin(mr) + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * mr))  # minutes

    minutes = (t.hour * 60 + t.minute + t.second / 60).to_numpy()
    true_solar = np.mod(minutes + eq_time + 4 * lon, 1440)
    hour_angle = np.where(true_solar / 4 < 0, true_solar / 4 + 180, true_solar / 4 - 180)

    latr, declr, har = np.radians(lat), np.radians(decl), np.radians(hour_angle)
    cos_zen = np.clip(np.sin(latr) * np.sin(declr) + np.cos(latr) * np.cos(declr) * np.cos(har), -1, 1)
    zen = np.degrees(np.arccos(cos_zen))
    elev = 90 - zen

    # Refraction (NOAA approximation)
    te = np.tan(np.radians(np.clip(elev, -89, 89)))
    refr = np.select(
        [elev > 85, elev > 5, elev > -0.575],
        [0.0, 58.1 / te - 0.07 / te**3 + 0.000086 / te**5,
         1735 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))],
        -20.772 / te) / 3600
    elev = elev + refr

    az_num = np.sin(latr) * cos_zen - np.sin(declr)
    az_den = np.cos(latr) * np.sin(np.radians(zen))
    az = np.degrees(np.arccos(np.clip(az_num / np.where(az_den == 0, 1e-12, az_den), -1, 1)))
    az = np.where(hour_angle > 0, np.mod(az + 180, 360), np.mod(540 - az, 360))
    return elev.astype(float), az.astype(float)
