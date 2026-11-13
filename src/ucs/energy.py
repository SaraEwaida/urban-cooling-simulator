"""Surface energy balance for every pixel, stepped through time.

For each pixel's top surface (roof, tree crown, water or ground):

    C dTs/dt = K_down + L_net - H - LE - K_deep (Ts - T_deep)

* K_down: absorbed short-wave = (1 - albedo) x [direct x sunlit + diffuse x SVF
  + reflected from surroundings x (1 - SVF)]
* L_net: emissivity x (L_in - sigma Ts^4), L_in = SVF x sky + (1 - SVF) x walls at air temperature;
  sky emissivity from Brutsaert (1975) with a cloud correction
* H: convection h_c (Ts - Ta), h_c = 5.7 + 3.8 u (McAdams), u = ERA5 10 m wind reduced by
  sheltering, u x (a + b x SVF)
* LE: vegetation rho L (qsat(Ts) - qa) / (r_a + r_s), with stomatal resistance
  r_s = r_s_min / f(light), f = K / (K + 100 W m-2) (Jarvis 1976), so leaves close at night;
  other surfaces evap x rho L (qsat(Ts) - qa) / r_a; r_a = rho cp / h_c
* Storage: ground uses force-restore (Deardorff 1978) with a deep layer that follows the
  daily mean; roofs are a thin slab over an air-conditioned interior; crowns and water are slabs.

Each step linearises the fluxes around Ts, so the update is stable for any time step.
"""
import numpy as np

SIGMA = 5.670374e-8
RHO_AIR = 1.15      # kg m-3 near 40 C
CP_AIR = 1005.0     # J kg-1 K-1
LV = 2.45e6         # J kg-1
TAU = 86400.0       # s, the diurnal period for force-restore
K0 = 273.15


def qsat(t_c, p_pa):
    es = 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))
    return 0.622 * es / (p_pa - 0.378 * es)


def dqsat_dt(t_c, p_pa):
    return qsat(t_c, p_pa) * 17.625 * 243.04 / (t_c + 243.04) ** 2


def sky_longwave(t_air_c, dew_c, cloud_pct):
    e_hpa = 6.1094 * np.exp(17.625 * dew_c / (dew_c + 243.04))
    tk = t_air_c + K0
    eps = 1.24 * (e_hpa / tk) ** (1 / 7)
    eps = np.minimum(1.0, eps * (1 + 0.22 * (cloud_pct / 100.0) ** 2))
    return eps * SIGMA * tk**4


def thermal_params(classes, codes: dict, thermal: dict):
    """Per-pixel effective heat capacity C, deep coupling K, deep temperature mode and h_c factor."""
    n = classes.shape
    C = np.zeros(n, "float32")
    K = np.zeros(n, "float32")
    deep_fixed = np.full(n, np.nan, "float32")  # fixed deep temperature (roof interior)
    restore = np.zeros(n, bool)                 # force-restore ground
    hfac = np.ones(n, "float32")
    rs_min = np.full(n, np.nan, "float32")      # stomatal resistance (vegetation only)
    for name, code in codes.items():
        m = classes == code
        p = thermal[name]
        hfac[m] = p.get("h_c_factor", 1.0)
        if "r_s_min" in p:
            rs_min[m] = p["r_s_min"]
        if p["model"] == "ground":
            Ceff = p["thermal_inertia"] * np.sqrt(TAU) / (2 * np.sqrt(np.pi))
            C[m], K[m], restore[m] = Ceff, Ceff * 2 * np.pi / TAU, True
        elif p["model"] == "roof":
            C[m], K[m], deep_fixed[m] = p["heat_capacity"], p["u_value"], p["interior_c"]
        else:  # slab
            C[m] = p["heat_capacity"]
    C[classes == 0] = 1e5
    return C, K, deep_fixed, restore, hfac, rs_min


class SurfaceModel:
    def __init__(self, albedo, emissivity, evap, svf, classes, codes, thermal, wind_a, wind_b, env_albedo):
        self.alb = np.nan_to_num(albedo, nan=0.2).astype("float32")
        self.eps = np.nan_to_num(emissivity, nan=0.95).astype("float32")
        self.beta = np.nan_to_num(evap, nan=0.0).astype("float32")
        self.svf = svf.astype("float32")
        self.C, self.K, self.deep_fixed, self.restore, self.hfac, self.rs_min = thermal_params(classes, codes, thermal)
        self.veg = ~np.isnan(self.rs_min)
        self.wind = (wind_a + wind_b * self.svf).astype("float32")
        self.env_albedo = env_albedo
        self.ts = None
        self.td = None

    def init(self, t_air_c, deep_offset_c=3.0):
        self.ts = np.full(self.svf.shape, t_air_c, "float32")
        self.td = np.where(np.isnan(self.deep_fixed), t_air_c + deep_offset_c, self.deep_fixed).astype("float32")

    def fluxes(self, f, sunlit, ts):
        """All terms (W m-2, positive into the surface for gains) and the derivative of the net gain."""
        ta, p = f["temperature_2m"], f["surface_pressure"] * 100.0
        direct, diffuse = max(f["direct_radiation"], 0.0), max(f["diffuse_radiation"], 0.0)
        k_in = direct * sunlit + diffuse * self.svf + self.env_albedo * (direct + diffuse) * (1 - self.svf)
        k_abs = (1 - self.alb) * k_in
        l_sky = sky_longwave(ta, f["dew_point_2m"], f["cloud_cover"])
        l_in = self.svf * l_sky + (1 - self.svf) * SIGMA * (ta + K0) ** 4
        tk = ts + K0
        l_net = self.eps * (l_in - SIGMA * tk**4)
        u = np.maximum(f["wind_speed_10m"] * self.wind, 0.1)
        dT = ts - ta
        hc = self.hfac * np.maximum(5.7 + 3.8 * u, 1.52 * np.abs(dT) ** (1 / 3))
        h = hc * dT
        qa = qsat(f["dew_point_2m"], p)
        ra = RHO_AIR * CP_AIR / hc
        light = k_in / (k_in + 100.0)
        rs = np.where(self.veg, np.nan_to_num(self.rs_min) / np.maximum(light, 0.01), 0.0)
        cond = np.where(self.veg, 1.0 / (ra + rs), self.beta / ra)   # moisture conductance
        le = np.maximum(cond * RHO_AIR * LV * (qsat(ts, p) - qa), 0.0)
        dle = np.where(le > 0, cond * RHO_AIR * LV * dqsat_dt(ts, p), 0.0)
        deriv = -4 * self.eps * SIGMA * tk**3 - hc - dle
        return {"k_abs": k_abs, "l_net": l_net, "h": h, "le": le}, deriv

    def step(self, f, sunlit, dt):
        terms, deriv = self.fluxes(f, sunlit, self.ts)
        net = terms["k_abs"] + terms["l_net"] - terms["h"] - terms["le"]
        g_deep = self.K * (self.ts - self.td)
        new = self.ts + dt * (net - g_deep) / (self.C + dt * (self.K - deriv))
        # deep ground layer relaxes to the surface over one day (force-restore)
        self.td = np.where(self.restore, self.td + dt * (new - self.td) / TAU, self.td).astype("float32")
        self.ts = new.astype("float32")
        return terms
