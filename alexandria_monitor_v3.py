"""
Alexandria Port — Constellation Monitor v3.0
============================================
Professional SAR satellite constellation tracker for maritime surveillance.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import math
import time
import yaml
from datetime import timedelta, datetime, timezone

import numpy as np
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from geopy.distance import geodesic
from skyfield.api import EarthSatellite, load, wgs84
from streamlit_folium import st_folium

# Import custom logger
from logger import get_logger, get_ui_logs

# ─────────────────────────────────────────────
# INIT LOGGER
# ─────────────────────────────────────────────
log = get_logger("constellation")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Alexandria Port — Constellation Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONFIGURATION LOAD (YAML)
# ─────────────────────────────────────────────
if os.path.exists("config.yaml"):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ALEX_LAT             = cfg['site']['lat']
    ALEX_LON             = cfg['site']['lon']
    DEFAULT_COVERAGE     = cfg['defaults']['coverage_radius_km']
    DEFAULT_FORECAST     = cfg['defaults']['forecast_days']
    DEFAULT_MIN_ELEV     = cfg['defaults']['min_elevation_deg']
    DEFAULT_REFRESH      = cfg['defaults']['refresh_interval_sec']
    ORBIT_ALTITUDE_KM    = cfg['constellation']['altitude_km']
    INCLINATION_DEG      = cfg['constellation']['inclination_deg']
    log.info("Loaded configuration from config.yaml")
else:
    # Fallbacks if YAML is missing
    ALEX_LAT, ALEX_LON   = 31.20, 29.91
    DEFAULT_COVERAGE     = 600
    DEFAULT_FORECAST     = 2
    DEFAULT_MIN_ELEV     = 5
    DEFAULT_REFRESH      = 10
    ORBIT_ALTITUDE_KM    = 550
    INCLINATION_DEG      = 32.0
    log.warning("config.yaml not found. Using hardcoded defaults.")

ALEX_COORDS        = (ALEX_LAT, ALEX_LON)
ORBIT_PERIOD_MIN   = int(24 * 60 / 15.22)   # ≈ 95 min

# ─────────────────────────────────────────────
# CSS — Dark Maritime Theme v3
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp { background: linear-gradient(135deg, #04080f 0%, #070e1a 50%, #04091a 100%); }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(90deg, #071525 0%, #0a2d44 45%, #071525 100%);
    border: 1px solid #174060;
    border-radius: 14px;
    padding: 26px 36px 22px;
    margin-bottom: 24px;
    box-shadow: 0 2px 40px rgba(0,160,220,0.10);
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg,
        transparent, transparent 3px,
        rgba(0,180,255,0.025) 3px, rgba(0,180,255,0.025) 4px);
}
.hero-banner::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00c8f0, transparent);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.75rem; font-weight: 700;
    color: #00d4ff; letter-spacing: 2.5px; margin: 0;
    text-shadow: 0 0 28px rgba(0,212,255,0.45);
}
.hero-subtitle {
    font-size: 0.85rem; color: #5aaccf;
    margin-top: 7px; letter-spacing: 0.8px; line-height: 1.8;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.25);
    border-radius: 4px; padding: 2px 8px; font-size: 0.72rem;
    color: #00d4ff; font-family: 'Space Mono', monospace;
    letter-spacing: 1px; margin-right: 6px; vertical-align: middle;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace; color: #4a8aaa;
    font-size: 0.68rem; letter-spacing: 2.5px; text-transform: uppercase;
    margin: 22px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid #122a3d;
}

/* ── Alert banners ── */
.alert-banner {
    background: linear-gradient(90deg, rgba(220,60,0,0.12), rgba(255,100,0,0.08), rgba(220,60,0,0.12));
    border: 1px solid #cc4400;
    border-radius: 8px; padding: 14px 22px;
    color: #ff8855; font-weight: 600; font-size: 0.92rem;
    letter-spacing: 0.4px; animation: pulse-alert 2.5s infinite;
}
@keyframes pulse-alert {
    0%, 100% { box-shadow: 0 0 10px rgba(220,60,0,0.25); }
    50%       { box-shadow: 0 0 24px rgba(220,60,0,0.55); }
}
.nominal-banner {
    background: rgba(0,60,100,0.18); border: 1px solid #154060;
    border-radius: 8px; padding: 14px 22px;
    color: #5aaccf; font-size: 0.88rem;
}
.warn-banner {
    background: rgba(180,100,0,0.12); border: 1px solid #7a5000;
    border-radius: 8px; padding: 14px 22px; color: #ddaa44; font-size: 0.88rem;
}

/* ── Custom alerts for Optimizer ── */
.alert-success {
    background: rgba(0, 200, 100, 0.12); border: 1px solid #00c864;
    border-radius: 8px; padding: 14px 22px; color: #00ffaa; font-size: 0.88rem;
}
.alert-warning {
    background: rgba(200, 160, 0, 0.12); border: 1px solid #c8a000;
    border-radius: 8px; padding: 14px 22px; color: #ffcc00; font-size: 0.88rem;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(7,14,26,0.85); border: 1px solid #122a3d;
    border-radius: 9px; padding: 13px 16px;
}
[data-testid="metric-container"] label { color: #4a7a96 !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; }
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #040810 !important; border-right: 1px solid #0e2030;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #06101a; border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #122a3d; gap: 0; padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #3a6a86; font-family: 'Space Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.5px; padding: 12px 20px;
    border-radius: 6px 6px 0 0; transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important; background: #071828 !important;
    border-bottom: 2px solid #00d4ff !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #5aaccf !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #122a3d; border-radius: 8px; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #5aaccf !important; font-family: 'Space Mono', monospace; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem !important;
    letter-spacing: 1px;
    border-radius: 6px;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,212,255,0.18) !important;
}

hr { border-color: #0e2030 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TLE DATA  (Walker Delta 3/3/1 Constellation)
# ─────────────────────────────────────────────
TLE_DATA = [
    {
        "name":       "SAR-Alpha",
        "line1":      "1 99991U 26001A   26132.50000000  .00000000  00000-0  00000-0 0  9994",
        "line2":      "2 99991  32.0000   0.0000 0001000   0.0000   0.0000 15.22000000    05",
        "color":      "#00d4ff",
        "fill_rgba":  "rgba(0,212,255,0.07)",
    },
    {
        "name":       "SAR-Beta",
        "line1":      "1 99992U 26001B   26132.50000000  .00000000  00000-0  00000-0 0  9995",
        "line2":      "2 99992  32.0000 120.0000 0001000   0.0000 120.0000 15.22000000    00",
        "color":      "#00ffaa",
        "fill_rgba":  "rgba(0,255,170,0.07)",
    },
    {
        "name":       "SAR-Gamma",
        "line1":      "1 99993U 26001C   26132.50000000  .00000000  00000-0  00000-0 0  9996",
        "line2":      "2 99993  32.0000 240.0000 0001000   0.0000 240.0000 15.22000000    04",
        "color":      "#ffaa00",
        "fill_rgba":  "rgba(255,170,0,0.07)",
    },
]

# ─────────────────────────────────────────────
# EGYPT PORTS DATA
# ─────────────────────────────────────────────
EGYPT_PORTS = {
    "Alexandria":        {"lat": 31.200, "lon": 29.910, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "Port Said":         {"lat": 31.257, "lon": 32.284, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "Suez":              {"lat": 29.966, "lon": 32.549, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "Damietta":          {"lat": 31.419, "lon": 31.814, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "El Dekheila":       {"lat": 31.133, "lon": 29.794, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "East Port Said":    {"lat": 31.260, "lon": 32.380, "type": "MAJOR",       "icon": "⚓", "color": "#00d4ff"},
    "Ain Sokhna":        {"lat": 29.590, "lon": 32.347, "type": "INDUSTRIAL",  "icon": "🏭", "color": "#ffaa00"},
    "Nuweiba":           {"lat": 28.966, "lon": 34.651, "type": "PASSENGER",   "icon": "🚢", "color": "#00ffaa"},
    "Sharm El Sheikh":   {"lat": 27.857, "lon": 34.283, "type": "PASSENGER",   "icon": "🚢", "color": "#00ffaa"},
    "Hurghada":          {"lat": 27.257, "lon": 33.812, "type": "PASSENGER",   "icon": "🚢", "color": "#00ffaa"},
    "Safaga":            {"lat": 26.748, "lon": 33.937, "type": "INDUSTRIAL",  "icon": "🏭", "color": "#ffaa00"},
    "Aqaba (EG side)":   {"lat": 29.513, "lon": 34.916, "type": "CARGO",       "icon": "📦", "color": "#ff8800"},
    "Abu Qir":           {"lat": 31.320, "lon": 30.064, "type": "INDUSTRIAL",  "icon": "🏭", "color": "#ffaa00"},
    "Marsa Matruh":      {"lat": 31.352, "lon": 27.237, "type": "CARGO",       "icon": "📦", "color": "#ff8800"},
    "Berenice":          {"lat": 23.909, "lon": 35.489, "type": "CARGO",       "icon": "📦", "color": "#ff8800"},
}

# ─────────────────────────────────────────────
# POWER CONSTANTS
# ─────────────────────────────────────────────
# ── Solar array ──────────────────────────────────────────────────────────────
SOLAR_PANEL_AREA_M2   = 1.8          # m²  — deployable wing area
SOLAR_CELL_EFF        = 0.295        # 29.5% GaAs triple-junction
SOLAR_IRRADIANCE      = 1361.0       # W/m²  (AM0, solar constant)
PANEL_DEGRADATION     = 0.995        # per year — 0.5 % annual
PANEL_TEMP_DERATE     = 0.88         # temperature derating factor
SOLAR_POWER_MAX_W     = (SOLAR_PANEL_AREA_M2 * SOLAR_CELL_EFF *
                         SOLAR_IRRADIANCE * PANEL_DEGRADATION * PANEL_TEMP_DERATE)

# ── Battery ───────────────────────────────────────────────────────────────────
BATTERY_CAPACITY_WH   = 60.0         # Wh  — Li-ion pack
BATTERY_DOD_MAX       = 0.20         # max depth-of-discharge (20 %)
BATTERY_CHARGE_EFF    = 0.95         # charge efficiency
BATTERY_INITIAL_SOC   = 0.92         # starting state-of-charge

# ── Subsystem power draw (watts) ─────────────────────────────────────────────
PWR_SAR_IMAGING       = 95.0         # W  — SAR payload active
PWR_SAR_STANDBY       = 4.0          # W  — payload in standby/sleep
PWR_DOWNLINK          = 28.0         # W  — X-band transmitter (pass period)
PWR_OBC               = 6.5          # W  — on-board computer (always on)
PWR_ADCS              = 8.0          # W  — attitude determination & control
PWR_COMMS_TT_C        = 3.5          # W  — TT&C beacon (always on)
PWR_THERMAL           = 5.0          # W  — heaters (eclipse only)
PWR_BATTERY_MGMT      = 1.2          # W  — battery management system

# ── Derived mode totals ───────────────────────────────────────────────────────
PWR_SAFE_MODE         = PWR_OBC + PWR_ADCS + PWR_COMMS_TT_C + PWR_BATTERY_MGMT
PWR_IDLE_SUNLIT       = PWR_SAFE_MODE + PWR_SAR_STANDBY
PWR_IDLE_ECLIPSE      = PWR_IDLE_SUNLIT + PWR_THERMAL
PWR_IMAGING_MODE      = PWR_IDLE_SUNLIT + PWR_SAR_IMAGING
PWR_DOWNLINK_MODE     = PWR_IDLE_SUNLIT + PWR_DOWNLINK

# ── Orbit geometry (Walker 550 km, 32°) ───────────────────────────────────────
EARTH_RADIUS_KM       = 6371.0
ORBIT_PERIOD_SEC      = ORBIT_PERIOD_MIN * 60  # 👈 السطر ده اللي كان ناقص
ECLIPSE_FRACTION      = 0.362        # fraction of orbit in shadow @ 550 km, ~32°
ECLIPSE_DURATION_SEC  = ORBIT_PERIOD_SEC * ECLIPSE_FRACTION
SUNLIT_DURATION_SEC   = ORBIT_PERIOD_SEC - ECLIPSE_DURATION_SEC

# ── Duty cycle budgets ────────────────────────────────────────────────────────
MAX_IMAGING_DUTY      = 0.30         # 30 % of sunlit arc
DOWNLINK_WINDOW_SEC   = 180          # 3-minute average pass

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fmt_countdown(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 0:
        return "Overdue"
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def get_ground_track(sat, ts, t_start: datetime, minutes: int = 100, step: int = 2):
    coords = []
    for i in range(0, minutes, step):
        t  = ts.from_datetime(t_start + timedelta(minutes=i))
        sp = sat.at(t).subpoint()
        coords.append((sp.latitude.degrees, sp.longitude.degrees))
    return coords

def compute_passes(sat, target, ts, t0, t1, min_el: float = 0.0):
    passes = []
    try:
        t_ev, evtypes = sat.find_events(target, t0, t1, altitude_degrees=min_el)
        cur = {}
        for t, etype in zip(t_ev, evtypes):
            dt = t.utc_datetime()
            if etype == 0:   # rise
                cur = {"rise": dt, "max_el": 0.0, "max_az": 0.0, "max_el_t": None, "set": None}
            elif etype == 1: # culmination
                alt, az, _ = (sat - target).at(t).altaz()
                cur["max_el"]  = round(alt.degrees, 1)
                cur["max_az"]  = round(az.degrees, 1)
                cur["max_el_t"] = dt
            elif etype == 2: # set
                if cur.get("rise"):
                    cur["set"] = dt
                    dur = (dt - cur["rise"]).total_seconds()
                    cur["duration_s"] = dur
                    passes.append(dict(cur))
                    cur = {}
    except Exception as e:
        log.error(f"Failed to compute passes: {e}")
    return passes

def compute_port_coverage(port_name, port_info, sats, ts, t0, t1, t0_dt, min_elevation, cov_radius):
    port_loc = wgs84.latlon(port_info["lat"], port_info["lon"])
    
    port_results = []
    for sat, meta in zip(sats, TLE_DATA):
        dist = geodesic(
            (port_info["lat"], port_info["lon"]),
            (sat.at(t0).subpoint().latitude.degrees, 
             sat.at(t0).subpoint().longitude.degrees)
        ).km
        
        topo_alt, topo_az, _ = (sat - port_loc).at(t0).altaz()
        passes = compute_passes(sat, port_loc, ts, t0, t1, min_elevation)
        future = [p for p in passes if p.get("rise") and p["rise"] > t0_dt]
        
        port_results.append({
            "sat":        meta["name"],
            "color":      meta["color"],
            "dist_km":    dist,
            "elev_deg":   topo_alt.degrees,
            "in_cov":     dist < cov_radius,
            "passes":     future,
            "next_rise":  future[0]["rise"] if future else None,
            "next_max_el":future[0].get("max_el", 0) if future else 0,
        })
        
    in_cov_now   = any(r["in_cov"] for r in port_results)
    next_pass    = min(
        (r for r in port_results if r["next_rise"]),
        key=lambda r: r["next_rise"],
        default=None
    )
    total_passes = sum(len(r["passes"]) for r in port_results)
    
    return {
        "name":         port_name,
        "lat":          port_info["lat"],
        "lon":          port_info["lon"],
        "type":         port_info["type"],
        "icon":         port_info["icon"],
        "color":        port_info["color"],
        "in_cov_now":   in_cov_now,
        "next_sat":     next_pass["sat"]      if next_pass else "—",
        "next_rise":    next_pass["next_rise"] if next_pass else None,
        "next_eta":     fmt_countdown(next_pass["next_rise"] - t0_dt) if next_pass else "No pass",
        "next_max_el":  next_pass["next_max_el"] if next_pass else 0,
        "total_passes": total_passes,
        "sat_details":  port_results,
    }

@st.cache_data(ttl=300)
def get_all_ports_coverage(_sats_proxy, _ts_proxy, t0_str, t1_str, _min_el, _cov_r):
    t0c  = _ts_proxy.utc(*[float(x) for x in t0_str.split(",")])
    t1c  = _ts_proxy.utc(*[float(x) for x in t1_str.split(",")])
    t0c_dt = t0c.utc_datetime()
    
    all_port_data = []
    for pname, pinfo in EGYPT_PORTS.items():
        pd_  = compute_port_coverage(pname, pinfo, _sats_proxy, _ts_proxy, t0c, t1c, t0c_dt, _min_el, _cov_r)
        all_port_data.append(pd_)
    return all_port_data

def elevation_series(sat, target, ts, t_start: datetime, hours: float = 48, step_min: int = 10) -> pd.DataFrame:
    rows, n = [], int(hours * 60 / step_min)
    for i in range(n):
        dt = t_start + timedelta(minutes=i * step_min)
        t  = ts.from_datetime(dt)
        alt, az, _ = (sat - target).at(t).altaz()
        rows.append({"time": dt, "elevation": max(0.0, alt.degrees), "azimuth": az.degrees})
    return pd.DataFrame(rows)

def pass_quality(max_el: float) -> str:
    if max_el >= 60: return "⭐ EXCELLENT"
    if max_el >= 30: return "HIGH"
    if max_el >= 15: return "MEDIUM"
    return "LOW"

def pass_quality_color(q: str) -> str:
    return {"⭐ EXCELLENT": "#00ff88", "HIGH": "#00d4ff",
            "MEDIUM": "#ffaa00", "LOW": "#ff6644"}.get(q, "#6cb8d4")

def render_optimizer(results, all_passes, gaps_h):
    st.markdown("### 🔬 Optimization Analysis")
    issues, recommendations = [], []
    avg_gap = np.mean(gaps_h) if gaps_h else 0
    max_gap = max(gaps_h) if gaps_h else 0
    best_el = max((p.get('max_el', 0) for p in all_passes), default=0)

    if avg_gap > 3:
        issues.append(f"⚠️ High avg revisit gap ({avg_gap:.1f}h) — target < 2h")
        recommendations.append("➜ Raise orbital altitude to 700–800 km to widen swath width")
    if max_gap > 6:
        issues.append(f"🔴 Coverage blackout detected ({max_gap:.1f}h max gap)")
        recommendations.append("➜ Add a 4th satellite at RAAN = 90° to fill the gap")
    if best_el < 20:
        issues.append(f"⚠️ Low max elevation ({best_el:.1f}°) — poor imaging geometry")
        recommendations.append("➜ Increase inclination to 40–45° for better Alexandria coverage")

    if not issues:
        st.markdown('<div class="alert-success">✅ Constellation performing optimally for this target.</div>', unsafe_allow_html=True)
    else:
        for i in issues:
            st.markdown(f'<div class="alert-warning">{i}</div><br>', unsafe_allow_html=True)

    if recommendations:
        st.markdown("#### 📐 Recommendations")
        for r in recommendations:
            st.markdown(f"""
            <div style="background:rgba(0,80,140,0.12); border-left:3px solid #0080c0;
                        border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:8px;
                        color:#6ab8d8; font-size:0.88rem;">
                {r}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧪 What-If Simulator")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_sats  = st.slider("Number of Satellites", 3, 12, 3)
    with c2:
        sim_alt   = st.slider("Orbital Altitude (km)", 400, 1200, 550, 50)
    with c3:
        sim_incl  = st.slider("Inclination (°)", 20, 98, 32)

    period_min   = 2 * np.pi * np.sqrt(((6371 + sim_alt) * 1e3)**3 / 3.986e14) / 60
    swath_km     = 2 * sim_alt * np.tan(np.radians(25))
    daily_passes = int(24 * 60 / period_min * sim_sats)
    est_revisit  = round(24 / max(daily_passes / 4, 0.1), 2)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Orbital Period",      f"{period_min:.1f} min")
    s2.metric("Est. Swath Width",    f"{swath_km:.0f} km")
    s3.metric("Daily Passes (est.)", str(daily_passes))
    s4.metric("Est. Revisit",        f"{est_revisit:.1f} h")

def compute_eclipse_periods(sat, ts, t0_dt, forecast_days: int = 2):
    """
    Detect eclipse entry/exit by checking sunlit flag every 30 s.
    Returns list of dicts {enter, exit, duration_s}.
    """
    from skyfield.api import load as sf_load
    eph   = sf_load("de421.bsp")
    earth = eph["earth"]
    sun   = eph["sun"]

    step_s   = 30
    n_steps  = int(forecast_days * 24 * 3600 / step_s)
    eclipses = []
    in_ecl   = False
    ecl_start = None

    for i in range(n_steps):
        from datetime import timedelta, timezone
        dt = t0_dt + timedelta(seconds=i * step_s)
        t  = ts.from_datetime(dt)
        sat_geo = sat.at(t)
        # Vector from sat to sun
        sat_pos  = (earth + sat).at(t).observe(sun)
        sun_alt  = sat_pos.apparent().altaz()[0].degrees
        # Negative sun altitude → eclipse
        ecl_now  = sun_alt < 0

        if ecl_now and not in_ecl:
            in_ecl    = True
            ecl_start = dt
        elif not ecl_now and in_ecl:
            in_ecl = False
            if ecl_start:
                dur = (dt - ecl_start).total_seconds()
                eclipses.append({"enter": ecl_start, "exit": dt, "duration_s": dur})
                ecl_start = None

    return eclipses

def simulate_power_budget(passes_for_sat: list, t0_dt, forecast_hours: float = 48,
                          step_min: int = 5, imaging_duty: float = MAX_IMAGING_DUTY,
                          initial_soc: float = BATTERY_INITIAL_SOC,
                          solar_power_max_w: float = SOLAR_POWER_MAX_W,
                          battery_capacity_wh: float = BATTERY_CAPACITY_WH) -> pd.DataFrame:
    """
    Time-step power simulation for one satellite.
    Returns DataFrame with columns:
        time, mode, power_gen_W, power_cons_W, net_W, soc, soc_pct, battery_wh
    Modes: 'IMAGING', 'DOWNLINK', 'IDLE_SUN', 'ECLIPSE', 'SAFE'
    """
    from datetime import timedelta
    n_steps  = int(forecast_hours * 60 / step_min)
    dt_s     = step_min * 60          # step size in seconds
    soc      = initial_soc
    rows     = []

    # Build pass lookup  {floor_minute: 'IMAGING' | 'DOWNLINK'}
    pass_windows = {}
    for p in passes_for_sat:
        if not (p.get("rise") and p.get("set")):
            continue
        rise_m = int((p["rise"] - t0_dt).total_seconds() // 60)
        set_m  = int((p["set"]  - t0_dt).total_seconds() // 60)
        # Imaging: first part of the pass window (duty-cycle limited)
        imaging_m  = int((set_m - rise_m) * imaging_duty * 0.9)
        downlink_m = int((set_m - rise_m) * 0.15)
        for m in range(rise_m, set_m):
            offset = m - rise_m
            if offset < imaging_m:
                pass_windows[m] = "IMAGING"
            elif offset < imaging_m + downlink_m:
                pass_windows[m] = "DOWNLINK"

    for i in range(n_steps):
        from datetime import timedelta
        dt_i = t0_dt + timedelta(minutes=i * step_min)
        m_i  = i * step_min

        # ── Determine illumination (simple sinusoidal eclipse model) ──
        orbit_phase = ((i * step_min * 60) % ORBIT_PERIOD_SEC) / ORBIT_PERIOD_SEC
        in_eclipse  = orbit_phase > (1 - ECLIPSE_FRACTION)

        # ── Determine mode ────────────────────────────────────────────
        mode = pass_windows.get(m_i, "ECLIPSE" if in_eclipse else "IDLE_SUN")

        # Override: if battery critically low → safe mode
        if soc < (1 - BATTERY_DOD_MAX - 0.02):
            mode = "SAFE"

        # ── Power generation ──────────────────────────────────────────
        if in_eclipse:
            gen_w = 0.0
        else:
            # Slight cosine derate over illumination arc
            sun_angle = math.pi * ((orbit_phase / (1 - ECLIPSE_FRACTION)) - 0.5)
            gen_w = solar_power_max_w * max(0.0, math.cos(sun_angle))

        # ── Power consumption ─────────────────────────────────────────
        cons_map = {
            "IMAGING":   PWR_IMAGING_MODE,
            "DOWNLINK":  PWR_DOWNLINK_MODE,
            "IDLE_SUN":  PWR_IDLE_SUNLIT,
            "ECLIPSE":   PWR_IDLE_ECLIPSE,
            "SAFE":      PWR_SAFE_MODE,
        }
        cons_w = cons_map.get(mode, PWR_IDLE_SUNLIT)

        # ── Energy balance ────────────────────────────────────────────
        net_w     = gen_w - cons_w
        delta_wh  = net_w * (dt_s / 3600)

        if delta_wh > 0:   # charging
            soc = min(1.0, soc + (delta_wh * BATTERY_CHARGE_EFF) / battery_capacity_wh)
        else:              # discharging
            soc = max(0.0, soc + delta_wh / battery_capacity_wh)

        soc_pct   = soc * 100
        batt_wh   = soc * battery_capacity_wh

        rows.append({
            "time":         dt_i,
            "mode":         mode,
            "power_gen_W":  round(gen_w,  2),
            "power_cons_W": round(cons_w, 2),
            "net_W":        round(net_w,  2),
            "soc":          round(soc,    4),
            "soc_pct":      round(soc_pct,2),
            "battery_wh":   round(batt_wh, 3),
        })

    return pd.DataFrame(rows)

def duty_cycle_summary(df: pd.DataFrame) -> dict:
    """Compute per-mode time fractions and energy statistics from simulation df."""
    total     = len(df)
    mode_cnts = df["mode"].value_counts()

    def pct(m): return round(mode_cnts.get(m, 0) / total * 100, 1)

    avg_gen   = df["power_gen_W"].mean()
    avg_cons  = df["power_cons_W"].mean()
    min_soc   = df["soc_pct"].min()
    dod_max   = 100 - min_soc
    safe_evts = (df["mode"] == "SAFE").sum()
    critical_windows = df[df["soc_pct"] < (1 - BATTERY_DOD_MAX) * 100]

    return {
        "pct_imaging":   pct("IMAGING"),
        "pct_downlink":  pct("DOWNLINK"),
        "pct_idle_sun":  pct("IDLE_SUN"),
        "pct_eclipse":   pct("ECLIPSE"),
        "pct_safe":      pct("SAFE"),
        "avg_gen_W":     round(avg_gen,  1),
        "avg_cons_W":    round(avg_cons, 1),
        "energy_margin": round(avg_gen - avg_cons, 1),
        "min_soc_pct":   round(min_soc, 1),
        "max_dod_pct":   round(dod_max, 1),
        "safe_events":   int(safe_evts),
        "n_critical":    len(critical_windows),
        "power_status":  ("⚠️ CRITICAL" if dod_max > BATTERY_DOD_MAX * 100
                          else "✅ NOMINAL"),
    }

MODE_COLORS = {
    "IMAGING":   "#00d4ff",
    "DOWNLINK":  "#00ffaa",
    "IDLE_SUN":  "#ffaa00",
    "ECLIPSE":   "#334466",
    "SAFE":      "#ff4422",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; color:#00d4ff; font-size:0.95rem;
                font-weight:700; padding:10px 0 6px; border-bottom:1px solid #122a3d;
                margin-bottom:14px; letter-spacing:1.5px;">
        🛰️ CONSTELLATION MONITOR
        <div style="font-size:0.65rem; color:#4a7a96; margin-top:3px; letter-spacing:0.5px;">
        Alexandria Port · SAR LEO Suite v3.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔄 Live Tracking")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh Interval (sec)", 5, 60, DEFAULT_REFRESH, 5)

    st.markdown("---")
    st.markdown("### ⚙️ Mission Parameters")
    coverage_radius = st.slider("Coverage Radius (km)", 200, 1500, DEFAULT_COVERAGE, 50)
    forecast_days   = st.slider("Forecast Window (days)", 1, 7, DEFAULT_FORECAST)
    min_elevation   = st.slider("Min Pass Elevation (°)", 0, 45, DEFAULT_MIN_ELEV, 5)
    show_tracks     = st.checkbox("Show Ground Tracks on Map", value=True)
    show_cov_circle = st.checkbox("Show Coverage Circle on Map", value=True)

    st.markdown("---")
    st.markdown("### ⚡ Power Budget")
    imaging_duty_pct = st.slider(
        "Imaging Duty Cycle (%)", 5, 50,
        int(MAX_IMAGING_DUTY * 100), 5,
        help="Fraction of sunlit arc used for SAR imaging"
    )
    imaging_duty = imaging_duty_pct / 100
    
    custom_battery_wh = st.number_input(
        "Battery Capacity (Wh)", min_value=10.0,
        max_value=300.0, value=float(BATTERY_CAPACITY_WH), step=5.0
    )
    
    custom_solar_w = st.number_input(
        "Solar Array Power (W)", min_value=10.0,
        max_value=1500.0, value=round(SOLAR_POWER_MAX_W, 1), step=5.0
    )
    
    power_forecast_h = st.slider("Power Forecast (hours)", 12, 168, 48, 12)

    st.markdown("---")
    st.markdown("### 📍 Target Site")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.76rem;
                color:#5aaccf; line-height:2.2;">
        <b style="color:#00d4ff">Alexandria Port</b><br>
        LAT&nbsp;&nbsp;: {ALEX_LAT}° N<br>
        LON&nbsp;&nbsp;: {ALEX_LON}° E<br>
        ALT&nbsp;&nbsp;: ~5 m MSL<br>
        TZ&nbsp;&nbsp;&nbsp;: UTC +2 (EET)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛰 Constellation Spec")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.74rem;
                color:#5aaccf; line-height:2.2;">
        Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Walker Delta<br>
        Sats&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 3<br>
        Inclin.&nbsp;&nbsp;: {INCLINATION_DEG}°<br>
        Phase Δ&nbsp;&nbsp;: 120°<br>
        Alt.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {ORBIT_ALTITUDE_KM} km LEO<br>
        Period&nbsp;&nbsp;&nbsp;: ~95 min<br>
        Mode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: SAR Imaging
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INIT SKYFIELD
# ─────────────────────────────────────────────
@st.cache_resource
def _timescale():
    return load.timescale()

ts    = _timescale()
t0    = ts.now()
t0_dt = t0.utc_datetime()
t1    = ts.from_datetime(t0_dt + timedelta(days=forecast_days))
alex  = wgs84.latlon(ALEX_LAT, ALEX_LON)

log.info("Parsing TLE data and initializing satellites.")
try:
    sats = [EarthSatellite(d["line1"], d["line2"], d["name"]) for d in TLE_DATA]
except Exception as e:
    log.error(f"TLE parse failed: {e}")
    sats = []

# ─────────────────────────────────────────────
# COMPUTE CURRENT POSITIONS + PASSES
# ─────────────────────────────────────────────
results          = []
in_coverage_now  = 0
log.info(f"Computing positions and passes for forecast window: {forecast_days} days.")

for sat, meta in zip(sats, TLE_DATA):
    sp  = sat.at(t0).subpoint()
    lat = sp.latitude.degrees
    lon = sp.longitude.degrees
    alt = sp.elevation.km

    dist          = geodesic((lat, lon), ALEX_COORDS).km
    topo_alt, topo_az, _ = (sat - alex).at(t0).altaz()
    elev_deg      = topo_alt.degrees
    az_deg        = topo_az.degrees
    cov_pct       = max(0.0, (1 - dist / coverage_radius) * 100)
    in_cov        = dist < coverage_radius

    passes        = compute_passes(sat, alex, ts, t0, t1, min_elevation)
    future_passes = [p for p in passes if p.get("rise") and p["rise"] > t0_dt]

    if future_passes:
        fp              = future_passes[0]
        next_pass_label = fmt_countdown(fp["rise"] - t0_dt)
        next_rise       = fp["rise"]
    elif in_cov:
        next_pass_label = "IN COVERAGE NOW"
        next_rise       = t0_dt
    else:
        next_pass_label = "No pass in window"
        next_rise       = None

    if in_cov:
        in_coverage_now += 1

    ground_track = get_ground_track(sat, ts, t0_dt, minutes=ORBIT_PERIOD_MIN, step=2) if show_tracks else []

    results.append({
        "name":       sat.name,
        "sat_obj":    sat,
        "meta":       meta,
        "lat":        lat,  "lon": lon,  "alt_km": alt,
        "dist_km":    dist, "elev_deg": elev_deg, "az_deg": az_deg,
        "cov_pct":    cov_pct, "in_cov": in_cov,
        "passes":     passes,
        "next_label": next_pass_label,
        "next_rise":  next_rise,
        "track":      ground_track,
        "color":      meta["color"],
        "fill_rgba":  meta["fill_rgba"],
    })

# Compute all ports directly via cached logic
t0_key = ",".join(str(x) for x in t0.utc)
t1_key = ",".join(str(x) for x in t1.utc)
all_port_data = get_all_ports_coverage(sats, ts, t0_key, t1_key, min_elevation, coverage_radius)

# ── Build global sorted pass list (with sat_name attached) ────────────────
all_passes = sorted(
    [dict(p, sat_name=r["name"], sat_color=r["color"])
     for r in results for p in r["passes"]
     if p.get("rise") and p["rise"] > t0_dt],
    key=lambda p: p["rise"],
)
total_pass_count = len(all_passes)
next_any         = all_passes[0] if all_passes else None
next_any_eta     = fmt_countdown(next_any["rise"] - t0_dt) if next_any else "—"

# ── Revisit gap stats ─────────────────────────────────────────────────────
gaps_h = []
for i in range(len(all_passes) - 1):
    pa, pb = all_passes[i], all_passes[i + 1]
    if pa.get("set") and pb.get("rise"):
        g = (pb["rise"] - pa["set"]).total_seconds() / 3600
        if g >= 0:
            gaps_h.append(round(g, 3))

avg_revisit = f"{np.mean(gaps_h):.1f} h" if gaps_h else "—"
max_gap     = f"{max(gaps_h):.1f} h"     if gaps_h else "—"

log.info("Pass computation finished.")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
utc_str = t0_dt.strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">🛰️ ALEXANDRIA PORT — CONSTELLATION MONITOR</div>
  <div class="hero-subtitle">
    <span class="hero-badge">SAR MARITIME</span>
    <span class="hero-badge">LIVE</span>
    <span class="hero-badge">LEO {ORBIT_ALTITUDE_KM} km</span>
    &nbsp;Real-time constellation tracking · 3-Sat Walker Delta · {INCLINATION_DEG}° Inclination · 120° Phase Spacing
    <br>
    🕐 Current UTC: <b style="color:#00d4ff">{utc_str}</b>
    &nbsp;·&nbsp; Forecast window: <b style="color:#00d4ff">{forecast_days}d</b>
    &nbsp;·&nbsp; Coverage radius: <b style="color:#00d4ff">{coverage_radius} km</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
min_dist = min((r["dist_km"] for r in results), default=0)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Satellites Online",      f"{len(sats)} / {len(TLE_DATA)}", "All operational" if len(sats)==len(TLE_DATA) else "Degraded")
k2.metric("In Coverage Now",        str(in_coverage_now),
          "Active imaging" if in_coverage_now > 0 else "Standby")
k3.metric("Closest Satellite",      f"{min_dist:,.0f} km")
k4.metric("Next Pass ETA",          next_any_eta,
          next_any["sat_name"] if next_any else None)
k5.metric("Passes in Forecast",     str(total_pass_count))
k6.metric("Avg Revisit Time",       avg_revisit, f"Max gap {max_gap}")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📡  Live Dashboard",
    "🗺️  Egypt Ports",
    "📅  Pass Schedule",
    "📊  Analytics",
    "⚙️  Constellation",
    "🔬  Optimization",
    "📋  System Logs",
    "⚡  Power Budget"
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tab1:
    if in_coverage_now > 0:
        names = ", ".join(r["name"] for r in results if r["in_cov"])
        verb  = "is" if in_coverage_now == 1 else "are"
        st.markdown(f"""
        <div class="alert-banner">
            🚨 ACTIVE COVERAGE ALERT — <b>{names}</b> {verb} currently within
            the {coverage_radius} km imaging range of Alexandria Port.
            Acquisition window OPEN.
        </div>""", unsafe_allow_html=True)
    else:
        next_html = (f"  Next pass: <b>{next_any['sat_name']}</b> in "
                     f"<b>{next_any_eta}</b>  "
                     f"(max el. {next_any.get('max_el', 0):.1f}°)."
                     if next_any else "")
        st.markdown(f"""
        <div class="nominal-banner">
            ◉ SYSTEM NOMINAL — No satellites currently within the
            {coverage_radius} km coverage zone.{next_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="section-header">Satellite Status</div>', unsafe_allow_html=True)
    cols = st.columns(3)

    for idx, res in enumerate(results):
        bdr = "#cc4400" if res["in_cov"] else "#122a3d"
        shd = "0 0 22px rgba(200,60,0,0.35)" if res["in_cov"] else "none"
        status_color = "#ff8855" if res["in_cov"] else "#3aaa6a"
        status_text  = "🔴 IN COVERAGE" if res["in_cov"] else "🟢 NOMINAL"

        with cols[idx]:
            st.markdown(f"""
            <div style="border:1.5px solid {bdr}; border-radius:10px;
                        padding:16px 18px 10px;
                        background:linear-gradient(160deg,#07131f,#040c16);
                        margin-bottom:8px; box-shadow:{shd};">
                <div style="font-family:'Space Mono',monospace; font-size:0.95rem;
                             font-weight:700; color:#00d4ff; margin:0 0 12px;
                             padding-bottom:9px; border-bottom:1px solid {bdr};
                             display:flex; justify-content:space-between;">
                    <span>🛰 {res['name']}</span>
                    <span style="font-size:0.65rem; color:{status_color};
                                 letter-spacing:1px; align-self:center;">
                        {status_text}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Distance",   f"{res['dist_km']:,.0f} km")
            m2.metric("Altitude",   f"{res['alt_km']:.1f} km")

            m3, m4 = st.columns(2)
            m3.metric("Elevation",  f"{res['elev_deg']:.1f}°")
            m4.metric("Azimuth",    f"{res['az_deg']:.1f}°")

            m5, m6 = st.columns(2)
            m5.metric("Coverage",   f"{res['cov_pct']:.1f}%")
            m6.metric("Passes",     str(len([p for p in res["passes"]
                                             if p.get("rise") and p["rise"] > t0_dt])))

            st.metric("Next Pass ETA", res["next_label"])
            st.caption(f"📍 {res['lat']:.3f}°N  /  {res['lon']:.3f}°E")

    st.markdown('<div class="section-header">Interactive Ground Track Map</div>',
                unsafe_allow_html=True)

    fmap = folium.Map(
        location=[ALEX_LAT, ALEX_LON],
        zoom_start=3,
        tiles="CartoDB dark_matter",
    )

    if show_cov_circle:
        folium.Circle(
            location=[ALEX_LAT, ALEX_LON],
            radius=coverage_radius * 1000,
            color="#00d4ff", fill=True, fill_opacity=0.04,
            opacity=0.35,
            tooltip=f"Coverage Zone: {coverage_radius} km",
        ).add_to(fmap)
        folium.Circle(
            location=[ALEX_LAT, ALEX_LON],
            radius=(coverage_radius // 2) * 1000,
            color="#00ffaa", fill=True, fill_opacity=0.05,
            opacity=0.25,
            tooltip=f"Inner Zone: {coverage_radius//2} km",
        ).add_to(fmap)

    folium.Marker(
        location=[ALEX_LAT, ALEX_LON],
        icon=folium.DivIcon(html="""
            <div style="font-size:22px; text-align:center;
                        filter:drop-shadow(0 0 6px #00d4ff);">⚓</div>
        """),
        tooltip="<b>Alexandria Port</b><br>31.20°N / 29.91°E",
        popup=folium.Popup("Alexandria Port — Primary Target Site", max_width=200),
    ).add_to(fmap)

    for res in results:
        folium.Marker(
            location=[res["lat"], res["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="
                    background:{res['color']}; border-radius:50%;
                    width:14px; height:14px; border:2px solid #fff;
                    box-shadow:0 0 10px {res['color']},0 0 20px {res['color']}88;">
                </div>
            """),
            tooltip=(f"<b>🛰 {res['name']}</b><br>"
                     f"Lat: {res['lat']:.2f}°N  Lon: {res['lon']:.2f}°E<br>"
                     f"Alt: {res['alt_km']:.1f} km<br>"
                     f"Dist to Alex: {res['dist_km']:,.0f} km"),
            popup=folium.Popup(
                f"<b>{res['name']}</b><br>"
                f"Distance: {res['dist_km']:,.0f} km<br>"
                f"Elevation: {res['elev_deg']:.1f}°<br>"
                f"Azimuth: {res['az_deg']:.1f}°",
                max_width=220,
            ),
        ).add_to(fmap)

        folium.Marker(
            location=[res["lat"] + 2.0, res["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="color:{res['color']}; font-size:10px; font-weight:700;
                            font-family:monospace; white-space:nowrap;
                            text-shadow:0 0 5px #000,0 0 5px #000,0 0 5px #000;">
                    {res['name']}
                </div>"""),
        ).add_to(fmap)

        if show_tracks and res["track"]:
            segs, prev = [[]], None
            for lat_g, lon_g in res["track"]:
                if prev is not None and abs(lon_g - prev) > 180:
                    segs.append([])
                segs[-1].append([lat_g, lon_g])
                prev = lon_g
            for seg in segs:
                if len(seg) > 1:
                    folium.PolyLine(
                        seg, color=res["color"], weight=1.8,
                        opacity=0.65, dash_array="6 4",
                        tooltip=f"{res['name']} ground track (next {ORBIT_PERIOD_MIN} min)",
                    ).add_to(fmap)

    st_folium(fmap, width=None, height=540, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — EGYPT PORTS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    # ── KPIs ──────────────────────────────────────────────────────────
    ports_in_cov = sum(1 for p in all_port_data if p["in_cov_now"])
    ports_w_pass = sum(1 for p in all_port_data if p["next_rise"])
    pk1, pk2, pk3, pk4 = st.columns(4)
    pk1.metric("Total Ports Tracked",  str(len(EGYPT_PORTS)))
    pk2.metric("Ports In Coverage Now", str(ports_in_cov),
               "Active imaging" if ports_in_cov else "None active")
    pk3.metric("Ports With Upcoming Pass", str(ports_w_pass))
    pk4.metric("Coverage Radius",      f"{coverage_radius} km")
    
    st.markdown("")

    # ── Map ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Egypt Ports — SAR Coverage Map</div>',
                unsafe_allow_html=True)

    port_map = folium.Map(
        location=[29.5, 31.5],
        zoom_start=6,
        tiles="CartoDB dark_matter",
    )

    # Draw per-port coverage circles
    for pd_ in all_port_data:
        ring_color = "#ff4400" if pd_["in_cov_now"] else "#00d4ff"
        folium.Circle(
            location=[pd_["lat"], pd_["lon"]],
            radius=coverage_radius * 1000,
            color=ring_color, fill=True, fill_opacity=0.03,
            opacity=0.18,
            tooltip=f"{pd_['name']} coverage zone",
        ).add_to(port_map)

    # Port markers
    for pd_ in all_port_data:
        status_color = "#ff4400" if pd_["in_cov_now"] else pd_["color"]
        pulse_ring   = f"box-shadow:0 0 14px {status_color};" if pd_["in_cov_now"] else ""
        folium.Marker(
            location=[pd_["lat"], pd_["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="
                    background:{status_color}22;
                    border:2px solid {status_color};
                    border-radius:50%;
                    width:20px; height:20px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:11px; {pulse_ring}">
                    {pd_['icon']}
                </div>"""),
            tooltip=f"""
                <b>{pd_['name']}</b><br>
                Type: {pd_['type']}<br>
                Status: {'🔴 IN COVERAGE' if pd_['in_cov_now'] else '🟢 Nominal'}<br>
                Next pass: {pd_['next_eta']}<br>
                Passes in window: {pd_['total_passes']}
            """,
            popup=folium.Popup(f"""
                <div style='font-family:monospace; font-size:11px; color:#222;'>
                <b>{pd_['name']}</b><br>
                📍 {pd_['lat']:.3f}°N / {pd_['lon']:.3f}°E<br>
                🛰 Next: {pd_['next_sat']} in {pd_['next_eta']}<br>
                📐 Max El: {pd_['next_max_el']:.1f}°<br>
                📅 Total passes: {pd_['total_passes']}
                </div>""", max_width=220),
        ).add_to(port_map)

    # Satellite current positions
    for res in results:
        folium.Marker(
            location=[res["lat"], res["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="
                    background:{res['color']};
                    border-radius:50%; width:12px; height:12px;
                    border:2px solid #fff;
                    box-shadow:0 0 10px {res['color']};">
                </div>"""),
            tooltip=f"🛰 {res['name']} — Alt: {res['alt_km']:.1f} km",
        ).add_to(port_map)

        if show_tracks and res["track"]:
            segs, prev = [[]], None
            for la, lo in res["track"]:
                if prev and abs(lo - prev) > 180: segs.append([])
                segs[-1].append([la, lo])
                prev = lo
            for seg in segs:
                if len(seg) > 1:
                    folium.PolyLine(
                        seg, color=res["color"], weight=1.5,
                        opacity=0.5, dash_array="5 4",
                    ).add_to(port_map)

    st_folium(port_map, width=None, height=580, returned_objects=[])

    # ── Port table ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Port Coverage Summary</div>',
                unsafe_allow_html=True)

    # Filter controls
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        filter_type = st.multiselect(
            "Port Type", options=["MAJOR","INDUSTRIAL","PASSENGER","CARGO"],
            default=["MAJOR","INDUSTRIAL","PASSENGER","CARGO"])
    with cf2:
        filter_status = st.selectbox(
            "Coverage Status", ["All", "In Coverage Now", "Not In Coverage"])
    with cf3:
        sort_by = st.selectbox(
            "Sort By", ["Next Pass ETA", "Port Name", "Total Passes", "Max Elevation"])

    # Build table rows
    rows = []
    for pd_ in all_port_data:
        if pd_["type"] not in filter_type:
            continue
        if filter_status == "In Coverage Now"  and not pd_["in_cov_now"]: continue
        if filter_status == "Not In Coverage"  and pd_["in_cov_now"]:     continue

        rows.append({
            "Port":           f"{pd_['icon']} {pd_['name']}",
            "Type":           pd_["type"],
            "Lat / Lon":      f"{pd_['lat']:.3f}°N / {pd_['lon']:.3f}°E",
            "Status":         "🔴 IN COVERAGE" if pd_["in_cov_now"] else "🟢 Nominal",
            "Next Satellite": pd_["next_sat"],
            "Next Pass ETA":  pd_["next_eta"],
            "Max Elevation":  f"{pd_['next_max_el']:.1f}°",
            "Total Passes":   pd_["total_passes"],
        })

    if sort_by == "Next Pass ETA":
        # put "No pass" entries at end
        rows.sort(key=lambda r: (r["Next Pass ETA"] == "No pass", r["Next Pass ETA"]))
    elif sort_by == "Port Name":
        rows.sort(key=lambda r: r["Port"])
    elif sort_by == "Total Passes":
        rows.sort(key=lambda r: -r["Total Passes"])
    elif sort_by == "Max Elevation":
        rows.sort(key=lambda r: -float(r["Max Elevation"].replace("°", "")))

    if rows:
        port_df = pd.DataFrame(rows)

        def _style_status(val):
            return "color: #ff4400; font-weight:700;" if "IN COVERAGE" in val else "color: #00cc88;"

        def _style_passes(val):
            return "color: #00d4ff;" if int(val) > 5 else "color: #5aaccf;"

        st.dataframe(
            port_df.style
                .map(_style_status, subset=["Status"])
                .map(_style_passes, subset=["Total Passes"]),
            use_container_width=True,
            hide_index=True,
            height=min(60 + len(port_df) * 35, 540),
        )

        # CSV export
        csv = port_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Export Port Coverage Table (CSV)",
            data=csv,
            file_name=f"egypt_ports_coverage_{t0_dt.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No ports match current filters.")

    # ── Per-port detail expanders ─────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Per-Port Satellite Detail</div>',
                unsafe_allow_html=True)

    for pd_ in sorted(all_port_data, key=lambda x: x["name"]):
        if pd_["type"] not in filter_type: continue
        
        with st.expander(f"{pd_['icon']} {pd_['name']} — {pd_['type']}  |  {pd_['next_eta']}"):
            dc1, dc2, dc3 = st.columns(3)
            for i, sr in enumerate(pd_["sat_details"]):
                col = [dc1, dc2, dc3][i]
                with col:
                    sc = "#ff4400" if sr["in_cov"] else sr["color"]
                    st.markdown(f"""
                    <div style="border:1px solid {sc}44; border-radius:8px;
                                padding:12px 14px; background:rgba(0,0,0,0.2);">
                        <div style="color:{sc}; font-family:'Space Mono',monospace;
                                     font-size:0.8rem; font-weight:700; margin-bottom:8px;">
                            🛰 {sr['sat']}
                        </div>
                        <div style="color:#5aaccf; font-size:0.78rem; line-height:2;">
                            Dist&nbsp;: {sr['dist_km']:,.0f} km<br>
                            Elev&nbsp;: {sr['elev_deg']:.1f}°<br>
                            Passes: {len(sr['passes'])}<br>
                            Next&nbsp;: {fmt_countdown(sr['next_rise'] - t0_dt) if sr['next_rise'] else '—'}
                        </div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — PASS SCHEDULE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Upcoming Pass Events</div>',
                unsafe_allow_html=True)

    if all_passes:
        rows = []
        for p in all_passes:
            rise = p["rise"]
            sett = p.get("set")
            maxe = p.get("max_el_t")
            dur  = p.get("duration_s", 0)
            q    = pass_quality(p.get("max_el", 0))
            rows.append({
                "Satellite":         p["sat_name"],
                "Rise Time (UTC)":   rise.strftime("%Y-%m-%d  %H:%M:%S"),
                "Max El. Time":      maxe.strftime("%H:%M:%S") if maxe else "—",
                "Set Time":          sett.strftime("%H:%M:%S") if sett else "—",
                "Duration":          f"{int(dur//60)}m {int(dur%60):02d}s" if dur else "—",
                "Max Elevation (°)": f"{p.get('max_el',0):.1f}°",
                "Azimuth @ Max":     f"{p.get('max_az',0):.1f}°",
                "Pass Quality":      q,
            })

        pass_df = pd.DataFrame(rows)

        def _style_q(val):
            return f"color: {pass_quality_color(val)}"

        st.dataframe(
            pass_df.style.map(_style_q, subset=["Pass Quality"]),
            use_container_width=True,
            hide_index=True,
            height=min(40 + len(pass_df) * 35, 520),
        )

        csv_bytes = pass_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Export Pass Schedule (CSV)",
            data=csv_bytes,
            file_name=f"alex_pass_schedule_{t0_dt.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.markdown("")
        s1, s2, s3, s4 = st.columns(4)
        durations = [p.get("duration_s", 0) for p in all_passes if p.get("duration_s")]
        s1.metric("Total Passes",    str(total_pass_count))
        s2.metric("Avg Duration",    f"{np.mean(durations)/60:.1f} min" if durations else "—")
        s3.metric("Best Max Elev.",  f"{max(p.get('max_el',0) for p in all_passes):.1f}°")
        s4.metric("Next Pass",       next_any_eta, next_any["sat_name"] if next_any else None)

        st.markdown('<div class="section-header">Pass Timeline</div>',
                    unsafe_allow_html=True)

        gantt_rows = []
        col_map    = {r["name"]: r["color"] for r in results}
        for p in all_passes:
            if p.get("rise") and p.get("set"):
                gantt_rows.append({
                    "Satellite": p["sat_name"],
                    "Start":     p["rise"],
                    "Finish":    p["set"],
                    "Quality":   pass_quality(p.get("max_el", 0)),
                    "Max El":    f"{p.get('max_el',0):.1f}°",
                })

        if gantt_rows:
            gdf = pd.DataFrame(gantt_rows)
            fig_g = px.timeline(
                gdf, x_start="Start", x_end="Finish", y="Satellite",
                color="Satellite",
                color_discrete_map=col_map,
                hover_data={"Quality": True, "Max El": True,
                            "Satellite": False, "Start": True, "Finish": True},
            )
            fig_g.update_yaxes(categoryorder="array",
                                categoryarray=[r["name"] for r in results])
            fig_g.update_layout(
                xaxis=dict(
                    title="UTC Time", gridcolor="#122a3d", color="#5aaccf",
                    range=[t0_dt, t0_dt + timedelta(days=forecast_days)],
                ),
                yaxis=dict(title="", gridcolor="#122a3d", color="#5aaccf"),
                plot_bgcolor="#06101a", paper_bgcolor="#04080f",
                font=dict(color="#5aaccf", family="Space Mono, monospace"),
                height=200,
                margin=dict(l=80, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_g, use_container_width=True)

    else:
        st.markdown("""
        <div class="warn-banner">
            ⚠ No passes found in the current forecast window.
            Try increasing the forecast days or lowering the minimum elevation angle.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Elevation Angle Over Time</div>',
                unsafe_allow_html=True)

    with st.spinner("Computing elevation series…"):
        fig_el = go.Figure()
        for res in results:
            df_el = elevation_series(res["sat_obj"], alex, ts, t0_dt,
                                     hours=forecast_days * 24, step_min=10)
            fig_el.add_trace(go.Scatter(
                x=df_el["time"], y=df_el["elevation"],
                name=res["name"],
                mode="lines",
                line=dict(color=res["color"], width=1.5),
                fill="tozeroy", fillcolor=res["fill_rgba"],
                hovertemplate=(f"<b>{res['name']}</b><br>"
                               f"%{{x|%b %d %H:%M}}<br>"
                               f"El: %{{y:.1f}}°<extra></extra>"),
            ))

    fig_el.add_hline(
        y=min_elevation, line_dash="dash", line_color="#ff6600",
        annotation_text=f"Min Elevation ({min_elevation}°)",
        annotation_font_color="#ff9966",
        annotation_position="bottom right",
    )
    fig_el.update_layout(
        xaxis=dict(title="UTC Time", gridcolor="#122a3d", color="#5aaccf",
                   range=[t0_dt, t0_dt + timedelta(days=forecast_days)]),
        yaxis=dict(title="Elevation (°)", range=[0, 90],
                   gridcolor="#122a3d", color="#5aaccf"),
        plot_bgcolor="#06101a", paper_bgcolor="#04080f",
        font=dict(color="#5aaccf", family="Space Mono, monospace"),
        height=360,
        margin=dict(l=55, r=20, t=20, b=50),
        legend=dict(bgcolor="#06101a", bordercolor="#122a3d",
                    borderwidth=1, orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig_el, use_container_width=True)

    st.markdown('<div class="section-header">Sky Plot  (Azimuth / Elevation)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Satellite trajectories across the sky as seen from Alexandria Port.  "
        "Centre = Zenith (90°) · Outer ring = Horizon (0°).  "
        "Only passes within the forecast window are shown."
    )

    fig_sky = go.Figure()
    for res in results:
        shown = 0
        for p in res["passes"]:
            if not (p.get("rise") and p.get("set") and p["rise"] > t0_dt):
                continue
            rise_dt, set_dt = p["rise"], p["set"]
            n = max(4, int((set_dt - rise_dt).total_seconds() / 20))
            az_v, el_v = [], []
            for i in range(n + 1):
                dt_i = rise_dt + (set_dt - rise_dt) * (i / n)
                t_i  = ts.from_datetime(dt_i)
                alt_i, az_i, _ = (res["sat_obj"] - alex).at(t_i).altaz()
                az_v.append(az_i.degrees)
                el_v.append(max(0.0, alt_i.degrees))

            r_vals = [90 - e for e in el_v]
            fig_sky.add_trace(go.Scatterpolar(
                r=r_vals, theta=az_v, mode="lines+markers",
                marker=dict(size=3, color=res["color"]),
                line=dict(color=res["color"], width=2),
                name=f"{res['name']}  {rise_dt.strftime('%m-%d %H:%M')}",
                hovertemplate=(
                    f"<b>{res['name']}</b><br>"
                    "Az: %{theta:.1f}°<br>El: %{customdata:.1f}°<extra></extra>"
                ),
                customdata=el_v,
                legendgroup=res["name"],
                showlegend=(shown == 0),
            ))
            shown += 1

    fig_sky.update_layout(
        polar=dict(
            bgcolor="#06101a",
            radialaxis=dict(
                range=[0, 90],
                tickvals=[0, 30, 60, 90],
                ticktext=["90°", "60°", "30°", "0°"],
                gridcolor="#1a3a52", linecolor="#1a3a52", color="#5aaccf",
            ),
            angularaxis=dict(
                direction="clockwise", rotation=90,
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                gridcolor="#1a3a52", linecolor="#1a3a52", color="#5aaccf",
            ),
        ),
        plot_bgcolor="#04080f", paper_bgcolor="#04080f",
        font=dict(color="#5aaccf", family="Space Mono, monospace"),
        height=500,
        legend=dict(bgcolor="#06101a", bordercolor="#122a3d", borderwidth=1,
                    orientation="v", x=1.05),
        margin=dict(l=30, r=140, t=30, b=30),
    )
    st.plotly_chart(fig_sky, use_container_width=True)

    st.markdown('<div class="section-header">Coverage Gap Analysis</div>',
                unsafe_allow_html=True)

    if len(gaps_h) >= 2:
        bar_colors = ["#ff4400" if g > 4 else "#ffaa00" if g > 2 else "#00d4ff"
                      for g in gaps_h]
        fig_gap = go.Figure(go.Bar(
            x=list(range(1, len(gaps_h) + 1)),
            y=gaps_h,
            marker_color=bar_colors,
            marker_line_color="rgba(0,0,0,0.3)",
            marker_line_width=0.5,
            text=[f"{g:.1f}h" for g in gaps_h],
            textposition="outside",
            textfont=dict(color="#5aaccf", size=9),
            hovertemplate="Gap #%{x}<br>Duration: %{y:.2f} h<extra></extra>",
        ))
        fig_gap.add_hline(y=np.mean(gaps_h), line_dash="dot", line_color="#00d4ff",
                          annotation_text=f"Avg {np.mean(gaps_h):.1f}h",
                          annotation_font_color="#00d4ff",
                          annotation_position="top right")
        fig_gap.update_layout(
            xaxis=dict(title="Gap #", gridcolor="#122a3d", color="#5aaccf"),
            yaxis=dict(title="Gap Duration (h)", gridcolor="#122a3d", color="#5aaccf"),
            plot_bgcolor="#06101a", paper_bgcolor="#04080f",
            font=dict(color="#5aaccf", family="Space Mono, monospace"),
            height=290, margin=dict(l=55, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Min Gap",         f"{min(gaps_h):.2f} h")
        g2.metric("Max Gap",         f"{max(gaps_h):.2f} h")
        g3.metric("Avg Gap",         f"{np.mean(gaps_h):.2f} h")
        g4.metric("Std Dev",         f"{np.std(gaps_h):.2f} h")
    else:
        st.info("Insufficient pass data for gap analysis. Increase the forecast window.")

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — CONSTELLATION CONFIG
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Current Orbital Parameters</div>',
                unsafe_allow_html=True)

    orbit_df = pd.DataFrame([{
        "Satellite":          r["name"],
        "Lat (°N)":           round(r["lat"],     3),
        "Lon (°E)":           round(r["lon"],     3),
        "Altitude (km)":      round(r["alt_km"],  1),
        "Dist. to Alex (km)": round(r["dist_km"], 0),
        "Topocentric El (°)": round(r["elev_deg"],2),
        "Azimuth (°)":        round(r["az_deg"],  2),
        "Coverage Score":     f"{r['cov_pct']:.1f}%",
        "Next Pass ETA":      r["next_label"],
        "Status":             "IN COVERAGE" if r["in_cov"] else "NOMINAL",
    } for r in results])

    st.dataframe(orbit_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Two-Line Element Sets (TLE)</div>',
                unsafe_allow_html=True)

    tle_cols = st.columns(3)
    for idx, meta in enumerate(TLE_DATA):
        with tle_cols[idx]:
            with st.expander(f"📄 {meta['name']}", expanded=True):
                st.code(f"{meta['name']}\n{meta['line1']}\n{meta['line2']}",
                        language="text")

    st.markdown('<div class="section-header">Constellation Phase Distribution</div>',
                unsafe_allow_html=True)

    phases = [0, 120, 240]
    fig_const = go.Figure()

    ring_t = list(range(361))
    fig_const.add_trace(go.Scatterpolar(
        r=[1] * 361, theta=ring_t,
        mode="lines",
        line=dict(color="#1a3a52", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    for ph, res in zip(phases, results):
        arc_t = [(ph - 60 + i) % 360 for i in range(121)]
        fig_const.add_trace(go.Scatterpolar(
            r=[1.0] * 121, theta=arc_t,
            mode="lines",
            line=dict(color=res["color"], width=4),
            opacity=0.25,
            showlegend=False, hoverinfo="skip",
        ))
        fig_const.add_trace(go.Scatterpolar(
            r=[0, 1], theta=[ph, ph],
            mode="lines+markers+text",
            line=dict(color=res["color"], width=2),
            marker=dict(size=[0, 16], color=[res["color"], res["color"]],
                        line=dict(color="white", width=1.5),
                        symbol=["circle", "circle"]),
            text=["", res["name"]],
            textposition="top center",
            textfont=dict(color=res["color"], size=11, family="Space Mono"),
            name=res["name"],
        ))

    fig_const.update_layout(
        polar=dict(
            bgcolor="#06101a",
            radialaxis=dict(visible=False, range=[0, 1.5]),
            angularaxis=dict(
                direction="clockwise", rotation=90,
                tickvals=list(range(0, 360, 30)),
                ticktext=[f"{i}°" for i in range(0, 360, 30)],
                gridcolor="#122a3d", color="#5aaccf",
                linecolor="#122a3d",
            ),
        ),
        plot_bgcolor="#04080f", paper_bgcolor="#04080f",
        font=dict(color="#5aaccf", family="Space Mono, monospace"),
        height=400,
        title=dict(
            text=f"Walker Delta Constellation — 120° Phase Spacing · {INCLINATION_DEG}° Inclination",
            font=dict(color="#5aaccf", size=11, family="Space Mono"),
            x=0.5,
        ),
        legend=dict(bgcolor="#06101a", bordercolor="#122a3d", borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.05),
        margin=dict(l=60, r=60, t=55, b=50),
    )
    st.plotly_chart(fig_const, use_container_width=True)

    st.markdown('<div class="section-header">Constellation Health Summary</div>',
                unsafe_allow_html=True)

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Total Satellites",    f"{len(sats)}")
    h2.metric("Inclination",         f"{INCLINATION_DEG}°")
    h3.metric("Orbital Altitude",    f"~{ORBIT_ALTITUDE_KM} km")
    h4.metric("Orbital Period",      f"~{ORBIT_PERIOD_MIN} min")
    h5.metric("RAAN Spacing",        "120°")

# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════
with tab6:
    render_optimizer(results, all_passes, gaps_h)

# ══════════════════════════════════════════════════════════════════════
#  TAB 7 — SYSTEM LOGS
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">Event Logs</div>', unsafe_allow_html=True)
    
    logs = get_ui_logs()
    if logs:
        for entry in logs:
            color = {'DEBUG':'#4a7a96','INFO':'#00d4ff','WARNING':'#ffaa00','ERROR':'#ff4444'}.get(entry['level'],'#fff')
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.76rem;
                        padding:4px 10px; border-left:2px solid {color}22; margin-bottom:2px;">
                <span style="color:#2a5a76;">[{entry['time']}]</span>
                <span style="color:{color}; margin:0 8px;">{entry['level']:<8}</span>
                <span style="color:#8aaccf;">{entry['msg']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No logs available yet.")

# ══════════════════════════════════════════════════════════════════════
#  TAB 8 — POWER BUDGET
# ══════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">Power Budget Overview</div>',
                unsafe_allow_html=True)

    # ── Top-level power specs ─────────────────────────────────────────────────────
    pb1, pb2, pb3, pb4, pb5 = st.columns(5)
    pb1.metric("Solar Array (peak)",  f"{custom_solar_w:.1f} W")
    pb2.metric("Battery Capacity",    f"{custom_battery_wh} Wh")
    pb3.metric("Max DoD",             f"{int(BATTERY_DOD_MAX*100)} %")
    pb4.metric("SAR Payload",         f"{PWR_SAR_IMAGING} W")
    pb5.metric("Eclipse Fraction",    f"{ECLIPSE_FRACTION*100:.1f} %")

    st.markdown("")

    # ── Per-satellite power simulation ────────────────────────────────────────────
    sim_dfs   = {}
    summaries = {}

    with st.spinner("Simulating power budgets…"):
        for res in results:              
            passes_for_sat = [p for p in res["passes"]
                              if p.get("rise") and p["rise"] > t0_dt]
            df_sim = simulate_power_budget(
                passes_for_sat, t0_dt,
                forecast_hours=power_forecast_h,
                imaging_duty=imaging_duty,
                solar_power_max_w=custom_solar_w,
                battery_capacity_wh=custom_battery_wh
            )
            sim_dfs[res["name"]]   = df_sim
            summaries[res["name"]] = duty_cycle_summary(df_sim)

    # ── Summary cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Satellite Power Status</div>',
                unsafe_allow_html=True)

    pw_cols = st.columns(3)
    for idx, res in enumerate(results):
        s   = summaries[res["name"]]
        bdr = "#cc4400" if s["safe_events"] > 0 else \
              "#997700" if s["max_dod_pct"] > 15 else "#1a3a52"
        with pw_cols[idx]:
            st.markdown(f"""
            <div style="border:1.5px solid {bdr}; border-radius:10px;
                        padding:16px 18px; background:linear-gradient(160deg,#07131f,#040c16);
                        margin-bottom:8px;">
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem;
                             font-weight:700; color:{res['color']}; margin-bottom:10px;
                             padding-bottom:8px; border-bottom:1px solid {bdr};
                             display:flex; justify-content:space-between;">
                    <span>⚡ {res['name']}</span>
                    <span style="font-size:0.65rem; align-self:center;
                                 color:{'#ff6644' if s['safe_events']>0 else '#00dd88'};">
                        {s["power_status"]}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            cm1, cm2 = st.columns(2)
            cm1.metric("Avg Generation", f"{s['avg_gen_W']} W")
            cm2.metric("Avg Consumption", f"{s['avg_cons_W']} W")

            cm3, cm4 = st.columns(2)
            cm3.metric("Energy Margin",   f"{s['energy_margin']:+.1f} W",
                       "surplus" if s["energy_margin"] >= 0 else "deficit")
            cm4.metric("Min SoC",         f"{s['min_soc_pct']:.1f} %")

            st.metric("Max Depth of Discharge", f"{s['max_dod_pct']:.1f} %",
                      f"{'⚠️ exceeds limit' if s['max_dod_pct'] > BATTERY_DOD_MAX*100 else 'within limit'}")

            # Mini donut — duty cycle
            dc_fig = go.Figure(go.Pie(
                labels=["Imaging", "Downlink", "Idle (Sun)", "Eclipse", "Safe"],
                values=[s["pct_imaging"], s["pct_downlink"], s["pct_idle_sun"],
                        s["pct_eclipse"], s["pct_safe"]],
                marker_colors=["#00d4ff", "#00ffaa", "#ffaa00", "#334466", "#ff4422"],
                hole=0.55,
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            dc_fig.update_layout(
                showlegend=False,
                margin=dict(l=5, r=5, t=5, b=5),
                height=120,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"{s['pct_imaging']}%<br><span style='font-size:8px'>imaging</span>",
                    x=0.5, y=0.5, font_size=11, font_color=res["color"],
                    showarrow=False,
                )],
            )
            st.plotly_chart(dc_fig, use_container_width=True, key=f"dc_{res['name']}")

    st.markdown("")

    # ── Satellite selector for deep-dive ──────────────────────────────────────────
    st.markdown('<div class="section-header">Deep-Dive: State-of-Charge & Power Flow</div>',
                unsafe_allow_html=True)

    sel_sat = st.selectbox(
        "Select satellite",
        [r["name"] for r in results],
        key="pwr_sat_select",
    )
    df_sel = sim_dfs[sel_sat]
    
    # التعديل هنا: هنجيب بيانات القمر بالكامل عشان نستخدم الـ rgba الصحيح
    sel_res = next(r for r in results if r["name"] == sel_sat)
    sel_color = sel_res["color"]
    sel_fill = sel_res["fill_rgba"] 

    # ── Combined SoC + Power chart (dual y-axis) ─────────────────────────────────
    fig_pwr = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.35, 0.25],
        vertical_spacing=0.04,
        subplot_titles=["Battery State-of-Charge (%)",
                        "Power Generation vs Consumption (W)",
                        "Operating Mode"],
    )

    # Row 1 — SoC
    fig_pwr.add_trace(go.Scatter(
        x=df_sel["time"], y=df_sel["soc_pct"],
        name="SoC %", mode="lines",
        line=dict(color=sel_color, width=2),
        fill="tozeroy", fillcolor=sel_fill, # 👈 التعديل هنا: استخدمنا المتغير الصحيح
        hovertemplate="%{x|%b %d %H:%M}<br>SoC: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # Row 2 — Generation vs Consumption
    fig_pwr.add_trace(go.Scatter(
        x=df_sel["time"], y=df_sel["power_gen_W"],
        name="Generation", mode="lines",
        line=dict(color="#00ffaa", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,255,170,0.07)",
        hovertemplate="%{x|%b %d %H:%M}<br>Gen: %{y:.1f} W<extra></extra>",
    ), row=2, col=1)

    fig_pwr.add_trace(go.Scatter(
        x=df_sel["time"], y=df_sel["power_cons_W"],
        name="Consumption", mode="lines",
        line=dict(color="#ff6644", width=1.8),
        hovertemplate="%{x|%b %d %H:%M}<br>Cons: %{y:.1f} W<extra></extra>",
    ), row=2, col=1)

    fig_pwr.add_trace(go.Scatter(
        x=df_sel["time"], y=df_sel["net_W"],
        name="Net", mode="lines",
        line=dict(color="#ffcc44", width=1.4, dash="dot"),
        hovertemplate="%{x|%b %d %H:%M}<br>Net: %{y:+.1f} W<extra></extra>",
    ), row=2, col=1)

    # Row 3 — Operating mode as colour bands
    mode_num = {m: i for i, m in enumerate(MODE_COLORS.keys())}
    df_sel   = df_sel.copy()
    df_sel["mode_n"] = df_sel["mode"].map(mode_num)

    for mode, color in MODE_COLORS.items():
        mask = df_sel["mode"] == mode
        if mask.any():
            fig_pwr.add_trace(go.Bar(
                x=df_sel.loc[mask, "time"],
                y=[1] * mask.sum(),
                name=mode,
                marker_color=color,
                marker_line_width=0,
                showlegend=True,
                width=5 * 60 * 1000,   # step width in ms
                hovertemplate=f"{mode}<extra></extra>",
            ), row=3, col=1)

    # Layout
    for row_i in range(1, 4):
        fig_pwr.update_yaxes(
            gridcolor="#122a3d", color="#5aaccf", row=row_i, col=1,
            zeroline=False,
        )
        fig_pwr.update_xaxes(
            gridcolor="#122a3d", color="#5aaccf", row=row_i, col=1,
        )

    fig_pwr.update_yaxes(title_text="SoC (%)",   range=[0, 105], row=1, col=1)
    fig_pwr.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig_pwr.update_yaxes(title_text="",  showticklabels=False, range=[0, 1], row=3, col=1)
    fig_pwr.update_xaxes(
        title_text="UTC Time", row=3, col=1,
        range=[t0_dt, t0_dt + timedelta(hours=power_forecast_h)],
    )

    fig_pwr.update_layout(
        plot_bgcolor="#06101a",
        paper_bgcolor="#04080f",
        font=dict(color="#5aaccf", family="Space Mono, monospace"),
        height=680,
        margin=dict(l=65, r=20, t=50, b=50),
        legend=dict(bgcolor="#06101a", bordercolor="#122a3d", borderwidth=1,
                    orientation="h", y=1.04, x=0, font_size=10),
        barmode="stack",
        hovermode="x unified",
    )

    # Annotate titles
    for i, title in enumerate(["Battery State-of-Charge (%)",
                                "Power Generation vs Consumption (W)",
                                "Operating Mode"]):
        fig_pwr.layout.annotations[i].update(
            font=dict(color="#4a8aaa", size=11, family="Space Mono"),
            x=0,
        )

    st.plotly_chart(fig_pwr, use_container_width=True, key="pwr_detail")

    # ── Per-pass power feasibility table ─────────────────────────────────────────
    st.markdown('<div class="section-header">Pass-Level Power Feasibility</div>',
                unsafe_allow_html=True)

    sel_res   = next(r for r in results if r["name"] == sel_sat)
    feas_rows = []

    for p in sel_res["passes"]:
        if not (p.get("rise") and p.get("rise") > t0_dt):
            continue

        dur_s    = p.get("duration_s", 0) or 0
        dur_min  = dur_s / 60
        img_s    = dur_s * imaging_duty * 0.9
        dl_s     = dur_s * 0.15
        idle_s   = dur_s - img_s - dl_s

        energy_req = (
            PWR_SAR_IMAGING * img_s / 3600 +
            PWR_DOWNLINK_MODE * dl_s / 3600 +
            PWR_IDLE_SUNLIT * idle_s / 3600
        )
        
        energy_avail = custom_battery_wh * BATTERY_DOD_MAX
        solar_top = custom_solar_w * 0.65 * dur_s / 3600
        net_batt  = energy_req - solar_top
        batt_pct_used = (net_batt / custom_battery_wh * 100) if dur_s > 0 else 0.0
        feasible  = batt_pct_used <= BATTERY_DOD_MAX * 100

        feas_rows.append({
            "Rise Time":        p["rise"].strftime("%m-%d  %H:%M:%S"),
            "Duration":         f"{dur_min:.1f} min",
            "Max Elevation":    f"{p.get('max_el', 0):.1f}°",
            "Energy Required":  f"{energy_req:.2f} Wh",
            "Solar Top-Up":     f"{solar_top:.2f} Wh",
            "Net Battery Draw": f"{net_batt:.2f} Wh",
            "Battery Use":      f"{batt_pct_used:.1f} %",
            "Feasibility":      "✅ YES" if feasible else "⛔ INSUFFICIENT",
        })

    if feas_rows:
        feas_df = pd.DataFrame(feas_rows)
        def _style_feas(val):
            return "color: #00dd88" if "YES" in str(val) else "color: #ff4422"
        st.dataframe(
            feas_df.style.map(_style_feas, subset=["Feasibility"]),
            use_container_width=True, hide_index=True,
            height=min(40 + len(feas_df) * 35, 480),
        )

        feas_csv = feas_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Export Power Feasibility Report (CSV)",
            data=feas_csv,
            file_name=f"{sel_sat}_power_feasibility_{t0_dt.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="pwr_csv",
        )
    else:
        st.info("No upcoming passes for the selected satellite within the forecast window.")

    # ── Power subsystem breakdown ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Subsystem Power Breakdown</div>',
                unsafe_allow_html=True)

    sub_cols = st.columns(2)

    # Bar chart — per-subsystem draw
    subsystems = {
        "SAR Imaging":       PWR_SAR_IMAGING,
        "X-Band Downlink":   PWR_DOWNLINK,
        "ADCS":              PWR_ADCS,
        "On-Board Computer": PWR_OBC,
        "TT&C Comms":        PWR_COMMS_TT_C,
        "Thermal (eclipse)": PWR_THERMAL,
        "Battery Mgmt":      PWR_BATTERY_MGMT,
        "SAR Standby":       PWR_SAR_STANDBY,
    }
    sub_colors = [
        "#00d4ff", "#00ffaa", "#ffaa00", "#aa88ff",
        "#ff8844", "#6699cc", "#88ccbb", "#335577",
    ]

    with sub_cols[0]:
        fig_sub = go.Figure(go.Bar(
            x=list(subsystems.values()),
            y=list(subsystems.keys()),
            orientation="h",
            marker_color=sub_colors,
            marker_line_width=0,
            text=[f"{v} W" for v in subsystems.values()],
            textposition="outside",
            textfont=dict(color="#5aaccf", size=10),
            hovertemplate="%{y}: %{x} W<extra></extra>",
        ))
        fig_sub.update_layout(
            xaxis=dict(title="Power Draw (W)", gridcolor="#122a3d", color="#5aaccf"),
            yaxis=dict(color="#5aaccf", gridcolor="#122a3d"),
            plot_bgcolor="#06101a", paper_bgcolor="#04080f",
            font=dict(color="#5aaccf", family="Space Mono, monospace"),
            height=340, margin=dict(l=160, r=60, t=20, b=50),
            title=dict(text="Subsystem Power Draw", font=dict(size=11, color="#4a8aaa"),
                       x=0.5),
        )
        st.plotly_chart(fig_sub, use_container_width=True, key="sub_bar")

    # Waterfall — mode totals
    with sub_cols[1]:
        modes_w = {
            "Safe Mode":     PWR_SAFE_MODE,
            "Idle (Sun)":    PWR_IDLE_SUNLIT,
            "Idle (Eclipse)":PWR_IDLE_ECLIPSE,
            "Downlink":      PWR_DOWNLINK_MODE,
            "SAR Imaging":   PWR_IMAGING_MODE,
        }
        fig_mode = go.Figure(go.Bar(
            x=list(modes_w.keys()),
            y=list(modes_w.values()),
            marker_color=["#334466", "#ffaa00", "#336688", "#00ffaa", "#00d4ff"],
            text=[f"{v:.0f} W" for v in modes_w.values()],
            textposition="outside",
            textfont=dict(color="#5aaccf", size=11),
            hovertemplate="%{x}<br>Total: %{y:.0f} W<extra></extra>",
        ))
        fig_mode.add_hline(
            y=custom_solar_w, line_dash="dash", line_color="#00ff88",
            annotation_text=f"Solar peak {custom_solar_w:.0f} W",
            annotation_font_color="#00ff88",
            annotation_position="top left",
        )
        fig_mode.update_layout(
            xaxis=dict(color="#5aaccf", gridcolor="#122a3d"),
            yaxis=dict(title="Total Power (W)", gridcolor="#122a3d", color="#5aaccf"),
            plot_bgcolor="#06101a", paper_bgcolor="#04080f",
            font=dict(color="#5aaccf", family="Space Mono, monospace"),
            height=340, margin=dict(l=55, r=20, t=20, b=50),
            title=dict(text="Operating Mode Power Totals", font=dict(size=11, color="#4a8aaa"),
                       x=0.5),
        )
        st.plotly_chart(fig_mode, use_container_width=True, key="mode_bar")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; font-size:0.72rem; color:#2a5060;
            font-family:'Space Mono',monospace; padding:8px 0 4px; line-height:2;">
    CONSTELLATION MONITOR v3.0 &nbsp;·&nbsp;
    EPOCH: {t0.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC &nbsp;·&nbsp;
    COVERAGE RADIUS: {coverage_radius} km &nbsp;·&nbsp;
    FORECAST: {forecast_days}d &nbsp;·&nbsp;
    MIN ELEVATION: {min_elevation}° &nbsp;·&nbsp;
    Powered by Skyfield + Plotly + Folium
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
