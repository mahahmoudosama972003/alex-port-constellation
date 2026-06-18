"""
Alexandria Port — Constellation Monitor v4.0
============================================

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import math
import time
from datetime import timedelta, datetime, timezone

import numpy as np
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from geopy.distance import geodesic
from skyfield.api import EarthSatellite, load, wgs84
from streamlit_folium import st_folium

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
# CONSTANTS
# ─────────────────────────────────────────────
ALEX_LAT, ALEX_LON = 31.20, 29.91
ALEX_COORDS        = (ALEX_LAT, ALEX_LON)
DEFAULT_COVERAGE   = 600   # km
ORBIT_PERIOD_MIN   = int(24 * 60 / 15.22)   # ≈ 95 min

# ─────────────────────────────────────────────
# TLE DATA  (SAR Walker constellation, 32° / 550 km)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# TLE DATA  (Walker Delta 3/3/1 Constellation, 32° / 550 km)
# ─────────────────────────────────────────────
TLE_DATA = [
    {
        "name":       "SAR-Alpha",
        "line1":      "1 99991U 26001A   26132.50000000  .00000000  00000-0  00000-0 0  9994",
        # RAAN = 0.0000, Mean Anomaly = 0.0000
        "line2":      "2 99991  32.0000   0.0000 0001000   0.0000   0.0000 15.22000000    05",
        "color":      "#00d4ff",
        "fill_rgba":  "rgba(0,212,255,0.07)",
    },
    {
        "name":       "SAR-Beta",
        "line1":      "1 99992U 26001B   26132.50000000  .00000000  00000-0  00000-0 0  9995",
        # RAAN = 120.0000, Mean Anomaly = 120.0000
        "line2":      "2 99992  32.0000 120.0000 0001000   0.0000 120.0000 15.22000000    00",
        "color":      "#00ffaa",
        "fill_rgba":  "rgba(0,255,170,0.07)",
    },
    {
        "name":       "SAR-Gamma",
        "line1":      "1 99993U 26001C   26132.50000000  .00000000  00000-0  00000-0 0  9996",
        # RAAN = 240.0000, Mean Anomaly = 240.0000
        "line2":      "2 99993  32.0000 240.0000 0001000   0.0000 240.0000 15.22000000    04",
        "color":      "#ffaa00",
        "fill_rgba":  "rgba(255,170,0,0.07)",
    },
]

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
    """Return list of (lat, lon) tuples for the satellite's future ground track."""
    coords = []
    for i in range(0, minutes, step):
        t  = ts.from_datetime(t_start + timedelta(minutes=i))
        sp = sat.at(t).subpoint()
        coords.append((sp.latitude.degrees, sp.longitude.degrees))
    return coords


def compute_passes(sat, target, ts, t0, t1, min_el: float = 0.0):
    """Compute structured pass dicts (rise/culmination/set) for one satellite."""
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
    except Exception:
        pass
    return passes


def elevation_series(sat, target, ts, t_start: datetime,
                     hours: float = 48, step_min: int = 10) -> pd.DataFrame:
    """Return DataFrame(time, elevation°, azimuth°) over the given window."""
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

    st.markdown("###  Live Tracking")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh Interval (sec)", 5, 60, 10, 5)
    if auto_refresh:
        st.success(f"Auto-refreshing every {refresh_rate}s")

    st.markdown("---")
    st.markdown("### ⚙️ Mission Parameters")
    coverage_radius = st.slider("Coverage Radius (km)", 200, 1500, DEFAULT_COVERAGE, 50)
    forecast_days   = st.slider("Forecast Window (days)", 1, 7, 2)
    min_elevation   = st.slider("Min Pass Elevation (°)", 0, 45, 5, 5)
    show_tracks     = st.checkbox("Show Ground Tracks on Map", value=True)
    show_cov_circle = st.checkbox("Show Coverage Circle on Map", value=True)

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
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.74rem;
                color:#5aaccf; line-height:2.2;">
        Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Walker Delta<br>
        Sats&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 3<br>
        Inclin.&nbsp;&nbsp;: 32°<br>
        Phase Δ&nbsp;&nbsp;: 120°<br>
        Alt.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 550 km LEO<br>
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
sats  = [EarthSatellite(d["line1"], d["line2"], d["name"]) for d in TLE_DATA]

# ─────────────────────────────────────────────
# COMPUTE CURRENT POSITIONS + PASSES
# ─────────────────────────────────────────────
results          = []
in_coverage_now  = 0

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

    ground_track = get_ground_track(sat, ts, t0_dt, minutes=ORBIT_PERIOD_MIN, step=2) \
                   if show_tracks else []

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
    <span class="hero-badge">LEO 550 km</span>
    &nbsp;Real-time constellation tracking · 3-Sat Walker Delta · 32° Inclination · 120° Phase Spacing
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
min_dist = min(r["dist_km"] for r in results)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Satellites Online",      f"{len(sats)} / {len(sats)}", "All operational")
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📡  Live Dashboard",
    "📅  Pass Schedule",
    "📊  Analytics",
    "⚙️  Constellation",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tab1:

    # ── Alert banner ──────────────────────────────────────────────────
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

    # ── Satellite status cards ─────────────────────────────────────────
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

    # ── Interactive Folium Map ─────────────────────────────────────────
    st.markdown('<div class="section-header">Interactive Ground Track Map</div>',
                unsafe_allow_html=True)

    fmap = folium.Map(
        location=[ALEX_LAT, ALEX_LON],
        zoom_start=3,
        tiles="CartoDB dark_matter",
    )

    # Coverage circle
    if show_cov_circle:
        folium.Circle(
            location=[ALEX_LAT, ALEX_LON],
            radius=coverage_radius * 1000,
            color="#00d4ff", fill=True, fill_opacity=0.04,
            opacity=0.35,
            tooltip=f"Coverage Zone: {coverage_radius} km",
        ).add_to(fmap)
        # Inner "critical zone" (50% radius)
        folium.Circle(
            location=[ALEX_LAT, ALEX_LON],
            radius=(coverage_radius // 2) * 1000,
            color="#00ffaa", fill=True, fill_opacity=0.05,
            opacity=0.25,
            tooltip=f"Inner Zone: {coverage_radius//2} km",
        ).add_to(fmap)

    # Alexandria port anchor marker
    folium.Marker(
        location=[ALEX_LAT, ALEX_LON],
        icon=folium.DivIcon(html="""
            <div style="font-size:22px; text-align:center;
                        filter:drop-shadow(0 0 6px #00d4ff);">⚓</div>
        """),
        tooltip="<b>Alexandria Port</b><br>31.20°N / 29.91°E",
        popup=folium.Popup("Alexandria Port — Primary Target Site", max_width=200),
    ).add_to(fmap)

    # Satellites
    for res in results:
        # Current position dot
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

        # Satellite name label
        folium.Marker(
            location=[res["lat"] + 2.0, res["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="color:{res['color']}; font-size:10px; font-weight:700;
                            font-family:monospace; white-space:nowrap;
                            text-shadow:0 0 5px #000,0 0 5px #000,0 0 5px #000;">
                    {res['name']}
                </div>"""),
        ).add_to(fmap)

        # Ground track (handle anti-meridian splits)
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
#  TAB 2 — PASS SCHEDULE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Upcoming Pass Events</div>',
                unsafe_allow_html=True)

    if all_passes:
        # ── Pass table ────────────────────────────────────────────────
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

        # ── Stats row ─────────────────────────────────────────────────
        st.markdown("")
        s1, s2, s3, s4 = st.columns(4)
        durations = [p.get("duration_s", 0) for p in all_passes if p.get("duration_s")]
        s1.metric("Total Passes",    str(total_pass_count))
        s2.metric("Avg Duration",    f"{np.mean(durations)/60:.1f} min" if durations else "—")
        s3.metric("Best Max Elev.",  f"{max(p.get('max_el',0) for p in all_passes):.1f}°")
        s4.metric("Next Pass",       next_any_eta, next_any["sat_name"] if next_any else None)

        # ── Gantt timeline ────────────────────────────────────────────
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
#  TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab3:

    # ── Elevation over time ───────────────────────────────────────────
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

    # ── Sky plot ─────────────────────────────────────────────────────
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

    # ── Coverage gap analysis ─────────────────────────────────────────
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
#  TAB 4 — CONSTELLATION CONFIG
# ══════════════════════════════════════════════════════════════════════
with tab4:

    # ── Orbital parameters table ──────────────────────────────────────
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

    # ── TLE raw data ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Two-Line Element Sets (TLE)</div>',
                unsafe_allow_html=True)

    tle_cols = st.columns(3)
    for idx, meta in enumerate(TLE_DATA):
        with tle_cols[idx]:
            with st.expander(f"📄 {meta['name']}", expanded=True):
                st.code(f"{meta['name']}\n{meta['line1']}\n{meta['line2']}",
                        language="text")

    # ── Walker constellation diagram ──────────────────────────────────
    st.markdown('<div class="section-header">Constellation Phase Distribution</div>',
                unsafe_allow_html=True)

    phases = [0, 120, 240]
    fig_const = go.Figure()

    # Orbit ring
    ring_t = list(range(361))
    fig_const.add_trace(go.Scatterpolar(
        r=[1] * 361, theta=ring_t,
        mode="lines",
        line=dict(color="#1a3a52", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Coverage arcs (120° each)
    for ph, res in zip(phases, results):
        arc_t = [(ph - 60 + i) % 360 for i in range(121)]
        fig_const.add_trace(go.Scatterpolar(
            r=[1.0] * 121, theta=arc_t,
            mode="lines",
            line=dict(color=res["color"], width=4),
            opacity=0.25,
            showlegend=False, hoverinfo="skip",
        ))
        # Spoke + dot
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
            text="Walker Delta Constellation — 120° Phase Spacing · 32° Inclination",
            font=dict(color="#5aaccf", size=11, family="Space Mono"),
            x=0.5,
        ),
        legend=dict(bgcolor="#06101a", bordercolor="#122a3d", borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.05),
        margin=dict(l=60, r=60, t=55, b=50),
    )
    st.plotly_chart(fig_const, use_container_width=True)

    # ── Constellation summary stats ───────────────────────────────────
    st.markdown('<div class="section-header">Constellation Health Summary</div>',
                unsafe_allow_html=True)

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Total Satellites",    f"{len(sats)}")
    h2.metric("Inclination",         "32.0°")
    h3.metric("Orbital Altitude",    "~550 km")
    h4.metric("Orbital Period",      f"~{ORBIT_PERIOD_MIN} min")
    h5.metric("RAAN Spacing",        "120°")

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
