# ============================================================
# Soyang River Dam Floodplain Inundation App
# Manning equation + relative DEM + spatially varying water surface
# ============================================================

import os
import io
import numpy as np
import streamlit as st
import pandas as pd
import geopandas as gpd
import rasterio
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Point

try:
    from scipy.ndimage import distance_transform_edt, binary_closing, binary_fill_holes
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# App setup
# ============================================================
#st.set_page_config(layout="wide")
st.title("🌊 Soyang River Dam Floodplain Inundation App")

REL_DEM_PATH = "rel_dem1.tif"
TAILRACE_PATH = "Tailrace_discharge.shp"
CROSS_SECTION_PATH = "CrossSections.shp"

DAM_LON = 127.815
DAM_LAT = 37.945

# Reference hydraulic points used only to calibrate the Manning response
RATING_POINTS = [
    {"RivID": 1, "Stage": 0.111457, "Discharge": 4.328249, "n": 0.045},
    {"RivID": 2, "Stage": 0.732764, "Discharge": 8.493151, "n": 0.045},
    {"RivID": 3, "Stage": 1.222962, "Discharge": 19.965510, "n": 0.045},
    {"RivID": 4, "Stage": 1.405198, "Discharge": 34.733889, "n": 0.045},
]


# ============================================================
# Sidebar
# ============================================================
st.sidebar.subheader("Tailrace Discharge")

use_reservoir_outflow = False
reservoir_df = st.session_state.get("reservoir_output_df", None)

if reservoir_df is not None and len(reservoir_df) > 0:
    use_reservoir_outflow = st.sidebar.checkbox(
        "Use PPO optimized outflow from reservoir module",
        value=False
    )

if use_reservoir_outflow:
    reservoir_df = reservoir_df.copy()
    reservoir_df["date"] = pd.to_datetime(reservoir_df["date"])

    selected_date = st.sidebar.selectbox(
        "Select reservoir operation date",
        reservoir_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        index=0
    )

    selected_row = reservoir_df[
        reservoir_df["date"].dt.strftime("%Y-%m-%d") == selected_date
    ].iloc[0]

    #selected_inflow = float(selected_row["PPO_Inflow"])
    selected_outflow = float(selected_row["PPO_Optimized_Outflow"])
    selected_storage = float(selected_row["PPO_Optimized_Storage"])

    st.sidebar.markdown("### Reservoir Conditions")
    st.sidebar.write(f"**Date:** {selected_date}")
    #st.sidebar.write(f"**PPO inflow:** {selected_inflow:.3f} m³/s")
    st.sidebar.write(f"**PPO optimized outflow:** {selected_outflow:.3f} m³/s")
    st.sidebar.write(f"**PPO storage:** {selected_storage:.3f} Mm³")

    generate_from_reservoir = st.sidebar.button(
        "Generate floodplain map using selected PPO release"
    )

    if "selected_reservoir_q" not in st.session_state:
        st.session_state["selected_reservoir_q"] = selected_outflow

    if generate_from_reservoir:
        st.session_state["selected_reservoir_q"] = selected_outflow
        st.sidebar.success(
            f"Floodplain map updated using PPO outflow: {selected_outflow:.3f} m³/s"
        )

    user_q = float(st.session_state["selected_reservoir_q"])
    low_flow_mode = user_q < 100.0

else:
    low_flow_mode = st.sidebar.checkbox("Low-flow mode (<100 m³/s)", value=True)

    if low_flow_mode:
        user_q = st.sidebar.slider(
            "Tailrace discharge, Q (m³/s)",
            min_value=1.0,
            max_value=100.0,
            value=34.858,
            step=0.1,
        )
    else:
        user_q = st.sidebar.slider(
            "Tailrace discharge, Q (m³/s)",
            min_value=1.0,
            max_value=8000.0,
            value=2500.0,
            step=10.0,
        )

user_n = st.sidebar.number_input(
    "Manning's roughness coefficient, n",
    min_value=0.020,
    max_value=0.100,
    value=0.045,
    step=0.001,
    format="%.3f",
)

st.sidebar.caption(
    "Typical values: 0.030 (smooth channel), 0.045 (natural stream), "
    "0.060+ (vegetated floodplain)"
)

st.sidebar.markdown("---")
st.sidebar.header("Map Controls")

basemap_choice = st.sidebar.selectbox(
    "Basemap",
    ["Satellite imagery", "OpenStreetMap", "Light basemap", "OpenTopoMap", "No basemap"],
    index=0,
)

show_flood = st.sidebar.checkbox("Show floodplain / channel inundation extent", value=True)
show_tailrace = st.sidebar.checkbox("Show tailrace discharge", value=True)
show_cross_sections = st.sidebar.checkbox("Show cross sections if available", value=True)
show_dam = st.sidebar.checkbox("Show Soyang River Dam marker", value=True)

fill_raster_gaps = st.sidebar.checkbox(
    "Fill small REL-DEM gaps for smoother display",
    value=True,
)

smooth_extent = st.sidebar.checkbox(
    "Smooth connected floodplain extent",
    value=True,
)

flood_opacity = st.sidebar.slider(
    "Inundation opacity",
    min_value=0.30,
    max_value=1.00,
    value=0.90,
    step=0.05,
)

max_overlay_size = st.sidebar.selectbox(
    "Raster display size",
    [1200, 1600, 2200, 3000],
    index=2,
)

zoom_start = st.sidebar.slider(
    "Initial map zoom",
    min_value=10,
    max_value=18,
    value=14,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.header("Hydraulic Display Controls")

stage_profile_min_fraction = st.sidebar.slider(
    "Upstream-to-downstream stage variation",
    min_value=0.05,
    max_value=1.00,
    value=0.08,
    step=0.01,
    help="Lower values create stronger longitudinal water-surface variation along the tailrace.",
)

lateral_falloff = st.sidebar.slider(
    "Floodplain lateral falloff",
    min_value=0.000,
    max_value=0.020,
    value=0.003,
    step=0.001,
    help="Higher values reduce water depth away from the tailrace line.",
)

invert_longitudinal_gradient = st.sidebar.checkbox(
    "Invert upstream/downstream gradient",
    value=False,
)

display_depth_cap = st.sidebar.number_input(
    "Maximum displayed depth intensity (m)",
    min_value=0.50,
    max_value=50.00,
    value=2.00 if low_flow_mode else 10.00,
    step=0.50,
    format="%.2f",
)

export_dpi = st.sidebar.selectbox(
    "Static PNG export DPI",
    [150, 200, 300],
    index=2,
)


# ============================================================
# File checks
# ============================================================
missing = [p for p in [REL_DEM_PATH, TAILRACE_PATH] if not os.path.exists(p)]

if missing:
    st.error("Missing required file(s):\n\n" + "\n".join(missing))
    st.stop()


# ============================================================
# Functions
# ============================================================
@st.cache_data
def load_vector(path):
    return gpd.read_file(path)


@st.cache_data
def read_raster_for_display(path, max_size):
    with rasterio.open(path) as src:
        raster_crs = src.crs
        bounds = src.bounds

        scale = max(src.height / max_size, src.width / max_size, 1)
        out_height = int(src.height / scale)
        out_width = int(src.width / scale)

        arr = src.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
        )

        data = arr.astype(float).filled(np.nan)

        bounds_4326 = transform_bounds(
            raster_crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            densify_pts=21,
        )

        return data, raster_crs, bounds, bounds_4326


def fill_nodata_nearest(arr):
    if not SCIPY_OK:
        return arr

    mask = ~np.isfinite(arr)

    if not np.any(mask) or np.all(mask):
        return arr

    idx = distance_transform_edt(
        mask,
        return_distances=False,
        return_indices=True,
    )

    return arr[tuple(idx)]


def compute_manning_stage(q, n, rating_points):
    """
    Manning equation:
        Q = (1/n) A R^(2/3) S^(1/2)

    The app uses a calibrated Manning-style response:
        H = a (Q n)^b
    """

    q_vals = np.array([p["Discharge"] for p in rating_points], dtype=float)
    h_vals = np.array([p["Stage"] for p in rating_points], dtype=float)
    n_vals = np.array([p["n"] for p in rating_points], dtype=float)

    x = q_vals * n_vals
    y = h_vals

    valid = (x > 0) & (y > 0)

    log_x = np.log(x[valid])
    log_y = np.log(y[valid])

    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    stage = a * ((q * n) ** b)

    return float(stage), float(a), float(b)


def estimate_display_cell_area_m2(bounds_4326, raster_shape):
    west, south, east, north = bounds_4326
    rows, cols = raster_shape

    center_lat = (south + north) / 2
    meters_per_degree_lat = 111_320
    meters_per_degree_lon = 111_320 * np.cos(np.deg2rad(center_lat))

    pixel_width_m = abs((east - west) / cols) * meters_per_degree_lon
    pixel_height_m = abs((north - south) / rows) * meters_per_degree_lat

    return pixel_width_m * pixel_height_m


def build_spatial_water_surface(
    base_stage,
    raster_bounds,
    raster_shape,
    bounds_4326,
    tailrace_projected,
    min_fraction=0.08,
    lateral_falloff=0.003,
    invert_gradient=False,
):
    """
    Builds a spatially varying water surface.

    The app still computes the base stage from Q and Manning's n.
    This function distributes that stage along and away from the tailrace
    so the displayed extent has realistic longitudinal and lateral variation.
    """

    rows, cols = raster_shape

    transform = from_bounds(
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top,
        cols,
        rows,
    )

    shapes = [(geom, 1) for geom in tailrace_projected.geometry if geom is not None and not geom.is_empty]

    channel_mask = rasterize(
        shapes,
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    if not np.any(channel_mask) or not SCIPY_OK:
        return np.full(raster_shape, base_stage, dtype=float)

    cell_area_m2 = estimate_display_cell_area_m2(bounds_4326, raster_shape)
    cell_size_m = max(np.sqrt(cell_area_m2), 0.001)

    distance_m, nearest_indices = distance_transform_edt(
        ~channel_mask,
        sampling=(cell_size_m, cell_size_m),
        return_indices=True,
    )

    nearest_rows = nearest_indices[0]
    nearest_cols = nearest_indices[1]

    x = raster_bounds.left + (nearest_cols + 0.5) * ((raster_bounds.right - raster_bounds.left) / cols)
    y = raster_bounds.top - (nearest_rows + 0.5) * ((raster_bounds.top - raster_bounds.bottom) / rows)

    channel_rows, channel_cols = np.where(channel_mask)

    channel_x = raster_bounds.left + (channel_cols + 0.5) * ((raster_bounds.right - raster_bounds.left) / cols)
    channel_y = raster_bounds.top - (channel_rows + 0.5) * ((raster_bounds.top - raster_bounds.bottom) / rows)

    coords = np.column_stack([channel_x, channel_y])

    if coords.shape[0] < 2:
        return np.full(raster_shape, base_stage, dtype=float)

    center = coords.mean(axis=0)
    coords_centered = coords - center

    _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
    axis = vh[0]

    projected_all = (np.column_stack([x.ravel(), y.ravel()]) - center) @ axis
    projected_all = projected_all.reshape(raster_shape)

    p_min = np.nanmin(projected_all[channel_mask])
    p_max = np.nanmax(projected_all[channel_mask])

    if p_max == p_min:
        along_norm = np.ones(raster_shape)
    else:
        along_norm = (projected_all - p_min) / (p_max - p_min)
        along_norm = np.clip(along_norm, 0.0, 1.0)

    if invert_gradient:
        along_norm = 1.0 - along_norm

    longitudinal_factor = min_fraction + (1.0 - min_fraction) * along_norm
    channel_stage = base_stage * longitudinal_factor

    local_stage = channel_stage - (lateral_falloff * distance_m)
    local_stage = np.maximum(local_stage, 0.0)

    return local_stage


def make_flood_rgba(depth_grid, flood_mask, color_max):
    colors = [
        "#f2eaff",
        "#c9b8ff",
        "#8d78ff",
        "#4b3dff",
        "#1616d9",
        "#000066",
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "floodplain_intensity",
        colors,
        N=256,
    )

    rgba = np.zeros((*depth_grid.shape, 4), dtype=np.uint8)

    if np.sum(flood_mask) == 0:
        return rgba, cmap, None

    norm = mcolors.Normalize(vmin=0.0, vmax=max(color_max, 0.01))

    display_values = np.clip(np.nan_to_num(depth_grid, nan=0.0), 0.0, color_max)

    rgba_float = cmap(norm(display_values))
    rgba_float[~flood_mask, 3] = 0.0
    rgba_float[flood_mask, 3] = 1.0

    return (rgba_float * 255).astype(np.uint8), cmap, norm


def add_gdf_to_folium(map_obj, gdf_4326, name, color, weight):
    folium.GeoJson(
        gdf_4326,
        name=name,
        style_function=lambda x: {
            "color": color,
            "weight": weight,
            "opacity": 1.0,
            "fillOpacity": 0.0,
        },
    ).add_to(map_obj)


def make_static_png(
    rel_dem,
    depth_grid,
    flood_mask,
    raster_bounds,
    tailrace_projected,
    cross_sections_projected,
    dam_projected,
    q,
    n,
    color_max,
    dpi=300,
):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

    extent = [
        raster_bounds.left,
        raster_bounds.right,
        raster_bounds.bottom,
        raster_bounds.top,
    ]

    ax.imshow(
        rel_dem,
        extent=extent,
        cmap="Greys",
        alpha=0.25,
        origin="upper",
        zorder=1,
    )

    if np.sum(flood_mask) > 0:
        colors = [
            "#f2eaff",
            "#c9b8ff",
            "#8d78ff",
            "#4b3dff",
            "#1616d9",
            "#000066",
        ]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "floodplain_intensity",
            colors,
            N=256,
        )

        norm = mcolors.Normalize(vmin=0.0, vmax=max(color_max, 0.01))

        img = ax.imshow(
            np.where(flood_mask, np.clip(depth_grid, 0.0, color_max), np.nan),
            extent=extent,
            cmap=cmap,
            norm=norm,
            alpha=0.95,
            origin="upper",
            zorder=3,
        )

        cbar = fig.colorbar(img, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label("Relative inundation depth intensity", fontsize=10, weight="bold")

    tailrace_projected.plot(
        ax=ax,
        color="#00BFFF",
        linewidth=2.8,
        label="Tailrace discharge",
        zorder=5,
    )

    if cross_sections_projected is not None:
        cross_sections_projected.plot(
            ax=ax,
            color="red",
            linewidth=2.0,
            label="Cross sections",
            zorder=6,
        )

    dam_projected.plot(
        ax=ax,
        color="#E6007E",
        marker="s",
        markersize=100,
        label="Soyang River Dam",
        zorder=10,
    )

    ax.set_title(
        f"Soyang River Dam Floodplain / Channel Inundation Map\n"
        f"Q = {q:.3f} m³/s; Manning n = {n:.3f}",
        fontsize=13,
        weight="bold",
    )

    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        title="Legend",
    )

    ax.annotate(
        "N",
        xy=(0.96, 0.94),
        xytext=(0.96, 0.86),
        arrowprops=dict(facecolor="black", width=4, headwidth=13),
        ha="center",
        va="center",
        fontsize=12,
        xycoords=ax.transAxes,
    )

    ax.set_xlim(raster_bounds.left, raster_bounds.right)
    ax.set_ylim(raster_bounds.bottom, raster_bounds.top)
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf


# ============================================================
# Load data
# ============================================================
tailrace = load_vector(TAILRACE_PATH)

cross_sections = None
if os.path.exists(CROSS_SECTION_PATH):
    cross_sections = load_vector(CROSS_SECTION_PATH)

rel_dem, raster_crs, raster_bounds, bounds_4326 = read_raster_for_display(
    REL_DEM_PATH,
    max_overlay_size,
)

if raster_crs is None:
    st.error("The relative DEM raster has no CRS.")
    st.stop()

if tailrace.crs is None:
    st.error("Tailrace_discharge.shp has no CRS.")
    st.stop()

tailrace_projected = tailrace.to_crs(raster_crs)

if fill_raster_gaps:
    rel_dem_calc = fill_nodata_nearest(rel_dem)
else:
    rel_dem_calc = rel_dem.copy()


# ============================================================
# Manning computation and spatial water surface
# ============================================================
base_stage, coeff_a, exponent_b = compute_manning_stage(
    q=user_q,
    n=user_n,
    rating_points=RATING_POINTS,
)

local_stage_grid = build_spatial_water_surface(
    base_stage=base_stage,
    raster_bounds=raster_bounds,
    raster_shape=rel_dem_calc.shape,
    bounds_4326=bounds_4326,
    tailrace_projected=tailrace_projected,
    min_fraction=stage_profile_min_fraction,
    lateral_falloff=lateral_falloff,
    invert_gradient=invert_longitudinal_gradient,
)

valid_mask = np.isfinite(rel_dem_calc) & (rel_dem_calc >= 0.0) & (rel_dem_calc < 500.0)

raw_depth = np.where(
    valid_mask,
    local_stage_grid - rel_dem_calc,
    np.nan,
)

raw_depth = np.where(raw_depth > 0.0, raw_depth, np.nan)

flood_mask_raw = np.isfinite(raw_depth)

if smooth_extent and SCIPY_OK and np.sum(flood_mask_raw) > 0:
    flood_mask = binary_closing(flood_mask_raw, structure=np.ones((3, 3)))
    flood_mask = binary_fill_holes(flood_mask)
    flood_mask = flood_mask & np.isfinite(raw_depth)
else:
    flood_mask = flood_mask_raw

depth_grid = np.where(flood_mask, raw_depth, np.nan)

mapped_cells = int(np.sum(flood_mask))
cell_area_m2 = estimate_display_cell_area_m2(bounds_4326, rel_dem_calc.shape)
mapped_extent_km2 = mapped_cells * cell_area_m2 / 1_000_000

color_max = display_depth_cap


# ============================================================
# Sidebar outputs
# ============================================================
st.sidebar.metric("Tailrace discharge", f"{user_q:.3f} m³/s")
st.sidebar.metric("Manning's n", f"{user_n:.3f}")
st.sidebar.metric("Approx. mapped floodplain extent", f"{mapped_extent_km2:.4f} km²")


# ============================================================
# Prepare vectors
# ============================================================
tailrace_4326 = tailrace.to_crs("EPSG:4326")

if cross_sections is not None and cross_sections.crs is not None:
    cross_sections_4326 = cross_sections.to_crs("EPSG:4326")
    cross_sections_projected = cross_sections.to_crs(raster_crs)
else:
    cross_sections_4326 = None
    cross_sections_projected = None

dam_4326 = gpd.GeoDataFrame(
    {"name": ["Soyang River Dam"]},
    geometry=[Point(DAM_LON, DAM_LAT)],
    crs="EPSG:4326",
)

dam_projected = dam_4326.to_crs(raster_crs)


# ============================================================
# Folium map
# ============================================================
west, south, east, north = bounds_4326
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=zoom_start,
    tiles=None,
    control_scale=True,
)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite imagery",
    control=True,
    show=(basemap_choice == "Satellite imagery"),
).add_to(m)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    control=True,
    show=(basemap_choice == "OpenStreetMap"),
).add_to(m)

folium.TileLayer(
    tiles="CartoDB positron",
    name="Light basemap",
    control=True,
    show=(basemap_choice == "Light basemap"),
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="Map data: OpenStreetMap contributors, SRTM | Map style: OpenTopoMap",
    name="OpenTopoMap",
    control=True,
    show=(basemap_choice == "OpenTopoMap"),
).add_to(m)

rgba, cmap, norm = make_flood_rgba(
    depth_grid=depth_grid,
    flood_mask=flood_mask,
    color_max=color_max,
)

if show_flood and mapped_cells > 0:
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=flood_opacity,
        name="Floodplain / channel inundation extent",
        interactive=True,
        cross_origin=False,
        zindex=4,
    ).add_to(m)

if show_tailrace:
    add_gdf_to_folium(
        m,
        tailrace_4326,
        "Tailrace discharge",
        "#00BFFF",
        5,
    )

if show_cross_sections and cross_sections_4326 is not None:
    add_gdf_to_folium(
        m,
        cross_sections_4326,
        "Cross sections",
        "red",
        3,
    )

if show_dam:
    folium.Marker(
        location=[DAM_LAT, DAM_LON],
        popup="Soyang River Dam",
        tooltip="Soyang River Dam",
        icon=folium.Icon(color="pink", icon="info-sign"),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[south, west], [north, east]])


# ============================================================
# Layout
# ============================================================
left, right = st.columns([3, 1])

with left:
    st.subheader("Interactive Floodplain / Channel Inundation Map")
    st_folium(m, width=1150, height=720)

with right:
    st.subheader("Current Scenario")
    st.write(f"**Tailrace discharge:** {user_q:.3f} m³/s")
    st.write(f"**Manning's n:** {user_n:.3f}")
    st.write(f"**Approx. mapped floodplain extent:** {mapped_extent_km2:.4f} km²")
    st.caption(
        "Mapped extent represents the main channel and adjacent engineered floodplain "
        "conveyance area."
    )

    if mapped_cells > 0 and norm is not None:
        fig_cb, ax_cb = plt.subplots(figsize=(3.2, 0.65))
        cb = fig_cb.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            orientation="horizontal",
        )
        cb.set_label("Relative inundation depth intensity", fontsize=9, weight="bold")
        ax_cb.tick_params(labelsize=8)
        st.pyplot(fig_cb)
        plt.close(fig_cb)


# ============================================================
# Downloads
# ============================================================
st.markdown("---")
st.subheader("Downloads")

html_string = m.get_root().render()

st.download_button(
    label="Download interactive map as HTML",
    data=html_string,
    file_name="Soyang_floodplain_inundation_app.html",
    mime="text/html",
)

png_buf = make_static_png(
    rel_dem=rel_dem_calc,
    depth_grid=depth_grid,
    flood_mask=flood_mask,
    raster_bounds=raster_bounds,
    tailrace_projected=tailrace_projected,
    cross_sections_projected=cross_sections_projected,
    dam_projected=dam_projected,
    q=user_q,
    n=user_n,
    color_max=color_max,
    dpi=export_dpi,
)

st.download_button(
    label="Download static thesis map as PNG",
    data=png_buf,
    file_name="Soyang_floodplain_inundation_map.png",
    mime="image/png",
)