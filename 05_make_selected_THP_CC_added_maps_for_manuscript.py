# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:32:00 2026

@author: ahmed.attia
"""

# ============================================================
# 05_make_selected_THP_CC_added_maps_for_manuscript.py
#
# Purpose:
#   Apply trained added cover-crop ML models across THP points
#   and create selected manuscript-ready maps clipped by CDL.
#
# Scientific comparison:
#   N0-L2 - N0-L0
#   N1-L1 - N0-L0
#
# Selected maps:
#   Cover crop types:
#       legume
#       legume_nonlegume
#
#   Periods:
#       2030s
#       2070s
#
#   Variables:
#       pred_CC_extra_dSOC_pct
#       pred_CC_extra_dN2O_pct
#       predicted added cover-crop trade-off class
#
# Main manuscript figures:
#   Figure_7_CC_added_dSOC_pct_selected.png
#   Figure_8_CC_added_dN2O_pct_selected.png
#   Figure_9_CC_added_tradeoff_selected.png
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ------------------------------------------------------------
# 1. User paths
# ------------------------------------------------------------

base_dir = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs"
)

ml_file = base_dir / "THP_cover_crop_added_effect_ML_dataset.csv"

model_dir = base_dir / "models"

cc_soc_model_file = model_dir / "RF_CC_added_dSOC_pct_model.joblib"
cc_n2o_model_file = model_dir / "RF_CC_added_dN2O_pct_model.joblib"
predictor_file = model_dir / "CC_added_predictor_list.joblib"

# Change these to your actual files
thp_boundary_file  = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/SpatialData_TX/TXhighPlains_counties.shp"
)

# Texas county boundaries
county_boundary_file = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/ML_study/SpatialData/TX_counties.shp"
)

cdl_raster_file = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/ML_study/SpatialData/THP_CDL_ras/CDL_THP_2023_3857.tif"
)

output_dir = base_dir / "THP_ML_selected_CC_added_maps"
output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. User settings
# ------------------------------------------------------------

target_crs = "EPSG:4326"
map_resolution  = 0.01

selected_cover_crop_types = ["legume", "legume_nonlegume"]
selected_periods = ["2030s", "2070s"]

# Use "mean" or "median" across RCP45 and RCP85
rcp_aggregation = "mean"

# CDL crop codes to retain
cdl_crop_codes = [
    1, 2, 4, 24,          # corn, cotton, sorghum, winter wheat
    5, 21, 22, 23,        # soybean, barley, durum wheat, spring wheat
    28, 29, 36, 37,       # oats, millet, alfalfa, other hay/non-alfalfa
    58, 59, 60, 61,       # clover/wildflowers, sod/grass seed, switchgrass, fallow/idle cropland
    225, 226, 230, 234, 236, 238, 240, 254,
    253                   # non-irrigated winter wheat
]
# Trade-off thresholds
soc_threshold = 5
n2o_threshold = 5

# Trade-off codes:
# 1 = added win-win
# 2 = limited added benefit
# 3 = added trade-off
tradeoff_labels = {
    1: "High added SOC / low N$_2$O penalty",
    2: "High added SOC / moderate N$_2$O penalty",
    3: "Moderate added benefit",
    4: "Strong N$_2$O trade-off"
}

# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------

def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def create_template_from_boundary(boundary_gdf, resolution, crs):
    boundary_proj = boundary_gdf.to_crs(crs)

    minx, miny, maxx, maxy = boundary_proj.total_bounds

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    transform = from_origin(minx, maxy, resolution, resolution)

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "lzw"
    }

    return meta, boundary_proj


def rasterize_boundary_mask(boundary_gdf, template_meta):
    shapes = [(geom, 1) for geom in boundary_gdf.geometry]

    mask_arr = rasterize(
        shapes=shapes,
        out_shape=(template_meta["height"], template_meta["width"]),
        transform=template_meta["transform"],
        fill=0,
        dtype="uint8"
    )

    return mask_arr == 1


def reproject_cdl_to_template(cdl_file, template_meta, output_file):
    with rasterio.open(cdl_file) as src:
        dst_meta = template_meta.copy()
        dst_meta.update({
            "count": 1,
            "dtype": src.dtypes[0],
            "nodata": src.nodata
        })

        with rasterio.open(output_file, "w", **dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_meta["transform"],
                dst_crs=dst_meta["crs"],
                resampling=Resampling.nearest
            )


def create_cdl_mask(cdl_file, template_meta, crop_codes, temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)

    cdl_reprojected = temp_dir / "cdl_reprojected_to_template.tif"

    reproject_cdl_to_template(
        cdl_file=cdl_file,
        template_meta=template_meta,
        output_file=cdl_reprojected
    )

    with rasterio.open(cdl_reprojected) as src:
        cdl = src.read(1)

    return np.isin(cdl, crop_codes)


def interpolate_points_to_grid(points_gdf, value_col, template_meta):
    transform = template_meta["transform"]
    width = template_meta["width"]
    height = template_meta["height"]

    xs = np.arange(width) * transform.a + transform.c + transform.a / 2
    ys = np.arange(height) * transform.e + transform.f + transform.e / 2

    grid_x, grid_y = np.meshgrid(xs, ys)

    valid = points_gdf.dropna(subset=[value_col]).copy()

    if len(valid) < 3:
        raise ValueError(f"Not enough valid points for {value_col}.")

    x = valid.geometry.x.values
    y = valid.geometry.y.values
    z = valid[value_col].values

    grid_linear = griddata(
        points=(x, y),
        values=z,
        xi=(grid_x, grid_y),
        method="linear"
    )

    grid_nearest = griddata(
        points=(x, y),
        values=z,
        xi=(grid_x, grid_y),
        method="nearest"
    )

    grid_out = np.where(np.isnan(grid_linear), grid_nearest, grid_linear)

    return grid_out.astype("float32")


def write_geotiff(output_path, array, template_meta, dtype="float32", nodata=-9999):
    meta = template_meta.copy()
    meta.update({
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "lzw"
    })

    arr = array.copy()
    arr_out = np.where(np.isnan(arr), nodata, arr).astype(dtype)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr_out, 1)


def classify_added_cc_tradeoff_from_predictions(dsoc_grid, dn2o_grid,
                                                 soc_high=7.5,
                                                 n2o_low=8,
                                                 n2o_high=15):
    """
    4-class classification for added cover-crop effect relative to N0-L0.

    1 = high added SOC / low N2O penalty
        dSOC >= soc_high and dN2O <= n2o_low

    2 = high added SOC / moderate N2O penalty
        dSOC >= soc_high and n2o_low < dN2O <= n2o_high

    3 = moderate added benefit
        dSOC < soc_high and dN2O <= n2o_high

    4 = strong N2O trade-off
        dN2O > n2o_high
    """

    out = np.full(dsoc_grid.shape, np.nan, dtype="float32")

    valid = ~np.isnan(dsoc_grid) & ~np.isnan(dn2o_grid)

    class1 = (
        valid &
        (dsoc_grid >= soc_high) &
        (dn2o_grid <= n2o_low)
    )

    class2 = (
        valid &
        (dsoc_grid >= soc_high) &
        (dn2o_grid > n2o_low) &
        (dn2o_grid <= n2o_high)
    )

    class3 = (
        valid &
        (dsoc_grid < soc_high) &
        (dn2o_grid <= n2o_high)
    )

    class4 = (
        valid &
        (dn2o_grid > n2o_high)
    )

    out[class1] = 1
    out[class2] = 2
    out[class3] = 3
    out[class4] = 4

    return out


def get_extent_from_template(template_meta):
    transform = template_meta["transform"]
    width = template_meta["width"]
    height = template_meta["height"]

    xmin = transform.c
    xmax = transform.c + width * transform.a
    ymax = transform.f
    ymin = transform.f + height * transform.e

    return [xmin, xmax, ymin, ymax]


def make_continuous_panel_figure(grid_dict, variable_name, title, colorbar_label,
                                 output_png, template_meta, counties_gdf,
                                 thp_boundary_gdf, cmap_name="viridis",
                                 vmin=None, vmax=None):
    """
    Make 2 x 2 panel:
        rows = cover_crop_type
        columns = periods
    """

    extent = get_extent_from_template(template_meta)

    arrays = []
    for cc_type in selected_cover_crop_types:
        for per in selected_periods:
            arrays.append(grid_dict[(cc_type, per, variable_name)])

    valid_values = np.concatenate([a[~np.isnan(a)] for a in arrays])

    if vmin is None:
        vmin = np.nanpercentile(valid_values, 2)

    if vmax is None:
        vmax = np.nanpercentile(valid_values, 98)

    fig, axes = plt.subplots(
        nrows=len(selected_cover_crop_types),
        ncols=len(selected_periods),
        figsize=(10.5, 9.5),
        constrained_layout=False,
        gridspec_kw={
            "wspace": 0.04,
            "hspace": 0.10
        }
    )

    row_labels = {
        "legume": "N0-L2 added effect",
        "legume_nonlegume": "N1-L1 added effect"
    }

    panel_labels = ["(A)", "(B)", "(C)", "(D)"]
    panel_idx = 0

    for i, cc_type in enumerate(selected_cover_crop_types):
        for j, per in enumerate(selected_periods):
            ax = axes[i, j]
            arr = grid_dict[(cc_type, per, variable_name)]

            im = ax.imshow(
                arr,
                extent=extent,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_name,
                interpolation="nearest"
            )

            counties_gdf.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.35,
                zorder=3
            )

            thp_boundary_gdf.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.8,
                zorder=4
            )

            ax.set_title(
                f"{panel_labels[panel_idx]} {row_labels.get(cc_type, cc_type)}, {per}",
                fontsize=13,
                fontweight="bold",
                pad=6
            )
            panel_idx += 1

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.03,
        right=0.86,
        bottom=0.04,
        top=0.90,
        wspace=0.03,
        hspace=0.14
    )

    cbar_ax = fig.add_axes([0.88, 0.18, 0.025, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(colorbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(title, fontsize=18, y=0.975)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_panel_figure(grid_dict, output_png, template_meta,
                               counties_gdf, thp_boundary_gdf):
    extent = get_extent_from_template(template_meta)

    cmap = ListedColormap([
        "#1a9850",  # dark green
        "#91cf60",  # light green
        "#fee08b",  # yellow
        "#d73027"   # red
    ])
    
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, axes = plt.subplots(
        nrows=len(selected_cover_crop_types),
        ncols=len(selected_periods),
        figsize=(10.5, 9.5),
        constrained_layout=False,
        gridspec_kw={
            "wspace": 0.04,
            "hspace": 0.10
        }
    )

    row_labels = {
        "legume": "N0-L2 added effect",
        "legume_nonlegume": "N1-L1 added effect"
    }

    panel_labels = ["(A)", "(B)", "(C)", "(D)"]
    panel_idx = 0

    for i, cc_type in enumerate(selected_cover_crop_types):
        for j, per in enumerate(selected_periods):
            ax = axes[i, j]
            arr = grid_dict[(cc_type, per, "tradeoff")]

            im = ax.imshow(
                arr,
                extent=extent,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest"
            )

            counties_gdf.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.35,
                zorder=3
            )

            thp_boundary_gdf.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.8,
                zorder=4
            )

            ax.set_title(
                f"{panel_labels[panel_idx]} {row_labels.get(cc_type, cc_type)}, {per}",
                fontsize=13,
                fontweight="bold",
                pad=6
            )
            panel_idx += 1

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.03,
        right=0.86,
        bottom=0.04,
        top=0.90,
        wspace=0.03,
        hspace=0.14
    )

    cbar_ax = fig.add_axes([0.88, 0.18, 0.04, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        tradeoff_labels[1],
        tradeoff_labels[2],
        tradeoff_labels[3],
        tradeoff_labels[4]
    ])
    cbar.set_label(r"Added cover-crop SOC–N$_2$O class", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle("Predicted added cover-crop SOC–N$_2$O trade-off zones", fontsize=18, y=0.975)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 4. Load ML data and models
# ------------------------------------------------------------

df = pd.read_csv(ml_file)
df = clean_column_names(df)

cc_soc_model = joblib.load(cc_soc_model_file)
cc_n2o_model = joblib.load(cc_n2o_model_file)
predictors = joblib.load(predictor_file)

missing_predictors = [p for p in predictors if p not in df.columns]
if missing_predictors:
    raise ValueError(f"Missing predictors in ML dataset: {missing_predictors}")

print("\nLoaded CC added-effect ML dataset:", df.shape)
print("Predictors used:")
print(predictors)


# ------------------------------------------------------------
# 5. Predict at THP points
# ------------------------------------------------------------

df["pred_CC_extra_dSOC_pct"] = cc_soc_model.predict(df[predictors])
df["pred_CC_extra_dN2O_pct"] = cc_n2o_model.predict(df[predictors])

point_output = output_dir / "THP_selected_point_predictions_CC_added.csv"
df.to_csv(point_output, index=False)

print("\nSaved point predictions:")
print(point_output)


# ------------------------------------------------------------
# 6. Prepare template and CDL mask
# ------------------------------------------------------------

thp = gpd.read_file(thp_boundary_file).to_crs(target_crs)

counties = gpd.read_file(county_boundary_file).to_crs(target_crs)
counties_thp = gpd.clip(counties, thp)

template_meta, thp_proj = create_template_from_boundary(
    boundary_gdf=thp,
    resolution=map_resolution,
    crs=target_crs
)

boundary_mask = rasterize_boundary_mask(thp_proj, template_meta)

cdl_mask = create_cdl_mask(
    cdl_file=cdl_raster_file,
    template_meta=template_meta,
    crop_codes=cdl_crop_codes,
    temp_dir=output_dir / "temp"
)

combined_mask = boundary_mask & cdl_mask

print("\nPixels inside THP + CDL mask:", combined_mask.sum())


# ------------------------------------------------------------
# 7. Convert point predictions to projected GeoDataFrame
# ------------------------------------------------------------

points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
).to_crs(target_crs)

points["x_proj"] = points.geometry.x
points["y_proj"] = points.geometry.y


# ------------------------------------------------------------
# 8. Interpolate selected maps
# ------------------------------------------------------------

grid_dict = {}

for cc_type in selected_cover_crop_types:
    for period in selected_periods:

        sub = points[
            (points["cover_crop_type"] == cc_type) &
            (points["period"] == period)
        ].copy()

        if sub.empty:
            print(f"Skipping empty subset: {cc_type}, {period}")
            continue

        group_cols = [
            "site", "soil_id", "lon", "lat", "x_proj", "y_proj"
        ]

        if rcp_aggregation == "mean":
            sub_agg = (
                sub
                .groupby(group_cols, as_index=False)[
                    ["pred_CC_extra_dSOC_pct", "pred_CC_extra_dN2O_pct"]
                ]
                .mean()
            )
        elif rcp_aggregation == "median":
            sub_agg = (
                sub
                .groupby(group_cols, as_index=False)[
                    ["pred_CC_extra_dSOC_pct", "pred_CC_extra_dN2O_pct"]
                ]
                .median()
            )
        else:
            raise ValueError("rcp_aggregation must be 'mean' or 'median'.")

        sub_agg = gpd.GeoDataFrame(
            sub_agg,
            geometry=gpd.points_from_xy(sub_agg["x_proj"], sub_agg["y_proj"]),
            crs=target_crs
        )

        dsoc_grid = interpolate_points_to_grid(
            points_gdf=sub_agg,
            value_col="pred_CC_extra_dSOC_pct",
            template_meta=template_meta
        )

        dn2o_grid = interpolate_points_to_grid(
            points_gdf=sub_agg,
            value_col="pred_CC_extra_dN2O_pct",
            template_meta=template_meta
        )

        dsoc_grid = np.where(combined_mask, dsoc_grid, np.nan)
        dn2o_grid = np.where(combined_mask, dn2o_grid, np.nan)

        tradeoff_grid = classify_added_cc_tradeoff_from_predictions(
            dsoc_grid=dsoc_grid,
            dn2o_grid=dn2o_grid,
            soc_high=7.5,
            n2o_low=8,
            n2o_high=15
        )

        tradeoff_grid = np.where(combined_mask, tradeoff_grid, np.nan)

        grid_dict[(cc_type, period, "pred_CC_extra_dSOC_pct")] = dsoc_grid
        grid_dict[(cc_type, period, "pred_CC_extra_dN2O_pct")] = dn2o_grid
        grid_dict[(cc_type, period, "tradeoff")] = tradeoff_grid

        safe_cc_type = cc_type.replace("_", "")
        safe_period = period.replace(" ", "")

        dsoc_tif = output_dir / f"pred_CC_extra_dSOC_pct_{safe_cc_type}_{safe_period}_{rcp_aggregation}RCP_CDLmasked.tif"
        dn2o_tif = output_dir / f"pred_CC_extra_dN2O_pct_{safe_cc_type}_{safe_period}_{rcp_aggregation}RCP_CDLmasked.tif"
        trade_tif = output_dir / f"pred_CC_added_tradeoff_{safe_cc_type}_{safe_period}_{rcp_aggregation}RCP_CDLmasked.tif"

        write_geotiff(dsoc_tif, dsoc_grid, template_meta, dtype="float32", nodata=-9999)
        write_geotiff(dn2o_tif, dn2o_grid, template_meta, dtype="float32", nodata=-9999)
        write_geotiff(trade_tif, tradeoff_grid, template_meta, dtype="int16", nodata=-9999)

        print("Saved:")
        print(" ", dsoc_tif)
        print(" ", dn2o_tif)
        print(" ", trade_tif)


# ------------------------------------------------------------
# 9. Make manuscript-ready panel figures
# ------------------------------------------------------------

figure7 = output_dir / "Figure_7_CC_added_dSOC_pct_selected.png"
figure8 = output_dir / "Figure_8_CC_added_dN2O_pct_selected.png"
figure9 = output_dir / "Figure_9_CC_added_tradeoff_selected.png"

make_continuous_panel_figure(
    grid_dict=grid_dict,
    variable_name="pred_CC_extra_dSOC_pct",
    title="Predicted added SOC benefit of cover crops relative to N0-L0",
    colorbar_label="Predicted added SOC change (%)",
    output_png=figure7,
    template_meta=template_meta,
    counties_gdf=counties_thp,
    thp_boundary_gdf=thp_proj,
    cmap_name="turbo_r",
    vmin = 5.5,
    vmax = 10.0
)

make_continuous_panel_figure(
    grid_dict=grid_dict,
    variable_name="pred_CC_extra_dN2O_pct",
    title=r"Predicted added N$_2$O effect of cover crops relative to N0-L0",
    colorbar_label=r"Predicted added N$_2$O change (%)",
    output_png=figure8,
    template_meta=template_meta,
    counties_gdf=counties_thp,
    thp_boundary_gdf=thp_proj,
    cmap_name="YlOrRd",
    vmin = 5,
    vmax = 22
)

make_tradeoff_panel_figure(
    grid_dict=grid_dict,
    output_png=figure9,
    template_meta=template_meta,
    counties_gdf=counties_thp,
    thp_boundary_gdf=thp_proj
)

print("\nSaved manuscript figures:")
print(figure7)
print(figure8)
print(figure9)

print("\nFinished selected THP added cover-crop manuscript maps.")