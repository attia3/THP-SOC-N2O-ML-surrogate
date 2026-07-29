# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:32:47 2026

@author: ahmed.attia
"""

# ============================================================
# 00_extract_soil_properties_from_US_SOL.py
#
# Purpose:
#   Extract ML-ready soil properties from DSSAT US.SOL
#   for only the soil_id values used in the THP DSSAT output.
#
# Inputs:
#   1. THP_rotation_outputs.csv
#   2. US.SOL
#
# Output:
#   soil_properties_from_US_SOL.csv
#
# Notes:
#   DSSAT soil layer columns usually include:
#   SLB   = lower depth of soil layer, cm
#   SLLL  = lower limit, cm3 cm-3
#   SDUL  = drained upper limit, cm3 cm-3
#   SSAT  = saturation, cm3 cm-3
#   SRGF  = root growth factor
#   SSKS  = saturated hydraulic conductivity
#   SBDM  = bulk density, g cm-3
#   SLOC  = soil organic carbon, g kg-1 or %
#          depending on the source/conversion.
#          In many DSSAT soil files this is organic carbon percentage.
#   SLCL  = clay, %
#   SLSI  = silt, %
#   SLHW  = pH in water
# ============================================================

import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. User paths
# ------------------------------------------------------------

dssat_output_file = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/THP_rotation_outputs_rebuilt.csv"
us_sol_file = "C:/DSSAT48/Soil/US.SOL"
output_soil_file = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/soil_properties_from_US_SOL.csv"


# ------------------------------------------------------------
# 2. Helper functions
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


def parse_float(value):
    """Convert DSSAT missing values such as -99 or -99.0 to NaN."""
    try:
        x = float(value)
        if x <= -90:
            return np.nan
        return x
    except Exception:
        return np.nan


def parse_soil_profile(lines, start_idx):
    """
    Parse one DSSAT soil profile starting at a line beginning with '*'.

    Returns:
        profile_id, metadata dict, layer dataframe
    """

    header_line = lines[start_idx].rstrip("\n")

    # First token after * is the soil profile ID
    # Example:
    # *US01006921    USA   SandyLoam   200    ISRIC soilgrids + HC27
    profile_id = header_line[1:].split()[0]

    # Try to extract texture and depth from header
    header_tokens = header_line[1:].split()
    texture_class = np.nan
    profile_depth_cm = np.nan

    if len(header_tokens) >= 4:
        # Often: ID COUNTRY TEXTURE DEPTH ...
        texture_class = header_tokens[2]
        profile_depth_cm = parse_float(header_tokens[3])

    # Move through the profile until the next '*' or end of file
    end_idx = start_idx + 1
    while end_idx < len(lines) and not lines[end_idx].startswith("*"):
        end_idx += 1

    profile_lines = lines[start_idx:end_idx]

    # Find SITE line
    site_lat = np.nan
    site_lon = np.nan
    scs_family = np.nan

    for i, line in enumerate(profile_lines):
        if line.startswith("@SITE"):
            if i + 1 < len(profile_lines):
                site_values = profile_lines[i + 1].split()
                # Example:
                # -99 US 70.541 -149.875 HC_GEN0012
                if len(site_values) >= 4:
                    site_lat = parse_float(site_values[2])
                    site_lon = parse_float(site_values[3])
                if len(site_values) >= 5:
                    scs_family = site_values[4]

    # Find layer table header
    layer_header_idx = None
    layer_cols = None

    for i, line in enumerate(profile_lines):
        if line.startswith("@") and "SLB" in line and "SLLL" in line:
            layer_header_idx = i
            layer_cols = line.replace("@", "").split()
            break

    if layer_header_idx is None:
        return profile_id, {
            "soil_texture_class": texture_class,
            "profile_depth_cm": profile_depth_cm,
            "soil_lat_USSOL": site_lat,
            "soil_lon_USSOL": site_lon,
            "scs_family": scs_family
        }, pd.DataFrame()

    # Parse layer rows until blank line, next header, or next profile
    layer_rows = []

    for line in profile_lines[layer_header_idx + 1:]:
        line_stripped = line.strip()

        if line_stripped == "":
            break

        if line_stripped.startswith("@"):
            break

        if line_stripped.startswith("*"):
            break

        parts = line_stripped.split()

        # Need at least SLB and SLMH plus numeric columns
        if len(parts) < 3:
            continue

        # Some soil files have SLMH as text after SLB
        # The header is:
        # SLB SLMH SLLL SDUL SSAT SRGF SSKS SBDM SLOC SLCL SLSI ...
        row = {}

        for col, val in zip(layer_cols, parts):
            if col == "SLMH":
                row[col] = val
            else:
                row[col] = parse_float(val)

        layer_rows.append(row)

    layers = pd.DataFrame(layer_rows)

    metadata = {
        "soil_texture_class": texture_class,
        "profile_depth_cm": profile_depth_cm,
        "soil_lat_USSOL": site_lat,
        "soil_lon_USSOL": site_lon,
        "scs_family": scs_family
    }

    return profile_id, metadata, layers


def weighted_mean_to_depth(layers, var, max_depth):
    """
    Depth-weighted mean of a soil variable from 0 to max_depth cm.

    Uses SLB as lower boundary of each layer and assumes previous
    layer lower boundary as upper boundary.
    """

    if layers.empty or var not in layers.columns or "SLB" not in layers.columns:
        return np.nan

    df = layers[["SLB", var]].copy()
    df = df.dropna(subset=["SLB", var])
    df = df.sort_values("SLB")

    if df.empty:
        return np.nan

    upper = 0.0
    weighted_sum = 0.0
    total_thickness = 0.0

    for _, row in df.iterrows():
        lower = float(row["SLB"])

        layer_upper = upper
        layer_lower = lower

        # overlap with 0-max_depth
        overlap_upper = max(layer_upper, 0.0)
        overlap_lower = min(layer_lower, max_depth)

        thickness = overlap_lower - overlap_upper

        if thickness > 0:
            weighted_sum += row[var] * thickness
            total_thickness += thickness

        upper = lower

        if upper >= max_depth:
            break

    if total_thickness == 0:
        return np.nan

    return weighted_sum / total_thickness


def derive_soil_properties(profile_id, metadata, layers):
    """
    Convert one layer table into one ML-ready row.
    """

    out = {
        "soil_id": profile_id,
        **metadata
    }

    # Layer-wise weighted means
    for depth in [30, 60, 100, 200]:
        suffix = f"0_{depth}"

        out[f"ll_{suffix}"] = weighted_mean_to_depth(layers, "SLLL", depth)
        out[f"dul_{suffix}"] = weighted_mean_to_depth(layers, "SDUL", depth)
        out[f"sat_{suffix}"] = weighted_mean_to_depth(layers, "SSAT", depth)
        out[f"bd_{suffix}"] = weighted_mean_to_depth(layers, "SBDM", depth)
        out[f"soc_{suffix}"] = weighted_mean_to_depth(layers, "SLOC", depth)
        out[f"clay_{suffix}"] = weighted_mean_to_depth(layers, "SLCL", depth)
        out[f"silt_{suffix}"] = weighted_mean_to_depth(layers, "SLSI", depth)
        out[f"ph_{suffix}"] = weighted_mean_to_depth(layers, "SLHW", depth)
        out[f"ksat_{suffix}"] = weighted_mean_to_depth(layers, "SSKS", depth)

        # Plant available water
        if not pd.isna(out[f"dul_{suffix}"]) and not pd.isna(out[f"ll_{suffix}"]):
            out[f"paw_{suffix}"] = out[f"dul_{suffix}"] - out[f"ll_{suffix}"]
        else:
            out[f"paw_{suffix}"] = np.nan

        # Sand is not directly listed in DSSAT layer table.
        # Approximate as 100 - clay - silt.
        if not pd.isna(out[f"clay_{suffix}"]) and not pd.isna(out[f"silt_{suffix}"]):
            out[f"sand_{suffix}"] = 100 - out[f"clay_{suffix}"] - out[f"silt_{suffix}"]
        else:
            out[f"sand_{suffix}"] = np.nan

    # Top-layer variables may also be useful
    if not layers.empty:
        layers_sorted = layers.sort_values("SLB").copy()
        first = layers_sorted.iloc[0]

        for var, name in [
            ("SLLL", "ll_top"),
            ("SDUL", "dul_top"),
            ("SSAT", "sat_top"),
            ("SBDM", "bd_top"),
            ("SLOC", "soc_top"),
            ("SLCL", "clay_top"),
            ("SLSI", "silt_top"),
            ("SLHW", "ph_top"),
            ("SSKS", "ksat_top")
        ]:
            out[name] = first[var] if var in layers_sorted.columns else np.nan

        if not pd.isna(out.get("clay_top")) and not pd.isna(out.get("silt_top")):
            out["sand_top"] = 100 - out["clay_top"] - out["silt_top"]
        else:
            out["sand_top"] = np.nan

    return out


# ------------------------------------------------------------
# 3. Read DSSAT output to get required soil IDs
# ------------------------------------------------------------

dssat_df = pd.read_csv(dssat_output_file)
dssat_df = clean_column_names(dssat_df)

if "soil_id" not in dssat_df.columns:
    raise ValueError("The DSSAT output file must contain a soil_id column.")

needed_soil_ids = (
    dssat_df["soil_id"]
    .astype(str)
    .str.strip()
    .unique()
)

needed_soil_ids = set(needed_soil_ids)

print("\nNumber of unique soil IDs needed:", len(needed_soil_ids))


# ------------------------------------------------------------
# 4. Read and parse US.SOL
# ------------------------------------------------------------

with open(us_sol_file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

profile_start_indices = [
    i for i, line in enumerate(lines)
    if line.startswith("*")
]

print("Number of profiles in US.SOL:", len(profile_start_indices))

soil_rows = []

for start_idx in profile_start_indices:
    profile_id, metadata, layers = parse_soil_profile(lines, start_idx)

    if profile_id in needed_soil_ids:
        row = derive_soil_properties(profile_id, metadata, layers)
        soil_rows.append(row)

soil_props = pd.DataFrame(soil_rows)

print("\nExtracted soil profiles:", soil_props.shape[0])

missing_ids = sorted(list(needed_soil_ids - set(soil_props["soil_id"].astype(str))))

if missing_ids:
    print("\nWARNING: Some soil IDs from the DSSAT output were not found in US.SOL.")
    print("Number missing:", len(missing_ids))
    print("First 20 missing IDs:", missing_ids[:20])
else:
    print("\nAll required soil IDs were found in US.SOL.")


# ------------------------------------------------------------
# 5. Save soil properties
# ------------------------------------------------------------

soil_props.to_csv(output_soil_file, index=False)

print("\nSaved:")
print(output_soil_file)

print("\nPreview:")
print(soil_props.head())






