# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:12:43 2026

@author: ahmed.attia
"""

# ============================================================
# 01_build_ml_datasets.py
# Purpose:
#   Build ML-ready datasets from THP DSSAT rotation outputs.
#
# Outputs:
#   1. THP_BAU_relative_ML_dataset.csv
#   2. THP_cover_crop_added_effect_ML_dataset.csv
#
# Main logic:
#   Track 1: Improved rotations relative to BAU
#       N0-L0 - BAU
#       N0-L2 - BAU
#       N1-L1 - BAU
#
#   Track 2: Cover-crop added effect relative to N0-L0
#       N0-L2 - N0-L0
#       N1-L1 - N0-L0
# ============================================================

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. User inputs
# ------------------------------------------------------------

input_file = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/THP_rotation_outputs_rebuilt.csv"
soil_file = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/soil_properties_from_US_SOL.csv"

output_bau_relative = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/THP_BAU_relative_ML_dataset.csv"
output_cc_added = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/THP_cover_crop_added_effect_ML_dataset.csv"


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

def clean_column_names(df):
    """Clean column names for easier scripting."""
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


def safe_pct_change(new, base):
    """Percentage change with protection against zero division."""
    return np.where(
        base == 0,
        np.nan,
        100 * (new - base) / base
    )


def classify_tradeoff(row, soc_threshold=5, n2o_threshold=5,
                             soc_col="dSOC_pct", n2o_col="dN2O_pct"):
    """
    Three-class SOC-N2O trade-off classification.

    Classes:
        win_win:
            SOC gain is meaningful and N2O does not meaningfully increase.

        weak_benefit:
            SOC gain is small or weak, but N2O does not meaningfully increase.

        tradeoff_or_loss:
            Either N2O meaningfully increases, or SOC response is poor with N2O penalty.
    """

    if pd.isna(row[soc_col]) or pd.isna(row[n2o_col]):
        return np.nan

    soc_gain = row[soc_col] > soc_threshold
    n2o_increase = row[n2o_col] > n2o_threshold

    if soc_gain and not n2o_increase:
        return "win_win"
    elif not soc_gain and not n2o_increase:
        return "weak_benefit"
    else:
        return "tradeoff_or_loss"
    

def classify_added_cc_effect(row, soc_threshold=5, n2o_threshold=5,
                             soc_col="CC_extra_dSOC_pct",
                             n2o_col="CC_extra_dN2O_pct"):
    
    soc_added = row["CC_extra_dSOC_pct"] > soc_threshold
    n2o_penalty = row["CC_extra_dN2O_pct"] > n2o_threshold

    if soc_added and not n2o_penalty:
        return "added_win_win"
    elif not soc_added and not n2o_penalty:
        return "limited_added_benefit"
    else:
        return "added_tradeoff"
    
    
# ------------------------------------------------------------
# 3. Load DSSAT rotation output file
# ------------------------------------------------------------

df = pd.read_csv(input_file)
df = clean_column_names(df)

print("\nLoaded DSSAT output")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. Check required columns
# ------------------------------------------------------------

required_cols = [
    "RCP", "period", "climate", "site", "soil_id",
    "Rotation_number", "lon", "lat",
    "OCTAM_mean", "N2OEM_mean", "CO2EM_mean",
    "Tmax", "Tmin", "Srada", "Prec", "CO2A"
]

missing_required = [c for c in required_cols if c not in df.columns]

if missing_required:
    raise ValueError(
        f"The following required columns are missing: {missing_required}"
    )

print("\nBasic data checks")
print("RCPs:", df["RCP"].unique())
print("Periods:", df["period"].unique())
print("Climate models:", df["climate"].unique())
print("Rotation numbers:", df["Rotation_number"].unique())
print("Number of sites:", df["site"].nunique())


# ------------------------------------------------------------
# 5. Map rotation numbers to rotation names
# ------------------------------------------------------------
# IMPORTANT:
# Confirm this mapping with your original rotation code.
# If the rotation numbers are different, change this dictionary.

rotation_map = {
    1: "BAU",
    2: "N0-L0",
    3: "N0-L2",
    4: "N1-L1"
}

df["rotation"] = df["Rotation_number"].map(rotation_map)

if df["rotation"].isna().any():
    bad_rotations = df.loc[df["rotation"].isna(), "Rotation_number"].unique()
    raise ValueError(
        f"Some Rotation_number values were not mapped: {bad_rotations}"
    )

print("\nRotation mapping")
print(
    df[["Rotation_number", "rotation"]]
    .drop_duplicates()
    .sort_values("Rotation_number")
)


# ------------------------------------------------------------
# 6. Average weather and target variables across climate models
# ------------------------------------------------------------
# We do NOT use climate model name as a predictor.
# Instead, we average the relevant variables across GCMs.

group_cols = [
    "RCP", "period", "site", "soil_id",
    "lon", "lat", "rotation"
]

weather_cols = [
    "Tmax", "Tmin", "Srada", "Prec", "CO2A"
]

target_cols = [
    "OCTAM_mean",    # SOC
    "N2OEM_mean",   # N2O emissions
    "CO2EM_mean",   # soil CO2 emissions
    "ONTAM_mean",   # SON, optional if present
    "NLCM_mean"     # N leaching, optional if present
]

weather_cols = [c for c in weather_cols if c in df.columns]
target_cols = [c for c in target_cols if c in df.columns]

avg_cols = weather_cols + target_cols

df_avg = (
    df[group_cols + avg_cols]
    .groupby(group_cols, as_index=False)
    .mean()
)

df_avg = df_avg.rename(columns={
    "Tmax": "Tmax_mean",
    "Tmin": "Tmin_mean",
    "Srada": "Srad_mean",
    "Prec": "Prec_mean",
    "CO2A": "CO2A_mean"
})

print("\nAveraged across climate models")
print("Shape:", df_avg.shape)
print(df_avg.head())


# ------------------------------------------------------------
# 7. Pivot rotations to wide format
# ------------------------------------------------------------

id_cols = [
    "RCP", "period", "site", "soil_id", "lon", "lat"
]

value_cols = [
    "OCTAM_mean",
    "N2OEM_mean",
    "CO2EM_mean",
    "ONTAM_mean",
    "NLCM_mean"
]

value_cols = [c for c in value_cols if c in df_avg.columns]

wide = df_avg.pivot_table(
    index=id_cols,
    columns="rotation",
    values=value_cols
)

wide.columns = [f"{var}_{rot}" for var, rot in wide.columns]
wide = wide.reset_index()

print("\nWide dataset")
print("Shape:", wide.shape)
print(wide.head())


# ------------------------------------------------------------
# 8. Check all required rotations exist after pivot
# ------------------------------------------------------------

expected_rotations = ["BAU", "N0-L0", "N0-L2", "N1-L1"]

for rot in expected_rotations:
    soc_col = f"OCTAM_mean_{rot}"
    if soc_col not in wide.columns:
        raise ValueError(
            f"Expected SOC column missing after pivot: {soc_col}. "
            "Check rotation mapping."
        )


# ------------------------------------------------------------
# 9. Create Track 1:
#    Improved-rotation effect relative to BAU
# ------------------------------------------------------------

improved_rotations = ["N0-L0", "N0-L2", "N1-L1"]

bau_rows = []

for rot in improved_rotations:
    tmp = wide[id_cols].copy()
    tmp["comparison"] = f"{rot}_minus_BAU"
    tmp["rotation"] = rot

    # Rotation descriptors
    tmp["has_cover_crop"] = np.where(rot in ["N0-L2", "N1-L1"], 1, 0)

    if rot == "N0-L0":
        tmp["cover_crop_type"] = "none"
    elif rot == "N0-L2":
        tmp["cover_crop_type"] = "legume"
    elif rot == "N1-L1":
        tmp["cover_crop_type"] = "legume_nonlegume"

    # SOC
    tmp["SOC_BAU"] = wide["OCTAM_mean_BAU"]
    tmp["SOC_rotation"] = wide[f"OCTAM_mean_{rot}"]
    tmp["dSOC"] = tmp["SOC_rotation"] - tmp["SOC_BAU"]
    tmp["dSOC_pct"] = safe_pct_change(
        tmp["SOC_rotation"],
        tmp["SOC_BAU"]
    )

    # N2O
    if f"N2OEM_mean_{rot}" in wide.columns:
        tmp["N2O_BAU"] = wide["N2OEM_mean_BAU"]
        tmp["N2O_rotation"] = wide[f"N2OEM_mean_{rot}"]
        tmp["dN2O"] = tmp["N2O_rotation"] - tmp["N2O_BAU"]
        tmp["dN2O_pct"] = safe_pct_change(
            tmp["N2O_rotation"],
            tmp["N2O_BAU"]
        )

    # CO2
    if f"CO2EM_mean_{rot}" in wide.columns:
        tmp["CO2_BAU"] = wide["CO2EM_mean_BAU"]
        tmp["CO2_rotation"] = wide[f"CO2EM_mean_{rot}"]
        tmp["dCO2"] = tmp["CO2_rotation"] - tmp["CO2_BAU"]
        tmp["dCO2_pct"] = safe_pct_change(
            tmp["CO2_rotation"],
            tmp["CO2_BAU"]
        )

    # SON
    if f"ONTAM_mean_{rot}" in wide.columns:
        tmp["SON_BAU"] = wide["ONTAM_mean_BAU"]
        tmp["SON_rotation"] = wide[f"ONTAM_mean_{rot}"]
        tmp["dSON"] = tmp["SON_rotation"] - tmp["SON_BAU"]
        tmp["dSON_pct"] = safe_pct_change(
            tmp["SON_rotation"],
            tmp["SON_BAU"]
        )

    # N leaching
    if f"NLCM_mean_{rot}" in wide.columns:
        tmp["Nleach_BAU"] = wide["NLCM_mean_BAU"]
        tmp["Nleach_rotation"] = wide[f"NLCM_mean_{rot}"]
        tmp["dNleach"] = tmp["Nleach_rotation"] - tmp["Nleach_BAU"]
        tmp["dNleach_pct"] = safe_pct_change(
            tmp["Nleach_rotation"],
            tmp["Nleach_BAU"]
        )

    bau_rows.append(tmp)

bau_ml_df = pd.concat(bau_rows, ignore_index=True)

def make_climate_period(row):
    if row["period"] == "Baseline":
        return "Ref_baseline"
    else:
        return f"{row['RCP']}_{row['period']}"

bau_ml_df["climate_period"] = bau_ml_df.apply(make_climate_period, axis=1)


# ------------------------------------------------------------
# 10. Add averaged weather variables to Track 1
# ------------------------------------------------------------

weather_avg_cols = [
    "Tmax_mean", "Tmin_mean", "Srad_mean", "Prec_mean", "CO2A_mean"
]

weather_avg_cols = [c for c in weather_avg_cols if c in df_avg.columns]

weather_df = (
    df_avg[id_cols + weather_avg_cols]
    .drop_duplicates(subset=id_cols)
)

bau_ml_df = bau_ml_df.merge(
    weather_df,
    on=id_cols,
    how="left"
)


# ------------------------------------------------------------
# 11. Define trade-off classes for Track 1
# ------------------------------------------------------------

if "dN2O_pct" in bau_ml_df.columns:
    bau_ml_df["tradeoff_class"] = bau_ml_df.apply(
        lambda row: classify_tradeoff(
            row,
            soc_threshold=5,
            n2o_threshold=5,
            soc_col="dSOC_pct",
            n2o_col="dN2O_pct"
        ),
        axis=1
    )


print("\nTrack 1: BAU-relative dataset")
print("Shape:", bau_ml_df.shape)
print(bau_ml_df.head())

if "tradeoff_class" in bau_ml_df.columns:
    print("\nTrade-off class distribution:")
    print(bau_ml_df["tradeoff_class"].value_counts(dropna=False))


# ------------------------------------------------------------
# 12. Create Track 2:
#     Added cover-crop effect relative to N0-L0
# ------------------------------------------------------------

cc_rotations = ["N0-L2", "N1-L1"]

cc_rows = []

for rot in cc_rotations:
    tmp = wide[id_cols].copy()
    tmp["comparison"] = f"{rot}_minus_N0_L0"
    tmp["rotation"] = rot
    tmp["has_cover_crop"] = 1

    if rot == "N0-L2":
        tmp["cover_crop_type"] = "legume"
    elif rot == "N1-L1":
        tmp["cover_crop_type"] = "legume_nonlegume"

    # SOC added effect
    tmp["SOC_noCC"] = wide["OCTAM_mean_N0-L0"]
    tmp["SOC_CC_rotation"] = wide[f"OCTAM_mean_{rot}"]
    tmp["CC_extra_dSOC"] = tmp["SOC_CC_rotation"] - tmp["SOC_noCC"]
    tmp["CC_extra_dSOC_pct"] = safe_pct_change(
        tmp["SOC_CC_rotation"],
        tmp["SOC_noCC"]
    )

    # N2O added effect
    if f"N2OEM_mean_{rot}" in wide.columns:
        tmp["N2O_noCC"] = wide["N2OEM_mean_N0-L0"]
        tmp["N2O_CC_rotation"] = wide[f"N2OEM_mean_{rot}"]
        tmp["CC_extra_dN2O"] = tmp["N2O_CC_rotation"] - tmp["N2O_noCC"]
        tmp["CC_extra_dN2O_pct"] = safe_pct_change(
            tmp["N2O_CC_rotation"],
            tmp["N2O_noCC"]
        )

    # CO2 added effect
    if f"CO2EM_mean_{rot}" in wide.columns:
        tmp["CO2_noCC"] = wide["CO2EM_mean_N0-L0"]
        tmp["CO2_CC_rotation"] = wide[f"CO2EM_mean_{rot}"]
        tmp["CC_extra_dCO2"] = tmp["CO2_CC_rotation"] - tmp["CO2_noCC"]
        tmp["CC_extra_dCO2_pct"] = safe_pct_change(
            tmp["CO2_CC_rotation"],
            tmp["CO2_noCC"]
        )

    # SON added effect
    if f"ONTAM_mean_{rot}" in wide.columns:
        tmp["SON_noCC"] = wide["ONTAM_mean_N0-L0"]
        tmp["SON_CC_rotation"] = wide[f"ONTAM_mean_{rot}"]
        tmp["CC_extra_dSON"] = tmp["SON_CC_rotation"] - tmp["SON_noCC"]
        tmp["CC_extra_dSON_pct"] = safe_pct_change(
            tmp["SON_CC_rotation"],
            tmp["SON_noCC"]
        )

    # N leaching added effect
    if f"NLCM_mean_{rot}" in wide.columns:
        tmp["Nleach_noCC"] = wide["NLCM_mean_N0-L0"]
        tmp["Nleach_CC_rotation"] = wide[f"NLCM_mean_{rot}"]
        tmp["CC_extra_dNleach"] = tmp["Nleach_CC_rotation"] - tmp["Nleach_noCC"]
        tmp["CC_extra_dNleach_pct"] = safe_pct_change(
            tmp["Nleach_CC_rotation"],
            tmp["Nleach_noCC"]
        )

    cc_rows.append(tmp)

cc_ml_df = pd.concat(cc_rows, ignore_index=True)

cc_ml_df = cc_ml_df.merge(
    weather_df,
    on=id_cols,
    how="left"
)

cc_ml_df["climate_period"] = cc_ml_df.apply(make_climate_period, axis=1)
# ------------------------------------------------------------
# 13. Define trade-off classes for Track 2
# ------------------------------------------------------------

if "CC_extra_dN2O_pct" in cc_ml_df.columns:
    cc_ml_df["CC_extra_tradeoff_class"] = cc_ml_df.apply(
        lambda row: classify_added_cc_effect(
            row,
            soc_threshold=5,
            n2o_threshold=5,
            soc_col="CC_extra_dSOC_pct",
            n2o_col="CC_extra_dN2O_pct"
        ),
        axis=1
    )

print("\nTrack 2: Cover-crop added-effect dataset")
print("Shape:", cc_ml_df.shape)
print(cc_ml_df.head())

if "CC_extra_tradeoff_class" in cc_ml_df.columns:
    print("\nCover-crop added-effect trade-off class distribution:")
    print(cc_ml_df["CC_extra_tradeoff_class"].value_counts(dropna=False))


# ------------------------------------------------------------
# 14. Merge soil properties if available
# ------------------------------------------------------------

try:
    soil = pd.read_csv(soil_file)
    soil = clean_column_names(soil)

    if "soil_id" not in soil.columns:
        raise ValueError("soil_properties.csv must contain a soil_id column.")

    bau_ml_df = bau_ml_df.merge(soil, on="soil_id", how="left")
    cc_ml_df = cc_ml_df.merge(soil, on="soil_id", how="left")

    print("\nSoil properties merged successfully.")
    print("Track 1 shape after soil merge:", bau_ml_df.shape)
    print("Track 2 shape after soil merge:", cc_ml_df.shape)

except FileNotFoundError:
    print(
        "\nWARNING: soil_properties.csv was not found. "
        "Datasets were saved without soil properties."
    )

except Exception as e:
    print("\nWARNING: Soil merge failed.")
    print(str(e))


# ------------------------------------------------------------
# 15. Save outputs
# ------------------------------------------------------------

bau_ml_df.to_csv(output_bau_relative, index=False)
cc_ml_df.to_csv(output_cc_added, index=False)

print("\nSaved output files:")
print(output_bau_relative)
print(output_cc_added)