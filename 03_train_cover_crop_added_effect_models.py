# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:52:59 2026

@author: ahmed.attia
"""

# ============================================================
# 03_train_cover_crop_added_effect_models.py
# Purpose:
#   Train ML models for the added effect of cover crops relative
#   to improved no-cover-crop rotation N0-L0.
#
# Targets:
#   CC_extra_dSOC_pct
#   CC_extra_dN2O_pct
#   CC_extra_tradeoff_class
#
# Predictors:
#   soil variables + weather variables + RCP + period + cover_crop_type
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict


# ------------------------------------------------------------
# 1. Load cover-crop added-effect dataset
# ------------------------------------------------------------

input_file = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/THP_cover_crop_added_effect_ML_dataset.csv"
model_dir = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/models"
)
df = pd.read_csv(input_file)

print("\nLoaded cover-crop added-effect ML dataset")
print("Shape:", df.shape)
print(df.head())


# ------------------------------------------------------------
# 2. Define predictor groups
# ------------------------------------------------------------

weather_predictors = [
    "Tmax_mean",
    "Tmin_mean",
    "Srad_mean",
    "Prec_mean",
    "CO2A_mean"
]

soil_predictors = [
    # Main SOC-relevant surface layer
    "sand_0_30",
    "silt_0_30",
    "clay_0_30",
    "bd_0_30",
    "soc_0_30",
    "ph_0_30",
    "ll_0_30",
    "dul_0_30",
    "sat_0_30",
    "paw_0_30",
    "ksat_0_30",

    # Deeper water and rooting-zone properties
    "sand_0_100",
    "silt_0_100",
    "clay_0_100",
    "bd_0_100",
    "soc_0_100",
    "ph_0_100",
    "ll_0_100",
    "dul_0_100",
    "sat_0_100",
    "paw_0_100",
    "ksat_0_100"
]
categorical_predictors = [
    "climate_period",
    "cover_crop_type"
]

weather_predictors = [c for c in weather_predictors if c in df.columns]
soil_predictors = [c for c in soil_predictors if c in df.columns]
categorical_predictors = [c for c in categorical_predictors if c in df.columns]

predictors = soil_predictors + weather_predictors + categorical_predictors

joblib.dump(
    predictors,
    model_dir / "CC_added_predictor_list.joblib"
)

print("\nPredictors used:")
for p in predictors:
    print(" -", p)

if len(soil_predictors) == 0:
    print(
        "\nWARNING: No soil predictors found. "
        "The model will run, but publication-quality results need soil properties."
    )


# ------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------

X_all = df[predictors].copy()

numeric_features = X_all.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_all.select_dtypes(exclude=[np.number]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)


# ------------------------------------------------------------
# 4. Regression helper function
# ------------------------------------------------------------

def train_regression_model(df, target, predictors, group_col="site"):
    """Train and evaluate Random Forest regression with GroupKFold by site."""

    model_df = df.dropna(subset=[target]).copy()

    X = model_df[predictors]
    y = model_df[target]
    groups = model_df[group_col]

    rf = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", rf)
    ])

    cv = GroupKFold(n_splits=5)

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    }

    scores = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )

    print(f"\nModel target: {target}")
    print("N:", len(model_df))
    print("Test R2:", round(np.mean(scores["test_r2"]), 3))
    print("Test MAE:", round(-np.mean(scores["test_mae"]), 3))
    print("Test RMSE:", round(-np.mean(scores["test_rmse"]), 3))
    print("Train R2:", round(np.mean(scores["train_r2"]), 3))

    pipe.fit(X, y)

    return pipe, scores


# ------------------------------------------------------------
# 5. Train added SOC effect model
# ------------------------------------------------------------

cc_soc_model, cc_soc_scores = train_regression_model(
    df=df,
    target="CC_extra_dSOC_pct",
    predictors=predictors,
    group_col="site"
)

cc_soc_model, cc_soc_scores = train_regression_model(
    df=df,
    target="CC_extra_dSOC_pct",
    predictors=predictors,
    group_col="site"
)

joblib.dump(
    cc_soc_model,
    model_dir / "RF_CC_added_dSOC_pct_model.joblib"
)
# ------------------------------------------------------------
# 6. Train added N2O effect model
# ------------------------------------------------------------

if "CC_extra_dN2O_pct" in df.columns:
    cc_n2o_model, cc_n2o_scores = train_regression_model(
        df=df,
        target="CC_extra_dN2O_pct",
        predictors=predictors,
        group_col="site"
    )
else:
    print("\nWARNING: CC_extra_dN2O_pct not found. N2O model skipped.")

joblib.dump(
       cc_n2o_model,
       model_dir / "RF_CC_added_dN2O_pct_model.joblib"
   )
# ------------------------------------------------------------
# 7. Cover-crop added-effect trade-off classification
# ------------------------------------------------------------

if "CC_extra_tradeoff_class" in df.columns:

    clf_df = df.dropna(subset=["CC_extra_tradeoff_class"]).copy()

    X = clf_df[predictors]
    y = clf_df["CC_extra_tradeoff_class"]
    groups = clf_df["site"]

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1
    )

    clf_pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", clf)
    ])

    cv = GroupKFold(n_splits=5)

    y_pred = cross_val_predict(
        clf_pipe,
        X,
        y,
        groups=groups,
        cv=cv
    )

    print("\nCover-crop added-effect trade-off classification model")
    print(classification_report(y, y_pred))

    clf_pipe.fit(X, y)

else:
    print("\nWARNING: CC_extra_tradeoff_class not found. Classification model skipped.")

joblib.dump(
    clf_pipe,
    model_dir / "RF_CC_added_tradeoff_classifier.joblib"
)
# ------------------------------------------------------------
# 8. Period-specific summaries
# ------------------------------------------------------------

summary = (
    df
    .groupby(["period", "RCP", "cover_crop_type"])
    .agg(
        mean_CC_extra_dSOC_pct=("CC_extra_dSOC_pct", "mean"),
        sd_CC_extra_dSOC_pct=("CC_extra_dSOC_pct", "std"),
        mean_CC_extra_dN2O_pct=("CC_extra_dN2O_pct", "mean") if "CC_extra_dN2O_pct" in df.columns else ("CC_extra_dSOC_pct", "mean"),
        n=("site", "count")
    )
    .reset_index()
)

summary.to_csv("C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs/CC_added_effect_period_RCP_summary.csv", index=False)

print("\nSaved summary:")
print("CC_added_effect_period_RCP_summary.csv")
print(summary.head())