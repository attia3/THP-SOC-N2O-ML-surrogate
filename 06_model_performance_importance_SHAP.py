# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:18:35 2026

@author: ahmed.attia
"""

# ============================================================
# 06_model_performance_importance_SHAP.py
#
# Purpose:
#   Generate model performance, permutation importance, and SHAP
#   interpretation figures for:
#
#   1. BAU-relative models:
#        dSOC_pct
#        dN2O_pct
#
#   2. Added cover-crop models:
#        CC_extra_dSOC_pct
#        CC_extra_dN2O_pct
#
# Notes:
#   - Uses GroupKFold by site for cross-validated predictions.
#   - Uses permutation importance on original predictors.
#   - Uses SHAP on transformed predictors inside the RF pipeline.
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import gaussian_kde

# ------------------------------------------------------------
# Optional SHAP import
# ------------------------------------------------------------

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(
        "\nWARNING: shap is not installed. "
        "Install with: pip install shap"
    )

# ------------------------------------------------------------
# 1. User paths
# ------------------------------------------------------------

base_dir = Path(
    r"C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs"
)

model_dir = base_dir / "models"

bau_file = base_dir / "THP_BAU_relative_ML_dataset.csv"
cc_file = base_dir / "THP_cover_crop_added_effect_ML_dataset.csv"

output_dir = base_dir / "THP_ML_model_diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)


# BAU models
bau_soc_model_file = model_dir / "RF_BAU_relative_dSOC_pct_model.joblib"
bau_n2o_model_file = model_dir / "RF_BAU_relative_dN2O_pct_model.joblib"
bau_predictor_file = model_dir / "BAU_relative_predictor_list.joblib"

# CC-added models
cc_soc_model_file = model_dir / "RF_CC_added_dSOC_pct_model.joblib"
cc_n2o_model_file = model_dir / "RF_CC_added_dN2O_pct_model.joblib"
cc_predictor_file = model_dir / "CC_added_predictor_list.joblib"


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------
def pretty_feature_name(name):
    replacements = {
        "CO2A_mean": r"CO$_2$",
        "Tmax_mean": "Tmax",
        "Tmin_mean": "Tmin",
        "Srad_mean": "Solar rad.",
        "Prec_mean": "Precip.",
        "climate_period_Ref_baseline": "Baseline",
        "climate_period_RCP45_2030s": "RCP4.5–2030s",
        "climate_period_RCP45_2070s": "RCP4.5–2070s",
        "climate_period_RCP85_2030s": "RCP8.5–2030s",
        "climate_period_RCP85_2070s": "RCP8.5–2070s",
        "rotation_N0-L0": "N0-L0",
        "rotation_N0-L2": "N0-L2",
        "rotation_N1-L1": "N1-L1",
        "cover_crop_type_legume": "Legume CC",
        "cover_crop_type_legume_nonlegume": "Mixed CC",
        "soc_0_30": "SOC 0–30 cm",
        "soc_0_100": "SOC 0–100 cm",
        "silt_0_30": "Silt 0–30 cm",
        "silt_0_100": "Silt 0–100 cm",
        "clay_0_30": "Clay 0–30 cm",
        "sand_0_30": "Sand 0–30 cm",
        "paw_0_100": "PAW 0–100 cm",
        "paw_0_30": "PAW 0–30 cm",
    }
    return replacements.get(name, name)


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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_cv_predictions(model, df, predictors, target, group_col="site", n_splits=5):
    """
    Create GroupKFold cross-validated predictions.
    """
    model_df = df.dropna(subset=[target]).copy()

    X = model_df[predictors]
    y = model_df[target].values
    groups = model_df[group_col].values

    cv = GroupKFold(n_splits=n_splits)

    y_pred = cross_val_predict(
        clone(model),
        X,
        y,
        groups=groups,
        cv=cv,
        n_jobs=-1
    )

    out = model_df.copy()
    out[f"pred_{target}"] = y_pred

    metrics = {
        "target": target,
        "n": len(out),
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": rmse(y, y_pred)
    }

    return out, metrics


def make_observed_predicted_figure(results_dict, output_png, title, axis_limits=None):
    """
    Make 1 x 2 observed vs predicted figure using density-colored scatter.
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        constrained_layout=True
    )

    panel_labels = ["(A)", "(B)"]

    for ax, key, panel_label in zip(axes, results_dict.keys(), panel_labels):
        df_plot, obs_col, pred_col, metrics = results_dict[key]

        x = df_plot[obs_col].values
        y = df_plot[pred_col].values

        if axis_limits is not None and key in axis_limits:
            min_val, max_val = axis_limits[key]
        else:
            min_val = np.nanmin([x.min(), y.min()])
            max_val = np.nanmax([x.max(), y.max()])

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        sc = ax.scatter(
            x, y,
            c=z,
            s=16,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9
        )

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            linewidth=1.2
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        ax.set_xlabel("Observed simulated response (%)", fontsize=11)
        ax.set_ylabel("Predicted response (%)", fontsize=11)

        ax.set_title(
            f"{panel_label} {key}\n"
            f"R² = {metrics['r2']:.2f}, RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}",
            fontsize=12,
            fontweight="bold"
        )

        ax.grid(False)
        ax.set_aspect("equal")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Point density", fontsize=10)

    fig.suptitle(title, fontsize=15, y=1.04)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_permutation_importance(model, df, predictors, target,
                                output_csv, output_png, title,
                                group_col="site", n_repeats=10,
                                top_n=15):
    """
    Fit model on all data and calculate permutation importance.
    Importance is calculated on original predictors.
    """

    model_df = df.dropna(subset=[target]).copy()
    X = model_df[predictors]
    y = model_df[target].values

    fitted_model = clone(model)
    fitted_model.fit(X, y)

    result = permutation_importance(
        fitted_model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    imp = pd.DataFrame({
        "feature": predictors,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std
    })

    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(output_csv, index=False)

    plot_df = imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_sd"])
    ax.set_xlabel("Permutation importance\n(decrease in model performance)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return imp


def get_transformed_feature_names(pipeline):
    """
    Extract transformed feature names from ColumnTransformer inside pipeline.
    Assumes pipeline steps:
        preprocess
        model
    """
    preprocess = pipeline.named_steps["preprocess"]

    try:
        names = preprocess.get_feature_names_out()
    except Exception:
        names = []
        for name, transformer, cols in preprocess.transformers_:
            if name == "remainder":
                continue

            if hasattr(transformer, "get_feature_names_out"):
                try:
                    tmp_names = transformer.get_feature_names_out(cols)
                except Exception:
                    tmp_names = cols
            else:
                tmp_names = cols

            names.extend(tmp_names)

    names = [str(n).replace("num__", "").replace("cat__", "") for n in names]
    return np.array(names)


def simplify_shap_feature_name(name):
    """
    Collapse one-hot encoded categories back to main variable names.
    Example:
        rotation_N0-L2 -> rotation
        period_2030s   -> period
    """
    # exact known categorical variables
    categorical_roots = [
        "RCP",
        "period",
        "rotation",
        "cover_crop_type"
    ]

    for root in categorical_roots:
        if name.startswith(root + "_"):
            return root

    return name

def make_full_shap_figure(
        model, df, predictors, target,
        output_png, title,
        sample_size=1000, top_n=12):

    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import clone

    model_df = df.dropna(subset=[target]).copy()

    print("\nTarget check:", target)
    print(model_df[target].describe())

    if len(model_df) > sample_size:
        model_df = model_df.sample(sample_size, random_state=42)

    X = model_df[predictors]
    y = model_df[target].values

    fitted_model = clone(model)
    fitted_model.fit(X, y)

    preprocess = fitted_model.named_steps["preprocess"]
    rf_model = fitted_model.named_steps["model"]

    X_trans = preprocess.transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = get_transformed_feature_names(fitted_model)
    feature_names_pretty = np.array([pretty_feature_name(f) for f in feature_names])

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_trans)

    mean_abs = np.abs(shap_values).mean(axis=0)

    imp = pd.DataFrame({
        "feature": feature_names_pretty,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    top_features = imp.head(top_n)["feature"].values
    top_idx = [np.where(feature_names_pretty == f)[0][0] for f in top_features]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: bar
    bar_df = imp.head(top_n).iloc[::-1]
    axes[0].barh(bar_df["feature"], bar_df["mean_abs_shap"])
    axes[0].set_xlabel("SHAP value", fontsize=10)
    axes[0].set_title("(A) Mean absolute SHAP importance", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="both", labelsize=9)
    axes[0].grid(True, axis="x", alpha=0.25)

    # Panel B: beeswarm
    shap.summary_plot(
        shap_values[:, top_idx],
        X_trans[:, top_idx],
        feature_names=feature_names_pretty[top_idx],
        max_display=top_n,
        show=False,
        plot_size=None
    )

    # SHAP creates its own current axis, so adjust globally
    plt.gca().set_title("(B) SHAP beeswarm distribution", fontsize=12, fontweight="bold")
    plt.gca().tick_params(axis="both", labelsize=9)
    plt.xlabel("SHAP value", fontsize=10)

    plt.suptitle(title, fontsize=16, y=1.03)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# 3. Load data and models
# ------------------------------------------------------------

bau_df = clean_column_names(pd.read_csv(bau_file))
cc_df = clean_column_names(pd.read_csv(cc_file))

bau_soc_model = joblib.load(bau_soc_model_file)
bau_n2o_model = joblib.load(bau_n2o_model_file)
bau_predictors = joblib.load(bau_predictor_file)

cc_soc_model = joblib.load(cc_soc_model_file)
cc_n2o_model = joblib.load(cc_n2o_model_file)
cc_predictors = joblib.load(cc_predictor_file)

print("\nLoaded data and models.")
print("BAU data:", bau_df.shape)
print("CC-added data:", cc_df.shape)


# ------------------------------------------------------------
# 4. Cross-validated model performance
# ------------------------------------------------------------

bau_soc_cv, bau_soc_metrics = make_cv_predictions(
    bau_soc_model,
    bau_df,
    bau_predictors,
    target="dSOC_pct"
)

bau_n2o_cv, bau_n2o_metrics = make_cv_predictions(
    bau_n2o_model,
    bau_df,
    bau_predictors,
    target="dN2O_pct"
)

cc_soc_cv, cc_soc_metrics = make_cv_predictions(
    cc_soc_model,
    cc_df,
    cc_predictors,
    target="CC_extra_dSOC_pct"
)

cc_n2o_cv, cc_n2o_metrics = make_cv_predictions(
    cc_n2o_model,
    cc_df,
    cc_predictors,
    target="CC_extra_dN2O_pct"
)

# Save CV predictions
bau_soc_cv.to_csv(output_dir / "CV_predictions_BAU_SOC.csv", index=False)
bau_n2o_cv.to_csv(output_dir / "CV_predictions_BAU_N2O.csv", index=False)
cc_soc_cv.to_csv(output_dir / "CV_predictions_CC_added_SOC.csv", index=False)
cc_n2o_cv.to_csv(output_dir / "CV_predictions_CC_added_N2O.csv", index=False)

metrics_df = pd.DataFrame([
    bau_soc_metrics,
    bau_n2o_metrics,
    cc_soc_metrics,
    cc_n2o_metrics
])

metrics_df.to_csv(output_dir / "model_performance_metrics.csv", index=False)

print("\nModel performance metrics:")
print(metrics_df)


# ------------------------------------------------------------
# 5. Performance figures
# ------------------------------------------------------------

make_observed_predicted_figure(
    results_dict={
        "BAU-relative SOC": (
            bau_soc_cv,
            "dSOC_pct",
            "pred_dSOC_pct",
            bau_soc_metrics
        ),
        r"BAU-relative N$_2$O": (
            bau_n2o_cv,
            "dN2O_pct",
            "pred_dN2O_pct",
            bau_n2o_metrics
        )
    },
    output_png=output_dir / "Figure_10_model_performance_BAU.png",
    title="Cross-validated performance of BAU-relative ML models",
    axis_limits={
        "BAU-relative SOC": (-10, 30),
        r"BAU-relative N$_2$O": (-50, 30)
    }
)


make_observed_predicted_figure(
    results_dict={
        "Added SOC": (
            cc_soc_cv,
            "CC_extra_dSOC_pct",
            "pred_CC_extra_dSOC_pct",
            cc_soc_metrics
        ),
        r"Added N$_2$O": (
            cc_n2o_cv,
            "CC_extra_dN2O_pct",
            "pred_CC_extra_dN2O_pct",
            cc_n2o_metrics
        )
    },
    output_png=output_dir / "Figure_11_model_performance_CC_added.png",
    title="Cross-validated performance of added cover-crop ML models",
    axis_limits={
        "Added SOC": (-5, 15),
        r"Added N$_2$O": (-20, 40)
    }
)


# ------------------------------------------------------------
# 6. Permutation importance
# ------------------------------------------------------------

make_permutation_importance(
    model=bau_soc_model,
    df=bau_df,
    predictors=bau_predictors,
    target="dSOC_pct",
    output_csv=output_dir / "Permutation_importance_BAU_SOC.csv",
    output_png=output_dir / "Figure_12a_permutation_importance_BAU_SOC.png",
    title="Permutation importance: BAU-relative SOC"
)

make_permutation_importance(
    model=bau_n2o_model,
    df=bau_df,
    predictors=bau_predictors,
    target="dN2O_pct",
    output_csv=output_dir / "Permutation_importance_BAU_N2O.csv",
    output_png=output_dir / "Figure_12b_permutation_importance_BAU_N2O.png",
    title=r"Permutation importance: BAU-relative N$_2$O"
)

make_permutation_importance(
    model=cc_soc_model,
    df=cc_df,
    predictors=cc_predictors,
    target="CC_extra_dSOC_pct",
    output_csv=output_dir / "Permutation_importance_CC_added_SOC.csv",
    output_png=output_dir / "Figure_13a_permutation_importance_CC_added_SOC.png",
    title="Permutation importance: added SOC benefit"
)

make_permutation_importance(
    model=cc_n2o_model,
    df=cc_df,
    predictors=cc_predictors,
    target="CC_extra_dN2O_pct",
    output_csv=output_dir / "Permutation_importance_CC_added_N2O.csv",
    output_png=output_dir / "Figure_13b_permutation_importance_CC_added_N2O.png",
    title=r"Permutation importance: added N$_2$O effect"
)


# ------------------------------------------------------------
# 7. SHAP figures
# ------------------------------------------------------------

make_full_shap_figure(
    model=bau_soc_model,
    df=bau_df,
    predictors=bau_predictors,
    target="dSOC_pct",
    output_png=output_dir / "Figure_14_SHAP_BAU_SOC.png",
    title="SHAP importance: BAU-relative SOC"
)

make_full_shap_figure(
    model=bau_n2o_model,
    df=bau_df,
    predictors=bau_predictors,
    target="dN2O_pct",
    output_png=output_dir / "Figure_15_SHAP_BAU_N2O.png",
    title=r"SHAP importance: BAU-relative N$_2$O"
)

make_full_shap_figure(
    model=cc_soc_model,
    df=cc_df,
    predictors=cc_predictors,
    target="CC_extra_dSOC_pct",
    output_png=output_dir / "Figure_16_SHAP_CC_added_SOC.png",
    title="SHAP importance: added SOC benefit"
)

make_full_shap_figure(
    model=cc_n2o_model,
    df=cc_df,
    predictors=cc_predictors,
    target="CC_extra_dN2O_pct",
    output_png=output_dir / "Figure_17_SHAP_CC_added_N2O.png",
    title=r"SHAP importance: added N$_2$O effect"
)


print("\nFinished all model diagnostic and interpretation figures.")
print("Outputs saved to:")
print(output_dir)