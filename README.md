# Machine-Learning Surrogate Modeling of SOC and N2O Responses to Diversified Crop Rotations in the Texas High Plains

This repository contains the machine-learning workflow developed to emulate DSSAT-simulated soil organic carbon (SOC) and nitrous oxide (N2O) responses to diversified crop rotations and cover-crop systems across the Texas High Plains.

The workflow uses Random Forest surrogate models to predict two sets of responses:

1. **BAU-relative responses**: SOC and N2O changes from improved rotations relative to the business-as-usual rotation.
2. **Added cover-crop effects**: incremental SOC and N2O responses from cover-crop systems relative to the improved no-cover-crop rotation.

The modeling framework was designed to support spatial assessment of climate-smart rotation strategies by combining process-based crop model simulations, machine-learning emulation, SHAP-based interpretation, and spatial response-zone mapping.

## Study region

The analysis focuses on the Texas High Plains, a semiarid agricultural region where crop production is strongly influenced by water availability, climate variability, and soil constraints.

## Modeling workflow

The repository includes scripts for:

* preparing BAU-relative and added cover-crop datasets;
* extracting and processing climate, soil, and management predictors;
* training Random Forest surrogate models;
* evaluating model performance using cross-validation;
* interpreting model behavior using SHAP analysis;
* generating spatial prediction maps for SOC and SOC-N2O response zones.

## Target variables

The main target variables are:

* SOC response relative to BAU (%);
* N2O response relative to BAU (%);
* added SOC benefit from cover crops relative to N0-L0 (%);
* added N2O effect from cover crops relative to N0-L0 (%).

## Models

Random Forest regression models were developed for each target variable. The models were trained using soil, climate, atmospheric CO2, rotation, and cover-crop descriptors as predictors.

## Repository status

This repository is being prepared to accompany the manuscript:

**Machine-Learning Surrogate Modeling of SOC and N2O Responses to Diversified Crop Rotations in the Texas High Plains**

The repository will be updated as the manuscript progresses through submission and review.

## Citation

A full citation will be added after publication. Until then, please cite this repository as:

Attia, A. Machine-Learning Surrogate Modeling of SOC and N2O Responses to Diversified Crop Rotations in the Texas High Plains. GitHub repository.

## Contact

For questions about the workflow or data availability, please contact:

Ahmed Attia
Texas A&M AgriLife Research
ahmed.attia3@outlook.com
ahmedatia80@gmail.com
amattia81@gmail.com

