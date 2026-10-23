# Western U.S. Wildfire Burn Severity Analysis (1984–2024)

This repository contains Python scripts and supporting results for a manuscript examining spatial and temporal patterns in moderate- and high-severity wildfire across the western United States.
 
The study uses Monitoring Trends in Burn Severity (MTBS) wildfire data and gridMET hydroclimatic variables to investigate regional changes in burn severity, identify dominant predictors using Random Forest regression, and evaluate changes in hydroclimatic conditions between 1984–2003 and 2004–2024.

## Repository Structure

- **scripts/** — Python scripts used for Random Forest regression, statistical analyses, and figure generation.
- **data/** — Documentation and supporting datasets used in the analysis.
- **results/** — Random Forest model outputs, including performance statistics, predictions, feature importance, and permutation importance.

## Methods

Random Forest regression models were used to evaluate relationships between hydroclimatic variables and wildfire burn severity across western U.S. EPA Level III ecoregions. The analysis includes model optimization, feature selection, and permutation importance to identify influential predictors.

## Data Sources

- Monitoring Trends in Burn Severity (MTBS)
- gridMET hydroclimatic data
- U.S. Environmental Protection Agency Level III Ecoregions

## Reproducibility

Supporting datasets, software dependencies, and instructions for reproducing the analysis will be documented as the repository is finalized.
