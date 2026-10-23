# %%
import os
import random
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor  # <-- regression (your targets are continuous)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Load dataset
# -----------------------
all_df = pd.read_csv(
    "/home/student/tsjohnson/Fire_Climate_Model/Manuscript_Codes/Final_manuscript_plots/MTBS_burn_area_severity_with_gridMET_window_min_max_monthly_standardized_1984-2024_Western_US_ecoregions_km2_select_regions.csv",
    parse_dates=["Ig_Date"]
)
all_df["year"] = all_df["Ig_Date"].dt.year

all_df["mod_and_high_percent"] = all_df["High_percent"]+all_df["Moderate_percent"]

print("Loaded rows:", len(all_df))
print("Loaded cols:", len(all_df.columns))
print("Example cols:", list(all_df.columns)[:25])

# -----------------------
# Modeling constants
# -----------------------
regions = ['4', '5', '6', '8','16', '77', '78', '85']
severities = ["Low", "Moderate", "High", "mod_and_high"]

# IMPORTANT: do NOT use "" as a saved label; it becomes blank in CSV and reads back as NaN
# We'll use a label for saving and map to target_col internally.
types = ["_percent"]  # percent or area

# Seasonal splits
seasonal_dfs = {
    "Summer_and_Fall": all_df[all_df['Ig_Date'].dt.month.isin([6, 7, 8, 9, 10, 11])].reset_index(drop=True)
}

# Feature sets
feature_sets = {
    "ALL_VARIABLES": [
        # ERC (max)
        "erc_1day_max","erc_3day_max","erc_7day_max","erc_14day_max","erc_30day_max",
        # FM100 (min)
        "fm100_1day_min","fm100_3day_min","fm100_7day_min","fm100_14day_min","fm100_30day_min",
        # PET (max)
        "pet_1day_max","pet_3day_max","pet_7day_max","pet_14day_max","pet_30day_max",
        # RHmin (min)
        "rmin_1day_min","rmin_3day_min","rmin_7day_min","rmin_14day_min","rmin_30day_min",
        # Tmax (max)
        "tmmx_1day_max","tmmx_3day_max","tmmx_7day_max","tmmx_14day_max","tmmx_30day_max",
        # VPD (max)
        "vpd_1day_max","vpd_3day_max","vpd_7day_max","vpd_14day_max","vpd_30day_max",
        # Wind (max)
        "vs_1day_max","vs_3day_max","vs_7day_max","vs_14day_max","vs_30day_max",
        # FWI (max)
        "FWI_1day_max","FWI_3day_max","FWI_7day_max","FWI_14day_max","FWI_30day_max",

        # Drought indices
        "spi14d","spi30d","spi90d",
        "spei14d","spei30d","spei90d",

        # Veg
        "SVI"
    ],
}

# -----------------------
# Outputs (collectors)
# -----------------------
test_r2_results = []
initial_rf_results_list = []     # impurity FI (final model)
predict_general_df = []          # train+test predictions (raw + transformed)
perm_importance_dfs = []         # permutation FI (final model)

# -----------------------
# Helper: subset search using CV on TRAIN ONLY (no test leakage)
# Returns (best_subset, best_cv_score)
# -----------------------
def find_best_subset_trainCV(X_train, y_train, features, base_cv_r2, best_params, seed=SEED):
    """
    Greedy backward search: try decreasing subset sizes; accept the first subset
    whose CV R^2 is within 1% of baseline. Uses 5-fold KFold CV on train only.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for k in range(len(features), 2, -1):
        for combo in combinations(features, k):
            rf = RandomForestRegressor(**best_params, random_state=seed, criterion='absolute_error', n_jobs=1)
            cv_scores = cross_val_score(rf, X_train[list(combo)], y_train, cv=kf, scoring="r2", n_jobs=1)
            cv_mean = float(np.mean(cv_scores))
            if cv_mean >= 0.99 * base_cv_r2:
                return list(combo), cv_mean
    return features, base_cv_r2

# -----------------------
# Grid for RF (deterministic)
# -----------------------
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10, 20, 30, 50],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 4, 6]
}

# ========================== MAIN LOOP ==========================
for season_name, season_df in seasonal_dfs.items():
    for feature_set_name, features in feature_sets.items():
        for region in regions:

            # Ensure region filter works even if US_L3CODE is numeric
            region_df = season_df[season_df['US_L3CODE'].astype(str) == region]

            for severity in severities:
                for sev_type in types:
                    is_area = (sev_type == "_area")
                    sev_type_label = "area" if is_area else "percent"
                    target_col = severity + ("_km2" if is_area else "_percent")

                    required_cols = features + [target_col, "year"]

                    # Skip if required columns not present
                    if not all(col in region_df.columns for col in required_cols):
                        continue

                    # Keep original row ids so you can trace points back to fires later
                    region_clean = region_df[required_cols].dropna().copy()
                    if len(region_clean) < 20:
                        print(f"⚠️ Skipping {region} | {season_name} | {severity}({sev_type_label}) | {feature_set_name} (n={len(region_clean)})")
                        continue

                    # --- Build X, y (raw)
                    X_full = region_clean[features]
                    y_full_raw = region_clean[target_col].copy()  # raw units
                    row_id_full = region_clean.index.to_numpy()   # original row ids
                    year_full = region_clean["year"].to_numpy()

                    # --- Transform only for (area) targets
                    if is_area:
                        y_full = np.log1p(y_full_raw)
                    else:
                        y_full = y_full_raw

                    # --- One split (freeze indices to reuse)
                    all_idx = np.arange(len(X_full))
                    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=SEED)

                    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
                    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

                    # also carry raw for plotting
                    y_train_raw = y_full_raw.iloc[train_idx]
                    y_test_raw  = y_full_raw.iloc[test_idx]

                    row_id_train = row_id_full[train_idx]
                    row_id_test  = row_id_full[test_idx]
                    year_train = year_full[train_idx]
                    year_test  = year_full[test_idx]

                    # --- Status print (progress tracking)
                    print(f"Running {season_name} | {feature_set_name} | region {region} | {severity}({sev_type_label})", flush=True)

                    # --- Grid search (CV on train only)
                    base_rf = RandomForestRegressor(random_state=SEED, criterion='absolute_error', n_jobs=1)
                    grid_search = GridSearchCV(
                        base_rf, param_grid, scoring='r2', refit=True, cv=3, n_jobs=1, verbose=2
                    )
                    try:
                        grid_search.fit(X_train, y_train)
                    except Exception as e:
                        print(f"GridSearch failed for {region} | {season_name} | {severity}({sev_type_label}) | {feature_set_name}: {e}")
                        continue
                    best_params = grid_search.best_params_

                    # --- Stage 1: rank features on TRAIN ONLY
                    full_rf = RandomForestRegressor(**best_params, random_state=SEED, criterion='absolute_error', n_jobs=1)
                    full_rf.fit(X_train, y_train)
                    imp_full = full_rf.feature_importances_
                    s_full = imp_full.sum()
                    if not ((imp_full >= 0).all() and 0.999 <= s_full <= 1.001):
                        print(f"❌ Bad FI (full) at {region} | {season_name} | {severity}({sev_type_label}) | {feature_set_name}: sum={s_full}, min={imp_full.min()}")
                        continue
                    full_importance = pd.Series(imp_full, index=X_train.columns).sort_values(ascending=False)

                    # --- Decide working feature set (apply top-75% ONLY for ALL_VARIABLES)
                    if feature_set_name == "ALL_VARIABLES":
                        q25 = full_importance.quantile(0.25)
                        working_features = full_importance[full_importance > q25].index.tolist()
                        if not working_features:
                            print(f"⚠️ No features passed threshold in {region} | {season_name} | {severity}({sev_type_label}) | {feature_set_name}")
                            continue
                    else:
                        working_features = list(X_train.columns)

                    # Slice TRAIN/TEST with frozen indices, same rows
                    Xtr = X_train[working_features]
                    Xte = X_test[working_features]

                    # --- Baseline CV on TRAIN ONLY (for subset search target)
                    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
                    baseline_rf = RandomForestRegressor(**best_params, random_state=SEED, criterion='absolute_error', n_jobs=1)
                    base_cv_scores = cross_val_score(baseline_rf, Xtr, y_train, cv=kf, scoring="r2", n_jobs=1)
                    base_cv_r2 = float(np.mean(base_cv_scores))

                    # --- Stage 2: choose smallest subset within 1% of baseline (TRAIN CV ONLY)
                    optimal_subset, optimal_cv_r2 = find_best_subset_trainCV(
                        Xtr, y_train, working_features, base_cv_r2, best_params, seed=SEED
                    )

                    # --- Final model: fit on ALL TRAIN (optimal subset), evaluate ONCE on TEST
                    final_rf = RandomForestRegressor(**best_params, random_state=SEED, criterion='absolute_error', n_jobs=1)
                    final_rf.fit(Xtr[optimal_subset], y_train)

                    y_pred_test = final_rf.predict(Xte[optimal_subset])
                    y_pred_train = final_rf.predict(Xtr[optimal_subset])

                    # --- Metrics ON TEST (first & only time test is used)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    test_rmse = float(np.sqrt(test_mse))

                    # Train metrics (helpful for diagnosing overfit)
                    train_r2 = r2_score(y_train, y_pred_train)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    train_rmse = float(np.sqrt(train_mse))

                    # Save run-level summary
                    test_r2_results.append({
                        "Season": season_name,
                        "Feature_Set": feature_set_name,
                        "Region": region,
                        "Burn_Severity": severity,
                        "Severity_Type": sev_type_label,
                        "n_total": int(len(X_full)),
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "Base_CV_R2_train": base_cv_r2,
                        "Optimal_CV_R2_train": optimal_cv_r2,
                        "Train_R2": train_r2,
                        "Test_R2": test_r2,
                        "Train_MAE": train_mae,
                        "Test_MAE": test_mae,
                        "Train_RMSE": train_rmse,
                        "Test_RMSE": test_rmse,
                        "Best_Params": json.dumps(best_params),
                        "Optimal_Subset": json.dumps(optimal_subset),
                    })

                    # --- Final impurity FI sanity
                    imp_final = final_rf.feature_importances_
                    s_final = imp_final.sum()
                    if not ((imp_final >= 0).all() and 0.999 <= s_final <= 1.001):
                        print(f"❌ Bad FI (final) at {region} | {season_name} | {severity}({sev_type_label}) | {feature_set_name}: sum={s_final}, min={imp_final.min()}")
                        continue

                    # Save final IMPURITY FI
                    final_importance = pd.Series(imp_final, index=optimal_subset).sort_values(ascending=True)
                    rf_result_df = pd.DataFrame({
                        "Season": season_name,
                        "Feature_Set": feature_set_name,
                        "Ecoregion": region,
                        "Burn Severity": severity,
                        "Severity Type": sev_type_label,
                        "Feature": final_importance.index,
                        "Feature Importance": final_importance.values,
                        "Train R2": train_r2,
                        "Test R2": test_r2,
                        "Train MAE": train_mae,
                        "Test MAE": test_mae,
                        "Train RMSE": train_rmse,
                        "Test RMSE": test_rmse,
                        "Base_CV_R2_train": base_cv_r2,
                        "Optimal_CV_R2_train": optimal_cv_r2,
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "Best Parameters": [json.dumps(best_params)] * len(final_importance),
                        "Optimal Subset": [json.dumps(optimal_subset)] * len(final_importance),
                    })
                    initial_rf_results_list.append(rf_result_df)

                    # Save predictions (TRAIN + TEST) with both transformed and raw units
                    # For percent: raw == transformed
                    # For area: y_true_raw is original area; y_true is log1p(area)
                    pred_train_df = pd.DataFrame({
                        "Season": season_name,
                        "Feature_Set": feature_set_name,
                        "Ecoregion": region,
                        "Burn Severity": severity,
                        "Severity Type": sev_type_label,
                        "Split": "train",
                        "Row_ID": row_id_train,
                        "year": year_train,
                        "Train R2": train_r2,
                        "Test R2": test_r2,
                        "y_true": y_train.values,
                        "y_pred": y_pred_train,
                        "y_true_raw": y_train_raw.values,
                        "y_pred_raw": (np.expm1(y_pred_train) if is_area else y_pred_train),
                    })

                    pred_test_df = pd.DataFrame({
                        "Season": season_name,
                        "Feature_Set": feature_set_name,
                        "Ecoregion": region,
                        "Burn Severity": severity,
                        "Severity Type": sev_type_label,
                        "Split": "test",
                        "Row_ID": row_id_test,
                        "year": year_test,
                        "Train R2": train_r2,
                        "Test R2": test_r2,
                        "y_true": y_test.values,
                        "y_pred": y_pred_test,
                        "y_true_raw": y_test_raw.values,
                        "y_pred_raw": (np.expm1(y_pred_test) if is_area else y_pred_test),
                    })

                    predict_general_df.append(pd.concat([pred_train_df, pred_test_df], ignore_index=True))

                    # Permutation importance (ΔR2) on TEST (no leakage; deterministic)
                    perm_results = permutation_importance(
                        final_rf, Xte[optimal_subset], y_test, n_repeats=10, random_state=SEED, n_jobs=1
                    )
                    perm_df = pd.DataFrame({
                        "Season": season_name,
                        "Feature_Set": feature_set_name,
                        "Ecoregion": region,
                        "Burn Severity": severity,
                        "Severity Type": sev_type_label,
                        "Feature": Xte[optimal_subset].columns,
                        "Perm_Importance_Mean_dR2": perm_results.importances_mean,
                        "Perm_Importance_STD_dR2": perm_results.importances_std,
                        "Selected": [True] * len(optimal_subset),
                        "Best_Params": [json.dumps(best_params)] * len(optimal_subset),
                        "Optimal_Subset": [json.dumps(optimal_subset)] * len(optimal_subset),
                        "n_test": int(len(X_test)),
                        "Test_R2": test_r2,
                    })
                    perm_importance_dfs.append(perm_df)

# -----------------------
# Save results
# -----------------------
out_dir = Path("/home/student/tsjohnson/Fire_Climate_Model/Manuscript_Codes/Final_manuscript_plots/FInal_RFR_Plots")
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(test_r2_results).to_csv(
    out_dir / "ModHighComb_withALL_20_max_min_included_rf_run_summary_only_all_variables_NEWREG_SUMMERFALL.csv",
    index=False
)

pd.concat(initial_rf_results_list, ignore_index=True).to_csv(
    out_dir / "ModHighComb_withALL_20_max_min_included_rf_variable_importance_summary_only_all_variables_NEWREG_SUMMERFALL.csv",
    index=False
)

pd.concat(predict_general_df, ignore_index=True).to_csv(
    out_dir / "ModHighComb_withALL_20_max_min_included_rf_predictions_train_test_only_all_variables_NEWREG_SUMMERFALL.csv",
    index=False
)

pd.concat(perm_importance_dfs, ignore_index=True).to_csv(
    out_dir / "ModHighComb_withALL_20_max_min_included_rf_permutation_importance_only_all_variables_NEWREG_SUMMERFALL.csv",
    index=False
)

print(f"\n✅ Done. Wrote outputs to: {out_dir}")