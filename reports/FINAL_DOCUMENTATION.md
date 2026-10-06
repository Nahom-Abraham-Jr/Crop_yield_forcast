# 🌾 Qiyas Ethiopian Smallholder Crop-Yield Challenge
## Final Technical Report & System Documentation

**Team Name**: Team 6  
**Competition**: Qiyas / AAU Ethiopian Crop Yield Forecasting Challenge 2026  
**Repository**: [Nahom-Abraham-Jr/Crop_yield_forcast](https://github.com/Nahom-Abraham-Jr/Crop_yield_forcast)  
**Live Vercel Application**: [https://crop-yield-forecast-team6.vercel.app](https://crop-yield-forecast-team6.vercel.app)  
**Live Streamlit Application**: [https://crop-yield-forecast-team6.streamlit.app](https://crop-yield-forecast-team6.streamlit.app)  
**Date**: October 2026  

---

## 👥 1. Team Roster & Registration Details

| No. | Member Full Name | Qiyas Registration ID | Technical Role & Contributions |
|:---:|:-----------------|:---------------------:|:-------------------------------|
| 1 | **Dawit Birhanu Mulu** | `qiyas-2026-004506` | Data Engineering, Pipeline Integration & Quality Auditing |
| 2 | **Fetene Erkutena Nibret** | `qiyas-2026-004836` | Model Architecture, 5-Fold Cross Validation & Hyperparameter Tuning |
| 3 | **Nahom Abraham Bekele** | `qiyas-2026-003344` | Feature Engineering, Temporal Window Alignment & Economic Modeling |
| 4 | **Haro Utura Kerro** | `qiyas-2026-007008` | Statistical Analysis, Figure Generation & Exploratory Data Analysis |
| 5 | **Kaleab Zerihun Teshome** | `qiyas-2026-003596` | Streamlit Web Application Development & UI/UX Experience |

---

## 🎯 2. Project Overview & Challenge Objective

Smallholder agriculture accounts for over 80% of crop production in Ethiopia and employs the majority of the rural workforce. However, yield prediction is traditionally post-harvest, leaving government authorities, emergency response bodies, and agricultural extension agents without early-warning systems against severe drought or crop failure.

The objective of this challenge is to predict `yield_tons_per_ha` for 3,751 unseen test farm plots based on survey characteristics, local weather time-series, and market economic indicators across 5 regions (*Oromia, Amhara, SNNPR, Tigray, Somali*) and 5 staple crops (*Teff, Wheat, Maize, Sorghum, Barley*).

---

## 🔄 3. Deliverable A: Data Cleaning & Integration Pipeline

### A1: Data Sanitization & Normalization
1. **Missing & Sentinel Values**: Sentinel values like `-999` were mapped to `np.nan`. Median imputation was applied to numeric features; mode imputation was applied to categorical features.
2. **Text Normalization**: All string fields (`region`, `crop_type`) were trimmed and lowercased to resolve casing mismatches (`'Oromia'` vs `'oromia '`).
3. **Month Parsing**: Handled both numeric months (`1-12`) and text strings (`'Aug'`, `'August'`) via a safe numeric converter.
4. **Market Price Outlier Cleaning**: Outliers exceeding 20,000 Birr/quintal caused by currency unit scaling were corrected.

### A2: Temporal & Spatial Data Integration
- **Left Table**: `plots_train.csv` (11,339 rows) / `plots_test.csv` (3,751 rows)
- **Growing Season Aggregation**: Weather data was aggregated over a 4-month window starting from `planting_month`:
  $$\text{season\_temp\_mean} = \frac{1}{4} \sum_{m=pm}^{pm+3} \text{temp\_mean}_m$$
  $$\text{season\_rainfall\_sum} = \sum_{m=pm}^{pm+3} \text{rainfall\_mm}_m$$
  $$\text{season\_extreme_heat\_days} = \sum_{m=pm}^{pm+3} \text{extreme\_heat\_days}_m$$
- **Market Price Join**: Merged on `(region, crop_type, survey_year)` to incorporate local Birr/quintal price signals.
- **Audit Result**: 100% match rate across all plots; row counts preserved exactly without data loss.

---

## 🔬 4. Deliverable B & C: Feature Engineering & Data Insights

### Key Engineered Ratios & Flags:
1. `fertilizer_per_ha_ratio`: $\frac{\text{fertilizer\_kg}}{\text{farm\_size\_ha} + 10^{-6}}$ — Captures input intensity per unit land.
2. `labor_per_ha_ratio`: $\frac{\text{labor\_days\_per\_ha}}{\text{farm\_size\_ha} + 10^{-6}}$ — Measures labor availability density.
3. `is_late_planting`: Binary indicator ($1$ if `planting_month` $> 6$, else $0$) capturing late-season planting risks.
4. `expected_revenue_birr_ha`: Estimated market gross revenue per hectare based on historical regional price indicators.

---

## 🤖 5. Deliverable D & E: Model Training, Benchmark & Evaluation

### Cross-Validation Strategy:
- **Protocol**: 5-Fold Cross-Validation (`random_state=42`)
- **Metrics**: Root Mean Squared Error (RMSE) and Coefficient of Determination ($R^2$)

### Model Benchmark Matrix:

| Model Rank | Algorithm Family | CV RMSE (tons/ha) | CV R² Score | Key Strengths & Remarks |
|:----------:|------------------|:-----------------:|:-----------:|:------------------------|
| 🥇 **1** | **Hist Gradient Boosting** | **0.4960** | **0.8740** | **Champion Model**. Native missing value handling, fast non-linear splits. |
| 🥈 **2** | **Random Forest Regressor** | **0.5450** | **0.8490** | Robust ensemble of 100 decision trees, highly resilient to feature noise. |
| 🥉 **3** | **Ridge Regression** | **0.9130** | **0.5740** | Regularized linear baseline. Interpretable but unable to model non-linear heat thresholds. |
| 4 | **Linear Regression (OLS)** | **0.9130** | **0.5740** | Standard OLS baseline; identical performance to Ridge under low dimensionality. |
| 5 | **Dummy Baseline** | **1.4000** | **-0.0010** | Mean predictor reference floor. |

### Hyperparameter Tuning (HistGradientBoosting):
- `max_iter`: 100
- `max_depth`: 15
- `min_samples_leaf`: 20
- `learning_rate`: 0.1

### Figure 12: Permutation Feature Importance Analysis
Permutation feature importance was calculated on out-of-fold validation sets. The top driving features influencing plot yields in Ethiopia are:
1. `soil_quality_index` (Weight: ~0.34) — Soil fertility is the single highest predictor of crop productivity.
2. `season_rainfall_sum` (Weight: ~0.26) — Cumulative growing season rainfall.
3. `fertilizer_per_ha_ratio` (Weight: ~0.18) — Commercial input application rate.
4. `altitude_m` (Weight: ~0.12) — Elevation and micro-climate temperature zoning.
5. `improved_seed` (Weight: ~0.08) — Seed genetics quality.

The figure has been rendered and saved to `team_6/figures/fig12_feature_importance.png`.

---

## 🖥️ 6. Streamlit Web Application (Qiyas Crop Intelligence Platform)

The interactive application in `team_6/app/app.py` provides a production-grade interface featuring:

1. **Interactive Control Center**:
   - Live Model Selector (allowing users to toggle between all 5 benchmarked models).
   - Quick Presets (e.g., *Oromia Wheat High Input*, *Amhara Teff Traditional*, *SNNPR Maize Optimal*, *Somali Sorghum Dryland*).
   - Full input controls for location, agronomics, and physical geography.

2. **Real-Time Forecasting Engine**:
   - Yield Prediction metric card (tons/ha) with confidence bounds.
   - Gross Revenue projection metric card (Ethiopian Birr / ha).
   - Regional Yield Category badge (*Exceptional*, *Above Average*, *Typical*, *Below Average*).

3. **Multi-Tab Analytics Suite**:
   - **Tab 1: Economic & Agronomic Analysis**: Breakdown of revenue drivers, cost-benefit estimates, and automated agronomic recommendations.
   - **Tab 2: All Models — CV Benchmark**: Comprehensive comparison table highlighting the active model in real-time.
   - **Tab 3: Benchmark & Regional Comparisons**: Visual comparison of predicted yield against Oromia, Amhara, SNNPR averages, and National targets.

---

## 📤 7. Submission Artifacts

1. **Predictions CSV**: `team_6/submission/team_6_submission.csv` (3,751 rows, columns: `plot_id`, `predicted_yield_tons_per_ha`)
2. **Serialized Model**: `team_6/models/final_model.joblib`
3. **Source Code & Notebooks**: All committed and pushed to GitHub repository [`Nahom-Abraham-Jr/Crop_yield_forcast`](https://github.com/Nahom-Abraham-Jr/Crop_yield_forcast).

---

*Report prepared and submitted by Team 6 for Qiyas Hackathon 2026.*
