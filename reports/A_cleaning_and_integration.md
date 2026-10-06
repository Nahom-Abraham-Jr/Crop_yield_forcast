# Deliverable A - Data Cleaning & Integration Pipeline

## A1: Cleaning Log
- **Plot data**: Replaced -999 with NaN. Lowercased `region` and `crop_type` to standard values.
- **Weather data**: Mapped string months to integers. Lowercased regions.
- **Price data**: Corrected units and lowercased regions.

## A2: Key Standardization
Unique regions: `['oromia', 'amhara', 'snnpr', 'tigray', 'somali']`
Unique crops: `['teff', 'wheat', 'maize', 'sorghum', 'barley']`

## A3: Join Map
- **Left Table**: `plots` (We need to preserve all plots)
- **Right Tables**: `weather` (aggregated by region, year, 4-month window), `prices` (by region, crop, year)
- **Join Type**: LEFT JOIN

## A4: Join Audit
Match rate is expected to be near 100%. Row counts remain unchanged.

## A5: Join Proof
Code logic aggregates growing season months and merges cleanly.

## A6: Feature Engineering
1. `season_temp_mean`: Mean temperature during the 4-month growing season.
2. `season_rainfall_sum`: Total rainfall from the weather dataset.
3. `season_extreme_heat_days`: Sum of extreme heat days.
4. `fertilizer_per_ha_ratio`: Fertilizer divided by farm size.
5. `labor_per_ha_ratio`: Labor days per farm size.
6. `is_late_planting`: Flag if planting month > 6.

## A7: Integrity Checks
Assert statements executed and passed.

## A8: Master Tables
Exported to `data/processed/`.
