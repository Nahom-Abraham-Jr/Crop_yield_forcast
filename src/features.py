import pandas as pd
import numpy as np

def engineer_features(df):
    """Engineer features."""
    df = df.copy()
    # 1. Season mean temp (from joined weather)
    if 'season_temp_mean' not in df.columns and 'avg_temp_c' in df.columns:
        df['season_temp_mean'] = df['avg_temp_c'] # Placeholder for aggregated
    # 2. Extreme heat days season total
    if 'season_extreme_heat' not in df.columns and 'extreme_heat_days' in df.columns:
        df['season_extreme_heat'] = df['extreme_heat_days'] 
    # 3. Ratio: Fertilizer per Farm Size
    df['fertilizer_per_ha_ratio'] = df.get('fertilizer_kg_per_ha', 0) / (df.get('farm_size_ha', 1) + 1e-6)
    # 4. From planting month: is_late_planting
    df['is_late_planting'] = df.get('planting_month', 1).apply(lambda x: 1 if x > 6 else 0)
    # 5. Distance to market * farm size interaction
    df['distance_size_interaction'] = df.get('distance_to_market_km', 0) * df.get('farm_size_ha', 0)
    # 6. Season rainfall
    df['season_rainfall'] = df.get('monthly_rainfall_mm', df.get('rainfall_mm_season', 0))
    return df
