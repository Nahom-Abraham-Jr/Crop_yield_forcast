import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Qiyas Crop Yield & Revenue Predictor")
st.write("Enter the plot details below. Weather and pricing data are fetched automatically behind the scenes!")

# Load model and lookup tables
@st.cache_resource
def load_resources():
    # Go up one dir from app/ to root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model = joblib.load(os.path.join(base_dir, 'models', 'final_model.joblib'))
    weather = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'regional_weather.csv'))
    prices = pd.read_csv(os.path.join(base_dir, 'data', 'raw', 'market_prices.csv'))
    
    # clean them the same way
    weather['region'] = weather['region'].astype(str).str.lower().str.strip()
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    weather['month'] = weather['month'].str.lower().str.slice(0,3).map(month_map)
    weather['month'] = pd.to_numeric(weather['month'], errors='coerce')
    
    prices['region'] = prices['region'].astype(str).str.lower().str.strip()
    prices['crop_type'] = prices['crop_type'].astype(str).str.lower().str.strip()
    prices.loc[prices['price_birr_per_quintal'] > 20000, 'price_birr_per_quintal'] /= 10
    
    return model, weather, prices

try:
    model, weather_df, prices_df = load_resources()
except Exception as e:
    st.error(f"Error loading model or data: {e}. Make sure you've run the notebook first!")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", ["Oromia", "Amhara", "SNNPR", "Tigray", "Somali"])
        crop_type = st.selectbox("Crop Type", ["Teff", "Wheat", "Maize", "Sorghum", "Barley"])
        survey_year = st.selectbox("Survey Year", [2021, 2022, 2023, 2024])
        planting_month = st.slider("Planting Month (1-12)", 1, 12, 5)
        altitude_m = st.number_input("Altitude (m)", value=1500)
        farm_size_ha = st.number_input("Farm Size (ha)", value=1.5)
        
    with col2:
        fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha)", value=50.0)
        improved_seed_used = st.selectbox("Improved Seed Used?", [0, 1])
        pest_disease_flag = st.selectbox("Pest/Disease Observed?", [0, 1])
        soil_quality_index = st.slider("Soil Quality Index (0-1)", 0.0, 1.0, 0.5)
        labor_days_per_ha = st.number_input("Labor Days per ha", value=30.0)
        distance_to_market_km = st.number_input("Distance to Market (km)", value=10.0)

    submitted = st.form_submit_button("Predict Yield & Revenue")

if submitted:
    r_lower = region.lower()
    c_lower = crop_type.lower()
    
    # 1. Look up Weather
    months = [(planting_month + i - 1) % 12 + 1 for i in range(4)]
    w = weather_df[(weather_df['region'] == r_lower) & (weather_df['year'] == survey_year) & (weather_df['month'].isin(months))]
    
    if len(w) == 0:
        season_temp = 20.0
        season_rain = 300.0
        season_heat = 0
        st.warning("⚠️ Could not find exact weather data for this region/year/season. Using regional averages.")
    else:
        season_temp = w['avg_temp_c'].mean()
        season_rain = w['monthly_rainfall_mm'].sum()
        season_heat = w['extreme_heat_days'].sum()

    # 2. Look up Price
    p = prices_df[(prices_df['region'] == r_lower) & (prices_df['crop_type'] == c_lower) & (prices_df['year'] == survey_year)]
    if len(p) == 0:
        price_val = 5000.0
        st.warning("⚠️ Could not find exact price data. Using standard 5000 birr/quintal fallback.")
    else:
        price_val = p['price_birr_per_quintal'].values[0]

    # 3. Create input dataframe for model
    input_data = pd.DataFrame([{
        'region': r_lower,
        'crop_type': c_lower,
        'survey_year': survey_year,
        'planting_month': planting_month,
        'altitude_m': altitude_m,
        'rainfall_mm_season': season_rain, # Fallback if model uses raw column
        'farm_size_ha': farm_size_ha,
        'fertilizer_kg_per_ha': fertilizer_kg_per_ha,
        'improved_seed_used': improved_seed_used,
        'pest_disease_flag': pest_disease_flag,
        'soil_quality_index': soil_quality_index,
        'labor_days_per_ha': labor_days_per_ha,
        'distance_to_market_km': distance_to_market_km,
        
        # Engineered features identical to training
        'season_temp_mean': season_temp,
        'season_rainfall_sum': season_rain,
        'season_extreme_heat': season_heat,
        'fertilizer_per_ha_ratio': fertilizer_kg_per_ha / (farm_size_ha + 1e-6),
        'labor_per_ha_ratio': labor_days_per_ha / (farm_size_ha + 1e-6),
        'is_late_planting': 1 if planting_month > 6 else 0,
        'interaction_dist_farm': distance_to_market_km * farm_size_ha
    }])
    
    # Predict
    pred_yield = model.predict(input_data)[0]
    # Revenue = yield_tons_per_ha * farm_size_ha * 10 (quintals/ton) * price
    revenue = pred_yield * farm_size_ha * 10 * price_val

    st.success(f"### Predicted Yield: **{pred_yield:.2f} tons/hectare**")
    st.info(f"### Estimated Revenue: **{revenue:,.2f} ETB**")
    
    st.write("---")
    st.write("🔍 **Behind the scenes lookups:**")
    st.write(f"- Looked up Market Price for {survey_year}: **{price_val:,.2f} birr/quintal**")
    st.write(f"- Looked up Growing Season Weather: **Avg {season_temp:.1f}°C** | **Total Rain {season_rain:.1f}mm**")
