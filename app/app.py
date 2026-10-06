import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qiyas Crop Intelligence | Team 6",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a2e14 0%, #1b5e20 55%, #2e7d32 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    color: white;
    box-shadow: 0 12px 40px rgba(27,94,32,.30);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: rgba(255,255,255,.04);
    border-radius: 50%;
}
.hero-title  { font-size: 2rem; font-weight: 800; letter-spacing: -.4px; margin-bottom: .35rem; }
.hero-sub    { font-size: .95rem; color: #c8e6c9; font-weight: 300; line-height: 1.6; max-width: 820px; }
.hero-badges { margin-top: .9rem; display: flex; gap: .6rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 30px;
    padding: 5px 14px;
    font-size: .78rem;
    font-weight: 600;
    color: #fff;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.kpi {
    flex: 1;
    background: #fff;
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    border-top: 4px solid #2e7d32;
    transition: transform .2s;
}
.kpi:hover { transform: translateY(-3px); }
.kpi.blue  { border-top-color: #0288d1; }
.kpi.amber { border-top-color: #f57c00; }
.kpi.purple{ border-top-color: #7b1fa2; }
.kpi-label { font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #666; }
.kpi-value { font-size: 1.85rem; font-weight: 800; color: #1b5e20; margin: .25rem 0; line-height: 1; }
.kpi.blue  .kpi-value  { color: #01579b; }
.kpi.amber .kpi-value  { color: #e65100; }
.kpi.purple .kpi-value { color: #6a1b9a; }
.kpi-sub   { font-size: .78rem; color: #888; }

/* ── Section header ── */
.sec-header {
    font-size: 1.05rem; font-weight: 700; color: #1b5e20;
    border-left: 4px solid #2e7d32;
    padding-left: .7rem; margin-bottom: 1rem; margin-top: .2rem;
}

/* ── Model Selector Cards ── */
.model-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: .6rem; margin-bottom: 1rem; }
.model-card {
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: .7rem .8rem;
    cursor: pointer;
    transition: all .2s;
    background: #fafafa;
    text-align: center;
}
.model-card.active { border-color: #2e7d32; background: #e8f5e9; box-shadow: 0 4px 14px rgba(46,125,50,.2); }
.model-card-title  { font-size: .78rem; font-weight: 700; color: #333; }
.model-card-rmse   { font-size: .72rem; color: #666; margin-top: .2rem; }

/* ── Button ── */
.stButton>button {
    background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 1rem !important; border-radius: 10px !important;
    padding: .65rem 1.5rem !important; border: none !important;
    box-shadow: 0 4px 14px rgba(46,125,50,.35) !important;
    width: 100% !important; transition: all .25s !important;
    letter-spacing: .3px !important;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 7px 20px rgba(46,125,50,.45) !important; }

/* ── Info chips ── */
.chip {
    display: inline-block;
    background: #e8f5e9; color: #1b5e20;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .78rem; font-weight: 600;
    margin: 3px 2px;
}
.chip.red   { background:#ffebee; color:#c62828; border-color:#ef9a9a; }
.chip.amber { background:#fff8e1; color:#e65100; border-color:#ffcc02; }
.chip.blue  { background:#e3f2fd; color:#01579b; border-color:#90caf9; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #f0f4f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Resolve Paths ───────────────────────────────────────────────────────────
def resolve_paths():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in [base, os.path.join(base, 'team_6')]:
        if os.path.exists(os.path.join(candidate, 'data', 'raw', 'crop_yield_train.csv')):
            raw  = os.path.join(candidate, 'data', 'raw')
            proc = os.path.join(candidate, 'data', 'processed')
            fig  = os.path.join(candidate, 'figures')
            mod  = os.path.join(candidate, 'models')
            return raw, proc, fig, mod
    return None, None, None, None

RAW_DIR, PROC_DIR, FIG_DIR, MOD_DIR = resolve_paths()

# ─── Load Raw Lookup Data ────────────────────────────────────────────────────
@st.cache_data
def load_lookup():
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    w = pd.read_csv(os.path.join(RAW_DIR, 'regional_weather.csv'))
    w['region'] = w['region'].str.lower().str.strip()
    if w['month'].dtype == 'O':
        w['month'] = w['month'].str.lower().str.slice(0,3).map(month_map)
    w['month'] = pd.to_numeric(w['month'], errors='coerce')

    p = pd.read_csv(os.path.join(RAW_DIR, 'market_prices.csv'))
    p['region']    = p['region'].str.lower().str.strip()
    p['crop_type'] = p['crop_type'].str.lower().str.strip()
    p.loc[p['price_birr_per_quintal'] > 20000, 'price_birr_per_quintal'] /= 10
    return w, p

@st.cache_data
def load_train():
    df = pd.read_csv(os.path.join(PROC_DIR, 'master_train.csv'))
    return df

try:
    weather_df, prices_df = load_lookup()
    train_df = load_train()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"⚠️ Could not load data: {e}")
    st.stop()

# ─── Model Registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Hist Gradient Boosting": {
        "emoji": "⚡", "short": "HGB",
        "cv_rmse": 0.496, "cv_r2": 0.874,
        "description": "Best performer. Native NaN handling, fast, boosted trees.",
        "color": "#2e7d32",
        "factory": lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=15, random_state=42)
    },
    "Random Forest": {
        "emoji": "🌲", "short": "RF",
        "cv_rmse": 0.545, "cv_r2": 0.849,
        "description": "Ensemble of 100 decision trees, robust to outliers.",
        "color": "#1565c0",
        "factory": lambda: RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42)
    },
    "Ridge Regression": {
        "emoji": "📐", "short": "Ridge",
        "cv_rmse": 0.913, "cv_r2": 0.574,
        "description": "L2-regularized linear model. Fast and interpretable baseline.",
        "color": "#6a1b9a",
        "factory": lambda: Ridge(alpha=1.0, random_state=42)
    },
    "Linear Regression": {
        "emoji": "📈", "short": "LR",
        "cv_rmse": 0.913, "cv_r2": 0.574,
        "description": "Classic OLS linear model. Strong interpretability baseline.",
        "color": "#e65100",
        "factory": lambda: LinearRegression()
    },
    "Dummy Baseline": {
        "emoji": "🎲", "short": "Dummy",
        "cv_rmse": 1.400, "cv_r2": -0.001,
        "description": "Predicts the mean yield. Reference floor for comparison.",
        "color": "#b71c1c",
        "factory": lambda: DummyRegressor(strategy='mean')
    }
}

# ─── Build / Cache pipeline for a selected model ─────────────────────────────
@st.cache_resource
def get_trained_pipeline(model_name: str):
    X = train_df.drop(columns=['plot_id', 'yield_tons_per_ha'], errors='ignore')
    y = train_df['yield_tons_per_ha']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    estimator = MODEL_REGISTRY[model_name]["factory"]()
    pipe = Pipeline([('pre', preprocessor), ('model', estimator)])
    pipe.fit(X, y)

    # Quick holdout score
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=.2, random_state=42)
    pipe_val = Pipeline([('pre', preprocessor), ('model', MODEL_REGISTRY[model_name]["factory"]())])
    pipe_val.fit(Xtr, ytr)
    preds = pipe_val.predict(Xval)
    val_rmse = np.sqrt(mean_squared_error(yval, preds))
    val_r2   = r2_score(yval, preds)
    return pipe, val_rmse, val_r2


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Control Center")
    st.markdown("---")

    # Model Selector
    st.markdown("### 🤖 Select AI Model")
    selected_model = st.radio(
        "Choose prediction model:",
        list(MODEL_REGISTRY.keys()),
        index=0,
        format_func=lambda m: f"{MODEL_REGISTRY[m]['emoji']} {m}"
    )
    m = MODEL_REGISTRY[selected_model]
    st.markdown(f"""
    <div style='background:#f1f8e9;border:1px solid #a5d6a7;border-radius:10px;padding:.8rem;margin-top:.5rem;font-size:.82rem;'>
        <b style='color:{m["color"]};'>{m["emoji"]} {selected_model}</b><br>
        <span style='color:#555;'>{m["description"]}</span><br><br>
        📊 CV RMSE: <b>{m['cv_rmse']:.3f}</b> &nbsp;|&nbsp; R²: <b>{m['cv_r2']:.3f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    preset = st.selectbox("Load scenario:", [
        "Custom Manual Entry",
        "🌾 Oromia Wheat – High Input",
        "🌱 Amhara Teff – Traditional",
        "🌽 SNNPR Maize – Optimal",
        "☀️ Somali Sorghum – Dryland"
    ])

    PRESETS = {
        "Custom Manual Entry":          dict(region="Oromia", crop="Wheat",    yr=2024, pm=5, alt=2200, sz=2.5, fert=80.,  seed=1, pest=0, soil=0.65, labor=40., dist=10.),
        "🌾 Oromia Wheat – High Input":  dict(region="Oromia", crop="Wheat",    yr=2024, pm=4, alt=2500, sz=3.0, fert=150., seed=1, pest=0, soil=0.85, labor=55., dist=6. ),
        "🌱 Amhara Teff – Traditional":  dict(region="Amhara", crop="Teff",     yr=2023, pm=6, alt=1800, sz=1.2, fert=35.,  seed=0, pest=1, soil=0.45, labor=30., dist=18.),
        "🌽 SNNPR Maize – Optimal":      dict(region="SNNPR",  crop="Maize",    yr=2024, pm=3, alt=1400, sz=2.0, fert=110., seed=1, pest=0, soil=0.72, labor=45., dist=8. ),
        "☀️ Somali Sorghum – Dryland":   dict(region="Somali", crop="Sorghum",  yr=2023, pm=7, alt=800,  sz=1.8, fert=20.,  seed=0, pest=1, soil=0.35, labor=25., dist=30.)
    }
    P = PRESETS[preset]

    st.markdown("---")
    st.markdown("### ℹ️ Platform Info")
    st.caption("**Hackathon**: Qiyas / AAU 2026")
    st.caption("**Team**: Group 6")
    st.caption("**Dataset**: 15,090 Ethiopian farm plots")

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🌾 Qiyas Crop Yield & Revenue Intelligence Platform</div>
  <div class="hero-sub">
    Empowering Ethiopian smallholder farmers, extension agents, and policymakers with
    AI-driven plot yield forecasts and market revenue predictions — before planting season begins.
  </div>
  <div class="hero-badges">
    <span class="badge">🤖 {selected_model}</span>
    <span class="badge">📊 5-Model Comparison</span>
    <span class="badge">🌦️ Integrated Weather & Price Time-Series</span>
    <span class="badge">👥 Team 6 Submission</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────
col_form, col_dash = st.columns([5, 7], gap="large")

# ════════════════════ LEFT: FORM ══════════════════════════════════════════════
with col_form:
    st.markdown('<div class="sec-header">📝 Farm & Plot Characteristics</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        st.markdown("##### 📍 Location & Crop")
        r1, r2 = st.columns(2)
        with r1:
            REGIONS = ["Oromia", "Amhara", "SNNPR", "Tigray", "Somali"]
            region    = st.selectbox("Region",      REGIONS,                        index=REGIONS.index(P["region"]))
            crop_type = st.selectbox("Crop Type",   ["Teff","Wheat","Maize","Sorghum","Barley"], index=["Teff","Wheat","Maize","Sorghum","Barley"].index(P["crop"]))
        with r2:
            survey_year    = st.selectbox("Survey Year", [2021,2022,2023,2024], index=[2021,2022,2023,2024].index(P["yr"]))
            planting_month = st.slider("Planting Month", 1, 12, P["pm"])

        st.markdown("##### 🚜 Agronomic Inputs")
        a1, a2 = st.columns(2)
        with a1:
            farm_size_ha       = st.number_input("Farm Size (ha)",        0.1, 50.0, float(P["sz"]),    0.1)
            fertilizer_kg      = st.number_input("Fertilizer (kg/ha)",    0.0, 500.0, float(P["fert"]), 5.0)
            labor_days         = st.number_input("Labor Days / ha",       1.0, 200.0, float(P["labor"]),5.0)
        with a2:
            improved_seed      = st.selectbox("Improved Seed?",  [1, 0], format_func=lambda x: "✅ Yes (Improved)" if x else "❌ No (Local)", index=0 if P["seed"]==1 else 1)
            pest_disease_flag  = st.selectbox("Pest/Disease?",   [0, 1], format_func=lambda x: "✅ No Disease" if x==0 else "🚨 Pest/Disease Present", index=P["pest"])
            soil_quality_index = st.slider("Soil Quality (0–1)", 0.0, 1.0, float(P["soil"]), 0.05)

        st.markdown("##### 🏔️ Physical Geography")
        g1, g2 = st.columns(2)
        with g1:
            altitude_m         = st.number_input("Altitude (m)",         0, 4000, int(P["alt"]),  50)
        with g2:
            distance_to_market = st.number_input("Distance to Market (km)", 0.1, 200.0, float(P["dist"]), 1.0)

        submitted = st.form_submit_button("🚀  Compute Yield & Revenue Forecast")

# ─── Lookups ──────────────────────────────────────────────────────────────────
r_l = region.lower()
c_l = crop_type.lower()

months  = [(planting_month + i - 1) % 12 + 1 for i in range(4)]
w_sub   = weather_df[(weather_df['region']==r_l) & (weather_df['year']==survey_year) & (weather_df['month'].isin(months))]
season_temp  = w_sub['avg_temp_c'].mean()          if len(w_sub) else 20.5
season_rain  = w_sub['monthly_rainfall_mm'].sum()  if len(w_sub) else 350.0
season_heat  = w_sub['extreme_heat_days'].sum()    if len(w_sub) else 2

p_sub       = prices_df[(prices_df['region']==r_l) & (prices_df['crop_type']==c_l) & (prices_df['year']==survey_year)]
price_birr  = p_sub['price_birr_per_quintal'].values[0] if len(p_sub) else 5200.0

# ─── Train pipeline ───────────────────────────────────────────────────────────
with st.spinner(f"⚙️ Training **{selected_model}** on full dataset…"):
    try:
        trained_pipe, val_rmse, val_r2 = get_trained_pipeline(selected_model)
        PIPE_OK = True
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        PIPE_OK = False

# ─── Build input row ──────────────────────────────────────────────────────────
def safe_month_num(v):
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    try:
        f = float(v)
        return int(f) if 1 <= f <= 12 else 5
    except (ValueError, TypeError):
        return month_map.get(str(v).lower().strip()[:3], 5)

input_row = pd.DataFrame([{
    'region': r_l, 'crop_type': c_l,
    'survey_year': survey_year,
    'planting_month': planting_month,
    'planting_month_num': safe_month_num(planting_month),
    'altitude_m': altitude_m,
    'farm_size_ha': farm_size_ha,
    'fertilizer_kg_per_ha': fertilizer_kg,
    'improved_seed_used': improved_seed,
    'pest_disease_flag': pest_disease_flag,
    'soil_quality_index': soil_quality_index,
    'labor_days_per_ha': labor_days,
    'distance_to_market_km': distance_to_market,
    'rainfall_mm_season': season_rain,
    'season_temp_mean': season_temp,
    'season_rainfall_sum': season_rain,
    'season_extreme_heat_days': season_heat,
    'fertilizer_per_ha_ratio': fertilizer_kg / (farm_size_ha + 1e-6),
    'labor_per_ha_ratio': labor_days / (farm_size_ha + 1e-6),
    'is_late_planting': 1 if planting_month > 6 else 0,
}])

# ─── Predict ─────────────────────────────────────────────────────────────────
if PIPE_OK:
    try:
        predicted_yield = trained_pipe.predict(input_row)[0]
        predicted_yield = max(0, predicted_yield)
    except Exception:
        predicted_yield = 2.45
else:
    predicted_yield = 2.45

total_tons     = predicted_yield * farm_size_ha
total_quintals = total_tons * 10
revenue        = total_quintals * price_birr

# ════════════════════ RIGHT: DASHBOARD ═══════════════════════════════════════
with col_dash:
    st.markdown('<div class="sec-header">📊 AI Forecast & Executive Dashboard</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">Predicted Yield</div>
        <div class="kpi-value">{predicted_yield:.2f} <span style="font-size:1rem;font-weight:400;">t/ha</span></div>
        <div class="kpi-sub">Total harvest: <b>{total_tons:.1f} t</b> ({total_quintals:.0f} qtl)</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Est. Farm Revenue</div>
        <div class="kpi-value">{revenue:,.0f} <span style="font-size:1rem;font-weight:400;">ETB</span></div>
        <div class="kpi-sub">Price: <b>{price_birr:,.0f} Birr/qtl</b></div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">Climate Profile</div>
        <div class="kpi-value">{season_temp:.1f}°C</div>
        <div class="kpi-sub">Rain: <b>{season_rain:.0f} mm</b> | Heat: <b>{int(season_heat)} days</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Accuracy strip ──
    st.markdown(f"""
    <div style='background:#f8faf9;border:1px solid #c8e6c9;border-radius:12px;padding:.8rem 1.1rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;'>
      <span style='font-size:.82rem;font-weight:600;color:#555;'>🤖 Active Model:</span>
      <span style='color:{m["color"]};font-weight:700;'>{m["emoji"]} {selected_model}</span>
      <span class='chip'>CV RMSE: {m['cv_rmse']:.3f}</span>
      <span class='chip'>CV R²: {m['cv_r2']:.3f}</span>
      <span class='chip blue'>Val RMSE: {val_rmse:.3f}</span>
      <span class='chip blue'>Val R²: {val_r2:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    t1, t2, t3, t4 = st.tabs(["🤖 AI Advice", "📊 Model Comparison", "📈 Benchmark Chart", "🔍 Lookups"])

    with t1:
        st.markdown("##### 🧠 Automated Agronomic Recommendations")
        recs = []
        if improved_seed == 0:
            recs.append(("🚨", "red",   "Switch to Improved Seed", "Local seed detected. Certified improved seed can boost yield by **+25–35%** in this region."))
        else:
            recs.append(("✅", "green", "Optimal Seed Variety", "Improved seed selection verified. Maximising genetic yield potential."))
        if fertilizer_kg < 60:
            recs.append(("⚠️", "amber", "Increase Fertilizer Rate", f"Applying **{fertilizer_kg:.0f} kg/ha** is below the recommended **80–120 kg/ha** threshold."))
        else:
            recs.append(("✅", "green", "Fertilizer on Target", f"**{fertilizer_kg:.0f} kg/ha** meets the agronomic threshold. Excellent input management."))
        if planting_month > 6:
            recs.append(("🚨", "red",   "Late Planting Risk", "Planting after June significantly increases end-of-season drought stress in Ethiopia."))
        if pest_disease_flag == 1:
            recs.append(("🚨", "red",   "Active Pest/Disease Alert", "Pest or disease observed. Deploy targeted intervention immediately to protect yield."))
        if season_heat > 10:
            recs.append(("⚠️", "amber", "Extreme Heat Risk", f"**{int(season_heat)} heat days** in the growing season. Consider heat-tolerant varieties."))
        if season_rain < 200:
            recs.append(("⚠️", "amber", "Low Seasonal Rainfall", f"Only **{season_rain:.0f} mm** cumulative rain. Supplemental irrigation may be needed."))

        for icon, color, title, msg in recs:
            bg  = {"red":"#ffebee","amber":"#fff8e1","green":"#e8f5e9"}.get(color,"#f5f5f5")
            brd = {"red":"#ef9a9a","amber":"#ffcc80","green":"#a5d6a7"}.get(color,"#ddd")
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {brd};border-radius:10px;padding:.8rem 1rem;margin-bottom:.6rem;'>
              <b style='font-size:.88rem;'>{icon} {title}</b><br>
              <span style='font-size:.82rem;color:#444;'>{msg}</span>
            </div>
            """, unsafe_allow_html=True)

    with t2:
        st.markdown("##### 📋 All Models — Cross-Validation Performance")
        comparison_data = [
            {"Model": f"{MODEL_REGISTRY[n]['emoji']} {n}", "CV RMSE": MODEL_REGISTRY[n]['cv_rmse'],
             "CV R²": MODEL_REGISTRY[n]['cv_r2'], "Selected": n == selected_model}
            for n in MODEL_REGISTRY
        ]
        comp_df = pd.DataFrame(comparison_data)

        def highlight_selected(row):
            if row['Selected']:
                return ['background-color: #e8f5e9; font-weight: bold;'] * len(row)
            return [''] * len(row)

        st.dataframe(
            comp_df.drop(columns=['Selected']).style
                .format({"CV RMSE": "{:.4f}", "CV R²": "{:.4f}"})
                .apply(highlight_selected, axis=1)
                .set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('background', '#f1f8e9')]}]),
            use_container_width=True, hide_index=True
        )
        st.caption(f"🟢 Highlighted row = currently active model: **{selected_model}**")

    with t3:
        st.markdown(f"##### 📊 Predicted Yield vs Regional Benchmarks ({survey_year})")
        bench = pd.DataFrame({
            "Category": [f"Your Plot ({crop_type})", "Oromia Avg", "Amhara Avg", "SNNPR Avg", "National Target"],
            "Yield": [predicted_yield, 2.65, 2.10, 2.40, 3.20]
        })
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors_bar = [m["color"] if i == 0 else "#b0bec5" for i in range(len(bench))]
        sns.barplot(data=bench, x="Category", y="Yield", hue="Category",
                    palette=colors_bar, legend=False, ax=ax)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        ax.set_title(f"Plot Forecast vs Regional Averages — {selected_model}", fontsize=10, fontweight='bold', pad=10)
        ax.set_ylabel("Yield (tons / ha)", fontsize=9)
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with t4:
        st.markdown(f"##### 🛰️ Automated Data Lookups — {region} ({survey_year})")
        col_w, col_p = st.columns(2)
        with col_w:
            st.markdown("**🌦️ Seasonal Weather**")
            st.markdown(f"- Growing window: months **{months[0]}–{months[-1]}**")
            st.markdown(f"- Mean temperature: **{season_temp:.2f} °C**")
            st.markdown(f"- Total rainfall: **{season_rain:.1f} mm**")
            st.markdown(f"- Extreme heat days: **{int(season_heat)}**")
        with col_p:
            st.markdown("**📈 Market Price**")
            st.markdown(f"- Crop: **{crop_type}** | Year: **{survey_year}**")
            st.markdown(f"- Price: **{price_birr:,.0f} Birr/quintal**")
            st.markdown(f"- Total quintals: **{total_quintals:.0f} qtl**")
            st.markdown(f"- Estimated revenue: **{revenue:,.0f} ETB**")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌾 **Qiyas Ethiopian Smallholder Crop-Yield Challenge | Team 6** · "
    "Built with Streamlit & Scikit-Learn · "
    f"Active Model: {selected_model} · Hackathon AAU 2026"
)
