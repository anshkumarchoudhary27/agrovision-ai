import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgriVision AI", page_icon="🌾", layout="wide")

@st.cache_resource
def load_files():
    model = joblib.load("yield_model.joblib")
    with open("metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_files()
except Exception:
    st.error("Model files missing. Upload yield_model.joblib and metadata.json beside app.py in GitHub.")
    st.stop()

st.markdown("""
<style>
.stApp{background:#f5f8f3}
[data-testid="stSidebar"]{background:#eaf2e8}
.hero{padding:28px;border-radius:16px;background:linear-gradient(110deg,#174b35,#39865a);color:white}
.hero h1{color:white}.hero p{color:#e4f1e6}
div[data-testid="stMetric"]{background:white;padding:14px;border-radius:12px;border:1px solid #e1e9df}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌱 AgriVision AI")
page = st.sidebar.radio("Choose a page", ["Home", "Yield Calculator", "ML Prediction", "About"])

if page == "Home":
    st.markdown('<div class="hero"><h1>Grow smarter. Harvest better.</h1><p>Crop yield performance calculator and Machine Learning prediction.</p></div>', unsafe_allow_html=True)
    st.subheader("Welcome to AgriVision AI")
    c1, c2 = st.columns(2)
    c1.info("🧮 **Yield Calculator**\n\nCalculate expected and actual crop production from land area and yield rate.")
    c2.success("🤖 **ML Prediction**\n\nEstimate yield using crop, country and historical climate inputs.")
    st.write("**Workflow:** Dataset → Preprocessing → Random Forest Regression → Evaluation → Prediction")
    st.warning("Academic prototype only. The ML dataset contains historical country-level records, not individual farm measurements.")

elif page == "Yield Calculator":
    st.markdown('<div class="hero"><h1>Crop Yield Performance Calculator</h1><p>Compare actual/planned production with your target.</p></div>', unsafe_allow_html=True)
    with st.form("calc"):
        a, b = st.columns(2)
        area = a.number_input("Land area (hectares)", min_value=0.01, value=2.0, step=0.5)
        crop = b.text_input("Crop name", "Wheat")
        expected = a.number_input("Expected yield (kg/hectare)", min_value=0.0, value=4000.0, step=100.0)
        actual = b.number_input("Actual/planned yield (kg/hectare)", min_value=0.0, value=3800.0, step=100.0)
        go = st.form_submit_button("Calculate", type="primary", use_container_width=True)
    if go:
        expected_t = area * expected / 1000
        actual_t = area * actual / 1000
        performance = actual / expected * 100 if expected > 0 else None
        st.subheader(f"Results — {crop}")
        x, y, z = st.columns(3)
        x.metric("Expected production", f"{expected_t:,.2f} tonnes")
        y.metric("Actual/planned production", f"{actual_t:,.2f} tonnes")
        z.metric("Yield performance", f"{performance:,.1f}%" if performance is not None else "N/A")
        if performance is not None:
            st.success("At or above target." if performance >= 100 else "Below the expected target.")
            st.write(f"Difference from target: **{actual_t - expected_t:+,.2f} tonnes**")
        else:
            st.info("Enter an expected yield greater than zero to calculate performance.")
        st.caption("This is a calculator, not an AI-generated estimate.")

elif page == "ML Prediction":
    st.markdown('<div class="hero"><h1>Machine Learning Yield Prediction</h1><p>Random Forest Regression trained on historical crop-yield data.</p></div>', unsafe_allow_html=True)
    st.warning("Dataset limitation: historical country-level observations from 1990–2013. This is not a farm-specific forecast or guaranteed harvest estimate.")
    ranges = meta["numeric_ranges"]
    with st.form("ml"):
        c1, c2 = st.columns(2)
        country = c1.selectbox("Country", meta["countries"], index=meta["countries"].index("India") if "India" in meta["countries"] else 0)
        crop = c1.selectbox("Crop", meta["crops"])
        year = c1.slider("Historical year", int(meta["year_min"]), int(meta["year_max"]), int(meta["year_max"]))
        rainfall = c2.number_input("Average annual rainfall (mm)", min_value=0.0, value=float(ranges["average_rain_fall_mm_per_year"]["median"]), step=10.0)
        temp = c2.number_input("Average temperature (°C)", min_value=-5.0, max_value=50.0, value=float(ranges["avg_temp"]["median"]), step=0.5)
        pesticides = c2.number_input("Pesticide use (tonnes; country-level dataset value)", min_value=0.0, value=float(ranges["pesticides_tonnes"]["median"]), step=10.0)
        area = st.number_input("Your land area (hectares)", min_value=0.01, value=2.0, step=0.5)
        go = st.form_submit_button("Predict crop yield", type="primary", use_container_width=True)
    if go:
        row = pd.DataFrame([{"Area":country,"Item":crop,"Year":int(year),
            "average_rain_fall_mm_per_year":float(rainfall),
            "pesticides_tonnes":float(pesticides),"avg_temp":float(temp)}])
        hg_ha = float(model.predict(row)[0])
        kg_ha = hg_ha / 10
        tonnes = kg_ha * area / 1000
        st.subheader("Prediction result")
        a,b,c = st.columns(3)
        a.metric("Predicted yield", f"{kg_ha:,.0f} kg/hectare")
        b.metric("Estimated total harvest", f"{tonnes:,.2f} tonnes")
        c.metric("Test R² score", f"{meta['metrics']['r2']:.3f}")
        st.caption(f"Test MAE: {meta['metrics']['mae']:,.0f} hg/ha · Test RMSE: {meta['metrics']['rmse']:,.0f} hg/ha")
        st.info("The result is a historical-pattern model estimate, not a guaranteed prediction.")

else:
    st.markdown('<div class="hero"><h1>About AgriVision AI</h1><p>An academic demonstration of machine learning for crop-yield estimation.</p></div>', unsafe_allow_html=True)
    st.write("**Technology:** Python, Pandas, Scikit-learn, Random Forest Regression, Joblib and Streamlit.")
    st.write("**Inputs:** Country, crop, year, annual rainfall, pesticide use and average temperature.")
    st.write("**Target:** Historical yield in hectograms per hectare (hg/ha), converted to kg/ha for display.")
    st.subheader("Model evaluation")
    a,b,c = st.columns(3)
    a.metric("R²", f"{meta['metrics']['r2']:.3f}")
    b.metric("MAE", f"{meta['metrics']['mae']:,.1f} hg/ha")
    c.metric("RMSE", f"{meta['metrics']['rmse']:,.1f} hg/ha")
    st.caption(f"Training records: {meta['rows']:,}. Dataset source: https://github.com/kdeeey/cropai")
    st.warning("The dataset is historical and country-level. It does not include local soil, irrigation, crop variety, farm management or current field conditions. Do not use this prototype as the sole basis for real farming decisions.")
