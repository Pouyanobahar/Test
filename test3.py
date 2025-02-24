import streamlit as st
import joblib
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load trained models
model_blasting_p20 = joblib.load("C:/P.Nobahar/Thesis/Meta models/3M scenarios/Results/blasting/best_model_p20.pkl")
model_blasting_p50 = joblib.load("C:/P.Nobahar/Thesis/Meta models/3M scenarios/Results/blasting/best_model_p50.pkl")
model_blasting_p80 = joblib.load("C:/P.Nobahar/Thesis/Meta models/3M scenarios/Results/blasting/best_model_p80.pkl")

# Prediction function
def blasting_prediction(ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod):
    input_data = pd.DataFrame([[ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod]], 
                              columns=["UCS (MPa)", "Young's Modulus (GPa)", "Burden (m)", "Spacing (m)", 
                                       "Hole Diameter (mm)", "Explosives Density - gr/cm3", "VOD - (m/s)"])
    
    return (
        model_blasting_p20.predict(input_data)[0],
        model_blasting_p50.predict(input_data)[0],
        model_blasting_p80.predict(input_data)[0],
        input_data  # Return input data for SHAP analysis
    )

# Streamlit UI
st.set_page_config(page_title="Mining Prediction Dashboard", layout="wide")
st.title("⛏️ Blasting Prediction & SHAP Analysis Dashboard")

# Sidebar inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    spacing = st.slider("Spacing (m)", 5.0, 8.0, 6.0, step=0.5)
    burden = st.slider("Burden (m)", 5.0, 8.0, 6.0, step=0.5)
    ucs = st.slider("UCS (MPa)", 46.0, 60.0, 50.0, step=2.0)
    youngs_mod = st.slider("Young's Modulus (GPa)", 8.0, 12.0, 10.0, step=1.0)
    hole_diameter = st.slider("Hole Diameter (mm)", 180, 240, 200, step=10)
    explosive_density = st.slider("Explosives Density - gr/cm3", 0.8, 1.2, 1.0, step=0.1)
    vod = st.slider("VOD - (m/s)", 4000, 6000, 4500, step=500)

# Predict automatically when parameters change
P20, P50, P80, input_data = blasting_prediction(ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod)

# Create DataFrame for visualization
df = pd.DataFrame({"Size Fraction": ["P20", "P50", "P80"], "Value": [P20, P50, P80]})

# Create Altair Bar Chart
chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
    x=alt.X("Size Fraction", title=""),
    y=alt.Y("Value", title="Prediction Output"),
    color=alt.Color("Size Fraction", scale=alt.Scale(scheme="dark2"))
).properties(width=600, height=400)

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

# Column 1: Bar Chart
with col1:
    st.subheader("📊 Predicted Blasting Outputs")
    st.altair_chart(chart, use_container_width=True)
    st.write(f"🔹 **P20:** {P20:.2f}")
    st.write(f"🔹 **P50:** {P50:.2f}")
    st.write(f"🔹 **P80:** {P80:.2f}")

# Column 2: SHAP Analysis
with col2:
    st.subheader("📉 SHAP Explanation")

    # Select Model
    selected_model = st.selectbox("Choose Target for SHAP Analysis", ["P20", "P50", "P80"])
    if selected_model == "P20":
        model = model_blasting_p20
    elif selected_model == "P50":
        model = model_blasting_p50
    else:
        model = model_blasting_p80

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    # SHAP Force Plot
    st.write("### 🔬 SHAP Force Plot")
    fig, ax = plt.subplots(figsize=(8, 2))
    shap.waterfall_plot(shap.Explanation(values=shap_values.values[0], 
                                         base_values=shap_values.base_values[0], 
                                         data=input_data.iloc[0]))
    st.pyplot(fig)


