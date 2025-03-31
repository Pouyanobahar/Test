import streamlit as st
import joblib
import pandas as pd
import altair as alt
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ------------------------------
# Set up page config and header banner
# ------------------------------
st.set_page_config(
    page_title="Mining Process Overview", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Top header with logo and title (update URL to your logo as needed)
def render_header():
    logo_path = "assets/genesis-mining-logo-vector.png"
    logo_base64 = get_base64_of_bin_file(logo_path)
    header_html = f"""
    <div style="display: flex; align-items: center; margin-top: 0px; padding: 0 0 10px 0;">
        <img src="data:image/png;base64,{logo_base64}" style="height: 100px; margin-right: 20px;">
        <h1 style="color: #2D3B44; margin: 0;">Mining Process Optimization Dashboard</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# ------------------------------
# Set up base directory and load models (using relative paths)
# ------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
models = {}

# Blasting models
models["blasting_p20"] = joblib.load(os.path.join(base_dir, "best_model_p20.pkl"))
models["blasting_p50"] = joblib.load(os.path.join(base_dir, "best_model_p50.pkl"))
models["blasting_p80"] = joblib.load(os.path.join(base_dir, "best_model_p80.pkl"))

# Screening models
models["OnScreen_p20"] = joblib.load(os.path.join(base_dir, "best_model_OS_p20.pkl"))
models["OnScreen_p50"] = joblib.load(os.path.join(base_dir, "best_model_OS_p50.pkl"))
models["OnScreen_p80"] = joblib.load(os.path.join(base_dir, "best_model_OS_p80.pkl"))
models["OnScreen_Massflow"] = joblib.load(os.path.join(base_dir, "best_model_OS_Mass.pkl"))

models["UnderScreen_p20"] = joblib.load(os.path.join(base_dir, "best_model_US_p20.pkl"))
models["UnderScreen_p50"] = joblib.load(os.path.join(base_dir, "best_model_US_p50.pkl"))
models["UnderScreen_p80"] = joblib.load(os.path.join(base_dir, "best_model_US_p80.pkl"))
models["UnderScreen_Massflow"] = joblib.load(os.path.join(base_dir, "best_model_US_Mass.pkl"))

# Crushing models
models["Crusher_p20"] = joblib.load(os.path.join(base_dir, "best_model_Crusher_p20.pkl"))
models["Crusher_p50"] = joblib.load(os.path.join(base_dir, "best_model_Crusher_p50.pkl"))
models["Crusher_p80"] = joblib.load(os.path.join(base_dir, "best_model_Crusher_p80.pkl"))
models["Crusher_Mass"] = joblib.load(os.path.join(base_dir, "best_model_Crusher_Mass.pkl"))

# ------------------------------
# Prediction Functions
# ------------------------------
def blasting_prediction(ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod):
    input_data = pd.DataFrame([[ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod]], 
                              columns=["UCS (MPa)", "Young's Modulus (GPa)", "Burden (m)", "Spacing (m)", 
                                       "Hole Diameter (mm)", "Explosives Density - gr/cm3", "VOD - (m/s)"])
    
    model_p20 = models.get("blasting_p20")
    model_p50 = models.get("blasting_p50")
    model_p80 = models.get("blasting_p80")
    
    if model_p20 is None or model_p50 is None or model_p80 is None:
        st.error("One or more blasting models failed to load. Please verify the file paths and that the files exist.")
        st.stop()
    p20 = model_p20.predict(input_data)[0]
    p50 = model_p50.predict(input_data)[0]
    p80 = model_p80.predict(input_data)[0]
    
    st.session_state.process_data['blasting'] = {'p20': p20, 'p50': p50, 'p80': p80}
    return p20, p50, p80, input_data

def screening_prediction(alpha, d50, css, p20_blast, p50_blast, p80_blast):
    input_data = pd.DataFrame([[alpha, d50, css, p20_blast, p50_blast, p80_blast]], 
                              columns=['Alpha', "D50, d50c", "CSS (mm)", 
                                       'WB Blasting Product - Solids - P20', 
                                       'WB Blasting Product - Solids - P50', 
                                       'WB Blasting Product - Solids - P80'])
    
    p20_os = models["OnScreen_p20"].predict(input_data)[0]
    p50_os = models["OnScreen_p50"].predict(input_data)[0]
    p80_os = models["OnScreen_p80"].predict(input_data)[0]
    mass_os = models["OnScreen_Massflow"].predict(input_data)[0]
    
    p20_us = models["UnderScreen_p20"].predict(input_data)[0]
    p50_us = models["UnderScreen_p50"].predict(input_data)[0]
    p80_us = models["UnderScreen_p80"].predict(input_data)[0]
    mass_us = models["UnderScreen_Massflow"].predict(input_data)[0]
    
    st.session_state.process_data['screening'] = {
        'onscreen': {'p20': p20_os, 'p50': p50_os, 'p80': p80_os, 'mass': mass_os},
        'underscreen': {'p20': p20_us, 'p50': p50_us, 'p80': p80_us, 'mass': mass_us}
    }
    
    return p20_os, p50_os, p80_os, mass_os, p20_us, p50_us, p80_us, mass_us, input_data

def crushing_prediction(css, p20_os, p50_os, p80_os, mass_os):
    p20_os = st.session_state.process_data['screening']['onscreen']['p20']
    p50_os = st.session_state.process_data['screening']['onscreen']['p50']
    p80_os = st.session_state.process_data['screening']['onscreen']['p80']
    mass_os = st.session_state.process_data['screening']['onscreen']['mass']
    
    input_data = pd.DataFrame([[css, p20_os, p50_os, p80_os, mass_os]], 
                              columns=["CSS (mm)", "Single Deck Screen OS - Solids - P20",
                                       "Single Deck Screen OS - Solids - P50",
                                       "Single Deck Screen OS - Solids - P80",
                                       "Single Deck Screen OS - Solids - Mass Flow"])
    
    p20_crush = models["Crusher_p20"].predict(input_data)[0]
    p50_crush = models["Crusher_p50"].predict(input_data)[0]
    p80_crush = models["Crusher_p80"].predict(input_data)[0]
    mass_crush = models["Crusher_Mass"].predict(input_data)[0]
    
    power = 150 + (0.5 * mass_os) + (100 / 2) - (css * 2)
    st.session_state.process_data['crushing'] = {
        'p20': p20_crush, 
        'p50': p50_crush, 
        'p80': p80_crush, 
        'mass': mass_crush, 
        'power': power
    }
    return p20_crush, p50_crush, p80_crush, mass_crush, power, input_data

# ------------------------------
# Visualization Helper for Sankey Diagram
# ------------------------------
def create_sankey_diagram(total_mass, os_mass, us_mass):
    # Unified color palette for nodes/links
    node_colors = ["#1f78b4", "#33a02c", "#b2df8a", "#fb9a99", "#a6cee3", "#e31a1c"]
    link_colors = [
        "rgba(31, 120, 180, 0.4)",
        "rgba(51, 160, 44, 0.4)",
        "rgba(166, 206, 227, 0.4)",
        "rgba(251, 154, 153, 0.4)",
        "rgba(227, 26, 28, 0.4)"
    ]
    fig = go.Figure(data=[go.Sankey(
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        node=dict(
            pad=7.5,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=["Blasting", "Screening", "On-Screen", "Crushing", "Under-Screen", "Final Product"],
            color=node_colors
        ),
        link=dict(
            source=[0, 1, 1, 2, 4],
            target=[1, 2, 4, 3, 5],
            value=[total_mass, os_mass, us_mass, os_mass, us_mass],
            color=link_colors
        )
    )])
    fig.update_layout(
        title_text="Material Flow Process",
        font_size=15,
        width=1200,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ------------------------------
# Custom CSS for Boxes & Hiding Streamlit Branding (optional)
# ------------------------------
custom_css = """
<style>
.box {
    background-color: #2D3B44;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    color: #FFFFFF;
}
.box h2, .box h3, .box p {
    color: #FFFFFF;
    margin: 0.5rem 0;
}
/* Optionally hide Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------
# Sidebar - Navigation & Global Parameters
# ------------------------------
current_page = st.sidebar.radio(
    "Menu",
    options=["👁️ Overview", "💥 Blasting", "🥅 Screening", "⚙️ Crusher", "📍 Location", "🎯 Optimization"]
)

# Global target & economic parameters with tooltips
Blasting_target_p80 = st.sidebar.slider("Blasting Target P80 (mm)", 500.0, 700.0, 600.0, key="blasting_target_p80", help="Desired fragmentation size for blasting")
Screening_target_p80 = st.sidebar.slider("Screening Target P80 (mm)", 10.0, 200.0, 60.0, key="screening_target_p80", help="Desired fragmentation size for screening")
Crushing_target_p80 = st.sidebar.slider("Crushing Target P80 (mm)", 10.0, 200.0, 80.0, key="crusher_target_p80", help="Desired fragmentation size for crushing")
cost_per_ton = st.sidebar.number_input("Processing Cost ($/ton)", 1.0, 10.0, 2.5, step=0.1, key="cost_per_ton")
energy_cost = st.sidebar.number_input("Energy Cost ($/kWh)", 0.05, 0.5, 0.12, step=0.01, key="energy_cost")

# ------------------------------
# Initialize Session State for Process Data
# ------------------------------
if "process_data" not in st.session_state:
    st.session_state["process_data"] = {
        "blasting": {"p20": 0, "p50": 0, "p80": 0},
        "screening": {
            "onscreen": {"p20": 0, "p50": 0, "p80": 0, "mass": 0},
            "underscreen": {"p20": 0, "p50": 0, "p80": 0, "mass": 0},
        },
        "crushing": {"p20": 0, "p50": 0, "p80": 0, "power": 0},
    }

# ------------------------------
# Main Page Display (Overview)
# ------------------------------
if current_page == "👁️ Overview":
    st.title("Mining Process Overview")
    st.write("Use this dashboard to analyze and optimize mining processes from blasting to crushing.")
    
    # Display key metrics in four equally sized columns
    col1, col2, col3, col4 = st.columns(4)
    blasting_p80 = st.session_state.process_data['blasting']['p80']
    screening_underscreen_p80 = st.session_state.process_data['screening']['underscreen']['p80']
    crushing_p80 = st.session_state.process_data['crushing']['p80']
    
    # Calculate deltas if values exist
    blasting_delta = None if blasting_p80 == 0 else f"{Blasting_target_p80 - blasting_p80:.1f} mm"
    screening_delta = None if screening_underscreen_p80 == 0 else f"{Screening_target_p80 - screening_underscreen_p80:.1f} mm"
    crushing_delta = None if crushing_p80 == 0 else f"{Crushing_target_p80 - crushing_p80:.1f} mm"
    
    with col1:
        st.markdown(f"""
        <div class="box">
            <h3>Blasting P80</h3>
            <p style="font-size: 2rem; font-weight: bold;">{blasting_p80:.1f} mm</p>
            <p style="font-size: 1.25rem;">Delta: {blasting_delta if blasting_delta is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="box">
            <h3>Screening P80</h3>
            <p style="font-size: 2rem; font-weight: bold;">{screening_underscreen_p80:.1f} mm</p>
            <p style="font-size: 1.25rem;">Delta: {screening_delta if screening_delta is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="box">
            <h3>Crushing P80</h3>
            <p style="font-size: 2rem; font-weight: bold;">{crushing_p80:.1f} mm</p>
            <p style="font-size: 1.25rem;">Delta: {crushing_delta if crushing_delta is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="box">
            <h3>Stockpile P80</h3>
            <p style="font-size: 2rem; font-weight: bold;">{crushing_p80/2:.1f} mm</p>
            <p style="font-size: 1.25rem;">Delta: {crushing_delta if crushing_delta is not None else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Process Flow Visualization")
        os_mass = st.session_state.process_data['screening']['onscreen']['mass'] or 60
        total_mass = 450
        us_mass = total_mass - os_mass
        sankey_fig = create_sankey_diagram(total_mass, os_mass, us_mass)
        st.plotly_chart(sankey_fig, use_container_width=True)
    
    with col2:
        st.subheader("Energy Consumption")
        energy_data = {
            "Blasting": 250,
            "Screening": 150,
            "Crushing": 200,
            "Milling": 300,
            "Transportation": 200
        }
        pie_colors = ['#1f78b4', '#33a02c', '#b2df8a', '#fb9a99', '#a6cee3']
        fig_energy = go.Figure(data=[go.Pie(
            labels=list(energy_data.keys()),
            values=list(energy_data.values()),
            hole=0.4,
            marker=dict(colors=pie_colors)
        )])
        fig_energy.update_layout(
            title_text="Energy Consumption",
            annotations=[dict(text='kWh', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    with col3:
        st.subheader("Cost Distribution")
        cost_data = {
            "Blasting": 2.5,
            "Screening": 1.8,
            "Crushing": 4.0,
            "Milling": 3.0,
            "Transportation": 2.2
        }
        fig_cost = go.Figure(data=[go.Pie(
            labels=list(cost_data.keys()),
            values=list(cost_data.values()),
            hole=0.4,
            marker=dict(colors=pie_colors)
        )])
        fig_cost.update_layout(
            title_text="Cost ($/ton)",
            annotations=[dict(text='$/ton', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col4:
        st.subheader("Project Location")
        latitude, longitude = -34.9285, 138.6007
        m = folium.Map(location=[latitude, longitude], zoom_start=3)
        folium.Marker(
            [latitude, longitude],
            popup="Project Location",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        st_folium(m, width=500, height=400)

    
    st.markdown("---")
    st.subheader("Process Optimization Recommendations")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.markdown("""#### Blasting
- Decrease burden by 0.5m  
- Increase explosive density for hard rock""")
    with rec_col2:
        st.markdown("""#### Screening
- Adjust screen angle  
- Increase washing water by 15%""")
    with rec_col3:
        st.markdown("""#### Crushing
- Decrease CSS by 5mm  
- Adjust eccentric speed to reduce power consumption""")
    
    st.markdown("---")
    st.subheader("KPIs")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    final_p80 = st.session_state.process_data['crushing']['p80']
    total_power = st.session_state.process_data['crushing']['power']
    feed_mass = st.session_state.process_data['screening']['onscreen']['mass'] + st.session_state.process_data['screening']['underscreen']['mass']
    energy_efficiency = total_power / feed_mass if feed_mass > 0 else 0
    product_quality = 100 - abs(final_p80 - Blasting_target_p80) / Blasting_target_p80 * 100 if Blasting_target_p80 > 0 else 0
    operating_cost = cost_per_ton + (energy_efficiency * energy_cost)
    
    # Optionally, you can include a gauge indicator for one key metric:
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = blasting_p80,
        title = {"text": "Blasting P80"},
        gauge = {"axis": {"range": [None, 700]},
                 "bar": {"color": "#1f78b4"}}
    ))
    with kpi1:
        st.plotly_chart(gauge_fig, use_container_width=True)
    with kpi2:
        st.metric("Energy Efficiency", f"{energy_efficiency:.1f} kW/h")
    with kpi3:
        st.metric("Product Quality", f"{product_quality:.1f} %")
    with kpi4:
        st.metric("Operating Cost", f"${product_quality:.1f}/ton")

    st.markdown("---")
    st.subheader("Report Generation")
    report_col1, report_col2 = st.columns([3, 1])
    with report_col1:
        report_options = st.multiselect(
            "Select sections to include in report:",
            [
                "Blasting Analysis", 
                "Screening Analysis", 
                "Crushing Analysis", 
                "Process Overview", 
                "Optimization Recommendations"
            ],
            ["Process Overview", "Optimization Recommendations"],
            key="report_options"
        )
    with report_col2:
        st.download_button(
            label="Generate Report",
            data="Sample Report Content - In a real app, this would be a PDF or CSV export",
            file_name="mining_process_report.csv",
            mime="text/csv",
            key="download_report"
        )

# ------------------------------
# Blasting Page (with Wide Left Column for Analysis)
# ------------------------------
elif current_page == "💥 Blasting":
    st.markdown("<h1 style='font-weight:bold;'>Blasting Analysis</h1>", unsafe_allow_html=True)
    blast_col1, blast_col2 = st.columns(2)
 
    with blast_col1:
        st.markdown('<div class="my-box">', unsafe_allow_html=True)
        with st.container():
            st.subheader("Input Parameters")
            with st.expander("Rock Properties", expanded=True):
                ucs = st.slider("UCS (MPa)", 46.0, 60.0, 50.0, step=2.0, key="ucs_blasting")
                youngs_mod = st.slider("Young's Modulus (GPa)", 8.0, 12.0, 10.0, step=1.0, key="youngs_mod_blasting")
            with st.expander("Blast Design", expanded=True):
                burden = st.slider("Burden (m)", 5.0, 8.0, 6.0, step=0.5, key="burden_blasting")
                spacing = st.slider("Spacing (m)", 5.0, 8.0, 6.0, step=0.5, key="spacing_blasting")
                hole_diameter = st.slider("Hole Diameter (mm)", 180, 240, 200, step=10, key="hole_diameter_blasting")
            with st.expander("Explosive Properties", expanded=True):
                explosive_density = st.slider("Explosives Density (g/cm³)", 0.8, 1.2, 1.0, step=0.1, key="explosive_density_blasting")
                vod = st.slider("VOD (m/s)", 4000, 6000, 4500, step=500, key="vod_blasting")
            powder_factor = (explosive_density * 1000 * (((hole_diameter/1000)**2)/4) * 3.1415) / (burden * spacing)
            st.info(f"Calculated Powder Factor: {powder_factor:.2f} kg/m³")
        st.markdown('</div>', unsafe_allow_html=True)
 
    with blast_col2:
        st.markdown('<div class="my-box">', unsafe_allow_html=True)
        # Prediction results
        P20, P50, P80, input_data = blasting_prediction(
            ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod
        )
        if P20 is None or P50 is None or P80 is None:
            st.error("Model predictions are unavailable. Please check that all models are loaded correctly.")
            st.stop()
        sizes = [0, P20, P50, P80, P80*1.1, P80*1.25]
        percentages = [0, 20, 50, 80, 90, 100]
        x_points = np.linspace(0, P80*1.25, 100)
        y_points = np.interp(x_points, sizes, percentages)
        blasting_oversize_value = np.interp(10, sizes, percentages)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_points, y=y_points, mode='lines', name='Size Distribution',
                                 line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=sizes, y=percentages, mode='markers', name='Key Points',
                                 marker=dict(size=10, color='red')))
        fig.update_layout(
            title='Fragmentation Size Distribution',
            xaxis_title='Fragment Size (mm)',
            yaxis_title='Passing (%)',
            xaxis=dict(type="log"),
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)
 
        p80_diff = P80 - Blasting_target_p80
        status = "✅ Close to target" if abs(p80_diff) < 20 else "❌ Far from target"
        st.markdown(
            f"""
            <div style='padding:10px;border-radius:8px;border:1px solid #444;
                        background-color:#3b3b3b;color:#FFFFFF;'>
                <strong>Target P80: {Blasting_target_p80:.1f} mm | 
                Difference: {p80_diff:.1f} mm | {status}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
 
    # Parameter Impact Analysis (outside the container boxes)
    st.subheader("Parameter Impact Analysis")
    shap_metric = st.selectbox("Select which metric to analyze with SHAP", ["P20", "P50", "P80"], key="shap_metric_blasting")
    if shap_metric == "P20":
        chosen_model = models["blasting_p20"]
    elif shap_metric == "P50":
        chosen_model = models["blasting_p50"]
    else:
        chosen_model = models["blasting_p80"]
    explainer = shap.Explainer(chosen_model)
    shap_values = explainer(input_data)
    fig_shap, ax = plt.subplots(figsize=(8, 3))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=input_data.iloc[0]
        ),
        show=False
    )
    st.pyplot(fig_shap)
 
    # Sensitivity Analysis (still part of the blasting page, but not inside any container)
    sensitivity_ranges = {
        "Burden (m)": np.linspace(5.0, 8.0, 20),
        "Spacing (m)": np.linspace(5.0, 8.0, 20),
        "Hole Diameter (mm)": np.linspace(180, 240, 20),
        "UCS (MPa)": np.linspace(46.0, 60.0, 20),
        "Young's Modulus (GPa)": np.linspace(8.0, 12.0, 20),
        "Explosives Density - gr/cm3": np.linspace(0.8, 1.2, 20),
        "VOD - (m/s)": np.linspace(4000, 6000, 20)
    }
    sensitivity_param = st.selectbox(
        "Select Parameter for Sensitivity Analysis",
        list(sensitivity_ranges.keys()),
        key="sensitivity_param_blasting"
    )
    base_input = pd.DataFrame([[ucs, youngs_mod, burden, spacing, hole_diameter, explosive_density, vod]],
                              columns=[
                                  "UCS (MPa)",
                                  "Young's Modulus (GPa)",
                                  "Burden (m)",
                                  "Spacing (m)",
                                  "Hole Diameter (mm)",
                                  "Explosives Density - gr/cm3",
                                  "VOD - (m/s)"
                              ])
    sens_col1, sens_col2, sens_col3 = st.columns(3)
    metrics = [("P20", sens_col1), ("P50", sens_col2), ("P80", sens_col3)]
    for metric, col in metrics:
        with col:
            st.subheader(f"Sensitivity Analysis for {metric}")
            chosen_model = models.get(f"blasting_{metric.lower()}")
            if chosen_model is None:
                st.error(f"Model for blasting {metric} is not loaded.")
                continue
            explainer = shap.Explainer(chosen_model)
            sensitivity_data = []
            for val in sensitivity_ranges[sensitivity_param]:
                modified_input = base_input.copy()
                modified_input[sensitivity_param] = val
                shap_values_metric = explainer(modified_input)
                prediction_metric = chosen_model.predict(modified_input)[0]
                param_index = list(base_input.columns).index(sensitivity_param)
                param_impact = shap_values_metric.values[0][param_index]
                sensitivity_data.append({
                    "Parameter Value": val,
                    f"{metric} Prediction": prediction_metric,
                    "SHAP Impact": param_impact
                })
            sens_df = pd.DataFrame(sensitivity_data)
            chart = px.line(
                sens_df,
                x="Parameter Value",
                y="SHAP Impact",
                title=f"SHAP Impact of {sensitivity_param} on {metric}",
                labels={"Parameter Value": sensitivity_param, "SHAP Impact": f"Impact on {metric}"}
            )
            chart.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(chart)
            st.markdown("### Sensitivity Analysis Summary")
            st.metric("Max Impact", f"{sens_df['SHAP Impact'].max():.2f}")
 
    # Drilling & Blasting Cost section (now correctly outdented)
    st.markdown("---")
    st.subheader("Drilling & Blasting Cost")
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
 
    with cost_col1:
        st.markdown("### Drilling Parameters")
        hole_length = st.number_input("Hole Length (m)", min_value=0.0, value=10.0, key="hole_length_cost")
        stemming = st.number_input("Stemming (m)", min_value=0.0, value=3.0, key="stemming_cost")
        sub_drilling = st.number_input("Subdrilling (m)", min_value=0.0, value=2.0, key="sub_drilling_cost")
        blasting_area = st.number_input("Blasting area (m2)", min_value=500, value=2000, step=50, key="blasting_area_cost")
        num_holes = blasting_area // (burden * spacing)
 
    with cost_col2:
        st.markdown("### Economic Parameters")
        explosive_types = {"ANFO": 0.9, "Heavy ANFO": 1.1, "Emulsion": 1.2, "Watergel": 1.0}
        detonator_types = {"Non-electric": 15, "Electric": 25, "Electronic": 35}
        accessory_cost = st.number_input("Accessories Cost per blast ($)", min_value=0.0, value=5000.0, key="accessory_cost")
        labor_cost = st.number_input("Labor Cost per blast ($)", min_value=0.0, value=2000.0, key="labor_cost")
        selected_explosive = st.selectbox("Select Explosive Type", list(explosive_types.keys()), key="selected_explosive")
        selected_detonator = st.selectbox("Select Detonator Type", list(detonator_types.keys()), key="selected_detonator")
        explosive_cost_per_kg = explosive_types[selected_explosive]
        detonator_cost = detonator_types[selected_detonator]
 
    def calculate_charge(hole_length, stemming, sub_drilling, explosive_density, hole_diameter_mm, num_holes):
        hole_diameter_m = hole_diameter_mm / 1000
        charge_length = hole_length - stemming + sub_drilling
        charge_volume = np.pi * (hole_diameter_m / 2)**2 * charge_length
        charge_mass = charge_volume * explosive_density * 1000  # kg
        total_charge = charge_mass * num_holes
        return charge_mass, total_charge
 
    charge_mass, total_charge = calculate_charge(hole_length, stemming, sub_drilling, explosive_density, hole_diameter, num_holes)
    total_explosive_cost = total_charge * explosive_cost_per_kg
    total_drilling_cost = hole_length * num_holes * 50  # Example drilling rate: $50/m
    total_detonator_cost = detonator_cost * num_holes
    total_blasting_cost = total_explosive_cost + total_detonator_cost + accessory_cost + total_drilling_cost + labor_cost
    cost_distribution = pd.DataFrame({
        'Cost Component': ['Explosives', 'Drilling', 'Detonator', 'Accessories', 'Operational Cost'],
        'Cost ($)': [total_explosive_cost, total_drilling_cost, total_detonator_cost, accessory_cost, labor_cost]
    })
 
    with cost_col3:
        st.subheader("Cost Distribution")
        pie_fig = px.pie(cost_distribution, names='Cost Component', values='Cost ($)')
        st.plotly_chart(pie_fig, use_container_width=True)
 
    with cost_col4:
        inner_col1, inner_col2 = st.columns(2)
        with inner_col1:
            st.subheader("Total Blasting Cost")
            st.markdown(f"""
            <div class="box">
            <ul>
            <li><strong>Explosives Cost:</strong> ${total_explosive_cost:.2f}</li>
            <li><strong>Drilling Cost:</strong> ${total_drilling_cost:.2f}</li>
            <li><strong>Detonator Cost:</strong> ${(detonator_cost*num_holes):.2f}</li>
            <li><strong>Accessories Cost:</strong> ${accessory_cost:.2f}</li>
            </ul>
            <h4>Grand Total: ${total_blasting_cost:.2f}</h4>
            </div>
            """, unsafe_allow_html=True)
 
        with inner_col2:
            st.subheader("Total Blasting Explosives")
            st.markdown(f"""
            <div class="box">
            <ul>
            <li><strong>Total Holes:</strong> {num_holes:.2f}</li>
            <li><strong>Total Charge:</strong> {total_charge:.2f} kg</li>
            <li><strong>Total Drilling:</strong> {(hole_length*num_holes):.2f} m</li>
            <li><strong>Total Detonators:</strong> {num_holes:.0f}</li>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------
# Screening and Crusher pages remain similar...
# (For brevity, keep your existing implementations for "🥅 Screening", "⚙️ Crusher", and "📍 Location" pages)
# ------------------------------
elif current_page == "🥅 Screening":
    st.markdown("<h1 style='font-weight:bold;'>Screening Analysis</h1>", unsafe_allow_html=True)
    screen_col1, screen_col2 = st.columns(2)
    with screen_col1:
        st.subheader("Input Parameters for Screening")
        p20_blast = st.session_state.process_data['blasting']['p20']
        p50_blast = st.session_state.process_data['blasting']['p50']
        p80_blast = st.session_state.process_data['blasting']['p80']
        with st.expander("Blasting Feed", expanded=True):
            st.write(f"Blasting P20: {p20_blast:.2f} mm")
            st.write(f"Blasting P50: {p50_blast:.2f} mm")
            st.write(f"Blasting P80: {p80_blast:.2f} mm")
        with st.expander("Screening Parameters", expanded=True):
            alpha = st.slider("Alpha (Deck Angle)", 8.0, 12.0, 10.0, step=0.5, key="alpha_screening", help="Screen deck angle in degrees")
            d50 = st.slider("D50 (Cut Size)", 5.0, 9.0, 7.0, step=0.5, key="d50_screening", help="Size at which 50% of material passes")
            css = st.slider("Screen Aperture (mm)", 5.0, 15.0, 10.0, step=1.0, key="css_screening", help="Size of screen openings")
    with screen_col2:
        p20_os, p50_os, p80_os, mass_os, p20_us, p50_us, p80_us, mass_us, input_data = screening_prediction(
            alpha, d50, css, p20_blast, p50_blast, p80_blast
        )
        total_mass = mass_os + mass_us
        os_percent = (mass_os / total_mass * 100) if total_mass > 0 else 0
        us_percent = (mass_us / total_mass * 100) if total_mass > 0 else 0
        results_tab1, results_tab2 = st.tabs(["Results Summary", "Detailed Analysis"])
        with results_tab1:
            st.subheader("Screening Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("On-Screen Mass", f"{mass_os:.1f} t/h", f"{os_percent:.1f}%")
                st.metric("On-Screen P80", f"{p80_os:.2f} mm", f"{p80_os - p80_blast:.2f} mm", delta_color="off")
            with res_col2:
                st.metric("Under-Screen Mass", f"{mass_us:.1f} t/h", f"{us_percent:.1f}%")
                st.metric("Under-Screen P80", f"{p80_us:.2f} mm", f"{p80_us - p80_blast:.2f} mm", delta_color="off")
            pie_fig = go.Figure(data=[go.Pie(
                labels=['On-Screen', 'Under-Screen'],
                values=[os_percent, us_percent],
                hole=0.4,
                marker_colors=['#33a02c', '#1f78b4']
            )])
            pie_fig.update_layout(title_text="Mass Distribution")
            st.plotly_chart(pie_fig)
        with results_tab2:
            sizes_os = [0, p20_os, p50_os, p80_os, (p50_os+(p80_os-p50_os)*5/3)]
            percentages_os = [0, 20, 50, 80, 100]
            sizes_us = [0, p20_us, p50_us, p80_us, (p50_us+(p80_us-p50_us)*5/3)]
            percentages_us = [0, 20, 50, 80, 100]
            x_points_os = np.linspace(0, (p50_os+(p80_os-p50_os)*5/3)*1.25, 100)
            x_points_us = np.linspace(0, (p50_us+(p80_us-p50_us)*5/3)*1.25, 100)
            y_points_os = np.interp(x_points_os, sizes_os, percentages_os)
            y_points_us = np.interp(x_points_us, sizes_us, percentages_us)
            Onscreen_oversize_value = np.interp(css, sizes_os, percentages_os)
            Underscreen_oversize_value = np.interp(css, sizes_us, percentages_us)
            blasting_oversize_value = 15
            fig = go.Figure()
            fig.update_layout(
                title='Fragmentation Size Distribution',
                xaxis_title='Fragment Size (mm)',
                yaxis_title='Passing (%)',
                xaxis=dict(type="log"),
                yaxis=dict(range=[0, 105])
            )
            fig.add_trace(go.Scatter(x=x_points_os, y=y_points_os, mode='lines', name='On-Screen', line=dict(color='#33a02c')))
            fig.add_trace(go.Scatter(x=x_points_us, y=y_points_us, mode='lines', name='Under-Screen', line=dict(color='#1f78b4')))
            fig.add_trace(go.Scatter(x=sizes_os, y=percentages_os, mode='markers', name='On-Screen Points', marker=dict(color='#33a02c', size=10)))
            fig.add_trace(go.Scatter(x=sizes_us, y=percentages_us, mode='markers', name='Under-Screen Points', marker=dict(color='#1f78b4', size=10)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
            f"""
            <div style='padding:10px; border-radius:8px; border:1px solid #444; background-color:#3b3b3b; color:#FFFFFF;'>
                <strong>Screen Efficiency: { ((Underscreen_oversize_value - blasting_oversize_value) / (Underscreen_oversize_value - Onscreen_oversize_value))*100:.1f} %</strong>
            </div>
            """,
            unsafe_allow_html=True)
            st.subheader("Parameter Impact Analysis (SHAP)")
            st.info("SHAP Impact Analysis for Screening (Placeholder)")
            st.subheader("Screen Efficiency Analysis")
            css_values = np.linspace(5, 15, 10)
            efficiencies = []
            for css_val in css_values:
                onscreen_value = np.interp(css_val, sizes_os, percentages_os)
                underscreen_value = np.interp(css_val, sizes_us, percentages_us)
                efficiency = (((underscreen_value - blasting_oversize_value) / (underscreen_value - onscreen_value)) * 100)
                efficiencies.append(efficiency)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=css_values, y=efficiencies, mode='lines', name='Screen Efficiency'))
            fig.update_layout(
                title="Efficiency vs. CSS",
                xaxis_title="CSS (mm)",
                yaxis_title="Efficiency (%)",
                legend_title="Legend",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

elif current_page == "⚙️ Crusher":
    st.markdown("<h1 style='font-weight:bold;'>Crushing Analysis</h1>", unsafe_allow_html=True)
    onscreen_p20 = st.session_state.process_data['screening']['onscreen']['p20']
    onscreen_p50 = st.session_state.process_data['screening']['onscreen']['p50']
    onscreen_p80 = st.session_state.process_data['screening']['onscreen']['p80']
    onscreen_mass = st.session_state.process_data['screening']['onscreen']['mass']
    crush_col1, crush_col2 = st.columns(2)
    with crush_col1:
        st.subheader("Crusher Settings")
        with st.expander("Feed Properties", expanded=True):
            st.write(f"Feed P20: {onscreen_p20:.2f} mm")
            st.write(f"Feed P50: {onscreen_p50:.2f} mm")
            st.write(f"Feed P80: {onscreen_p80:.2f} mm")
            st.write(f"Feed Rate: {onscreen_mass:.1f} t/h")
        with st.expander("Crusher Parameters", expanded=True):
            css = st.slider("Closed Side Setting (mm)", 10, 50, 30, step=1, key="css_crusher")
            eccentric_speed = st.slider("Eccentric Speed (rpm)", 100, 300, 200, step=10, key="eccentric_speed")
            crusher_type = st.selectbox("Crusher Type", ["Jaw", "Cone", "Impact"], key="crusher_type")
        if crusher_type == "Jaw":
            reduction_ratio = 6
        elif crusher_type == "Cone":
            reduction_ratio = 8
        else:
            reduction_ratio = 15
        st.info(f"Expected Reduction Ratio: {reduction_ratio}:1")
    with crush_col2:
        p20_crush, p50_crush, p80_crush, mass_crush, power, input_data = crushing_prediction(
            css, onscreen_p20, onscreen_p50, onscreen_p80, onscreen_mass
        )
        st.subheader("Feed vs. Product Size Relationship")
        crush_subcol1, crush_subcol2 = st.columns(2)
        with crush_subcol1:    
            st.metric("Crusher Product Mass", f"{mass_crush:.1f} t/h")
        with crush_subcol2:
            st.metric("Crusher Product P80", f"{p80_crush:.2f} mm")
        product_sizes = [0, p20_crush, p50_crush, p80_crush, p80_crush*1.1, p80_crush*1.25]
        percentages = [0, 20, 50, 80, 90, 100]
        product_x_points = np.linspace(0, p80_crush*1.25, 100)
        product_y_points = np.interp(product_x_points, product_sizes, percentages)
        feed_sizes = [0, onscreen_p20, onscreen_p50, onscreen_p80, onscreen_p80*1.1, onscreen_p80*1.25]
        feed_x_points = np.linspace(0, onscreen_p80*1.25, 100)
        feed_y_points = np.interp(feed_x_points, feed_sizes, percentages)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_x_points, y=product_y_points, mode='lines', 
                                 name='Product Size Distribution', line=dict(color='#fb9a99')))
        fig.add_trace(go.Scatter(x=product_sizes, y=percentages, mode='markers', 
                                 name='Key Points', marker=dict(size=10, color='red')))
        fig.add_trace(go.Scatter(x=feed_x_points, y=feed_y_points, mode='lines', 
                                 name='Feed Size Distribution', line=dict(color='#33a02c')))
        fig.add_trace(go.Scatter(x=feed_sizes, y=percentages, mode='markers', 
                                 name='Key Points', marker=dict(size=10, color='red')))
        fig.add_vline(x=Crushing_target_p80, line_dash="dash", line_color="red", annotation_text="Target P80")
        fig.update_layout(
            title='Fragmentation Size Distribution',
            xaxis_title='Fragment Size (mm)',
            yaxis_title='Passing (%)',
            xaxis=dict(type="log", range=[0, 2.8457]),
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Feed vs. Product Size Relationship")
    feed_values = np.arange(400, 600, 20)
    p80_values = []
    power_values = []
    css = 10
    onscreen_p20 = st.session_state.process_data['screening']['onscreen']['p20']
    onscreen_p50 = st.session_state.process_data['screening']['onscreen']['p50']
    onscreen_mass = st.session_state.process_data['screening']['onscreen']['mass']
    for feed_val in feed_values:
        result = crushing_prediction(css, onscreen_p20, onscreen_p50, feed_val, onscreen_mass)
        p80_val = result[2] if isinstance(result, tuple) else result
        power_val = 10 * 5 * (1 / (p80_val ** 0.5) - 1 / (feed_val ** 0.5))
        p80_values.append(p80_val)
        power_values.append(power_val)
    feed_df = pd.DataFrame({
        'Feed (mm)': feed_values,
        'Crusher P80 (mm)': p80_values,
        'Power (kW)': power_values
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feed_df['Feed (mm)'], 
        y=feed_df['Crusher P80 (mm)'], 
        name='Crusher P80 (mm)', 
        line=dict(color='blue'),
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=feed_df['Feed (mm)'], 
        y=feed_df['Power (kW)'], 
        name='Power (kW)', 
        line=dict(color='red'),
        mode='lines+markers',
        yaxis='y2'
    ))
    fig.add_vline(x=onscreen_p80, line_dash="dash", line_color="green", annotation_text="Current p80")
    fig.add_hline(y=Crushing_target_p80, line_dash="dash", line_color="purple", annotation_text="Target P80")
    fig.update_layout(
        title='Effect of Feed P80 on Crusher P80 and Power',
        xaxis=dict(
            title='Feed (mm)',
            range=[feed_values.min(), feed_values.max()]
        ),
        yaxis=dict(
            title='Crusher P80 (mm)',
            side='left',
            showgrid=False,
            range=[50, 150]
        ),
        yaxis2=dict(
            title='Power (kW)',
            side='right',
            overlaying='y',
            showgrid=False,
            range=[2.5, 3]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

elif current_page == "📍 Location":
    st.title("Location")
    st.write("Location visualization and analysis content goes here.")

elif current_page == "🎯 Optimization":
    st.title("Optimization Settings")
    st.write("Optimization settings and recommendations content goes here.")
