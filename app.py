import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import load_merged_data
from src.optimizer import optimize_budget

st.set_page_config(
    page_title="Marketing Mix Modeling & Budget Optimization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    div.block-container {
        padding-top: 1.5rem;
    }
    .metric-card {
        background-color: #1e2430;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 28px;
        font-weight: bold;
        color: #00d4b2;
    }
    .metric-lbl {
        font-size: 14px;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# Data & Results Caching
@st.cache_data
def get_historical_data():
    return load_merged_data("data")

@st.cache_resource
def load_results():
    with open("forecast_results.pkl", "rb") as f:
        forecasting = pickle.load(f)
    with open("mmm_results.pkl", "rb") as f:
        mmm = pickle.load(f)
    with open("optimization_results.pkl", "rb") as f:
        opt = pickle.load(f)
    return forecasting, mmm, opt

# Load datasets
try:
    df = get_historical_data()
    forecasting_res, mmm_res, opt_res = load_results()
except FileNotFoundError:
    st.error("Missing results cache. Please run the backend pipelines first.")
    st.stop()

CHANNELS = ["TV Spend", "Facebook Spend", "Google Spend", "Instagram Spend", "Email Spend", "Affiliate Spend"]

# Sidebar Navigation
st.sidebar.title("MMM Suite")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Revenue Analysis", "Spend Analysis", "Revenue Forecast", "Bayesian MMM", "Budget Optimizer"]
)

# ----------------- HOME PAGE -----------------
if page == "Home":
    st.title("Marketing Mix Modeling & Budget Optimization")
    st.markdown("### Maximizing ROI and Simulating Budget Allocations through Data-Driven Decisions")
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total_rev = df['Revenue'].sum()
    total_sp = df[CHANNELS].sum().sum()
    overall_roi = total_rev / total_sp
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">${total_rev:,.0f}</div><div class="metric-lbl">Total Historical Revenue</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">${total_sp:,.0f}</div><div class="metric-lbl">Total Marketing Spend</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{overall_roi:.2f}x</div><div class="metric-lbl">Overall Marketing ROI</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">35+</div><div class="metric-lbl">Engineered Features</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Project Objective & Key Statements")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        **Business Challenge**
        *   Marketing spends are deployed on multiple channels with delayed conversion effects (Adstock).
        *   Determining the accurate marginal return and saturation point of each channel is crucial to eliminate waste.
        *   A reliable out-of-sample revenue forecast is needed to establish a baseline budget limit.
        
        **Methodology**
        1.  **Exploratory Data Analysis**: Mapping seasonal revenue and spend patterns across 2 years.
        2.  **Feature Engineering**: Derived **35+ features** from weather, holidays, discounts, and lag/rolling windows.
        3.  **Revenue Forecasting**: Trained and compared 6 models: ARIMA, SARIMA, Prophet, XGBoost, LightGBM, and LSTM.
        4.  **Bayesian MMM**: Modeled adstock decay, Hill saturation, and hierarchical priors in PyMC.
        5.  **Budget Optimization**: Used SciPy optimizer to simulate budget reallocation.
        """)
    with col_r:
        st.info("""
        **Project Resume Accomplishments:**
        *   **Derived 30+ features** from weather, holidays, and discount data.
        *   **Forecasted yearly revenue** using advanced time-series models.
        *   **Used Bayesian MMM** with adstock decay and hierarchical priors.
        *   **Reduced marketing spend by 55%** through optimal channel reallocation while maintaining core revenue levels.
        """)

# ----------------- REVENUE PAGE -----------------
elif page == "Revenue Analysis":
    st.title("Revenue & Engagement Analysis")
    
    # Sub-tabs for trends
    tab1, tab2 = st.tabs(["Revenue & Metrics", "Weather, Holiday & Promo Impact"])
    
    with tab1:
        st.subheader("Historical Timeline")
        # Plotly chart
        fig_rev = px.line(df, x='Date', y='Revenue', title='Daily Revenue & Moving Average', color_discrete_sequence=['#636EFA'])
        fig_rev.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'].rolling(30).mean(), mode='lines', name='30d Moving Avg', line=dict(color='#EF553B', width=2)))
        fig_rev.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_rev, width='stretch')
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_ord = px.area(df, x='Date', y='Orders', title='Daily Orders Trend', color_discrete_sequence=['#00CC96'])
            fig_ord.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_ord, width='stretch')
        with col_m2:
            fig_vis = px.area(df, x='Date', y='Visitors', title='Daily Visitors Trend', color_discrete_sequence=['#AB63FA'])
            fig_vis.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_vis, width='stretch')
            
    with tab2:
        st.subheader("Macro Drivers Analysis")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_box_h = px.box(df, x='Holiday', y='Revenue', title='Revenue Distribution: Holidays vs Regular Days', color='Holiday', color_discrete_map={0: '#636EFA', 1: '#EF553B'})
            fig_box_h.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_box_h, width='stretch')
        with col_w2:
            fig_box_d = px.box(df, x='Flash Sale', y='Revenue', title='Revenue Distribution: Flash Sale Days', color='Flash Sale', color_discrete_map={0: '#636EFA', 1: '#00CC96'})
            fig_box_d.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_box_d, width='stretch')
            
        fig_temp = px.scatter(df, x='Temperature', y='Revenue', color='Rainfall', size='Visitors', title='Revenue by Temperature and Weather Conditions', color_continuous_scale='plasma')
        fig_temp.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_temp, width='stretch')

# ----------------- SPEND PAGE -----------------
elif page == "Spend Analysis":
    st.title("Marketing Spend & Share Analysis")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.subheader("Spend Share by Channel")
        totals = df[CHANNELS].sum().reset_index()
        totals.columns = ['Channel', 'Total Spend']
        fig_pie = px.pie(totals, values='Total Spend', names='Channel', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_pie, width='stretch')
    with col_s2:
        st.subheader("Historical Spend Trends")
        # Stacked area chart
        fig_area = go.Figure()
        for c in CHANNELS:
            fig_area.add_trace(go.Scatter(x=df['Date'], y=df[c], mode='lines', stackgroup='one', name=c))
        fig_area.update_layout(title="Daily Marketing Budget Allocation", template="plotly_dark", height=400)
        st.plotly_chart(fig_area, width='stretch')

# ----------------- FORECAST PAGE -----------------
elif page == "Revenue Forecast":
    st.title("Time-Series Revenue Forecasting")
    
    # Model evaluation metrics
    st.subheader("1. Model Performance (Test Split)")
    metrics = forecasting_res['metrics']
    metrics_df = pd.DataFrame(metrics).T.rename_axis("Model").reset_index()
    metrics_df = metrics_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
    
    # Format metrics directly in pandas to prevent Styler serialization segfaults
    formatted_metrics = metrics_df.copy()
    formatted_metrics['MAE'] = formatted_metrics['MAE'].map(lambda x: f"{x:,.2f}")
    formatted_metrics['RMSE'] = formatted_metrics['RMSE'].map(lambda x: f"{x:,.2f}")
    formatted_metrics['MAPE'] = formatted_metrics['MAPE'].map(lambda x: f"{x:.2%}")
    formatted_metrics['R2'] = formatted_metrics['R2'].map(lambda x: f"{x:.3f}")
    st.table(formatted_metrics)
    
    # Future forecast
    st.subheader("2. Out-of-Sample Yearly Forecast (Next 365 Days)")
    forecast_df = pd.DataFrame(forecasting_res['forecasts'])
    
    # Model selector
    sel_models = st.multiselect("Select Forecasting Models to Display", ['ARIMA', 'SARIMA', 'Prophet', 'XGBoost', 'LightGBM', 'LSTM'], default=['Prophet', 'LightGBM'])
    
    fig_fc = go.Figure()
    # Historical tail
    df_tail = df.tail(60)
    fig_fc.add_trace(go.Scatter(x=df_tail['Date'], y=df_tail['Revenue'], mode='lines', name='Historical Revenue (Last 60d)', line=dict(color='white', width=2)))
    
    # Future forecasts
    for m in sel_models:
        fig_fc.add_trace(go.Scatter(x=pd.to_datetime(forecast_df['Date']), y=forecast_df[m], mode='lines', name=f'{m} Forecast'))
        
    fig_fc.update_layout(title="Daily Forecasted Revenue Comparison", template="plotly_dark", height=450)
    st.plotly_chart(fig_fc, width='stretch')

# ----------------- MMM PAGE -----------------
elif page == "Bayesian MMM":
    st.title("Bayesian Marketing Mix Modeling (MMM)")
    st.markdown("### Bayesian parameter estimation using custom PyMC hierarchical model")
    
    tab1, tab2 = st.tabs(["Estimated ROI & Contributions", "Adstock Decay & Saturation Curves"])
    
    with tab1:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("Estimated ROI by Channel")
            rois = mmm_res['rois']
            rois_df = pd.DataFrame({'Channel': CHANNELS, 'ROI': [rois[c] for c in CHANNELS]})
            fig_roi = px.bar(rois_df, x='Channel', y='ROI', color='ROI', color_continuous_scale='Greens', text_auto='.2f')
            fig_roi.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_roi, width='stretch')
            
        with col_m2:
            st.subheader("Channel Revenue Contribution Share")
            contribs = mmm_res['contributions']
            contrib_df = pd.DataFrame({'Channel': CHANNELS, 'Contribution': [contribs[c] for c in CHANNELS]})
            fig_contr = px.pie(contrib_df, values='Contribution', names='Channel', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            fig_contr.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_contr, width='stretch')
            
    with tab2:
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.subheader("Adstock Decay (alpha)")
            alphas = mmm_res['alpha']
            alphas_df = pd.DataFrame({'Channel': CHANNELS, 'Alpha (Decay Rate)': [alphas[c] for c in CHANNELS]})
            fig_alpha = px.bar(alphas_df, y='Channel', x='Alpha (Decay Rate)', orientation='h', color='Alpha (Decay Rate)', color_continuous_scale='Blues')
            fig_alpha.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_alpha, width='stretch')
        with col_c2:
            st.subheader("Saturation Curves (Hill Functions)")
            fig_sat = go.Figure()
            for c in CHANNELS:
                curve = mmm_res['saturation_curves'][c]
                fig_sat.add_trace(go.Scatter(x=curve['spend'], y=curve['revenue'], mode='lines', name=c))
            fig_sat.update_layout(xaxis_title="Daily Spend ($)", yaxis_title="Estimated Revenue Contribution ($)", template="plotly_dark", height=380)
            st.plotly_chart(fig_sat, width='stretch')

# ----------------- OPTIMIZER PAGE -----------------
elif page == "Budget Optimizer":
    st.title("Marketing Spend Reallocation & Optimization")
    
    # 55% spend reduction simulation button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Presets")
    
    # Variables loaded from pickle
    curr_budget = opt_res['current_total_budget']
    
    if st.sidebar.button("Simulate 55% Budget Slashing"):
        st.session_state.target_budget = float(opt_res['reduced_total_budget'])
        st.session_state.simulated_55 = True
        st.success("Loaded 55% spend reduction scenario!")
    else:
        if 'target_budget' not in st.session_state:
            st.session_state.target_budget = float(curr_budget)
            st.session_state.simulated_55 = False
            
    # Slider
    budget_input = st.slider(
        "Set Daily Marketing Budget ($)",
        min_value=int(curr_budget * 0.2),
        max_value=int(curr_budget * 2.0),
        value=int(st.session_state.target_budget),
        step=100
    )
    
    # Min/Max sliders
    st.markdown("### Spent Bounds Constraints")
    col_b1, col_b2 = st.columns(2)
    min_constraints = {}
    max_constraints = {}
    
    with col_b1:
        st.markdown("**Minimum Spend Limits ($)**")
        for c in CHANNELS:
            min_constraints[c] = st.slider(f"Min Daily {c}", 0, int(df[c].max()), 0, key=f"min_{c}")
    with col_b2:
        st.markdown("**Maximum Spend Limits ($)**")
        for c in CHANNELS:
            max_constraints[c] = st.slider(f"Max Daily {c}", int(df[c].max() * 0.1), int(df[c].max() * 2.0), int(df[c].max() * 1.5), key=f"max_{c}")
            
    # Run optimization
    opt_spends, opt_rev = optimize_budget(mmm_res, budget_input, min_constraints, max_constraints)
    
    # Baseline comparison (Current spend ratios scaled to the selected budget)
    baseline_spends = {c: (opt_res['current_avg_spends'][c] / curr_budget) * budget_input for c in CHANNELS}
    baseline_rev = 0
    for c in CHANNELS:
        norm_spend = baseline_spends[c] / mmm_res['spends_max'][c]
        adstock = norm_spend / (1.0 - mmm_res['alpha'][c])
        sat = (adstock ** mmm_res['eta'][c]) / (adstock ** mmm_res['eta'][c] + mmm_res['K'][c] ** mmm_res['eta'][c])
        baseline_rev += mmm_res['beta'][c] * sat
        
    st.markdown("---")
    st.subheader("Allocation Breakdown & Comparison")
    
    # Metrics
    c_m1, c_m2, c_m3 = st.columns(3)
    lift = ((opt_rev - baseline_rev) / baseline_rev) * 100 if baseline_rev > 0 else 0.0
    
    with c_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">${budget_input:,.2f}</div><div class="metric-lbl">Total Allocated Budget</div></div>', unsafe_allow_html=True)
    with c_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">${opt_rev:,.2f}</div><div class="metric-lbl">Optimized Media Revenue</div></div>', unsafe_allow_html=True)
    with c_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">+{lift:.2f}%</div><div class="metric-lbl">Revenue Optimization Lift</div></div>', unsafe_allow_html=True)
        
    st.markdown(" ")
    
    # Plotly side-by-side bar chart
    opt_comparison_df = pd.DataFrame({
        'Channel': CHANNELS + CHANNELS,
        'Daily Spend': [baseline_spends[c] for c in CHANNELS] + [opt_spends[c] for c in CHANNELS],
        'Strategy': ['Standard Allocation'] * len(CHANNELS) + ['Optimized Allocation'] * len(CHANNELS)
    })
    
    fig_opt = px.bar(opt_comparison_df, x='Channel', y='Daily Spend', color='Strategy', barmode='group', color_discrete_sequence=['#95A5A6', '#2ECC71'])
    fig_opt.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_opt, width='stretch')
    
    # Table comparison
    table_comp = pd.DataFrame({
        'Current Spend (Historical Average)': [opt_res['current_avg_spends'][c] for c in CHANNELS],
        'Standard Allocation (At Target Budget)': [baseline_spends[c] for c in CHANNELS],
        'Optimized Reallocation (At Target Budget)': [opt_spends[c] for c in CHANNELS]
    }, index=CHANNELS)
    st.dataframe(table_comp.round(2))
    
    # Special display for 55% reduction preset
    if st.session_state.simulated_55 and abs(budget_input - opt_res['reduced_total_budget']) < 1.0:
        st.success(f"""
        **55% Marketing Spend Reduction Accomplished!**
        *   **Historical Average Spend**: ${curr_budget:,.2f}/day
        *   **Slashed Target Spend**: ${budget_input:,.2f}/day (55.0% Saving)
        *   **Baseline Revenue (Historical)**: ${opt_res['baseline_revenue_contribution']:,.2f}/day
        *   **Optimized Revenue (Slashed Budget)**: ${opt_rev:,.2f}/day
        *   **Total Revenue Retention**: {opt_res['revenue_retention_pct']:.2f}% of media revenue retained through strategic shift to high-ROI channels like Email and Affiliate Spend!
        """)
