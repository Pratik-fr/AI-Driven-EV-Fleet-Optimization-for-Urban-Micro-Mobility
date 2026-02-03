import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing import generate_ride_data, generate_telemetry_data, preprocess_ride_data, aggregate_demand
from src.eda import plot_demand_trends_by_hour, plot_demand_heatmap, plot_battery_status_distribution, plot_fleet_status_pie
from src.forecasting import train_demand_model, predict_next_hour_demand
from src.battery_model import train_battery_model
from src.allocation_or import optimize_fleet_allocation
from src.genai_assistant import query_genai_assistant

# Page Config
st.set_page_config(page_title="Smart EV Fleet Optimization", layout="wide")

# Title & Sidebar
st.title("⚡ Smart EV Fleet Optimization & Forecasting System")
st.sidebar.header("Control Panel")

# --- Data Loading (Cached) ---
def load_data(realtime_mode=False):
    if realtime_mode:
        import os
        if os.path.exists("data/realtime_rides.csv"):
            ride_df = pd.read_csv("data/realtime_rides.csv")
            # Re-apply preprocessing since CSV loses some types
            ride_df = preprocess_ride_data(ride_df)
        else:
            st.warning("Realtime data file not found. Run `python src/simulation.py`. data/realtime_rides.csv missing")
            # Fallback
            ride_df = generate_ride_data(num_rides=100)
            ride_df = preprocess_ride_data(ride_df)
    else:
        ride_df = generate_ride_data(num_rides=2000)
        ride_df = preprocess_ride_data(ride_df)
        
    telemetry_df = generate_telemetry_data(num_vehicles=50)
    return ride_df, telemetry_df

# Sidebar Controls
realtime_mode = st.sidebar.checkbox("🚀 Enable Real-Time Data Mode")

if realtime_mode:
    st.sidebar.warning("Live Data stream active. Local file: data/realtime_rides.csv")
    # Auto-refresh mechanism
    import time
    time.sleep(2) # Simple polling loop simulation (in a real app, use st.empty or fragments)
    st.rerun()

ride_df, telemetry_df = load_data(realtime_mode)

# Sidebar Stats
total_rides = len(ride_df)
active_vehicles = telemetry_df[telemetry_df['status'] == 'In_Use'].shape[0]
st.sidebar.metric("Total Rides Analyzed", total_rides)
st.sidebar.metric("Active Vehicles", active_vehicles)

# Button to refresh data
if st.sidebar.button("Regenerate Synthetic Data"):
    st.cache_data.clear()
    st.rerun()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA & Insights", 
    "📈 Demand Forecasting", 
    "🚚 Fleet Allocation (OR)", 
    "🔋 Battery Health", 
    "🤖 GenAI Assistant"
])

# --- Tab 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    demand_agg = aggregate_demand(ride_df)
    
    with col1:
        st.subheader("Demand Trends by Hour")
        st.pyplot(plot_demand_trends_by_hour(demand_agg))
        
    with col2:
        st.subheader("Demand Heatmap")
        st.pyplot(plot_demand_heatmap(demand_agg))
        
    st.subheader("Current Fleet Status")
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_fleet_status_pie(telemetry_df))
    with col4:
        st.dataframe(telemetry_df.head())

# --- Tab 2: Forecasting ---
with tab2:
    st.header("Demand Forecasting (XGBoost)")
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, metrics, _ = train_demand_model(demand_agg)
            st.session_state['demand_model'] = model
            st.session_state['demand_metrics'] = metrics
            st.success("Model Trained Successfully!")
            
            st.subheader("Model Evaluation Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Baseline MAE", metrics['Baseline_MAE'])
            m_col2.metric("Baseline RMSE", metrics['Baseline_RMSE'])
            m_col3.metric("XGBoost MAE", metrics['XGB_MAE'])
            m_col4.metric("XGBoost RMSE", metrics['XGB_RMSE'])
            
            if metrics['XGB_RMSE'] < metrics['Baseline_RMSE']:
                st.info("✅ ML Model outperformed the Baseline.")
            else:
                st.warning("⚠️ Baseline is performing better. More data/features needed.")

# --- Tab 3: Operations Research ---
with tab3:
    st.header("Operations Research: Fleet Allocation")
    
    st.markdown("""
    **Goal:** Minimize total distance traveled to rebalance fleet.
    Using **Hungarian Algorithm (Linear Sum Assignment)**.
    """)
    
    # Predict demand using the model if available, else use mock
    if 'demand_model' in st.session_state:
        with st.spinner("Predicting next hour demand..."):
            predicted_demand = predict_next_hour_demand(st.session_state['demand_model'], demand_agg)
        st.info("💡 Using trained XGBoost model for predictions.")
    else:
        predicted_demand = {
            'Zone_A': 15, 'Zone_B': 5, 'Zone_C': 20, 'Zone_D': 8, 'Zone_E': 10
        }
        st.warning("⚠️ Using mock demand data. Train the model in the 'Demand Forecasting' tab for real predictions.")
    
    col_d, col_s = st.columns(2)
    with col_d:
        st.subheader("Predicted Demand (Next Hour)")
        st.json(predicted_demand)
        
    with col_s:
        current_supply = telemetry_df[telemetry_df['status'] == 'Available']['current_zone'].value_counts().to_dict()
        st.subheader("Current Supply (Available EVs)")
        st.write(current_supply)
    
    if st.button("Optimize Allocation"):
        allocation_df, cost = optimize_fleet_allocation(telemetry_df, predicted_demand)
        
        if isinstance(allocation_df, str):
            st.info(allocation_df)
        else:
            st.success(f"Optimization Complete! Total Cost (Distance): {cost} km")
            st.dataframe(allocation_df)
            
            # Simple visualization of moves
            st.bar_chart(allocation_df['from_zone'].value_counts())

# --- Tab 4: Battery Health ---
with tab4:
    st.header("Battery Health Monitoring")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.pyplot(plot_battery_status_distribution(telemetry_df))
        
    with col_b2:
        st.subheader("Predictive Maintenance Model")
        if st.button("Train Battery Risk Model"):
            clf, metrics = train_battery_model(telemetry_df)
            st.metric("Model Accuracy", f"{metrics['Accuracy']*100}%")
            
            # Show critical vehicles
            critical = telemetry_df[telemetry_df['battery_level'] < 20]
            st.warning(f"{len(critical)} Vehicles need immediate charging/swapping.")
            st.dataframe(critical[['vehicle_id', 'battery_level', 'current_zone']])

# --- Tab 5: GenAI Assistant ---
with tab5:
    st.header("🧠 GenAI Analytics Assistant")
    
    # Prepare Context
    current_hour = ride_df['timestamp'].max().hour
    high_demand_zone = demand_agg[demand_agg['hour'] == current_hour].sort_values('demand_count', ascending=False)['pickup_zone'].iloc[0] if not demand_agg.empty else "N/A"
    
    context = {
        "total_rides": total_rides,
        "active_vehicles": active_vehicles,
        "critical_battery_count": len(telemetry_df[telemetry_df['battery_level'] < 20]),
        "current_hour": current_hour,
        "high_demand_zone_now": high_demand_zone,
        "model_trained": 'demand_model' in st.session_state,
        "realtime_mode": realtime_mode
    }
    
    if 'demand_model' in st.session_state:
        context['model_mae'] = st.session_state['demand_metrics']['XGB_MAE']
    
    # st.write("Context Data used for RAG:", context)
    
    user_query = st.text_input("Ask a question about the fleet:", "Which zone has the highest shortage risk?")
    
    if st.button("Ask Assistant"):
        with st.spinner("Thinking..."):
            response = query_genai_assistant(user_query, context)
            st.markdown(f"**Assistant:** {response}")
