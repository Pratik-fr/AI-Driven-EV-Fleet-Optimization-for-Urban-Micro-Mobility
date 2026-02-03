import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import generate_ride_data, generate_telemetry_data, preprocess_ride_data, aggregate_demand
from src.forecasting import train_demand_model
from src.battery_model import train_battery_model
from src.allocation_or import optimize_fleet_allocation

def test_pipeline():
    print("1. Generating Data...")
    ride_df = generate_ride_data(num_rides=500)
    telemetry_df = generate_telemetry_data(num_vehicles=20)
    
    assert not ride_df.empty, "Ride data is empty"
    assert not telemetry_df.empty, "Telemetry data is empty"
    print("   Data generation successful.")

    print("2. Preprocessing...")
    ride_df_clean = preprocess_ride_data(ride_df)
    demand_agg = aggregate_demand(ride_df_clean)
    
    assert 'hour' in demand_agg.columns, "Feature engineering failed"
    print("   Preprocessing successful.")

    print("3. Training Demand Model...")
    model, metrics, _ = train_demand_model(demand_agg)
    print(f"   Model trained. Metrics: {metrics}")
    assert model is not None, "Demand model is None"

    print("4. Training Battery Model...")
    clf, batt_metrics = train_battery_model(telemetry_df)
    print(f"   Battery model trained. Accuracy: {batt_metrics['Accuracy']}")
    assert clf is not None, "Battery model is None"

    print("5. Optimization...")
    # Mock demand
    predicted_demand = {'Zone_A': 10, 'Zone_B': 5}
    allocation, cost = optimize_fleet_allocation(telemetry_df, predicted_demand)
    
    if isinstance(allocation, pd.DataFrame):
        print(f"   Optimization generated plan with cost: {cost}")
    else:
        print(f"   Optimization message: {allocation}")
        
    print("\n✅ Integration Test Passed Successfully!")

if __name__ == "__main__":
    test_pipeline()
