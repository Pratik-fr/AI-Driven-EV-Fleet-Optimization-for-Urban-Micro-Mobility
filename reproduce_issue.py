
import pandas as pd
import numpy as np
from src.preprocessing import generate_ride_data, aggregate_demand
from src.forecasting import train_demand_model, predict_next_hour_demand

def test_forecasting_flow():
    print("Generating synthetic data...")
    ride_df = generate_ride_data(num_rides=500)
    print("Pre-processing data...")
    # aggregate_demand adds hour, day_of_week, is_weekend
    demand_df = aggregate_demand(ride_df)
    
    print("Columns in demand_df:", demand_df.columns)
    
    print("Training model...")
    model, metrics, _ = train_demand_model(demand_df)
    print("Model trained.")
    print("Metrics:", metrics)
    
    # Check features expected by model
    if hasattr(model, 'feature_names_in_'):
        print("Model expects features:", model.feature_names_in_)
    else:
        print("Model expects features (booster):", model.get_booster().feature_names)

    print("Predicting next hour demand...")
    predictions = predict_next_hour_demand(model, demand_df)
    print("Predictions:", predictions)
    print("Success!")

if __name__ == "__main__":
    test_forecasting_flow()
