import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

def feature_engineering_for_demand(demand_df):
    """
    Prepares features for the demand model.
    """
    df = demand_df.copy()

    # Ensure is_weekend is int
    if 'is_weekend' in df.columns:
        df['is_weekend'] = df['is_weekend'].astype(int)
    
    # Lag features: Demand 1 hour ago, 24 hours ago
    df['lag_1h'] = df.groupby('pickup_zone')['demand_count'].shift(1)
    df['lag_24h'] = df.groupby('pickup_zone')['demand_count'].shift(24)
    
    # Rolling mean features
    df['rolling_mean_3h'] = df.groupby('pickup_zone')['demand_count'].shift(1).rolling(window=3).mean()
    
    # Drop NaNs created by lag
    df.dropna(inplace=True)
    
    # Encode Zone (One-Hot or Label Encoding) - Using simple mapping for now or One-Hot
    # Using get_dummies for zones. Ensure dtype is int to be included in numeric features.
    df = pd.get_dummies(df, columns=['pickup_zone'], prefix='zone', drop_first=False, dtype=int)
    
    return df

def train_demand_model(demand_df):
    """
    Trains an XGBoost model for demand forecasting.
    """
    # Prepare data
    model_df = feature_engineering_for_demand(demand_df)
    
    # Define features and target
    drop_cols = ['time_bin', 'demand_count', 'timestamp'] # + any other non-numeric
    features = [c for c in model_df.columns if c not in drop_cols and c in model_df.select_dtypes(include=[np.number]).columns]
    target = 'demand_count'
    
    X = model_df[features]
    y = model_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 1. Baseline Model: Moving Average (already computed as rolling_mean_3h)
    # We use lag_24h as a simplistic baseline (Same time yesterday)
    y_pred_baseline = X_test['lag_24h']
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    
    # 2. ML Model: XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_xgb = model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    
    metrics = {
        'Baseline_MAE': round(mae_baseline, 2),
        'Baseline_RMSE': round(rmse_baseline, 2),
        'XGB_MAE': round(mae_xgb, 2),
        'XGB_RMSE': round(rmse_xgb, 2)
    }
    
    return model, metrics, feature_engineering_for_demand # Returning function to reproduce features

def predict_next_hour_demand(model, demand_df):
    """
    Predicts demand for the next hour for all zones based on the latest data.
    """
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    
    # Get the most recent time bin
    latest_time = demand_df['time_bin'].max()
    next_time = latest_time + pd.Timedelta(hours=1)
    
    predictions = {}
    
    # We need to build a feature vector for each zone for the 'next_time'
    for zone in zones:
        # Get historical data for this zone to compute lags
        zone_data = demand_df[demand_df['pickup_zone'] == zone].sort_values('time_bin')
        
        if len(zone_data) < 24:
            # Not enough data for lag_24h, return a baseline or random
            predictions[zone] = int(zone_data['demand_count'].mean()) if not zone_data.empty else 5
            continue
            
        # Features for next hour
        hour = next_time.hour
        day_of_week = next_time.dayofweek
        is_weekend = int(day_of_week >= 5)
        
        lag_1h = zone_data.iloc[-1]['demand_count']
        lag_24h = zone_data[zone_data['time_bin'] <= (next_time - pd.Timedelta(hours=24))].iloc[-1]['demand_count'] if not zone_data[zone_data['time_bin'] <= (next_time - pd.Timedelta(hours=24))].empty else lag_1h
        
        rolling_mean_3h = zone_data.iloc[-3:]['demand_count'].mean()
        
        # Prepare feature dict
        feat_dict = {
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'lag_1h': [lag_1h],
            'lag_24h': [lag_24h],
            'rolling_mean_3h': [rolling_mean_3h]
        }
        
        # Add zone one-hot features
        for z in zones:
            feat_dict[f'zone_{z}'] = [1 if z == zone else 0]
            
        X_next = pd.DataFrame(feat_dict)
        
        # Ensure column order matches training (XGBoost is sensitive to this)
        # Using the booster's feature names to order columns
        booster = model.get_booster()
        # feature_names might be available directly on model in scikit-learn API wrapper
        if hasattr(model, 'feature_names_in_'):
             expected_cols = model.feature_names_in_
        else:
             expected_cols = booster.feature_names
             
        # Reorder columns to match model expectation
        # Missing columns (shouldn't happen with correct logic) filled with 0
        # Extra columns (shouldn't happen) dropped
        X_next = X_next.reindex(columns=expected_cols, fill_value=0)
        
        pred = model.predict(X_next)[0]
        predictions[zone] = max(0, int(round(pred)))
        
    return predictions
