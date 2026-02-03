import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Synthetic Data Generation ---

def generate_ride_data(num_rides=1000, start_date=None, days=30):
    """
    Generates synthetic ride data for the EV fleet.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    data = []
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    
    for i in range(num_rides):
        ride_id = f"ride_{i:05d}"
        
        # Random timestamp within the period
        random_seconds = random.randint(0, days * 24 * 3600)
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        pickup_zone = random.choice(zones)
        drop_zone = random.choice(zones)
        
        # Introduce some valid and invalid rides
        # E.g., very short rides or very long rides
        duration_min = max(2, int(np.random.normal(15, 5))) # Normal distrib centered at 15 mins
        distance_km = round(duration_min * 0.25 + np.random.normal(0, 0.5), 2) # correlated with duration
        distance_km = max(0.5, distance_km)

        # Introduce Missing Values (Data Quality Issues)
        if random.random() < 0.02:
            duration_min = np.nan
        if random.random() < 0.02:
            drop_zone = None
            
        data.append({
            'ride_id': ride_id,
            'timestamp': timestamp,
            'pickup_zone': pickup_zone,
            'drop_zone': drop_zone,
            'ride_duration_min': duration_min,
            'distance_km': distance_km
        })
        
    df = pd.DataFrame(data)
    return df

def generate_telemetry_data(num_vehicles=50):
    """
    Generates synthetic telemetry data for the EV fleet (current status).
    """
    data = []
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    statuses = ['Available', 'In_Use', 'Maintenance', 'Low_Battery']
    
    for i in range(num_vehicles):
        vehicle_id = f"EV_{i:03d}"
        battery_level = random.randint(5, 100)
        current_zone = random.choice(zones)
        
        # Logic: Low battery implies Low_Battery status
        if battery_level < 20:
            status = 'Low_Battery'
        else:
            status = random.choices(statuses, weights=[0.6, 0.3, 0.1, 0.0])[0]
            
        data.append({
            'vehicle_id': vehicle_id,
            'battery_level': battery_level,
            'current_zone': current_zone,
            'status': status,
            'last_update': datetime.now()
        })
        
    df = pd.DataFrame(data)
    return df


# --- Preprocessing & Feature Engineering ---

def preprocess_ride_data(df):
    """
    Cleans and enhances the ride data.
    """
    df = df.copy()
    
    # Handle Missing Values
    # Fill missing duration with median
    df['ride_duration_min'] = df['ride_duration_min'].fillna(df['ride_duration_min'].median())
    
    # Drop rows where critical info like pickup_zone is missing (if any) or drop_zone
    df.dropna(subset=['pickup_zone', 'drop_zone'], inplace=True)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    df['date_key'] = df['timestamp'].dt.date
    
    # Simple Holiday Logic (Example)
    # Assume 1st and 15th of the month are holidays
    df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x.day in [1, 15] else 0)
    
    return df

def aggregate_demand(df, time_bin='1H'):
    """
    Aggregates ride data to create a demand timeseries per zone.
    """
    # Round time to nearest bin
    df['time_bin'] = df['timestamp'].dt.floor(time_bin)
    
    demand_df = df.groupby(['time_bin', 'pickup_zone']).size().reset_index(name='demand_count')
    
    # Add time features to demand_df for modeling
    demand_df['hour'] = demand_df['time_bin'].dt.hour
    demand_df['day_of_week'] = demand_df['time_bin'].dt.dayofweek # 0=Monday
    demand_df['is_weekend'] = demand_df['day_of_week'] >= 5
    
    return demand_df
