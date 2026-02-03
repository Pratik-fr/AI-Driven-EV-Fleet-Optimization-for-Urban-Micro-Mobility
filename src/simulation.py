import pandas as pd
import time
import random
from datetime import datetime, timedelta
import os
from src.preprocessing import generate_ride_data

# File paths
REALTIME_DATA_FILE = "data/realtime_rides.csv"

def initialize_realtime_data():
    """Initializes the realtime data file with some historical data."""
    if not os.path.exists("data"):
        os.makedirs("data")
        
    print("Converting historical data to realtime file...")
    # Start with 500 records
    df = generate_ride_data(num_rides=500, days=2)
    df.to_csv(REALTIME_DATA_FILE, index=False)
    print(f"Initialized {REALTIME_DATA_FILE} with {len(df)} records.")

def simulate_realtime_stream(interval_sec=2):
    """
    Appends a new ride every interval_sec to the CSV.
    """
    if not os.path.exists(REALTIME_DATA_FILE):
        initialize_realtime_data()
        
    print(f"Starting Real-Time Simulator. Appending new rides every {interval_sec} seconds...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Generate 1 new ride
            # We use 'days=0' to imply 'now' or very recent
            new_ride_df = generate_ride_data(num_rides=1, start_date=datetime.now(), days=0)
            
            # Append to CSV
            new_ride_df.to_csv(REALTIME_DATA_FILE, mode='a', header=False, index=False)
            
            # Print status
            ride_info = new_ride_df.iloc[0]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New Ride: {ride_info['pickup_zone']} -> {ride_info['drop_zone']}")
            
            time.sleep(interval_sec)
            
    except KeyboardInterrupt:
        print("\nSimulator stopped.")

if __name__ == "__main__":
    simulate_realtime_stream()
