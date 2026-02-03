import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_demand_trends_by_hour(demand_df):
    """
    Plots average demand by hour of the day.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=demand_df, x='hour', y='demand_count', hue='pickup_zone', marker="o")
    plt.title("Average Demand by Hour & Zone")
    plt.xlabel("Hour of Day")
    plt.ylabel("Ride Count")
    plt.grid(True)
    return plt

def plot_demand_heatmap(demand_df):
    """
    Plots a heatmap of demand by day of week and hour.
    """
    # Pivot for heatmap: Index=Day, Columns=Hour, Values=Demand
    heatmap_data = demand_df.groupby(['day_of_week', 'hour'])['demand_count'].mean().unstack()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f")
    plt.title("Demand Heatmap (Day vs Hour)")
    return plt

def plot_battery_status_distribution(telemetry_df):
    """
    Plots distribution of battery levels.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(telemetry_df['battery_level'], bins=20, kde=True, color='green')
    plt.title("Current Fleet Battery Level Distribution")
    plt.xlabel("Battery Level (%)")
    plt.axvline(20, color='red', linestyle='--', label='Critical Threshold (20%)')
    plt.legend()
    return plt

def plot_fleet_status_pie(telemetry_df):
    """
    Pie chart of vehicle status.
    """
    status_counts = telemetry_df['status'].value_counts()
    
    plt.figure(figsize=(6, 6))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title("Current Fleet Status Breakdown")
    return plt
