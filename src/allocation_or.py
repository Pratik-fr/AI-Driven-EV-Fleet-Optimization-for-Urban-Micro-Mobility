import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def get_zone_distance_matrix():
    """
    Returns a mock distance matrix (km) between zones.
    Rows: From Zone, Cols: To Zone
    Order: Zone_A, Zone_B, Zone_C, Zone_D, Zone_E
    """
    #        A   B   C   D   E
    dists = [
        [0,  5,  12, 8,  15], # A
        [5,  0,  6,  9,  11], # B
        [12, 6,  0,  4,  7],  # C
        [8,  9,  4,  0,  5],  # D
        [15, 11, 7,  5,  0]   # E
    ]
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    return pd.DataFrame(dists, index=zones, columns=zones)

def optimize_fleet_allocation(current_fleet, predicted_demand):
    """
    Allocates available EVs to zones with demand shortage.
    
    current_fleet: DataFrame with 'vehicle_id', 'current_zone', 'status'
    predicted_demand: Dict or Series {zone: demand_count}
    """
    
    # 1. Identify Supply (Available EVs)
    available_evs = current_fleet[current_fleet['status'] == 'Available'].copy()
    
    if available_evs.empty:
        return "No available EVs to rebalance."
    
    # 2. Identify Demand (Shortage)
    # Calculate current supply per zone
    current_supply = current_fleet[current_fleet['status'] == 'Available']['current_zone'].value_counts()
    
    shortages = [] # list of (zone, needed_count)
    
    for zone, demand in predicted_demand.items():
        # simplified check: if projected demand > current available
        # In reality, we'd check net flow, but here:
        curr = current_supply.get(zone, 0)
        gap = max(0, int(demand) - curr)
        for _ in range(gap):
            shortages.append(zone)
            
    if not shortages:
        return "No projected shortages. No rebalancing needed."
    
    # 3. Construct Cost Matrix
    # Rows: Available EVs (Supply)
    # Cols: Shortage Slots (Demand)
    
    dist_matrix = get_zone_distance_matrix()
    cost_matrix = np.zeros((len(available_evs), len(shortages)))
    
    ev_list = available_evs.to_dict('records') # [{'vehicle_id':.., 'current_zone':..}, ...]
    
    for i, ev in enumerate(ev_list):
        from_zone = ev['current_zone']
        for j, to_zone in enumerate(shortages):
            # Cost = Distance
            cost = dist_matrix.loc[from_zone, to_zone]
            cost_matrix[i, j] = cost
            
    # 4. Solve Assignment Problem (Hungarian Algorithm)
    # linear_sum_assignment finds min cost matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 5. Format Results
    allocations = []
    total_cost = 0
    
    for row, col in zip(row_ind, col_ind):
        ev = ev_list[row]
        target_zone = shortages[col]
        cost = cost_matrix[row, col]
        
        # Only assign if moving to a different zone
        if ev['current_zone'] != target_zone:
            allocations.append({
                'vehicle_id': ev['vehicle_id'],
                'from_zone': ev['current_zone'],
                'to_zone': target_zone,
                'cost': cost
            })
            total_cost += cost
            
    result_df = pd.DataFrame(allocations)
    return result_df, total_cost
