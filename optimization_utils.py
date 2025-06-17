"""
Utility functions for energy system optimization
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, Tuple, Optional


def optimize_energy_system(
    fixed_load: np.ndarray,
    solar_cf: np.ndarray,
    wind_cf: np.ndarray,
    flexible_ev_energy: float,
    flex_start: int,
    flex_end: int,
    solar_capital_cost: float,
    wind_capital_cost: float,
    storage_capital_cost: float,
    storage_duration: float,
    round_trip_efficiency: float
) -> Dict:
    """
    Optimize energy system capacities and operation using linear programming.
    
    Parameters:
    -----------
    fixed_load : np.ndarray
        Fixed hourly load profile (MW) for 24 hours
    solar_cf : np.ndarray
        Solar capacity factor profile (0-1) for 24 hours
    wind_cf : np.ndarray
        Wind capacity factor profile (0-1) for 24 hours
    flexible_ev_energy : float
        Total energy requirement for flexible EV charging (MWh)
    flex_start : int
        Start hour for flexible charging window (0-23)
    flex_end : int
        End hour for flexible charging window (0-23)
    solar_capital_cost : float
        Capital cost for solar ($/kW)
    wind_capital_cost : float
        Capital cost for wind ($/kW)
    storage_capital_cost : float
        Capital cost for storage ($/kWh)
    storage_duration : float
        Storage duration in hours
    round_trip_efficiency : float
        Storage round trip efficiency (%)
    
    Returns:
    --------
    dict : Optimization results containing:
        - success: bool
        - solar_capacity: float (MW)
        - wind_capacity: float (MW)
        - storage_capacity: float (MW)
        - solar_generation: np.ndarray (MW)
        - wind_generation: np.ndarray (MW)
        - storage_charge: np.ndarray (MW)
        - storage_discharge: np.ndarray (MW)
        - storage_soc: np.ndarray (MWh)
        - flex_ev_charging: np.ndarray (MW)
        - slack_generation: np.ndarray (MW)
        - objective_value: float ($)
        - message: str
    """
    T = 24  # Time periods (hours)
    
    # Validate inputs
    if len(fixed_load) != T or len(solar_cf) != T or len(wind_cf) != T:
        raise ValueError("All time series must have exactly 24 hours")
    
    if not (0 <= flex_start <= 23) or not (0 <= flex_end <= 23):
        raise ValueError("Flex start and end must be between 0 and 23")
    
    if round_trip_efficiency <= 0 or round_trip_efficiency > 100:
        raise ValueError("Round trip efficiency must be between 0 and 100")
    
    # Define variable indices
    idx = 0
    solar_cap_idx = idx; idx += 1
    wind_cap_idx = idx; idx += 1
    storage_cap_idx = idx; idx += 1
    storage_soc_idx = list(range(idx, idx + T)); idx += T
    solar_gen_idx = list(range(idx, idx + T)); idx += T
    wind_gen_idx = list(range(idx, idx + T)); idx += T
    storage_charge_idx = list(range(idx, idx + T)); idx += T
    storage_discharge_idx = list(range(idx, idx + T)); idx += T
    slack_idx = list(range(idx, idx + T)); idx += T
    
    # Add ramp variables for storage
    storage_ramp_up_idx = list(range(idx, idx + T-1)); idx += T-1
    storage_ramp_down_idx = list(range(idx, idx + T-1)); idx += T-1
    
    n_vars = idx
    if flexible_ev_energy > 0:
        flex_ev_idx = list(range(idx, idx + T)); idx += T
        # Add ramp variables for flexible EV
        ev_ramp_up_idx = list(range(idx, idx + T-1)); idx += T-1
        ev_ramp_down_idx = list(range(idx, idx + T-1)); idx += T-1
        n_vars = idx
    
    # Initialize coefficient vector (minimize cost)
    c = np.zeros(n_vars)
    
    # Set objective coefficients (annualized costs)
    c[solar_cap_idx] = solar_capital_cost * 0.1
    c[wind_cap_idx] = wind_capital_cost * 0.1
    c[storage_cap_idx] = storage_capital_cost * storage_duration * 0.1
    c[slack_idx] = 10000  # High penalty for slack
    
    # Add small ramp costs for storage (encourage smooth operation)
    ramp_cost = 0.1  # Small cost per MW ramping
    c[storage_ramp_up_idx] = ramp_cost
    c[storage_ramp_down_idx] = ramp_cost
    
    # Add small ramp costs for flexible EVs if they exist
    if flexible_ev_energy > 0:
        c[ev_ramp_up_idx] = ramp_cost
        c[ev_ramp_down_idx] = ramp_cost
    
    # Initialize constraint matrices
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    # Power balance constraints for each hour
    for t in range(T):
        row = np.zeros(n_vars)
        
        # Generation
        row[solar_gen_idx[t]] = 1
        row[wind_gen_idx[t]] = 1
        row[storage_discharge_idx[t]] = 1
        row[slack_idx[t]] = 1
        
        # Load
        row[storage_charge_idx[t]] = -1
        if flexible_ev_energy > 0:
            row[flex_ev_idx[t]] = -1
        
        A_eq.append(row)
        b_eq.append(fixed_load[t])
    
    # Generation capacity constraints
    for t in range(T):
        # Solar generation <= capacity * capacity factor
        row = np.zeros(n_vars)
        row[solar_gen_idx[t]] = 1
        row[solar_cap_idx] = -solar_cf[t]
        A_ub.append(row)
        b_ub.append(0)
        
        # Wind generation <= capacity * capacity factor
        row = np.zeros(n_vars)
        row[wind_gen_idx[t]] = 1
        row[wind_cap_idx] = -wind_cf[t]
        A_ub.append(row)
        b_ub.append(0)
    
    # Storage constraints
    eta = np.sqrt(round_trip_efficiency/100)
    for t in range(T):
        # Storage state evolution
        row = np.zeros(n_vars)
        if t == 0:
            row[storage_soc_idx[t]] = 1
            row[storage_charge_idx[t]] = -eta
            row[storage_discharge_idx[t]] = 1/eta
        else:
            row[storage_soc_idx[t]] = 1
            row[storage_soc_idx[t-1]] = -1
            row[storage_charge_idx[t]] = -eta
            row[storage_discharge_idx[t]] = 1/eta
        A_eq.append(row)
        b_eq.append(0)
        
        # Storage SOC <= capacity * duration
        row = np.zeros(n_vars)
        row[storage_soc_idx[t]] = 1
        row[storage_cap_idx] = -storage_duration
        A_ub.append(row)
        b_ub.append(0)
        
        # Charge/discharge <= capacity
        row = np.zeros(n_vars)
        row[storage_charge_idx[t]] = 1
        row[storage_cap_idx] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        row = np.zeros(n_vars)
        row[storage_discharge_idx[t]] = 1
        row[storage_cap_idx] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    # Flexible EV constraints
    if flexible_ev_energy > 0:
        # Total energy constraint
        row = np.zeros(n_vars)
        row[flex_ev_idx] = 1
        A_eq.append(row)
        b_eq.append(flexible_ev_energy)
        
        # Charging window constraints
        flex_profile = np.zeros(24)
        for t in range(24):
            if flex_start <= flex_end:
                if flex_start <= t <= flex_end:
                    flex_profile[t] = 1
            else:  # Handle overnight window
                if t >= flex_start or t <= flex_end:
                    flex_profile[t] = 1
        
        for t in range(T):
            if flex_profile[t] == 0:
                # No charging outside window
                row = np.zeros(n_vars)
                row[flex_ev_idx[t]] = 1
                A_eq.append(row)
                b_eq.append(0)
    
    # Storage ramping constraints
    for t in range(T-1):
        # Storage power change = discharge[t+1] - discharge[t] - (charge[t+1] - charge[t])
        # This equals ramp_up - ramp_down
        row = np.zeros(n_vars)
        row[storage_discharge_idx[t+1]] = 1
        row[storage_discharge_idx[t]] = -1
        row[storage_charge_idx[t+1]] = -1
        row[storage_charge_idx[t]] = 1
        row[storage_ramp_up_idx[t]] = -1
        row[storage_ramp_down_idx[t]] = 1
        A_eq.append(row)
        b_eq.append(0)
    
    # Flexible EV ramping constraints
    if flexible_ev_energy > 0:
        for t in range(T-1):
            # EV power change = ev[t+1] - ev[t] = ramp_up - ramp_down
            row = np.zeros(n_vars)
            row[flex_ev_idx[t+1]] = 1
            row[flex_ev_idx[t]] = -1
            row[ev_ramp_up_idx[t]] = -1
            row[ev_ramp_down_idx[t]] = 1
            A_eq.append(row)
            b_eq.append(0)
    
    # Variable bounds
    bounds = []
    bounds.append((0, None))  # solar capacity
    bounds.append((0, None))  # wind capacity
    bounds.append((0, None))  # storage capacity
    
    for _ in range(T):
        bounds.append((0, None))  # storage SOC
    for _ in range(T):
        bounds.append((0, None))  # solar generation
    for _ in range(T):
        bounds.append((0, None))  # wind generation
    for _ in range(T):
        bounds.append((0, None))  # storage charge
    for _ in range(T):
        bounds.append((0, None))  # storage discharge
    for _ in range(T):
        bounds.append((0, None))  # slack
    
    # Ramp variable bounds
    for _ in range(T-1):
        bounds.append((0, None))  # storage ramp up
    for _ in range(T-1):
        bounds.append((0, None))  # storage ramp down
    
    if flexible_ev_energy > 0:
        for _ in range(T):
            bounds.append((0, None))  # flex EV charge
        for _ in range(T-1):
            bounds.append((0, None))  # EV ramp up
        for _ in range(T-1):
            bounds.append((0, None))  # EV ramp down
    
    # Convert to arrays
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    # Solve linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                    bounds=bounds, method='highs')
    
    if result.success:
        # Extract results
        x = result.x
        solar_capacity = x[solar_cap_idx]
        wind_capacity = x[wind_cap_idx]
        storage_capacity = x[storage_cap_idx]
        storage_soc = np.array(x[storage_soc_idx])
        solar_generation = np.array(x[solar_gen_idx])
        wind_generation = np.array(x[wind_gen_idx])
        storage_charging = np.array(x[storage_charge_idx])
        storage_discharging = np.array(x[storage_discharge_idx])
        slack_generation = np.array(x[slack_idx])
        
        if flexible_ev_energy > 0:
            flex_ev_charging = np.array(x[flex_ev_idx])
        else:
            flex_ev_charging = np.zeros(T)
        
        return {
            'success': True,
            'solar_capacity': solar_capacity,
            'wind_capacity': wind_capacity,
            'storage_capacity': storage_capacity,
            'solar_generation': solar_generation,
            'wind_generation': wind_generation,
            'storage_charge': storage_charging,
            'storage_discharge': storage_discharging,
            'storage_soc': storage_soc,
            'flex_ev_charging': flex_ev_charging,
            'slack_generation': slack_generation,
            'objective_value': result.fun,
            'message': 'Optimization successful'
        }
    else:
        return {
            'success': False,
            'message': result.message,
            'objective_value': None
        }