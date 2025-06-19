import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pypsa
from optimization_utils import optimize_energy_system

def generate_household_load_profiles(num_households=20, hours=24):
    """
    Generate diverse synthetic household load profiles for a single day.
    
    Parameters:
    num_households (int): Number of household profiles to generate
    hours (int): Number of hours in the day (24)
    
    Returns:
    pd.DataFrame: DataFrame with time index and load profiles for each household
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create time index for 24 hours
    time_index = pd.date_range(start='2024-01-01 00:00', periods=hours, freq='H')
    
    # Initialize DataFrame
    df = pd.DataFrame(index=time_index)
    
    # Define different household archetypes
    archetypes = [
        # Working couple (low daytime, high evening)
        {'base': 0.3, 'morning': (7, 1.5, 1.5), 'evening': (19, 3.5, 2), 'midday': None, 'night_owl': False},
        
        # Family with kids (multiple peaks)
        {'base': 0.5, 'morning': (7, 2.5, 1.5), 'evening': (18, 4.0, 2.5), 'midday': (12, 1.5, 1), 'night_owl': False},
        
        # Work from home (consistent daytime use)
        {'base': 0.8, 'morning': (8, 1.5, 2), 'evening': (20, 2.0, 2), 'midday': None, 'night_owl': False},
        
        # Retired/elderly (early schedule)
        {'base': 0.4, 'morning': (6, 2.0, 1.5), 'evening': (17, 2.5, 2), 'midday': (13, 1.2, 1.5), 'night_owl': False},
        
        # Night shift worker (inverted schedule)
        {'base': 0.3, 'morning': (20, 2.5, 2), 'evening': (8, 3.0, 2), 'midday': None, 'night_owl': True},
        
        # Student/young adult (late schedule)
        {'base': 0.2, 'morning': (10, 1.0, 2), 'evening': (22, 3.0, 3), 'midday': None, 'night_owl': True},
        
        # Energy efficient home (low overall)
        {'base': 0.15, 'morning': (7, 0.8, 1), 'evening': (19, 1.5, 1.5), 'midday': None, 'night_owl': False},
        
        # High consumption home (pool, AC, etc)
        {'base': 1.2, 'morning': (8, 2.0, 2), 'evening': (18, 4.5, 2), 'midday': (14, 2.5, 3), 'night_owl': False}
    ]
    
    # Generate load profiles for each household
    for i in range(num_households):
        # Select archetype with some randomization
        archetype = archetypes[i % len(archetypes)]
        
        # Add randomization to base parameters
        base_load = archetype['base'] * np.random.uniform(0.8, 1.2)
        
        # Generate hourly load values
        load_profile = np.ones(hours) * base_load
        
        # Add morning peak
        if archetype['morning']:
            m_time, m_mag, m_width = archetype['morning']
            m_time += np.random.uniform(-1, 1)
            m_mag *= np.random.uniform(0.7, 1.3)
            for hour in range(hours):
                load_profile[hour] += m_mag * np.exp(-0.5 * ((hour - m_time) / m_width) ** 2)
        
        # Add evening peak
        if archetype['evening']:
            e_time, e_mag, e_width = archetype['evening']
            e_time += np.random.uniform(-1, 1)
            e_mag *= np.random.uniform(0.7, 1.3)
            for hour in range(hours):
                load_profile[hour] += e_mag * np.exp(-0.5 * ((hour - e_time) / e_width) ** 2)
        
        # Add midday peak if present
        if archetype['midday']:
            mid_time, mid_mag, mid_width = archetype['midday']
            mid_time += np.random.uniform(-0.5, 0.5)
            mid_mag *= np.random.uniform(0.8, 1.2)
            for hour in range(hours):
                load_profile[hour] += mid_mag * np.exp(-0.5 * ((hour - mid_time) / mid_width) ** 2)
        
        # Night owl behavior (higher nighttime baseline)
        if archetype['night_owl']:
            for hour in [22, 23, 0, 1, 2]:
                load_profile[hour] += base_load * np.random.uniform(0.3, 0.6)
        
        # Add time-correlated noise
        noise = np.random.normal(0, 0.05, hours)
        for j in range(1, hours):
            noise[j] = 0.7 * noise[j-1] + 0.3 * noise[j]
        load_profile += noise * base_load
        
        # Add occasional spikes (appliance use)
        for _ in range(np.random.randint(0, 3)):
            spike_hour = np.random.randint(0, hours)
            spike_magnitude = np.random.uniform(0.5, 2.0)
            spike_duration = np.random.randint(1, 3)
            for h in range(max(0, spike_hour), min(hours, spike_hour + spike_duration)):
                load_profile[h] += spike_magnitude
        
        # Ensure non-negative
        load_profile = np.maximum(load_profile, 0)
        
        df[f'Household_{i+1}'] = load_profile
    
    return df

# Streamlit app
st.title('Households to the energy system')

st.markdown("A simple demonstration of the current practices for modelling household behaviour.")

st.markdown("Deepening a shared understanding of current modelling processes.")

st.markdown("An opportunity to reflect on the possible 'intervention points' for SFL.")

# Section 1: Exogenous Energy Demand
st.header('Section 1: Exogenous Energy Demand')

st.markdown('Exogenous to the optimisation model / determined prior to and as an input for the optimisation model.')

# Generate data
def load_data():
    return generate_household_load_profiles(num_households=20, hours=24)

df = load_data()

# Plot individual household load profiles
st.subheader('1.1 Individual Household Load Profiles')

fig1, ax1 = plt.subplots(figsize=(12, 6))

for column in df.columns:
    ax1.plot(df.index.hour, df[column], alpha=0.5, linewidth=1)

ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Load (kW)')
ax1.set_title('20 Household Load Profiles')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 23)
ax1.set_xticks(range(0, 24, 2))

st.pyplot(fig1)

# Calculate and plot aggregate load profile
st.subheader('1.2 Aggregate Load Profile')

df['Aggregate'] = df.sum(axis=1)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.index.hour, df['Aggregate'], color='darkblue', linewidth=2.5)
ax2.fill_between(df.index.hour, df['Aggregate'], alpha=0.3, color='darkblue')

ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Total Load (kW)')
ax2.set_title('Aggregate Load Profile (Sum of 20 Households)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 23)
ax2.set_xticks(range(0, 24, 2))

st.pyplot(fig2)

# EV Charging Section
st.subheader('1.3 New technologies: EV Example')

# EV parameters widgets
col1, col2 = st.columns(2)

with col1:
    num_ev_households = st.slider(
        'Number of Households with EVs',
        min_value=0,
        max_value=20,
        value=10,
        help='Select how many of the 20 households have EVs'
    )
    
    ev_energy_per_day = st.slider(
        'EV Energy Use per Day (kWh)',
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        help='Average daily energy consumption per EV'
    )

with col2:
    charging_start = st.slider(
        'Charging Window Start (Hour)',
        min_value=0,
        max_value=23,
        value=18,
        help='Default start time for EV charging'
    )
    
    charging_end = st.slider(
        'Charging Window End (Hour)',
        min_value=0,
        max_value=23,
        value=6,
        help='Default end time for EV charging (can be next day)'
    )
    
    charge_time_std = st.slider(
        'Charge Start Time Std Dev (Hours)',
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.5,
        help='Standard deviation for randomizing charge start times'
    )

# Add max charge rate slider below the columns
max_charge_rate = st.slider(
    'Maximum Charge Rate (kW)',
    min_value=3.0,
    max_value=22.0,
    value=7.0,
    step=0.5,
    help='Maximum charging power per EV (3kW = slow, 7kW = standard home, 11-22kW = fast)'
)

def generate_ev_charging_profiles(num_evs, energy_per_ev, start_hour, end_hour, std_dev, max_rate, hours=24):
    """
    Generate EV charging profiles based on parameters.
    
    Returns:
    pd.DataFrame: DataFrame with EV charging profiles
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create time index
    time_index = pd.date_range(start='2024-01-01 00:00', periods=hours, freq='H')
    ev_df = pd.DataFrame(index=time_index)
    
    # Calculate charging duration needed
    charger_power = max_rate  # Use the provided max charge rate
    charge_duration = energy_per_ev / charger_power
    
    # Handle overnight charging window
    if end_hour < start_hour:
        available_hours = (24 - start_hour) + end_hour
    else:
        available_hours = end_hour - start_hour
    
    for i in range(num_evs):
        # Randomize start time around the default start time
        actual_start = np.random.normal(start_hour, std_dev)
        actual_start = int(np.clip(actual_start, 0, 23))
        
        # Generate charging profile
        charging_profile = np.zeros(hours)
        
        # Calculate hours needed for charging
        hours_needed = int(np.ceil(charge_duration))
        
        # Fill in charging hours
        for h in range(hours_needed):
            hour_idx = (actual_start + h) % 24
            if hours_needed > 0:
                # Last hour might be partial
                if h == hours_needed - 1:
                    remaining_energy = energy_per_ev - (h * charger_power)
                    charging_profile[hour_idx] = min(remaining_energy, charger_power)
                else:
                    charging_profile[hour_idx] = charger_power
        
        ev_df[f'EV_{i+1}'] = charging_profile
    
    return ev_df

# Generate EV charging profiles
if num_ev_households > 0:
    ev_profiles = generate_ev_charging_profiles(
        num_ev_households,
        ev_energy_per_day,
        charging_start,
        charging_end,
        charge_time_std,
        max_charge_rate
    )
    
    # Plot individual EV charging profiles
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # Highlight charging window
    if charging_end < charging_start:  # Overnight window
        ax3.axvspan(charging_start, 24, alpha=0.2, color='lightgreen', label='Charging Window')
        ax3.axvspan(0, charging_end, alpha=0.2, color='lightgreen')
    else:
        ax3.axvspan(charging_start, charging_end, alpha=0.2, color='lightgreen', label='Charging Window')
    
    for column in ev_profiles.columns:
        ax3.plot(ev_profiles.index.hour, ev_profiles[column], alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Charging Power (kW)')
    ax3.set_title(f'{num_ev_households} EV Charging Profiles')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_ylim(0, max(8, max_charge_rate * 1.1))
    ax3.legend()
    
    st.pyplot(fig3)
    
    # Calculate total demand including EVs
    ev_profiles['Total_EV_Load'] = ev_profiles.sum(axis=1)
    df['Total_with_EVs'] = df['Aggregate'] + ev_profiles['Total_EV_Load']
    
    # Plot aggregate demand with EVs
    st.subheader('1.3.2 Total Demand Including EV Charging')
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    # Highlight charging window
    if charging_end < charging_start:  # Overnight window
        ax4.axvspan(charging_start, 24, alpha=0.1, color='lightgreen', label='Charging Window')
        ax4.axvspan(0, charging_end, alpha=0.1, color='lightgreen')
    else:
        ax4.axvspan(charging_start, charging_end, alpha=0.1, color='lightgreen', label='Charging Window')
    
    # Plot original aggregate
    ax4.plot(df.index.hour, df['Aggregate'], 
             color='darkblue', linewidth=2.5, label='Base Load', alpha=0.7)
    ax4.fill_between(df.index.hour, df['Aggregate'], 
                     alpha=0.2, color='darkblue')
    
    # Plot EV load
    ax4.plot(ev_profiles.index.hour, ev_profiles['Total_EV_Load'], 
             color='green', linewidth=2.5, label='EV Load', alpha=0.7)
    ax4.fill_between(ev_profiles.index.hour, df['Aggregate'], 
                     df['Total_with_EVs'], alpha=0.3, color='green')
    
    # Plot total with EVs
    ax4.plot(df.index.hour, df['Total_with_EVs'], 
             color='red', linewidth=3, label='Total Load', linestyle='--')
    
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Load (kW)')
    ax4.set_title('Aggregate Load Profile with EV Charging')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 23)
    ax4.set_xticks(range(0, 24, 2))
    ax4.legend()
    
    st.pyplot(fig4)
else:
    st.info('Select number of households with EVs to see charging analysis')

# Section 2: Endogenous Energy Demand
st.header('Section 2: Endogenous Energy Demand')

st.markdown('Endogenous to the optimisation model / determined as an output of the optimisation model.')

# Flexible EV Charging
st.subheader('2.1 Example: Flexible EV Charging')

# Flexible charging parameters
st.write('Configure the flexible charging window to optimize EV charging times')

flex_start = st.slider(
    'Flexible Charging Start Time',
    min_value=0,
    max_value=23,
    value=18,
    key='flex_start',
    help='Earliest time EVs can start charging in flexible mode'
)

flex_end = st.slider(
    'Flexible Charging End Time',
    min_value=0,
    max_value=23,
    value=6,
    key='flex_end',
    help='Latest time EVs must finish charging in flexible mode'
)

flex_adoption = st.slider(
    'Flexible Charging Adoption Rate (%)',
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    key='flex_adoption',
    help='Percentage of EVs that participate in flexible charging'
)

# Visual representation of flexible charging window
fig_flex_window, ax_flex_window = plt.subplots(figsize=(10, 2))

# Create a 24-hour timeline
hours = list(range(24))
timeline = [0] * 24

# Mark the charging window
if flex_end < flex_start:  # Overnight window
    for h in hours:
        if h >= flex_start or h <= flex_end:
            timeline[h] = 1
else:
    for h in hours:
        if flex_start <= h <= flex_end:
            timeline[h] = 1

# Plot the timeline
ax_flex_window.bar(hours, timeline, width=1, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
ax_flex_window.set_xlim(-0.5, 23.5)
ax_flex_window.set_ylim(0, 1.2)
ax_flex_window.set_xticks(range(0, 24, 2))
ax_flex_window.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax_flex_window.set_yticks([])
ax_flex_window.set_xlabel('Hour of Day')
ax_flex_window.set_title('Flexible Charging Window')

# Add start and end markers
ax_flex_window.axvline(flex_start, color='darkgreen', linestyle='--', alpha=0.8, linewidth=2)
ax_flex_window.axvline(flex_end, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
ax_flex_window.text(flex_start, 1.1, 'Start', ha='center', va='bottom', color='darkgreen', fontweight='bold')
ax_flex_window.text(flex_end, 1.1, 'End', ha='center', va='bottom', color='darkred', fontweight='bold')

st.pyplot(fig_flex_window)

# Section 3: Large Scale Generation
st.header('Section 3: Large Scale Generation')

# Large Scale Solar Resource
st.subheader('3.1 Large Scale Solar Resource')

st.markdown("The cost and availability of solar energy is defined with the final capacity determined at model runtime.")

def generate_solar_capacity_factor(hours=24):
    """
    Generate a realistic solar capacity factor profile for a single day.
    
    Returns:
    np.array: Hourly capacity factors (0-1)
    """
    hours_array = np.arange(hours)
    
    # Solar parameters
    sunrise = 6.0  # 6 AM
    sunset = 18.0  # 6 PM
    peak_hour = 12.0  # Solar noon
    
    # Initialize capacity factor array
    capacity_factor = np.zeros(hours)
    
    # Generate solar profile for daylight hours
    for hour in hours_array:
        if sunrise <= hour <= sunset:
            # Use a cosine function to model solar intensity
            # Maximum at solar noon, zero at sunrise/sunset
            angle = np.pi * (hour - sunrise) / (sunset - sunrise)
            base_cf = np.sin(angle)
            
            # Add some cloud variability
            cloud_factor = 0.9 + 0.1 * np.sin(hour * 2)  # Small variations
            
            capacity_factor[hour] = base_cf * cloud_factor * 0.85  # Max ~85% capacity factor
    
    return capacity_factor

# Generate solar capacity factor data
solar_cf = generate_solar_capacity_factor()
time_index = pd.date_range(start='2024-01-01 00:00', periods=24, freq='H')

# Create solar generation chart
fig_solar, ax_solar = plt.subplots(figsize=(12, 6))

# Plot capacity factor as area chart
ax_solar.fill_between(range(24), solar_cf, alpha=0.3, color='orange', label='Solar Capacity')
ax_solar.plot(range(24), solar_cf, color='darkorange', linewidth=2.5)

# Add markers for sunrise and sunset
ax_solar.axvline(x=6, color='gray', linestyle='--', alpha=0.5, label='Sunrise')
ax_solar.axvline(x=18, color='gray', linestyle='--', alpha=0.5, label='Sunset')

# Formatting
ax_solar.set_xlabel('Hour of Day')
ax_solar.set_ylabel('Capacity Factor')
ax_solar.set_title('Solar Generator Capacity Factor Profile')
ax_solar.grid(True, alpha=0.3)
ax_solar.set_xlim(0, 23)
ax_solar.set_ylim(0, 1)
ax_solar.set_xticks(range(0, 24, 2))
ax_solar.legend()

# Add percentage labels on y-axis
ax_solar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))

st.pyplot(fig_solar)

# Solar cost parameters
solar_capital_cost = st.slider(
    'Solar Capital Cost ($/kW)',
    min_value=500,
    max_value=2000,
    value=750,
    step=50,
    help='Capital cost per kW of solar capacity'
)

# Large Scale Wind Resource
st.subheader('3.2 Large Scale Wind Resource')

st.markdown("The cost and availability of wind energy is defined with the final capacity determined at model runtime.")

def generate_wind_capacity_factor(hours=24):
    """
    Generate a flatter wind capacity factor profile for a single day.
    This encourages solar deployment by making wind less variable.
    
    Returns:
    np.array: Hourly capacity factors (0-1)
    """
    np.random.seed(123)  # Different seed for wind variability
    
    hours_array = np.arange(hours)
    
    # Much flatter base wind pattern - consistent throughout the day
    base_cf = 0.35  # Consistent base capacity factor
    
    # Very slight variation to maintain some realism
    base_pattern = np.ones(hours) * base_cf
    
    # Add minimal hour-to-hour variation (±5%)
    for hour in hours_array:
        base_pattern[hour] += 0.05 * np.sin(hour * np.pi / 12)
    
    # Add very small random variability
    variability = np.random.normal(0, 0.02, hours)  # Reduced from 0.08 to 0.02
    
    # Add some correlation between consecutive hours
    for i in range(1, hours):
        variability[i] = 0.9 * variability[i-1] + 0.1 * variability[i]  # Increased correlation
    
    # Combine base pattern with variability
    capacity_factor = base_pattern + variability
    
    # Clip to realistic bounds (0.2 to 0.5) - narrower range for flatter profile
    capacity_factor = np.clip(capacity_factor, 0.2, 0.5)
    
    return capacity_factor

# Generate wind capacity factor data
wind_cf = generate_wind_capacity_factor()

# Create wind generation chart
fig_wind, ax_wind = plt.subplots(figsize=(12, 6))

# Plot capacity factor as area chart
ax_wind.fill_between(range(24), wind_cf, alpha=0.3, color='skyblue', label='Wind Capacity')
ax_wind.plot(range(24), wind_cf, color='darkblue', linewidth=2.5)

# Add average line
avg_wind = np.mean(wind_cf)
ax_wind.axhline(y=avg_wind, color='red', linestyle='--', alpha=0.5, label=f'Daily Average ({avg_wind:.1%})')

# Formatting
ax_wind.set_xlabel('Hour of Day')
ax_wind.set_ylabel('Capacity Factor')
ax_wind.set_title('Wind Generator Capacity Factor Profile')
ax_wind.grid(True, alpha=0.3)
ax_wind.set_xlim(0, 23)
ax_wind.set_ylim(0, 1)
ax_wind.set_xticks(range(0, 24, 2))
ax_wind.legend()

# Add percentage labels on y-axis
ax_wind.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))

st.pyplot(fig_wind)

# Wind cost parameters
wind_capital_cost = st.slider(
    'Wind Capital Cost ($/kW)',
    min_value=1000,
    max_value=3000,
    value=1500,
    step=50,
    help='Capital cost per kW of wind capacity'
)

# Utility Scale Storage
st.subheader('3.3 Utility Scale Storage')

st.markdown("The characteristics of energy storage is defined as an input, with the final installed capacity and "
            "operational profile determined at runtime.")

st.write('Configure parameters for grid-scale battery energy storage system')

storage_duration = st.slider(
    'Storage Duration (Hours)',
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help='Hours of storage at rated power output'
)

round_trip_efficiency = st.slider(
    'Round Trip Efficiency (%)',
    min_value=70,
    max_value=95,
    value=85,
    step=1,
    help='Efficiency of charge/discharge cycle'
)

storage_capital_cost = st.slider(
    'Storage Capital Cost ($/kWh)',
    min_value=100,
    max_value=500,
    value=250,
    step=10,
    help='Capital cost per kWh of storage capacity'
)

# Section 4: System Optimisation
st.header('Section 4: System Optimisation')

st.write('Click the button below to run the system optimization using linear programming.')

if st.button('Run Optimization', type='primary'):
    with st.spinner('Running optimization...'):
        # Calculate loads and convert from kW to MW
        base_load = df['Aggregate'].values / 1000  # Convert kW to MW
        
        # Add inflexible EV load if EVs exist
        if num_ev_households > 0:
            inflexible_ev_ratio = 1 - (flex_adoption / 100)
            inflexible_ev_load = ev_profiles['Total_EV_Load'].values * inflexible_ev_ratio / 1000  # Convert kW to MW
            fixed_load = base_load + inflexible_ev_load
            
            # Calculate flexible EV energy requirement (convert kWh to MWh)
            flexible_ev_energy = ev_profiles['Total_EV_Load'].sum() * (flex_adoption / 100) / 1000
        else:
            fixed_load = base_load
            flexible_ev_energy = 0
        
        # Run optimization using utility function
        result = optimize_energy_system(
            fixed_load=fixed_load,
            solar_cf=solar_cf,
            wind_cf=wind_cf,
            flexible_ev_energy=flexible_ev_energy,
            flex_start=flex_start,
            flex_end=flex_end,
            solar_capital_cost=solar_capital_cost,
            wind_capital_cost=wind_capital_cost,
            storage_capital_cost=storage_capital_cost,
            storage_duration=storage_duration,
            round_trip_efficiency=round_trip_efficiency
        )
        
        if result['success']:
            st.success('Optimization completed successfully!')
            
            # Extract results from utility function
            solar_capacity = result['solar_capacity']
            wind_capacity = result['wind_capacity']
            storage_capacity = result['storage_capacity']
            solar_generation = result['solar_generation']
            wind_generation = result['wind_generation']
            storage_charging = result['storage_charge']
            storage_discharging = result['storage_discharge']
            storage_soc = result['storage_soc']
            slack_generation = result['slack_generation']
            flex_ev_charging = result['flex_ev_charging']
            
            # Create network object to store results (for compatibility with plotting code)
            time_index = pd.date_range(start='2024-01-01 00:00', periods=24, freq='H')
            network = type('obj', (object,), {
                'generators': pd.DataFrame({
                    'p_nom_opt': [solar_capacity, wind_capacity]
                }, index=['solar', 'wind']),
                'storage_units': pd.DataFrame({
                    'p_nom_opt': [storage_capacity]
                }, index=['battery']),
                'generators_t': type('obj', (object,), {
                    'p': pd.DataFrame({
                        'solar': solar_generation,
                        'wind': wind_generation,
                        'slack': slack_generation
                    })
                })(),
                'storage_units_t': type('obj', (object,), {
                    'p': pd.DataFrame({'battery': storage_discharging - storage_charging}),
                    'state_of_charge': pd.DataFrame({'battery': storage_soc}),
                    'p_store': pd.DataFrame({'battery': storage_charging})
                })(),
                'objective': result['objective_value'],
                'flex_ev_charging': flex_ev_charging,
                'snapshots': time_index
            })
            
        else:
            st.error('Optimization failed!')
            st.write(f"Status: {result['message']}")

        # Chart 1a: Demand Profile (excluding storage)
        st.subheader('4.1a Demand Profile')
        
        fig_demand, ax_demand = plt.subplots(figsize=(12, 6))
        
        # Highlight flexible charging window if flexible EVs exist
        if flexible_ev_energy > 0:
            if flex_end < flex_start:  # Overnight window
                ax_demand.axvspan(flex_start, 24, alpha=0.1, color='lightgreen', label='Flexible Charging Window')
                ax_demand.axvspan(0, flex_end, alpha=0.1, color='lightgreen')
            else:
                ax_demand.axvspan(flex_start, flex_end, alpha=0.1, color='lightgreen', label='Flexible Charging Window')
        
        # Plot base household load (convert back to kW for display)
        ax_demand.fill_between(range(24), 0, base_load * 1000, 
                              label='Household Load', color='lightblue', alpha=0.7)
        
        # Plot inflexible EV demand if applicable
        if num_ev_households > 0:
            inflexible_ev_load_kw = inflexible_ev_load * 1000
            ax_demand.fill_between(range(24), base_load * 1000, (base_load + inflexible_ev_load) * 1000,
                                 label='Inflexible EV Charging', color='purple', alpha=0.5)
            current_top = (base_load + inflexible_ev_load) * 1000
        else:
            current_top = base_load * 1000
        
        # Plot flexible EV on top if applicable
        if flexible_ev_energy > 0:
            flex_ev_charging_kw = network.flex_ev_charging * 1000
            total_load_kw = current_top + flex_ev_charging_kw
            ax_demand.fill_between(range(24), current_top, total_load_kw,
                                 label='Flexible EV Charging', color='red', alpha=0.5)
            ax_demand.plot(range(24), total_load_kw, 'darkred', linewidth=2, 
                         label='Total Demand')
        else:
            ax_demand.plot(range(24), current_top, 'darkblue', linewidth=2, 
                         label='Total Demand')
        
        ax_demand.set_xlabel('Hour of Day')
        ax_demand.set_ylabel('Demand (kW)')
        ax_demand.set_title('Electricity Demand Profile (Excluding Storage)')
        ax_demand.legend()
        ax_demand.grid(True, alpha=0.3)
        ax_demand.set_xlim(0, 23)
        ax_demand.set_xticks(range(0, 24, 2))
        
        st.pyplot(fig_demand)
        
        # Chart 1b: Supply and Demand Balance
        st.subheader('4.1b Supply and Demand Balance')
        
        fig_balance, ax_balance = plt.subplots(figsize=(12, 6))
        
        # Plot generation (stacked area) - convert to kW
        ax_balance.fill_between(range(24), 0, 
                               network.generators_t.p["solar"].values * 1000,
                               label='Solar', color='orange', alpha=0.7)
        
        wind_bottom = network.generators_t.p["solar"].values * 1000
        ax_balance.fill_between(range(24), wind_bottom,
                               wind_bottom + network.generators_t.p["wind"].values * 1000,
                               label='Wind', color='skyblue', alpha=0.7)
        
        # Add storage discharge to supply
        storage_power = network.storage_units_t.p["battery"].values * 1000
        storage_discharge = np.maximum(storage_power, 0)
        storage_charge = np.minimum(storage_power, 0)  # Negative values
        
        total_renewable = (network.generators_t.p["solar"].values + network.generators_t.p["wind"].values) * 1000
        ax_balance.fill_between(range(24), total_renewable,
                               total_renewable + storage_discharge,
                               label='Storage Discharge', color='green', alpha=0.7)
        
        # Plot storage charging as negative (below zero)
        ax_balance.fill_between(range(24), 0, storage_charge,
                               label='Storage Charging', color='darkgreen', alpha=0.5)
        
        # Plot total demand as a single line (without storage)
        if flexible_ev_energy > 0:
            total_demand = (fixed_load + network.flex_ev_charging) * 1000
        else:
            total_demand = fixed_load * 1000
        
        ax_balance.plot(range(24), total_demand, 'k-', linewidth=3, 
                       label='Total Demand')
        
        # Add horizontal line at zero for reference
        ax_balance.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax_balance.set_xlabel('Hour of Day')
        ax_balance.set_ylabel('Power (kW)')
        ax_balance.set_title('Supply and Demand Balance')
        ax_balance.legend()
        ax_balance.grid(True, alpha=0.3)
        ax_balance.set_xlim(0, 23)
        ax_balance.set_xticks(range(0, 24, 2))
        
        st.pyplot(fig_balance)
        
        # Chart 2: Optimal Capacities
        st.subheader('4.2 Optimal System Capacities')
        
        fig_cap, ax_cap = plt.subplots(figsize=(10, 6))
        
        # Convert MW to kW for display
        capacities = {
            'Solar': network.generators.loc['solar', 'p_nom_opt'] * 1000,
            'Wind': network.generators.loc['wind', 'p_nom_opt'] * 1000,
            'Storage': network.storage_units.loc['battery', 'p_nom_opt'] * 1000
        }
        
        colors = ['orange', 'skyblue', 'green']
        bars = ax_cap.bar(range(len(capacities)), list(capacities.values()), color=colors)
        ax_cap.set_xticks(range(len(capacities)))
        ax_cap.set_xticklabels(list(capacities.keys()))
        ax_cap.set_ylabel('Capacity (kW)')
        ax_cap.set_title('Optimal Generation and Storage Capacities')
        
        # Add value labels on bars
        for bar, value in zip(bars, capacities.values()):
            height = bar.get_height()
            ax_cap.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}',
                       ha='center', va='bottom')
        
        ax_cap.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig_cap)
        
        # Chart 3: System Costs
        st.subheader('4.3 Total System Cost')
        
        # Calculate costs
        solar_cost = network.generators.loc['solar', 'p_nom_opt'] * solar_capital_cost
        wind_cost = network.generators.loc['wind', 'p_nom_opt'] * wind_capital_cost
        storage_cost = network.storage_units.loc['battery', 'p_nom_opt'] * storage_capital_cost * storage_duration
        total_cost = solar_cost + wind_cost + storage_cost
        
        # Annualize costs (assuming 20-year lifetime, 5% discount rate)
        annualization_factor = 0.08
        annual_costs = {
            'Solar': solar_cost * annualization_factor,
            'Wind': wind_cost * annualization_factor,
            'Storage': storage_cost * annualization_factor
        }
        
        fig_cost, (ax_cost1, ax_cost2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Pie chart of cost breakdown
        colors = ['orange', 'skyblue', 'green']
        ax_cost1.pie(annual_costs.values(), labels=annual_costs.keys(), colors=colors,
                    autopct='%1.1f%%', startangle=90)
        ax_cost1.set_title('Annual Cost Breakdown')
        
        # Bar chart of costs
        ax_cost2.bar(range(len(annual_costs)), list(annual_costs.values()), color=colors)
        ax_cost2.set_xticks(range(len(annual_costs)))
        ax_cost2.set_xticklabels(list(annual_costs.keys()))
        ax_cost2.set_ylabel('Annual Cost ($)')
        ax_cost2.set_title('Annual System Costs by Component')
        ax_cost2.grid(True, alpha=0.3, axis='y')
        
        # Add total cost annotation
        total_annual = sum(annual_costs.values())
        ax_cost2.axhline(y=total_annual, color='red', linestyle='--', alpha=0.7)
        ax_cost2.text(1, total_annual * 1.05, f'Total: ${total_annual:,.0f}/year',
                     ha='center', color='red', fontweight='bold')
        
        st.pyplot(fig_cost)
        
        # Display key metrics
        st.subheader('Key Optimization Results')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Total Annual Cost', f'${total_annual:,.0f}')
        with col2:
            # Convert MWh to kWh for cost per kWh
            total_energy_mwh = fixed_load.sum() + flexible_ev_energy
            total_energy_kwh = total_energy_mwh * 1000
            cost_per_kwh = total_annual / total_energy_kwh
            st.metric('Annual Cost per kWh', f'${cost_per_kwh:.3f}')
        
