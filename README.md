# Energy System Optimization Demo

This Streamlit app demonstrates energy system optimization with flexible EV charging, renewable generation, and energy storage.

## Features

- **Household Load Profiles**: Generates diverse synthetic load profiles for 20 households
- **EV Charging Analysis**: Models electric vehicle charging with configurable parameters
- **Flexible Charging**: Optimizes EV charging within time windows
- **Renewable Generation**: Models solar and wind capacity factors
- **Energy Storage**: Includes battery storage with configurable parameters
- **System Optimization**: Uses linear programming to find optimal capacities

## Live Demo

[View on Streamlit Community Cloud](https://your-app-url.streamlit.app)

## Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cem-demo.git
cd cem-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Section 1**: View and configure household load profiles and EV charging
2. **Section 2**: Set flexible EV charging parameters
3. **Section 3**: Configure renewable generation and storage costs
4. **Section 4**: Run optimization to find optimal system capacities

## Technologies

- Streamlit for the web interface
- NumPy and Pandas for data manipulation
- Matplotlib for visualizations
- SciPy for linear programming optimization
- PyPSA for power system analysis components