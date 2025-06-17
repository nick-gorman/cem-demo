"""
Unit tests for energy system optimization
"""
import unittest
import numpy as np
from optimization_utils import optimize_energy_system


class TestOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up common test data"""
        # Base case parameters
        self.fixed_load = np.ones(24) * 50  # 50 MW constant load
        
        # Solar CF: zero at night, peak at noon
        self.solar_cf = np.zeros(24)
        for h in range(6, 18):  # 6 AM to 6 PM
            self.solar_cf[h] = np.sin((h - 6) * np.pi / 12) * 0.8
        
        # Wind CF: variable with higher output at night
        np.random.seed(42)
        self.wind_cf = 0.3 + 0.2 * np.sin(np.arange(24) * np.pi / 12) + 0.1 * np.random.randn(24)
        self.wind_cf = np.clip(self.wind_cf, 0, 0.9)
        
        # Cost parameters
        self.solar_cost = 1000  # $/kW
        self.wind_cost = 1500   # $/kW
        self.storage_cost = 250  # $/kWh
        
        # Storage parameters
        self.storage_duration = 4  # hours
        self.efficiency = 85  # %
        
        # EV parameters
        self.flex_ev_energy = 100  # MWh
        self.flex_start = 22  # 10 PM
        self.flex_end = 6     # 6 AM (overnight)
    
    def test_basic_optimization(self):
        """Test basic optimization with all components"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(result['solar_capacity'], 0)
        self.assertGreater(result['wind_capacity'], 0)
        self.assertGreater(result['storage_capacity'], 0)
    
    def test_power_balance(self):
        """Test that power balance is maintained at each hour"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        for t in range(24):
            generation = (result['solar_generation'][t] + 
                         result['wind_generation'][t] + 
                         result['storage_discharge'][t] +
                         result['slack_generation'][t])
            
            load = (self.fixed_load[t] + 
                   result['storage_charge'][t] + 
                   result['flex_ev_charging'][t])
            
            self.assertAlmostEqual(generation, load, places=5,
                msg=f"Power balance not maintained at hour {t}")
    
    def test_capacity_constraints(self):
        """Test that generation doesn't exceed capacity limits"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        for t in range(24):
            # Solar constraint
            self.assertLessEqual(
                result['solar_generation'][t],
                result['solar_capacity'] * self.solar_cf[t] + 1e-6
            )
            
            # Wind constraint
            self.assertLessEqual(
                result['wind_generation'][t],
                result['wind_capacity'] * self.wind_cf[t] + 1e-6
            )
            
            # Storage constraints
            self.assertLessEqual(
                result['storage_charge'][t],
                result['storage_capacity'] + 1e-6
            )
            self.assertLessEqual(
                result['storage_discharge'][t],
                result['storage_capacity'] + 1e-6
            )
    
    def test_storage_energy_balance(self):
        """Test storage state of charge evolution"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        eta = np.sqrt(self.efficiency / 100)
        
        for t in range(24):
            if t == 0:
                expected_soc = (eta * result['storage_charge'][t] - 
                               result['storage_discharge'][t] / eta)
            else:
                expected_soc = (result['storage_soc'][t-1] + 
                               eta * result['storage_charge'][t] - 
                               result['storage_discharge'][t] / eta)
            
            self.assertAlmostEqual(
                result['storage_soc'][t], 
                expected_soc, 
                places=5,
                msg=f"Storage SOC balance not maintained at hour {t}"
            )
    
    def test_flexible_ev_total_energy(self):
        """Test that flexible EVs charge exactly the required energy"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        total_ev_charged = np.sum(result['flex_ev_charging'])
        self.assertAlmostEqual(
            total_ev_charged, 
            self.flex_ev_energy, 
            places=5,
            msg="Flexible EV total energy constraint not satisfied"
        )
    
    def test_flexible_ev_window_constraint(self):
        """Test that flexible EVs only charge during allowed window"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        for t in range(24):
            # For overnight window (22 to 6)
            if self.flex_start > self.flex_end:
                in_window = (t >= self.flex_start or t <= self.flex_end)
            else:
                in_window = (self.flex_start <= t <= self.flex_end)
            
            if not in_window:
                self.assertAlmostEqual(
                    result['flex_ev_charging'][t], 
                    0, 
                    places=5,
                    msg=f"EV charging outside allowed window at hour {t}"
                )
    
    def test_no_flexible_ev(self):
        """Test optimization without flexible EVs"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            0,  # No flexible EVs
            self.flex_start,
            self.flex_end,
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(np.sum(result['flex_ev_charging']), 0)
    
    def test_daytime_charging_window(self):
        """Test with daytime charging window"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            9,   # 9 AM
            17,  # 5 PM
            self.solar_cost,
            self.wind_cost,
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        
        # Check window constraint
        for t in range(24):
            if not (9 <= t <= 17):
                self.assertAlmostEqual(
                    result['flex_ev_charging'][t], 
                    0, 
                    places=5,
                    msg=f"EV charging outside daytime window at hour {t}"
                )
        
        # Should still meet total energy
        total_ev_charged = np.sum(result['flex_ev_charging'])
        self.assertAlmostEqual(total_ev_charged, self.flex_ev_energy, places=5)
    
    def test_high_renewable_scenario(self):
        """Test with very cheap renewables"""
        result = optimize_energy_system(
            self.fixed_load,
            self.solar_cf,
            self.wind_cf,
            self.flex_ev_energy,
            self.flex_start,
            self.flex_end,
            100,   # Very cheap solar
            100,   # Very cheap wind
            self.storage_cost,
            self.storage_duration,
            self.efficiency
        )
        
        self.assertTrue(result['success'])
        # Should have minimal slack generation
        self.assertLess(np.sum(result['slack_generation']), 1.0)
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors"""
        # Wrong array length
        with self.assertRaises(ValueError):
            optimize_energy_system(
                np.ones(23),  # Wrong length
                self.solar_cf,
                self.wind_cf,
                self.flex_ev_energy,
                self.flex_start,
                self.flex_end,
                self.solar_cost,
                self.wind_cost,
                self.storage_cost,
                self.storage_duration,
                self.efficiency
            )
        
        # Invalid flex hours
        with self.assertRaises(ValueError):
            optimize_energy_system(
                self.fixed_load,
                self.solar_cf,
                self.wind_cf,
                self.flex_ev_energy,
                25,  # Invalid hour
                self.flex_end,
                self.solar_cost,
                self.wind_cost,
                self.storage_cost,
                self.storage_duration,
                self.efficiency
            )
        
        # Invalid efficiency
        with self.assertRaises(ValueError):
            optimize_energy_system(
                self.fixed_load,
                self.solar_cf,
                self.wind_cf,
                self.flex_ev_energy,
                self.flex_start,
                self.flex_end,
                self.solar_cost,
                self.wind_cost,
                self.storage_cost,
                self.storage_duration,
                150  # Invalid efficiency > 100%
            )


if __name__ == '__main__':
    unittest.main()