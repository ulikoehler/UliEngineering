import unittest
import numpy as np

from UliEngineering.Electronics.LogarithmicAmplifier import (
    logarithmic_amplifier_output_voltage,
    logarithmic_amplifier_input_current
)

class TestLogarithmicAmplifier(unittest.TestCase):
    def test_logarithmic_amplifier_output_voltage(self):
        # Test with known values
        ipd = 1e-6  # 1 µA
        gain = 0.2  # 0.2 V/decade
        intercept = 1e-9  # 1 nA
        expected_output_voltage = gain * np.log10(ipd / intercept)
        self.assertAlmostEqual(
            logarithmic_amplifier_output_voltage(ipd, gain, intercept),
            expected_output_voltage,
            places=6
        )
        
        
    def test_logarithmic_amplifier_output_voltage_ad5303(self):
        """Example from AD5303 datasheet, with amperes rather than watts"""
        # Test with known values
        ipd = "3mA"
        gain = "200mV" # /decade
        intercept = "110 pA"
        expected_output_voltage = 1.487 # V, from datasheet example
        self.assertAlmostEqual(
            logarithmic_amplifier_output_voltage(ipd, gain, intercept),
            expected_output_voltage,
            places=3 # Datasheet gives 3 digits only
        )

    def test_logarithmic_amplifier_input_current(self):
        # Test with known values
        vout = 0.6  # 0.6 V
        gain = 0.2  # 0.2 V/decade
        intercept = 1e-9  # 1 nA
        expected_input_current = intercept * np.power(10, vout / gain)
        self.assertAlmostEqual(
            logarithmic_amplifier_input_current(vout, gain, intercept),
            expected_input_current,
            places=6
        )
        
        
    def test_logarithmic_amplifier_input_current_ad5303(self):
        """Example from AD5303 datasheet, with amperes rather than watts"""
        # Test with known values
        vout = "1.487 V"
        gain = "200mV" # /decade
        intercept = "110 pA"
        expected_input_current = 3e-3 # A, from datasheet example
        self.assertAlmostEqual(
            logarithmic_amplifier_input_current(vout, gain, intercept),
            expected_input_current,
            places=3
        )

if __name__ == '__main__':
    unittest.main()