import unittest
import numpy as np

from UliEngineering.Electronics.LogarithmicAmplifier import (
    logarithmic_amplifier_output_voltage,
    logarithmic_amplifier_input_current
)
from UliEngineering.Electronics.Diode import VoltageV, CurrentA

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
        """Example from AD5303 datasheet, with amperes rather than watts."""
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
        """Example from AD5303 datasheet, with amperes rather than watts."""
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

class TestAnnotatedTypes(unittest.TestCase):
    def test_type_annotations_exist(self):
        """Test that the type annotations are available."""
        self.assertIsNotNone(VoltageV)
        self.assertIsNotNone(CurrentA)

    def test_logarithmic_amplifier_functions_various_units(self):
        """Test logarithmic amplifier functions with various unit inputs."""
        # Test with different unit representations
        v1 = logarithmic_amplifier_output_voltage("1 µA", 0.2, "1 nA")
        v2 = logarithmic_amplifier_output_voltage("0.001 mA", 0.2, "0.001 µA")
        self.assertAlmostEqual(v1, v2)

        # Test input current with different units
        i1 = logarithmic_amplifier_input_current("0.6 V", 0.2, "1 nA")
        i2 = logarithmic_amplifier_input_current("600 mV", 0.2, "0.001 µA")
        self.assertAlmostEqual(i1, i2)

if __name__ == '__main__':
    unittest.main()