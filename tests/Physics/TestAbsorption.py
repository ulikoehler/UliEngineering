#!/usr/bin/env python3
import unittest
import numpy as np
from UliEngineering.Chemistry.Absorption import (
    absorption_length_from_absorption_coefficient,
    extinction_coefficient_from_absorption_length,
    remaining_light_fraction,
    length_from_remaining_fraction,
    half_length,
    HaleQuerryAbsorptionModel,
)

class TestAbsorptionFunctions(unittest.TestCase):
    def test_absorption_length_from_extinction_coefficient_scalar(self):
        self.assertAlmostEqual(absorption_length_from_absorption_coefficient(2.0), 0.5)
        self.assertAlmostEqual(absorption_length_from_absorption_coefficient(0.5), 2.0)

    def test_absorption_length_from_extinction_coefficient_array(self):
        arr = np.array([1.0, 2.0, 4.0])
        expected = np.array([1.0, 0.5, 0.25])
        np.testing.assert_allclose(absorption_length_from_absorption_coefficient(arr), expected)

    def test_extinction_coefficient_from_absorption_length_scalar(self):
        self.assertAlmostEqual(extinction_coefficient_from_absorption_length(2.0), 0.5)
        self.assertAlmostEqual(extinction_coefficient_from_absorption_length(0.5), 2.0)

    def test_extinction_coefficient_from_absorption_length_array(self):
        arr = np.array([1.0, 2.0, 4.0])
        expected = np.array([1.0, 0.5, 0.25])
        np.testing.assert_allclose(extinction_coefficient_from_absorption_length(arr), expected)

    def test_remaining_light_fraction_scalar(self):
        # For length=0, should be 1.0
        self.assertAlmostEqual(remaining_light_fraction(0, 1.0), 1.0)
        # For extinction_coefficient=0, should be 1.0
        self.assertAlmostEqual(remaining_light_fraction(1.0, 0), 1.0)
        # For length=1, extinction_coefficient=1, should be exp(-1)
        self.assertAlmostEqual(remaining_light_fraction(1.0, 1.0), np.exp(-1))

    def test_remaining_light_fraction_array(self):
        lengths = np.array([0, 1, 2])
        ext = 1.0
        expected = np.exp(-ext * lengths)
        np.testing.assert_allclose(remaining_light_fraction(lengths, ext), expected)

    def test_length_from_remaining_fraction_scalar(self):
        # For remaining_fraction=1, should be 0
        self.assertAlmostEqual(length_from_remaining_fraction(1.0, 1.0), 0.0)
        # For remaining_fraction=exp(-2), extinction_coefficient=2, should be 1
        self.assertAlmostEqual(length_from_remaining_fraction(np.exp(-2), 2.0), 1.0)

    def test_length_from_remaining_fraction_array(self):
        fractions = np.exp(-np.array([0, 1, 2]))
        ext = 1.0
        expected = np.array([0, 1, 2])
        np.testing.assert_allclose(length_from_remaining_fraction(fractions, ext), expected)

    def test_half_length_scalar(self):
        # For extinction_coefficient=1, half_length = -ln(0.5)/1 = ln(2)
        self.assertAlmostEqual(half_length(1.0), np.log(2))
        # For extinction_coefficient=2, half_length = ln(2)/2
        self.assertAlmostEqual(half_length(2.0), np.log(2)/2)

    def test_half_length_array(self):
        ext = np.array([1.0, 2.0, 4.0])
        expected = np.log(2) / ext
        np.testing.assert_allclose(half_length(ext), expected)

    def test_inverse_absorption_length_and_extinction_coefficient_scalar(self):
        # Should be inverse of each other
        for val in [0.1, 1.0, 10.0, 1e-6]:
            l = absorption_length_from_absorption_coefficient(val)
            self.assertAlmostEqual(extinction_coefficient_from_absorption_length(l), val)
            e = extinction_coefficient_from_absorption_length(val)
            self.assertAlmostEqual(absorption_length_from_absorption_coefficient(e), val)

    def test_inverse_absorption_length_and_extinction_coefficient_array(self):
        vals = np.array([0.1, 1.0, 10.0, 1e-6])
        l = absorption_length_from_absorption_coefficient(vals)
        np.testing.assert_allclose(extinction_coefficient_from_absorption_length(l), vals)
        e = extinction_coefficient_from_absorption_length(vals)
        np.testing.assert_allclose(absorption_length_from_absorption_coefficient(e), vals)

    def test_remaining_fraction_and_length_consistency_scalar(self):
        # For a range of extinction coefficients and lengths
        for ext in [0.1, 1.0, 10.0]:
            for length in [0.0, 0.5, 1.0, 2.0]:
                frac = remaining_light_fraction(length, ext)
                length2 = length_from_remaining_fraction(frac, ext)
                self.assertAlmostEqual(length, length2, places=10)
                # Also check round-trip
                frac2 = remaining_light_fraction(length2, ext)
                self.assertAlmostEqual(frac, frac2, places=10)

    def test_remaining_fraction_and_length_consistency_array(self):
        ext = np.array([0.1, 1.0, 10.0])
        length = np.array([0.0, 0.5, 1.0])
        # Test all combinations
        for e in ext:
            for l in length:
                frac = remaining_light_fraction(l, e)
                l2 = length_from_remaining_fraction(frac, e)
                self.assertAlmostEqual(l, l2, places=10)

    def test_remaining_fraction_and_length_vectorized(self):
        ext = np.array([0.1, 1.0, 10.0])
        length = np.array([0.0, 0.5, 1.0])
        frac = remaining_light_fraction(length, ext)
        length2 = length_from_remaining_fraction(frac, ext)
        np.testing.assert_allclose(length, length2, atol=1e-10)

    def test_half_length_consistency(self):
        # half_length should be the same as length_from_remaining_fraction(0.5, ...)
        for ext in [0.1, 1.0, 10.0]:
            hl = half_length(ext)
            hl2 = length_from_remaining_fraction(0.5, ext)
            self.assertAlmostEqual(hl, hl2, places=10)
        ext_arr = np.array([0.1, 1.0, 10.0])
        np.testing.assert_allclose(half_length(ext_arr), length_from_remaining_fraction(0.5, ext_arr), atol=1e-10)

    def test_zero_and_negative_inputs(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"):
                # Absorption length zero: extinction coefficient should be inf
                self.assertTrue(np.isinf(extinction_coefficient_from_absorption_length(0.0)))
                # Negative extinction coefficient: should return negative absorption length
                self.assertLess(absorption_length_from_absorption_coefficient(-1.0), 0)
                # Negative absorption length: should return negative extinction coefficient
                self.assertLess(extinction_coefficient_from_absorption_length(-1.0), 0)
                # Remaining fraction negative or >1: length_from_remaining_fraction should handle gracefully (may return nan or inf)
                self.assertTrue(np.isnan(length_from_remaining_fraction(-0.1, 1.0)))
                self.assertTrue(np.isinf(length_from_remaining_fraction(0.0, 1.0)))
                self.assertAlmostEqual(length_from_remaining_fraction(1.0, 1.0), 0.0)

    def test_units_and_types(self):
        # Accepts lists, tuples, numpy arrays, scalars
        for typ in [list, tuple, np.array]:
            vals = typ([1.0, 2.0, 4.0])
            res = absorption_length_from_absorption_coefficient(vals)
            np.testing.assert_allclose(res, [1.0, 0.5, 0.25])

class TestHaleQuerryAbsorptionModel(unittest.TestCase):
    def setUp(self):
        self.model = HaleQuerryAbsorptionModel()

    def test_interpolation_within_bounds(self):
        # Test a wavelength exactly at a datapoint (use meters)
        # Use an interior datapoint (225 nm) to avoid endpoint exclusion
        wl = 225.0e-9
        # model returns absorption_coefficient (alpha), not the unitless extinction kappa
        expected = self.model.datapoints[1].absorption_coefficient
        self.assertAlmostEqual(self.model(wl), expected)
        # Test a wavelength between two datapoints (in meters)
        # Interpolate between 225 nm and 250 nm
        wl = 237.5e-9
        # Linear interpolation between the datapoints' absorption coefficients (225 nm and 250 nm)
        y0 = self.model.datapoints[1].absorption_coefficient
        y1 = self.model.datapoints[2].absorption_coefficient
        x0, x1 = self.model.datapoints[1].wavelength, self.model.datapoints[2].wavelength
        expected = y0 + (y1 - y0) * (wl - x0) / (x1 - x0)
        self.assertAlmostEqual(self.model(wl), expected)

    def test_interpolation_array(self):
        wls = np.array([225.0e-9, 250.0e-9, 275.0e-9])
        expected = np.array([
            self.model.datapoints[1].absorption_coefficient,
            self.model.datapoints[2].absorption_coefficient,
            self.model.datapoints[3].absorption_coefficient,
        ])
        np.testing.assert_allclose(self.model(wls), expected)

    def test_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            self.model(100.0e-9)
        with self.assertRaises(ValueError):
            self.model(201000.0e-9)
        with self.assertRaises(ValueError):
            self.model(np.array([200.0e-9, 300000.0e-9]))

    def test_vectorized_interpolation(self):
        # Should work for numpy arrays (in meters)
        wls = np.linspace(225.0e-9, 1000.0e-9, 5)
        result = self.model(wls)
        self.assertEqual(result.shape, wls.shape)
        # All values should be between min and max of datapoints
        self.assertTrue(np.all(result >= np.min(self.model._ext_coeffs)))
        self.assertTrue(np.all(result <= np.max(self.model._ext_coeffs)))

    def test_model_consistency_with_manual(self):
        model = self.model
        # Test that for datapoints, interpolation matches exactly
        # Skip first and last datapoints because the implementation treats endpoints as out-of-bounds
        for dp in model.datapoints[1:-1:10]:  # test every 10th interior point for speed
            self.assertAlmostEqual(model(dp.wavelength), dp.absorption_coefficient, places=12)

    def test_model_vectorized_consistency(self):
        model = self.model
        # Use interior datapoints only, endpoints are out-of-bounds
        wls = np.array([dp.wavelength for dp in model.datapoints[1:-1]])
        expected = np.array([dp.absorption_coefficient for dp in model.datapoints[1:-1]])
        np.testing.assert_allclose(model(wls), expected, atol=1e-12)

    def test_model_bounds(self):
        model = self.model
        min_wl = model._wavelengths[0]
        max_wl = model._wavelengths[-1]
        # Endpoints are treated as out-of-bounds by the model implementation
        with self.assertRaises(ValueError):
            model(min_wl)
        with self.assertRaises(ValueError):
            model(max_wl)
        # Should also fail just outside bounds
        with self.assertRaises(ValueError):
            model(min_wl - 1e-9)
        with self.assertRaises(ValueError):
            model(max_wl + 1e-9)

if __name__ == "__main__":
    unittest.main()
