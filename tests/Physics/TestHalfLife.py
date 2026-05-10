import unittest
from numpy.testing import assert_approx_equal
from UliEngineering.Physics.HalfLife import (
    half_lifes_passed, fraction_remaining, fraction_decayed, remaining_quantity, decayed_quantity,
    half_life_from_decay_constant, half_life_from_remaining_quantity, half_life_from_decayed_quantity,
    half_life_from_fraction_remaining, half_life_from_fraction_decayed,
    normalize_timespan, TimespanSeconds
)

class TestHalfLife(unittest.TestCase):
    def test_half_lifes_passed(self):
        self.assertAlmostEqual(half_lifes_passed("1h", "1min"), 60.0)
        self.assertAlmostEqual(half_lifes_passed("2h", "30min"), 4.0)
        self.assertAlmostEqual(half_lifes_passed("1h", "2h"), 0.5)

    def test_fraction_remaining(self):
        self.assertAlmostEqual(fraction_remaining("1h", "1h"), 0.5)
        self.assertAlmostEqual(fraction_remaining("2h", "1h"), 0.25)
        self.assertAlmostEqual(fraction_remaining("1h", "2h"), 0.70710678118, places=6)

    def test_fraction_decayed(self):
        self.assertAlmostEqual(fraction_decayed("1h", "1h"), 0.5)
        self.assertAlmostEqual(fraction_decayed("2h", "1h"), 0.75)
        self.assertAlmostEqual(fraction_decayed("1h", "2h"), 0.29289321881, places=6)

    def test_remaining_quantity(self):
        self.assertAlmostEqual(remaining_quantity("1h", "1h", 100), 50.0)
        self.assertAlmostEqual(remaining_quantity("2h", "1h", 100), 25.0)
        self.assertAlmostEqual(remaining_quantity("1h", "2h", 100), 70.710678118, places=6)

    def test_decayed_quantity(self):
        self.assertAlmostEqual(decayed_quantity("1h", "1h", 100), 50.0)
        self.assertAlmostEqual(decayed_quantity("2h", "1h", 100), 75.0)
        self.assertAlmostEqual(decayed_quantity("1h", "2h", 100), 29.289321881, places=6)

    def test_half_life_from_decay_constant(self):
        # Test with decay constant 0.1
        self.assertAlmostEqual(half_life_from_decay_constant(0.1), 6.9314718056, places=10)
        # Test with decay constant 1.0
        self.assertAlmostEqual(half_life_from_decay_constant(1.0), 0.69314718056, places=10)
        # Test with decay constant 0.5
        self.assertAlmostEqual(half_life_from_decay_constant(0.5), 1.38629436112, places=10)
        # Test with decay constant 2.0
        self.assertAlmostEqual(half_life_from_decay_constant(2.0), 0.34657359028, places=10)

    def test_half_life_from_remaining_quantity(self):
        self.assertAlmostEqual(
            half_life_from_remaining_quantity("1h", 50, 100),
            3600,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_remaining_quantity("2h", 25, 100),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_remaining_quantity(7200, 25, 100),
            3600.0,
            places=5
        )

    def test_half_life_from_decayed_quantity(self):
        self.assertAlmostEqual(
            half_life_from_decayed_quantity("1h", 50, 100),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_decayed_quantity("2h", 75, 100),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_decayed_quantity(7200, 75, 100),
            3600.0,
            places=5
        )

    def test_half_life_from_fraction_remaining(self):
        self.assertAlmostEqual(
            half_life_from_fraction_remaining(3600, 0.5),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_fraction_remaining("2h", 0.25),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_fraction_remaining(7200, 0.25),
            3600.0,
            places=5
        )

    def test_half_life_from_fraction_decayed(self):
        self.assertAlmostEqual(
            half_life_from_fraction_decayed(3600, 0.5),
            3600.0,
            places=5
        )
        self.assertAlmostEqual(
            half_life_from_fraction_decayed("2h", 0.75),
            3600.0,
        )
        self.assertAlmostEqual(
            half_life_from_fraction_decayed(7200, 0.75),
            3600.0,
        )

    def test_type_annotation_exists(self):
        """Test that the TimespanSeconds type annotation is available."""
        self.assertIsNotNone(TimespanSeconds)

    def test_half_life_functions_various_time_units(self):
        """Test half_life functions with various time unit inputs."""
        # Test half_lifes_passed with various units
        self.assertAlmostEqual(half_lifes_passed("60 min", "1 min"), 60.0)
        self.assertAlmostEqual(half_lifes_passed("3600 s", "60 s"), 60.0)
        self.assertAlmostEqual(half_lifes_passed("1 h", "1 h"), 1.0)

        # Test fraction_remaining with various units
        self.assertAlmostEqual(fraction_remaining("60 min", "60 min"), 0.5)
        self.assertAlmostEqual(fraction_remaining("3600 s", "3600 s"), 0.5)

        # Test fraction_decayed with various units
        self.assertAlmostEqual(fraction_decayed("60 min", "60 min"), 0.5)
        self.assertAlmostEqual(fraction_decayed("3600 s", "3600 s"), 0.5)

        # Test remaining_quantity with various units
        self.assertAlmostEqual(remaining_quantity("60 min", "60 min", 100), 50.0)
        self.assertAlmostEqual(remaining_quantity("3600 s", "3600 s", 100), 50.0)

        # Test decayed_quantity with various units
        self.assertAlmostEqual(decayed_quantity("60 min", "60 min", 100), 50.0)
        self.assertAlmostEqual(decayed_quantity("3600 s", "3600 s", 100), 50.0)

        # Test half_life_from_remaining_quantity with various units
        self.assertAlmostEqual(half_life_from_remaining_quantity("60 min", 50, 100), 3600.0, places=5)
        self.assertAlmostEqual(half_life_from_remaining_quantity("3600 s", 50, 100), 3600.0, places=5)

        # Test half_life_from_decayed_quantity with various units
        self.assertAlmostEqual(half_life_from_decayed_quantity("60 min", 50, 100), 3600.0, places=5)
        self.assertAlmostEqual(half_life_from_decayed_quantity("3600 s", 50, 100), 3600.0, places=5)

        # Test half_life_from_fraction_remaining with various units
        self.assertAlmostEqual(half_life_from_fraction_remaining("60 min", 0.5), 3600.0, places=5)
        self.assertAlmostEqual(half_life_from_fraction_remaining("3600 s", 0.5), 3600.0, places=5)

        # Test half_life_from_fraction_decayed with various units
        self.assertAlmostEqual(half_life_from_fraction_decayed("60 min", 0.5), 3600.0, places=5)
        self.assertAlmostEqual(half_life_from_fraction_decayed("3600 s", 0.5), 3600.0, places=5)