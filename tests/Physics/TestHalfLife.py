import unittest
from UliEngineering.Physics.HalfLife import half_lifes_passed, fraction_remaining, fraction_decayed, remaining_quantity, decayed_quantity

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
