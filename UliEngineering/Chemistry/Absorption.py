#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Length import normalize_length
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c as speed_of_light

__all__ = [
    "HaleQuerryAbsorptionData",
    "HaleQuerryAbsorptionModel",
    "absorption_length_from_absorption_coefficient",
    "extinction_coefficient_from_absorption_length",
    "remaining_light_fraction",
    "length_from_remaining_fraction",
    "half_length",
]

@normalize_numeric_args
@returns_unit("m")
def absorption_length_from_absorption_coefficient(absorption_coefficient):
    """
    Compute the absorption length (in meters) from the extinction coefficient (in 1/m).
    Absorption length is defined as the distance over which the intensity drops to 1/e.
    Formula: absorption_length = 1 / absorption_coefficient):
    
    NOTE: The absopriotn coefficient must bei in 1/m, not in 1/cm or any other unit.

    Parameters:
    - absorption_coefficient):: Extinction coefficient in 1/m (scalar, list, or ndarray).

    Returns:
    - Absorption length in meters.
    """
    return np.reciprocal(absorption_coefficient)

@normalize_numeric_args
@returns_unit("1/m")
def extinction_coefficient_from_absorption_length(absorption_length):
    """
    Compute the extinction coefficient (in 1/m) from the absorption length (in meters).
    Formula: extinction_coefficient = 1 / absorption_length

    Parameters:
    - absorption_length: Absorption length in meters (scalar, list, or ndarray).

    Returns:
    - Extinction coefficient in 1/m.
    """
    return np.reciprocal(absorption_length)

@returns_unit("")
def remaining_light_fraction(length, absorption_coefficient):
    """
    Compute the remaining fraction of light after passing through a medium of given length (in meters)
    with a given extinction coefficient (in 1/m).

    Formula: fraction = exp(-absorption_coefficient * length)

    Parameters:
    - length: Length of the medium (scalar, list, or ndarray, any unit parsable by normalize_length)
    - absorption_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Remaining fraction of light (unitless)
    """
    length = normalize_length(length)
    absorption_coefficient = normalize_numeric(absorption_coefficient)
    return np.exp(-absorption_coefficient * length)

@normalize_numeric_args
@returns_unit("m")
def length_from_remaining_fraction(remaining_fraction, absorption_coefficient):
    """
    Compute the length of the medium (in meters) given the remaining fraction of light
    and the extinction coefficient (in 1/m).

    Formula: length = -ln(remaining_fraction) / absorption_coefficient

    Parameters:
    - remaining_fraction: Remaining fraction of light (scalar, list, or ndarray, unitless)
    - absorption_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Length in meters.
    """
    return -np.log(remaining_fraction) / absorption_coefficient

@normalize_numeric_args
@returns_unit("m")
def half_length(absorption_coefficient):
    """
    Compute the half-length, i.e., the length of medium where the remaining fraction of light is 0.5,
    for a given extinction coefficient (in 1/m).

    Parameters:
    - absorption_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Half-length in meters.
    """
    return length_from_remaining_fraction(0.5, absorption_coefficient)

@normalize_numeric_args
@returns_unit("1/m")
def absorption_coefficient_from_extinction_coefficient(extinction_coefficient, wavelength):
    """
    Compute the absorption coefficient (alpha, in 1/m) from the extinction coefficient (kappa, unitless)
    and the wavelength (in meters).

    Uses the formula:
        alpha = (2 * omega * kappa) / c
    where omega = 2 * pi * c / wavelength
    
    Source:
    SOLID STATE PHYSICS, Part II, M. S. Dresselhaus, Formula 5.2
    https://web.mit.edu/course/6/6.732/www/6.732-pt2.pdf

    Parameters:
    - extinction_coefficient: Extinction coefficient kappa (unitless)
    - wavelength: Wavelength (meters or any unit parsable by normalize_length)

    Returns:
    - Absorption coefficient alpha in 1/m
    """
    wavelength = normalize_length(wavelength)
    omega = 2 * np.pi * speed_of_light / wavelength
    return 2 * omega * extinction_coefficient / speed_of_light

@dataclass
class HaleQuerryAbsorptionData:
    wavelength: float # in meters (m)
    extinction_coefficient: float
    refractive_index: float
    absorption_coefficient: float = None # Initialized from extinction_coefficient

    def __post_init__(self):
        # Compute absorption_coefficient using the extinction_coefficient and wavelength
        self.absorption_coefficient = absorption_coefficient_from_extinction_coefficient(
            self.extinction_coefficient, self.wavelength
        )

class HaleQuerryAbsorptionModel:
    """
    Hale-Querry absorption model for water. Valid from 200nm to 200μm.
    Interpolated using a piecewise linear function.

    Input wavelength must be in nanometers (nm).

    Source: Hale, G. M., & Querry, M. R. (1973). Optical constants of water in the 200-nm to 200-μm wavelength region, Table 1
    Document available at: https://opg.optica.org/ao/viewmedia.cfm?uri=ao-12-3-555&seq=0
    """
    datapoints = [
        HaleQuerryAbsorptionData(200.0e-9, 1.1e-7, 1.396),
        HaleQuerryAbsorptionData(225.0e-9, 4.9e-8, 1.373),
        HaleQuerryAbsorptionData(250.0e-9, 3.35e-8, 1.362),
        HaleQuerryAbsorptionData(275.0e-9, 2.35e-8, 1.354),
        HaleQuerryAbsorptionData(300.0e-9, 1.6e-8, 1.349),
        HaleQuerryAbsorptionData(325.0e-9, 1.08e-8, 1.346),
        HaleQuerryAbsorptionData(350.0e-9, 6.5e-9, 1.343),
        HaleQuerryAbsorptionData(375.0e-9, 3.5e-9, 1.341),
        HaleQuerryAbsorptionData(400.0e-9, 1.86e-9, 1.339),
        HaleQuerryAbsorptionData(425.0e-9, 1.3e-9, 1.338),
        HaleQuerryAbsorptionData(450.0e-9, 1.02e-9, 1.337),
        HaleQuerryAbsorptionData(475.0e-9, 9.35e-10, 1.336),
        HaleQuerryAbsorptionData(500.0e-9, 1.00e-9, 1.335),
        HaleQuerryAbsorptionData(525.0e-9, 1.32e-9, 1.334),
        HaleQuerryAbsorptionData(550.0e-9, 1.96e-9, 1.333),
        HaleQuerryAbsorptionData(575.0e-9, 3.60e-9, 1.333),
        HaleQuerryAbsorptionData(600.0e-9, 1.09e-8, 1.332),
        HaleQuerryAbsorptionData(625.0e-9, 1.39e-8, 1.332),
        HaleQuerryAbsorptionData(650.0e-9, 1.64e-8, 1.331),
        HaleQuerryAbsorptionData(675.0e-9, 2.29e-8, 1.331),
        HaleQuerryAbsorptionData(700.0e-9, 3.35e-8, 1.331),
        HaleQuerryAbsorptionData(725.0e-9, 9.15e-8, 1.330),
        HaleQuerryAbsorptionData(750.0e-9, 1.56e-7, 1.330),
        HaleQuerryAbsorptionData(775.0e-9, 1.48e-7, 1.330),
        HaleQuerryAbsorptionData(800.0e-9, 1.25e-7, 1.329),
        HaleQuerryAbsorptionData(825.0e-9, 1.82e-7, 1.329),
        HaleQuerryAbsorptionData(850.0e-9, 2.93e-7, 1.329),
        HaleQuerryAbsorptionData(875.0e-9, 3.91e-7, 1.328, 6.0),
        HaleQuerryAbsorptionData(900.0e-9, 4.86e-7, 1.328, 6.1),
        HaleQuerryAbsorptionData(925.0e-9, 1.06e-6, 1.328, 6.2),
        HaleQuerryAbsorptionData(950.0e-9, 2.93e-6, 1.327, 6.3),
        HaleQuerryAbsorptionData(975.0e-9, 3.48e-6, 1.327, 6.4),
        HaleQuerryAbsorptionData(1000.0e-9, 2.89e-6, 1.327, 6.5),
        HaleQuerryAbsorptionData(1200.0e-9, 9.89e-6, 1.324, 6.6),
        HaleQuerryAbsorptionData(1400.0e-9, 1.38e-4, 1.321, 6.7),
        HaleQuerryAbsorptionData(1600.0e-9, 8.55e-5, 1.317, 6.8),
        HaleQuerryAbsorptionData(1800.0e-9, 1.15e-4, 1.312, 6.9),
        HaleQuerryAbsorptionData(2000.0e-9, 1.1e-3, 1.306, 7.0),
        HaleQuerryAbsorptionData(2200.0e-9, 2.89e-4, 1.296, 7.1),
        HaleQuerryAbsorptionData(2400.0e-9, 9.56e-4, 1.279, 7.2),
        HaleQuerryAbsorptionData(2600.0e-9, 3.17e-3, 1.242, 7.3),
        HaleQuerryAbsorptionData(2650.0e-9, 6.7e-3, 1.219, 7.4),
        HaleQuerryAbsorptionData(2700.0e-9, 0.019, 1.188, 7.5),
        HaleQuerryAbsorptionData(2750.0e-9, 0.059, 1.157, 7.6),
        HaleQuerryAbsorptionData(2800.0e-9, 0.115, 1.142, 7.7),
        HaleQuerryAbsorptionData(2850.0e-9, 0.185, 1.149, 7.8),
        HaleQuerryAbsorptionData(2900.0e-9, 0.268, 1.201, 7.9),
        HaleQuerryAbsorptionData(2950.0e-9, 0.298, 1.292, 8.0),
        HaleQuerryAbsorptionData(3000.0e-9, 0.272, 1.371, 8.2),
        HaleQuerryAbsorptionData(3050.0e-9, 0.240, 1.426, 8.4),
        HaleQuerryAbsorptionData(3100.0e-9, 0.192, 1.467, 8.6),
        HaleQuerryAbsorptionData(3150.0e-9, 0.135, 1.483, 8.8),
        HaleQuerryAbsorptionData(3200.0e-9, 0.0924, 1.478, 9.0),
        HaleQuerryAbsorptionData(3250.0e-9, 0.0610, 1.467, 9.2),
        HaleQuerryAbsorptionData(3300.0e-9, 0.0368, 1.450, 9.4),
        HaleQuerryAbsorptionData(3350.0e-9, 0.0261, 1.432, 9.6),
        HaleQuerryAbsorptionData(3400.0e-9, 0.0195, 1.420),
        HaleQuerryAbsorptionData(3450.0e-9, 0.0132, 1.410),
        HaleQuerryAbsorptionData(3500.0e-9, 0.0094, 1.400),
        HaleQuerryAbsorptionData(3600.0e-9, 0.00515, 1.385),
        HaleQuerryAbsorptionData(3700.0e-9, 0.00360, 1.374),
        HaleQuerryAbsorptionData(3800.0e-9, 0.00340, 1.364),
        HaleQuerryAbsorptionData(3900.0e-9, 0.00380, 1.357),
        HaleQuerryAbsorptionData(4000.0e-9, 0.00460, 1.351),
        HaleQuerryAbsorptionData(4100.0e-9, 0.00562, 1.346),
        HaleQuerryAbsorptionData(4200.0e-9, 0.00688, 1.342),
        HaleQuerryAbsorptionData(4300.0e-9, 0.00845, 1.338),
        HaleQuerryAbsorptionData(4400.0e-9, 0.0103, 1.334),
        HaleQuerryAbsorptionData(4500.0e-9, 0.0134, 1.332),
        HaleQuerryAbsorptionData(4600.0e-9, 0.0147, 1.330),
        HaleQuerryAbsorptionData(4700.0e-9, 0.0157, 1.330),
        HaleQuerryAbsorptionData(4800.0e-9, 0.0150, 1.330),
        HaleQuerryAbsorptionData(4900.0e-9, 0.0137, 1.328),
        HaleQuerryAbsorptionData(5000.0e-9, 0.0124, 1.325),
        HaleQuerryAbsorptionData(5100.0e-9, 0.0111, 1.322),
        HaleQuerryAbsorptionData(5200.0e-9, 0.0101, 1.317),
        HaleQuerryAbsorptionData(5300.0e-9, 0.0090, 1.312),
        HaleQuerryAbsorptionData(5400.0e-9, 0.0103, 1.305),
        HaleQuerryAbsorptionData(5500.0e-9, 0.0116, 1.298),
        HaleQuerryAbsorptionData(5600.0e-9, 0.0142, 1.289),
        HaleQuerryAbsorptionData(5700.0e-9, 0.0203, 1.277),
        HaleQuerryAbsorptionData(5800.0e-9, 0.0330, 1.262),
        HaleQuerryAbsorptionData(5900.0e-9, 0.0622, 1.248),
        HaleQuerryAbsorptionData(6000.0e-9, 0.107, 1.265),
        HaleQuerryAbsorptionData(6100.0e-9, 0.131, 1.319),
        HaleQuerryAbsorptionData(6200.0e-9, 0.0880, 1.363),
        HaleQuerryAbsorptionData(6300.0e-9, 0.0570, 1.357),
        HaleQuerryAbsorptionData(6400.0e-9, 0.0449, 1.347),
        HaleQuerryAbsorptionData(6500.0e-9, 0.0392, 1.339),
        HaleQuerryAbsorptionData(6600.0e-9, 0.0356, 1.334),
        HaleQuerryAbsorptionData(6700.0e-9, 0.0337, 1.329),
        HaleQuerryAbsorptionData(6800.0e-9, 0.0327, 1.324),
        HaleQuerryAbsorptionData(6900.0e-9, 0.0322, 1.321),
        HaleQuerryAbsorptionData(7000.0e-9, 0.0320, 1.317),
        HaleQuerryAbsorptionData(7100.0e-9, 0.0320, 1.314),
        HaleQuerryAbsorptionData(7200.0e-9, 0.0321, 1.312),
        HaleQuerryAbsorptionData(7300.0e-9, 0.0322, 1.309),
        HaleQuerryAbsorptionData(7400.0e-9, 0.0324, 1.307),
        HaleQuerryAbsorptionData(7500.0e-9, 0.0326, 1.304),
        HaleQuerryAbsorptionData(7600.0e-9, 0.0328, 1.302),
        HaleQuerryAbsorptionData(7700.0e-9, 0.0331, 1.299),
        HaleQuerryAbsorptionData(7800.0e-9, 0.0335, 1.297),
        HaleQuerryAbsorptionData(7900.0e-9, 0.0339, 1.294),
        HaleQuerryAbsorptionData(8000.0e-9, 0.0343, 1.291),
        HaleQuerryAbsorptionData(8200.0e-9, 0.0351, 1.286),
        HaleQuerryAbsorptionData(8400.0e-9, 0.0361, 1.281),
        HaleQuerryAbsorptionData(8600.0e-9, 0.0372, 1.275),
        HaleQuerryAbsorptionData(8800.0e-9, 0.0385, 1.269),
        HaleQuerryAbsorptionData(9000.0e-9, 0.0399, 1.262),
        HaleQuerryAbsorptionData(9200.0e-9, 0.0415, 1.255),
        HaleQuerryAbsorptionData(9400.0e-9, 0.0433, 1.247),
        HaleQuerryAbsorptionData(9600.0e-9, 0.0454, 1.239),
        HaleQuerryAbsorptionData(9800.0e-9, 0.0479, 1.229),
        HaleQuerryAbsorptionData(10000.0e-9, 0.0508, 1.218),
        HaleQuerryAbsorptionData(10500.0e-9, 0.0662, 1.185),
        HaleQuerryAbsorptionData(11000.0e-9, 0.0968, 1.153),
        HaleQuerryAbsorptionData(11500.0e-9, 0.142, 1.126),
        HaleQuerryAbsorptionData(12000.0e-9, 0.199, 1.111),
        HaleQuerryAbsorptionData(12500.0e-9, 0.259, 1.123),
        HaleQuerryAbsorptionData(13000.0e-9, 0.305, 1.146),
        HaleQuerryAbsorptionData(13500.0e-9, 0.343, 1.177),
        HaleQuerryAbsorptionData(14000.0e-9, 0.370, 1.210),
        HaleQuerryAbsorptionData(14500.0e-9, 0.388, 1.241),
        HaleQuerryAbsorptionData(15000.0e-9, 0.402, 1.270),
        HaleQuerryAbsorptionData(15500.0e-9, 0.414, 1.297),
        HaleQuerryAbsorptionData(16000.0e-9, 0.422, 1.325),
        HaleQuerryAbsorptionData(16500.0e-9, 0.428, 1.351),
        HaleQuerryAbsorptionData(17000.0e-9, 0.429, 1.376),
        HaleQuerryAbsorptionData(17500.0e-9, 0.429, 1.401),
        HaleQuerryAbsorptionData(18000.0e-9, 0.426, 1.423),
        HaleQuerryAbsorptionData(18500.0e-9, 0.421, 1.443),
        HaleQuerryAbsorptionData(19000.0e-9, 0.414, 1.461),
        HaleQuerryAbsorptionData(19500.0e-9, 0.404, 1.476),
        HaleQuerryAbsorptionData(20000.0e-9, 0.393, 1.480),
        HaleQuerryAbsorptionData(21000.0e-9, 0.382, 1.487),
        HaleQuerryAbsorptionData(22000.0e-9, 0.373, 1.500),
        HaleQuerryAbsorptionData(23000.0e-9, 0.367, 1.511),
        HaleQuerryAbsorptionData(24000.0e-9, 0.361, 1.521),
        HaleQuerryAbsorptionData(25000.0e-9, 0.356, 1.531),
        HaleQuerryAbsorptionData(26000.0e-9, 0.350, 1.539),
        HaleQuerryAbsorptionData(27000.0e-9, 0.344, 1.545),
        HaleQuerryAbsorptionData(28000.0e-9, 0.338, 1.549),
        HaleQuerryAbsorptionData(29000.0e-9, 0.333, 1.551),
        HaleQuerryAbsorptionData(30000.0e-9, 0.328, 1.551),
        HaleQuerryAbsorptionData(32000.0e-9, 0.324, 1.546),
        HaleQuerryAbsorptionData(34000.0e-9, 0.329, 1.536),
        HaleQuerryAbsorptionData(36000.0e-9, 0.343, 1.527),
        HaleQuerryAbsorptionData(38000.0e-9, 0.361, 1.522),
        HaleQuerryAbsorptionData(40000.0e-9, 0.385, 1.519),
        HaleQuerryAbsorptionData(42000.0e-9, 0.409, 1.522),
        HaleQuerryAbsorptionData(44000.0e-9, 0.436, 1.530),
        HaleQuerryAbsorptionData(46000.0e-9, 0.462, 1.541),
        HaleQuerryAbsorptionData(48000.0e-9, 0.488, 1.555),
        HaleQuerryAbsorptionData(50000.0e-9, 0.514, 1.587),
        HaleQuerryAbsorptionData(60000.0e-9, 0.587, 1.703),
        HaleQuerryAbsorptionData(70000.0e-9, 0.576, 1.821),
        HaleQuerryAbsorptionData(80000.0e-9, 0.547, 1.886),
        HaleQuerryAbsorptionData(90000.0e-9, 0.536, 1.924),
        HaleQuerryAbsorptionData(100000.0e-9, 0.532, 1.957),
        HaleQuerryAbsorptionData(110000.0e-9, 0.531, 1.966),
        HaleQuerryAbsorptionData(120000.0e-9, 0.526, 2.004),
        HaleQuerryAbsorptionData(130000.0e-9, 0.514, 2.036),
        HaleQuerryAbsorptionData(140000.0e-9, 0.500, 2.056),
        HaleQuerryAbsorptionData(150000.0e-9, 0.495, 2.069),
        HaleQuerryAbsorptionData(160000.0e-9, 0.496, 2.081),
        HaleQuerryAbsorptionData(170000.0e-9, 0.497, 2.094),
        HaleQuerryAbsorptionData(180000.0e-9, 0.499, 2.107),
        HaleQuerryAbsorptionData(190000.0e-9, 0.501, 2.119),
        HaleQuerryAbsorptionData(200000.0e-9, 0.504, 2.130),
    ]

    def __init__(self):
        # Extract wavelength and extinction coefficient arrays
        self._wavelengths = np.array([d.wavelength for d in self.datapoints])
        self._ext_coeffs = np.array([d.absorption_coefficient for d in self.datapoints])
        # Build linear interpolation model
        self._interp = interp1d(
            self._wavelengths,
            self._ext_coeffs,
            kind="linear",
            bounds_error=True
        )

    def __call__(self, wavelength):
        """
        Interpolate the extinction coefficient for the given wavelength (in meters).
        Raises ValueError if wavelength is out of bounds.
        """
        wavelength = normalize_length(wavelength)
        # Convert nm to μm for interpolation
        min_wl = self._wavelengths[0]
        max_wl = self._wavelengths[-1]
        if np.any(np.less_equal(wavelength, min_wl)) or np.any(np.greater_equal(wavelength, max_wl)):
            raise ValueError(f"Wavelength {wavelength*1e9} nm is out of bounds ({min_wl*1e9} nm..{max_wl*1e9} nm) for Hale-Querry model.")
        return self._interp(wavelength)
