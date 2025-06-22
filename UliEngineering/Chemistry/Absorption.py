#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Length import normalize_length
import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    "AbsorptionData",
    "HaleQuerryAbsorptionModel",
    "absorption_length_from_extinction_coefficient",
    "extinction_coefficient_from_absorption_length",
    "remaining_light_fraction",
    "length_from_remaining_fraction",
    "half_length",
]

@dataclass
class AbsorptionData:
    wavelength: float
    extinction_coefficient: float
    refractive_index: float
    extra: Optional[float] = None
    
class HaleQuerryAbsorptionModel:
    """
    Hale-Querry absorption model for water. Valid from 200nm to 200μm.
    Interpolated using a piecewise linear function.

    Input wavelength must be in nanometers (nm).

    Source: Hale, G. M., & Querry, M. R. (1973). Optical constants of water in the 200-nm to 200-μm wavelength region, Table 1
    Document available at: https://opg.optica.org/ao/viewmedia.cfm?uri=ao-12-3-555&seq=0
    """
    datapoints = [
        AbsorptionData(200.0, 1.1e-7, 1.396),
        AbsorptionData(225.0, 4.9e-8, 1.373),
        AbsorptionData(250.0, 3.35e-8, 1.362),
        AbsorptionData(275.0, 2.35e-8, 1.354),
        AbsorptionData(300.0, 1.6e-8, 1.349),
        AbsorptionData(325.0, 1.08e-8, 1.346),
        AbsorptionData(350.0, 6.5e-9, 1.343),
        AbsorptionData(375.0, 3.5e-9, 1.341),
        AbsorptionData(400.0, 1.86e-9, 1.339),
        AbsorptionData(425.0, 1.3e-9, 1.338),
        AbsorptionData(450.0, 1.02e-9, 1.337),
        AbsorptionData(475.0, 9.35e-10, 1.336),
        AbsorptionData(500.0, 1.00e-9, 1.335),
        AbsorptionData(525.0, 1.32e-9, 1.334),
        AbsorptionData(550.0, 1.96e-9, 1.333),
        AbsorptionData(575.0, 3.60e-9, 1.333),
        AbsorptionData(600.0, 1.09e-8, 1.332),
        AbsorptionData(625.0, 1.39e-8, 1.332),
        AbsorptionData(650.0, 1.64e-8, 1.331),
        AbsorptionData(675.0, 2.29e-8, 1.331),
        AbsorptionData(700.0, 3.35e-8, 1.331),
        AbsorptionData(725.0, 9.15e-8, 1.330),
        AbsorptionData(750.0, 1.56e-7, 1.330),
        AbsorptionData(775.0, 1.48e-7, 1.330),
        AbsorptionData(800.0, 1.25e-7, 1.329),
        AbsorptionData(825.0, 1.82e-7, 1.329),
        AbsorptionData(850.0, 2.93e-7, 1.329),
        AbsorptionData(875.0, 3.91e-7, 1.328, 6.0),
        AbsorptionData(900.0, 4.86e-7, 1.328, 6.1),
        AbsorptionData(925.0, 1.06e-6, 1.328, 6.2),
        AbsorptionData(950.0, 2.93e-6, 1.327, 6.3),
        AbsorptionData(975.0, 3.48e-6, 1.327, 6.4),
        AbsorptionData(1000.0, 2.89e-6, 1.327, 6.5),
        AbsorptionData(1200.0, 9.89e-6, 1.324, 6.6),
        AbsorptionData(1400.0, 1.38e-4, 1.321, 6.7),
        AbsorptionData(1600.0, 8.55e-5, 1.317, 6.8),
        AbsorptionData(1800.0, 1.15e-4, 1.312, 6.9),
        AbsorptionData(2000.0, 1.1e-3, 1.306, 7.0),
        AbsorptionData(2200.0, 2.89e-4, 1.296, 7.1),
        AbsorptionData(2400.0, 9.56e-4, 1.279, 7.2),
        AbsorptionData(2600.0, 3.17e-3, 1.242, 7.3),
        AbsorptionData(2650.0, 6.7e-3, 1.219, 7.4),
        AbsorptionData(2700.0, 0.019, 1.188, 7.5),
        AbsorptionData(2750.0, 0.059, 1.157, 7.6),
        AbsorptionData(2800.0, 0.115, 1.142, 7.7),
        AbsorptionData(2850.0, 0.185, 1.149, 7.8),
        AbsorptionData(2900.0, 0.268, 1.201, 7.9),
        AbsorptionData(2950.0, 0.298, 1.292, 8.0),
        AbsorptionData(3000.0, 0.272, 1.371, 8.2),
        AbsorptionData(3050.0, 0.240, 1.426, 8.4),
        AbsorptionData(3100.0, 0.192, 1.467, 8.6),
        AbsorptionData(3150.0, 0.135, 1.483, 8.8),
        AbsorptionData(3200.0, 0.0924, 1.478, 9.0),
        AbsorptionData(3250.0, 0.0610, 1.467, 9.2),
        AbsorptionData(3300.0, 0.0368, 1.450, 9.4),
        AbsorptionData(3350.0, 0.0261, 1.432, 9.6),
        AbsorptionData(3400.0, 0.0195, 1.420),
        AbsorptionData(3450.0, 0.0132, 1.410),
        AbsorptionData(3500.0, 0.0094, 1.400),
        AbsorptionData(3600.0, 0.00515, 1.385),
        AbsorptionData(3700.0, 0.00360, 1.374),
        AbsorptionData(3800.0, 0.00340, 1.364),
        AbsorptionData(3900.0, 0.00380, 1.357),
        AbsorptionData(4000.0, 0.00460, 1.351),
        AbsorptionData(4100.0, 0.00562, 1.346),
        AbsorptionData(4200.0, 0.00688, 1.342),
        AbsorptionData(4300.0, 0.00845, 1.338),
        AbsorptionData(4400.0, 0.0103, 1.334),
        AbsorptionData(4500.0, 0.0134, 1.332),
        AbsorptionData(4600.0, 0.0147, 1.330),
        AbsorptionData(4700.0, 0.0157, 1.330),
        AbsorptionData(4800.0, 0.0150, 1.330),
        AbsorptionData(4900.0, 0.0137, 1.328),
        AbsorptionData(5000.0, 0.0124, 1.325),
        AbsorptionData(5100.0, 0.0111, 1.322),
        AbsorptionData(5200.0, 0.0101, 1.317),
        AbsorptionData(5300.0, 0.0090, 1.312),
        AbsorptionData(5400.0, 0.0103, 1.305),
        AbsorptionData(5500.0, 0.0116, 1.298),
        AbsorptionData(5600.0, 0.0142, 1.289),
        AbsorptionData(5700.0, 0.0203, 1.277),
        AbsorptionData(5800.0, 0.0330, 1.262),
        AbsorptionData(5900.0, 0.0622, 1.248),
        AbsorptionData(6000.0, 0.107, 1.265),
        AbsorptionData(6100.0, 0.131, 1.319),
        AbsorptionData(6200.0, 0.0880, 1.363),
        AbsorptionData(6300.0, 0.0570, 1.357),
        AbsorptionData(6400.0, 0.0449, 1.347),
        AbsorptionData(6500.0, 0.0392, 1.339),
        AbsorptionData(6600.0, 0.0356, 1.334),
        AbsorptionData(6700.0, 0.0337, 1.329),
        AbsorptionData(6800.0, 0.0327, 1.324),
        AbsorptionData(6900.0, 0.0322, 1.321),
        AbsorptionData(7000.0, 0.0320, 1.317),
        AbsorptionData(7100.0, 0.0320, 1.314),
        AbsorptionData(7200.0, 0.0321, 1.312),
        AbsorptionData(7300.0, 0.0322, 1.309),
        AbsorptionData(7400.0, 0.0324, 1.307),
        AbsorptionData(7500.0, 0.0326, 1.304),
        AbsorptionData(7600.0, 0.0328, 1.302),
        AbsorptionData(7700.0, 0.0331, 1.299),
        AbsorptionData(7800.0, 0.0335, 1.297),
        AbsorptionData(7900.0, 0.0339, 1.294),
        AbsorptionData(8000.0, 0.0343, 1.291),
        AbsorptionData(8200.0, 0.0351, 1.286),
        AbsorptionData(8400.0, 0.0361, 1.281),
        AbsorptionData(8600.0, 0.0372, 1.275),
        AbsorptionData(8800.0, 0.0385, 1.269),
        AbsorptionData(9000.0, 0.0399, 1.262),
        AbsorptionData(9200.0, 0.0415, 1.255),
        AbsorptionData(9400.0, 0.0433, 1.247),
        AbsorptionData(9600.0, 0.0454, 1.239),
        AbsorptionData(9800.0, 0.0479, 1.229),
        AbsorptionData(10000.0, 0.0508, 1.218),
        AbsorptionData(10500.0, 0.0662, 1.185),
        AbsorptionData(11000.0, 0.0968, 1.153),
        AbsorptionData(11500.0, 0.142, 1.126),
        AbsorptionData(12000.0, 0.199, 1.111),
        AbsorptionData(12500.0, 0.259, 1.123),
        AbsorptionData(13000.0, 0.305, 1.146),
        AbsorptionData(13500.0, 0.343, 1.177),
        AbsorptionData(14000.0, 0.370, 1.210),
        AbsorptionData(14500.0, 0.388, 1.241),
        AbsorptionData(15000.0, 0.402, 1.270),
        AbsorptionData(15500.0, 0.414, 1.297),
        AbsorptionData(16000.0, 0.422, 1.325),
        AbsorptionData(16500.0, 0.428, 1.351),
        AbsorptionData(17000.0, 0.429, 1.376),
        AbsorptionData(17500.0, 0.429, 1.401),
        AbsorptionData(18000.0, 0.426, 1.423),
        AbsorptionData(18500.0, 0.421, 1.443),
        AbsorptionData(19000.0, 0.414, 1.461),
        AbsorptionData(19500.0, 0.404, 1.476),
        AbsorptionData(20000.0, 0.393, 1.480),
        AbsorptionData(21000.0, 0.382, 1.487),
        AbsorptionData(22000.0, 0.373, 1.500),
        AbsorptionData(23000.0, 0.367, 1.511),
        AbsorptionData(24000.0, 0.361, 1.521),
        AbsorptionData(25000.0, 0.356, 1.531),
        AbsorptionData(26000.0, 0.350, 1.539),
        AbsorptionData(27000.0, 0.344, 1.545),
        AbsorptionData(28000.0, 0.338, 1.549),
        AbsorptionData(29000.0, 0.333, 1.551),
        AbsorptionData(30000.0, 0.328, 1.551),
        AbsorptionData(32000.0, 0.324, 1.546),
        AbsorptionData(34000.0, 0.329, 1.536),
        AbsorptionData(36000.0, 0.343, 1.527),
        AbsorptionData(38000.0, 0.361, 1.522),
        AbsorptionData(40000.0, 0.385, 1.519),
        AbsorptionData(42000.0, 0.409, 1.522),
        AbsorptionData(44000.0, 0.436, 1.530),
        AbsorptionData(46000.0, 0.462, 1.541),
        AbsorptionData(48000.0, 0.488, 1.555),
        AbsorptionData(50000.0, 0.514, 1.587),
        AbsorptionData(60000.0, 0.587, 1.703),
        AbsorptionData(70000.0, 0.576, 1.821),
        AbsorptionData(80000.0, 0.547, 1.886),
        AbsorptionData(90000.0, 0.536, 1.924),
        AbsorptionData(100000.0, 0.532, 1.957),
        AbsorptionData(110000.0, 0.531, 1.966),
        AbsorptionData(120000.0, 0.526, 2.004),
        AbsorptionData(130000.0, 0.514, 2.036),
        AbsorptionData(140000.0, 0.500, 2.056),
        AbsorptionData(150000.0, 0.495, 2.069),
        AbsorptionData(160000.0, 0.496, 2.081),
        AbsorptionData(170000.0, 0.497, 2.094),
        AbsorptionData(180000.0, 0.499, 2.107),
        AbsorptionData(190000.0, 0.501, 2.119),
        AbsorptionData(200000.0, 0.504, 2.130),
    ]

    def __init__(self):
        # Extract wavelength and extinction coefficient arrays
        self._wavelengths = np.array([d.wavelength for d in self.datapoints])
        self._ext_coeffs = np.array([d.extinction_coefficient for d in self.datapoints])
        # Build linear interpolation model
        self._interp = interp1d(
            self._wavelengths,
            self._ext_coeffs,
            kind="linear",
            bounds_error=True
        )

    def __call__(self, wavelength):
        """
        Interpolate the extinction coefficient for the given wavelength (in nanometers, nm).
        Raises ValueError if wavelength is out of bounds.
        """
        # Convert nm to μm for interpolation
        min_wl = self._wavelengths[0]
        max_wl = self._wavelengths[-1]
        if np.any(np.less(wavelength, min_wl)) or np.any(np.greater(wavelength, max_wl)):
            raise ValueError(f"Wavelength {wavelength} nm is out of bounds ({min_wl*1000}–{max_wl*1000} nm) for Hale-Querry model.")
        return self._interp(wavelength)

@normalize_numeric_args
@returns_unit("m")
def absorption_length_from_extinction_coefficient(extinction_coefficient):
    """
    Compute the absorption length (in meters) from the extinction coefficient (in 1/m).
    Absorption length is defined as the distance over which the intensity drops to 1/e.
    Formula: absorption_length = 1 / extinction_coefficient

    Parameters:
    - extinction_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray).

    Returns:
    - Absorption length in meters.
    """
    return np.reciprocal(extinction_coefficient)

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
def remaining_light_fraction(length, extinction_coefficient):
    """
    Compute the remaining fraction of light after passing through a medium of given length (in meters)
    with a given extinction coefficient (in 1/m).

    Formula: fraction = exp(-extinction_coefficient * length)

    Parameters:
    - length: Length of the medium (scalar, list, or ndarray, any unit parsable by normalize_length)
    - extinction_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Remaining fraction of light (unitless)
    """
    length = normalize_length(length)
    extinction_coefficient = normalize_numeric(extinction_coefficient)
    return np.exp(-extinction_coefficient * length)

@normalize_numeric_args
@returns_unit("m")
def length_from_remaining_fraction(remaining_fraction, extinction_coefficient):
    """
    Compute the length of the medium (in meters) given the remaining fraction of light
    and the extinction coefficient (in 1/m).

    Formula: length = -ln(remaining_fraction) / extinction_coefficient

    Parameters:
    - remaining_fraction: Remaining fraction of light (scalar, list, or ndarray, unitless)
    - extinction_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Length in meters.
    """
    return -np.log(remaining_fraction) / extinction_coefficient

@normalize_numeric_args
@returns_unit("m")
def half_length(extinction_coefficient):
    """
    Compute the half-length, i.e., the length of medium where the remaining fraction of light is 0.5,
    for a given extinction coefficient (in 1/m).

    Parameters:
    - extinction_coefficient: Extinction coefficient in 1/m (scalar, list, or ndarray)

    Returns:
    - Half-length in meters.
    """
    return length_from_remaining_fraction(0.5, extinction_coefficient)
