#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Light utilities."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ._normalize import normalize_with_known_units
import numpy as np

__all__ = ["lumen_to_candela_by_apex_angle",
           "normalize_luminous_flux", "LuminousFluxLumen",
           "normalize_angle_degrees", "AngleDegrees"]

def normalize_luminous_flux(flux: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(flux, {"lm": 1.0, "lumen": 1.0, "lumens": 1.0}, quantity_name="luminous flux")

def normalize_angle_degrees(angle: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(angle, {"°": 1.0, "deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 57.29577951308232}, quantity_name="angle")

LuminousFluxLumen = Annotated[NormalizedComputable, normalize_luminous_flux]
AngleDegrees = Annotated[NormalizedComputable, normalize_angle_degrees]

@returns_unit("cd")
def lumen_to_candela_by_apex_angle(flux: LuminousFluxLumen, angle: AngleDegrees):
    """
    Compute the luminous intensity from the luminous flux.
    
    assuming that the flux of <flux> is distributed equally around.
    a cone with apex angle <angle>.

    Parameters
    ----------
    flux : LuminousFluxLumen
        The luminous flux in lumens.
    angle : AngleDegrees
        The apex angle of the emission cone, in degrees.

    Returns
    -------
    float
        Luminous intensity in candela.

    Examples
    --------
    >>> autoFormat(lumen_to_candela_by_apex_angle, "25 lm", "120°")
    '7.96 cd'
    
    """
    flux = normalize_luminous_flux(flux)
    angle = normalize_angle_degrees(angle)
    solid_angle = 2*np.pi*(1.-np.cos(np.deg2rad(angle)/2.0))
    return flux / solid_angle
