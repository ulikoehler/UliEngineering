#!/usr/bin/env python3
"""
Microstrip utilities
"""
import scipy.constants
import numpy as np
from UliEngineering.Length import normalize_length
from UliEngineering.Units import Unit

__all__ = ["Z0", "microstrip_impedance",
           "RelativePermittivity", "DielectricHeight"]

Z0 = scipy.constants.physical_constants['characteristic impedance of vacuum'][0]


class RelativePermittivity():
    """
    Default values for relative permittivity of different materials
    """
    FR4 = 4.8


class DielectricHeight():
    """
    Default values for the thickness of the dielectric for
    different PCB configurations.
    """
    L4_1p6mm = "140 μm"
    """
    4 layer PCB with 1.6mm total thickness
    https://www.multi-circuit-boards.eu/en/pcb-design-aid/layer-buildup/standard-buildup.html
    """

def microstrip_impedance(w, h=DielectricHeight.L4_1p6mm, t="35 μm", e_r=RelativePermittivity.FR4) -> Unit("Ω"):
    """
    Compute the impedance of an outer-layer microstrip using

    We use a more exact equation involving the strip height

    Ref: https://www.allaboutcircuits.com/tools/microstrip-impedance-calculator/

    Parameters
    ----------
    w : float or 1D np array
        The trace width of the microstrip
    h : float or 1D np array
        Trace height of the substrate between the bottom
        of the microstrip and the ground plane
    t : float or 1D np array
        The trace thickness of the microstrip
    e_r : float or 1D np array
        Relative permittivity of the dielectric
    """
    w = normalize_length(w)
    h = normalize_length(h)
    t = normalize_length(t)
    # Formula from https://www.allaboutcircuits.com/tools/microstrip-impedance-calculator/
    Y0 = np.square(t / (w * np.pi + 1.1 * t * np.pi))
    Y1 = np.sqrt(np.square(t / h) + Y0)
    Y2 = (e_r + 1) / (2 * e_r)
    weff = w + (t / np.pi) * np.log((4 * np.e) / Y1) * Y2
    hdweff = (h / weff)
    X0 = (14 * e_r + 8) / (11 * e_r)
    X1 = 4 * X0 * hdweff
    X2 = np.sqrt(16 * np.square(hdweff) * np.square(X0) +
                 Y2 * np.square(np.pi))
    return (Z0 / (2 * np.pi * np.sqrt(2) * np.sqrt(e_r + 1))) * np.log(1 + 4 * hdweff * (X1 + X2))
