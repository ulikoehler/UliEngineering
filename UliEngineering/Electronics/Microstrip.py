#!/usr/bin/env python3
"""
Microstrip utilities
"""
import scipy.constants
import numpy as np
import math
from collections import namedtuple

from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.EngineerIO.Length import normalize_length

__all__ = ["Z0", "microstrip_impedance", "differential_microstrip_impedance",
           "RelativePermittivity", "microstrip_width"]

Z0 = scipy.constants.physical_constants['characteristic impedance of vacuum'][0]


class RelativePermittivity():
    """
    Default values for relative permittivity of different materials
    
    Best to choose a specific value for your material, since these vary widely
    """
    FR4 = 4.8 # Varies widely (approximate range: 3.9..4.8)

@returns_unit("m")
def microstrip_width(Z0="50 Ω", h="140 μm", t="35 μm", e_r=RelativePermittivity.FR4, max_iter:int=1000, tol:float=1e-9):
    """
    Compute the width of a single-ended outer-layer microstrip given its impedance,
    height, thickness, and the relative permittivity of the substrate.

    This uses an iterative approach to solve microstrip_impedance()
    for width since there's no closed-form solution.
    (Newton-Raphson method)

    Parameters
    ----------
    Z0 : number or engineer string
        The characteristic impedance of the microstrip in ohms
    h : number or engineer string
        Trace height of the substrate between the bottom
        of the microstrip and the ground plane (converted to meters)
    t : number or engineer string
        The trace thickness of the microstrip (converted to meters)
    e_r : number or engineer string
        Relative permittivity of the dielectric
    max_iter : int
        Maximum number of iterations for the solver
    tol : float
        Tolerance for convergence in ohms

    Returns
    -------
    float
        The width of the microstrip in meters
    """
    Z0 = normalize_numeric(Z0)
    h = normalize_length(h)
    t = normalize_length(t)
    e_r = normalize_numeric(e_r)
    # Initial guess using simplified formula (for thin traces)
    w_guess = h * (8 * math.exp(2 * Z0 * math.sqrt(e_r + 1) / 377) - 2) / (math.exp(2 * Z0 * math.sqrt(e_r + 1) / 377) + 2)

    # Iterative solution
    for _ in range(max_iter):
        current_Z0 = microstrip_impedance(w_guess, h, t, e_r)
        error = current_Z0 - Z0

        if abs(error) < tol:
            return w_guess

        # Numerical derivative approximation
        delta = max(w_guess * 1e-6, 1e-12)
        Z0_plus = microstrip_impedance(w_guess + delta, h, t, e_r)
        derivative = (Z0_plus - current_Z0) / delta
        
        if derivative == 0:
            break

        # Newton-Raphson update
        w_new = w_guess - error / derivative
        if w_new <= 0:
            w_new = w_guess / 2.0
        w_guess = w_new

    # Check if we converged
    current_Z0 = microstrip_impedance(w_guess, h, t, e_r)
    if abs(current_Z0 - Z0) > tol:
        raise ValueError(f"Could not converge to Z0={Z0} Ohm. Best guess w={w_guess} m gave Z0={current_Z0} Ohm")
    
    return w_guess

@returns_unit("Ω")
def microstrip_impedance(w, h="140 μm", t="35 μm", e_r=RelativePermittivity.FR4):
    """
    Compute the impedance of a single-eded
    outer-layer microstrip using its width, height and
    the relative permittivity of the substrate.

    We use a more exact equation involving the strip height

    Ref: https://www.allaboutcircuits.com/tools/microstrip-impedance-calculator/

    Parameters
    ----------
    w : number or engineer string
        The trace width of the microstrip
    h : number or engineer string
        Trace height of the substrate between the bottom
        of the microstrip and the ground plane
    t : number or engineer string
        The trace thickness of the microstrip
    e_r : number or engineer string
        Relative permittivity of the dielectric
    """
    w = normalize_length(w)
    h = normalize_length(h)
    t = normalize_length(t)
    e_r = normalize_numeric(e_r)
    # Formula from https://www.allaboutcircuits.com/tools/microstrip-impedance-calculator/
    Y0 = np.square(t / (w * math.pi + 1.1 * t * math.pi))
    Y1 = math.sqrt(np.square(t / h) + Y0)
    Y2 = (e_r + 1) / (2 * e_r)
    weff = w + (t / math.pi) * np.log((4 * math.e) / Y1) * Y2
    hdweff = (h / weff)
    X0 = (14 * e_r + 8) / (11 * e_r)
    X1 = 4 * X0 * hdweff
    X2 = math.sqrt(16 * np.square(hdweff) * np.square(X0) +
                 Y2 * np.square(math.pi))
    return (Z0 / (2 * math.pi * math.sqrt(2) * math.sqrt(e_r + 1))) * np.log(1 + 4 * hdweff * (X1 + X2))

DifferentialMicrostripImpedance = namedtuple("DifferentialMicrostripImpedance", [
    "single_ended_impedance",
    "differential_impedance",
    "even_mode_impedance",
    "odd_mode_impedance"
])

@returns_unit("Ω")
@normalize_numeric_args
def differential_microstrip_impedance(w, d, h="140μm", t="35 μm", e_r=RelativePermittivity.FR4):
    """
    Compute the impedance of a differential (edge-coupled)
    outer-layer microstrip using its width, height, the distance
    between the edges of the microstrips,
    the height of the substrate beneath the microstrip
    and the relative permittivity of the substrate.

    NOTE: Due to the available closed-form formulae, the differential impedance
          does not affect the differential impedance. It is only used for the
          single-ended impedance.

    NOTE: Odd modes are present when the microstrip is driven to a different polarity
    NOTE: Even modes are present when the microstrip is driven to the same polarity

    Ref: https://www.eeweb.com/tools/edge-coupled-microstrip-impedance

    Parameters
    ----------
    w : number or engineer string
        The trace width of the microstrip
    d : number or engineer string
        Distance between the edges of the microstrip lines
    h : number or engineer string
        Trace height of the substrate between the bottom
        of the microstrip and the ground plane
    t : number or engineer string
        The trace thickness of the microstrip
    e_r : number or engineer string
        Relative permittivity of the dielectric
    """
    # Formula from https://www.allaboutcircuits.com/tools/edge-coupled-microstrip-impedance-calculator/
    # Secondary source: https://www.eeweb.com/tools/edge-coupled-microstrip-impedance
    # NOTE: Uppercase first-letter variables are just utilitarian
    # and do not have a separate formula on the paper
    # Simple variables & parameters
    eta_0 = 377
    u = w / h
    g = d / h
    g10 = g**10
    # Step 1: Determine effective e_r 
    E0 = math.sqrt(w/(w + 12*h))
    E1 = (e_r + 1)/2
    E2 = (e_r-1)/2
    er_eff1 = E1 + E2 * (E0 + 0.04 * (1.-u)**2)
    er_eff2 = E1 + E2 * (E0)
    #var er_eff1 = ((er+1)/2)+((er-1)/2)*(Math.sqrt(w/(w+12*h))+.04*Math.pow((1-(w/h)),2));
    #var er_eff2 = ((er+1)/2)+((er-1)/2)*(Math.sqrt(w/(w+12*h)));
    er_eff = er_eff1 if u < 1 else er_eff2
    a0 = 0.7287*(er_eff - (e_r+1) / 2) * (1 - math.exp(-0.179*u));
    b0 = (0.747 * e_r) / (.15 + e_r);
    c0 = b0 - (b0 - 0.207) * math.exp(-0.414 * u);
    d0 = 0.694 * math.exp(-0.562 * u) + 0.593
    # weff
    Y0 = (t / (w*math.pi + 1.1 * t *math.pi))**2
    Y1 = math.sqrt((t / h)**2 + Y0)
    Y2 = (er_eff + 1) / (2 * er_eff)
    weff = w + (t / math.pi) * math.log((4 * math.e) / Y1) * Y2
    hdweff = h/weff
    # Surface characteristic impedance
    ZT0 = (eta_0/(2*math.pi*math.sqrt(2)*math.sqrt(er_eff+1)))
    ZT1 = (14*er_eff+8) / (11*er_eff)
    ZT2 = math.sqrt(16* hdweff**2 * ZT1**2 + Y1*math.pi**2)
    z0surf = ZT0*math.log(1+4*hdweff*(4*hdweff*(ZT1)+ZT2))
    # Q intermediates
    q1 = 0.8695 * u**0.194
    q2 = 1 + 0.7519*g + 1.89*g**2.31
    q3 = 0.1975 + (16.6 + (8.4 / g)**6)**-0.387 + (1./241.)*math.log(g10/(1 + (g/3.4)**10))
    q4 = (2 * q1) / (q2 * (math.exp(-g) * u**q3 + (2 - math.exp(-g)) * u**-q3))
    q5 = 1.794 + 1.14 * math.log(1 + (0.638 / (g + 0.517 * g**2.43)))
    q6 = 0.2305 + (1./281.3)*math.log(g10/(1 + (g/5.8)**10)) + (1./5.1)*math.log(1 + 0.598 * g**1.154)
    q7 = (10 + 190 * g**2) / (1+82.3 * g**3)
    q8 = math.exp(-6.5-0.95 * math.log(g) - (g / .15)**5)
    q9 = math.log(q7) * (q8 + 1./16.5)
    q10 = (1./q2) * (q2 * q4 - q5 * math.exp(math.log(u) * q6 * u**-q9))
    # Odd mode effective e_r
    er_eff_o = ((0.5*(e_r + 1) + a0 - er_eff) * math.exp(-c0 * g**d0)) + er_eff
    # Odd characteristic impedance
    z0odd = (z0surf * math.sqrt(er_eff/er_eff_o)) / (1 - (z0surf/eta_0) * q10 * math.sqrt(er_eff))
    z0diff = z0odd*2
    # ...
    v = (u * (20 + g**2)) / (10 + g**2) + g*math.exp(-g)
    ae_v = (1 + (math.log((v**4 + (v/52)**2) / (v**4 + 0.432)) / 49)) + (math.log(1 + (v/18.1)**3) / 18.7)
    be_er =  0.564 * ((e_r - 0.9)/(e_r + 3))**0.053
    er_eff_e = 0.5*(e_r + 1) + 0.5*(e_r - 1) * (10/v)**(-ae_v * be_er)
    # Even characteristic impedance
    z0even = (z0surf * math.sqrt(er_eff/er_eff_e)) / (1 - (z0surf/eta_0) * q4 * math.sqrt(er_eff))
    # Single-ended impedances
    single_ended_impedance = microstrip_impedance(w, h=h, t=t, e_r=e_r)
    return DifferentialMicrostripImpedance(single_ended_impedance, z0diff, z0even, z0odd)
