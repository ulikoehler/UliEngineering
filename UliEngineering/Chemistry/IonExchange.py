#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ion exchange equilibrium and selectivity.

Provides functions for:
- Ion exchange selectivity coefficient
- Separation factor
- Distribution coefficient (Kd)
- Ion exchange capacity
- Donnan equilibrium
"""
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import numpy as np

__all__ = [
    "selectivity_coefficient",
    "separation_factor",
    "distribution_coefficient",
    "ion_exchange_capacity_from_breakthrough",
    "donnan_potential",
    "donnan_ratio",
]


@normalize_numeric_args
@returns_unit("")
def selectivity_coefficient(q_A, C_B, q_B, C_A, z_A=1, z_B=1):
    """
    Compute the selectivity coefficient for ion exchange of A over B.

    For exchange: z_B * A^(z_A) + z_A * B_resin^(z_B) ⇌ z_B * A_resin^(z_A) + z_A * B^(z_B)

    K_AB = (q_A^z_B * C_B^z_A) / (q_B^z_A * C_A^z_B)

    Parameters
    ----------
    q_A : float
        Resin-phase concentration of ion A.
    C_B : float
        Solution-phase concentration of ion B.
    q_B : float
        Resin-phase concentration of ion B.
    C_A : float
        Solution-phase concentration of ion A.
    z_A : int
        Charge of ion A (default: 1).
    z_B : int
        Charge of ion B (default: 1).

    Returns
    -------
    float
        Selectivity coefficient K_AB.
    """
    return (q_A**z_B * C_B**z_A) / (q_B**z_A * C_A**z_B)


@normalize_numeric_args
@returns_unit("")
def separation_factor(q_A, C_B, q_B, C_A):
    """
    Compute the separation factor α for ion exchange.

    α_AB = (q_A * C_B) / (q_B * C_A)

    Parameters
    ----------
    q_A : float
        Resin-phase concentration of A.
    C_B : float
        Solution-phase concentration of B.
    q_B : float
        Resin-phase concentration of B.
    C_A : float
        Solution-phase concentration of A.

    Returns
    -------
    float
        Separation factor α (>1 means A is preferred).
    """
    return (q_A * C_B) / (q_B * C_A)


@normalize_numeric_args
@returns_unit("L/g")
def distribution_coefficient(q, C):
    """
    Compute the distribution coefficient Kd.

    Kd = q / C

    Parameters
    ----------
    q : float
        Amount adsorbed per unit mass of exchanger (e.g., mol/g or meq/g).
    C : float
        Equilibrium solution concentration (mol/L or meq/L).

    Returns
    -------
    float
        Distribution coefficient in L/g.
    """
    return q / C


@normalize_numeric_args
@returns_unit("mol/g")
def ion_exchange_capacity_from_breakthrough(C_feed, V_breakthrough, mass_resin):
    """
    Compute ion exchange capacity from breakthrough experiment.

    Q = C_feed * V_breakthrough / m_resin

    Parameters
    ----------
    C_feed : float
        Feed concentration in mol/L.
    V_breakthrough : float
        Volume processed until breakthrough in liters.
    mass_resin : float
        Mass of ion exchange resin in grams.

    Returns
    -------
    float
        Ion exchange capacity in mol/g.
    """
    return C_feed * V_breakthrough / mass_resin


@normalize_numeric_args
@returns_unit("V")
def donnan_potential(z, C_in, C_out, T=298.15):
    """
    Compute the Donnan membrane potential.

    E_D = (R*T) / (z*F) * ln(C_out / C_in)

    Parameters
    ----------
    z : float
        Charge number of the ion.
    C_in : float
        Ion concentration inside the membrane (mol/L).
    C_out : float
        Ion concentration outside the membrane (mol/L).
    T : float
        Temperature in Kelvin (default: 298.15 K).

    Returns
    -------
    float
        Donnan potential in Volts.
    """
    from scipy.constants import R, physical_constants
    F = physical_constants["Faraday constant"][0]
    return (R * T) / (z * F) * np.log(C_out / C_in)


@normalize_numeric_args
@returns_unit("")
def donnan_ratio(z, C_fixed, C_solution):
    """
    Compute the Donnan ratio for a membrane with fixed charge.

    For a monovalent case with fixed charge concentration C_fixed:
        r = C_in / C_out

    For symmetric 1:1 electrolyte:
        C_co = (-C_fixed + √(C_fixed² + 4*C_solution²)) / 2
        r = C_co / C_solution

    Parameters
    ----------
    z : float
        Charge number (absolute value, for the co-ion).
    C_fixed : float
        Fixed charge concentration in the membrane (mol/L).
    C_solution : float
        External electrolyte concentration (mol/L).

    Returns
    -------
    float
        Donnan ratio (C_co-ion_in / C_solution).
    """
    C_co = (-C_fixed + np.sqrt(C_fixed**2 + 4.0 * C_solution**2)) / 2.0
    return C_co / C_solution
