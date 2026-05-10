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
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
import numpy as np

__all__ = [
    "selectivity_coefficient",
    "separation_factor",
    "distribution_coefficient",
    "ion_exchange_capacity_from_breakthrough",
    "donnan_potential",
    "donnan_ratio",
    "normalize_concentration", "ConcentrationMolar",
    "normalize_volume", "VolumeLiter",
    "normalize_mass", "MassGram",
]


def normalize_concentration(c: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(c, {"mol/L": 1.0, "M": 1.0, "mM": 1e-3, "µM": 1e-6, "mol/m³": 1e-3}, quantity_name="concentration")

def normalize_volume(V: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(V, {"L": 1.0, "mL": 1e-3, "µL": 1e-6, "m³": 1000.0, "cm³": 1e-3}, quantity_name="volume")

def normalize_mass(m: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(m, {"g": 1.0, "mg": 1e-3, "µg": 1e-6, "kg": 1000.0}, quantity_name="mass")

ConcentrationMolar = Annotated[NormalizedComputable, normalize_concentration]
VolumeLiter = Annotated[NormalizedComputable, normalize_volume]
MassGram = Annotated[NormalizedComputable, normalize_mass]


@returns_unit("")
def selectivity_coefficient(q_A: ConcentrationMolar, C_B: ConcentrationMolar, q_B: ConcentrationMolar, C_A: ConcentrationMolar, z_A=1, z_B=1):
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
    q_A = normalize_concentration(q_A) if isinstance(q_A, str) else q_A
    C_B = normalize_concentration(C_B) if isinstance(C_B, str) else C_B
    q_B = normalize_concentration(q_B) if isinstance(q_B, str) else q_B
    C_A = normalize_concentration(C_A) if isinstance(C_A, str) else C_A
    return (q_A**z_B * C_B**z_A) / (q_B**z_A * C_A**z_B)


@returns_unit("")
def separation_factor(q_A: ConcentrationMolar, C_B: ConcentrationMolar, q_B: ConcentrationMolar, C_A: ConcentrationMolar):
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
    q_A = normalize_concentration(q_A) if isinstance(q_A, str) else q_A
    C_B = normalize_concentration(C_B) if isinstance(C_B, str) else C_B
    q_B = normalize_concentration(q_B) if isinstance(q_B, str) else q_B
    C_A = normalize_concentration(C_A) if isinstance(C_A, str) else C_A
    return (q_A * C_B) / (q_B * C_A)


@returns_unit("L/g")
def distribution_coefficient(q, C: ConcentrationMolar):
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
    C = normalize_concentration(C) if isinstance(C, str) else C
    return q / C


@returns_unit("mol/g")
def ion_exchange_capacity_from_breakthrough(C_feed: ConcentrationMolar, V_breakthrough: VolumeLiter, mass_resin: MassGram):
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
    C_feed = normalize_concentration(C_feed) if isinstance(C_feed, str) else C_feed
    V_breakthrough = normalize_volume(V_breakthrough) if isinstance(V_breakthrough, str) else V_breakthrough
    mass_resin = normalize_mass(mass_resin) if isinstance(mass_resin, str) else mass_resin
    return C_feed * V_breakthrough / mass_resin


@returns_unit("V")
def donnan_potential(z, C_in: ConcentrationMolar, C_out: ConcentrationMolar, T=298.15):
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
    C_in = normalize_concentration(C_in) if isinstance(C_in, str) else C_in
    C_out = normalize_concentration(C_out) if isinstance(C_out, str) else C_out
    T = normalize_temperature(T) if isinstance(T, str) else T
    F = physical_constants["Faraday constant"][0]
    return (R * T) / (z * F) * np.log(C_out / C_in)


@returns_unit("")
def donnan_ratio(z, C_fixed: ConcentrationMolar, C_solution: ConcentrationMolar):
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
    C_fixed = normalize_concentration(C_fixed) if isinstance(C_fixed, str) else C_fixed
    C_solution = normalize_concentration(C_solution) if isinstance(C_solution, str) else C_solution
    C_co = (-C_fixed + np.sqrt(C_fixed**2 + 4.0 * C_solution**2)) / 2.0
    return C_co / C_solution
