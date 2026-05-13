#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitzer equations for activity coefficients.

The Pitzer model is a semi-empirical extension of Debye-Hückel theory.
for concentrated electrolyte solutions (up to ~6 mol/kg).

For a single-electrolyte solution MX:
    ln(γ±) = |z+ * z-| * f^γ + m * B^γ + m² * C^γ

where:
    f^γ  = -A_φ * (√I / (1 + b*√I) + (2/b) * ln(1 + b*√I))
    B^γ  = 2*β₀ + 2*β₁/(α²*I) * [1 - (1 + α*√I - α²*I/2) * exp(-α*√I)]
    C^γ  = 1.5 * C^φ

Standard parameters: b = 1.2, α = 2.0, A_φ ≈ 0.3915 at 25 °C
"""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from ..Physics._normalize import normalize_with_known_units
import numpy as np

__all__ = [
    "pitzer_f_gamma",
    "pitzer_B_gamma",
    "pitzer_C_gamma",
    "pitzer_activity_coefficient",
    "pitzer_osmotic_coefficient",
    "PITZER_A_PHI_25C",
    "PITZER_PARAMETERS",
    "normalize_molality", "MolalityMolKg",
]

PITZER_A_PHI_25C = 0.3915  # Debye-Hückel slope for osmotic coefficient at 25 °C


def normalize_molality(m: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(m, {"mol/kg": 1.0, "m": 1.0, "mmol/kg": 1e-3, "µmol/kg": 1e-6}, quantity_name="molality")

MolalityMolKg = Annotated[NormalizedComputable, normalize_molality]

# Pitzer parameters for common electrolytes at 25 °C
# Source: Pitzer (1991), Activity Coefficients in Electrolyte Solutions
# Format: (β₀, β₁, C^φ)
PITZER_PARAMETERS = {
    "NaCl": (0.0765, 0.2664, 0.00127),
    "KCl": (0.04835, 0.2122, -0.00084),
    "HCl": (0.1775, 0.2945, 0.00080),
    "NaOH": (0.0864, 0.253, 0.0044),
    "KOH": (0.1298, 0.320, 0.0041),
    "CaCl2": (0.3159, 1.614, -0.00034),
    "MgCl2": (0.3524, 1.6815, 0.00519),
    "Na2SO4": (0.01958, 1.113, 0.00497),
    "MgSO4": (0.2210, 3.343, 0.0250),
    "NaBr": (0.0973, 0.2791, 0.00116),
    "KBr": (0.0569, 0.2212, -0.00180),
    "LiCl": (0.1494, 0.3074, 0.00359),
    "NaNO3": (0.0068, 0.1783, -0.00072),
    "KNO3": (-0.0816, 0.0494, 0.00660),
    "NH4Cl": (0.0522, 0.1918, -0.00301),
}


@returns_unit("")
def pitzer_f_gamma(I: MolalityMolKg, A_phi=PITZER_A_PHI_25C, b=1.2):
    """
    Compute the Pitzer f^γ (electrostatic) term.

    f^γ = -A_φ * (√I / (1 + b*√I) + (2/b) * ln(1 + b*√I)).

    Parameters
    ----------
    I : float
        Ionic strength in mol/kg.
    A_phi : float
        Debye-Hückel slope for osmotic coefficient (default: 0.3915 at 25 °C).
    b : float
        Pitzer universal parameter (default: 1.2).

    Returns
    -------
    float
        f^γ term (dimensionless).
    
    """
    I = normalize_molality(I) if isinstance(I, str) else I
    sqrt_I = np.sqrt(I)
    return -A_phi * (sqrt_I / (1.0 + b * sqrt_I) + (2.0 / b) * np.log(1.0 + b * sqrt_I))


@returns_unit("")
def pitzer_B_gamma(I: MolalityMolKg, beta0, beta1, alpha=2.0):
    """
    Compute the Pitzer B^γ (ion-interaction) term.

    B^γ = 2*β₀ + 2*β₁/(α²*I) * [1 - (1 + α*√I - α²*I/2) * exp(-α*√I)].

    Parameters
    ----------
    I : float
        Ionic strength in mol/kg.
    beta0 : float
        Pitzer β₀ parameter.
    beta1 : float
        Pitzer β₁ parameter.
    alpha : float
        Pitzer α parameter (default: 2.0 for 1-1, 1-2, 2-1 electrolytes).

    Returns
    -------
    float
        B^γ term (dimensionless).
    
    """
    I = normalize_molality(I) if isinstance(I, str) else I
    sqrt_I = np.sqrt(I)
    x = alpha * sqrt_I
    return 2.0 * beta0 + 2.0 * beta1 / (alpha**2 * I) * (1.0 - (1.0 + x - x**2 / 2.0) * np.exp(-x))


@returns_unit("")
def pitzer_C_gamma(C_phi):
    """
    Compute the Pitzer C^γ term from C^φ.

    C^γ = 1.5 * C^φ.

    Parameters
    ----------
    C_phi : float
        Pitzer C^φ parameter.

    Returns
    -------
    float
        C^γ term.
    
    """
    return 1.5 * C_phi


@returns_unit("")
def pitzer_activity_coefficient(m: MolalityMolKg, z_plus, z_minus, nu_plus, nu_minus,
                                 beta0, beta1, C_phi,
                                 A_phi=PITZER_A_PHI_25C, b=1.2, alpha=2.0):
    """
    Compute the mean activity coefficient using the Pitzer model.

    ln(γ±) = |z+ * z-| * f^γ + m * (2*ν+*ν-/ν) * B^γ + m² * (2*(ν+*ν-)^(3/2)/ν) * C^γ.

    Parameters
    ----------
    m : float
        Molality of the electrolyte in mol/kg.
    z_plus : float
        Charge number of the cation.
    z_minus : float
        Charge number of the anion (positive).
    nu_plus : float
        Stoichiometric coefficient of the cation.
    nu_minus : float
        Stoichiometric coefficient of the anion.
    beta0 : float
        Pitzer β₀ parameter.
    beta1 : float
        Pitzer β₁ parameter.
    C_phi : float
        Pitzer C^φ parameter.
    A_phi : float
        Debye-Hückel slope (default: 0.3915 at 25 °C).
    b : float
        Pitzer universal parameter (default: 1.2).
    alpha : float
        Pitzer α parameter (default: 2.0).

    Returns
    -------
    float
        Mean activity coefficient γ± (dimensionless).
    
    """
    m = normalize_molality(m) if isinstance(m, str) else m
    nu = nu_plus + nu_minus
    # Ionic strength for single electrolyte
    I = 0.5 * m * (nu_plus * z_plus**2 + nu_minus * z_minus**2)

    f_g = pitzer_f_gamma(I, A_phi, b)
    B_g = pitzer_B_gamma(I, beta0, beta1, alpha)
    C_g = pitzer_C_gamma(C_phi)

    ln_gamma = (np.abs(z_plus * z_minus) * f_g +
                m * (2.0 * nu_plus * nu_minus / nu) * B_g +
                m**2 * (2.0 * (nu_plus * nu_minus)**1.5 / nu) * C_g)

    return np.exp(ln_gamma)


@returns_unit("")
def pitzer_osmotic_coefficient(m: MolalityMolKg, z_plus, z_minus, nu_plus, nu_minus,
                                beta0, beta1, C_phi,
                                A_phi=PITZER_A_PHI_25C, b=1.2, alpha=2.0):
    """
    Compute the osmotic coefficient using the Pitzer model.

    φ - 1 = |z+*z-| * f^φ + m * (2*ν+*ν-/ν) * B^φ + m² * (2*(ν+*ν-)^(3/2)/ν) * C^φ.

    where:
        f^φ = -A_φ * √I / (1 + b*√I)
        B^φ = β₀ + β₁ * exp(-α*√I)

    Parameters
    ----------
    m : float
        Molality in mol/kg.
    z_plus, z_minus : float
        Charge numbers (z_minus positive).
    nu_plus, nu_minus : float
        Stoichiometric coefficients.
    beta0, beta1 : float
        Pitzer parameters.
    C_phi : float
        Pitzer C^φ parameter.
    A_phi : float
        Debye-Hückel slope (default: 0.3915).
    b : float
        Pitzer parameter (default: 1.2).
    alpha : float
        Pitzer parameter (default: 2.0).

    Returns
    -------
    float
        Osmotic coefficient φ (dimensionless).
    
    """
    m = normalize_molality(m) if isinstance(m, str) else m
    nu = nu_plus + nu_minus
    I = 0.5 * m * (nu_plus * z_plus**2 + nu_minus * z_minus**2)
    sqrt_I = np.sqrt(I)

    # f^φ
    f_phi = -A_phi * sqrt_I / (1.0 + b * sqrt_I)

    # B^φ
    B_phi = beta0 + beta1 * np.exp(-alpha * sqrt_I)

    phi_minus_1 = (np.abs(z_plus * z_minus) * f_phi +
                   m * (2.0 * nu_plus * nu_minus / nu) * B_phi +
                   m**2 * (2.0 * (nu_plus * nu_minus)**1.5 / nu) * C_phi)

    return 1.0 + phi_minus_1
