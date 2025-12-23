#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reliability conversion utilities

This module provides conversion helpers between common reliability metrics:
- FIT (failures in 1e9 hours)
- PFH / PFHd (failures per hour)
- MTTF / MTTFd (mean time to failure / dangerous failure)
- B10d (cycles until 10% dangerous failures)

Source and formulas are based on:
https://www.nsbiv.ch/fileadmin/user_upload/documents/fachartikel/150812HR_Umrechnungsformeln_f%C3%BCr_MTTFd-Berechnungen.pdf

This module used to live at `UliEngineering.Electronics.Reliability`; that module
now contains a small shim that imports everything from here and emits a
deprecation warning.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "FIT_to_MTTF",
    "FIT_to_MTTFd",
    "MTTF_to_FIT",
    "MTTFd_to_FIT",
    "B10d_to_MTTFd",
    "MTTFd_to_B10d",
    "cycles_per_year",
    "PFH_to_MTTF",
    "PFHd_to_MTTFd",
    "MTTF_to_PFH",
    "MTTFd_to_PFHd",
    "FIT_to_PFH",
    "PFH_to_FIT",
    "FIT_to_PFHd",
    "PFHd_to_FIT",
]

Unit = Literal["years", "hours", "days"]

_HOURS_PER_YEAR = 365 * 24
_HOURS_PER_DAY = 24


def _unit_to_hours_factor(unit: Unit) -> float:
    if unit == "hours":
        return 1.0
    if unit == "days":
        return float(_HOURS_PER_DAY)
    # years
    return float(_HOURS_PER_YEAR)


def FIT_to_MTTF(fit: float, unit: Unit = "years") -> float:
    if fit <= 0:
        raise ValueError("FIT must be > 0")
    mttf_years = 1e9 / (_HOURS_PER_YEAR * fit)
    if unit == "years":
        return mttf_years
    if unit == "hours":
        return mttf_years * _HOURS_PER_YEAR
    return mttf_years * (_HOURS_PER_YEAR / _HOURS_PER_DAY)


def FIT_to_MTTFd(fit: float, unit: Unit = "years") -> float:
    # MTTFd is defined as twice the MTTF for the same FIT
    return 2.0 * FIT_to_MTTF(fit, unit)


def MTTF_to_FIT(mttf: float, unit: Unit = "years") -> float:
    if mttf <= 0:
        raise ValueError("MTTF must be > 0")
    hours = mttf * _unit_to_hours_factor(unit)
    mttf_years = hours / _HOURS_PER_YEAR
    return 1e9 / (_HOURS_PER_YEAR * mttf_years)


def MTTFd_to_FIT(mttfd: float, unit: Unit = "years") -> float:
    if mttfd <= 0:
        raise ValueError("MTTFd must be > 0")
    mttf_equiv = mttfd / 2.0
    return MTTF_to_FIT(mttf_equiv, unit)


def cycles_per_year(t_cycle: float, d_op: float = 365.0, h_op: float = 24.0) -> float:
    if t_cycle <= 0:
        raise ValueError("t_cycle must be > 0")
    if d_op <= 0 or h_op <= 0:
        raise ValueError("d_op and h_op must be > 0")
    return d_op * h_op * 3600.0 / float(t_cycle)


def B10d_to_MTTFd(B10d: float, t_cycle: float, d_op: float = 365.0, h_op: float = 24.0, unit: Unit = "years") -> float:
    if B10d <= 0:
        raise ValueError("B10d must be > 0")
    n_op = cycles_per_year(t_cycle, d_op, h_op)
    mttfd_years = B10d / (0.1 * n_op)
    if unit == "years":
        return mttfd_years
    if unit == "hours":
        return mttfd_years * _HOURS_PER_YEAR
    return mttfd_years * (_HOURS_PER_YEAR / _HOURS_PER_DAY)


def MTTFd_to_B10d(mttfd: float, t_cycle: float, d_op: float = 365.0, h_op: float = 24.0, unit: Unit = "years") -> float:
    if mttfd <= 0:
        raise ValueError("MTTFd must be > 0")
    hours = mttfd * _unit_to_hours_factor(unit)
    mttfd_years = hours / _HOURS_PER_YEAR
    n_op = cycles_per_year(t_cycle, d_op, h_op)
    return 0.1 * n_op * mttfd_years


# ----- PFH / PFHd conversions -----


def PFH_to_MTTF(pfh: float, unit: Unit = "years") -> float:
    if pfh <= 0:
        raise ValueError("PFH must be > 0")
    mttf_years = 1.0 / (_HOURS_PER_YEAR * pfh)
    if unit == "years":
        return mttf_years
    if unit == "hours":
        return mttf_years * _HOURS_PER_YEAR
    return mttf_years * (_HOURS_PER_YEAR / _HOURS_PER_DAY)


def PFHd_to_MTTFd(pfhd: float, unit: Unit = "years") -> float:
    if pfhd <= 0:
        raise ValueError("PFHd must be > 0")
    mttfd_years = 1.0 / (_HOURS_PER_YEAR * pfhd)
    if unit == "years":
        return mttfd_years
    if unit == "hours":
        return mttfd_years * _HOURS_PER_YEAR
    return mttfd_years * (_HOURS_PER_YEAR / _HOURS_PER_DAY)


def MTTF_to_PFH(mttf: float, unit: Unit = "years") -> float:
    if mttf <= 0:
        raise ValueError("MTTF must be > 0")
    hours = mttf * _unit_to_hours_factor(unit)
    return 1.0 / hours


def MTTFd_to_PFHd(mttfd: float, unit: Unit = "years") -> float:
    if mttfd <= 0:
        raise ValueError("MTTFd must be > 0")
    hours = mttfd * _unit_to_hours_factor(unit)
    return 1.0 / hours


# ----- FIT <-> PFH conversions -----


def FIT_to_PFH(fit: float) -> float:
    if fit <= 0:
        raise ValueError("FIT must be > 0")
    return float(fit) / 1e9


def PFH_to_FIT(pfh: float) -> float:
    if pfh <= 0:
        raise ValueError("PFH must be > 0")
    return float(pfh) * 1e9


def FIT_to_PFHd(fitd: float) -> float:
    if fitd <= 0:
        raise ValueError("FITd must be > 0")
    return float(fitd) / 1e9


def PFHd_to_FIT(pfhd: float) -> float:
    if pfhd <= 0:
        raise ValueError("PFHd must be > 0")
    return float(pfhd) * 1e9
