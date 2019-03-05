#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate MOSFETs
"""
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = ["mosfet_gate_charge_losses"]

def mosfet_gate_charge_losses(total_gate_charge, vsupply, frequency="100 kHz") -> Unit("W"):
    """
    Compute the gate charge loss of a MOSFET in a switch-mode
    power-supply application.
    
    Ref: 
    http://rohmfs.rohm.com/en/products/databook/applinote/ic/power/switching_regulator/power_loss_appli-e.pdf

    Parameters
    ----------
    total_gate_charge: number or Engineer string
        The total gate charge in Coulomb.
        For multiple MOSFETs such as in synchronous applications,
        add their gate charges together.
    vsupply: number or Engineer string
        The gate driver supply voltage in Volts
    frequency: number or Engineer string
        The switching frequency in Hz
    """
    total_gate_charge = normalize_numeric(total_gate_charge)
    vsupply = normalize_numeric(vsupply)
    frequency = normalize_numeric(frequency)
    return total_gate_charge * vsupply * frequency