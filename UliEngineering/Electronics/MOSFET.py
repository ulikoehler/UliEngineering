#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate MOSFETs
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "mosfet_gate_charge_losses", "mosfet_gate_charge_loss_per_cycle",
    "mosfet_gate_capacitance_from_gate_charge"]

@returns_unit("W")
@normalize_numeric_args
def mosfet_gate_charge_losses(total_gate_charge, vsupply, frequency="100 kHz"):
    """
    Compute the gate charge loss of a MOSFET in a switch-mode
    power-supply application as a total power (integrated per second).
    
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
    return mosfet_gate_charge_loss_per_cycle(total_gate_charge, vsupply) * frequency


@returns_unit("J")
@normalize_numeric_args
def mosfet_gate_charge_loss_per_cycle(total_gate_charge, vsupply):
    """
    Compute the gate charge loss of a MOSFET in a switch-mode
    power-supply application per switching cycle.
    
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
    """
    return total_gate_charge * vsupply

@returns_unit("F")
@normalize_numeric_args
def mosfet_gate_capacitance_from_gate_charge(total_gate_charge, vsupply):
    """
    Compute the gate capacitance of a MOSFET in a switch-mode
    power-supply application.
    
    Parameters
    ----------
    total_gate_charge: number or Engineer string
        The total gate charge in Coulomb.
        For multiple MOSFETs such as in synchronous applications,
        add their gate charges together.
    vsupply: number or Engineer string
        The gate driver supply voltage in Volts
    """
    return total_gate_charge / vsupply