"""
Utilities for concentration
"""
from numpy import ndarray
import numpy as np

from . import EngineerIO
from .Decorators import returns_unit
from .UnitInfo import EngineerIOConfiguration, UnitAlias, UnitInfo
from .Defaults import default_si_prefix_map
from scipy.constants import N_A  # Avogadro's number

__all__ = [
    "normalize_mass_concentration", "convert_mass_concentration_to_per_liter", "EngineerMassConcentrationIO",
    "normalize_amount_concentration", "convert_amount_concentration_to_grams_per_liter", "EngineerAmountConcentrationIO"
]

def amount_concentration_unit_infos():
    return [
        UnitInfo('1/l', 1/N_A, ['per liter', 'per litre', '1/L', '1/litre', '1/liter', 'per l']),
        UnitInfo('mol/l', 1.0, ['M', 'mol/L', 'molar', 'mole per liter', 'moles per liter']),
        UnitAlias('mmol/l', aliases=['mmol/L', 'millimolar', 'millimole per liter', 'millimoles per liter']),
        UnitAlias('µmol/l', aliases=['umol/l', 'umol/L', 'µmol/L', 'micromolar', 'micromole per liter', 'micromoles per liter']),
        UnitAlias('nmol/l', aliases=['nmol/L', 'nanomolar', 'nanomole per liter', 'nanomoles per liter']),
        UnitAlias('pmol/l', aliases=['pmol/L', 'picomolar', 'picomole per liter', 'picomoles per liter']),
        # Medical units (per milliliter)
        UnitInfo('mol/ml', 1000.0, ['mol/mL', 'mol/ml', 'mole per milliliter', 'moles per milliliter']),
        UnitInfo('mmol/ml', 1.0, ['mmol/mL', 'mmol/ml', 'millimole per milliliter', 'millimoles per milliliter']),
        UnitInfo('µmol/ml', 0.001, ['umol/ml', 'umol/mL', 'µmol/ml', 'µmol/mL', 'micromole per milliliter', 'micromoles per milliliter']),
        UnitInfo('nmol/ml', 0.000001, ['nmol/mL', 'nanomole per milliliter', 'nanomoles per milliliter']),
        UnitInfo('pmol/ml', 0.000000001, ['pmol/mL', 'picomole per milliliter', 'picomoles per milliliter']),
        # Medical units (per microliter)
        UnitInfo('mol/µl', 1_000_000.0, ['mol/ul', 'mol/uL', 'mol/µl', 'mole per microliter', 'moles per microliter']),
        UnitInfo('mmol/µl', 1000.0, ['mmol/ul', 'mmol/uL', 'mmol/µl', 'millimole per microliter', 'millimoles per microliter']),
        UnitInfo('µmol/µl', 1.0, ['umol/ul', 'umol/uL', 'µmol/ul', 'µmol/uL', 'micromole per microliter', 'micromoles per microliter']),
        UnitInfo('nmol/µl', 0.001, ['nmol/ul', 'nmol/uL', 'nmol/µl', 'nanomole per microliter', 'nanomoles per microliter']),
        UnitInfo('pmol/µl', 0.000001, ['pmol/ul', 'pmol/uL', 'pmol/µl', 'picomole per microliter', 'picomoles per microliter']),
        # Medical units (per nanoliter)
        UnitInfo('mol/nl', 1_000_000_000.0, ['mol/nl', 'mol/nL', 'mole per nanoliter', 'moles per nanoliter']),
        UnitInfo('mmol/nl', 1_000_000.0, ['mmol/nl', 'mmol/nL', 'millimole per nanoliter', 'millimoles per nanoliter']),
        UnitInfo('µmol/nl', 1000.0, ['umol/nl', 'umol/nL', 'µmol/nl', 'µmol/nL', 'micromole per nanoliter', 'micromoles per nanoliter']),
        UnitInfo('nmol/nl', 1.0, ['nmol/nl', 'nmol/nL', 'nanomole per nanoliter', 'nanomoles per nanoliter']),
        UnitInfo('pmol/nl', 0.001, ['pmol/nl', 'pmol/nL', 'picomole per nanoliter', 'picomoles per nanoliter']),
        # Percent, ppm, ppb
        UnitInfo('%', 0.01, ['percent', '%', '% v/v', '% w/v']),
        UnitInfo('ppm', 1e-6, ['parts per million']),
        UnitInfo('ppb', 1e-9, ['parts per billion']),
        UnitInfo('ppt', 1e-12, ['parts per trillion']),
    ]

def mass_concentration_unit_infos():
    return [
        UnitInfo('g/l', 1.0, ['g/L', 'gram per liter', 'grams per liter']),
        UnitAlias('mg/l', aliases=['mg/L', 'milligram per liter', 'milligrams per liter']),
        UnitAlias('µg/l', aliases=['ug/l', 'ug/L', 'µg/L', 'microgram per liter', 'micrograms per liter']),
        UnitAlias('ng/l', aliases=['ng/L', 'nanogram per liter', 'nanograms per liter']),
        UnitInfo('g/ml', 1000.0, ['g/mL', 'g/ml', 'gram per milliliter', 'grams per milliliter']),
        UnitInfo('mg/ml', 1.0, ['mg/mL', 'mg/ml', 'milligram per milliliter', 'milligrams per milliliter']),
        UnitInfo('µg/ml', 0.001, ['ug/ml', 'ug/mL', 'µg/mL', 'microgram per milliliter', 'micrograms per milliliter']),
        UnitInfo('ng/ml', 0.000001, ['ng/mL', 'nanogram per milliliter', 'nanograms per milliliter']),
        UnitInfo('pg/ml', 0.000000001, ['pg/mL', 'picogram per milliliter', 'picograms per milliliter']),
        UnitInfo('g/µl', 1_000_000.0, ['g/ul', 'g/uL', 'g/µl', 'gram per microliter', 'grams per microliter']),
        UnitInfo('mg/µl', 1000.0, ['mg/ul', 'mg/uL', 'mg/µl', 'milligram per microliter', 'milligrams per microliter']),
        UnitInfo('µg/µl', 1.0, ['ug/ul', 'ug/uL', 'µg/ul', 'µg/uL', 'microgram per microliter', 'micrograms per microliter']),
        UnitInfo('ng/µl', 0.001, ['ng/ul', 'ng/uL', 'ng/µl', 'nanogram per microliter', 'nanograms per microliter']),
        UnitInfo('pg/µl', 0.000001, ['pg/ul', 'pg/uL', 'pg/µl', 'picogram per microliter', 'picograms per microliter']),
        UnitInfo('g/nl', 1_000_000_000.0, ['g/nl', 'g/nL', 'g/nanoliter', 'gram per nanoliter', 'grams per nanoliter']),
        UnitInfo('mg/nl', 1_000_000.0, ['mg/nl', 'mg/nL', 'milligram per nanoliter', 'milligrams per nanoliter']),
        UnitInfo('µg/nl', 1000.0, ['ug/nl', 'ug/nL', 'µg/nl', 'µg/nL', 'microgram per nanoliter', 'micrograms per nanoliter']),
        UnitInfo('ng/nl', 1.0, ['ng/nl', 'ng/nL', 'nanogram per nanoliter', 'nanograms per nanoliter']),
        UnitInfo('pg/nl', 0.001, ['pg/nl', 'pg/nL', 'picogram per nanoliter', 'picograms per nanoliter']),
        # Percent, ppm, ppb
        UnitInfo('%', 0.01, ['percent', '%', '% v/v', '% w/v']),
        UnitInfo('ppm', 1e-6, ['parts per million']),
        UnitInfo('ppb', 1e-9, ['parts per billion']),
        UnitInfo('ppt', 1e-12, ['parts per trillion']),
    ]

def _create_amount_concentration_config():
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=amount_concentration_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=default_si_prefix_map(include_length_unit_prefixes=True)
    )

def _create_mass_concentration_config():
    config = EngineerIOConfiguration.default()
    return EngineerIOConfiguration(
        units=mass_concentration_unit_infos(),
        unit_prefixes=config.unit_prefixes,
        si_prefix_map=default_si_prefix_map(include_length_unit_prefixes=True)
    )

class EngineerAmountConcentrationIO(EngineerIO):
    """
    EngineerIO subclass specialized for amount concentration unit parsing and conversion.
    """
    _instance = None

    def __init__(self):
        super().__init__(config=_create_amount_concentration_config())

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @returns_unit("1/l")
    def normalize_amount_concentration(self, s):
        if s is None:
            return None
        if isinstance(s, list):
            return np.asarray([self.normalize_amount_concentration(v) for v in s])
        if isinstance(s, ndarray):
            return np.asarray([self.normalize_amount_concentration(v) for v in s])
        return self.normalize(s).value

    @returns_unit("1/l")
    def convert_amount_concentration_to_grams_per_liter(self, value, unit):
        return self.normalize_amount_concentration(f"{value} {unit}")

class EngineerMassConcentrationIO(EngineerIO):
    """
    EngineerIO subclass specialized for mass concentration unit parsing and conversion.
    """
    _instance = None

    def __init__(self):
        super().__init__(config=_create_mass_concentration_config())

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @returns_unit("mol/l")
    def normalize_mass_concentration(self, s):
        if s is None:
            return None
        if isinstance(s, list):
            return np.asarray([self.normalize_mass_concentration(v) for v in s])
        if isinstance(s, ndarray):
            return np.asarray([self.normalize_mass_concentration(v) for v in s])
        return self.normalize(s).value

    @returns_unit("1/l")
    def convert_mass_concentration_to_per_liter(self, value, unit):
        return self.normalize_mass_concentration(f"{value} {unit}")

@returns_unit("1/l")
def convert_amount_concentration_to_grams_per_liter(value, unit, instance=None):
    if instance is None:
        instance = EngineerAmountConcentrationIO.instance()
    return instance.convert_amount_concentration_to_grams_per_liter(value, unit)

@returns_unit("1/l")
def normalize_amount_concentration(s, instance=None):
    if instance is None:
        instance = EngineerAmountConcentrationIO.instance()
    return instance.normalize_amount_concentration(s)

@returns_unit("1/l")
def convert_mass_concentration_to_per_liter(value, unit, instance=None):
    if instance is None:
        instance = EngineerMassConcentrationIO.instance()
    return instance.convert_mass_concentration_to_per_liter(value, unit)

@returns_unit("1/l")
def normalize_mass_concentration(s, instance=None):
    if instance is None:
        instance = EngineerMassConcentrationIO.instance()
    return instance.normalize_mass_concentration(s)
