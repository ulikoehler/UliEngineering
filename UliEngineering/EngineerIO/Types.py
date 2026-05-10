#!/usr/bin/env python3

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

NormalizableScalar = str | bytes | int | float | np.generic
NormalizableArgument = NormalizableScalar | np.ndarray | Iterable[NormalizableScalar]
NormalizedArgument = int | float | np.generic | np.ndarray
NormalizedComputable = float | np.ndarray

@dataclass
class UnitSplitResult:
    remainder: str = ''
    unit_prefix: str = ''
    unit: str = ''

@dataclass
class SplitResult:
    prefix: str = ''
    number: str = ''
    unit_prefix_char: str = ''
    unit_prefix: str = ''
    unit: str = ''

@dataclass
class NormalizeResult:
    # Resulting numeric value (in the normalized unit)
    prefix: str = ''
    value: float = field(default=0.0)
    # Original numeric value (before normalization)
    original_number: float = field(default=0.0)
    # Unit prefix (e.g. "kilo", "mega")
    unit_prefix: str = field(default='')
    # Unit originally parsed from the value. (e.g. "m²", "cm²").
    # NOT neccessarily the unit of [value]
    unit: str = field(default='')
    # Multiplier from prefix
    prefix_multiplier: float = field(default=1.0)
    # Multiplier from unit factor
    unit_factor: float = field(default=1.0)
