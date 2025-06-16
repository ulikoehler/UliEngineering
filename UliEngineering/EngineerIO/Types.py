#!/usr/bin/env python3

from dataclasses import dataclass

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
    prefix: str = ''
    value: float = 0.0
    original_number: float = 0.0
    unit_prefix: str = ''
    unit: str = ''
    # Multiplier from prefix
    prefix_multiplier: float = 1.0
