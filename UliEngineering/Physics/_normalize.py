#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from UliEngineering.EngineerIO import EngineerIO
from UliEngineering.EngineerIO.Types import NormalizedComputable
from UliEngineering.Units import InvalidUnitInContextException


def normalize_with_known_units(
    value: Any,
    unit_factors: Mapping[str, float],
    *,
    default_factor: float = 1.0,
    quantity_name: str = "value",
) -> NormalizedComputable:
    if value is None:
        raise ValueError(f"Can't normalize {quantity_name} None")

    if isinstance(value, np.ndarray):
        return np.asarray(
            [normalize_with_known_units(item, unit_factors, default_factor=default_factor, quantity_name=quantity_name) for item in value],
            dtype=float,
        )

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, np.ndarray)):
        return np.asarray(
            [normalize_with_known_units(item, unit_factors, default_factor=default_factor, quantity_name=quantity_name) for item in value],
            dtype=float,
        )

    if isinstance(value, (int, float, np.generic)):
        return float(value) * default_factor

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported {quantity_name} value type: {type(value)!r}")

    raw_value = value.strip()
    compact_value = raw_value.replace(" ", "")
    for unit, factor in sorted(unit_factors.items(), key=lambda item: len(item[0]), reverse=True):
        if compact_value.endswith(unit):
            numeric_part = compact_value[:-len(unit)]
            if not numeric_part:
                raise ValueError(f"Missing numeric part in {quantity_name} string '{value}'")
            return EngineerIO.instance().normalize_numeric(numeric_part) * factor

    if any(ch.isspace() for ch in raw_value) or "/" in compact_value or "^" in compact_value:
        raise InvalidUnitInContextException(
            f"Invalid unit in {quantity_name} string '{value}'. Expected one of: {', '.join(sorted(unit_factors))}"
        )

    return EngineerIO.instance().normalize_numeric(compact_value) * default_factor