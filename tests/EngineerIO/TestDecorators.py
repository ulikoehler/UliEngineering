#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Annotated, Any
import unittest

import numpy as np

from UliEngineering.EngineerIO.Area import normalize_area
from UliEngineering.EngineerIO.Decorators import normalize_args, normalize_numeric_args
from UliEngineering.EngineerIO.Types import NormalizedComputable
from UliEngineering.Units import Hz, InvalidUnitInContextException, m


DEFAULT_FREQUENCY: Any = "1 kHz"


@normalize_args
def annotated_length(value: Annotated[NormalizedComputable, m]) -> NormalizedComputable:
    return value


@normalize_args
def annotated_area(value: Annotated[NormalizedComputable, normalize_area]) -> NormalizedComputable:
    return value


@normalize_args
def annotated_default_frequency(value: Annotated[NormalizedComputable, Hz] = DEFAULT_FREQUENCY) -> NormalizedComputable:
    return value


@normalize_args(exclude=["unit"])
def annotated_excluded_length(value: Annotated[NormalizedComputable, m], unit="raw") -> tuple[NormalizedComputable, str]:
    return value, unit


@normalize_numeric_args
def legacy_numeric_sum(a, b="1k"):
    return a + b


class TestDecorators(unittest.TestCase):
    def test_normalize_args_uses_unit_metadata(self):
        self.assertAlmostEqual(annotated_length("10 cm"), 0.1)
        self.assertAlmostEqual(annotated_length("2 mm"), 0.002)

    def test_normalize_args_vectorizes_iterables(self):
        normalized = annotated_length(np.asarray(["10 cm", "2 mm"]))
        np.testing.assert_allclose(normalized, np.asarray([0.1, 0.002]))

        normalized_tuple = annotated_length(("10 cm", "2 mm"))
        np.testing.assert_allclose(normalized_tuple, np.asarray([0.1, 0.002]))

    def test_normalize_args_uses_converter_metadata(self):
        self.assertAlmostEqual(annotated_area("25 cm²"), 0.0025)
        self.assertAlmostEqual(annotated_area("100 mm²"), 0.0001)

    def test_normalize_args_normalizes_defaults(self):
        self.assertAlmostEqual(annotated_default_frequency(), 1000.0)

    def test_normalize_args_honors_exclude(self):
        value, unit = annotated_excluded_length("1 m", unit="rpm")
        self.assertAlmostEqual(value, 1.0)
        self.assertEqual(unit, "rpm")

    def test_normalize_args_rejects_invalid_unit(self):
        with self.assertRaises(InvalidUnitInContextException):
            annotated_length("10 s")

    def test_normalize_numeric_args_remains_compatible(self):
        self.assertAlmostEqual(legacy_numeric_sum("2k"), 3000.0)