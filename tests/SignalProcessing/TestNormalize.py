#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
from UliEngineering.SignalProcessing.Normalize import *
import numpy as np
import unittest
import pytest

class TestNormalize(unittest.TestCase):
    
    @pytest.mark.filterwarnings("ignore: invalid value encountered in scalar divide")
    @pytest.mark.filterwarnings("ignore: Mean of empty slice")
    def test_center_to_zero(self):
        assert_allclose(center_to_zero([]).data, [])
        assert_allclose(center_to_zero(np.asarray([0.])).data, [0.])
        assert_allclose(center_to_zero(np.asarray([1.])).data, [0.])
        assert_allclose(center_to_zero(np.asarray([-1., 1.])).data, [-1., 1.])
        assert_allclose(center_to_zero(np.asarray([-2., 2.])).data, [-2., 2.])
        assert_allclose(center_to_zero(np.asarray([0., 1.])).data, [-0.5, 0.5])
        assert_allclose(center_to_zero(np.asarray([0., 2.])).data, [-1., 1])

    def test_normalize_max(self):
        assert_allclose(normalize_max([]).data, [])
        assert_allclose(normalize_max(np.asarray([0.])).data, [0.])
        assert_allclose(normalize_max(np.asarray([1.])).data, [1.])
        assert_allclose(normalize_max(np.asarray([-1., 1.])).data, [-1., 1.])
        assert_allclose(normalize_max(np.asarray([-2., 2.])).data, [-1., 1.])
        assert_allclose(normalize_max(np.asarray([0., 1.])).data, [0., 1.])
        assert_allclose(normalize_max(np.asarray([-1., 0.])).data, [-1., 0.])
        assert_allclose(normalize_max(np.asarray([-1., -2.])).data, [-1., -2.])
        assert_allclose(normalize_max(np.asarray([0., 2.])).data, [0., 1])
        assert_allclose(normalize_max(np.asarray([-10., 1.])).data, [-10., 1.])
        assert_allclose(normalize_max(np.asarray([-10., 2.])).data, [-5., 1])

    
    def test_normalize_minmax(self):
        assert_allclose(normalize_minmax([]).data, [])
        assert_allclose(normalize_minmax(np.asarray([0.])).data, [0.])
        assert_allclose(normalize_minmax(np.asarray([1.])).data, [0.])
        assert_allclose(normalize_minmax(np.asarray([-1., 1.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([-2., 2.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([0., 1.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([-1., 0.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([-1., -2.])).data, [1., 0.])
        assert_allclose(normalize_minmax(np.asarray([0., 2.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([0., 0.])).data, [0., 0.])
        assert_allclose(normalize_minmax(np.asarray([1., 1.])).data, [0., 0.])
        assert_allclose(normalize_minmax(np.asarray([-10., 1.])).data, [0., 1.])
        assert_allclose(normalize_minmax(np.asarray([-10., 2.])).data, [0., 1])
        assert_allclose(normalize_minmax(np.asarray([1., 2., 3.])).data, [0., 0.5, 1.])
        assert_allclose(normalize_minmax(np.asarray([3., 2., 1.])).data, [1., 0.5, 0.])

