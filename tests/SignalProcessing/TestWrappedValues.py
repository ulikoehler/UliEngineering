#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for UliEngineering.SignalProcessing.WrappedValues.unwrap

These tests cover normal operation, edge cases, custom thresholds,
NaN/Inf handling, and a round-trip property test using wrapped/modulo values.
"""
import math

import numpy as np
import pytest

from UliEngineering.SignalProcessing.WrappedValues import unwrap, OnlineUnwrapper

def test_no_wrap_returns_same():
    data = [0, 1, 2, 3, 4, 5]
    out = unwrap(data, wrap_value=100)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, np.array(data, dtype=np.float64))


def test_single_wrap_forward():
    # e.g. value jumps from near max down to small number
    series = [95, 98, 2, 5]
    un = unwrap(series, wrap_value=100)
    expected = np.array([95.0, 98.0, 102.0, 105.0])
    np.testing.assert_allclose(un, expected)


def test_single_wrap_backward():
    # e.g. small -> large (wrap in opposite direction)
    series = [10, 5, 350]
    # Using wrap_value=360, threshold default 180: diff from 5->350 = 345 > 180
    # so algorithm will subtract 360 on that step, producing a backward step
    un = unwrap(series, wrap_value=360)
    expected = np.array([10.0, 5.0, -10.0])
    np.testing.assert_allclose(un, expected)


def test_custom_threshold_changes_detection():
    # With a small threshold, a modest jump is considered a wrap
    series = [10, 25, 5]
    un_default = unwrap(series, wrap_value=100)  # threshold defaults to 50
    # default should not treat 25->5 (diff=-20) as wrap
    np.testing.assert_allclose(un_default, np.array([10.0, 25.0, 5.0]))

    un_small_thresh = unwrap(series, wrap_value=100, threshold=10)
    # threshold=10 makes 25->5 (diff=-20 < -10) be treated as forward wrap => add 100
    np.testing.assert_allclose(un_small_thresh, np.array([10.0, 25.0, 105.0]))


def test_large_step_not_multiple_wraps():
    # A very large jump larger than wrap_value but algorithm only corrects one wrap
    series = [0, 250]
    un = unwrap(series, wrap_value=100)
    # diff=250 -> subtract 100 => 150 remains (no multiple-wrap correction)
    np.testing.assert_allclose(un, np.array([0.0, 150.0]))


def test_input_types_and_dtype():
    # Works for python lists and numpy arrays; returns float64 numpy array
    li = [1, 2, 3]
    ar = np.array([1, 2, 3], dtype=np.int32)
    out1 = unwrap(li)
    out2 = unwrap(ar)
    assert out1.dtype == np.float64
    assert out2.dtype == np.float64
    np.testing.assert_allclose(out1, out2)


def test_empty_series_raises_index_error():
    with pytest.raises(IndexError):
        unwrap([])


def test_nan_and_inf_are_preserved():
    series = [1.0, np.nan, 2.0, np.inf, -np.inf]
    out = unwrap(series, wrap_value=100)
    # NaN should remain NaN and will propagate to subsequent values because of cumsum
    assert math.isnan(out[1])
    assert np.all(np.isnan(out[1:]))

    # Separately, ensure infinities are preserved when there is no NaN earlier
    series_inf = [1.0, np.inf, -np.inf]
    out2 = unwrap(series_inf, wrap_value=100)
    # First infinite value should remain infinite. Subsequent -inf may produce NaN
    # because of inf + (-inf) in the cumulative sum.
    assert out2[1] == np.inf
    assert np.isnan(out2[2]) or out2[2] == -np.inf


def test_round_trip_with_wrapped_modulo():
    rng = np.random.default_rng(12345)
    wrap_value = 360.0
    # Create a smooth increasing continuous series (differences << threshold)
    orig = np.linspace(100.0, 100.0 + 10000.0, 1000)
    wrapped = np.mod(orig, wrap_value)
    un = unwrap(wrapped, wrap_value=wrap_value)

    # The unwrapped result should differ from original by a constant multiple of wrap_value
    diff = un - orig
    # Round the ratio to nearest integer to get number of wraps offset
    multiples = np.round(diff / wrap_value)
    # All multiples should be (nearly) equal
    assert np.allclose(multiples, multiples[0])
    # After removing that integer multiple, we should be nearly identical
    corrected = un - multiples[0] * wrap_value
    np.testing.assert_allclose(corrected, orig, atol=1e-6)

def test_online_unwrapper_scalar():
    unwrapper = OnlineUnwrapper(wrap_value=100)
    assert unwrapper(10) == 10
    assert unwrapper(20) == 20
    # 20 -> 90: diff=70 > 50 => wrap backward => correction -100 => 90-100 = -10
    assert unwrapper(90) == -10
    # 90 -> 5: diff=-85 < -50 => wrap forward => correction +100 => 5 + (-100+100) = 5
    assert unwrapper(5) == 5

def test_online_unwrapper_chunks():
    unwrapper = OnlineUnwrapper(wrap_value=100)
    # Chunk 1
    res1 = unwrapper([10, 20])
    np.testing.assert_allclose(res1, [10, 20])
    # Chunk 2
    res2 = unwrapper([90, 5])
    np.testing.assert_allclose(res2, [-10, 5])

def test_online_unwrapper_consistency_with_unwrap():
    # Generate random walk data
    np.random.seed(42)
    # Random steps
    steps = np.random.randn(1000) * 10
    walk = np.cumsum(steps)
    # Wrap it
    wrapped = walk % 100

    # Expected result using the static function
    # Note: static unwrap might start at a different offset if the first value is far from 0?
    # No, static unwrap starts at arr[0].
    # Online unwrapper also starts at arr[0].
    expected = unwrap(wrapped, wrap_value=100)

    # Test scalar
    unwrapper = OnlineUnwrapper(wrap_value=100)
    res_scalar = [unwrapper(x) for x in wrapped]
    np.testing.assert_allclose(res_scalar, expected)

    # Test chunks
    unwrapper = OnlineUnwrapper(wrap_value=100)
    res_chunks = []
    chunk_size = 50
    for i in range(0, len(wrapped), chunk_size):
        chunk = wrapped[i:i+chunk_size]
        res_chunks.append(unwrapper(chunk))
    res_chunks = np.concatenate(res_chunks)
    np.testing.assert_allclose(res_chunks, expected)

def test_online_unwrapper_mixed_scalar_and_chunks():
    unwrapper = OnlineUnwrapper(wrap_value=100)
    # 1. Scalar
    assert unwrapper(10) == 10
    # 2. Chunk
    res = unwrapper([20, 90]) # 20 (ok), 90 (wrap -> -10)
    np.testing.assert_allclose(res, [20, -10])
    # 3. Scalar
    assert unwrapper(5) == 5 # 90->5 wrap -> 5
