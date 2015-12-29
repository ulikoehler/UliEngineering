#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for generating chunks from datasets
"""
import numpy as np
import functools

def evaluateGeneratorFunction(tp, as_list=False):
    """
    Given a tuple (n, g) returned by one of the generator functions,
    evaluate the generator at all values.

    By default, returns a generator. Can also return a list if as_list=True
    """
    n, g = tp
    gen = (g(i) for i in range(n))
    return list(gen) if as_list else gen


def __fixedSizeChunkGenerator(ofsTable, chunksize, y, perform_copy, i):
    """Worker for fixedSizeChunkGenerator()"""
    yofs = ofsTable[i]
    yslice = y[yofs:yofs + chunksize]
    return yslice.copy() if perform_copy else yslice

def fixedSizeChunkGenerator(y, chunksize, shiftsize, perform_copy=True):
    """
    A chunk-generating function that can be used for parallelFFTSum().
    Generates only full chunks with variable chunk / shift size.

    If perform_copy=False, the chunk is not copied in the generator function.
    Functions like parallelFFTSum() modify the data which might lead to
    undesired overwriting of data. However, setting perform_copy=False might
    improve the performance significantly if the downstream function does
    not require copies.

    This is a lazy function, it generates copies only on-demand.

    Returns (n, g) where g is a unary generator function (which takes the chunk
        number as an argument) and n is the number of chunks.
    """
    # Precompute offset table
    offsets = [ofs for ofs in range(0, y.shape[0] - (chunksize - 1), shiftsize)]
    return len(offsets), functools.partial(__fixedSizeChunkGenerator, offsets, chunksize, y, perform_copy)
