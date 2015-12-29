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


def __fixedSizeChunkGenerator(ofsTable, chunksize, y, i):
    """Worker for fixedSizeChunkGenerator()"""
    yofs = ofsTable[i]
    return y[yofs:yofs + chunksize].copy()

def fixedSizeChunkGenerator(y, chunksize, shiftsize):
    """
    A chunk-generating function that can be used for parallelFFTSum().
    Generates only full chunks with variable chunk / shift size.

    Returns copies of chunks, so this can be safely used for non-writable datasets.
    This is a lazy function, it generates copies only on-demand.

    Returns (n, g) where g is a unary generator function and n is the number of chunks.
    """
    # Precompute offset table
    offsets = [ofs for ofs in range(0, y.shape[0] - (chunksize - 1), shiftsize)]
    return len(offsets), functools.partial(__fixedSizeChunkGenerator, offsets, chunksize, y)
