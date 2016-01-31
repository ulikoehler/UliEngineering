#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for generating chunks from datasets
"""
import numpy as np
import functools
from toolz import functoolz
import random
import collections

__all__ = ["ChunkGenerator", "overlapping_chunks", "reshaped_chunks", "random_sample_chunks",
           "random_sample_chunks_nonoverlapping"]


class ChunkGenerator(object):
    """
    Chunk generator objects can lazily generate arbitrary chunks from arbitary data.
    They are based around an unary generator function that takes a chunk index
    and a predefined number of chunks.
    """
    def __init__(self, generator, num_chunks, func=None):
        self.generator = generator
        self.num_chunks = num_chunks
        self.func = functoolz.identity

    def __iter__(self):
        return (self.func(self.generator(i)) for i in range(self.num_chunks))

    def __getitem__(self, i):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop, key.step)]
        elif isinstance(key, int):
            return self.func(self.generator(i))
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(type(key))) 


    def __call__(self, i):  # Compatibility only. Slicing recommended
        return self.__getitem__(i)

    def __len__(self):
        return self.num_chunks

    def apply(self, fn):
        """
        Add a function to the current list of functions. The given function
        will be executed last in the list of functions.
        """
        if self.func == functoolz.identity:
            self.func = fn
        elif isinstance(self.func, functoolz.Compose):
            self.func.funcs.append(fn)
        else:
            self.func = functoolz.compose(fn, self.func)

    def as_list(self):
        return list(self)

    def as_array(self):
        return np.asarray(self.as_list())


def __overlapping_chunks_worker(offsets, chunksize, arr, copy, i):
    ofs = offsets[i]
    arrslice = arr[ofs:ofs + chunksize]
    return arrslice.copy() if copy else arrslice


def overlapping_chunks(arr, chunksize, shiftsize, copy=False):
    """
    A chunk-generating function that can be used for parallelFFTReduce().
    Generates only full chunks with variable chunk / shift size.

    If copy=False, the chunk is not copied in the generator function.
    Functions like parallelFFTSum() modify the data which might lead to
    undesired overwriting of data. However, setting copy=False might
    improve the performance significantly if the downstream function does
    not require copies and the arrays are large.

    This is a lazy function, it generates chunks only on-demand.

    Returns (g, n) where g is a unary generator function (which takes the chunk
        number as an argument) and n is the number of chunks.
    """
    # Precompute offset table
    chunksize = int(chunksize)
    offsets = np.asarray(range(0, arr.shape[0] - (chunksize - 1), shiftsize))
    gen = functools.partial(__overlapping_chunks_worker, offsets, chunksize, arr, copy)
    return ChunkGenerator(gen, offsets.size)


def random_sample_chunks_nonoverlapping(arr, chunksize, num_samples):
    """
    A chunk-generating function that randomly selects n non-overlapping chunks.
    This generator uses reshaped chunks (i.e. non overlapping zero-overhead chunks)
    as a basis and randomly selects a fraction of those chunks.
    This means that only start chunk number is randomized while the chunk phase
    is always the same.
    """
    chunksize = int(chunksize)
    arr2d = reshaped_chunks(arr, chunksize)
    indices = random.sample(range(arr2d.shape[0]), num_samples)
    return ChunkGenerator(lambda i: arr2d[indices[i]], num_samples)


def random_sample_chunks(arr, chunksize, num_samples):
    """
    A chunk-generating function that can be used for parallelFFTReduce().
    """
    chunksize = int(chunksize)
    start_idxs = range(arr.shape[0] - (chunksize - 1))
    indices = random.sample(start_idxs, num_samples)
    return ChunkGenerator(lambda i: arr[indices[i]:indices[i] + chunksize], num_samples)


def reshaped_chunks(arr, chunksize):
    """
    Generates virtual chunks of a numpy array by reshaping a view of the original array.
    Works really well with huge, mmapped arrays as no part of the array is copied.

    Automatically handles odd-sized arrays by discarding extra values.
    Works only with 1D arrays.
    """
    if arr.shape[0] == 0:
        return arr
    chunksize = int(chunksize)
    # We might need to cut off some records for odd-shaped arrays
    end = arr.shape[0] - (arr.shape[0] % chunksize)
    v = arr[:end].view()
    v.shape = (-1, chunksize)
    return v
