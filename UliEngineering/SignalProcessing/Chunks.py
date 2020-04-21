#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for generating chunks from datasets
"""
import numpy as np
import functools
from toolz import functoolz
import random
import concurrent.futures

__all__ = ["ChunkGenerator", "overlapping_chunks", "reshaped_chunks",
           "random_sample_chunks", "random_sample_chunks_nonoverlapping",
           "array_to_chunkgen", "IndexChunkGenerator", "sliding_window"]


class ChunkGenerator(object):
    """
    Chunk generator objects can lazily generate arbitrary chunks
    from arbitary data.
    They are based around an unary generator function that takes a chunk index
    and a predefined number of chunks.
    """
    def __init__(self, generator, num_chunks, func=None):
        self.generator = generator
        self.num_chunks = num_chunks
        self.func = func if func is not None else functoolz.identity

    def unprocessed_chunk(self, i):
        """
        Get the data for the ith chunk.
        In contrast to __getitem__() this slice is not processed using self.func().
        """
        return self.generator(i)

    def __iter__(self):
        return (self.func(self.generator(i)) for i in range(self.num_chunks))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.func(self.generator(key))
        else:
            raise TypeError("Invalid argument type for slicing: {0}".format(
                type(key)))

    def __call__(self, i):  # Compatibility only. Slicing recommended
        return self.__getitem__(i)

    def __len__(self):
        return self.num_chunks

    def apply(self, fn):
        """
        Add a function to the current list of functions. The given function
        will be executed last in the list of functions.

        Return self
        """
        # Try NOT to nest functoolz.compose, that might be expensive.
        # Instead emulate a SINGLE functoolz.compose call
        if self.func == functoolz.identity:
            self.func = fn
        elif isinstance(self.func, functoolz.Compose):
            self.func.funcs.append(fn)
        else:
            self.func = functoolz.compose(fn, self.func)
        return self

    def as_list(self):
        return list(self)

    def as_array(self):
        """
        Convert all values of this chunk to a NumPy array
        """
        return np.asarray(self.as_list())

    def _evaluate_worker(self, i):
        return (i, self[i])

    def evaluate_1d_parallel(self, executor=None):
        """
        Parallel evaluation of the chunks.

        The prerequisite for calling this function is that the
        functions applied to the chunk generator result in a scalar
        value.

        This is often used for time-intensive chunk filter functions.

        Note that in many cases using this approach is much slower than
        just using as_array().

        Parameters
        ----------
        executor : A concurrent.futures.Executor
            Since this writes directly to the result array.
            Depending on the application use either ProcessPoolExecutors
            or ThreadPoolExecutors.
        """
        arr = np.zeros(len(self))
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()
        # Generate futures for executor to execute
        futures = [executor.submit(self._evaluate_worker, i) for i in range(len(self))]
        # Wait until finished
        for future in concurrent.futures.as_completed(futures):
            i, result = future.result()
            arr[i] = result
        return arr

class IndexChunkGenerator(ChunkGenerator):
    """
    A chunk generator that operates on a data array-like object.
    In contrast to the generic chunk generator, this allows
    the user to retrieve the indexes used to generate a certain chunk.
    """
    def __init__(self, data, index_generator, num_chunks, func=None, copy=False):
        """
        Initialize an index chunk generator for a given array.
        The index_generator(i) function must return a slice() object.

        Keyword arguments:
        ------------------
        func : function-like or None
            The chunk postprocessor function. Applied to every chunk.
        """
        self.data = data
        self.index_generator = index_generator
        # Build generator function
        if copy:
            _generator = self._copy_generator
        else: # Dont copy - default
            _generator = self._nocopy_generator
        # Init chunk generator
        super().__init__(generator=_generator, num_chunks=num_chunks, func=func)

    def _nocopy_generator(self, i):
        return self.data[self.index_generator(i)]

    def _copy_generator(self, i):
        return self.data[self.index_generator(i)].copy()

    def original_indexes(self, i):
        """
        Get the indexes used to construct a chunk from self.data.
        Returns a slice() object.
        """
        return self.index_generator(i)

def _overlapping_chunks_worker(offsets, chunksize, i):
    return slice(offsets[i], offsets[i] + chunksize)

def overlapping_chunks(arr, chunksize, shiftsize, func=None, copy=False):
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
    if chunksize == 0:
        raise ValueError("chunksize must not be 0")
    # Precompute offset table
    chunksize = int(chunksize)
    offsets = np.asarray(range(0, arr.shape[0] - (chunksize - 1), shiftsize))
    gen = functools.partial(_overlapping_chunks_worker, offsets, chunksize)
    return IndexChunkGenerator(arr, gen, offsets.size, func=func, copy=copy)

def sliding_window(data, window_size, shift_size=1, window_func=None, copy=False):
    """
    Create a chunk generator that generates left-to-right sliding window chunks.

    This is a convenience wrapper of overlapping_chunks() that clearly
    states the intent of the operation ("sliding window")
    """
    return overlapping_chunks(data, window_size, shift_size, func=window_func, copy=copy)


def random_sample_chunks_nonoverlapping(arr, chunksize, num_samples, copy=False):
    """
    A chunk-generating function that randomly selects num_samples non-overlapping chunks.

    The random indexes are generated on initialization,
    so subsequent calls using the same index return the same sample.

    This generator uses reshaped chunks (i.e. non overlapping zero-overhead chunks)
    as a basis and randomly selects a fraction of those chunks.
    This means that only start chunk number is randomized while the chunk relative offset
    is always the same. In other workds,
    """
    chunksize = int(chunksize)
    arr2d = reshaped_chunks(arr, chunksize)
    indices = random.sample(range(arr2d.shape[0]), num_samples)
    return IndexChunkGenerator(arr2d, lambda i: indices[i], num_samples, copy=copy)


def random_sample_chunks(arr, chunksize, num_samples):
    """
    Generate num_samples completely random sample chunks of size chunksize.
    
    The random indexes are generated on initialization,
    so subsequent calls using the same index return the same sample.


    """
    chunksize = int(chunksize)
    start_idxs = range(arr.shape[0] - (chunksize - 1))
    indices = random.sample(start_idxs, num_samples)
    return IndexChunkGenerator(arr, lambda i: slice(indices[i], indices[i] + chunksize), num_samples)


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


def array_to_chunkgen(arr):
    """
    Convert a potentially multidimensional NumPy array-like
    to a ChunkGenerator(), using the values along the first axis.
    """
    return IndexChunkGenerator(arr, lambda i: i, arr.shape[0])
