#!/usr/bin/env python3
"""
Utilities for iterables
"""

__all__ = ["PeekableIteratorWrapper", "ListIterator"]

class ListIterator(object):
    """
    Takes an iterable (like a list)
    and exposes a generator-like interface.

    The given iterable must support len()
    and index-based access
    for this algorithm to work.

    Equivalent to (v for v in lst)
    except calling len() reveals
    how many values are left to iterate.

    Use .index to access the current index.
    """
    def __init__(self, lst):
        self.index = 0
        self._lst = lst
        self._remaining = len(self._lst)

    def __iter__(self):
        return self

    def __next__(self):
        if self._remaining <= 0:
            raise StopIteration
        v = self._lst[self.index]
        self.index += 1
        self._remaining -= 1
        return v

    def __len__(self):
        """Remaining values"""
        return self._remaining

    


def iterable_to_iterator(it):
    """
    Given an iterable (like a list), generates
    an iterable out
    """

class PeekableIteratorWrapper(object):
    """
    Wraps an iterator and provides the additional
    capability of 'peeking' and un-getting values.

    Works by storing un-got values in a buffer
    that is emptied on call to next() before
    touching the child iterator.

    The buffer is managed in a stack-like manner
    (LIFO) so you can un-get multiple values.
    """
    def __init__(self, child):
        """
        Initialize a PeekableIteratorWrapper
        with a given child iterator
        """
        self.buffer = []
        self.child = child

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.buffer) > 0:
            return self.buffer.pop()
        return next(self.child)
    
    def __len__(self):
        """
        Returns len(child). Only supported
        if child support len().
        """
        return len(self.child)

    def unget(self, v):
        """
        Un-gets v so that v will be returned
        on the next call to __next__ (unless
        another value is un-got after this).
        """
        self.buffer.append(v)

    def peek(self):
        """
        Get the next value without removing it from
        the iterator.

        Note: Multiple subsequent calls to peek()
        without any calls to __next__() in between
        will return the same value.
        """
        val = next(self)
        self.unget(val)
        return val
