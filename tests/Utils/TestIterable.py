#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from nose.tools import assert_equal, assert_true, raises, assert_false
from UliEngineering.Utils.Iterable import *
import unittest


class TestListIterator(unittest.TestCase):
    def test_to_list(self):
        iterable = [1,2,3,4,5]
        it = ListIterator(iterable)
        assert_equal(len(it), 5)
        assert_equal(list(it), iterable)
    
    def test_next(self):
        iterable = [3,4,5]
        it = ListIterator(iterable)
        assert_equal(it.index, 0)
        assert_equal(len(it), 3)
        # 1st
        assert_equal(next(it), 3)
        assert_equal(len(it), 2)
        assert_equal(it.index, 1)
        # 2nd
        assert_equal(next(it), 4)
        assert_equal(len(it), 1)
        assert_equal(it.index, 2)
        # 3rd
        assert_equal(next(it), 5)
        assert_equal(len(it), 0)
        assert_equal(list(it), [])
    

class TestPeekableIteratorWrapper(unittest.TestCase):
    def test_list_2without_peek(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        assert_equal(list(it), iterable)

    def test_list_unget(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        assert_equal(len(it), 5)
        it.unget(0)
        it.unget(-1)
        assert_equal(list(it), [-1, 0, 1, 2, 3, 4, 5])

    def test_list_peek_unget(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        assert_equal(it.peek(), 1)
        it.unget(0)
        assert_equal(it.peek(), 0)
        it.unget(-1)
        assert_equal(it.peek(), -1)
        assert_equal(list(it), [-1, 0, 1, 2, 3, 4, 5])


    def test_list_peek_unget_next(self):
        iterable = [1,2,3]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        # 1
        assert_equal(it.peek(), 1)
        assert_equal(next(it), 1)
        assert_equal(it.peek(), 2)
        assert_equal(it.peek(), 2)
        # 5
        it.unget(5)
        assert_equal(it.peek(), 5)
        assert_equal(next(it), 5)
        assert_equal(it.peek(), 2)
        # 2
        assert_equal(next(it), 2)
        assert_equal(it.peek(), 3)
        # 3
        assert_equal(next(it), 3)
        assert_equal(list(it), [])

    def test_has_next(self):
        # has_next() might influence next() so this is a separate test.
        # Short test
        it = PeekableIteratorWrapper(ListIterator([]))
        assert_false(it.has_next())
        # Main test
        iterable = [1,2,3]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        # 1
        assert_true(it.has_next())
        assert_equal(it.peek(), 1)
        assert_equal(next(it), 1)
        assert_equal(it.peek(), 2)
        assert_equal(it.peek(), 2)
        # 5
        assert_true(it.has_next())
        it.unget(5)
        assert_true(it.has_next())
        assert_equal(it.peek(), 5)
        assert_equal(next(it), 5)
        assert_equal(it.peek(), 2)
        assert_true(it.has_next())
        # 2
        assert_true(it.has_next())
        assert_equal(next(it), 2)
        assert_equal(it.peek(), 3)
        assert_true(it.has_next())
        # 3
        assert_true(it.has_next())
        assert_equal(next(it), 3)
        assert_false(it.has_next())
        assert_equal(list(it), [])
        assert_false(it.has_next())