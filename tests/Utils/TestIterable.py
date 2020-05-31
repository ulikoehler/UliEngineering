#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal
from UliEngineering.Utils.Iterable import *
import unittest


class TestListIterator(unittest.TestCase):
    def test_to_list(self):
        iterable = [1,2,3,4,5]
        it = ListIterator(iterable)
        self.assertEqual(len(it), 5)
        self.assertEqual(list(it), iterable)
    
    def test_next(self):
        iterable = [3,4,5]
        it = ListIterator(iterable)
        self.assertEqual(it.index, 0)
        self.assertEqual(len(it), 3)
        # 1st
        self.assertEqual(next(it), 3)
        self.assertEqual(len(it), 2)
        self.assertEqual(it.index, 1)
        # 2nd
        self.assertEqual(next(it), 4)
        self.assertEqual(len(it), 1)
        self.assertEqual(it.index, 2)
        # 3rd
        self.assertEqual(next(it), 5)
        self.assertEqual(len(it), 0)
        self.assertEqual(list(it), [])
    

class TestPeekableIteratorWrapper(unittest.TestCase):
    def test_list_2without_peek(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        self.assertEqual(list(it), iterable)

    def test_list_unget(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        self.assertEqual(len(it), 5)
        it.unget(0)
        it.unget(-1)
        self.assertEqual(list(it), [-1, 0, 1, 2, 3, 4, 5])

    def test_list_peek_unget(self):
        iterable = [1,2,3,4,5]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        self.assertEqual(it.peek(), 1)
        it.unget(0)
        self.assertEqual(it.peek(), 0)
        it.unget(-1)
        self.assertEqual(it.peek(), -1)
        self.assertEqual(list(it), [-1, 0, 1, 2, 3, 4, 5])


    def test_list_peek_unget_next(self):
        iterable = [1,2,3]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        # 1
        self.assertEqual(it.peek(), 1)
        self.assertEqual(next(it), 1)
        self.assertEqual(it.peek(), 2)
        self.assertEqual(it.peek(), 2)
        # 5
        it.unget(5)
        self.assertEqual(it.peek(), 5)
        self.assertEqual(next(it), 5)
        self.assertEqual(it.peek(), 2)
        # 2
        self.assertEqual(next(it), 2)
        self.assertEqual(it.peek(), 3)
        # 3
        self.assertEqual(next(it), 3)
        self.assertEqual(list(it), [])

    def test_has_next(self):
        # has_next() might influence next() so this is a separate test.
        # Short test
        it = PeekableIteratorWrapper(ListIterator([]))
        self.assertFalse(it.has_next())
        # Main test
        iterable = [1,2,3]
        it = PeekableIteratorWrapper(ListIterator(iterable))
        # 1
        self.assertTrue(it.has_next())
        self.assertEqual(it.peek(), 1)
        self.assertEqual(next(it), 1)
        self.assertEqual(it.peek(), 2)
        self.assertEqual(it.peek(), 2)
        # 5
        self.assertTrue(it.has_next())
        it.unget(5)
        self.assertTrue(it.has_next())
        self.assertEqual(it.peek(), 5)
        self.assertEqual(next(it), 5)
        self.assertEqual(it.peek(), 2)
        self.assertTrue(it.has_next())
        # 2
        self.assertTrue(it.has_next())
        self.assertEqual(next(it), 2)
        self.assertEqual(it.peek(), 3)
        self.assertTrue(it.has_next())
        # 3
        self.assertTrue(it.has_next())
        self.assertEqual(next(it), 3)
        self.assertFalse(it.has_next())
        self.assertEqual(list(it), [])
        self.assertFalse(it.has_next())


class TestSkipFirst(unittest.TestCase):
    def test_list(self):
        self.assertEqual([2,3,4,5], list(skip_first([1,2,3,4,5])))
        self.assertEqual([], list(skip_first([])))
    def test_gen(self):
        self.assertEqual([2,3,4,5], list(skip_first(v for v in [1,2,3,4,5])))
        self.assertEqual([], list(skip_first(v for v in [])))
    