#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.Utils.String import *
from parameterized import parameterized
import unittest

class TestSplitNth(unittest.TestCase):
    def testSimple(self):
        self.assertEqual("a", split_nth("a,b,c,d,e,f"))
        self.assertEqual("a", split_nth("a,b,c,d,e,f", nth=1))
        self.assertEqual("b", split_nth("a,b,c,d,e,f", nth=2))
        self.assertEqual("c", split_nth("a,b,c,d,e,f", nth=3))
        self.assertEqual("d", split_nth("a,b,c,d,e,f", nth=4))
        self.assertEqual("e", split_nth("a,b,c,d,e,f", nth=5))
        self.assertEqual("f", split_nth("a,b,c,d,e,f", nth=6))

    def testMultiChar(self):
        self.assertEqual("ab", split_nth("ab,cd,ef,gh,ij,kl"))
        self.assertEqual("ab", split_nth("ab,cd,ef,gh,ij,kl", nth=1))
        self.assertEqual("cd", split_nth("ab,cd,ef,gh,ij,kl", nth=2))
        self.assertEqual("ef", split_nth("ab,cd,ef,gh,ij,kl", nth=3))
        self.assertEqual("gh", split_nth("ab,cd,ef,gh,ij,kl", nth=4))
        self.assertEqual("ij", split_nth("ab,cd,ef,gh,ij,kl", nth=5))
        self.assertEqual("kl", split_nth("ab,cd,ef,gh,ij,kl", nth=6))

    def testOtherDelimiter(self):
        self.assertEqual("a", split_nth("a;b;c;d;e;f", delimiter=';'))
        self.assertEqual("a", split_nth("a;b;c;d;e;f", delimiter=';', nth=1))
        self.assertEqual("b", split_nth("a;b;c;d;e;f", delimiter=';', nth=2))
        self.assertEqual("c", split_nth("a;b;c;d;e;f", delimiter=';', nth=3))
        self.assertEqual("d", split_nth("a;b;c;d;e;f", delimiter=';', nth=4))
        self.assertEqual("e", split_nth("a;b;c;d;e;f", delimiter=';', nth=5))
        self.assertEqual("f", split_nth("a;b;c;d;e;f", delimiter=';', nth=6))
        self.assertEqual("", split_nth("aa;bb;;dd;ee;ff", delimiter=';', nth=3))
        self.assertEqual("", split_nth("aa;bb;;", delimiter=';', nth=3))
        self.assertEqual("", split_nth("aa;bb;;", delimiter=';', nth=4))

    def testEmpty(self):
        self.assertEqual("", split_nth("aa,bb,,dd,ee,ff", nth=3))
        self.assertEqual("", split_nth("aa,bb,,", nth=3))
        self.assertEqual("", split_nth("aa,bb,,", nth=4))
        self.assertEqual("", split_nth("", nth=1))

    @parameterized.expand(["a", "abc", "abcdef", "abc,def"])
    def testInvalidFirst(self, param):
        with self.assertRaises(ValueError):
            split_nth(param, nth=3)

    def testInvalidOther(self):
        with self.assertRaises(ValueError):
            split_nth("abc,def", nth=3)

class TestStringUtils(unittest.TestCase):
    def test_partition_at_numeric_to_nonnumeric_boundary(self):
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("foo.123bar"), ("foo.123", "bar"))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("123s"), ("123", "s"))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("123"), ("123", ""))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("foo"), ("foo", ""))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("foo1bar"), ("foo1", "bar"))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("foo.123"), ("foo.123", ""))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("foo123bar"), ("foo123", "bar"))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary("123foo456"), ("123", "foo456"))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary(s=""), ("", ""))
        self.assertEqual(partition_at_numeric_to_nonnumeric_boundary(s="123.456km"), ("123.456", "km"))
