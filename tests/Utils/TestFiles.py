#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_allclose
from UliEngineering.Utils.Files import *
from UliEngineering.Utils.Temporary import *
import unittest
import os.path

class TestFileUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = AutoDeleteTempfileGenerator()

    def testCountLinesFilelike(self):
        self.assertEqual(1, count_lines(io.StringIO("foo")))
        self.assertEqual(1, count_lines(io.StringIO("foo\n")))
        self.assertEqual(1, count_lines(io.StringIO("foo\n\n")))
        self.assertEqual(1, count_lines(io.StringIO("foo\n\n\n")))
        self.assertEqual(2, count_lines(io.StringIO("foo\n\n f\n")))
        self.assertEqual(2, count_lines(io.StringIO("foo\na\n\n")))
        self.assertEqual(2, count_lines(io.StringIO("foo\n\n\na")))

        self.assertEqual(3, count_lines(io.StringIO("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")))

    def testCountLinesTempfile(self):
        # Test 1
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\n\n\na")
        handle.close()
        self.assertEqual(2, count_lines(fname))
        # Test 2
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")
        handle.close()
        self.assertEqual(3, count_lines(fname))

    def testExtractColumn(self):
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\nbar\n\na")
        handle.close()
        # Read back
        self.assertEqual(extract_column(fname), ["foo", "bar", "a"])

    def testExtractNumericColumn(self):
        handle, fname = self.tmp.mkftemp()
        handle.write("3.2\n2.4\n\n1.5")
        handle.close()
        # Read back
        assert_allclose(extract_numeric_column(fname), [3.2, 2.4, 1.5])

    def testTextIO(self):
        tmpdir = self.tmp.mkdtemp()
        fname = os.path.join(tmpdir, "test.txt")

        write_textfile(fname, "foobar")
        # Check file
        self.assertTrue(os.path.isfile(fname))
        with open(fname) as infile:
            self.assertEqual(infile.read(), "foobar")
        # Read back
        txt = read_textfile(os.path.join(tmpdir, "test.txt"))
        self.assertEqual(txt, "foobar")

    def test_list_recursive(self):
        tmpdir = self.tmp.mkdtemp()
        # Make platform-dependent filename
        filename_test2 = os.path.join("dir", "test2.txt")
        # Create test files
        write_textfile(os.path.join(tmpdir, "test.txt"), "")
        write_textfile(os.path.join(tmpdir, filename_test2), "")

        self.assertEqual(["test.txt", filename_test2],
            list(list_recursive(tmpdir, relative=True, files_only=True)))
        self.assertEqual(["test.txt", "dir/", filename_test2],
            list(list_recursive(tmpdir, relative=True, files_only=False)))
        self.assertEqual([os.path.join(tmpdir, "test.txt"),
                       os.path.join(tmpdir, filename_test2)],
            list(list_recursive(tmpdir, relative=False, files_only=True)))

    def test_(self):
        inp = ['ne_10m_admin_0_countries.README.html',
               'ne_10m_admin_0_countries.VERSION.txt',
               'ne_10m_admin_0_countries.dbf',
               'ne_10m_admin_0_countries.prj',
               'ne_10m_admin_0_countries.shp',
               'ne_10m_admin_0_countries.shx',
               'ne_10m_admin_0_countries.cpg']
        exp = [['ne_10m_admin_0_countries.dbf',
                'ne_10m_admin_0_countries.prj',
                'ne_10m_admin_0_countries.shp']]
        self.assertEqual(exp, list(find_datasets_by_extension(
            inp, (".dbf", ".prj", ".shp"))))