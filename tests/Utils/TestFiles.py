#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false
from UliEngineering.Utils.Files import *
from UliEngineering.Utils.Temporary import *

class TestFileUtils(object):
    def __init__(self):
        self.tmp = AutoDeleteTempfileGenerator()

    def testCountLinesFilelike(self):
        assert_equal(1, count_lines(io.StringIO("foo")))
        assert_equal(1, count_lines(io.StringIO("foo\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n")))
        assert_equal(1, count_lines(io.StringIO("foo\n\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n f\n")))
        assert_equal(2, count_lines(io.StringIO("foo\na\n\n")))
        assert_equal(2, count_lines(io.StringIO("foo\n\n\na")))

        assert_equal(3, count_lines(io.StringIO("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")))

    def testCountLinesTempfile(self):
        # Test 1
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\n\n\na")
        handle.close()
        assert_equal(2, count_lines(fname))
        # Test 2
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\r\n\r\n\r\n\na\r\na\n\n\n\r\n\r\r\n")
        handle.close()
        assert_equal(3, count_lines(fname))

    def testExtractColumn(self):
        handle, fname = self.tmp.mkftemp()
        handle.write("foo\nbar\n\na")
        handle.close()
        # Read back
        assert_equal(extract_column(fname), ["foo", "bar", "a"])

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
        assert_true(os.path.isfile(fname))
        with open(fname) as infile:
            assert_equal(infile.read(), "foobar")
        # Read back
        txt = read_textfile(os.path.join(tmpdir, "test.txt"))
        assert_equal(txt, "foobar")

    def test_list_recursive(self):
        tmpdir = self.tmp.mkdtemp()
        # Create test files
        write_textfile(os.path.join(tmpdir, "test.txt"), "")
        write_textfile(os.path.join(tmpdir, "dir/test2.txt"), "")

        assert_equal(["test.txt", "dir/test2.txt"],
            list(list_recursive(tmpdir, relative=True, files_only=True)))
        assert_equal(["test.txt", "dir/", "dir/test2.txt"],
            list(list_recursive(tmpdir, relative=True, files_only=False)))
        assert_equal([os.path.join(tmpdir, "test.txt"),
                       os.path.join(tmpdir, "dir/test2.txt")],
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
        assert_equal(exp, list(find_datasets_by_extension(
            inp, (".dbf", ".prj", ".shp"))))