#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from UliEngineering.Filesystem.Hash import *
import tempfile
import hashlib
import os
import shutil

class TestHashFile(unittest.TestCase):
    def test_hash_file_native(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, world!")
            file_path = f.name
        assert hash_file_native(file_path) == hashlib.sha256(b"Hello, world!").hexdigest()
        os.unlink(file_path)

    def test_hash_file_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, world!")
            file_path = f.name
        self.assertEqual(hash_file_sha256(file_path), "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3")
        os.unlink(file_path)

    def test_hash_file_md5(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, world!")
            file_path = f.name
        self.assertEqual(hash_file_md5(file_path), "6cd3556deb0da54bca060b4c39479839")
        os.unlink(file_path)

    def test_hash_file_sha1(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, world!")
            file_path = f.name
        self.assertEqual(hash_file_sha1(file_path), "943a702d06f34599aee1f8da8ef9f7296031d699")
        os.unlink(file_path)
        
class TestHashDirectory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        # Create two files in main directory
        with open(os.path.join(self.tmpdir.name, "file1.txt"), "wb") as f:
            f.write(b"Hello, world!")
        with open(os.path.join(self.tmpdir.name, "file2.txt"), "wb") as f:
            f.write(b"Goodbye, world!")
        # Create a single file in a subdirectory
        subdir = os.path.join(self.tmpdir.name, "subdir")
        os.mkdir(subdir)
        with open(os.path.join(subdir, "file3.txt"), "wb") as f:
            f.write(b"Hello again, world!")

    def tearDown(self):
        shutil.rmtree(self.tmpdir.name)

    def test_hash_directory_non_recursive(self):
        results = hash_directory(self.tmpdir.name, recursive=False, relative_paths=True)
        self.assertEqual(len(results), 2)
        self.assertIn(("file1.txt", "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"), results)
        self.assertIn(("file2.txt", "a6ab91893bbd50903679eb6f0d5364dba7ec12cd3ccc6b06dfb04c044e43d300"), results)

    def test_hash_directory_recursive(self):
        results = hash_directory(self.tmpdir.name, recursive=True, relative_paths=True)
        print(results)
        self.assertEqual(len(results), 3)
        self.assertIn(("file1.txt", "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"), results)
        self.assertIn(("file2.txt", "a6ab91893bbd50903679eb6f0d5364dba7ec12cd3ccc6b06dfb04c044e43d300"), results)
        self.assertIn((os.path.join("subdir", "file3.txt"), "ef42b2ddfd8608161d0943b6c5cc349bf7b8f63c1261e393a348ffd24877b5ef"), results)

    def test_hash_directory_absolute_paths(self):
        results = hash_directory(self.tmpdir.name, recursive=True, relative_paths=False)
        self.assertEqual(len(results), 3)
        self.assertIn((os.path.join(self.tmpdir.name, "file1.txt"), "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"), results)
        self.assertIn((os.path.join(self.tmpdir.name, "file2.txt"), "a6ab91893bbd50903679eb6f0d5364dba7ec12cd3ccc6b06dfb04c044e43d300"), results)
        self.assertIn((os.path.join(self.tmpdir.name, "subdir", "file3.txt"), "ef42b2ddfd8608161d0943b6c5cc349bf7b8f63c1261e393a348ffd24877b5ef"), results)

