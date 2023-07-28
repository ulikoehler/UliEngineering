#!/usr/bin/env python3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import os

__all__ = ["hash_file_sha256", "hash_file_md5", "hash_file_sha1", "hash_directory"]

def hash_file(file_path, hash_type=hashlib.sha256, binary=False, buffer_size=65536):
    """
    Compute the hash of a file using the specified hash algorithm using Python's hashlib

    :param file_path: The path to the file to hash
    :type file_path: str
    :param hash_type: A function returning a hash object. Typically hashlib.sha256 or hashlib.md5
    :type hash_type: callable
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    hash_func = hash_type()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(buffer_size)
            if not data:
                break
            hash_func.update(data)
    return hash_func.digest() if binary else hash_func.hexdigest()

def hash_file_sha256(file_path, binary=False, buffer_size=65536):
    """
    Compute the SHA256 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return hash_file(file_path, hash_type=hashlib.sha256, binary=binary, buffer_size=buffer_size)

def hash_file_md5(file_path, binary=False, buffer_size=65536):
    """
    Compute the MD5 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return hash_file(file_path, hash_type=hashlib.md5, binary=binary, buffer_size=buffer_size)

def hash_file_sha1(file_path, binary=False, buffer_size=65536):
    """
    Compute the SHA1 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return hash_file(file_path, hash_type=hashlib.sha1, binary=binary, buffer_size=buffer_size)

def hash_directory(directory, recursive=False, hash_type=hashlib.sha256, binary=False, buffer_size=65536, concurrency=os.cpu_count()):
    """
    List all files in a directory (recursively, depending on options)
    and compute the hash of each file.
    
    The file hashes are computed concurrently using a ThreadPoolExecutor.
    
    Returns tuples (filename, hash)
    """
    results = [] # List of (filename, sha256sum) tuples
    with ThreadPoolExecutor(concurrency) as executor:
        futures = []
        # Walk through the directory and start futures
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                future = executor.submit(
                    hash_file, file_path, hash_type=hash_type,
                    binary=binary, buffer_size=buffer_size
                )
                futures.append((future, file_path))
            # if not recursive, break after first iteration
            if not recursive:
                break
        # Wait until finished and collect results
        for future, file_path in futures:
            sha256sum = future.result()
            results.append((file_path, sha256sum))
    return results
