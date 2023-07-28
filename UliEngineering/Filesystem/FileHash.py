#!/usr/bin/env python3
import hashlib

def compute_file_hash(file_path, hash_obj=hashlib.sha256, binary=False, buffer_size=65536):
    """
    Compute the hash of a file using the specified hash algorithm using Python's hashlib

    :param file_path: The path to the file to hash
    :type file_path: str
    :param hash_obj: A function returning a hash object. Typically hashlib.sha256 or hashlib.md5
    :type hash_obj: callable
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    hash_func = hash_obj()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(buffer_size)
            if not data:
                break
            hash_func.update(data)
    return hash_func.digest() if binary else hash_func.hexdigest()

def compute_file_sha256(file_path, binary=False, buffer_size=65536):
    """
    Compute the SHA256 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return compute_file_hash(file_path, hash_obj=hashlib.sha256, binary=binary, buffer_size=buffer_size)

def compute_file_md5(file_path, binary=False, buffer_size=65536):
    """
    Compute the MD5 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return compute_file_hash(file_path, hash_obj=hashlib.md5, binary=binary, buffer_size=buffer_size)

def compute_file_sha1(file_path, binary=False, buffer_size=65536):
    """
    Compute the SHA1 hash of a file.

    :param file_path: The path to the file to hash
    :type file_path: str
    :return: If binary, The hexadecimal digest of the file hash
    :rtype: str
    """
    return compute_file_hash(file_path, hash_obj=hashlib.sha1, binary=binary, buffer_size=buffer_size)