#!/usr/bin/env python3
"""
Concurrency utilities
"""
import concurrent.futures
import os

__all__ = ["new_thread_executor"]


def new_thread_executor(nthreads=None):
    """
    Create a new thread-based concurrent.futures executor that
    is optimized for CPU-bound work
    """
    if nthreads is None:
        nthreads = os.cpu_count() or 4
    return concurrent.futures.ThreadPoolExecutor(nthreads)