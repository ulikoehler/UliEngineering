#!/usr/bin/env python3
"""
Concurrency utilities
"""
import concurrent.futures
import os
import queue

__all__ = ["QueuedThreadExecutor"]

class QueuedThreadExecutor(concurrent.futures.ThreadPoolExecutor):
    """
    In contrast to the normal ThreadPoolExecutor, this executor has
    the advantage of having a configurable queue size,
    enabling more efficient processing especially with irregular,
    slow or asynchronous queue feeders
    """
    def __init__(self, nthreads=None, queue_size=100):
        if nthreads is None:
            nthreads = os.cpu_count() or 4
        super().__init__(nthreads)
        self._work_queue = queue.Queue(queue_size)