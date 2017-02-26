#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
Utilities for JSON encoding and decoding
"""
import json
import numpy as np


class NumPyEncoder(json.JSONEncoder):
    """
    A JSON encoder that is capable of encoding NumPy ndarray objects.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  # Generic scalars
            return obj.item()
        # Let the base class default method raise the TypeError
        raise TypeError("Unserializable object {} of type {}".format(
            obj, type(obj)))
