#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT-related code that is used elsewhere to break up cyclic imports
"""
from collections import namedtuple

FFTResult = namedtuple("FFTResult", ["frequencies", "amplitudes", "angles"])