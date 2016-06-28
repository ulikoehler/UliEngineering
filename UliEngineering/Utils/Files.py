#!/usr/bin/env python3
"""
File utilities
"""
from toolz import functoolz

def count_lines(flo, isline=functoolz.compose(bool, str.strip)):
	"""
	Count the lines in a file.

	Takes a file-like object. Strings are treated as filenames.
	Returns the number of lines.
	"""
	must_close = False # Close only if we opened it
	# Open it if it is a strig
	if isinstance(flo, str):
		flo = open(flo, "r")
		must_close = True
	try:
		num_lines = 0
		for line in flo:
			num_lines += 1 if isline(line) else 0
		return num_lines
	finally:
		if must_close:
			flo.close()
