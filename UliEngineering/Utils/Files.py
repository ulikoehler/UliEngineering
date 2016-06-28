#!/usr/bin/env python3
"""
File utilities
"""
from toolz import functoolz
import operator

__standard_isline = functoolz.compose(bool, str.strip)
__csv_firstcol = functoolz.compose(operator.itemgetter(0), lambda s: s.partition(','))

def count_lines(flo, isline=__standard_isline):
	"""
	Count the lines in a file.

	Takes a file-like object. Strings are treated as filenames.
	Returns the number of lines.
	"""
	# Open it if it is a string
	if isinstance(flo, str):
		with open(flo, "r") as infile:
			return count_lines(infile, isline=isline)
	# Actual counting code
	num_lines = 0
	for line in flo:
		num_lines += 1 if isline(line) else 0
	return num_lines

def extract_column(flo, isline=__standard_isline, postproc=functoolz.identity, extractcol=__csv_firstcol):
	"""
	Lazily extract a column from a file, for example extract a column from a CSV file.
	The values are run through a postprocessing function and placed in a list which is returned.
	Lines which do not pass the isline function are ignored.
	"""
	# Open it if it is a string
	if isinstance(flo, str):
		with open(flo, "r") as infile:
			return extract_column(infile, isline=isline, postproc=postproc)
	# Actual counting code
	columns = []
	for line in flo:
		if not isline(line): continue
		col = postproc(extractcol(line))
		columns.append(col)
	return columns

def numpy_resize_insert(arr, val, index, growth_factor=1.5, min_growth=1000, max_growth=1000000):
	"""
	Append a value to a 1D numpy array. Resize dynamically if required.
	Returns the new array (which may be the same as the old array)
	"""
	# Array large enough ? If so, we can resize directly
	if index < arr.size:
		arr[index] = val
		return arr
	# OK, need to resize
	required_growth = (index - arr.size) + 1
	factor_growth = int(arr.size * growth_factor)
	# Resize at least min_growth & at most max_growth, but at least required_growth (overrides all other rules)
	growth = max(min(max(factor_growth, min_growth), max_growth), required_growth)
	# Resize (empty space is filled with copies of original array)
	arr = np.resize(arr, arr.size + growth)
	arr[index] = val
	return arr


def extract_numeric_column(flo, isline=__standard_isline, postproc=functoolz.identity, extractcol=__csv_firstcol):
	"""
	Like extract_column, but 
	"""
	# Open it if it is a string
	if isinstance(flo, str):
		with open(flo, "r") as infile:
			return extract_column(infile, isline=isline, postproc=postproc)
	# Actual counting code #TODO
	columns = []
	for line in flo:
		if not isline(line): continue
		col = postproc(extractcol(line))
		columns.append(col)
	return columns


