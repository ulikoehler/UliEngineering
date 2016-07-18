#!/usr/bin/env python3
"""
Compression utilities
"""
import gzip
import bz2
import lzma
import os.path

__open_map = {
	"": open,
	".gz": gzip.open,
	".bz2": bz2.open,
	".lzma": lzma.open,
	".xz": lzma.open
}

"""
Mode map for opening binary files that maps
normal open() single-char modes to text modes
and everything else to binary modes
"""
__mode_map = {
	"r": "rt", "rb": "rb", "w": "wb", "wb": "wb",
	"x": "xt", "xb": "xb", "a": "at", "ab": "ab",
	"rt": "rt", "wt": "wt", "xt": "xt", "at": "at"
}

def auto_open(filename, mode="r", charset="utf8", **kwargs):
	"""
	Automatically open a potentially compressed file using the right
	library variant of open().
	The correct decompression algorithm is selected by filename extension.
	This function can be used instead of open() and automatically selects
	the right mode (text or binary).
	"""
	extension = os.path.splitext(filename)[1]
	if extension not in __open_map:
		raise ValueError(
			"Unable to find correct decompression for extension '{0}' in filename {1}".format(extension, filename))
	open_fn = __open_map[extension]
	mode = __mode_map[mode] if extension else mode
	return open_fn(filename, mode, **kwargs)

