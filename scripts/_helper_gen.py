# Helper that generates _build_all.py content
import os

Q=chr(39);DQ=chr(34);BS=chr(92);NL=chr(10);EM=chr(8212);RA=chr(8594)

builder = os.path.join(os.path.dirname(__file__), '_build_all.py')

# Build all lines of the target file
lines = []

