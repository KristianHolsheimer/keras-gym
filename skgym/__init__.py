import sys
import warnings

if sys.version_info[:2] < (3, 6):
    warnings.warn("scikit-gym was written for python version 3.6 and up")
