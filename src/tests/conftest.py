import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("MPLBACKEND", "Agg")
