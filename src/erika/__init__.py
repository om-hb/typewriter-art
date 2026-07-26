"""Sigma SM 8200i (Erika) typewriter back-end for typewriter-art.

Turns the output of ``optimize.py`` (``results/choices.json``) into a motion
plan the ``erika_ai`` ESP32 firmware can execute on the physical typewriter.
"""

import os
import sys

#: The typewriter-art ``src`` directory. Everything upstream -- optimize.py,
#: utils.py, charsets/, results/ -- is resolved relative to it, so the CLI
#: works from any working directory.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

__all__ = ["SRC_DIR", "erika_codes", "etp", "planner", "preview", "emulate"]
