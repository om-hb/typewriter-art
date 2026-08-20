import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_report_header(config):
    """Say up front whether the cross-repository drift guards can run."""
    from tests.test_erika import FIRMWARE_SRC

    if FIRMWARE_SRC:
        return f"erika drift guards: active against {FIRMWARE_SRC}"
    return "erika drift guards: DISABLED (erika_ai/src not found)"


def pytest_terminal_summary(terminalreporter):
    """Warn loudly when the Python/C++ code tables went unchecked.

    The Erika codes, the .etp opcodes and the upload chunk size each exist
    twice -- once in Python here, once in C++ in erika_ai. A dozen-odd tests
    compare them, and they only run with the firmware checked out. Without it the suite
    still reports all-green, which is exactly the impression not to give.

    This goes in the summary rather than the header because the header is
    suppressed under -q, which is how the suite is normally run.
    """
    from tests.test_erika import FIRMWARE_SRC

    if FIRMWARE_SRC:
        return
    terminalreporter.write_sep("!", "drift guards did not run", red=True)
    terminalreporter.write_line(
        "erika_ai/src was not found, so the Python and C++ copies of the Erika\n"
        "codes, the .etp opcodes, the upload chunk size and which opcodes carry\n"
        "an operand were NOT compared.\n"
        "Check out erika_ai beside this repository, or point ERIKA_FIRMWARE_SRC\n"
        "at its src directory."
    )
