"""The ``.etp`` print-job container -- "Erika Typewriter Print", version 1.

A print job is a flat list of motion and strike opcodes. All the geometry --
where the head is, how far to move, which key to hit -- is resolved on the
host; the firmware is a dumb interpreter that pushes bytes at the typewriter
whenever RTS says it is ready. That keeps the ESP32 side small enough to be
obviously correct, and lets the whole plan be verified offline.

Layout::

    offset size field
    0      4    magic  b"ETP1"
    4      1    version (1)
    5      1    flags   bit0 pitch12, bit1 home_each_row
    6      2    cols     uint16 LE   grid columns  (informational)
    8      2    rows     uint16 LE   grid rows     (informational)
    10     4    strikes  uint32 LE   number of glyph strikes, for progress
    14     4    body_len uint32 LE
    18     4    crc32    uint32 LE   CRC-32 of the body
    22     2    reserved
    24     ..   body

Opcodes are one byte, most followed by one operand byte. Operands are
unsigned; the encoder splits longer runs across repeated opcodes.

Keep this in sync with ``erika_ai/src/erika_image.h``.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"ETP1"
VERSION = 1
HEADER_SIZE = 24
HEADER_STRUCT = "<4sBBHHIII2x"

FLAG_PITCH12 = 0x01
FLAG_HOME_EACH_ROW = 0x02

# -- opcodes ---------------------------------------------------------------
OP_END = 0x00  # -            stop
OP_RIGHT = 0x01  # n            move right n half-steps
OP_LEFT = 0x02  # n            move left  n half-steps
OP_DOWN = 0x03  # n            feed paper down n half-lines
OP_UP = 0x04  # n            feed paper up   n half-lines
OP_CR = 0x05  # -            carriage return to left margin, no line feed
OP_STRIKE = 0x06  # code         type a glyph; head advances one full step
OP_STRIKE_NA = 0x07  # code         type a dead key; head does not advance
OP_DELAY = 0x08  # n            wait n * 10 ms
OP_MICRO_DOWN = 0x09  # n            n micro line steps down
OP_MICRO_UP = 0x0A  # n            n micro line steps up
OP_NEWLINE = 0x0B  # n            carriage return + n full line feeds

_HAS_OPERAND = {
    OP_RIGHT, OP_LEFT, OP_DOWN, OP_UP, OP_STRIKE, OP_STRIKE_NA,
    OP_DELAY, OP_MICRO_DOWN, OP_MICRO_UP, OP_NEWLINE,
}

OPCODE_NAMES = {
    OP_END: "END", OP_RIGHT: "RIGHT", OP_LEFT: "LEFT", OP_DOWN: "DOWN",
    OP_UP: "UP", OP_CR: "CR", OP_STRIKE: "STRIKE", OP_STRIKE_NA: "STRIKE_NA",
    OP_DELAY: "DELAY", OP_MICRO_DOWN: "MICRO_DOWN", OP_MICRO_UP: "MICRO_UP",
    OP_NEWLINE: "NEWLINE",
}

MAX_OPERAND = 0xFF


class EtpError(ValueError):
    pass


@dataclass
class Job:
    """A decoded print job."""

    body: bytes
    cols: int = 0
    rows: int = 0
    strikes: int = 0
    pitch: int = 10
    home_each_row: bool = True

    @property
    def flags(self) -> int:
        return (FLAG_PITCH12 if self.pitch == 12 else 0) | (
            FLAG_HOME_EACH_ROW if self.home_each_row else 0
        )


class Encoder:
    """Builds an opcode body, splitting over-long runs automatically."""

    def __init__(self):
        self._out = bytearray()
        self.strikes = 0

    def __len__(self) -> int:
        return len(self._out)

    def _emit(self, op: int, operand: int | None = None) -> None:
        self._out.append(op)
        if op in _HAS_OPERAND:
            if operand is None:
                raise EtpError(f"{OPCODE_NAMES[op]} requires an operand")
            if not 0 <= operand <= MAX_OPERAND:
                raise EtpError(f"operand {operand} out of range for {OPCODE_NAMES[op]}")
            self._out.append(operand)
        elif operand is not None:
            raise EtpError(f"{OPCODE_NAMES[op]} takes no operand")

    def _repeat(self, op: int, n: int) -> None:
        """Emit `op` as many times as needed to cover n units."""
        if n < 0:
            raise EtpError(f"negative count {n} for {OPCODE_NAMES[op]}")
        while n > MAX_OPERAND:
            self._emit(op, MAX_OPERAND)
            n -= MAX_OPERAND
        if n:
            self._emit(op, n)

    # -- motion ------------------------------------------------------------
    def right(self, half_steps: int) -> None:
        self._repeat(OP_RIGHT, half_steps)

    def left(self, half_steps: int) -> None:
        self._repeat(OP_LEFT, half_steps)

    def horizontal(self, delta_half_steps: int) -> None:
        if delta_half_steps > 0:
            self.right(delta_half_steps)
        elif delta_half_steps < 0:
            self.left(-delta_half_steps)

    def down(self, half_lines: int) -> None:
        self._repeat(OP_DOWN, half_lines)

    def up(self, half_lines: int) -> None:
        self._repeat(OP_UP, half_lines)

    def vertical(self, delta_half_lines: int) -> None:
        if delta_half_lines > 0:
            self.down(delta_half_lines)
        elif delta_half_lines < 0:
            self.up(-delta_half_lines)

    def carriage_return(self) -> None:
        self._emit(OP_CR)

    def newline(self, lines: int = 1) -> None:
        self._repeat(OP_NEWLINE, lines)

    def micro_down(self, n: int) -> None:
        self._repeat(OP_MICRO_DOWN, n)

    def micro_up(self, n: int) -> None:
        self._repeat(OP_MICRO_UP, n)

    # -- printing ----------------------------------------------------------
    def strike(self, code: int, advances: bool = True) -> None:
        # A motion code struck as a glyph would move the head without the plan
        # knowing about it, and everything after would land in the wrong place.
        # The firmware refuses these too; catch it here where it is debuggable.
        from erika import erika_codes as ec

        if code in ec.CONTROL_CODES:
            raise EtpError(
                f"0x{code:02X} ({ec.describe_code(code)}) is a motion code, "
                "not a glyph -- it cannot be struck"
            )
        self._emit(OP_STRIKE if advances else OP_STRIKE_NA, code)
        self.strikes += 1

    def delay_ms(self, ms: int) -> None:
        self._repeat(OP_DELAY, round(ms / 10))

    def end(self) -> None:
        self._emit(OP_END)

    def body(self) -> bytes:
        return bytes(self._out)


def pack(job: Job) -> bytes:
    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        VERSION,
        job.flags,
        job.cols,
        job.rows,
        job.strikes,
        len(job.body),
        zlib.crc32(job.body) & 0xFFFFFFFF,
    )
    assert len(header) == HEADER_SIZE, len(header)
    return header + job.body


def unpack(data: bytes) -> Job:
    if len(data) < HEADER_SIZE:
        raise EtpError(f"file is only {len(data)} bytes, need at least {HEADER_SIZE}")
    magic, version, flags, cols, rows, strikes, body_len, crc = struct.unpack(
        HEADER_STRUCT, data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise EtpError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise EtpError(f"unsupported version {version}")
    body = data[HEADER_SIZE : HEADER_SIZE + body_len]
    if len(body) != body_len:
        raise EtpError(f"truncated body: {len(body)} of {body_len} bytes")
    actual = zlib.crc32(body) & 0xFFFFFFFF
    if actual != crc:
        raise EtpError(f"CRC mismatch: header says {crc:08X}, body is {actual:08X}")
    return Job(
        body=body,
        cols=cols,
        rows=rows,
        strikes=strikes,
        pitch=12 if flags & FLAG_PITCH12 else 10,
        home_each_row=bool(flags & FLAG_HOME_EACH_ROW),
    )


def iter_ops(body: bytes):
    """Yield (offset, opcode, operand-or-None), validating as it goes."""
    i = 0
    while i < len(body):
        op = body[i]
        if op not in OPCODE_NAMES:
            raise EtpError(f"unknown opcode 0x{op:02X} at offset {i}")
        if op in _HAS_OPERAND:
            if i + 1 >= len(body):
                raise EtpError(f"truncated operand for {OPCODE_NAMES[op]} at {i}")
            yield i, op, body[i + 1]
            i += 2
        else:
            yield i, op, None
            i += 1


def disassemble(job: Job, limit: int | None = None) -> str:
    """Human-readable listing, with a running head-position model.

    The position columns are the whole point: they let you sanity-check that
    the plan lands where the optimizer intended without a typewriter.
    """
    from erika import erika_codes as ec

    lines = [
        f"; ETP1  grid {job.cols}x{job.rows}  strikes {job.strikes}  "
        f"pitch {job.pitch}  home_each_row {job.home_each_row}",
        f"; body {len(job.body)} bytes",
        ";  offset  col   row  opcode",
    ]
    x = y = 0  # half-steps from left margin, half-lines from the datum
    n = 0
    for off, op, operand in iter_ops(job.body):
        if op == OP_RIGHT:
            x += operand
        elif op == OP_LEFT:
            x -= operand
        elif op == OP_DOWN:
            y += operand
        elif op == OP_UP:
            y -= operand
        elif op == OP_CR:
            x = 0
        elif op == OP_NEWLINE:
            x, y = 0, y + 2 * operand
        elif op == OP_STRIKE:
            x += 2

        text = OPCODE_NAMES[op]
        if op in (OP_STRIKE, OP_STRIKE_NA):
            g = ec.glyph_for_code(operand)
            label = repr(g.char) if g else f"0x{operand:02X}"
            text = f"{text:<9} {label}"
        elif operand is not None:
            text = f"{text:<9} {operand}"
        lines.append(f"  {off:6d}  {x / 2:5.1f} {y / 2:5.1f}  {text}")

        n += 1
        if limit is not None and n >= limit:
            lines.append(f"  ... ({len(job.body)} bytes total)")
            break
    return "\n".join(lines)


def load(path: str) -> Job:
    with open(path, "rb") as f:
        return unpack(f.read())


def save(path: str, job: Job) -> int:
    data = pack(job)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Inspect an .etp print job")
    p.add_argument("file")
    p.add_argument("--limit", "-n", type=int, default=60,
                   help="max opcodes to list (0 for all)")
    a = p.parse_args(argv)
    job = load(a.file)
    print(disassemble(job, limit=a.limit or None))


if __name__ == "__main__":
    main()
