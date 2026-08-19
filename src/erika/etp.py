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

The version in the header is what protects a job from the wrong firmware, and
the firmware refuses an opcode it does not know -- so an opcode added here is
safe to add without a version bump: an older device fails loudly on the first
one it meets rather than typing something else. That is why ``OP_SET_FORCE``
did not become version 2.

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
OP_SET_FORCE = 0x0C  # n            strike force (Anschlagstärke) for what follows
OP_RAW = 0x0D  # byte         send this byte to the machine, untouched
OP_NO_ADVANCE = 0x0E  # -     the next strike prints where the head stands

_HAS_OPERAND = {
    OP_RIGHT, OP_LEFT, OP_DOWN, OP_UP, OP_STRIKE, OP_STRIKE_NA,
    OP_DELAY, OP_MICRO_DOWN, OP_MICRO_UP, OP_NEWLINE, OP_SET_FORCE,
    OP_RAW,
}

OPCODE_NAMES = {
    OP_END: "END", OP_RIGHT: "RIGHT", OP_LEFT: "LEFT", OP_DOWN: "DOWN",
    OP_UP: "UP", OP_CR: "CR", OP_STRIKE: "STRIKE", OP_STRIKE_NA: "STRIKE_NA",
    OP_DELAY: "DELAY", OP_MICRO_DOWN: "MICRO_DOWN", OP_MICRO_UP: "MICRO_UP",
    OP_NEWLINE: "NEWLINE", OP_SET_FORCE: "SET_FORCE", OP_RAW: "RAW",
    OP_NO_ADVANCE: "NO_ADVANCE",
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
        """Type the key with this code.

        Only a key. The interface's vocabulary does not stop at the motion block
        -- it runs to 0xAF -- and the codes above it fail in two different ways
        if one reaches the machine where a glyph was meant:

            a motion code          moves the head, so every mark after it lands
                                   in the wrong place. Wrong, and visible.

            an operand-carrying    eats the byte after it, so the firmware and
            code                   the machine disagree about where opcodes
                                   begin for the rest of the job. Every byte is
                                   still legal and the CRC still passes over an
                                   intact file; the sheet is half an hour of
                                   arbitrary motion and nothing says why.

        So the test is the wheel's own range rather than a list of the codes we
        happen to have named. The firmware makes the same check, against the
        same two bounds; this one is here because it is the one with a stack
        trace attached.
        """
        from erika import erika_codes as ec

        if not ec.is_glyph_code(code):
            raise EtpError(
                f"0x{code:02X} ({ec.describe_code(code)}) is not a key on the "
                f"wheel -- those run 0x{ec.MIN_GLYPH_CODE:02X}.."
                f"0x{ec.MAX_GLYPH_CODE:02X} -- so it cannot be struck"
                + (
                    ". It is a motion: striking it would move the head without "
                    "the plan knowing"
                    if code in ec.CONTROL_CODES
                    else ""
                )
                + (
                    ". It takes the following byte with it: striking it would "
                    "put the interpreter one byte out of step for the rest of "
                    "the job"
                    if code in ec.OPERAND_CODES
                    else ""
                )
            )
        self._emit(OP_STRIKE if advances else OP_STRIKE_NA, code)
        self.strikes += 1

    def set_force(self, force: int) -> None:
        """Set the strike force (Anschlagstärke) for every strike that follows.

        One opcode here, two bytes on the wire. A machine that does not honour
        the command types the force byte as a character instead, so the value
        has to be one that is harmless when typed -- inside the wheel's range.
        Above it lies the motion block, and above that the commands, seven of
        which would swallow the byte after the force as their own operand. That
        is the same pair of failures `strike()` refuses above, and the reason
        the probe sheet only sweeps values the wheel could have typed.
        """
        from erika import erika_codes as ec

        if not ec.is_usable_force(force):
            raise EtpError(
                f"strike force 0x{force:02X} ({ec.describe_code(force)}) is a "
                "command, not an inert character. If this machine ignores the "
                "force command the byte arrives as one, and that one would "
                + (
                    "take the byte after it with it and desynchronise the rest "
                    "of the job"
                    if force in ec.OPERAND_CODES
                    else "move the head"
                )
                + f" -- pick a force of 0x{ec.MAX_FORCE:02X} or below"
            )
        self._emit(OP_SET_FORCE, force)

    def no_advance(self) -> None:
        """The next strike prints where the head stands (Doppeldruck, 0xA9).

        How a stack of characters in one cell is typed. The alternative, and
        what the planner did before this existed, is to strike and then
        backspace -- which spends the escapement's repeatability on every
        stacked character, and a picture is mostly stacked characters. The
        calibration sheet has a section for backspace registration precisely
        because it is a thing that can go wrong.

        The byte count is the same either way: for k glyphs in a cell it is
        2k-1 wire bytes with backspaces and 2k-1 with this. The difference is
        entirely in where the marks land.

        Deliberately not OP_STRIKE_NA, which means "this key does not advance"
        and is true of the four dead keys mechanically. This one is a property
        of the *next* strike rather than of the key, which is what the machine's
        own code is, and conflating them would need the firmware to carry a
        fourth hand-mirrored table of which codes are dead keys.
        """
        self._emit(OP_NO_ADVANCE)

    # -- probing -----------------------------------------------------------
    def raw(self, byte: int) -> None:
        """Put one byte on the wire exactly as given.

        The escape hatch, and the only opcode about which the firmware has no
        opinion at all. It exists because the interface answers to about sixty
        control codes and this pipeline uses eleven of them, and the only way to
        find out what the other fifty do on *this* machine is to send them and
        look at the paper. ``erika.pipeline codes`` is that sheet.

        What it costs, and why nothing but a probe sheet should use it:

        - Neither the firmware nor ``disassemble`` nor the emulator knows what
          the byte does, beyond the handful in ``erika_codes.CONTROL_CODE_NAMES``
          they have been taught. Offline verification against the optimizer's
          mockup -- the property the whole .etp arrangement exists for -- is only
          as good as that model.
        - Several codes take the byte after them as an operand, so a raw command
          is two raw bytes and getting the pair wrong desynchronises the machine
          rather than the file. ``raw_command`` is the way to say it that cannot
          get that wrong.
        - The firmware pays a full character delay for a raw byte, being unable
          to tell a mode switch from an inch of carriage travel. A command that
          moves further than that wants an explicit ``delay_ms`` after it.
        - A job containing raw bytes cannot be resumed part-way with
          ``IMG PRINT pass N``: the firmware replays them, because a raw byte is
          more often machine state than motion, and replaying a *motion* while
          fast-forwarding would type the rest of the sheet in the wrong place.

        Once a code has been settled on paper it should stop being raw and
        become an opcode with a position model, which is what the rest of the
        chain needs in order to keep checking itself.
        """
        if not 0 <= byte <= 0xFF:
            raise EtpError(f"raw byte {byte} is not a byte")
        self._emit(OP_RAW, byte)

    def raw_command(self, code: int, operand: int | None = None) -> None:
        """One control code and, where it takes one, its operand.

        Refuses the two ways of getting the pair wrong: an operand for a code
        that does not take one, and -- the one that matters -- no operand for a
        code that does, which leaves the machine treating whatever comes next as
        the count. The file would be intact and the CRC would pass.
        """
        from erika import erika_codes as ec

        wants = code in ec.OPERAND_CODES
        if wants and operand is None:
            raise EtpError(
                f"0x{code:02X} ({ec.describe_code(code)}) takes an operand -- "
                "without one the machine reads the next byte as its own"
            )
        if operand is not None and not wants:
            raise EtpError(
                f"0x{code:02X} ({ec.describe_code(code)}) takes no operand"
            )
        self.raw(code)
        if operand is not None:
            self.raw(operand)

    def carriage_steps(self, steps: int) -> None:
        """Move the carriage by a signed count of 1/120" steps, via 0xA5."""
        from erika import erika_codes as ec

        while steps:
            take = max(-ec.MAX_STEPS_PER_COMMAND,
                       min(ec.MAX_STEPS_PER_COMMAND, steps))
            self.raw_command(ec.CARRIAGE_STEPS, ec.encode_step_operand(take))
            steps -= take

    def platen_steps(self, steps: int) -> None:
        """Feed the paper by a signed count of 1/240" steps, via 0xA6.

        Splits around the counts the table forbids as well as around the
        operand's range: a run of five goes out as 2 + 2 + 1 rather than as one
        command the mechanism is documented to refuse.
        """
        from erika import erika_codes as ec

        while steps:
            take = max(-ec.MAX_STEPS_PER_COMMAND,
                       min(ec.MAX_STEPS_PER_COMMAND, steps))
            if abs(take) in ec.FORBIDDEN_PLATEN_STEPS:
                # Split off a 2 and let the loop deal with what is left, which is
                # 1, 2 or a 2-and-something -- never a forbidden count again.
                take = 2 if take > 0 else -2
            self.raw_command(ec.PLATEN_STEPS, ec.encode_step_operand(take))
            steps -= take

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

    raw = [operand for _, op, operand in iter_ops(job.body) if op == OP_RAW]
    unmodelled = sorted(
        {
            b
            for b in raw
            if b not in ec.OPERAND_CODES and b != ec.NO_ADVANCE
            and b in ec.CONTROL_CODE_NAMES
        }
    )
    lines = [
        f"; ETP1  grid {job.cols}x{job.rows}  strikes {job.strikes}  "
        f"pitch {job.pitch}  home_each_row {job.home_each_row}",
        f"; body {len(job.body)} bytes",
    ]
    if unmodelled:
        # Said out loud rather than passed over: the columns below are a model,
        # and these bytes are outside it. Most are mode switches with no effect
        # on position, but "most" is not the same as "checked".
        lines.append(
            "; the column and row below do NOT account for "
            + ", ".join(f"0x{b:02X} {ec.CONTROL_CODE_NAMES[b]}" for b in unmodelled)
        )
    lines.append(";  offset  col   row  opcode")
    x = y = 0  # half-steps from left margin, half-lines from the datum
    # Sub-half-step and sub-half-line residue, in the machine's own motor steps.
    # Only 0xA5 and 0xA6 can produce it, and only through OP_RAW.
    x_fine = y_fine = 0
    per_half_step = ec.carriage_steps_per_half_step(job.pitch)
    force: int | None = None
    pending_raw: int | None = None  # a raw command waiting for its operand byte
    no_advance = False  # 0xA9 seen: the next strike does not move the carriage
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
            if no_advance:
                no_advance = False
            else:
                x += 2
        elif op == OP_SET_FORCE:
            force = operand
        elif op == OP_RAW:
            if pending_raw is not None:
                steps = ec.decode_step_operand(operand)
                if pending_raw == ec.CARRIAGE_STEPS:
                    x_fine += steps
                    carry, x_fine = divmod(x_fine, per_half_step)
                    x += carry
                elif pending_raw == ec.PLATEN_STEPS:
                    y_fine += steps
                    carry, y_fine = divmod(y_fine, ec.PLATEN_STEPS_PER_HALF_LINE)
                    y += carry
                pending_raw = None
            elif operand in ec.OPERAND_CODES:
                pending_raw = operand
            elif operand == ec.NO_ADVANCE:
                no_advance = True
        elif op == OP_NO_ADVANCE:
            no_advance = True

        text = OPCODE_NAMES[op]
        if op == OP_RAW:
            text = f"{text:<9} 0x{operand:02X}"
            named = ec.CONTROL_CODE_NAMES.get(operand)
            if pending_raw is not None and pending_raw == operand:
                text += f"  {named} (operand follows)"
            elif named:
                text += f"  {named}"
        elif op in (OP_STRIKE, OP_STRIKE_NA):
            g = ec.glyph_for_code(operand)
            label = repr(g.char) if g else f"0x{operand:02X}"
            # The force in force is what this strike lands with, and a picture
            # that types at several is unreadable without it.
            if force is not None:
                label = f"{label} @f{force}"
            text = f"{text:<9} {label}"
        elif op == OP_SET_FORCE:
            text = f"{text:<9} 0x{operand:02X}"
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
