"""Tests for the Sigma SM 8200i back-end.

The important ones are at the bottom: a synthetic image is planned, encoded,
run through a virtual typewriter, and the marks it leaves are compared with
where the plan said they should go. That exercises every stage between the
optimizer's output and the paper.
"""

from __future__ import annotations

import json
import os
import re
import struct

import numpy as np
import pytest

from erika import emulate
from erika import erika_codes as ec
from erika import etp, planner, preview
from erika.planner import Charset, PlanError

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_firmware_src() -> str | None:
    """Locate erika_ai/src, which lives in a separate repository.

    Six tests below are the only guard against the Erika codes, the .etp
    opcodes and the upload chunk size drifting apart between their Python and
    C++ copies -- including two that check the *shape* of the opcode stream
    rather than its constants, which is what adding an opcode gets wrong. They
    need the firmware checked out, and the conventional layout is as a sibling of
    this repository -- but that is a convention, not a guarantee: checkout names
    and locations differ between machines.

    So the path is overridable, and conftest.py shouts if it cannot be found.
    A silently skipped drift guard is worse than none, because the suite still
    goes green.
    """
    override = os.environ.get("ERIKA_FIRMWARE_SRC")
    if override:
        return override if os.path.isdir(override) else None
    workspace = os.path.dirname(os.path.dirname(SRC))
    guess = os.path.normpath(os.path.join(workspace, "erika_ai", "src"))
    return guess if os.path.isdir(guess) else None


FIRMWARE_SRC = _find_firmware_src()
NO_FIRMWARE = FIRMWARE_SRC is None
NO_FIRMWARE_REASON = (
    "erika_ai/src not found -- the Python/C++ drift guards did NOT run. "
    "Check out erika_ai beside this repository, or set ERIKA_FIRMWARE_SRC."
)


# ---------------------------------------------------------------------------
# code tables
# ---------------------------------------------------------------------------


def test_glyph_codes_are_unique_and_not_control_codes():
    codes = [g.code for g in ec.GLYPHS + ec.DEAD_KEY_GLYPHS]
    assert len(codes) == len(set(codes))
    assert not set(codes) & ec.CONTROL_CODES


def test_dead_keys_do_not_advance():
    assert all(not g.advances for g in ec.DEAD_KEY_GLYPHS)
    assert all(g.advances for g in ec.GLYPHS)


def test_cell_aspect_matches_the_machines_geometry():
    # 10 pitch: 2.54 mm wide cells on 4.233 mm lines.
    assert ec.cell_aspect(10) == pytest.approx(25.4 / 6 / 2.54)
    assert ec.cell_aspect(12) == pytest.approx(2.0)


def _parse_cpp_defines(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for name, value in re.findall(r"#define\s+(\w+)\s+(0x[0-9A-Fa-f]+|\d+)", text):
        out[name] = int(value, 0)
    for name, value in re.findall(r"^\s*(ETP_\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)",
                                  text, re.M):
        out[name] = int(value, 0)
    return out


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_motion_codes_match_python():
    """The two code tables are written out by hand in both languages.

    Nothing enforces that at build time, and a silent divergence would only
    show up as a garbled sheet of paper, so check it here.
    """
    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    expected = {
        "ERIKA_SPACE": ec.SPACE,
        "ERIKA_BACKSPACE": ec.BACKSPACE,
        "ERIKA_HALF_STEP_FWD": ec.HALF_STEP_FORWARD,
        "ERIKA_HALF_STEP_BACK": ec.HALF_STEP_BACK,
        "ERIKA_HALF_LINE_FWD": ec.HALF_LINE_FORWARD,
        "ERIKA_HALF_LINE_BACK": ec.HALF_LINE_BACK,
        "ERIKA_NEWLINE": ec.NEWLINE,
        "ERIKA_CARRIAGE_RETURN": ec.CARRIAGE_RETURN,
        "ERIKA_MICRO_LINE_FWD": ec.MICRO_LINE_FORWARD,
        "ERIKA_MICRO_LINE_BACK": ec.MICRO_LINE_BACK,
        "ERIKA_SET_STRIKE_FORCE": ec.SET_STRIKE_FORCE,
        # Unused by default -- IMG STEPS and IMG ACK turn them on -- but mirrored
        # by hand all the same, and a mirrored constant is a constant that drifts.
        "ERIKA_CARRIAGE_STEPS": ec.CARRIAGE_STEPS,
        "ERIKA_PLATEN_STEPS": ec.PLATEN_STEPS,
        "ERIKA_CARRIAGE_STEPS_PER_INCH": ec.CARRIAGE_STEPS_PER_INCH,
        "ERIKA_PLATEN_STEPS_PER_INCH": ec.PLATEN_STEPS_PER_INCH,
        "ERIKA_MAX_STEPS_PER_COMMAND": ec.MAX_STEPS_PER_COMMAND,
        "ERIKA_REPORT_WHEN_PRINTED": 0x96,
        # Rückwärtsdruck, which the planner reaches for on a serpentine's
        # reverse passes. The pair has to stay a pair on both sides: the firmware
        # sends the OFF again on every exit, and a drifted OFF would leave the
        # machine typing right to left with nothing to say why.
        "ERIKA_BACKWARD_PRINT_ON": ec.BACKWARD_PRINT_ON,
        "ERIKA_BACKWARD_PRINT_OFF": ec.BACKWARD_PRINT_OFF,
    }
    for name, value in expected.items():
        assert defines.get(name) == value, f"{name} differs from erika_codes.py"


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_and_python_agree_on_which_bytes_are_keys():
    """The bound both ends use to decide whether a byte may be struck.

    The firmware has no glyph table to consult -- only these two numbers -- so if
    they drift apart, one side accepts strikes the other refuses. Drifting *up*
    is the dangerous direction: the codes above the wheel include seven that take
    the byte after them as an operand, and one of those reaching the machine as a
    strike leaves the interpreter reading opcodes out of operands for the rest of
    the job, with every byte legal and the CRC intact.
    """
    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    assert defines.get("ERIKA_MIN_GLYPH_CODE") == ec.MIN_GLYPH_CODE
    assert defines.get("ERIKA_MAX_GLYPH_CODE") == ec.MAX_GLYPH_CODE


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_firmware_checks_a_strike_against_the_range_not_a_list():
    """The shape of the check, not its constants.

    A list of the codes we have named is not the guard: the published table runs
    to 0xAF and the pipeline has read only part of it, so anything unnamed above
    the wheel has to be refused too. Reverting this to isMotionCode() would put
    back exactly the hole it was written to close, and the constants above would
    still agree.
    """
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    match = re.search(r"case ETP_STRIKE:.*?case ETP_SET_FORCE:", text, re.S)
    assert match, "could not find the STRIKE case in fetchNext()"
    assert "isTypeableGlyph(operand)" in match.group(0), (
        "the STRIKE case no longer checks the operand against the wheel's range"
    )


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_opcodes_match_python():
    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    expected = {
        "ETP_END": etp.OP_END, "ETP_RIGHT": etp.OP_RIGHT, "ETP_LEFT": etp.OP_LEFT,
        "ETP_DOWN": etp.OP_DOWN, "ETP_UP": etp.OP_UP, "ETP_CR": etp.OP_CR,
        "ETP_STRIKE": etp.OP_STRIKE, "ETP_STRIKE_NA": etp.OP_STRIKE_NA,
        "ETP_DELAY": etp.OP_DELAY, "ETP_MICRO_DOWN": etp.OP_MICRO_DOWN,
        "ETP_MICRO_UP": etp.OP_MICRO_UP, "ETP_NEWLINE": etp.OP_NEWLINE,
        "ETP_SET_FORCE": etp.OP_SET_FORCE,
        "ETP_BACKWARD_ON": etp.OP_BACKWARD_ON,
        "ETP_BACKWARD_OFF": etp.OP_BACKWARD_OFF,
        "ETP_HEADER_SIZE": etp.HEADER_SIZE, "ETP_VERSION": etp.VERSION,
        "ETP_FLAG_PITCH12": etp.FLAG_PITCH12,
        "ETP_FLAG_HOME_EACH_ROW": etp.FLAG_HOME_EACH_ROW,
    }
    for name, value in expected.items():
        assert defines.get(name) == value, f"{name} differs from etp.py"


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_knows_which_opcodes_carry_an_operand():
    """The opcode table is not the only thing duplicated -- the *shape* is too.

    ``fetchNext()`` decides how many bytes to read from one hand-written list of
    opcodes that take an operand. An opcode added to etp.py but missing there
    puts the interpreter one byte out of step for the rest of the job: every
    later opcode is read from an operand and every operand from an opcode, so a
    picture becomes a stream of arbitrary motions. The device cannot notice --
    the bytes are all individually legal -- and the CRC passes, because the file
    is intact. Nothing but this comparison would catch it.
    """
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    match = re.search(r"bool needsOperand\s*=\s*\((.*?)\);", text, re.S)
    assert match, "could not find the needsOperand expression in fetchNext()"
    firmware = set(re.findall(r"op == (ETP_\w+)", match.group(1)))
    ours = {f"ETP_{etp.OPCODE_NAMES[op]}" for op in etp._HAS_OPERAND}
    assert firmware == ours, (
        "opcodes that take an operand differ between etp.py and fetchNext(): "
        f"only in Python {sorted(ours - firmware)}, "
        f"only in firmware {sorted(firmware - ours)}"
    )


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_has_a_case_for_every_opcode():
    """An opcode the firmware does not decode fails the job -- loudly, which is
    the right outcome for old firmware meeting a new job, and the wrong one for
    firmware that was simply never updated. Both look the same on the device, so
    the difference has to be caught here."""
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    handled = set(re.findall(r"case (ETP_\w+):", text))
    ours = {f"ETP_{name}" for name in etp.OPCODE_NAMES.values()}
    assert not ours - handled, f"fetchNext() has no case for {sorted(ours - handled)}"


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_upload_chunk_size_matches_the_host_tool():
    from erika import send

    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "image_receiver.h"))
    assert defines["IMG_UPLOAD_CHUNK"] == send.CHUNK_SIZE


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_a_full_data_line_fits_the_firmware_line_buffer():
    """A chunk must base64-encode to something the device can hold.

    The device drops overlong lines. If CHUNK_SIZE ever outgrows IMG_MAX_LINE
    every upload would fail, so pin the relationship here rather than finding
    out on a board.
    """
    import base64

    from erika import send

    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "image_receiver.h"))
    longest = len("D ") + len(base64.b64encode(b"\x00" * send.CHUNK_SIZE))
    assert longest < defines["IMG_MAX_LINE"], (
        f"a full data line is {longest} chars but IMG_MAX_LINE is "
        f"{defines['IMG_MAX_LINE']}"
    )


# ---------------------------------------------------------------------------
# upload protocol
# ---------------------------------------------------------------------------


class FakeDevice:
    """The firmware's line protocol, as a Python stand-in.

    Mirrors ImageReceiver::handleLine() closely enough to exercise the host's
    framing and ACK accounting: base64 decode, running total, and the
    UPLOAD-READY / ACK / OK replies -- and the two lines a resync sends, whose
    firmware answers are the whole point of test_resync_survives_a_board_in_setup.
    """

    def __init__(self, chunk=None):
        from erika import send

        self.chunk = chunk or send.CHUNK_SIZE
        self.received = bytearray()
        self.total = 0
        self.active = False
        self.sent_lines: list[str] = []

    def line(self, text: str) -> str:
        import base64

        head, _, rest = text.partition(" ")
        if head == "IMG":
            sub, _, arg = rest.partition(" ")
            if sub.upper() == "UPLOAD":
                self.active = True
                self.total = int(arg)
                self.received = bytearray()
                return f"UPLOAD-READY {self.chunk}"
            if sub.upper() == "PING":
                return f"PONG chunk={self.chunk}"
            if sub.upper() == "CANCEL":
                # Cancelling nothing is an error, not a no-op. A resync sends
                # this first, so that ERR is the normal answer to the very first
                # line the host ever sends.
                if not self.active:
                    return "ERR no upload in progress"
                self.active = False
                return "OK upload cancelled"
            if sub.upper() == "STATUS":
                # Answered in every state, which is what makes it a usable
                # sentinel for resync.
                return "STATE READY" if self.received else "STATE IDLE"
            return "ERR unknown"
        if head == "D" and self.active:
            data = base64.b64decode(rest)
            if len(data) > self.chunk:
                return "ERR too much data"
            self.received += data
            return f"ACK {len(self.received)}"
        if head == "Z" and self.active:
            self.active = False
            if len(self.received) != self.total:
                return f"ERR short upload: {len(self.received)}/{self.total}"
            return f"OK {len(self.received)} bytes stored and verified"
        return "ERR not an IMG line"


class FakeLink:
    """Just enough of Link for upload() to run against a FakeDevice."""

    def __init__(self, device: FakeDevice, noise: bool = False):
        self.device = device
        self.noise = noise
        self._pending: list[str] = []

    def send_line(self, text: str) -> None:
        self.device.sent_lines.append(text)
        if self.noise:
            # The firmware logs to the same stream; replies must survive it.
            self._pending.append("[BT] Client connected")
        self._pending.append(self.device.line(text))

    def await_reply(self, *prefixes, timeout=30.0) -> str:
        wanted = prefixes or ("OK", "ERR", "ACK", "UPLOAD-READY")
        while self._pending:
            line = self._pending.pop(0)
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if any(line.startswith(p) for p in wanted):
                return line
        raise TimeoutError(f"no reply matching {wanted}")

    def drain(self, seconds: float = 0.4) -> list[str]:
        out, self._pending = self._pending, []
        return out

    def resync(self) -> None:
        self._pending = []


class SleepySerial:
    """A port whose board is still in setup(): it takes lines in and answers
    nothing, then answers all of them at once when loop() finally runs.

    That is the real timeline -- Serial.setRxBufferSize() buys 4 kB before
    Serial.begin(), WiFiConfig::begin() then waits up to 15s for association, and
    poll() drains the whole buffer in one pass afterwards. Modelled on the write
    count rather than on a clock, so the test does not sleep: the wake-up is the
    moment the `wake_after`th line is taken in.
    """

    def __init__(self, device: FakeDevice, wake_after: int = 2):
        self.device = device
        self.wake_after = wake_after
        self.port = None
        self.baudrate = None
        self.timeout = None
        self.write_timeout = None
        self.dtr = True
        self.rts = True
        self.is_open = False
        self._held: list[str] = []
        self._wire: list[str] = []

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def write(self, raw: bytes) -> int:
        text = raw.decode().strip()
        self.device.sent_lines.append(text)
        reply = self.device.line(text)
        (self._held if self.wake_after else self._wire).append(reply)
        if self.wake_after:
            self.wake_after -= 1
            if not self.wake_after:
                self._wire += self._held
                self._held = []
        return len(raw)

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        # b"" is how pyserial reports the read timeout Link.read_line raises on.
        return (self._wire.pop(0) + "\r\n").encode() if self._wire else b""

    def reset_input_buffer(self) -> None:
        # Only what has reached the host: the replies still inside the device are
        # beyond reach, which is exactly why draining could not clear them.
        self._wire = []


def test_resync_survives_a_board_still_in_setup(monkeypatch):
    """The resync's own reply must not become the next command's answer.

    IMG CANCEL is answered "ERR no upload in progress" whenever nothing is in
    flight, and a board still in setup() answers it only after the next line has
    gone out -- so draining until silence cleared nothing, and the first real
    command read that ERR as its own. Connecting to a healthy machine failed with
    the firmware's correct answer to a question the host had asked itself.
    """
    import types

    from erika import send

    device = FakeDevice()
    port = SleepySerial(device)
    monkeypatch.setattr(send, "_require_serial", lambda: types.SimpleNamespace(
        Serial=lambda: port))

    link = send.Link("/dev/fake", reset=False)
    link.resync()

    assert link.command("IMG PING", echo=False).startswith("PONG")
    # The sentinel, not a silence window: everything ahead of the STATE is stale.
    assert device.sent_lines == ["IMG CANCEL", "IMG STATUS", "IMG PING"]


@pytest.mark.parametrize("noise", [False, True])
def test_upload_delivers_the_file_byte_for_byte(tmp_path, charset, noise, capsys):
    from erika import send

    job = planner.encode(
        planner.build_plan(
            _write_choices(tmp_path, _random_choices(charset, 5, 9, FOUR_LAYERS, seed=2)),
            charset,
        )
    )
    path = os.path.join(tmp_path, "job.etp")
    etp.save(path, job)
    original = open(path, "rb").read()

    device = FakeDevice()
    send.upload(FakeLink(device, noise=noise), path, progress=False)

    assert bytes(device.received) == original
    # And the reassembled bytes are a valid job, not just the right length.
    assert etp.unpack(bytes(device.received)).strikes == job.strikes


class DropsNthLine(FakeDevice):
    """A device whose receive buffer overruns on one particular data line.

    This is the failure that actually happened on hardware: the stock 256-byte
    serial buffer could not hold a whole data line, so a chunk went missing at
    an unpredictable point in the transfer.
    """

    def __init__(self, drop_at: int, drops: int = 1, **kw):
        super().__init__(**kw)
        self.drop_at = drop_at
        self.drops_left = drops
        self.seen = 0

    def line(self, text: str) -> str:
        if text.startswith("IMG UPLOAD"):
            self.seen = 0  # line numbering is per attempt
        if text.startswith("D "):
            self.seen += 1
            if self.seen == self.drop_at and self.drops_left > 0:
                self.drops_left -= 1
                # Dropped on the wire: the device answers with its unchanged
                # total, which is how the host notices.
                return f"ACK {len(self.received)}"
        return super().line(text)


def test_upload_detects_a_dropped_data_line(tmp_path, charset):
    from erika import send

    path = os.path.join(tmp_path, "job.etp")
    etp.save(path, _tiny_job())

    with pytest.raises(RuntimeError, match="data line was lost"):
        send.upload(FakeLink(DropsNthLine(drop_at=1)), path, progress=False,
                    retries=0)


def test_upload_retries_a_transfer_that_drops_a_line(tmp_path, charset, capsys):
    """One bad line should cost a retry, not the whole job."""
    from erika import send

    job = planner.encode(
        planner.build_plan(
            _write_choices(tmp_path, _random_choices(charset, 6, 10, FOUR_LAYERS, seed=8)),
            charset,
        )
    )
    path = os.path.join(tmp_path, "job.etp")
    etp.save(path, job)
    original = open(path, "rb").read()

    device = DropsNthLine(drop_at=3, drops=1)
    send.upload(FakeLink(device), path, progress=False, retries=2)

    assert bytes(device.received) == original
    assert "retrying" in capsys.readouterr().out


def test_upload_gives_up_after_the_retry_budget(tmp_path):
    from erika import send

    path = os.path.join(tmp_path, "job.etp")
    etp.save(path, _tiny_job())

    # Drops the first line of every attempt.
    device = DropsNthLine(drop_at=1, drops=99)
    with pytest.raises(RuntimeError, match="data line was lost"):
        send.upload(FakeLink(device), path, progress=False, retries=2)


def test_upload_frames_the_conversation_correctly(tmp_path, charset):
    import base64

    from erika import send

    # Big enough to need several data lines.
    job = planner.encode(
        planner.build_plan(
            _write_choices(tmp_path, _random_choices(charset, 8, 12, FOUR_LAYERS, seed=4)),
            charset,
        )
    )
    path = os.path.join(tmp_path, "job.etp")
    size = etp.save(path, job)

    device = FakeDevice()
    send.upload(FakeLink(device), path, progress=False)

    lines = device.sent_lines
    assert lines[0] == f"IMG UPLOAD {size}"
    assert lines[-1] == "Z"

    data_lines = [l for l in lines[1:-1] if l.startswith("D ")]
    assert len(data_lines) == len(lines) - 2, "unexpected traffic between the frames"
    assert len(data_lines) > 1, "test job is too small to exercise chunking"
    for line in data_lines:
        assert len(base64.b64decode(line[2:])) <= device.chunk


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


def _tiny_job() -> etp.Job:
    enc = etp.Encoder()
    enc.right(5)
    enc.strike(ec.glyph_for_char("A").code)
    enc.newline(1)
    enc.end()
    return etp.Job(body=enc.body(), cols=8, rows=2, strikes=enc.strikes)


def test_pack_unpack_round_trip():
    job = _tiny_job()
    back = etp.unpack(etp.pack(job))
    assert back.body == job.body
    assert (back.cols, back.rows, back.strikes) == (job.cols, job.rows, job.strikes)
    assert back.pitch == 10 and back.home_each_row is True


def test_flags_survive_the_round_trip():
    job = _tiny_job()
    job.pitch = 12
    job.home_each_row = False
    back = etp.unpack(etp.pack(job))
    assert back.pitch == 12 and back.home_each_row is False


def test_corrupt_body_is_rejected():
    data = bytearray(etp.pack(_tiny_job()))
    data[etp.HEADER_SIZE] ^= 0xFF
    with pytest.raises(etp.EtpError, match="CRC"):
        etp.unpack(bytes(data))


def test_truncated_file_is_rejected():
    data = etp.pack(_tiny_job())
    with pytest.raises(etp.EtpError, match="truncated"):
        etp.unpack(data[:-2])
    with pytest.raises(etp.EtpError, match="at least"):
        etp.unpack(data[:10])


def test_bad_magic_is_rejected():
    data = b"NOPE" + etp.pack(_tiny_job())[4:]
    with pytest.raises(etp.EtpError, match="magic"):
        etp.unpack(data)


def test_encoder_splits_runs_longer_than_one_operand():
    enc = etp.Encoder()
    enc.right(600)
    ops = list(etp.iter_ops(enc.body()))
    assert [operand for _, _, operand in ops] == [255, 255, 90]
    assert all(op == etp.OP_RIGHT for _, op, _ in ops)


def test_encoder_rejects_negative_moves():
    with pytest.raises(etp.EtpError):
        etp.Encoder().right(-1)


def test_motion_codes_cannot_be_struck_as_glyphs():
    """Both ends refuse this; a struck SPACE would shift the rest of the row."""
    for code in (ec.SPACE, ec.BACKSPACE, ec.CARRIAGE_RETURN, ec.HALF_STEP_FORWARD):
        with pytest.raises(etp.EtpError, match="motion"):
            etp.Encoder().strike(code)


def test_a_command_that_eats_the_next_byte_cannot_be_struck_either():
    """The worse half of the same hazard, and the one the old guard let through.

    A motion code struck as a glyph moves the head: wrong, and visible on the
    sheet. A code that carries an operand swallows the byte after it, so from
    there the firmware reads every opcode out of an operand -- every byte still
    legal, the CRC still passing over an intact file, and nothing to say why the
    picture became noise.
    """
    assert ec.OPERAND_CODES  # the guard is worth nothing if this is empty
    for code in sorted(ec.OPERAND_CODES):
        with pytest.raises(etp.EtpError, match="one byte out of step"):
            etp.Encoder().strike(code)


def test_nothing_above_the_wheels_range_can_be_struck():
    """The guard is the range, not a list of the codes that have been named.

    An unnamed byte above the wheel is not known to be harmless -- the published
    table runs to 0xAF and this pipeline has read only part of it -- so the
    default has to be refusal.
    """
    for code in range(ec.MAX_GLYPH_CODE + 1, 0x100):
        with pytest.raises(etp.EtpError):
            etp.Encoder().strike(code)
    with pytest.raises(etp.EtpError):
        etp.Encoder().strike(0x00)


def test_every_key_on_the_wheel_is_inside_the_range_the_guards_use():
    """Which is what makes a bound safe to use in place of a lookup -- on the
    firmware side there is no glyph table to consult, only these two numbers."""
    for g in ec.GLYPHS + ec.DEAD_KEY_GLYPHS:
        assert ec.is_glyph_code(g.code), f"{g.char!r} is outside the range"
    # And the bounds are tight: something really does sit on each of them.
    codes = {g.code for g in ec.GLYPHS + ec.DEAD_KEY_GLYPHS}
    assert ec.MIN_GLYPH_CODE in codes
    assert ec.MAX_GLYPH_CODE in codes


def test_iter_ops_rejects_unknown_opcodes():
    with pytest.raises(etp.EtpError, match="unknown opcode"):
        list(etp.iter_ops(b"\xEE"))


def test_header_is_the_documented_size():
    assert struct.calcsize(etp.HEADER_STRUCT) == etp.HEADER_SIZE == 24


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def charset() -> Charset:
    return Charset.load("sigma-10", SRC)


@pytest.mark.parametrize("name", ["sigma-10", "sigma-12"])
def test_charset_file_is_utf8(name):
    """A charset written in the platform's default encoding is unloadable.

    The glyph names contain non-ASCII characters, and Python's default text
    encoding is the locale's -- cp1252 on Windows. A glyphs.json generated
    without an explicit encoding= is readable only on the machine that made
    it, and fails on the first non-ASCII character everywhere else.
    """
    path = os.path.join(SRC, "charsets", name, "glyphs.json")
    text = open(path, "rb").read().decode("utf-8")  # strict: raises if not UTF-8
    assert any(ord(c) > 127 for c in text), "expected non-ASCII glyphs to guard"


def _write_choices(tmp_path, layers: dict) -> str:
    path = os.path.join(tmp_path, "choices.json")
    with open(path, "w") as f:
        json.dump(layers, f)
    return path


def test_quarter_cell_offsets_are_rejected_when_only_keystrokes_are_allowed(
    tmp_path, charset
):
    """They are typeable by default now -- the machine's motor steps reach them.
    What is still refused is a quarter-cell offset when the plan has been told to
    stay on the keyboard."""
    path = _write_choices(tmp_path, {"layer0_0.25_0": [[1, 2]]})
    with pytest.raises(PlanError, match="not typeable by keystroke"):
        planner.build_plan(path, charset, fine=False)
    assert planner.build_plan(path, charset).strikes  # and accepted by default


def test_image_wider_than_the_carriage_is_rejected(tmp_path, charset):
    wide = [[1] * (charset.max_columns + 5)]
    path = _write_choices(tmp_path, {"layer0_0_0": wide})
    with pytest.raises(PlanError, match="carriage only reaches"):
        planner.build_plan(path, charset)


def test_out_of_range_character_index_is_rejected(tmp_path, charset):
    path = _write_choices(tmp_path, {"layer0_0_0": [[len(charset) + 1]]})
    with pytest.raises(PlanError, match="only has"):
        planner.build_plan(path, charset)


def test_blank_cells_are_not_typed(tmp_path, charset):
    path = _write_choices(tmp_path, {"layer0_0_0": [[0, 0, 5, 0]]})
    plan = planner.build_plan(path, charset)
    assert [s.x for s in plan.strikes] == [4]


def test_layer_offsets_become_half_cell_positions(tmp_path, charset):
    path = _write_choices(
        tmp_path,
        {
            "layer0_0_0": [[7]],
            "layer1_0_0.5": [[7]],
            "layer2_0.5_0": [[7]],
            "layer3_0.5_0.5": [[7]],
        },
    )
    plan = planner.build_plan(path, charset)
    assert {(s.y, s.x) for s in plan.strikes} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_paper_only_ever_feeds_forward(tmp_path, charset):
    rng = np.random.default_rng(7)
    grid = rng.integers(0, len(charset), size=(6, 9)).tolist()
    path = _write_choices(tmp_path, {"layer0_0_0": grid, "layer1_0.5_0.5": grid})
    plan = planner.build_plan(path, charset)
    ys = [s.y for s in plan.strikes]
    assert ys == sorted(ys)


def test_serpentine_alternates_sweep_direction(tmp_path, charset):
    grid = [[3, 3, 3, 3], [3, 3, 3, 3]]
    path = _write_choices(tmp_path, {"layer0_0_0": grid})
    plan = planner.build_plan(path, charset, home_each_row=False, boustrophedon=True)
    rows = {}
    for s in plan.strikes:
        rows.setdefault(s.y, []).append(s.x)
    first, second = rows[0], rows[2]
    assert first == sorted(first)
    assert second == sorted(second, reverse=True)


def test_encode_never_emits_an_upward_feed(tmp_path, charset):
    rng = np.random.default_rng(11)
    grid = rng.integers(0, len(charset), size=(5, 8)).tolist()
    path = _write_choices(tmp_path, {"layer0_0_0": grid, "layer1_0.5_0": grid})
    job = planner.encode(planner.build_plan(path, charset))
    assert not any(op == etp.OP_UP for _, op, _ in etp.iter_ops(job.body))


def test_whole_line_gaps_use_the_line_feed_mechanism(tmp_path, charset):
    # No vertical layer offsets, so every gap is a whole line: the planner
    # should reach for NEWLINE rather than stacking two half-line steps.
    path = _write_choices(tmp_path, {"layer0_0_0": [[4], [4], [4]]})
    job = planner.encode(planner.build_plan(path, charset))
    ops = [op for _, op, _ in etp.iter_ops(job.body)]
    assert etp.OP_NEWLINE in ops
    assert etp.OP_DOWN not in ops


# ---------------------------------------------------------------------------
# the whole chain, through a virtual typewriter
# ---------------------------------------------------------------------------


def _random_choices(charset, rows, cols, layers, seed=0, density=0.6):
    """A synthetic optimizer result: random glyphs, some cells left blank."""
    rng = np.random.default_rng(seed)
    out = {}
    for layer, (ov, oh) in enumerate(layers):
        grid = rng.integers(1, len(charset), size=(rows, cols))
        grid = np.where(rng.random((rows, cols)) < density, grid, 0)
        out[f"layer{layer}_{ov}_{oh}"] = grid.tolist()
    return out


FOUR_LAYERS = [(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)]


@pytest.mark.parametrize("home_each_row", [True, False])
@pytest.mark.parametrize("layers", [[(0, 0)], FOUR_LAYERS])
def test_virtual_typewriter_lands_every_strike_where_the_plan_said(
    tmp_path, charset, home_each_row, layers
):
    """Plan -> opcodes -> raw bytes -> carriage motion -> marks on paper.

    If the half-step expansion, the position bookkeeping or the opcode
    encoding is off by anything, the recovered positions drift from the
    planned ones and this catches it.
    """
    path = _write_choices(tmp_path, _random_choices(charset, 7, 11, layers, seed=3))
    plan = planner.build_plan(path, charset, home_each_row=home_each_row)
    job = planner.encode(plan)

    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert machine.overruns == 0

    recovered = emulate.impressions_to_strikes(machine, charset)
    assert len(recovered) == len(plan.strikes) == job.strikes
    for got, want in zip(recovered, plan.strikes):
        assert (got.y, got.x) == (want.y, want.x)
        assert charset.codes[got.index] == charset.codes[want.index]


def test_virtual_typewriter_reproduces_the_planned_image(tmp_path, charset):
    """The marks the machine leaves must composite to the planned picture."""
    from utils import prep_charset

    tiles, _, _ = prep_charset("sigma-10", SRC)
    path = _write_choices(tmp_path, _random_choices(charset, 6, 10, FOUR_LAYERS, seed=5))
    plan = planner.build_plan(path, charset)
    job = planner.encode(plan)

    machine = emulate.type_job(job, max_columns=charset.max_columns)
    typed = planner.Plan(
        strikes=emulate.impressions_to_strikes(machine, charset),
        cols=plan.cols,
        rows=plan.rows,
        charset=charset,
        layer_offsets=plan.layer_offsets,
    )

    intended = preview.render(plan, tiles)
    actual = preview.render(typed, tiles)
    assert np.array_equal(intended, actual)


# ---------------------------------------------------------------------------
# the indent: where on the paper the print goes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("home_each_row", [True, False])
def test_an_indent_moves_the_head_and_leaves_the_picture_alone(
    tmp_path, charset, home_each_row
):
    """The one invariant the feature rests on.

    An indent is a decision about where on the sheet the print sits, and the
    machine is the only thing that should be able to tell: the strikes, the grid
    and the rendered picture have to come out identical, because that render is
    what the plan is verified against. Get this wrong -- by folding the indent
    into the strike positions -- and every indented job reports a mismatch
    against optimize.py's mockup while being perfectly correct.
    """
    from utils import prep_charset

    tiles, _, _ = prep_charset("sigma-10", SRC)
    path = _write_choices(tmp_path, _random_choices(charset, 6, 10, FOUR_LAYERS, seed=11))
    plain = planner.build_plan(path, charset, home_each_row=home_each_row)
    shifted = planner.build_plan(path, charset, home_each_row=home_each_row, indent=12)

    assert [(s.y, s.x, s.index) for s in shifted.strikes] == [
        (s.y, s.x, s.index) for s in plain.strikes
    ]
    assert (shifted.cols, shifted.rows) == (plain.cols, plain.rows)
    assert np.array_equal(preview.render(plain, tiles), preview.render(shifted, tiles))

    # ...and the machine does put it 12 columns further right, which is the half
    # of it no render can show. Through the virtual typewriter rather than by
    # reading the opcodes, because what matters is where the head ends up after
    # every carriage return, backspace and backward run in the stream.
    job = planner.encode(shifted)
    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert machine.overruns == 0
    recovered = emulate.impressions_to_strikes(machine, charset)
    assert len(recovered) == len(shifted.strikes)
    for got, want in zip(recovered, shifted.strikes):
        assert (got.y, got.x) == (want.y, want.x + 2 * 12)


def test_an_indented_backward_run_still_lands_where_planned(tmp_path, charset):
    """The right-to-left path keeps the head's position itself, so it needs it too.

    Rückwärtsdruck moves the carriage and strikes on one byte, so inside a run the
    encoder emits no moves and updates its own idea of where the head is -- in two
    places. Either of them left holding a position in the picture rather than one
    on the paper puts every move *after* the run out by the whole indent, and
    nothing before it, which is a fault no preview can show: the plan and the
    mockup still agree, because neither of them knows about the indent at all.
    """
    grid = [[5] * 12 for _ in range(3)]
    path = _write_choices(tmp_path, {"layer0_0_0": grid})
    plan = planner.build_plan(path, charset, home_each_row=False, indent=9)
    job = planner.encode(plan)
    assert any(op == etp.OP_BACKWARD_ON for _, op, _ in etp.iter_ops(job.body))

    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert machine.overruns == 0
    recovered = emulate.impressions_to_strikes(machine, charset)
    assert len(recovered) == len(plan.strikes)
    for got, want in zip(recovered, plan.strikes):
        assert (got.y, got.x) == (want.y, want.x + 2 * 9)


def test_an_indent_costs_the_carriage_the_columns_it_uses(tmp_path, charset):
    """A print that fits and an indent that fits can still not fit together.

    Nothing downstream would notice: the plan verifies against the mockup, the
    job uploads, and the machine types into its right-hand stop with every
    character past it landing in the same column. So it is refused here, and the
    refusal has to name both ways out -- narrowing the print is not obviously the
    right one when the indent is the thing being asked for.
    """
    room = charset.max_columns - 4
    path = _write_choices(tmp_path, {"layer0_0_0": [[1] * room]})
    assert planner.build_plan(path, charset, indent=4).strikes  # exactly fits

    with pytest.raises(PlanError) as raised:
        planner.build_plan(path, charset, indent=5)
    message = str(raised.value)
    assert "Indent it by 4 or less" in message
    # And the row length it suggests instead really does fit at that indent.
    suggested = int(re.search(r"-r (\d+)", message).group(1))
    fits = _write_choices(tmp_path, {"layer0_0_0": [[1] * (suggested + 1)]})
    assert planner.build_plan(fits, charset, indent=5).strikes


def test_a_negative_indent_is_refused(tmp_path, charset):
    """There is nothing to the left of the left margin -- it is where CR goes."""
    path = _write_choices(tmp_path, {"layer0_0_0": [[1, 2, 3]]})
    with pytest.raises(PlanError, match="negative"):
        planner.build_plan(path, charset, indent=-1)


def test_an_indented_summary_says_which_columns_the_print_lands_in(tmp_path, charset):
    path = _write_choices(tmp_path, {"layer0_0_0": [[1] * 10]})
    plan = planner.build_plan(path, charset, indent=7)
    text = planner.summarize(plan, planner.encode(plan))
    assert "7 blank columns" in text
    assert "columns 8 to 17" in text
    # Nothing about an indent in a job that has none, so an ordinary summary
    # reads exactly as it did.
    plain = planner.build_plan(path, charset)
    assert "indent" not in planner.summarize(plain, planner.encode(plain))


def test_dead_keys_do_not_advance_the_carriage(tmp_path):
    """A dead key must leave the head where it was, or everything after shifts."""
    dead_charset = Charset(
        name="test", pitch=10, cell_w=24, cell_h=40, max_columns=65,
        codes=[ec.SPACE, ec.DEAD_KEY_GLYPHS[0].code, ec.glyph_for_char("A").code],
        advances=[True, False, True],
        chars=[" ", ec.DEAD_KEY_GLYPHS[0].char, "A"],
    )
    path = _write_choices(tmp_path, {"layer0_0_0": [[1, 2]]})
    plan = planner.build_plan(path, dead_charset)
    job = planner.encode(plan)

    machine = emulate.type_job(job)
    positions = [(i.y, i.x) for i in machine.impressions]
    assert positions == [(0, 0), (0, 2)]

    # ...and the encoder used the non-advancing opcode for it.
    ops = [op for _, op, _ in etp.iter_ops(job.body)]
    assert etp.OP_STRIKE_NA in ops


def _calibration_lines(tmp_path) -> dict[int, list]:
    """Type the calibration job on the virtual machine, grouped by line."""
    from erika import pipeline

    out = os.path.join(tmp_path, "cal.etp")
    pipeline.main(["calibrate", "-o", out])
    machine = emulate.type_job(etp.load(out))
    assert machine.overruns == 0, "calibration ran the carriage off the end"
    lines: dict[int, list] = {}
    for imp in machine.impressions:
        lines.setdefault(imp.y, []).append(imp)
    return lines


def test_calibration_half_step_test_prints_pairs_of_bars(tmp_path, capsys):
    """Part 2 must strike the ruler twice, offset by exactly a half step.

    If the two passes coincided, the sheet would look like part 1 and the
    operator would read a working machine as a broken one.
    """
    from erika.pipeline import RULER_CHAR

    code = ec.glyph_for_char(RULER_CHAR).code
    lines = _calibration_lines(tmp_path)
    # The two ruler lines are the ones made only of the ruler glyph.
    rulers = [
        sorted(i.x for i in marks)
        for _, marks in sorted(lines.items())
        if len(marks) > 1 and {i.code for i in marks} == {code}
    ]
    assert len(rulers) == 2, "expected a plain ruler and an offset one"
    plain, offset = rulers

    assert len(offset) == 2 * len(plain), "second pass should double the bars"
    # Half-steps land on odd x, whole steps on even. Pairs, not overstrikes.
    assert {x % 2 for x in offset} == {0, 1}
    assert sorted({b - a for a, b in zip(offset, offset[1:])}) == [1, 3]


def test_calibration_line_feed_test_stays_in_one_column(tmp_path, capsys):
    """Part 3 checks that two half-line feeds equal one whole line.

    It reads as a pitch comparison only if every mark shares a column -- an
    earlier version staggered alternate marks to stop them overlapping, which
    made a correct machine look like it was drifting sideways.
    """
    lines = _calibration_lines(tmp_path)
    # The ladder is the run of lines carrying a single mark each.
    ladder = [marks[0] for _, marks in sorted(lines.items()) if len(marks) == 1]
    assert len(ladder) >= 8, "ladder not found"
    assert len({m.x for m in ladder}) == 1, "marks must share one column"

    gaps = [b.y - a.y for a, b in zip(ladder, ladder[1:])]
    whole = [g for g in gaps if g == 2]  # two half-lines == one whole line
    half = [g for g in gaps if g == 1]
    assert len(whole) + len(half) == len(gaps), f"unexpected gaps: {gaps}"
    assert whole and half, "need both a reference group and a half-line group"
    assert gaps == whole + half, "reference group should come first"


def test_calibration_glyph_check_types_the_whole_charset(tmp_path, capsys):
    """Part 6 is the only check that the type wheel matches the code table.

    It has to be complete and verbatim: the operator compares the paper
    against the rows the tool prints, so a glyph missing from either side
    would go unnoticed -- and the optimizer chooses characters by how much ink
    they lay down, so a mismatched wheel corrupts every tone in the picture.
    """
    from erika.pipeline import glyph_check_rows

    lines = _calibration_lines(tmp_path)
    typed = {
        "".join(ec.glyph_for_code(i.code).char for i in marks)
        for _, marks in lines.items()
    }
    rows = glyph_check_rows()
    for row in rows:
        assert row in typed, f"glyph-check row not typed: {row!r}"
    assert "".join(rows) == "".join(g.char for g in ec.GLYPHS), "charset not covered"


def _area_cells(tmp_path, *extra: str) -> set[tuple[int, int]]:
    """Type the print-area job on the virtual machine; return the marked cells.

    Cells are (row, column) in whole cells -- the sheet only ever marks whole
    ones, so half-step coordinates would just be noise here.
    """
    from erika import pipeline

    out = os.path.join(tmp_path, "area.etp")
    assert pipeline.main(["area", "-o", out, *extra]) == 0
    job = etp.load(out)
    machine = emulate.type_job(job, max_columns=ec.MAX_COLUMNS[job.pitch])
    assert machine.overruns == 0, "print-area sheet ran the carriage off the end"
    assert not [op for _, op, _ in etp.iter_ops(job.body) if op == etp.OP_UP], (
        "the sheet must not reverse the platen"
    )
    assert all(imp.x % 2 == imp.y % 2 == 0 for imp in machine.impressions)
    return {(imp.y // 2, imp.x // 2) for imp in machine.impressions}


def _assert_corners(cells: set[tuple[int, int]], columns: int, rows: int) -> None:
    """The four corner cells are marked and nothing lies outside them."""
    corners = {(0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1)}
    assert corners <= cells, f"corners not marked: {sorted(corners - cells)}"
    assert max(c for _, c in cells) == columns - 1
    assert max(r for r, _ in cells) == rows - 1


@pytest.mark.parametrize("pitch", [10, 12])
def test_print_area_brackets_reach_the_carriage_limit(tmp_path, pitch):
    """The corners must sit on the extremes, or the sheet lies about the area.

    The width is the one hard limit the machine has, and this sheet is what an
    operator measures it against -- a bracket one column short would quietly
    shrink every print planned from it.
    """
    from erika.pipeline import DEFAULT_AREA_ROWS

    cells = _area_cells(tmp_path, "-p", str(pitch))
    _assert_corners(cells, ec.MAX_COLUMNS[pitch], DEFAULT_AREA_ROWS)


def test_print_area_marks_the_area_asked_for(tmp_path):
    """--columns / --rows exist to preview where a given -r would land."""
    _assert_corners(_area_cells(tmp_path, "--columns", "20", "--rows", "6"), 20, 6)


def test_print_area_refuses_a_width_the_carriage_cannot_reach(tmp_path):
    """Better to say so than to type a sheet that overstates the machine."""
    from erika import pipeline

    out = os.path.join(tmp_path, "area.etp")
    assert pipeline.main(["area", "-o", out, "--columns", "999"]) == 2
    assert not os.path.exists(out)


def test_expand_covers_every_opcode_the_encoder_can_emit():
    enc = etp.Encoder()
    enc.right(3)
    enc.left(3)
    enc.down(3)
    enc.up(3)
    enc.carriage_return()
    enc.newline(2)
    enc.micro_down(2)
    enc.micro_up(2)
    enc.delay_ms(30)
    enc.strike(ec.glyph_for_char("A").code)
    enc.strike(ec.DEAD_KEY_GLYPHS[0].code, advances=False)
    enc.backward_on()
    enc.strike(ec.glyph_for_char("B").code)
    enc.backward_off()
    enc.end()
    raw = emulate.expand(enc.body())
    assert raw  # no EmulationError, and something came out

    # An odd half-step run must end with the half-step key, not start with it.
    enc2 = etp.Encoder()
    enc2.right(5)
    assert emulate.expand(enc2.body()) == [ec.SPACE, ec.SPACE, ec.HALF_STEP_FORWARD]


def test_resume_by_pass_replays_paper_feeds_only(tmp_path, charset):
    """Skipping to a pass must leave the platen on the right line.

    Mirrors ErikaImagePrinter::fetchNext()'s skip branch: paper motion is
    replayed so the sheet ends up where it would have been, while strikes and
    carriage travel are suppressed so nothing is retyped.

    Passes are counted by carriage returns, and the planner emits exactly one
    per printed row -- so pass N is the N-th row, 1-based, which is the same
    number IMG STATUS reports back.
    """
    path = _write_choices(tmp_path, _random_choices(charset, 6, 8, FOUR_LAYERS, seed=9))
    plan = planner.build_plan(path, charset)
    job = planner.encode(plan)

    skip_to = 4
    machine = emulate.Typewriter(max_columns=charset.max_columns)
    passes = 0
    printed_anything = False
    for _, op, operand in etp.iter_ops(job.body):
        if op == etp.OP_END:
            break
        if op in (etp.OP_CR, etp.OP_NEWLINE):
            passes += 1
        skipping = passes < skip_to
        if skipping and op in (etp.OP_STRIKE, etp.OP_STRIKE_NA,
                               etp.OP_RIGHT, etp.OP_LEFT, etp.OP_DELAY):
            continue
        if not skipping and op in (etp.OP_STRIKE, etp.OP_STRIKE_NA):
            printed_anything = True
            break  # stop at the first character actually typed
        raw = bytes([op]) if operand is None else bytes([op, operand])
        machine.run(emulate.expand(raw))

    assert printed_anything, "skipping never ended"
    # Nothing was struck while fast-forwarding...
    assert machine.impressions == []
    # ...but the paper sits exactly where the resumed row expects it.
    rows = sorted({s.y for s in plan.strikes})
    assert machine.y == rows[skip_to - 1]


# ---------------------------------------------------------------------------
# the optimizer's per-layer background (fork fix in optimize.py)
# ---------------------------------------------------------------------------


def _record_backgrounds(layers_name: str, monkeypatch) -> list[np.ndarray]:
    """Run ``kword`` with its inner loop stubbed, keeping every background it built.

    Stubbed because the real pass is numba-jitted and compiling it costs the better
    part of ten seconds, which nothing here needs. Replacing the name inside
    ``optimize`` is what takes effect -- it binds it at import, so patching ``utils``
    would not.

    ``init_mode="random"`` is load-bearing: from a blank start every layer is bare
    paper, and a background that wrongly included the layer being optimized would be
    indistinguishable from a correct one.
    """
    import optimize  # noqa: PLC0415

    seen: list[np.ndarray] = []

    def stub(bg, mockup, target, chars, choices, layer_offset, **kwargs):
        seen.append(np.array(bg, copy=True))
        # A comparison count of zero would be a fair description of a stub, but
        # ``kword``'s timing report divides by it; and a zero error makes matplotlib
        # complain about identical y limits. Both are in the cosmetic tail, and both
        # are cheaper to satisfy than to work around.
        return choices, mockup, 1, 1.0

    monkeypatch.setattr(optimize, "layer_optimization_pass", stub)
    monkeypatch.chdir(SRC)
    np.random.seed(0)
    optimize.kword(
        charset="sigma-10",
        target=os.path.join("images", "mwdog_crop.png"),
        layers=layers_name,
        row_length=6,
        num_loops=1,
        init_mode="random",
        display=0,
        out_file=os.path.join("results", "_background_test.png"),
        nowait=True,
    )
    return seen


def test_a_single_layer_is_optimized_against_bare_paper(monkeypatch):
    """The fix that makes 1x1 verifiable, guarded at its cause.

    Upstream composites "the other layers" as
    ``layers[(layer_num + 1) % len(layer_offsets)]``, which for one layer wraps back
    round to that same layer. Every candidate glyph was therefore scored as if it
    were struck twice, and the mockup that came out could not be reproduced by any
    plan that strikes a cell once -- so ``erika.pipeline print`` reported a mismatch
    on every single-layer job, and the preview was the only thing worth trusting.

    A plan strikes once. The background for the only layer is bare paper.
    """
    backgrounds = _record_backgrounds("1x1", monkeypatch)
    assert len(backgrounds) == 1
    assert np.all(backgrounds[0] == 1.0)


def test_each_layer_is_optimized_against_the_others_and_not_itself(monkeypatch):
    """And the multi-layer path still composites, which is the regression to fear.

    Four layers must give four *different* backgrounds -- each one missing a
    different layer. Making them all bare paper, or all the product of everything,
    would both leave the suite green without this.
    """
    backgrounds = _record_backgrounds("4x1", monkeypatch)
    assert len(backgrounds) == 4
    assert len({bg.tobytes() for bg in backgrounds}) == 4
    # Each composites real ink: with a random start the other three are full of
    # glyphs, so bare paper here would mean they had been ignored.
    for background in backgrounds:
        assert background.min() < 1.0


# ---------------------------------------------------------------------------
# the carriage limit's off-by-one
# ---------------------------------------------------------------------------


def test_too_wide_error_suggests_a_row_length_that_actually_fits():
    """The advice in the refusal has to work when followed.

    ``resizeTarget`` pads the target by half a cell on every side, so a print of
    *n* characters per row occupies *n + 1* columns of cells. The message used to
    suggest ``-r <max_columns>``, which is exactly the value that had just been
    refused -- following it failed again with the same error.
    """
    charset = Charset.load("sigma-10", SRC)
    results = os.path.join(SRC, "results")
    os.makedirs(results, exist_ok=True)

    def choices_with(columns: int) -> str:
        path = os.path.join(results, f"_width_{columns}_choices.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"layer0_0_0": [[1] * columns]}, f)
        return path

    # One column past the carriage's reach.
    path = choices_with(charset.max_columns + 1)
    try:
        with pytest.raises(PlanError) as raised:
            planner.build_plan(path, charset)
    finally:
        os.remove(path)

    suggested = int(re.search(r"-r (\d+)", str(raised.value)).group(1))
    # Usable means strictly below the column count, not equal to it.
    assert suggested < charset.max_columns
    assert suggested == charset.max_columns - 1

    # And taking the advice works: accepted, and reaching the last column.
    path = choices_with(suggested)
    try:
        plan = planner.build_plan(path, charset)
    finally:
        os.remove(path)
    assert max(s.x for s in plan.strikes) // 2 + 1 == suggested


# ---------------------------------------------------------------------------
# strike force
#
# Nothing here has been on hardware. What these tests pin is the part that can
# be pinned without a typewriter: that the byte stream says what the plan meant,
# that a charset without forces is untouched by any of it, and that the two
# hazards the design worries about are actually refused.
# ---------------------------------------------------------------------------


def _force_charset(forces=(0x00, 0x03), chars="AB") -> Charset:
    """A small charset carrying every glyph once per force, as make_charset does."""
    glyphs = [ec.glyph_for_char(c) for c in chars]
    return Charset(
        name="test-forces", pitch=10, cell_w=24, cell_h=40, max_columns=65,
        codes=[ec.SPACE] + [g.code for f in forces for g in glyphs],
        advances=[True] + [True for _ in forces for _ in glyphs],
        chars=[" "] + [g.char for _ in forces for g in glyphs],
        forces=[None] + [f for f in forces for _ in glyphs],
        force_order=list(forces),
    )


def test_a_charset_without_forces_never_mentions_force(tmp_path, charset):
    """The whole point of the default: existing charsets print as they always did."""
    path = _write_choices(tmp_path, _random_choices(charset, 4, 6, FOUR_LAYERS, seed=1))
    job = planner.encode(planner.build_plan(path, charset))
    assert etp.OP_SET_FORCE not in [op for _, op, _ in etp.iter_ops(job.body)]


def test_force_is_asserted_once_per_change_not_once_per_strike(tmp_path):
    cs = _force_charset()
    # Two cells at each force, interleaved in the grid; grouping should sort
    # them into one run per force, so two assertions and not four.
    path = _write_choices(tmp_path, {"layer0_0_0": [[1, 3, 2, 4]]})
    job = planner.encode(planner.build_plan(path, cs))
    ops = [(op, operand) for _, op, operand in etp.iter_ops(job.body)]
    forces = [operand for op, operand in ops if op == etp.OP_SET_FORCE]
    # 0x00 for the first group, 0x03 for the second, then back to 0x00 at the end.
    assert forces == [0x00, 0x03, 0x00]


def test_force_grouping_types_the_hardest_strike_first(tmp_path):
    cs = _force_charset(forces=(0x00, 0x03, 0x06), chars="AB")
    grid = [[5, 1, 3, 6, 2, 4]]  # forces 0x06, 0x00, 0x03, 0x06, 0x00, 0x03
    path = _write_choices(tmp_path, {"layer0_0_0": grid})
    plan = planner.build_plan(path, cs)
    assert [cs.forces[s.index] for s in plan.strikes] == [0, 0, 3, 3, 6, 6]
    # ...and x still rises within each group, so the carriage sweeps one way.
    for a, b in zip(plan.strikes, plan.strikes[1:]):
        if cs.forces[a.index] == cs.forces[b.index]:
            assert a.x < b.x


def test_force_grouping_can_be_turned_off(tmp_path):
    cs = _force_charset()
    path = _write_choices(tmp_path, {"layer0_0_0": [[1, 3, 2, 4]]})
    plan = planner.build_plan(path, cs, group_by_force=False)
    assert [s.x for s in plan.strikes] == [0, 2, 4, 6]


def test_a_job_hands_the_machine_back_at_full_force(tmp_path):
    """Force outlives the job. Left soft, the next thing typed on this machine --
    by hand or by the firmware's chatbot -- comes out faint for no reason."""
    cs = _force_charset()
    path = _write_choices(tmp_path, {"layer0_0_0": [[3]]})  # a light glyph only
    job = planner.encode(planner.build_plan(path, cs))
    forces = [operand for _, op, operand in etp.iter_ops(job.body)
              if op == etp.OP_SET_FORCE]
    assert forces[-1] == cs.force_order[0]


def test_a_force_that_is_a_motion_code_is_refused():
    enc = etp.Encoder()
    with pytest.raises(etp.EtpError, match="move the head"):
        enc.set_force(ec.HALF_STEP_FORWARD)


def test_a_force_that_would_eat_the_byte_after_it_is_refused():
    """The same hole as the strike guard had. A machine that ignores the force
    command types the force byte, and 0xA5 typed is a carriage command that
    takes the byte after it -- which is the glyph the force was chosen for."""
    for code in sorted(ec.OPERAND_CODES):
        with pytest.raises(etp.EtpError, match="desynchronise"):
            etp.Encoder().set_force(code)


def test_no_force_above_the_wheels_range_is_offered():
    """A force has to be inert when typed, and only the wheel's own codes are
    known to be. Everything above is a motion or a command."""
    assert not any(ec.is_usable_force(v) for v in range(ec.MAX_FORCE + 1, 0x100))
    assert all(ec.is_usable_force(v) for v in range(0, ec.MAX_FORCE + 1))


@pytest.mark.parametrize("value", [0x00, 0x39, 0x67])
def test_a_force_of_any_usable_value_survives_the_wire(value):
    """Force 0 is the interesting one: the firmware used to signal "no trailing
    byte" with a zero, so a force of 0 would have been swallowed."""
    enc = etp.Encoder()
    enc.set_force(value)
    enc.strike(ec.glyph_for_char("A").code)
    codes = emulate.expand(enc.body())
    assert codes == [ec.SET_STRIKE_FORCE, value, ec.glyph_for_char("A").code]


def test_the_machine_takes_the_byte_after_the_force_command_as_the_force():
    """A machine that honours the command swallows the next byte whatever it is;
    one that does not types it. That asymmetry is why forces avoid motion codes,
    and it is modelled here rather than in expand() because it is the machine's
    behaviour, not the firmware's."""
    machine = emulate.Typewriter()
    a = ec.glyph_for_char("A").code
    machine.run([ec.SET_STRIKE_FORCE, 0x03, a])
    assert machine.force == 0x03
    assert [(i.x, i.force) for i in machine.impressions] == [(0, 0x03)]


def test_a_multi_force_plan_survives_the_virtual_typewriter(tmp_path):
    """The end-to-end guarantee, extended to force: the marks the machine leaves
    must map back to the same charset indices the plan chose -- which they only
    do if the recovery keys on (code, force). Keyed on the code alone every
    strike would come back as its full-force twin, and the render would agree
    with the plan while throwing the force away."""
    cs = _force_charset(forces=(0x00, 0x03), chars="AB")
    grid = [[1, 3, 2, 4], [4, 2, 3, 1]]
    path = _write_choices(tmp_path, {"layer0_0_0": grid})
    plan = planner.build_plan(path, cs)
    job = planner.encode(plan)

    machine = emulate.type_job(job, max_columns=cs.max_columns)
    assert machine.overruns == 0
    recovered = emulate.impressions_to_strikes(machine, cs)
    assert len(recovered) == len(plan.strikes)
    for got, want in zip(recovered, plan.strikes):
        assert (got.y, got.x, got.index) == (want.y, want.x, want.index)


def test_a_charset_naming_a_force_it_does_not_list_is_refused(tmp_path):
    """force_order is the typing order, so a force missing from it has no place
    in the plan -- and would otherwise raise ValueError from list.index() deep
    inside the sort, which says nothing about what is wrong."""
    path = os.path.join(tmp_path, "glyphs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "charset_name": "broken", "pitch": 10, "cell_width_px": 24,
            "cell_height_px": 40, "max_columns": 65, "forces": [0],
            "glyphs": [
                {"index": 0, "char": " ", "code": ec.SPACE, "advances": True,
                 "name": "space", "force": None},
                {"index": 1, "char": "A", "code": 0x30, "advances": True,
                 "name": "", "force": 7},
            ],
        }, f)
    with pytest.raises(PlanError, match="does not name"):
        Charset.load(path)


def test_the_probe_sheet_only_asks_for_forces_that_cannot_move_the_head(tmp_path):
    """The probe's whole safety argument. A value in 0x71..0x82 on a machine that
    ignores the command is a motion, and the sheet would be unreadable exactly
    where it has to be read."""
    from erika import pipeline

    out = os.path.join(tmp_path, "forces.etp")
    pipeline.main(["forces", "-o", out, "--run", "3"])
    job = etp.load(out)
    values = [operand for _, op, operand in etp.iter_ops(job.body)
              if op == etp.OP_SET_FORCE]
    assert values, "the probe asserted no forces at all"
    assert all(ec.is_usable_force(v) for v in values)
    machine = emulate.type_job(job)
    assert machine.overruns == 0


def _probe_forces(tmp_path, name, *extra):
    """The forces one probe sheet actually asks the machine for, in order."""
    from erika import pipeline

    out = os.path.join(tmp_path, name)
    assert pipeline.main(["forces", "-o", out, "--run", "3", *extra]) == 0
    job = etp.load(out)
    return [operand for _, op, operand in etp.iter_ops(job.body)
            if op == etp.OP_SET_FORCE], job


def test_a_coarse_probe_strides_the_values_at_the_step_it_was_given(tmp_path):
    """What --step is for: the whole byte is more than two hundred rows and ten
    sheets of paper, and a coarse pass first is what makes a wide sweep readable."""
    fine, _ = _probe_forces(tmp_path, "fine.etp", "--from", "0x00", "--to", "0x40")
    coarse, _ = _probe_forces(tmp_path, "coarse.etp",
                              "--from", "0x00", "--to", "0x40", "--step", "8")

    assert coarse == list(range(0x00, 0x41, 8))
    assert len(coarse) < len(fine) / 4
    # Evenly spaced, which is what makes a change in the ink obvious down the sheet.
    assert {b - a for a, b in zip(coarse, coarse[1:])} == {8}


def test_a_step_strides_the_values_and_drops_motion_codes_afterwards(tmp_path):
    """The order matters and only one of the two readings is a step.

    Striding the value space and dropping the motion codes after leaves a gap where
    the sweep lands on a code that would move the head. Dropping them *first* would
    renumber what is left and hand back values at uneven spacing -- N usable forces,
    but not every Nth candidate, which is unreadable as a sweep and the opposite of
    what a coarse pass is for.
    """
    forces, _ = _probe_forces(tmp_path, "over.etp",
                              "--from", "0x60", "--to", "0x74", "--step", "2")

    # 0x60..0x74 by twos straddles the top of the wheel's range; what lies above
    # it -- motions, and then commands -- goes.
    asked = range(0x60, 0x75, 2)
    assert forces == [v for v in asked if ec.is_usable_force(v)]
    # The gap is real -- something was dropped -- and every value still sits on the
    # stride from the first one.
    assert len(forces) < len(list(asked))
    assert all((v - 0x60) % 2 == 0 for v in forces)


def test_a_coarse_probe_is_still_safe_and_still_has_its_reference_row(tmp_path):
    """The two properties the sheet is built around have to survive a step."""
    from erika import emulate

    forces, job = _probe_forces(tmp_path, "coarse.etp", "--step", "3")
    assert forces
    assert all(ec.is_usable_force(v) for v in forces)

    machine = emulate.type_job(job)
    assert machine.overruns == 0
    first_force_row = next(i.y for i in machine.impressions if i.force is not None)
    assert [i for i in machine.impressions
            if i.y < first_force_row and i.force is None]


def test_the_whole_usable_range_fits_one_sheet_at_a_coarse_enough_step(tmp_path):
    """The case the flag exists for, stated as a number of lines of typing.

    "The whole byte" is asked for and something narrower comes back: only the
    wheel's own codes are safe to hand the machine as a force, so a sweep of
    0x00..0xFF is a sweep of 0x00..MAX_FORCE with a note saying what it dropped.
    That is still more than one sheet of paper.
    """
    from erika.pipeline import DEFAULT_AREA_ROWS

    _, fine = _probe_forces(tmp_path, "fine.etp", "--from", "0x00", "--to", "0xFF")
    _, coarse = _probe_forces(tmp_path, "coarse.etp",
                              "--from", "0x00", "--to", "0xFF", "--step", "16")

    assert fine.rows > DEFAULT_AREA_ROWS  # more than one sheet of paper
    assert coarse.rows <= DEFAULT_AREA_ROWS  # one


def test_a_step_is_a_stride_so_it_cannot_be_zero_or_negative(tmp_path):
    from erika import pipeline

    out = os.path.join(tmp_path, "forces.etp")
    for bad in ("0", "-3"):
        assert pipeline.main(["forces", "-o", out, "--step", bad]) == 2


def test_a_step_that_lands_only_on_motion_codes_is_refused(tmp_path):
    """Rather than writing a sheet with a heading and no rows under it."""
    from erika import pipeline

    out = os.path.join(tmp_path, "forces.etp")
    assert pipeline.main(
        ["forces", "-o", out, "--from", "0x72", "--to", "0x74"]
    ) == 2


def test_the_probe_says_how_it_swept(tmp_path, capsys):
    """The sheet's rows label their own values, but the run has to say the stride
    too -- a coarse pass brackets a neighbourhood rather than naming a value, and
    the advice for reading it back is different because of that."""
    from erika import pipeline

    out = os.path.join(tmp_path, "forces.etp")
    pipeline.main(["forces", "-o", out, "--run", "3", "--step", "4"])
    said = capsys.readouterr().out
    assert "in steps of 4" in said
    assert "--step 1" in said  # what to do about a row that differs

    pipeline.main(["forces", "-o", out, "--run", "3"])
    plain = capsys.readouterr().out
    assert "in steps of" not in plain
    assert "coarse pass" not in plain


def test_the_probe_samples_the_machine_before_it_sets_any_force(tmp_path):
    """The reference row. Without it there is nothing on the sheet to compare
    the swept rows against, and a machine that ignores the command looks the
    same as one that accepts every value."""
    from erika import pipeline

    out = os.path.join(tmp_path, "forces.etp")
    pipeline.main(["forces", "-o", out, "--run", "3"])
    machine = emulate.type_job(etp.load(out))
    first_force_row = next(i.y for i in machine.impressions if i.force is not None)
    sampled_before = [i for i in machine.impressions
                      if i.y < first_force_row and i.force is None]
    assert sampled_before


# ---------------------------------------------------------------------------
# the charset's ink model
# ---------------------------------------------------------------------------


def test_ink_model_leaves_paper_white_and_lifts_ink_off_black():
    """Both halves matter. Paper that is not 1.0 would tint the whole sheet; ink
    at 0.0 is what makes the optimizer refuse to mark a midtone cell at all."""
    from erika.make_charset import apply_ink_model

    tile = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = apply_ink_model(tile, ink=0.1, spread=0.0)
    assert out[0, 0] == pytest.approx(0.1)
    assert out[0, 2] == pytest.approx(1.0)
    assert 0.1 < out[0, 1] < 1.0


def test_a_lighter_strike_force_lays_down_less_ink():
    from erika.make_charset import apply_ink_model

    tile = np.zeros((4, 4), dtype=np.float32)
    full = apply_ink_model(tile, ink=0.1, spread=0.0, density=1.0)
    half = apply_ink_model(tile, ink=0.1, spread=0.0, density=0.5)
    assert full.mean() < half.mean() < 1.0


def test_ink_spread_pads_with_paper_not_with_ink():
    """The blur has to be told what lies outside the cell, and the answer is
    paper. Padded with zeros -- OpenCV's default -- every glyph would come back
    with a dark frame, which is ink where no key struck."""
    from erika.make_charset import apply_ink_model

    blank = np.ones((40, 24), dtype=np.float32)
    assert apply_ink_model(blank, ink=0.1, spread=1.5).min() == pytest.approx(1.0)

    tile = np.ones((40, 24), dtype=np.float32)
    tile[18:22, 10:14] = 0.0  # a mark in the middle, far from every edge
    out = apply_ink_model(tile, ink=0.0, spread=1.0)
    assert out[0, :].min() == pytest.approx(1.0)
    assert out[-1, :].min() == pytest.approx(1.0)
    assert out[:, 0].min() == pytest.approx(1.0)
    assert out[:, -1].min() == pytest.approx(1.0)


def test_a_scan_is_normalised_for_paper_but_not_for_ink(tmp_path):
    """The bug this replaced: mapping the 1st percentile to 0 took the darkest
    ink on the sheet and made it pure black -- destroying, in the one function
    that measures real ink, the property section 5.5 says the tonal range
    depends on."""
    import cv2

    from erika.make_charset import _sheet_from_scan

    # A sheet of slightly-grey "paper" with ink at 60/255 -- dark, not black.
    # Two marks rather than one, at opposite corners: a scan with no
    # registration marks on it is cropped to its ink, and a single mark would
    # leave no paper inside that crop for the assertions below to look at.
    sheet = np.full((80, 96), 240, dtype=np.uint8)
    sheet[10:30, 10:30] = 60
    sheet[50:70, 60:80] = 60
    path = os.path.join(tmp_path, "scan.png")
    cv2.imwrite(path, sheet)

    out, _, _ = _sheet_from_scan(path, n_tiles=8, cols=4, cell_w=24, cell_h=40)
    assert out.max() == pytest.approx(1.0, abs=0.01), "paper should read as white"
    assert out.min() > 0.15, f"ink was stretched towards black: {out.min()}"
    assert out.min() == pytest.approx(60 / 240, abs=0.05)


def test_resolve_densities_refuses_a_list_that_does_not_fit():
    from erika.make_charset import resolve_densities

    assert resolve_densities(1, None) == (1.0,)
    assert resolve_densities(3, (0.5, 0.25)) == (1.0, 0.5, 0.25)
    with pytest.raises(ValueError, match="need 2 densities"):
        resolve_densities(3, (0.5,))
    with pytest.raises(ValueError, match="in .0, 1."):
        resolve_densities(2, (1.5,))


def test_make_charset_refuses_a_force_that_would_move_the_head(tmp_path):
    from erika.make_charset import make_charset

    with pytest.raises(ValueError, match="commands"):
        make_charset(name="bad", base_path=str(tmp_path),
                     forces=(0x00, ec.HALF_LINE_FORWARD))


# ---------------------------------------------------------------------------
# soft matching (erika/softmatch.py)
# ---------------------------------------------------------------------------


def test_soft_match_refuses_a_block_that_does_not_divide_the_cell():
    """A partial block at the right or bottom edge would weight the middle of
    every cell above its edges -- a bias in the direction this is meant to
    remove."""
    from erika import softmatch

    softmatch.validate(0.5, 8, (40, 24))
    with pytest.raises(ValueError, match="does not divide"):
        softmatch.validate(0.5, 7, (40, 24))
    with pytest.raises(ValueError, match="weight must be"):
        softmatch.validate(1.5, 8, (40, 24))
    with pytest.raises(ValueError, match="at least 1"):
        softmatch.validate(0.5, 0, (40, 24))


def test_soft_match_installs_and_removes_itself_cleanly():
    """It replaces a name in *optimize*'s namespace, because that is where the
    binding the loop uses lives. Patching utils would not take effect."""
    import optimize
    import utils

    from erika import softmatch

    assert optimize.layer_optimization_pass is utils.layer_optimization_pass
    try:
        softmatch.install(0.5, 8)
        assert optimize.layer_optimization_pass is not utils.layer_optimization_pass
    finally:
        softmatch.uninstall()
    assert optimize.layer_optimization_pass is utils.layer_optimization_pass


def _tiny_run(charset_name, pass_fn, target, loops=4, seed=0):
    """One small optimizer run against `target`, returning (mockup, choices)."""
    from utils import prep_charset

    chars, _, _ = prep_charset(charset_name, SRC)
    char_h, char_w = chars.shape[1], chars.shape[2]
    rows, cols = target.shape[0] // char_h, target.shape[1] // char_w
    offsets = [(0, 0), (0, char_w // 2), (char_h // 2, 0), (char_h // 2, char_w // 2)]
    padded = np.pad(target, ((0, char_h // 2), (0, char_w // 2)), constant_values=1)
    mockup = np.ones_like(padded)
    layers = np.array([mockup.copy() for _ in offsets], dtype="float32")
    choices = np.zeros((len(offsets), rows * cols), dtype="uint16")

    def paint(n, off):
        for i, ch in enumerate(choices[n]):
            r, c = divmod(i, cols)
            layers[n][r * char_h + off[0]:(r + 1) * char_h + off[0],
                      c * char_w + off[1]:(c + 1) * char_w + off[1]] = chars[ch]

    np.random.seed(seed)
    for _ in range(loops):
        for n, off in enumerate(offsets):
            bg = layers[(n + 1) % 4] * layers[(n + 2) % 4] * layers[(n + 3) % 4]
            choices[n], mockup, _, _ = pass_fn(
                bg, mockup, padded, chars, choices[n], np.array(off),
                asymmetry=0.1, mode="greedy", temperature=1e-3)
            paint(n, off)
    return mockup, choices, chars


def test_soft_matching_puts_ink_where_per_pixel_error_refuses_to():
    """The point of the whole module, on the case that shows it: a flat panel of
    light grey. Per pixel, any mark in it costs more than bare paper, so the
    optimizer leaves it blank and the tone is simply wrong. Scored on local
    average tone it halftones instead."""
    import utils

    from erika import softmatch

    # Light grey -- inside the range that comes out as bare paper.
    target = np.full((5 * 40, 8 * 24), 200 / 255, dtype="float32")

    plain, plain_choices, _ = _tiny_run("sigma-10", utils.layer_optimization_pass, target)
    soft, soft_choices, _ = _tiny_run(
        "sigma-10",
        lambda *a, **k: softmatch._pass(*a, k["asymmetry"], k["mode"],
                                        k["temperature"], 0.9, 8),
        target,
    )
    assert (plain_choices == 0).mean() > 0.9, "expected the plain pass to give up"
    assert (soft_choices == 0).mean() < (plain_choices == 0).mean()
    # And the tone it produces is closer to what was asked for.
    assert abs(soft.mean() - target.mean()) < abs(plain.mean() - target.mean())


def test_soft_matching_leaves_a_mockup_the_plan_can_reproduce():
    """The property `pipeline print` checks on every run, and the one a change to
    the loss could quietly break: the mockup must stay a faithful composite of
    the chosen glyphs, or the plan is diffed against a picture of nothing."""
    from erika import softmatch

    target = np.full((4 * 40, 6 * 24), 150 / 255, dtype="float32")
    mockup, choices, chars = _tiny_run(
        "sigma-10",
        lambda *a, **k: softmatch._pass(*a, k["asymmetry"], k["mode"],
                                        k["temperature"], 0.8, 8),
        target,
    )
    # Recomposite from the choices alone, the way preview.render does.
    char_h, char_w = chars.shape[1], chars.shape[2]
    cols = target.shape[1] // char_w
    rebuilt = np.ones_like(mockup)
    offsets = [(0, 0), (0, char_w // 2), (char_h // 2, 0), (char_h // 2, char_w // 2)]
    for n, off in enumerate(offsets):
        for i, ch in enumerate(choices[n]):
            r, c = divmod(i, cols)
            rebuilt[r * char_h + off[0]:(r + 1) * char_h + off[0],
                    c * char_w + off[1]:(c + 1) * char_w + off[1]] *= chars[ch]
    assert np.abs(rebuilt - mockup).max() < 1e-5


# ---------------------------------------------------------------------------
# optimised cropping (erika/align.py)
# ---------------------------------------------------------------------------


def test_the_crop_family_is_the_papers_and_starts_from_identity():
    """64 crops for the paper's quarters, and identity first -- so the report can
    say what alignment bought, and a tie leaves the picture unresampled."""
    from erika import align

    family = align.candidates()
    assert len(family) == 64
    assert family[0].is_identity
    assert sum(c.is_identity for c in family) == 1
    assert {c.a for c in family} == {0, 0.25, 0.5, 0.75}
    with pytest.raises(ValueError, match="at least 1"):
        align.candidates(0)


def test_the_scale_is_one_more_character_across_the_row_at_most():
    """`s = (c + n) / n`: the family fine-tunes alignment, it does not reframe the
    picture. A scale that discarded a tenth of the photograph would be a
    different feature wearing this one's name."""
    from erika import align

    assert align.Crop(0, 0, 0).scale(40) == 1.0
    assert align.Crop(0, 0, 0.75).scale(40) == pytest.approx(1.01875)
    widest = max(c.scale(40) for c in align.candidates())
    assert widest < 1 + 1 / 40


def test_an_identity_crop_does_not_touch_a_single_pixel():
    """Which is what makes a search that finds nothing free. warpAffine with a
    unit scale and an integer translation copies pixel for pixel; if that ever
    stops being true the picture pays for the search even when it lost."""
    from erika import align

    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(120, 90), dtype=np.uint8)
    same = align.transform(image, align.Crop(0, 0, 0), 20, (40, 24), (1.0, 1.0))
    assert np.array_equal(same, image)


def test_a_shift_lands_where_the_paper_says_it_should():
    """The whole point of the module, as arithmetic.

    `a` is in *character widths measured in the resized frame*, but the warp is
    applied to the source, so the conversion runs through resizeTarget's scaling.
    Getting it wrong is invisible -- the picture still looks like the picture, just
    aligned to the wrong grid -- so it is pinned against a feature whose position
    can be read off exactly: a single black column on white.
    """
    from utils import resizeTarget

    from erika import align

    row_length, cell = 20, (40, 24)
    edge_at = 300
    image = np.full((400, 600), 255, dtype=np.uint8)
    image[:, edge_at] = 0

    def column_in_prepared(crop):
        warped = align.transform(image, crop, row_length, cell, (1.0, 1.0))
        prepared, padding = resizeTarget(warped, row_length, cell, (1.0, 1.0))
        darkest = int(np.argmin(prepared.mean(axis=0)))
        return darkest - padding["left"]

    base = column_in_prepared(align.Crop(0, 0, 0))
    # Half a character width left, in the frame the parameter is written in.
    shifted = column_in_prepared(align.Crop(0.5, 0, 0))
    assert shifted - base == pytest.approx(-cell[1] / 2, abs=1)
    # And a full character width is a whole cell.
    whole = column_in_prepared(align.Crop(1.0, 0, 0))
    assert whole - base == pytest.approx(-cell[1], abs=1)


def test_a_vertical_shift_uses_the_vertical_scale():
    """The two axes have different scales -- resizeTarget corrects the character
    aspect in y only -- so one shared divisor would be right in x and wrong in y.
    """
    from utils import resizeTarget

    from erika import align

    row_length, cell = 20, (40, 24)
    image = np.full((400, 600), 255, dtype=np.uint8)
    image[200, :] = 0

    def row_in_prepared(crop):
        warped = align.transform(image, crop, row_length, cell, (0.9, 1.0))
        prepared, padding = resizeTarget(warped, row_length, cell, (0.9, 1.0))
        return int(np.argmin(prepared.mean(axis=1))) - padding["top"]

    base = row_in_prepared(align.Crop(0, 0, 0))
    shifted = row_in_prepared(align.Crop(0, 0.5, 0))
    assert shifted - base == pytest.approx(-cell[0] / 2, abs=1)


def test_align_composites_the_way_the_optimizer_does():
    """The guard the module's own docstring points at.

    ``_greedy_cycles`` reimplements the middle of ``kword`` -- it has to, or the
    search would write choices.json 64 times over the run that is about to need
    it -- and the hazard in a reimplementation is the offset arithmetic and the
    index mapping, not the loss. So the choices it returns are recomposited
    independently here and required to give back the same mockup, which is the
    same invariant ``preview.render`` checks for a real plan.
    """
    from utils import prep_charset

    from erika import align

    chars, _, _ = prep_charset("sigma-10", SRC)
    cell_h, cell_w = chars.shape[1], chars.shape[2]
    target = np.full((4 * cell_h, 6 * cell_w), 140 / 255, dtype="float32")
    offsets = [(0, 0), (0, cell_w // 2), (cell_h // 2, 0), (cell_h // 2, cell_w // 2)]

    mockup, choices, padded = align._greedy_cycles(chars, target, offsets, 1, 0.1)

    cols = target.shape[1] // cell_w
    rebuilt = np.ones_like(mockup)
    for layer, offset in enumerate(offsets):
        for i, choice in enumerate(choices[layer]):
            row, col = divmod(i, cols)
            rebuilt[row * cell_h + offset[0]:(row + 1) * cell_h + offset[0],
                    col * cell_w + offset[1]:(col + 1) * cell_w + offset[1]] *= chars[choice]
    assert np.abs(rebuilt - mockup).max() < 1e-5
    assert mockup.shape == padded.shape


#: A test pattern on the character pitch: bars half a cell wide, one per cell.
#: The paper's figure 13 uses a checkerboard for the same reason -- there is
#: nothing in it but alignment, so nothing else can explain a change in score.
def _pitch_pattern(row_length: int, cell_w: int, cell_h: int, roll: int):
    image = np.full((6 * cell_h, row_length * cell_w), 255, dtype=np.uint8)
    for col in range(row_length):
        image[:, col * cell_w:col * cell_w + cell_w // 2] = 0
    return np.roll(image, roll, axis=1)


def _search_pattern(tmp_path, roll, cell_w=24, cell_h=40, row_length=12, steps=4):
    import cv2

    from erika import align

    path = os.path.join(tmp_path, f"pitch{roll}.png")
    cv2.imwrite(path, _pitch_pattern(row_length, cell_w, cell_h, roll))
    result, _ = align.search(path, "sigma-10", row_length, "4x1",
                             steps=steps, loops=2, base_path=SRC)
    return result


def test_the_chosen_shift_tracks_where_the_pattern_actually_sits(tmp_path):
    """The search has to *find* the misalignment, not merely improve on it.

    So the pattern is moved a known quarter cell and the chosen shift has to move
    with it by the same quarter -- which is the assertion that would fail if the
    search were ranking noise, and the reason the default is two greedy cycles
    rather than the paper's one. Modulo half a cell, for the reason the next test
    pins.
    """
    plain = _search_pattern(tmp_path, 0)
    quarter = _search_pattern(tmp_path, 6)  # a quarter of a 24px cell
    moved = (plain.crop.a - quarter.crop.a) % 0.5
    assert moved == pytest.approx(0.25), (
        f"pattern moved a quarter cell but the chosen shift went from "
        f"{plain.crop.a} to {quarter.crop.a}"
    )
    for result in (plain, quarter):
        assert result.gain > 0


def test_half_a_cell_is_not_a_misalignment_under_a_half_cell_layer_scheme(tmp_path):
    """A property of this pipeline that is easy to expect the wrong way round.

    ``4x1`` puts layers at 0 and 0.5 of a cell, so the set of placements is
    already half-cell periodic: shifting the target half a cell maps every layer-0
    cell onto a layer-1 cell, which is a relabelling and not a harder problem.
    Hence the tolerance rather than equality -- the interior is equivalent and what
    is left over is the border, where a shift changes which half-cell of padding
    the picture sits against. That residue is small here because the pattern is
    periodic and uniform; on a photograph it is bigger, which is why a half-cell
    crop is not simply a no-op in practice.

    Two consequences worth having pinned. Alignment has little to offer against a
    *half*-cell error, and the effective period in each axis is half a cell rather
    than a whole one -- so the paper's quarters sample that period twice over, and
    an ``--align-steps 2`` probes almost nothing but the scale and the border.
    """
    plain = _search_pattern(tmp_path, 0)
    half = _search_pattern(tmp_path, 12)
    assert half.identity_score == pytest.approx(plain.identity_score, abs=0.05)


def test_alignment_takes_every_phase_to_about_the_same_place(tmp_path):
    """What the feature is actually for. The unaligned score swings by more than a
    point depending on where the picture happens to sit; afterwards the phase
    barely matters, which is the whole claim of section 5.3.1."""
    results = [_search_pattern(tmp_path, roll) for roll in (0, 6, 12, 18)]
    before = [r.identity_score for r in results]
    after = [r.score for r in results]
    assert np.ptp(before) > 0.8, "the test pattern is not phase-sensitive enough"
    assert np.ptp(after) < np.ptp(before) / 2
    assert min(after) > max(before)


def test_align_writes_a_target_the_optimizer_can_be_pointed_at(tmp_path):
    """Also covers the destination not existing yet, which is every first run:
    ``samefile`` stats both paths, so the same-input guard has to check the
    destination is there before asking."""
    from erika import align

    destination = "results/_test-aligned.png"
    assert not os.path.exists(os.path.join(SRC, destination))
    result, path = align.apply_to_file(
        "images/mwdog_crop.png", "sigma-10", 12, "4x1", steps=2, loops=1,
        base_path=SRC, destination=destination,
    )
    written = os.path.join(SRC, path)
    try:
        assert os.path.isfile(written)
        assert result.candidates == 8
        assert result.seconds > 0
    finally:
        os.remove(written)


def test_align_refuses_to_overwrite_its_own_input():
    """Aligning an already-aligned target is not idempotent -- each pass resamples
    and the score carries noise -- so replacing the input with a second-generation
    copy is the wrong thing to do quietly."""
    from erika import align

    # A tracked file, so the test does not depend on anything in results/ -- and
    # the guard fires before the search runs, so nothing writes over the sample.
    same = "images/mwdog_crop.png"
    with pytest.raises(ValueError, match="overwrite its own input"):
        align.apply_to_file(
            same, "sigma-10", 12, "4x1", steps=1, loops=1,
            base_path=SRC, destination=same,
        )
    assert os.path.getsize(os.path.join(SRC, same)) > 0


# ---------------------------------------------------------------------------
# raw bytes and direct step control (erika/etp.py, erika/emulate.py)
# ---------------------------------------------------------------------------


def test_a_raw_command_will_not_go_out_without_its_operand():
    """The mistake that breaks the machine rather than the file.

    A command that takes an operand and is sent without one leaves the machine
    reading whatever comes next as its count. The file is intact, the CRC
    passes, and the firmware -- which has no opinion about raw bytes -- has
    nothing to notice.
    """
    enc = etp.Encoder()
    with pytest.raises(etp.EtpError, match="takes an operand"):
        enc.raw_command(ec.CARRIAGE_STEPS)
    with pytest.raises(etp.EtpError, match="takes no operand"):
        enc.raw_command(0x8E, 1)


def test_a_move_longer_than_one_command_is_split():
    enc = etp.Encoder()
    enc.carriage_steps(300)
    operands = [operand for _, op, operand in etp.iter_ops(enc.body())
                if op == etp.OP_RAW]
    # command, count, command, count, ... and the counts come to 300.
    assert operands[0::2] == [ec.CARRIAGE_STEPS] * 3
    assert sum(ec.decode_step_operand(v) for v in operands[1::2]) == 300


def test_a_backwards_move_uses_the_tables_own_two_s_complement():
    enc = etp.Encoder()
    enc.carriage_steps(-12)
    operands = [operand for _, op, operand in etp.iter_ops(enc.body())
                if op == etp.OP_RAW]
    assert operands == [ec.CARRIAGE_STEPS, 0xF4]  # 256 - 12
    assert ec.decode_step_operand(0xF4) == -12


@pytest.mark.parametrize("steps", list(range(1, 20)) + [40, 127, -5, -40])
def test_a_platen_feed_never_asks_for_a_forbidden_step_count(steps):
    """"Die Schritte 3, 4, 5, 6 sind verboten!" -- so a feed that lands on one
    has to be split, and the split has to come to the same distance."""
    enc = etp.Encoder()
    enc.platen_steps(steps)
    operands = [operand for _, op, operand in etp.iter_ops(enc.body())
                if op == etp.OP_RAW]
    counts = [ec.decode_step_operand(v) for v in operands[1::2]]
    assert operands[0::2] == [ec.PLATEN_STEPS] * len(counts)
    assert sum(counts) == steps
    assert not [c for c in counts if abs(c) in ec.FORBIDDEN_PLATEN_STEPS]


@pytest.mark.parametrize("pitch", (10, 12))
def test_carriage_steps_and_the_escapement_come_to_the_same_place(pitch):
    """The claim the probe sheet is built to test, checked against the model.

    A cell asked for in 1/120" steps has to land where a cell asked for as two
    half-steps lands, or the two ways of moving the head cannot be mixed -- and
    mixing them is the whole point of adding the direct commands.
    """
    cell = 2 * ec.carriage_steps_per_half_step(pitch)

    stepped = etp.Encoder()
    for _ in range(20):
        stepped.strike(ec.glyph_for_char("!").code)
        stepped.carriage_steps(cell)

    escapement = etp.Encoder()
    for _ in range(20):
        escapement.strike(ec.glyph_for_char("!").code)
        escapement.right(2)

    def marks(enc):
        machine = emulate.Typewriter(pitch=pitch).run(emulate.expand(enc.body()))
        return [(i.y, i.x) for i in machine.impressions]

    assert marks(stepped) == marks(escapement)


def test_a_step_smaller_than_a_half_step_is_carried_rather_than_lost():
    """Quarter-cell offsets rest on this: three carriage steps at pitch 10 are a
    quarter of a cell, and four of them have to come to one whole cell rather
    than to nothing four times over."""
    enc = etp.Encoder()
    for _ in range(4):
        enc.carriage_steps(3)  # a quarter cell at pitch 10
    enc.strike(ec.glyph_for_char("!").code)
    machine = emulate.Typewriter(pitch=10).run(emulate.expand(enc.body()))
    assert [(i.y, i.x) for i in machine.impressions] == [(0, 2)]  # one whole cell


def test_no_advance_prints_the_next_glyph_where_the_head_stands():
    """0xA9 in the order a plan would use it: the code, the glyphs that stack,
    and then the one that moves on."""
    enc = etp.Encoder()
    enc.raw(0xA9)
    enc.strike(ec.glyph_for_char("-").code)
    enc.strike(ec.glyph_for_char("O").code)
    enc.strike(ec.glyph_for_char("X").code)
    machine = emulate.Typewriter().run(emulate.expand(enc.body()))
    assert [(i.x, ec.describe_code(i.code)) for i in machine.impressions] == [
        (0, "'-'"), (0, "'O'"), (2, "'X'"),
    ]


def test_the_emulator_records_a_mode_switch_rather_than_guessing_at_it():
    """What 0x8C does to this machine is what the probe sheet exists to find
    out. A model that assumed an answer would have the test suite certify it.

    0x8E used to be the example here and is now modelled, which is the shape of
    the thing: a code moves out of `probes` when a sheet has said what it does,
    and not before.
    """
    enc = etp.Encoder()
    enc.raw(0x8C)  # correction ribbon on -- erase or white ink, depending on tape
    enc.raw_command(0xAA, 4)
    enc.strike(ec.glyph_for_char("A").code)
    machine = emulate.Typewriter().run(emulate.expand(enc.body()))
    assert machine.probes == [(0x8C, None), (0xAA, 4)]
    assert [(i.x) for i in machine.impressions] == [0]  # nothing moved the head


def test_the_disassembly_says_which_bytes_its_columns_do_not_account_for():
    enc = etp.Encoder()
    enc.raw(0x8C)
    enc.strike(ec.glyph_for_char("A").code)
    enc.end()
    text = etp.disassemble(etp.Job(body=enc.body()))
    assert "do NOT account for" in text
    assert "CORRECTION_ON" in text
    # A job that only steps is fully modelled, so it gets no such warning.
    plain = etp.Encoder()
    plain.carriage_steps(12)
    plain.end()
    assert "do NOT account for" not in etp.disassemble(etp.Job(body=plain.body()))


# ---------------------------------------------------------------------------
# the control code probe sheet (erika.pipeline codes)
# ---------------------------------------------------------------------------


def _codes_sheet(tmp_path, *extra):
    from erika import pipeline

    out = os.path.join(tmp_path, "codes.etp")
    assert pipeline.main(["codes", "-o", out, *extra]) == 0
    return etp.load(out)


@pytest.mark.parametrize("pitch", ("10", "12"))
def test_the_probe_sheet_stays_on_the_paper(tmp_path, pitch):
    """Every section returns the head to the margin, so a code that displaces it
    cannot carry the error into the next section -- and nothing runs the
    carriage off the end even if every probe does exactly what it says."""
    job = _codes_sheet(tmp_path, "--pitch", pitch)
    machine = emulate.type_job(job)
    assert machine.overruns == 0


def test_the_probe_sheet_asks_the_machine_the_questions_it_says_it_does(tmp_path):
    job = _codes_sheet(tmp_path)
    raw = [operand for _, op, operand in etp.iter_ops(job.body) if op == etp.OP_RAW]
    for code in (0xAA, ec.CARRIAGE_STEPS, ec.PLATEN_STEPS, 0xA9, 0x8E, 0x8C, 0x89):
        assert code in raw, f"the sheet never sends 0x{code:02X}"
    # And it puts the pitch back, or everything typed after it comes out narrow.
    assert raw[-1] == 0x87


def test_the_emulator_reproduces_section_7_the_way_the_paper_read_it(tmp_path):
    """The model tied back to the measurement it came from.

    The printed sheet reported the five letters reading EDCBA with the A at
    column 19 -- one cell left of the column 20 the head started at. That single
    observation is the whole of the position model for 0x8E, so this is the test
    that fails if the model is ever quietly inverted into the mirror of the
    forward one, which is the plausible wrong answer.
    """
    job = _codes_sheet(tmp_path)
    machine = emulate.type_job(job)
    letters = [(i.x / 2, ec.glyph_for_code(i.code).char)
               for i in machine.impressions]
    groups = [letters[k:k + 5] for k in range(len(letters) - 4)
              if [c for _, c in letters[k:k + 5]] == list("ABCDE")]
    # Two of them: the section heading says ABCDE, and then section 7 types it.
    # The heading runs forwards, which is what makes the second one legible.
    assert len(groups) == 2
    assert groups[0] == [(23, "A"), (24, "B"), (25, "C"), (26, "D"), (27, "E")]
    assert groups[1] == [(19, "A"), (18, "B"), (17, "C"), (16, "D"), (15, "E")]


def test_the_probe_sheets_two_combs_ask_for_the_same_distance(tmp_path):
    """Section 2's whole argument: the upper comb is typed with the escapement
    and the lower one in 1/120" steps, and they are only comparable by eye if
    the plan asked for the same place twice."""
    job = _codes_sheet(tmp_path)
    machine = emulate.type_job(job)
    ruler = ec.glyph_for_char("!").code
    rows = {}
    for imp in machine.impressions:
        if imp.code == ruler:
            rows.setdefault(imp.y, []).append(imp.x)
    combs = [xs for xs in rows.values() if len(xs) == 20]
    assert len(combs) == 2, "expected the escapement comb and the stepped one"
    assert combs[0] == combs[1]


# ---------------------------------------------------------------------------
# direct step expansion (IMG STEPS ON -- erika_ai/src/erika_image.cpp)
# ---------------------------------------------------------------------------


MODES = (emulate.STEPS_OFF, emulate.STEPS_AUTO, emulate.STEPS_ALL)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("home_each_row", [True, False])
@pytest.mark.parametrize("pitch", ["sigma-10", "sigma-12"])
def test_direct_steps_land_where_the_keystrokes_would(tmp_path, pitch,
                                                      home_each_row, mode):
    """The claim `IMG STEPS` rests on, checked on a whole plan.

    A move can go out as a run of SPACE and half-step keys or as a count of
    1/120" carriage steps, and "auto" mixes the two within a single job. All
    three have to put the head in the same place -- so the setting is a matter of
    bytes and units rather than of geometry, and switching it does not change
    the picture. If they disagree the difference is a whole print job of drift,
    and here is the only cheap place to notice.
    """
    cs = Charset.load(pitch, SRC)
    plan = planner.build_plan(
        _write_choices(tmp_path, _random_choices(cs, 7, 11, FOUR_LAYERS, seed=5)),
        cs,
        home_each_row=home_each_row,
    )
    job = planner.encode(plan)

    keystrokes = emulate.type_job(job, max_columns=cs.max_columns)
    stepped = emulate.type_job(job, max_columns=cs.max_columns, direct_steps=mode)

    assert stepped.impressions == keystrokes.impressions
    assert stepped.overruns == keystrokes.overruns == 0


def test_using_step_commands_for_everything_costs_bytes(tmp_path, charset):
    """The measurement that made the setting three-state rather than a switch.

    A step command is two bytes whatever the distance and a keystroke run is one
    byte per half cell, so the commands lose on the one-cell hops that most of a
    picture is made of. "all" is for the units, not the bytes, and saying so
    needs this to be a number rather than an impression.
    """
    plan = planner.build_plan(
        _write_choices(tmp_path, _random_choices(charset, 8, 20, FOUR_LAYERS,
                                                 seed=3, density=0.25)),
        charset,
    )
    body = planner.encode(plan).body
    plain = len(emulate.expand(body))
    assert len(emulate.expand(body, direct_steps=emulate.STEPS_ALL)) > plain


def test_auto_never_costs_more_bytes_than_the_keystrokes(tmp_path, charset):
    """Which is the whole of what "auto" promises."""
    for seed, density in ((3, 0.25), (4, 0.6), (5, 0.9)):
        plan = planner.build_plan(
            _write_choices(tmp_path, _random_choices(charset, 8, 20, FOUR_LAYERS,
                                                     seed=seed, density=density)),
            charset,
        )
        body = planner.encode(plan).body
        assert (len(emulate.expand(body, direct_steps=emulate.STEPS_AUTO))
                <= len(emulate.expand(body)))


def test_auto_takes_the_long_moves_and_leaves_the_short_ones(tmp_path, charset):
    """A blank run is where a print's time goes, and it is also the only place
    the commands are cheaper -- so this is what "auto" has to be doing."""
    short = etp.Encoder()
    short.right(2)  # one cell: one SPACE against a two-byte command
    assert emulate.expand(short.body(), direct_steps=emulate.STEPS_AUTO) == [ec.SPACE]

    long_run = etp.Encoder()
    long_run.right(20)  # ten cells
    assert emulate.expand(long_run.body(), direct_steps=emulate.STEPS_AUTO) == [
        ec.CARRIAGE_STEPS, 120,
    ]


def test_a_micro_line_is_two_platen_steps():
    """Both sides have to agree on this or a sub-half-line offset moves the paper
    a twentieth of a line in one expansion and a tenth in the other.

    Three of them come to six steps, which the platen refuses, so they go out as
    2 + 2 + 1 -- six bytes where the keystrokes were three. Exactly why "auto"
    leaves a feed this small alone.
    """
    enc = etp.Encoder()
    enc.micro_down(3)
    assert emulate.expand(enc.body(), direct_steps=emulate.STEPS_ALL) == [
        ec.PLATEN_STEPS, 2, ec.PLATEN_STEPS, 2, ec.PLATEN_STEPS, 2,
    ]
    assert emulate.expand(enc.body(), direct_steps=emulate.STEPS_AUTO) == [
        ec.MICRO_LINE_FORWARD
    ] * 3


def test_a_long_move_is_split_into_commands_the_operand_can_carry():
    enc = etp.Encoder()
    enc.right(60)  # 60 half-steps = 360 carriage steps at pitch 10
    raw = emulate.expand(enc.body(), direct_steps=emulate.STEPS_ALL)
    counts = [ec.decode_step_operand(v) for v in raw[1::2]]
    assert raw[0::2] == [ec.CARRIAGE_STEPS] * len(counts)
    assert sum(counts) == 360
    assert max(counts) <= ec.MAX_STEPS_PER_COMMAND


def test_a_direct_feed_never_asks_the_platen_for_a_forbidden_count():
    """The firmware splits these too, and it has to split them the same way."""
    for half_lines in range(1, 12):
        enc = etp.Encoder()
        enc.down(half_lines)
        raw = emulate.expand(enc.body(), direct_steps=emulate.STEPS_ALL)
        counts = [ec.decode_step_operand(v) for v in raw[1::2]]
        assert sum(counts) == half_lines * ec.PLATEN_STEPS_PER_HALF_LINE
        assert not [c for c in counts if abs(c) in ec.FORBIDDEN_PLATEN_STEPS]


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_firmware_has_the_same_three_settings():
    """expand() is a re-implementation of fetchNext(), and a mode that exists on
    only one side is a job that comes out differently on the machine than in the
    check that cleared it."""
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.h"), encoding="utf-8").read()
    match = re.search(r"enum DirectSteps\s*:\s*uint8_t\s*\{(.*?)\}", text, re.S)
    assert match, "could not find the DirectSteps enum"
    names = set(re.findall(r"Steps(\w+)\s*=", match.group(1)))
    assert names == {"Off", "Auto", "All"}
    assert {m.capitalize() for m in MODES} == names


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_nothing_the_preamble_sends_could_move_the_head_or_eat_a_byte():
    """`IMG PREPARE` writes a handful of bytes before the first opcode, and they
    are the only bytes in a job that no plan accounted for.

    So they have to be inert as far as position goes. A byte in the motion block
    would displace the whole picture before it started, one from the wheel's own
    range would type a character into the top-left corner, and one that carries
    an operand would eat the first opcode of the body. None of those would look
    like a preamble problem on the paper.
    """
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.h"), encoding="utf-8").read()
    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    names = [
        "ERIKA_RESET", "ERIKA_LINE_SPACING_1", "ERIKA_LINE_SPACING_1_5",
        "ERIKA_LINE_SPACING_2", "ERIKA_PITCH_10", "ERIKA_PITCH_12",
        "ERIKA_PITCH_15", "ERIKA_KEYBOARD_OFF", "ERIKA_KEYBOARD_ON",
        "ERIKA_REPORT_WHEN_PRINTED",
    ]
    for name in names:
        code = defines.get(name)
        assert code is not None, f"{name} is gone from the firmware"
        assert code not in ec.CONTROL_CODES, f"{name} is a motion code"
        assert not ec.is_glyph_code(code), f"{name} is a key on the wheel"
        assert code not in ec.OPERAND_CODES, f"{name} carries an operand"
        # And it is a code the published table names, not an invention.
        assert code in ec.CONTROL_CODE_NAMES, f"0x{code:02X} is not in the table"

    # The postamble exists to undo 0x91, and it has to be the byte that does.
    match = re.search(r"_postamble\[_postambleLen\+\+\] = (\w+);",
                      open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"),
                           encoding="utf-8").read())
    assert match and match.group(1) == "ERIKA_KEYBOARD_ON"
    assert "ERIKA_KEYBOARD_OFF" in text  # the thing it is undoing


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_postamble_runs_on_every_exit_from_a_job():
    """The property the whole preamble depends on. A machine left in duplex has a
    keyboard that does nothing, and nothing on the device says why -- so finish,
    abort and fail all have to put it back, and a failure is exactly the path
    somebody would forget."""
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    for func in ("abort", "fail"):
        body = re.search(
            rf"void ErikaImagePrinter::{func}\(.*?\n}}", text, re.S
        )
        assert body, f"could not find {func}()"
        assert "flushPostamble()" in body.group(0), (
            f"{func}() does not send the postamble, so it can leave the keyboard dead"
        )
    # finish() is reached only after the postamble has drained through tick().
    assert re.search(r"_bodyEnded = true;\s*\n\s*if \(nextPostambleByte\(\)\)", text)


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_switch_codes_mean_the_same_thing_on_both_sides():
    """The six codes that name a slide-switch position, mirrored.

    These travel in both directions -- sent they set the switch's setting,
    received they report the operator moving it -- so the firmware reads its own
    pinning bytes back through the same table it decodes the machine's reports
    with. A code that meant 10 on one side and 12 on the other would pin a job to
    the wrong pitch *and* report that it had succeeded.
    """
    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    assert {defines["ERIKA_PITCH_10"]: 10,
            defines["ERIKA_PITCH_12"]: 12,
            defines["ERIKA_PITCH_15"]: 15} == ec.PITCH_FOR_CODE
    assert {defines["ERIKA_LINE_SPACING_1"]: 10,
            defines["ERIKA_LINE_SPACING_1_5"]: 15,
            defines["ERIKA_LINE_SPACING_2"]: 20} == ec.SPACING_FOR_CODE
    # And the pitches a job may actually be planned for are the ones the header
    # flag can express, which is fewer than the machine answers to.
    assert set(ec.PITCH_WIDTH_MM) < set(ec.PITCH_FOR_CODE.values())
    for pitch in ec.PITCH_WIDTH_MM:
        assert ec.PITCH_FOR_CODE[ec.pitch_code(pitch)] == pitch


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_a_job_pins_its_pitch_unless_the_operator_says_not_to():
    """`IMG PREPARE` is three levels, and which one is the default is the whole
    point of the split.

    Nothing can ask the machine where its slide switches are, so a job that does
    not pin its pitch is a job that hopes. What it costs to pin is a switch that
    reads wrong until it is next moved; what it costs not to is a sheet typed at
    12 characters per inch from a plan laid out for 10, which loads, verifies,
    prints and comes out with every glyph overlapping its neighbour. So the
    pinning level is on at boot -- and separate from the keyboard codes, which
    are off at boot because 0x92 going missing leaves a typewriter that cannot
    type.
    """
    header = open(os.path.join(FIRMWARE_SRC, "erika_image.h"), encoding="utf-8").read()
    match = re.search(r"enum Prepare\s*:\s*uint8_t\s*\{(.*?)\}", header, re.S)
    assert match, "could not find the Prepare enum"
    assert set(re.findall(r"Prepare(\w+)\s*=", match.group(1))) == {
        "Off", "Pitch", "All"
    }
    assert re.search(r"Prepare _prepare = PreparePitch;", header), (
        "the default is no longer the level that pins the pitch"
    )

    source = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    body = re.search(r"void ErikaImagePrinter::buildPreamble\(\).*?\n}", source, re.S)
    assert body, "could not find buildPreamble()"
    # The keyboard pair -- the half that can leave the machine unable to type --
    # is reached only at the level that is off by default.
    for byte in ("ERIKA_KEYBOARD_OFF", "ERIKA_KEYBOARD_ON", "ERIKA_RESET"):
        line = re.search(rf"^.*{byte}.*$", body.group(0), re.M)
        assert line, f"{byte} is gone from the preamble"
        assert "PrepareAll" in body.group(0)
    assert body.group(0).count("PrepareAll") >= 2

    # And the pitch pinned is the job's own, not a setting.
    pins = re.search(r"void ErikaImagePrinter::queuePins\(\).*?\n}", source, re.S)
    assert pins, "could not find queuePins()"
    assert "_hdr.pitch12() ? ERIKA_PITCH_12 : ERIKA_PITCH_10" in pins.group(0)
    assert "ERIKA_LINE_SPACING_1" in pins.group(0)


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_a_switch_moved_mid_print_is_read_and_overruled():
    """The machine reports 0x84..0x89 when a switch moves, and mid-job is when it
    happens -- the sheet is coming out narrow, so somebody reaches for the pitch
    switch. While a picture types, the firmware's print loop is the only reader of
    the link, so if tick() does not look, nobody ever will: the report is dropped
    by flushInput() at the end of the job and no code exists to ask afterwards.
    """
    source = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"), encoding="utf-8").read()
    tick = re.search(r"void ErikaImagePrinter::tick\(\).*?\n}", source, re.S)
    assert tick and "watchSwitches()" in tick.group(0), (
        "tick() does not read what the machine says, so a switch moved mid-print "
        "is lost"
    )
    watch = re.search(r"void ErikaImagePrinter::watchSwitches\(\).*?\n}", source, re.S)
    assert watch, "could not find watchSwitches()"
    assert "pollSettings()" in watch.group(0)
    assert "queuePins()" in watch.group(0), "it notices but does nothing about it"
    assert "PrepareOff" in watch.group(0), (
        "it must not overrule a machine the operator asked it to leave alone"
    )
    # The reports are decoded in one place, and both directions go through it.
    interface = open(os.path.join(FIRMWARE_SRC, "erika_interface.cpp"),
                     encoding="utf-8").read()
    raw = re.search(r"void ErikaInterface::writeRaw\(.*?\n}", interface, re.S)
    assert raw and "noteSettingCode" in raw.group(0), (
        "a pitch code this firmware sends has to update the same log a report does"
    )
    for name in ("ErikaInterface::readInput", "ErikaInterface::flushInput"):
        func = re.search(rf"{name}\(.*?\n}}", interface, re.S)
        assert func and "noteSettingCode" in func.group(0), (
            f"{name}() drops the machine's switch reports"
        )


# ---------------------------------------------------------------------------
# overstriking without a backspace (planner --no-advance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("home_each_row", [True, False])
def test_no_advance_puts_every_strike_exactly_where_the_backspace_did(
    tmp_path, charset, home_each_row
):
    """The whole claim. Typing a stack with Doppeldruck rather than a backspace
    is supposed to change how the marks get there and not where they are -- so
    the same plan, encoded both ways, has to come off the virtual machine
    identically."""
    plan = planner.build_plan(
        _write_choices(tmp_path, _random_choices(charset, 6, 10, FOUR_LAYERS, seed=7)),
        charset,
        home_each_row=home_each_row,
    )
    backspaced = emulate.type_job(planner.encode(plan))
    stacked = emulate.type_job(planner.encode(plan, no_advance=True))

    assert stacked.impressions == backspaced.impressions
    assert stacked.overruns == backspaced.overruns == 0


def test_no_advance_replaces_the_backspaces_rather_than_adding_to_them(tmp_path,
                                                                      charset):
    """What has to be true is that the backspaces are gone: leaving them in *and*
    suppressing the advance would walk left across the row.

    On the wire it is the same number of bytes -- one 0xA9 where there was one
    BACKSPACE -- so this is a different mechanism rather than a saving. In the
    file it *is* a saving, because a backspace is an OP_LEFT and its operand
    while NO_ADVANCE carries none, and a job has to fit in SPIFFS.
    """
    plan = planner.build_plan(
        # One layer scheme where every layer sits at the same offset, so every
        # inked cell is a stack of up to four.
        _write_choices(tmp_path, _random_choices(charset, 5, 8,
                                                 [(0, 0)] * 4, seed=11)),
        charset,
    )
    plain = planner.encode(plan, no_advance=False)
    stacked = planner.encode(plan)

    def count(job, want):
        return sum(1 for _, op, _ in etp.iter_ops(job.body) if op == want)

    assert count(stacked, etp.OP_NO_ADVANCE) > 0
    assert count(stacked, etp.OP_LEFT) < count(plain, etp.OP_LEFT)
    assert len(stacked.body) < len(plain.body)
    # ...and the same number of bytes actually reach the typewriter.
    assert len(emulate.expand(stacked.body)) == len(emulate.expand(plain.body))


def test_a_dead_key_is_not_given_a_no_advance_it_does_not_need(tmp_path, charset):
    """It already does not advance. Prefixing it would suppress the advance of
    whatever came next instead, which is a different cell."""
    dead = [i for i, adv in enumerate(charset.advances) if not adv]
    if not dead:
        pytest.skip("this charset has no dead keys")
    grid = [[dead[0], dead[0]]]
    path = _write_choices(tmp_path, {"layer0_0_0": grid, "layer1_0_0": grid})
    job = planner.encode(planner.build_plan(path, charset), no_advance=True)
    assert not [1 for _, op, _ in etp.iter_ops(job.body) if op == etp.OP_NO_ADVANCE]


def test_the_last_glyph_of_a_stack_still_advances(tmp_path, charset):
    """Or the next cell is typed on top of this one."""
    enc = etp.Encoder()
    enc.no_advance()
    enc.strike(ec.glyph_for_char("-").code)
    enc.strike(ec.glyph_for_char("O").code)
    enc.strike(ec.glyph_for_char("X").code)
    machine = emulate.Typewriter().run(emulate.expand(enc.body()))
    assert [i.x for i in machine.impressions] == [0, 0, 2]


def test_the_plan_summary_says_when_it_used_doppeldruck(tmp_path, charset):
    """It is off by default and unconfirmed, so a sheet that came out wrong
    should be able to say whether it was on."""
    plan = planner.build_plan(
        _write_choices(tmp_path, _random_choices(charset, 4, 6,
                                                 [(0, 0)] * 3, seed=13)),
        charset,
    )
    job = planner.encode(plan)
    assert "without advancing" in planner.summarize(plan, job)
    plain = planner.encode(plan, no_advance=False)
    assert "without advancing" not in planner.summarize(plan, plain)


# ---------------------------------------------------------------------------
# typing right to left (0x8E, planner backward=)
#
# Section 7 of the control-code sheet answered one question and left three. It
# answered the strike: five letters typed from column 20 read EDCBA with the A
# at column *19*, so the head moves one cell left and then marks. It did not ask
# what a motion key, a Doppeldruck or a dead key does in that mode, and each of
# those would put every mark after it in the wrong cell.
#
# So these tests come in two halves. One says the saving is real and lands where
# the plan said. The other says the planner stays inside what the sheet actually
# measured -- and that is the half that matters, because the failure it guards
# against is a sheet that looks like a registration fault.
# ---------------------------------------------------------------------------

#: One layer at whole-cell offsets: the case backward printing is *for*. Anything
#: with a half-cell horizontal offset puts consecutive strikes half a cell apart,
#: which is not the move 0x8E makes.
WHOLE_CELL_LAYERS = [(0, 0)]


def _ops(job) -> list[int]:
    return [op for _, op, _ in etp.iter_ops(job.body)]


def _count(job, want) -> int:
    return sum(1 for op in _ops(job) if op == want)


def _serpentine_plan(tmp_path, charset, layers=None, seed=17, density=0.95,
                     rows=6, cols=14):
    path = _write_choices(
        tmp_path,
        _random_choices(charset, rows, cols, layers or WHOLE_CELL_LAYERS,
                        seed=seed, density=density),
    )
    return planner.build_plan(path, charset, home_each_row=False)


def test_a_reverse_pass_types_backwards_instead_of_backspacing(tmp_path, charset):
    plan = _serpentine_plan(tmp_path, charset)
    backwards = planner.encode(plan)
    sweeping = planner.encode(plan, backward=False)

    assert _count(backwards, etp.OP_BACKWARD_ON) > 0
    assert _count(sweeping, etp.OP_BACKWARD_ON) == 0
    # The backspaces are what it replaces, and the wire is where the saving is:
    # a byte the machine has to digest costs a character delay whatever it says.
    assert _count(backwards, etp.OP_LEFT) < _count(sweeping, etp.OP_LEFT)
    assert len(emulate.expand(backwards.body)) < len(
        emulate.expand(sweeping.body)
    )


def test_every_backward_strike_lands_where_the_plan_said(tmp_path, charset):
    """The property everything else rests on, checked the way the rest of the
    pipeline checks itself: run the opcodes through the virtual machine and
    compare every impression against the plan.

    Several densities, because what a run *is* depends on where the gaps fall --
    a dense row is one long run and a sparse one is a scatter of short ones with
    ordinary carriage moves between them."""
    for density in (1.0, 0.8, 0.5, 0.2):
        plan = _serpentine_plan(tmp_path, charset, density=density, seed=23)
        job = planner.encode(plan)
        machine = emulate.type_job(job, max_columns=charset.max_columns)
        assert machine.overruns == 0
        assert [(i.y, i.x, i.code) for i in machine.impressions] == [
            (s.y, s.x, charset.codes[s.index]) for s in plan.strikes
        ], f"density {density}"


def test_a_backward_run_costs_one_byte_a_cell_and_a_sweep_three(tmp_path, charset):
    """The arithmetic behind MIN_BACKWARD_RUN, on the wire rather than on paper.

    A full row of one layer is one run from end to end, so the reverse pass is
    exactly its glyphs plus the two mode switches -- against a glyph and two
    backspaces per cell the other way."""
    cols = 10
    grid = [[3] * cols, [4] * cols]
    path = _write_choices(tmp_path, {"layer0_0_0": grid})
    plan = planner.build_plan(path, charset, home_each_row=False)

    reverse_pass = [s for s in plan.strikes if s.y == 2]
    assert [s.x for s in reverse_pass] == sorted(
        (s.x for s in reverse_pass), reverse=True
    ), "the second pass should sweep right to left"

    backwards = len(emulate.expand(planner.encode(plan).body))
    sweeping = len(emulate.expand(planner.encode(plan, backward=False).body))
    # The pass above ends with the escapement one cell right of its last strike,
    # which is exactly where a backward run wants to begin -- so the run arrives
    # for nothing and costs one byte at each end plus one per cell: cols + 2.
    # The sweep pays a backspace to get onto the first cell, a byte for it, and
    # three for every cell after: 3 * cols - 1.
    assert sweeping - backwards == (3 * cols - 1) - (cols + 2)


def test_nothing_but_a_strike_goes_between_the_two_mode_switches(tmp_path, charset):
    """The invariant that keeps the plan inside what the sheet measured.

    A motion, a Doppeldruck, a force change or a paper feed in there would each
    be an assumption about a mode nobody has asked about, and none of them would
    look like a mode problem on the sheet -- they would look like the carriage
    slipping."""
    for layers, density in (
        (WHOLE_CELL_LAYERS, 1.0),
        (WHOLE_CELL_LAYERS, 0.5),
        ([(0, 0)] * 3, 0.9),          # every cell a stack
        (FOUR_LAYERS, 0.9),           # half-cell offsets across
        ([(0, 0), (0.5, 0)], 0.7),    # two paper positions per row
    ):
        plan = _serpentine_plan(tmp_path, charset, layers=layers, density=density)
        inside = False
        for op in _ops(planner.encode(plan)):
            if op == etp.OP_BACKWARD_ON:
                assert not inside, "backward printing turned on twice"
                inside = True
            elif op == etp.OP_BACKWARD_OFF:
                inside = False
            elif inside:
                assert op == etp.OP_STRIKE, (
                    f"{etp.OPCODE_NAMES[op]} while typing backwards, and no sheet "
                    "has said what the machine does with it"
                )


def test_backward_printing_is_always_turned_off_again(tmp_path, charset):
    """It is a mode, and a mode outlives the job. A plan that ends in it would
    leave the next thing typed on the machine -- by the firmware's chatbot, or
    by hand -- running right to left."""
    plan = _serpentine_plan(tmp_path, charset, density=1.0)
    ops = _ops(planner.encode(plan))
    assert ops.count(etp.OP_BACKWARD_ON) == ops.count(etp.OP_BACKWARD_OFF) > 0
    assert ops.index(etp.OP_BACKWARD_ON) < ops.index(etp.OP_BACKWARD_OFF)
    last_switch = max(i for i, op in enumerate(ops)
                      if op in (etp.OP_BACKWARD_ON, etp.OP_BACKWARD_OFF))
    assert ops[last_switch] == etp.OP_BACKWARD_OFF
    assert etp.OP_END in ops[last_switch:]


def test_a_run_too_short_to_pay_stays_on_the_backspaces(tmp_path, charset):
    """Two cells is a byte at best and a tie at worst, so the mechanism that has
    been on paper longest keeps it -- the same rule the firmware's
    stepsAreWorthIt follows."""
    # A reverse pass of exactly two adjacent cells, and one of exactly three.
    cases = {"two": [[1, 1, 1], [0, 1, 1]], "three": [[1, 1, 1], [1, 1, 1]]}
    paths = {}
    for name, grid in cases.items():
        (tmp_path / name).mkdir()
        paths[name] = _write_choices(tmp_path / name, {"layer0_0_0": grid})
    for path, expected in ((paths["two"], 0), (paths["three"], 1)):
        plan = planner.build_plan(path, charset, home_each_row=False)
        assert _count(planner.encode(plan), etp.OP_BACKWARD_ON) == expected


def test_a_stack_is_never_typed_backwards(tmp_path, charset):
    """0xA9 means "print where the head stands", and in a mode whose whole
    content is which way the head goes before it prints, nobody has asked what
    that means. A cell with two glyphs in it therefore ends a run."""
    every_cell_stacked = _write_choices(
        tmp_path, {"layer0_0_0": [[5] * 8] * 2, "layer1_0_0": [[6] * 8] * 2}
    )
    plan = planner.build_plan(every_cell_stacked, charset, home_each_row=False)
    job = planner.encode(plan)
    assert _count(job, etp.OP_NO_ADVANCE) > 0
    assert _count(job, etp.OP_BACKWARD_ON) == 0
    # And it is still typed correctly, on the backspaces.
    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert [(i.y, i.x) for i in machine.impressions] == [
        (s.y, s.x) for s in plan.strikes
    ]


def test_a_run_that_begins_where_a_stack_ended_still_lands(tmp_path, charset):
    """The one adjacency worth constructing by hand.

    A cell with two glyphs in it is typed with 0xA9, which leaves the head
    standing *on* the cell rather than one past it -- and a backward run wants to
    begin one cell to the right of its first mark. So the run has to step right
    before it turns the mode on, which reads like the wrong direction and is the
    right one. Row 0 is only there to be the forward pass, so that row 1 is the
    reversed one: 10, 10, 8, 6, 4, a stack and then a run.
    """
    path = _write_choices(tmp_path, {
        "layer0_0_0": [[2, 2, 0, 0, 0, 0], [0, 0, 3, 4, 5, 6]],
        "layer1_0_0": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7]],
    })
    plan = planner.build_plan(path, charset, home_each_row=False)
    assert [s.x for s in plan.strikes] == [0, 2, 10, 10, 8, 6, 4]

    job = planner.encode(plan)
    assert _count(job, etp.OP_NO_ADVANCE) == 1
    assert _count(job, etp.OP_BACKWARD_ON) == 1
    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert [(i.y, i.x, i.code) for i in machine.impressions] == [
        (s.y, s.x, charset.codes[s.index]) for s in plan.strikes
    ]


def test_a_dead_key_is_never_typed_backwards(tmp_path):
    """It does not feed at all, and "does not feed" in a mode that is entirely
    about the direction of the feed is the least answerable of the three."""
    dead = ec.DEAD_KEY_GLYPHS[0]
    cs = Charset(
        name="test-dead", pitch=10, cell_w=24, cell_h=40, max_columns=65,
        codes=[ec.SPACE, ec.glyph_for_char("A").code, dead.code],
        advances=[True, True, False],
        chars=[" ", "A", dead.char],
    )
    # A reverse pass of five cells with a dead key in the middle of it.
    path = _write_choices(tmp_path, {"layer0_0_0": [[1] * 5,
                                                    [1, 1, 2, 1, 1]]})
    plan = planner.build_plan(path, cs, home_each_row=False)
    job = planner.encode(plan)
    inside = False
    for _, op, operand in etp.iter_ops(job.body):
        if op == etp.OP_BACKWARD_ON:
            inside = True
        elif op == etp.OP_BACKWARD_OFF:
            inside = False
        elif inside:
            assert operand != dead.code
    machine = emulate.type_job(job, max_columns=cs.max_columns)
    assert [(i.y, i.x) for i in machine.impressions] == [
        (s.y, s.x) for s in plan.strikes
    ]


def test_a_force_change_ends_a_run(tmp_path):
    """Two bytes on the wire, which is the whole saving for two cells -- and it
    would sit inside a mode the machine has only been seen to hold across
    characters."""
    cs = _force_charset()
    # One row, alternating forces, typed with grouping off so the changes land
    # inside the pass rather than being sorted out of it.
    path = _write_choices(tmp_path, {"layer0_0_0": [[1] * 6, [1, 3, 1, 3, 1, 3]]})
    plan = planner.build_plan(path, cs, home_each_row=False, group_by_force=False)
    inside = False
    for op in _ops(planner.encode(plan)):
        if op == etp.OP_BACKWARD_ON:
            inside = True
        elif op == etp.OP_BACKWARD_OFF:
            inside = False
        elif inside:
            assert op != etp.OP_SET_FORCE


def test_a_plan_that_returns_the_carriage_never_types_backwards(tmp_path, charset):
    """Every pass runs left to right, so there is nothing to reverse -- and the
    byte stream has to be exactly what it was before any of this existed."""
    path = _write_choices(tmp_path, _random_choices(charset, 5, 9, FOUR_LAYERS,
                                                    seed=31))
    plan = planner.build_plan(path, charset, home_each_row=True)
    assert planner.encode(plan).body == planner.encode(plan, backward=False).body
    assert _count(planner.encode(plan), etp.OP_BACKWARD_ON) == 0


def test_the_opt_out_puts_the_backspaces_back(tmp_path, charset):
    """--backspace-sweep, for the sheet that comes out wrong and the question of
    which mechanism did it."""
    plan = _serpentine_plan(tmp_path, charset)
    job = planner.encode(plan, backward=False)
    assert _count(job, etp.OP_BACKWARD_ON) == 0
    assert _count(job, etp.OP_LEFT) > 0
    machine = emulate.type_job(job, max_columns=charset.max_columns)
    assert [(i.y, i.x) for i in machine.impressions] == [
        (s.y, s.x) for s in plan.strikes
    ]


def test_the_emulator_refuses_what_the_sheet_did_not_ask(tmp_path):
    """The model stops where the measurement stopped. A guess here would be a
    guess every test that runs a plan through this then certifies."""
    def wire(build):
        enc = etp.Encoder()
        enc.backward_on()
        build(enc)
        enc.backward_off()
        return emulate.expand(enc.body())

    stacked = wire(lambda e: (e.no_advance(),
                              e.strike(ec.glyph_for_char("A").code),
                              e.strike(ec.glyph_for_char("B").code)))
    with pytest.raises(emulate.EmulationError, match="0xA9"):
        emulate.Typewriter().run(stacked)

    dead = ec.DEAD_KEY_GLYPHS[0]
    with pytest.raises(emulate.EmulationError, match="dead key"):
        emulate.Typewriter().run(wire(lambda e: e.strike(dead.code,
                                                         advances=False)))

    with pytest.raises(emulate.EmulationError, match="motion keys"):
        emulate.Typewriter().run(wire(lambda e: e.left(2)))

    with pytest.raises(emulate.EmulationError, match="carriage steps"):
        emulate.Typewriter().run(wire(lambda e: e.carriage_steps(6)))

    # ...and the one it did ask is modelled, from the sheet's own section 7:
    # five letters from column 20 read EDCBA with the A at column 19.
    enc = etp.Encoder()
    enc.right(2 * 20)
    enc.backward_on()
    for char in "ABCDE":
        enc.strike(ec.glyph_for_char(char).code)
    enc.backward_off()
    machine = emulate.Typewriter().run(emulate.expand(enc.body()))
    assert [(i.x // 2, ec.glyph_for_code(i.code).char)
            for i in machine.impressions] == [
        (19, "A"), (18, "B"), (17, "C"), (16, "D"), (15, "E"),
    ]


def test_the_disassembly_follows_the_head_backwards(tmp_path, charset):
    """The columns are what a plan is checked against without a typewriter, so
    they have to run the same way the paper does."""
    plan = _serpentine_plan(tmp_path, charset, density=1.0)
    text = etp.disassemble(planner.encode(plan))
    assert "BACKWARD_ON" in text and "BACKWARD_OFF" in text
    # The strikes between the two switches walk leftwards, one cell at a time.
    cols, inside = [], False
    for line in text.splitlines():
        if "BACKWARD_ON" in line:
            inside = True
        elif "BACKWARD_OFF" in line:
            break
        elif inside and " STRIKE " in line:
            cols.append(float(line.split()[1]))
    assert len(cols) >= planner.MIN_BACKWARD_RUN
    assert all(b - a == -1.0 for a, b in zip(cols, cols[1:])), cols


def test_the_plan_summary_says_when_it_typed_backwards(tmp_path, charset):
    plan = _serpentine_plan(tmp_path, charset, density=1.0)
    assert "right to left" in planner.summarize(plan, planner.encode(plan))
    plain = planner.encode(plan, backward=False)
    assert "right to left" not in planner.summarize(plan, plain)


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_firmware_turns_backward_printing_off_on_every_exit():
    """The same property the postamble has, for the same reason: 0x8E is state
    on the machine, and a job that stops between it and 0x8D leaves the operator
    with a typewriter that types right to left and nothing to say why. A failure
    is exactly the path somebody would forget."""
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.cpp"),
                encoding="utf-8").read()
    for func in ("abort", "fail", "finish"):
        body = re.search(rf"void ErikaImagePrinter::{func}\(.*?\n}}", text, re.S)
        assert body, f"could not find {func}()"
        assert "flushBackwardPrint()" in body.group(0), (
            f"{func}() can leave the machine printing right to left"
        )
    # And it is the OFF code that gets sent, not the ON one again.
    flush = re.search(r"void ErikaImagePrinter::flushBackwardPrint\(.*?\n}",
                      text, re.S)
    assert flush and "ERIKA_BACKWARD_PRINT_OFF" in flush.group(0)


# ---------------------------------------------------------------------------
# offsets finer than a keystroke (planner --fine)
# ---------------------------------------------------------------------------

QUARTER_LAYERS = [(0, 0), (0, 0.25), (0.25, 0), (0.25, 0.25)]


def test_refusing_a_fine_offset_says_which_switch_did_it(tmp_path, charset):
    """The refusal is now a consequence of a flag rather than of the machine, so
    it has to name the flag -- otherwise it reads as "your scheme is impossible",
    which it was until the control-code sheet came back and is not any more."""
    path = _write_choices(tmp_path, {"layer0_0.25_0": [[1, 2]]})
    with pytest.raises(PlanError, match="turned it off"):
        planner.build_plan(path, charset, fine=False)


@pytest.mark.parametrize("pitch,ok", [("sigma-10", True), ("sigma-12", False)])
def test_a_quarter_cell_is_whole_carriage_steps_at_pitch_10_and_not_at_12(
    tmp_path, pitch, ok
):
    """Three steps at pitch 10, two and a half at pitch 12. Half a motor step
    does not exist, so the same layer scheme is typeable at one pitch and not at
    the other -- which the error has to say rather than blaming the scheme."""
    cs = Charset.load(pitch, SRC)
    path = _write_choices(tmp_path, _random_choices(cs, 4, 6, QUARTER_LAYERS, seed=2))
    if ok:
        plan = planner.build_plan(path, cs, fine=True)
        assert {s.fx for s in plan.strikes} == {0, 3}
    else:
        with pytest.raises(PlanError, match="not a whole number of"):
            planner.build_plan(path, cs, fine=True)


def test_a_quarter_line_is_ten_platen_steps_at_any_pitch(tmp_path, charset):
    """Vertical is the easy half: a line is 40 platen steps whatever the pitch,
    so every offset built from halves, quarters, fifths or eighths is exact."""
    for offset, steps in ((0.25, 10), (0.2, 8), (0.125, 5), (0.75, 30)):
        half, residue = planner.offset_to_units(offset, "vertical", 10, fine=True)
        assert half * ec.PLATEN_STEPS_PER_HALF_LINE + residue == steps


def test_daisy_full_becomes_typeable_at_pitch_15():
    """The scheme the planner used to call physically unrealisable. Its eighth-
    cell offsets are one carriage step each at 15 characters per inch, and its
    fifth-line offsets are eight platen steps -- both exact, neither reachable
    by any key on the machine."""
    for offset in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        planner.offset_to_units(offset, "horizontal", 15, fine=True)
    for offset in (0.2, 0.4, 0.6, 0.8):
        planner.offset_to_units(offset, "vertical", 15, fine=True)
    # And not at pitch 10, where an eighth of a cell is a step and a half.
    with pytest.raises(PlanError, match="not a whole number of"):
        planner.offset_to_units(0.125, "horizontal", 10, fine=True)


def test_a_fine_plan_lands_a_quarter_cell_from_the_grid(tmp_path, charset):
    """End to end through the virtual machine: the residue survives the encoder,
    the firmware's expansion and the machine's own arithmetic, and comes back
    out as a mark a quarter of a cell off the half-cell grid."""
    path = _write_choices(tmp_path, {"layer0_0_0": [[1]], "layer1_0_0.25": [[1]]})
    plan = planner.build_plan(path, charset, fine=True)
    job = planner.encode(plan)
    machine = emulate.type_job(job, max_columns=charset.max_columns)

    places = [(i.y, i.fy, i.x, i.fx) for i in machine.impressions]
    assert places == [(0, 0, 0, 0), (0, 0, 0, 3)]
    assert machine.overruns == 0


def test_a_half_cell_plan_is_byte_identical_with_fine_switched_on(tmp_path,
                                                                 charset):
    """The property that makes --fine safe to have. Nothing about a scheme built
    from halves reaches the new opcodes, so turning the flag on cannot change a
    picture that never needed it."""
    path = _write_choices(tmp_path, _random_choices(charset, 6, 9, FOUR_LAYERS,
                                                    seed=17))
    plain = planner.encode(planner.build_plan(path, charset))
    fine = planner.encode(planner.build_plan(path, charset, fine=True))
    assert fine.body == plain.body


def test_a_fine_plan_still_only_feeds_the_paper_forward(tmp_path, charset):
    """The residue is part of the position, so it has to be part of the sort --
    otherwise a quarter-line layer is typed before the layer above it and the
    platen has to come back, which is where banding comes from."""
    path = _write_choices(tmp_path, _random_choices(charset, 5, 7, QUARTER_LAYERS,
                                                    seed=4))
    job = planner.encode(planner.build_plan(path, charset, fine=True))
    ops = [op for _, op, _ in etp.iter_ops(job.body)]
    assert etp.OP_UP not in ops
    assert etp.OP_UP_FINE not in ops
    assert etp.OP_MICRO_UP not in ops


def test_the_preview_places_a_fine_strike_off_the_grid(tmp_path, charset):
    """Or the offline check against the optimizer's mockup would be comparing a
    quarter-cell plan against a half-cell render and calling it exact."""
    import numpy as np
    from erika import preview

    tiles = np.ones((len(charset) + 1, charset.cell_h, charset.cell_w),
                    dtype=np.float32)
    tiles[1] = 0.0  # index 1 lays down solid ink, so its position is visible

    def inked_columns(offset):
        path = _write_choices(tmp_path, {f"layer0_0_{offset}": [[1]]})
        plan = planner.build_plan(path, charset, fine=True)
        img = preview.render(plan, tiles)
        return np.flatnonzero((img < 0.5).any(axis=0))

    at_zero = inked_columns(0)
    at_quarter = inked_columns(0.25)
    assert at_quarter[0] - at_zero[0] == charset.cell_w // 4


def test_the_listed_typeable_schemes_are_the_ones_the_planner_accepts():
    """TYPEABLE_LAYER_SCHEMES is written out so that importing pipeline does not
    read layers.json. That makes it a copy, and a copy needs a comparison."""
    from erika import pipeline

    assert pipeline.typeable_layer_schemes(10) == pipeline.TYPEABLE_LAYER_SCHEMES
    assert pipeline.typeable_layer_schemes(12) == pipeline.TYPEABLE_LAYER_SCHEMES


def test_which_schemes_fine_unlocks_depends_on_the_pitch():
    """The reason typeable_layer_schemes takes a pitch at all.

    A quarter of a cell is three carriage steps at pitch 10 and two at pitch 15,
    but two and a half at pitch 12 -- so pitch 12 gains nothing from --fine. An
    eighth is one step only at pitch 15, which is the pitch the slide switch does
    not offer and part 9 of the probe sheet asks about.
    """
    from erika import pipeline

    at10 = set(pipeline.typeable_layer_schemes(10, fine=True))
    at12 = set(pipeline.typeable_layer_schemes(12, fine=True))
    at15 = set(pipeline.typeable_layer_schemes(15, fine=True))
    keystroke = set(pipeline.TYPEABLE_LAYER_SCHEMES)

    assert at12 == keystroke  # nothing, and the flag should not pretend otherwise
    assert "16x1" in at10 - keystroke
    assert {"daisy_full", "daisy_x2"} <= at15 - at10


COARSE_V = (etp.OP_DOWN, etp.OP_UP, etp.OP_NEWLINE, etp.OP_MICRO_DOWN, etp.OP_MICRO_UP)
FINE_V = (etp.OP_DOWN_FINE, etp.OP_UP_FINE)
COARSE_H = (etp.OP_RIGHT, etp.OP_LEFT)
FINE_H = (etp.OP_RIGHT_FINE, etp.OP_LEFT_FINE)


@pytest.mark.parametrize("home_each_row", [True, False])
def test_a_move_never_changes_mechanism_part_way_through(tmp_path, charset,
                                                         home_each_row):
    """What part 5 of the control-code sheet showed, on paper.

    It feeds the platen five steps at a time, eight times; every gap came out
    equal except the first, which is the one that follows the detented line feed
    ending the section heading. A motor-step feed straight after the detent
    appears to lose a step or two taking up the detent -- invisible in part 4,
    which asks for forty steps from the same starting condition, and a fifth of
    the gap here.

    So a move is either all keystrokes or all motor steps, never a keystroke run
    with the remainder appended. This is the check that it stays that way.
    """
    path = _write_choices(tmp_path, _random_choices(charset, 6, 9, QUARTER_LAYERS,
                                                    seed=21))
    job = planner.encode(
        planner.build_plan(path, charset, home_each_row=home_each_row, fine=True)
    )
    ops = [op for _, op, _ in etp.iter_ops(job.body)]
    for a, b in zip(ops, ops[1:]):
        assert not (a in COARSE_V and b in FINE_V), "coarse feed then a fine one"
        assert not (a in FINE_V and b in COARSE_V), "fine feed then a coarse one"
        assert not (a in COARSE_H and b in FINE_H), "coarse move then a fine one"
        assert not (a in FINE_H and b in COARSE_H), "fine move then a coarse one"


def test_one_mechanism_per_move_is_also_the_shorter_stream(tmp_path, charset):
    """The agreement worth noticing: the remainder already costs an opcode and an
    operand, so folding the whole move into it removes one rather than adding."""
    path = _write_choices(tmp_path, _random_choices(charset, 6, 9, QUARTER_LAYERS,
                                                    seed=22))
    plan = planner.build_plan(path, charset, fine=True)
    job = planner.encode(plan)

    # What the two-mechanism form would have cost, counted from the same plan.
    split = etp.Encoder()
    x = y = x_fine = y_fine = 0
    per_half = ec.carriage_steps_per_half_step(charset.pitch)
    for s in plan.strikes:
        dv = ((s.y * ec.PLATEN_STEPS_PER_HALF_LINE + s.fy)
              - (y * ec.PLATEN_STEPS_PER_HALF_LINE + y_fine))
        whole, rest = planner._split(dv, ec.PLATEN_STEPS_PER_HALF_LINE)
        split.vertical(whole)
        split.vertical_fine(rest)
        dh = (s.x * per_half + s.fx) - (x * per_half + x_fine)
        whole, rest = planner._split(dh, per_half)
        split.horizontal(whole)
        split.horizontal_fine(rest)
        split.strike(charset.codes[s.index], charset.advances[s.index])
        y, y_fine, x, x_fine = s.y, s.fy, s.x, s.fx
        if charset.advances[s.index]:
            x += 2
    assert len(job.body) < len(split.body())


def test_a_half_cell_plan_still_uses_the_detented_mechanism(tmp_path, charset):
    """The change is for moves that need motor steps. A plan built from halves
    has none, and should still drive the line feed the calibration sheet
    measured -- which is the more repeatable of the two when it can be used."""
    path = _write_choices(tmp_path, _random_choices(charset, 5, 8, FOUR_LAYERS,
                                                    seed=23))
    job = planner.encode(planner.build_plan(path, charset, fine=True))
    ops = [op for _, op, _ in etp.iter_ops(job.body)]
    assert etp.OP_NEWLINE in ops
    assert not [op for op in ops if op in FINE_V + FINE_H]


def test_a_serpentine_pass_is_a_paper_position_not_a_row(tmp_path, charset):
    """A layer offset by a quarter of a line is its own sweep.

    _serpentine grouped by y alone, which was right while y was the whole of the
    paper position. With a residue under it, grouping by y folds two sweeps into
    one and reversing them puts the quarter-line strikes before the ones above
    them -- so the platen has to come back, which encode() refuses outright.
    """
    path = _write_choices(tmp_path, _random_choices(charset, 5, 8, QUARTER_LAYERS,
                                                    seed=24))
    plan = planner.build_plan(path, charset, home_each_row=False, fine=True)
    positions = [(s.y, s.fy) for s in plan.strikes]
    assert positions == sorted(positions), "a sweep was reversed into another one"
    # And it still alternates, which is the whole point of the serpentine.
    sweeps = []
    for s in plan.strikes:
        if not sweeps or (s.y, s.fy) != sweeps[-1][0]:
            sweeps.append(((s.y, s.fy), []))
        sweeps[-1][1].append(s.x)
    directions = {tuple(xs) == tuple(sorted(xs)) for _, xs in sweeps if len(xs) > 1}
    assert directions == {True, False}


# ---------------------------------------------------------------------------
# the defaults, now that the control-code sheet has been printed
# ---------------------------------------------------------------------------


def test_the_machines_own_mechanisms_are_the_default():
    """Every one of these was off until the sheet came back positive, and each is
    off again the moment somebody changes a default without meaning to. Cheap to
    pin, and the failure it prevents is a picture that is quietly worse."""
    import inspect

    from erika import pipeline

    assert inspect.signature(planner.build_plan).parameters["fine"].default is True
    assert inspect.signature(planner.encode).parameters["no_advance"].default is True

    args = pipeline.build_parser().parse_args(["print"])
    assert args.fine is True
    assert args.no_advance is True


def test_the_opt_outs_still_work_and_name_a_mechanism():
    """They exist for one purpose: a sheet came out wrong and the question is
    which mechanism did it. So each has to actually go back to the old one."""
    from erika import pipeline

    args = pipeline.build_parser().parse_args(["print", "--keystrokes-only"])
    assert args.fine is False
    args = pipeline.build_parser().parse_args(["print", "--backspace-overstrike"])
    assert args.no_advance is False


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_the_firmware_defaults_to_using_the_step_commands():
    """AUTO rather than ALL: the commands are cheaper on a blank run and dearer
    on the one-cell hops a picture is mostly made of, which was measured. And
    not the completion report, which no printed sheet can confirm.

    The pitch pins are the other way round -- on at boot -- and the difference is
    not how confirmed they are but what the two failures cost. An unconfirmed
    delay model comes out as a smudged sheet; an unpinned pitch comes out as a
    plan for 10 characters per inch typed at 12, and nothing on either side can
    notice, because no code in the interface's table reports a slide switch.
    See test_a_job_pins_its_pitch_unless_the_operator_says_not_to.
    """
    text = open(os.path.join(FIRMWARE_SRC, "erika_image.h"), encoding="utf-8").read()
    assert re.search(r"DirectSteps _directSteps = StepsAuto;", text)
    assert re.search(r"bool _completionReport = false;", text)
    assert re.search(r"Prepare _prepare = PreparePitch;", text)


def test_the_probe_sheet_asks_whether_the_type_fits_the_pitch(tmp_path):
    """Section 9 confirms the escapement and is structurally blind to the wheel.

    Its comb is typed with RULER_CHAR, which is chosen for being a narrow
    vertical mark -- and a narrow mark is exactly the one whose spacing can
    shrink without the glyphs touching. Section 10 is the one that can tell,
    and it can only tell if it uses a wide glyph and gives the reader something
    to compare: the same groups at the working pitch, and the wide glyph spaced
    out so an isolated one is visible next to a packed one.
    """
    from erika.pipeline import RULER_CHAR

    job = _codes_sheet(tmp_path)
    machine = emulate.type_job(job)
    wide = ec.glyph_for_char("M").code

    rows = {}
    for imp in machine.impressions:
        if imp.code == wide:
            rows.setdefault(imp.y, []).append(imp.x)
    # Evenly spaced runs only: section 8's correction row also has fifteen M on
    # it, in two groups with a gap, which is not a run of anything.
    runs = {
        y: sorted(xs)
        for y, xs in rows.items()
        if len(xs) == 15 and len({b - a for a, b in zip(sorted(xs), sorted(xs)[1:])}) == 1
    }
    assert len(runs) == 3, "expected a reference row, a pitch-15 row and a spaced one"

    packed = [xs for xs in runs.values() if {b - a for a, b in zip(xs, xs[1:])} == {2}]
    spaced = [xs for xs in runs.values() if {b - a for a, b in zip(xs, xs[1:])} == {4}]
    assert len(packed) == 2, "two rows must be packed, or there is nothing to compare"
    assert len(spaced) == 1, "one row must be spaced, or overlap cannot be told apart"

    # The thin glyph is the control and has to be on the sheet beside the wide one.
    assert RULER_CHAR != "M"
    thin = ec.glyph_for_char(RULER_CHAR).code
    thin_rows = {i.y for i in machine.impressions if i.code == thin}
    assert thin_rows & set(runs), "the control never shares a row with the wide glyph"


def test_the_pitch_is_restored_after_every_section_that_changes_it(tmp_path):
    """Two sections change the pitch now, and a pitch that sticks rescales
    everything after it -- including the other section's own reference row."""
    job = _codes_sheet(tmp_path)
    raw = [operand for _, op, operand in etp.iter_ops(job.body) if op == etp.OP_RAW]
    switches = [b for b in raw if b in (0x87, 0x88, 0x89)]
    assert switches.count(0x89) == 2
    # Strictly alternating, so nothing is left at 15.
    assert switches[0::2] == [0x89, 0x89]
    assert switches[1::2] == [0x87, 0x87]


# ---------------------------------------------------------------------------
# the beeper (erika.melody)
# ---------------------------------------------------------------------------


def _device_timeline(body: bytes) -> list[int]:
    """When each byte actually leaves the device, by the firmware's own rules.

    The model `erika.melody` computes its delays against, written out
    independently here so that the two have to agree: a byte cannot go out
    until a character delay has passed since the last one *and* until any
    standing ETP_DELAY has expired, and a delay begins when its opcode is
    decoded -- which is as soon as the interpreter is free, not when the next
    byte is due. The two overlap rather than add, and that is exactly the
    assumption `compile_to` subtracts a character delay on.
    """
    from erika import melody as mel

    sent: list[int] = []
    last, until = None, 0
    for _, op, operand in etp.iter_ops(body):
        if op == etp.OP_RAW:
            at = max(0 if last is None else last + mel.RAW_BYTE_COST_MS, until)
            sent.append(at)
            last = at
        elif op == etp.OP_DELAY:
            until = max(0 if last is None else last, until) + operand * 10
    return sent


def test_the_beeper_plays_the_rhythm_it_was_asked_for():
    """The load-bearing claim of the whole module: onset to onset on the wire
    is the slot the melody asked for.

    Nothing else in this pipeline can be checked after the fact -- a print job
    has a mockup to diff against and a tune leaves no trace -- so the timing
    model is only as good as this test.
    """
    from erika import melody as mel

    melody = mel.parse("q e e h -q q", tempo=100)
    enc = etp.Encoder()
    mel.compile_to(enc, melody)
    onsets = _device_timeline(enc.body())
    # Two bytes per beep, and the first of each pair is where a slot begins.
    starts = onsets[::2]
    intended, t = [], 0
    for e in melody.events:
        if not e.is_rest:
            intended.append(t)
        t += e.slot_ms
    assert len(starts) == len(intended) == melody.beeps
    for got, want in zip(starts, intended):
        # One rounding of one delay to the encoder's 10 ms grid, per note.
        assert abs(got - want) <= 5 * len(intended), (got, want)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= mel.MIN_SLOT_MS for g in gaps), gaps


def test_a_note_the_device_cannot_deliver_is_refused_rather_than_stretched():
    """Rounding a too-short note up to the floor would keep the job playable
    and silently change the rhythm, which is the one failure a tune cannot
    survive -- every note after it lands somewhere else."""
    from erika import melody as mel

    with pytest.raises(mel.MelodyError) as exc:
        mel.to_job(mel.parse("s s s s", tempo=120))
    assert "200" in str(exc.value) and "150" in str(exc.value)
    # And the same rhythm slowed down is fine.
    assert mel.to_job(mel.parse("s s s s", tempo=70)).body


def test_every_built_in_tune_is_playable_as_written():
    from erika import melody as mel

    for name in mel.TUNES:
        melody = mel.tune(name)
        assert mel.check(melody) == [], f"{name}: {mel.check(melody)}"
        assert mel.to_job(melody).body


def test_notes_are_separated_by_silence_because_nothing_else_separates_them():
    """One pitch means two adjacent notes of the same length are one long note
    unless there is a gap. The gate is what leaves it, so a tune shipped with a
    gate that closes it would be a bug nobody could see in the score."""
    from erika import melody as mel

    for name in mel.TUNES:
        for e in mel.tune(name).events:
            if not e.is_rest:
                assert e.gap_ms >= mel.MIN_GAP_MS, (name, e)


def test_a_melody_reaches_the_paper_not_at_all():
    from erika import melody as mel

    machine = emulate.type_job(mel.to_job(mel.tune("shave")))
    assert machine.impressions == []
    assert machine.overruns == 0
    assert {code for code, _ in machine.probes} == {ec.BELL}


def test_a_note_longer_than_the_bell_is_clamped_not_wrapped():
    """The sweep found the operand to be a plain unsigned length, so the cap is
    now the whole byte -- but it is still a cap, and a note past it has to come
    out as the longest beep the machine has rather than as a short one."""
    from erika import melody as mel

    assert mel.units_for(10_000) == ec.MAX_BELL_UNITS <= 255
    assert mel.units_for(1) == 1  # and never quantised out of existence
    long_note = mel.parse("w", tempo=20)  # 12 s, well past the cap
    operands = [operand for _, op, operand in etp.iter_ops(mel.to_job(long_note).body)
                if op == etp.OP_RAW]
    assert max(operands[1::2]) <= ec.MAX_BELL_UNITS
    assert any("goes quiet early" in p for p in mel.check(long_note))


def test_the_probe_sweeps_the_whole_operand_and_never_repeats_a_length():
    """What the sweep is for, now that it has answered the question it was
    built to ask.

    It reaches the top of the byte -- that is how the cap got there -- and every
    beep differs from the one before it, because the reading is "each longer
    than the last" and two equal lengths in the middle would make that judgement
    on a machine nobody can pause. It also has to stay outside `units_for`,
    which clamps; a probe that clamped would agree with itself no matter what
    the constants said.
    """
    from erika import melody as mel

    operands = [operand for _, op, operand in etp.iter_ops(mel.probe_job().body)
                if op == etp.OP_RAW]
    assert operands[::2] == [ec.BELL] * len(mel.PROBE_UNITS)
    lengths = operands[1::2]
    assert lengths == list(mel.PROBE_UNITS)
    assert max(lengths) == ec.MAX_BELL_UNITS  # the whole range, to its top
    assert all(b > a for a, b in zip(lengths, lengths[1:]))


def test_morse_survives_the_round_trip():
    """Decoding the compiled rhythm back to dots and dashes, which is the only
    way to check that the gaps land between the right symbols."""
    from erika import melody as mel

    melody = mel.morse("SOS ERIKA")
    unit = mel.DEFAULT_MORSE_UNIT_MS
    text = ""
    for e in melody.events:
        if e.is_rest:
            text += " " if e.slot_ms >= 4 * unit else "/"
        else:
            text += "-" if e.sound_ms == 3 * unit else "."
            if e.gap_ms > unit:
                text += "/"
    letters = ["".join(sym for sym in word.split("/") if sym)
               for word in text.split(" ")]
    assert " ".join(letters) == "".join(
        mel.MORSE[c] for c in "SOS"
    ) + " " + "".join(mel.MORSE[c] for c in "ERIKA")


def test_morse_refuses_a_unit_the_device_cannot_key():
    from erika import melody as mel

    with pytest.raises(mel.MelodyError):
        mel.morse("E", unit_ms=40)


def test_the_notation_says_what_it_could_not_read():
    from erika import melody as mel

    with pytest.raises(mel.MelodyError) as exc:
        mel.parse("q x q")
    assert "'x'" in str(exc.value) or "x" in str(exc.value)
    # A dot is one and a half, two dots one and three quarters -- not two.
    quarter = mel.quarter_ms(100)
    assert mel.parse("q.", tempo=100).events[0].slot_ms == round(quarter * 1.5)
    assert mel.parse("q..", tempo=100).events[0].slot_ms == round(quarter * 1.75)


@pytest.mark.skipif(NO_FIRMWARE, reason=NO_FIRMWARE_REASON)
def test_firmware_character_delay_matches_the_melody_timing_model():
    """`melody.RAW_BYTE_COST_MS` is a hand-copy of the firmware's own pacing,
    and the whole tempo ceiling rests on it.

    A drift here is quiet in the way this workspace warns about: the job still
    plays, every byte is still legal, and the rhythm is simply wrong -- slower
    if the firmware's delay grew, and running its notes together if it shrank.
    """
    from erika import melody as mel

    defines = _parse_cpp_defines(os.path.join(FIRMWARE_SRC, "erika_image.h"))
    assert defines.get("ERIKA_CHAR_DELAY_MS") == mel.RAW_BYTE_COST_MS


# ---------------------------------------------------------------------------
# recovering the cell grid from a scanned charset sheet
# ---------------------------------------------------------------------------


def _fake_scan(path, cols=20, rows=5, cell=30, line=50, margin=17,
              bearing=6, marks=True):
    """A synthetic scan of a charset sheet, at a scale nothing downstream knows.

    Each glyph cell gets a bar a quarter of the way across it, which is what the
    assertions read: a grid recovered a fraction of a cell out of step moves
    every bar by that fraction. `bearing` is how far a glyph's ink sits inside
    its own cell, and it is the whole reason the marks exist -- a crop to the
    outermost ink lands `bearing` px inside the block, not on its edge.
    """
    import cv2
    import numpy as np

    from erika.make_charset import sheet_mark_cells

    left_cell, first_cell, right_cell = sheet_mark_cells(cols)
    width = margin * 2 + (right_cell + 1) * cell
    img = np.full((margin * 2 + rows * line, width), 255, np.uint8)

    def put(cell_x, row, x0_frac, w_frac):
        x = margin + int((cell_x + x0_frac) * cell)
        y = margin + row * line
        img[y + line // 4 : y + 3 * line // 4,
            x : x + max(1, int(w_frac * cell))] = 40

    for r in range(rows):
        for c in range(cols):
            # A bar a quarter across the cell, inset by the bearing so that
            # cropping to the ink is wrong by exactly that much.
            put(first_cell + c, r, 0.25, 0.2)
        if marks:
            put(left_cell, r, 0.45, 0.1)
            put(right_cell, r, 0.45, 0.1)
    # Inset the block's own ink so a crop-to-ink is demonstrably off.
    img[:, margin + first_cell * cell : margin + first_cell * cell + bearing] = 255
    cv2.imwrite(str(path), img)
    return cols, rows


@pytest.mark.parametrize("margin", [0, 17, 60])
def test_a_scan_grid_is_recovered_from_the_registration_marks(tmp_path, margin):
    """The marks make the crop stop mattering, which is the point of them.

    Three margins, including none: the recovered grid has to be the same grid
    each time, because it is measured from two marks a known number of cells
    apart rather than assumed from the edges of the image.
    """
    from erika.make_charset import _sheet_from_scan

    path = tmp_path / f"scan-{margin}.png"
    cols, rows = _fake_scan(path, margin=margin)
    sheet, got_cols, got_rows = _sheet_from_scan(
        str(path), cols * rows, cols, cell_w=24, cell_h=40
    )
    assert (got_cols, got_rows) == (cols, rows)
    assert sheet.shape == (rows * 40, cols * 24)

    # Every tile's bar should sit a quarter of the way across it. Read the
    # darkest column of each tile and check where it fell.
    import numpy as np

    for r in range(rows):
        for c in range(cols):
            tile = sheet[r * 40 : (r + 1) * 40, c * 24 : (c + 1) * 24]
            ink = np.flatnonzero(tile.min(axis=0) < 0.6)
            assert ink.size, f"tile ({r}, {c}) came out blank"
            start = ink[0] / 24
            assert abs(start - 0.25) < 0.06, (
                f"tile ({r}, {c}) bar starts at {start:.3f} of a cell, not 0.25 "
                "-- the grid is out of step"
            )


def test_a_scan_without_marks_still_works_and_says_it_is_guessing(tmp_path, capsys):
    """An older sheet, or a crop that cut the marks off. The fallback is the old
    behaviour, and the old behaviour is the thing the marks replaced -- so it has
    to be loud rather than silent."""
    from erika.make_charset import _sheet_from_scan

    path = tmp_path / "nomarks.png"
    cols, rows = _fake_scan(path, margin=0, marks=False)
    sheet, _, _ = _sheet_from_scan(str(path), cols * rows, cols, 24, 40)
    assert sheet.shape == (rows * 40, cols * 24)
    said = capsys.readouterr().out
    assert "no registration marks" in said
    assert "eighth of a cell" in said


def test_the_marks_are_where_the_sheet_types_them(tmp_path):
    """The layout is written down once and used by two modules -- the sheet types
    it and the scan measures it. A disagreement is a charset silently a fraction
    of a cell out of step, which is the failure the marks were added to remove.
    """
    from erika import pipeline
    from erika.make_charset import SHEET_MARK_CHAR, sheet_mark_cells

    out = os.path.join(tmp_path, "sheet.etp")
    assert pipeline.main(["sheet", "-o", out, "--sheet-cols", "20"]) == 0
    machine = emulate.type_job(etp.load(out))
    mark = ec.glyph_for_char(SHEET_MARK_CHAR).code
    left, first, right = sheet_mark_cells(20)

    rows = {}
    for imp in machine.impressions:
        rows.setdefault(imp.y, []).append(imp)
    assert rows, "the sheet typed nothing"
    for y, line in rows.items():
        cells = sorted(i.x // 2 for i in line)
        assert cells[0] == left, f"row {y} does not open at cell {left}"
        assert cells[-1] == right, f"row {y} does not close at cell {right}"
        assert line[0].code == mark and line[-1].code == mark
    assert machine.overruns == 0


def test_the_marks_are_struck_hard_even_on_a_light_row(tmp_path):
    """A mark is a measurement, not a sample. One typed at a light force on a
    light-force row is a measurement that might not be found on the scan."""
    from erika import pipeline

    out = os.path.join(tmp_path, "sheet.etp")
    assert pipeline.main(["sheet", "-o", out, "--forces", "0x00,0x30"]) == 0
    machine = emulate.type_job(etp.load(out))
    mark = ec.glyph_for_char("!").code
    forces = {i.force for i in machine.impressions
              if i.code == mark and i.x // 2 in (0, 23)}
    assert forces == {0x00}, f"marks were struck at {forces}, not the hardest force"


def test_the_crop_to_ink_the_marks_replaced_really_was_off(tmp_path):
    """The failure, demonstrated rather than asserted from arithmetic.

    An unmarked sheet cropped exactly to its outermost ink -- which is what the
    sheet used to ask for -- gives a grid displaced by one side bearing, because
    the ink of the outer column starts inside its own cell. The same sheet with
    marks comes back in step. Both are measured here off the same synthetic
    scan, so the difference is the marks and nothing else.
    """
    import numpy as np

    from erika.make_charset import _sheet_from_scan

    def bar_offsets(path):
        sheet, _, rows = _sheet_from_scan(str(path), 20 * 5, 20, 24, 40)
        out = []
        for r in range(rows):
            for c in range(20):
                tile = sheet[r * 40 : (r + 1) * 40, c * 24 : (c + 1) * 24]
                ink = np.flatnonzero(tile.min(axis=0) < 0.6)
                out.append(ink[0] / 24 if ink.size else None)
        return [o for o in out if o is not None]

    marked = tmp_path / "marked.png"
    _fake_scan(marked, margin=40, bearing=6, marks=True)

    # The old way: no marks, and cropped hard against the ink.
    import cv2
    bare = tmp_path / "bare.png"
    _fake_scan(bare, margin=0, bearing=6, marks=False)
    im = cv2.imread(str(bare), cv2.IMREAD_GRAYSCALE)
    inked_x = np.flatnonzero((im < 200).any(axis=0))
    inked_y = np.flatnonzero((im < 200).any(axis=1))
    cv2.imwrite(str(bare), im[inked_y[0] : inked_y[-1] + 1,
                              inked_x[0] : inked_x[-1] + 1])

    with_marks = bar_offsets(marked)
    crop_to_ink = bar_offsets(bare)

    worst_marked = max(abs(o - 0.25) for o in with_marks)
    worst_cropped = max(abs(o - 0.25) for o in crop_to_ink)
    assert worst_marked < 0.06, f"the marked grid is out by {worst_marked:.3f} cells"
    assert worst_cropped > 2 * worst_marked, (
        "the crop-to-ink grid was supposed to be the worse of the two; it came "
        f"out at {worst_cropped:.3f} cells against {worst_marked:.3f}"
    )


def test_the_sheet_and_the_charset_builder_agree_on_the_grid():
    """`sheet --sheet-cols` and `charset --sheet-cols` are one number.

    The sheet types its glyphs `sheet_cols` to a line and the builder slices the
    scan on that same grid, so the two subcommands have to be tellable the same
    value. For one commit `charset` could not be told at all: it took no
    `--sheet-cols` and `_build_charset` did not forward one, so a sheet typed at
    anything but the default was sliced on a grid of 20 regardless.

    That failure is invisible downstream, which is why it is worth a test. Every
    tile comes back a blend of two neighbouring glyphs -- still a plausible tile,
    still the right count, so `_verify_mapping` passes and the charset loads. The
    only symptom is that every tonal decision in every picture is made against
    ink that no key on the machine produces.
    """
    from erika.pipeline import build_parser

    parser = build_parser()
    subs = next(
        action.choices
        for action in parser._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    for command in ("sheet", "charset"):
        defaults = {a.dest: a.default for a in subs[command]._actions}
        assert "sheet_cols" in defaults, f"`{command}` takes no --sheet-cols"

    assert (
        {a.dest: a.default for a in subs["sheet"]._actions}["sheet_cols"]
        == {a.dest: a.default for a in subs["charset"]._actions}["sheet_cols"]
    ), "the two default to different grids, so an untouched pair disagrees"


def test_the_charset_builder_forwards_the_grid_it_was_given():
    """And the flag reaches ``make_charset``, not just the parser.

    Separate from the test above because they fail for different reasons: that
    one catches a flag nobody can pass, this one catches a flag that is accepted
    and then dropped -- which looks identical from the command line.
    """
    from erika.pipeline import _build_charset, build_parser

    args = build_parser().parse_args(
        ["charset", "--from-scan", "scan.png", "--sheet-cols", "26"]
    )
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)

    _build_charset(args, spy, lambda text: (), lambda text: None)
    assert seen["sheet_cols"] == 26
    assert seen["scan"] == "scan.png"


# ---------------------------------------------------------------------------
# Reading the strike-force probe sheet back
#
# The sheet is typed to be read by eye, and `forces --from-scan` reads it by
# arithmetic instead: every row is the same glyph struck the same number of
# times, so the ink per row is the transfer curve of the force command. What
# these tests defend is the geometry, because the failure mode is silent -- a
# misidentified row reads as a machine with a strange force curve, not as a bug.
# ---------------------------------------------------------------------------


def _draw_probe_scan(tmp_path, blocks, run, ink_for, pitch=10, cell_px=24,
                     bearing=0.125, margin=37):
    """A synthetic scan of the sheet ``probe_lines`` describes.

    Labels are drawn at *full* ink on every row, whatever the row's own force
    did. That is not what the machine types -- a row below the threshold is
    blank end to end -- and it is deliberate: it makes a reader whose window has
    slipped left onto the label measure black instead of the ink it was after.
    """
    import cv2

    from erika.force_scan import PROBE_TITLE, probe_lines

    lines = probe_lines(blocks, run)
    line_px = cell_px * ec.cell_aspect(pitch)
    cols = max(l.run_col + l.run_cells for l in lines if l.kind == "sample")
    width = int(margin * 2 + cols * cell_px)
    height = int(margin * 2 + len(lines) * line_px)
    im = np.full((height, width), 255, dtype="uint8")

    def box(index, first_cell, cells, value):
        top = int(round(margin + (index + 0.2) * line_px))
        bottom = int(round(margin + (index + 0.8) * line_px))
        left = int(round(margin + (first_cell + bearing) * cell_px))
        right = int(round(margin + (first_cell + cells - bearing) * cell_px))
        im[top:bottom, left:right] = value

    for index, line in enumerate(lines):
        if line.kind == "title":
            box(index, 0, len(PROBE_TITLE), 0)
        elif line.kind == "sample":
            box(index, 0, len(line.text), 0)  # the label, always black
            ink = ink_for(line.value)
            if ink > 0:
                box(index, line.run_col, line.run_cells,
                    int(round(255 * (1 - ink))))

    path = str(tmp_path / "probe.png")
    cv2.imwrite(path, im)
    return path


def test_the_probe_sheet_is_typed_from_the_list_the_reader_counts():
    """One layout, walked twice. The reader cannot count rows for itself."""
    from erika.force_scan import probe_lines

    blocks = {"custom": [0, 5, 10]}
    lines = probe_lines(blocks, 20)
    samples = [l for l in lines if l.kind == "sample"]

    assert [l.value for l in samples] == [None, 0, 5, 10], (
        "the reference row typed before any force command has to be first, and "
        "carry no force"
    )
    assert len({len(l.text) for l in samples if l.value is not None}) == 1, (
        "value labels must be one width, or the runs do not line up"
    )
    assert all(l.run_col > len(l.text) for l in samples), (
        "the run has to start clear of the label it is measured beside"
    )


def test_a_scanned_probe_sheet_reads_back_as_the_curve_that_drew_it(tmp_path):
    from erika.force_scan import read_scan

    blocks = {"custom": list(range(0, 101, 10))}
    # A plausible shape: solid at 0, nothing until the threshold, then a ramp
    # that saturates -- the machine's own answer, in miniature.
    curve = {0: 1.0, 10: 0.0, 20: 0.0, 30: 0.0, 40: 0.08, 50: 0.35,
             60: 0.62, 70: 0.82, 80: 0.94, 90: 1.0, 100: 1.0}
    scan = _draw_probe_scan(tmp_path, blocks, 20,
                            lambda v: 1.0 if v is None else curve[v])
    readings = read_scan(scan, blocks, 20)

    assert [r.value for r in readings] == [None] + list(curve), (
        "the rows that printed nothing still have to occupy their place, or "
        "every row after them is read as a different force"
    )
    by_value = {r.value: r for r in readings}
    for value, wanted in curve.items():
        assert by_value[value].ink == pytest.approx(wanted, abs=0.03), (
            f"force {value} read back as {by_value[value].ink:.3f}, not {wanted}"
        )
    assert [v for v in curve if not by_value[v].marked] == [10, 20, 30]


def test_the_suggestion_is_spaced_in_ink_and_not_in_value(tmp_path):
    """The whole point of reading the sheet rather than eyeballing it."""
    from erika.force_scan import read_scan, suggest

    blocks = {"custom": list(range(0, 101, 10))}
    curve = {0: 1.0, 10: 0.0, 20: 0.0, 30: 0.0, 40: 0.08, 50: 0.35,
             60: 0.62, 70: 0.82, 80: 0.94, 90: 1.0, 100: 1.0}
    scan = _draw_probe_scan(tmp_path, blocks, 20,
                            lambda v: 1.0 if v is None else curve[v])
    readings = read_scan(scan, blocks, 20)
    picked = suggest(readings, 4)
    ink = {r.value: r.ink for r in readings}

    # Hardest first means most ink first, which is *not* descending by value:
    # 0 is full strike, so a correct list here starts with the smallest number
    # on it. Sorting the result numerically would put full strike last and hand
    # the registration marks of a charset sheet to the lightest force there is.
    assert [ink[v] for v in picked] == sorted((ink[v] for v in picked),
                                              reverse=True)
    assert ink[picked[0]] == 1.0, "the hardest strike has to stay; see cmd_forces"
    assert len(set(picked)) == len(picked), "a value twice is a wasted block"
    # Quarter, half and three quarters of full ink land at 40, 50 and 60-70 --
    # the bottom of the value range, which is what even arithmetic would miss.
    assert all(40 <= v <= 70 for v in picked[1:]), (
        f"{picked} spaces the lighter forces by value rather than by ink"
    )


def test_a_row_below_the_ink_threshold_does_not_shift_the_rows_after_it(tmp_path):
    """The reason the grid is computed instead of detected.

    A force too weak to mark prints nothing, and its label prints nothing
    either, because the label is typed at the row's own force. A reader that
    found rows by looking for ink would hand every later row the wrong force
    and produce a curve that looks entirely reasonable.
    """
    from erika.force_scan import read_scan

    blocks = {"custom": [0, 20, 40, 60, 80]}
    scan = _draw_probe_scan(
        tmp_path, blocks, 20,
        lambda v: 1.0 if v is None else {0: 1.0, 20: 0.0, 40: 0.0,
                                         60: 0.5, 80: 0.9}[v],
    )
    by_value = {r.value: r.ink for r in read_scan(scan, blocks, 20)}
    assert by_value[60] == pytest.approx(0.5, abs=0.03)
    assert by_value[80] == pytest.approx(0.9, abs=0.03)


def test_the_reader_tolerates_a_cell_width_it_guessed_slightly_wrong(tmp_path):
    """The cell comes from the title's ink, so the side bearing is a guess.

    Half a cell of inset at each end of the run is what buys the tolerance.
    """
    from erika.force_scan import read_scan

    blocks = {"custom": [0, 50]}
    for bearing in (0.05, 0.3):
        scan = _draw_probe_scan(
            tmp_path, blocks, 20,
            lambda v: 1.0 if v is None else {0: 1.0, 50: 0.5}[v],
            bearing=bearing,
        )
        by_value = {r.value: r.ink for r in read_scan(scan, blocks, 20)}
        assert by_value[50] == pytest.approx(0.5, abs=0.05), (
            f"a side bearing of {bearing} cells threw the measurement off"
        )


def test_reading_a_sheet_with_the_wrong_sweep_is_refused_not_guessed(tmp_path):
    """A sweep that does not match the sheet is the sheet_cols failure again."""
    from erika.force_scan import read_scan

    typed = {"custom": [0, 50]}
    scan = _draw_probe_scan(tmp_path, typed, 20,
                            lambda v: 1.0 if v is None else 0.5)
    with pytest.raises(ValueError, match="outside the scan"):
        read_scan(scan, {"custom": list(range(0, 104))}, 20)


def test_full_strike_is_not_reported_as_the_ink_threshold(tmp_path):
    """0 marks because it is the hardest strike, not because it is the floor.

    Reading the sheet naively -- "the lowest value that took ink" -- names the
    top of the scale as the bottom of it, which is the one thing a reader of
    this report must not be told.
    """
    from erika.force_scan import read_scan, report

    blocks = {"custom": [0, 20, 40, 60, 95, 100]}
    curve = {0: 1.0, 20: 0.0, 40: 0.1, 60: 0.6, 95: 1.0, 100: 1.0}
    scan = _draw_probe_scan(tmp_path, blocks, 20,
                            lambda v: 1.0 if v is None else curve[v])
    text = report(read_scan(scan, blocks, 20), 3)

    assert "ink begins at 40" in text
    assert "stops changing from 95" in text
    assert "top of this scale" in text


# ---------------------------------------------------------------------------
# Squaring a scan up before its grid is sliced
# ---------------------------------------------------------------------------


def _charset_scan(path, angle=0.0, noise=0.0, dead_keys=False, cols=20, seed=11):
    """A synthetic scan of a charset sheet, optionally crooked and grainy.

    Bars rather than type, but with type's extents: the full line height and
    three quarters of the cell, which is what makes rotation contaminate a tile
    with its neighbour. The noise is paper, and it is not decoration -- it is
    what a scan has and a font-drawn sheet does not, and it broke the blank-cell
    test long before any of this rotated anything.
    """
    import cv2

    from erika.make_charset import sheet_mark_cells

    rng = np.random.default_rng(seed)
    glyphs = len(ec.all_glyphs(dead_keys=dead_keys))
    rows = (glyphs + cols - 1) // cols
    left_cell, first_cell, right_cell = sheet_mark_cells(cols)
    cell, line, margin = 36, 60, 60
    im = np.full((margin * 2 + rows * line, margin * 2 + (right_cell + 1) * cell),
                 250, np.uint8)

    def bar(cell_x, row, start, width, grey):
        x = margin + int((cell_x + start) * cell)
        y = margin + row * line
        im[y + 1: y + line - 1, x: x + max(1, int(width * cell))] = grey

    for row in range(rows):
        bar(left_cell, row, 0.45, 0.10, 30)
        bar(right_cell, row, 0.45, 0.10, 30)
        for column in range(cols):
            index = row * cols + column
            if index >= glyphs:
                break
            bar(first_cell + column, row, 0.125, 0.75, 40 + (index % 7) * 18)

    if angle:
        height, width = im.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        im = cv2.warpAffine(im, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=250)
    if noise:
        im = np.clip(im.astype(np.int16) + rng.normal(0, noise, im.shape),
                     0, 255).astype(np.uint8)
    cv2.imwrite(str(path), im)
    return str(path)


def test_the_skew_of_a_scan_is_found_to_a_fraction_of_a_degree(tmp_path):
    from erika import deskew

    import cv2

    for typed in (-2.0, -0.75, 0.5, 1.5):
        path = _charset_scan(tmp_path / f"s{typed}.png", angle=typed, noise=1.0)
        found = deskew.find_angle(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        assert found == pytest.approx(-typed, abs=0.06), (
            f"a sheet {typed} deg off was measured at {found}"
        )


def test_a_square_scan_is_handed_on_without_being_resampled(tmp_path):
    """The correction has to be worth more than the resample that applies it.

    A hundredth of a degree is inside the noise of the search, and rotating for
    it blurs every edge in the image by half a pixel -- several percent of a
    row's ink on a sheet whose rows are a couple of dozen pixels tall. That is a
    loss taken to remove nothing.
    """
    from erika import deskew

    import cv2

    path = _charset_scan(tmp_path / "square.png", angle=0.0, noise=1.0)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out, angle, note = deskew.straighten(im)
    assert angle == 0.0 and note is None
    assert out is im, "a scan already square must come back untouched, not re-rendered"


def test_a_crooked_scan_builds_the_same_charset_as_a_straight_one(tmp_path):
    """The point of the whole exercise, end to end.

    Without straightening, a degree moves the worst tile of this sheet by some
    26 grey levels out of 255 and two degrees fails the build outright. The grid
    is sliced axis-aligned, so the error grows toward the corners -- which is the
    half of the sheet a spot check does not look at.
    """
    from erika.make_charset import make_charset
    from utils import prep_charset

    base = str(tmp_path)
    os.makedirs(base, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(SRC, "layers.json"), base)

    def build(name, angle, deskew_scan):
        scan = _charset_scan(tmp_path / f"{name}.png", angle=angle, noise=1.5,
                             dead_keys=True)
        make_charset(name=name, pitch=10, scan=scan, sheet_cols=20,
                     dead_keys=True, base_path=base, deskew_scan=deskew_scan)
        tiles, _, _ = prep_charset(name, base)
        return np.asarray(tiles, dtype="float32").mean(axis=(1, 2))

    straight = build("straight", 0.0, True)
    crooked = build("crooked", 1.5, True)
    assert len(crooked) == len(straight)
    worst = float(np.abs(crooked - straight).max()) * 255
    assert worst < 3.0, f"straightening left {worst:.1f} grey levels on the worst tile"


def test_a_scanned_sheet_takes_its_tiles_by_position_not_by_ink(tmp_path):
    """On a scan, "is this cell blank" is not an answerable question.

    At the lightest strike force some glyphs put down no ink at all -- that is
    what a lightest strike force is *for* -- and dropping those cells shifts
    every index after them. The cells that must go are the ones past the last
    glyph, which the sheet's layout names without looking at the paper.
    """
    from erika.make_charset import SCAN_WHITE_THRESHOLD, scan_exclusions

    assert SCAN_WHITE_THRESHOLD > 1.0, (
        "a scanned sheet must keep every cell, and a cell of bare paper has a "
        "mean of exactly 1.0"
    )
    # 103 glyphs at 4 forces is 412 tiles in a 20x21 grid: 8 cells spare.
    excluded = scan_exclusions(20, 21, 412)
    assert len(excluded) == 8
    # chop_charset prepends a blank and filters on i+1, so cell k is number k+2.
    assert excluded == list(range(414, 422))
    assert 413 not in excluded, "that is the last glyph, not a spare cell"


def test_a_scanned_grid_that_is_not_on_the_glyphs_is_complained_about(tmp_path):
    """Nothing downstream can fail on this now, so it has to be caught here.

    Taking tiles by position means a grid half a row out still yields a full
    charset of plausible tiles that prints and is wrong. The populations are
    compared at the median, not at the extremes: on a real sheet the palest
    glyph and the darkest blank do overlap, because a few glyphs make no mark at
    the lightest force.
    """
    from erika.make_charset import check_scan_grid

    cols, rows, tiles = 20, 21, 412
    good = np.full((rows * 40, cols * 24), 1.0, dtype="float32")
    for index in range(tiles):
        r, c = divmod(index, cols)
        good[r * 40 + 4: (r + 1) * 40 - 4, c * 24 + 4: (c + 1) * 24 - 4] = 0.55
    assert check_scan_grid(good, cols, rows, tiles) is None

    # A few faint glyphs among the trailing paper is still fine: the medians
    # separate even when the extremes do not.
    for index in range(tiles - 5, tiles):
        r, c = divmod(index, cols)
        good[r * 40: (r + 1) * 40, c * 24: (c + 1) * 24] = 1.0
    assert check_scan_grid(good, cols, rows, tiles) is None

    blank = np.full((rows * 40, cols * 24), 1.0, dtype="float32")
    assert "not on the glyphs" in (check_scan_grid(blank, cols, rows, tiles) or "")


def test_the_charset_builder_can_be_told_not_to_straighten():
    """A flag that is accepted and dropped looks identical from the command line."""
    from erika.pipeline import _build_charset, build_parser

    seen = {}
    for argv, wanted in ((["charset", "--from-scan", "s.png"], True),
                         (["charset", "--from-scan", "s.png", "--no-deskew"], False)):
        args = build_parser().parse_args(argv)
        _build_charset(args, lambda **kw: seen.update(kw), lambda t: (), lambda t: None)
        assert seen["deskew_scan"] is wanted


def test_one_overhanging_glyph_does_not_merge_two_columns(tmp_path):
    """What broke on the second real sheet this ever read.

    The column projection runs down the whole sheet, so a single glyph that
    reaches into the gap beside it -- once, anywhere in twenty-one rows -- welds
    its column to the next one for the entire projection. Twenty columns became
    four, the spacing they exist to measure was lost with them, and the marks
    could not be told from the block. Asking for a share of the heaviest column
    rather than for two rows of ink is what fixes it.
    """
    import cv2

    from erika.make_charset import _find_sheet_marks, sheet_mark_cells

    cols = 20
    left_cell, first_cell, right_cell = sheet_mark_cells(cols)
    cell, line, margin, rows = 40, 60, 50, 21
    im = np.full((margin * 2 + rows * line, margin * 2 + (right_cell + 1) * cell),
                 250, np.uint8)
    for row in range(rows):
        for c in (left_cell, right_cell):
            x = margin + c * cell + int(cell * 0.45)
            im[margin + row * line + 8: margin + (row + 1) * line - 8,
               x: x + int(cell * 0.12)] = 30
        for column in range(cols):
            x = margin + (first_cell + column) * cell + int(cell * 0.15)
            im[margin + row * line + 6: margin + (row + 1) * line - 6,
               x: x + int(cell * 0.7)] = 60
    # One glyph, on one row, reaching across the gap into its neighbour.
    x = margin + (first_cell + 5) * cell + int(cell * 0.15)
    im[margin + 3 * line + 20: margin + 3 * line + 30, x: x + int(cell * 1.1)] = 60
    path = str(tmp_path / "overhang.png")
    cv2.imwrite(path, im)

    marks = _find_sheet_marks(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cols)
    assert marks is not None, "one overhanging glyph hid the registration marks"
    left, right = marks
    assert (right - left) / right_cell == pytest.approx(cell, abs=cell * 0.05)


def test_a_speck_of_dust_does_not_set_the_vertical_extent(tmp_path):
    """Also from the first real sheet: three pixels of dirt near the page edge.

    The extent was taken from any ink at all, so the grid became a quarter
    taller than the type and started nearly three rows above it. Every cell then
    held the wrong thing, and a charset full of the wrong thing is not an error
    anywhere downstream.
    """
    import cv2

    from erika.make_charset import _ink_extent

    im = np.full((2600, 1900), 250, np.uint8)
    im[300:2350, 300:1700] = 60          # the type
    im[2:5, 900:903] = 40                # dust near the top edge
    im[2590:2593, 1000:1003] = 40        # and near the bottom
    extent = _ink_extent(im < 250 * 0.75, axis=1)
    assert extent == (300, 2349), f"dust moved the extent to {extent}"


def test_a_glyph_that_did_not_print_at_the_hardest_force_is_refused(tmp_path):
    """A blank tile is one the optimizer may choose for a midtone.

    Taking a scan's tiles by position means a blank cell no longer shifts the
    count, which is what makes a light strike force usable -- so the check that
    a sheet actually printed has to be made deliberately, and only where it is
    true. At the hardest force every key should mark the paper.
    """
    from erika.make_charset import check_scan_hardest_block

    cols, rows, block = 20, 6, 103
    n_tiles = block
    sheet = np.full((rows * 40, cols * 24), 1.0, dtype="float32")
    for index in range(n_tiles):
        r, c = divmod(index, cols)
        sheet[r * 40 + 4: (r + 1) * 40 - 4, c * 24 + 4: (c + 1) * 24 - 4] = 0.5
    chars = [chr(65 + i % 26) for i in range(n_tiles)]
    assert check_scan_hardest_block(sheet, cols, rows, n_tiles, block, chars) is None

    r, c = divmod(7, cols)
    sheet[r * 40: (r + 1) * 40, c * 24: (c + 1) * 24] = 1.0
    fault = check_scan_hardest_block(sheet, cols, rows, n_tiles, block, chars)
    assert fault is not None and "tile 8" in fault and repr(chars[7]) in fault


def test_a_glyph_missing_from_a_lighter_force_is_not_refused(tmp_path):
    """Which is the whole point of typing the sheet at more than one force.

    The underscore and the accents stop marking two forces down on both sheets
    this was built against. Refusing that would refuse every multi-force charset
    worth having.
    """
    from erika.make_charset import check_scan_hardest_block

    cols, rows, block = 20, 11, 103
    n_tiles = block * 2
    sheet = np.full((rows * 40, cols * 24), 1.0, dtype="float32")
    for index in range(n_tiles):
        r, c = divmod(index, cols)
        grey = 0.5 if index < block else 0.93
        sheet[r * 40 + 4: (r + 1) * 40 - 4, c * 24 + 4: (c + 1) * 24 - 4] = grey
    for index in (block + 5, block + 40, n_tiles - 1):  # light force, no mark
        r, c = divmod(index, cols)
        sheet[r * 40: (r + 1) * 40, c * 24: (c + 1) * 24] = 1.0
    chars = [chr(65 + i % 26) for i in range(n_tiles)]
    assert check_scan_hardest_block(sheet, cols, rows, n_tiles, block, chars) is None


def test_the_rows_come_from_the_marks_and_not_from_the_ink(tmp_path):
    """Ink runs from the tallest glyph's top to the lowest one's bottom.

    Which is inside the block by an ascender at one end and a descender at the
    other, so dividing it by the row count gives a pitch that is short. Nothing
    in one row, a quarter of a cell by the twenty-first -- and the twenty-first
    row is where the lightest strike force was typed, so the tiles it damages
    most are the ones with least ink to spare. On the first real sheet the ink
    extent implied 98.5px where the marks say 99.9.
    """
    import cv2

    from erika.make_charset import _ink_extent, _mark_rows, sheet_mark_cells

    cols, rows = 20, 21
    left_cell, first_cell, right_cell = sheet_mark_cells(cols)
    cell, pitch, margin = 40, 100, 60
    im = np.full((margin * 2 + rows * pitch, margin * 2 + (right_cell + 1) * cell),
                 250, np.uint8)
    for row in range(rows):
        y = margin + row * pitch
        # The marks: a stem and a dot under it, like the '!' the sheet types.
        for c in (left_cell, right_cell):
            x = margin + c * cell + int(cell * 0.45)
            im[y + 20: y + 62, x: x + 5] = 30
            im[y + 70: y + 78, x: x + 5] = 30
        # Glyphs that sit well inside their cells, which is what misleads the
        # ink extent: 22 pixels of clearance at the top, 14 at the bottom.
        for column in range(cols):
            x = margin + (first_cell + column) * cell + 6
            im[y + 22: y + pitch - 14, x: x + cell - 12] = 60
    path = str(tmp_path / "rows.png")
    cv2.imwrite(path, im)

    read = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    left_c = margin + left_cell * cell + int(cell * 0.45) + 2
    right_c = margin + right_cell * cell + int(cell * 0.45) + 2
    band = _mark_rows(read, left_c, right_c, cell, rows)
    assert band is not None, "twenty-one marks down each edge and none were found"
    top, found = band
    assert found == pytest.approx(pitch, abs=0.5), (
        f"the marks say {found:.2f}px where the sheet was drawn at {pitch}"
    )

    # And what the ink alone would have said, which is the bug this replaces.
    block = read[:, margin + first_cell * cell:
                 margin + (first_cell + cols) * cell]
    extent = _ink_extent(block < 250 * 0.75, axis=1)
    from_ink = (extent[1] - extent[0] + 1) / rows
    assert from_ink < pitch - 1, "the fixture no longer reproduces the bug"
    assert abs(found - pitch) < abs(from_ink - pitch) / 3, (
        f"marks {found:.2f} are no better than ink {from_ink:.2f} against {pitch}"
    )


def _force_sheet(cols, rows, entries_n, block, cell_w, cell_h, slip=0.0):
    """A sheet where each force block can be drawn a little lower than the last.

    ``slip`` is the fraction of a cell that block *n* is shifted down by n*slip,
    which is what a row pitch a percent short does to a real scan.
    """
    sheet = np.ones((rows * cell_h, cols * cell_w), dtype="float32")
    for i in range(entries_n):
        r, c = divmod(i, cols)
        drop = int(round((i // block) * slip * cell_h))
        top = r * cell_h + 8 + drop
        sheet[top: min(top + 12, (r + 1) * cell_h),
              c * cell_w + 4: (c + 1) * cell_w - 4] = 1.0 - 0.2 * (i % block % 5 + 1)
    return sheet


def test_the_force_montage_is_only_built_when_there_is_something_to_compare():
    from erika import erika_codes as ec, force_view

    glyphs = ec.all_glyphs()[:20]
    entries = [(g, 0) for g in glyphs]
    sheet = _force_sheet(20, 1, 20, 20, 24, 40)
    assert force_view.build(sheet, entries, 20, 24, 40, [0]) is None, (
        "one force has nothing to put beside anything"
    )
    assert force_view.drift(sheet, entries, 20, 24, 40, [0]) is None


def test_the_force_montage_measures_a_grid_that_slips_between_blocks():
    """The number under the picture, and the reason both exist.

    A drifting row pitch cuts each force block lower than the last. Every tile
    still looks like its character, the count is right, the mapping verifies --
    and the tones are measured off centre, worst in the lightest block.
    """
    from erika import erika_codes as ec, force_view

    cols, block, forces = 20, 20, [0, 60, 45]
    glyphs = ec.all_glyphs()[:block]
    entries = [(g, f) for f in forces for g in glyphs]
    rows = (len(entries) + cols - 1) // cols

    steady = _force_sheet(cols, rows, len(entries), block, 24, 40, slip=0.0)
    slipped = _force_sheet(cols, rows, len(entries), block, 24, 40, slip=0.15)

    held = force_view.drift(steady, entries, cols, 24, 40, forces)
    moved = force_view.drift(slipped, entries, cols, 24, 40, forces)
    assert held is not None and held < 0.5, f"a steady grid measured {held}"
    assert moved > 4.0, f"a grid slipping 15% of a 40px cell measured only {moved}"

    montage = force_view.build(slipped, entries, cols, 24, 40, forces)
    assert montage is not None
    assert montage.shape[0] > 40 and montage.shape[1] > 3 * 24, "too small to read"


def test_the_montage_shows_the_lightest_marks_the_wheel_makes():
    """Chosen from the sheet, not named in the code.

    The glyphs that expose a drifting grid are the small ones that sit high or
    low in the cell -- a full stop, an accent -- and which those are differs by
    wheel. So the spread runs from least ink to most, and the light end is where
    the check actually happens.
    """
    from erika import erika_codes as ec, force_view

    cols, block, forces = 20, 20, [0, 60]
    glyphs = ec.all_glyphs()[:block]
    entries = [(g, f) for f in forces for g in glyphs]
    rows = (len(entries) + cols - 1) // cols
    sheet = _force_sheet(cols, rows, len(entries), block, 24, 40)

    picked = force_view._pick(sheet, entries, cols, 24, 40, block,  # noqa: SLF001
                              force_view.ROWS_SHOWN)
    assert len(picked) == force_view.ROWS_SHOWN
    ink = []
    for i in picked:
        r, c = divmod(i, cols)
        ink.append(1.0 - float(sheet[r * 40: (r + 1) * 40, c * 24: (c + 1) * 24].mean()))
    assert ink[0] < ink[-1], "the spread should run from least ink to most"


def _phase_sheet(path, rows=5, cols=6, pitch=100, top=30.0, offset=0):
    """Rows of type that fill the line height, sitting ``offset`` px low.

    An accent against the ceiling of the cell, a body, and an underscore against
    the floor -- which is what leaves no gap to cut along, and is the reason the
    phase has to be found rather than assumed. Ink from the very top of the cell
    to the very bottom means any boundary in the wrong place passes through
    something.
    """
    import cv2

    im = np.full((int(top + rows * pitch + 60), cols * 40), 250, np.uint8)
    for r in range(rows):
        y = int(top + r * pitch) + offset
        for c in range(cols):
            x = c * 40 + 6
            if c == 2:
                im[y + 1: y + 9, x: x + 20] = 60        # an accent, at the ceiling
            im[y + 13: y + 87, x: x + 26] = 70          # the body of the type
            if c == 1:
                im[y + 92: y + 99, x: x + 30] = 60      # an underscore, at the floor
            if c == 3:
                # A stroke running nearly the whole cell, like the '|' the wheel
                # has. Without one there are gaps for a boundary to slip into and
                # several phases cut nothing; the machine leaves no such gaps,
                # which is what folding a real sheet onto one cell shows.
                im[y: y + 96, x + 8: x + 14] = 65
    cv2.imwrite(str(path), im)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), np.array(
        [top + r * pitch for r in range(rows)]
    )


def test_the_cut_is_placed_where_it_severs_the_least_glyph(tmp_path):
    """The marks fix the pitch and only nearly fix the phase.

    A mark's ink runs cap height to baseline, which is not centred in its cell --
    there is an accent zone above it and a descender zone below, and they are not
    the same depth. What that leaves is a cut a few percent of a row too high,
    invisible on every glyph except the one it is not: the underscore is a bar at
    the very bottom of the type body, so a few percent of a row is half of it,
    and the half that goes missing turns up at the top of the tile below.
    """
    from erika.make_charset import _cut_risk, _row_phase

    im, tops = _phase_sheet(tmp_path / "phase.png", offset=7)
    nudge = _row_phase(im, tops, 100, 6, 24, 40, 30)
    assert nudge == pytest.approx(7, abs=3), (
        f"the cut moved {nudge}px where the sheet sits 7px lower"
    )
    args = (im, tops, 100, 6, 24, 40, 30)
    assert _cut_risk(*args, nudge) < _cut_risk(*args, 0), "not a minimum at all"


def test_the_cut_is_not_moved_when_the_marks_already_had_it_right(tmp_path):
    """A correction, not a search. Nudging a good phase costs a resample."""
    from erika.make_charset import _row_phase

    im, tops = _phase_sheet(tmp_path / "square.png", offset=0)
    assert _row_phase(im, tops, 100, 6, 24, 40, 30) == 0


def test_a_glyph_that_ends_at_the_cell_floor_is_not_pushed_out_of_it(tmp_path):
    """The trap in scoring this by ink near the boundary rather than across it.

    The underscore is *meant* to reach the floor of its cell; so is an accent the
    ceiling. Score a cut by how much ink lies beside it and the best answer
    becomes one that shoves the underscore wholly into the cell below -- the same
    error as before, carried to completion, and now with nothing left at the
    boundary to give it away. Ink on *both* sides of the line, in one column, is
    what a severed glyph looks like and a well-placed one does not.
    """
    from erika.make_charset import _cut_risk

    im, tops = _phase_sheet(tmp_path / "floor.png", offset=0)
    args = (im, tops, 100, 6, 24, 40, 30)
    assert _cut_risk(*args, 0) == pytest.approx(0.0, abs=0.5), (
        "type that fits its cell exactly is not being cut by anything"
    )
    # Displacing the underscore into the next cell must not look like an
    # improvement on leaving it where it belongs.
    assert _cut_risk(*args, -8) > _cut_risk(*args, 0)
