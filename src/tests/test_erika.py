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
    }
    for name, value in expected.items():
        assert defines.get(name) == value, f"{name} differs from erika_codes.py"


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
        with pytest.raises(etp.EtpError, match="motion code"):
            etp.Encoder().strike(code)


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


def test_quarter_cell_offsets_are_rejected(tmp_path, charset):
    path = _write_choices(tmp_path, {"layer0_0.25_0": [[1, 2]]})
    with pytest.raises(PlanError, match="not typeable"):
        planner.build_plan(path, charset)


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
    with pytest.raises(etp.EtpError, match="motion code"):
        enc.set_force(ec.HALF_STEP_FORWARD)


@pytest.mark.parametrize("value", [0x00, 0x39, 0xFF])
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
    sheet = np.full((80, 96), 240, dtype=np.uint8)
    sheet[10:30, 10:30] = 60
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

    with pytest.raises(ValueError, match="motion codes"):
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
