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

    Four tests below are the only guard against the Erika codes, the .etp
    opcodes and the upload chunk size drifting apart between their Python and
    C++ copies. They need the firmware checked out, and the conventional
    layout is as a sibling of this repository -- but that is a convention, not
    a guarantee: checkout names and locations differ between machines.

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
        "ETP_HEADER_SIZE": etp.HEADER_SIZE, "ETP_VERSION": etp.VERSION,
        "ETP_FLAG_PITCH12": etp.FLAG_PITCH12,
        "ETP_FLAG_HOME_EACH_ROW": etp.FLAG_HOME_EACH_ROW,
    }
    for name, value in expected.items():
        assert defines.get(name) == value, f"{name} differs from etp.py"


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
    UPLOAD-READY / ACK / OK replies.
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
