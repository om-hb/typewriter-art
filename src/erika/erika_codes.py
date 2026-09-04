"""Character and control codes for the Sigma SM 8200i / Erika S3004 interface.

This is the Python mirror of ``erika_ai/src/erika_char_map.cpp``. Keep the two
in sync: the ``code`` values here are written verbatim into the ``.etp`` print
job and handed straight to ``SoftwareSerial.write()`` by the firmware.

Reference: Erika IF2014 Anwenderhandbuch, page 10
https://github.com/cyroxx/erika3004/blob/master/docs/Erika-IF2014_AnwenderHandbuch.pdf
"""

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Motion / control codes
# --------------------------------------------------------------------------
# These are the codes the typewriter *emits* when the corresponding key is
# pressed (see the switch in ErikaInterface::readInput) and, on this interface,
# also the codes it *accepts* to perform the same motion.
#
# CAUTION: HALF_STEP_FORWARD (0x73) is the one code in this table that is not
# confirmed by the keycode comments in erika_interface.cpp -- 0x73 is the only
# gap in the otherwise contiguous 0x71..0x79 motion block, and the block is
# laid out in forward/backward pairs. Verify it once on real hardware with
#     python -m erika.pipeline calibrate
# and override it via --half-step-forward-code if your machine differs.

SPACE = 0x71  # Leerschritt          - one full step right (prints nothing)
BACKSPACE = 0x72  # Rücktaste            - one full step left
HALF_STEP_FORWARD = 0x73  # Halbschritt vorwärts - half step right   (see caution)
HALF_STEP_BACK = 0x74  # Halbschrittrücktaste - half step left
HALF_LINE_FORWARD = 0x75  # Halbzeilentaste vorw.- half line down
HALF_LINE_BACK = 0x76  # Halbzeilentaste rückw.- half line up
NEWLINE = 0x77  # Wagenrücklauf        - carriage return + line feed
CARRIAGE_RETURN = 0x78  # Expressrücktaste     - carriage return, no line feed
TAB = 0x79  # Tabulator
MICRO_LINE_FORWARD = 0x81  # Mikrozeilenschaltung vorwärts
MICRO_LINE_BACK = 0x82  # Mikrozeilenschaltung rückwärts

# --------------------------------------------------------------------------
# Strike force (Anschlagstärke)
# --------------------------------------------------------------------------
# A two-byte command: this code, then one byte naming the force. It is the
# only way this machine can lay down less than a full measure of ink, and the
# paper's section 5.5 calls a charset with varying strike force "the largest
# factor in obtaining a good tonal range in the midtones and highlights" --
# so it is the single most valuable code in this table.
#
# The manual gives the code and says the next character is the strength; it does
# not say what strengths exist or how they are spelled. Both readings of
# "Zeichen" were plausible -- a small integer, or the ASCII digit for it -- and
# neither could be settled from here, so
#
#     python -m erika.pipeline forces
#
# types a sheet that sweeps both and lets the paper answer. That sheet has now
# been typed; what it said is below, and it settles the spelling. On a different
# machine or a different wheel, type it again -- everything below the command is
# a measurement of one machine, not a property of the protocol.
SET_STRIKE_FORCE = 0xA3  # Anschlagstärke - next byte is the force

#: Values `pipeline forces` sweeps by default, as {label: (first, last)}: the
#: two readings of the manual, before either had been on paper. The `raw` one
#: turned out to be right (see below), which leaves these as a first pass on an
#: unknown machine rather than the sweep to run on this one. Both blocks stay
#: inside the wheel's own range (see MAX_FORCE): if this machine does not
#: honour SET_STRIKE_FORCE, the force byte arrives as an ordinary character,
#: and a character the wheel can type is a visible stray mark rather than a
#: motion that shifts the rest of the line or a command that eats the byte
#: after it.
FORCE_PROBE_BLOCKS = {
    "raw": (0x00, 0x09),
    "ascii": (0x30, 0x39),
}

# What the sheet said, on the Sigma SM 8200i this was written for, with a
# Courier 10 wheel: swept 0x00..0x64 in steps of 5, then again in steps of 1 at
# both ends.
#
#   0          solid. Full strike; nothing was seen to print darker.
#   1..39      no ink at all. The command is honoured -- the strike is simply
#              too weak to reach the paper.
#   40 (0x28)  first ink: isolated dots, not a character.
#   43 (0x2B)  first legible character. The practical floor.
#   55 (0x37)  fully formed characters.
#   95..103    saturated; 0x5F to 0x67 are indistinguishable from each other.
#
# Three things follow, and the first is the one the sheet existed to settle.
#
# **The operand is a raw count, not the ASCII digit for one.** Under the `ascii`
# reading only 0x30..0x39 would set a force and every other value would be typed
# as a stray character; what the paper shows instead is one continuous ramp
# across the whole swept range, 0x32 and 0x37 sitting on it like any other
# value. Harder is a larger number, with 0 the exception at the top.
#
# **The usable ladder is 43..95, and it is compressed.** 43 to 55 spans
# barely-there to fully formed and everything above 55 is the top of the range,
# so charset forces want picking from the bottom of it -- and by how the ink
# looks, not by even arithmetic. Note that FORCE_PROBE_BLOCKS' own two blocks
# undersample exactly there: on a new wheel, `--from 35 --to 103 --step 1` is
# the pass worth the paper.
#
# **The scale saturates below MAX_FORCE.** 0x5F is already the top and the
# ceiling is 0x67, so the bound MAX_FORCE imposes for an unrelated reason (a
# force byte has to be inert if it is typed) costs this machine no tonal range
# at all. That was luck, not design.
#
# One thing the sheet did not resolve: whether 0 is exactly the top of the ramp
# or a shade darker still. By eye it is the same. Nothing depends on it.

#: The force a job asks for when it wants everything the machine has. The probe
#: sheet confirms it: 0 prints solid, level with the top of the scale.
FULL_STRIKE_FORCE = 0x00


#: The motion codes: they move the head instead of marking the paper, and must
#: never appear as a glyph strike. This is the block 0x71..0x82.
CONTROL_CODES = frozenset(
    {
        SPACE,
        BACKSPACE,
        HALF_STEP_FORWARD,
        HALF_STEP_BACK,
        HALF_LINE_FORWARD,
        HALF_LINE_BACK,
        NEWLINE,
        CARRIAGE_RETURN,
        TAB,
        MICRO_LINE_FORWARD,
        MICRO_LINE_BACK,
    }
)

# --------------------------------------------------------------------------
# The rest of the interface's vocabulary
# --------------------------------------------------------------------------
# The motion block is not where the machine's commands stop -- it is where the
# ones this pipeline *uses* stop. The published table (see
# erika_ai/ressources/steuercodes.md and docs/control-codes.md in the
# workspace) runs to 0xAF, and what matters here is not what those codes do but
# that some of them **swallow the byte after them**.
#
# That is the difference between a wrong byte and a ruined job. A stray motion
# code moves the head, and every mark after it lands in the wrong place -- bad,
# and visible. A stray operand-carrying code eats the next byte, so from there
# every opcode is read out of an operand and every operand out of an opcode:
# the same one-byte-out-of-step failure the workspace CLAUDE.md describes for
# adding an opcode, with every byte still individually legal and the CRC still
# passing over an intact file.

#: Codes that take a following byte when *sent* to the machine. A byte from this
#: set arriving where a character was meant desynchronises the whole stream.
OPERAND_CODES = frozenset(
    {
        0xA1,  # Übertragungsrate  - next byte is the baud code
        0xA3,  # Anschlagstärke    - next byte is the force (SET_STRIKE_FORCE)
        0xA5,  # Wagensteuerung    - next byte is a 1/120" step count
        0xA6,  # Papiervorschub    - next byte is a 1/240" step count
        0xA7,  # Typenrad drehen   - next byte is a 3.6 deg step count
        0xA8,  # Farbbandtransport - next byte is a 10 deg step count
        0xAA,  # BEL               - next byte is the signal length
    }
)

#: Doppeldruck: the character after this one prints without advancing the
#: carriage. The way a stack of glyphs in one cell is typed without spending the
#: escapement's repeatability on a backspace between each pair.
NO_ADVANCE = 0xA9

#: Rückwärtsdruck: while this is on, the head steps one full cell *left* and
#: then strikes, so typing runs right to left. 0x8D puts it back.
#:
#: The order is the whole of it, and section 7 of `erika.pipeline codes` is what
#: settled it: five letters typed from column 20 came out reading EDCBA with the
#: A at column *19*, so the table's "erst Vorschub rückwärts, dann Zeichendruck"
#: is literal. The model is therefore ``x -= 2`` and then mark, which leaves the
#: head standing on the mark rather than one cell past it -- the mirror of
#: forward printing in cost but not in arithmetic.
#:
#: What the sheet did not ask, and so what nothing here may assume: whether the
#: motion keys invert with it, whether 0xA9 still means "print where the head
#: stands", and whether a dead key still declines to feed. The planner keeps a
#: backward run to plain advancing strikes with no motion inside it for exactly
#: that reason, and the emulator refuses the combinations rather than guessing.
BACKWARD_PRINT_ON = 0x8E
BACKWARD_PRINT_OFF = 0x8D

#: 0xA9 (Doppeldruck) does not eat the next byte, it *changes* it: the character
#: after it prints without advancing the carriage. So a stray 0xA9 does not
#: desynchronise the stream, it silently drops one advance -- which shifts
#: everything after it on the line. 0x8E is the same shape of hazard one step
#: further out: it changes what *every* following strike does to the position,
#: until 0x8D. Named here because these are the codes above the motion block
#: that are neither inert nor stream-breaking.
MODIFIER_CODES = frozenset({NO_ADVANCE, BACKWARD_PRINT_ON, BACKWARD_PRINT_OFF})

# --------------------------------------------------------------------------
# What the wheel can be told to type
# --------------------------------------------------------------------------
# The type wheel's codes are a contiguous range with a few unused slots in it,
# and every code above the range is a command of some kind. So the cheap,
# drift-proof test for "could this byte be a glyph" is the range itself -- one
# bound rather than a table -- and the test suite pins it against GLYPHS so it
# cannot quietly stop being true.
MIN_GLYPH_CODE = 0x01
MAX_GLYPH_CODE = 0x67

#: The highest byte a strike force may be. A machine that does not implement
#: SET_STRIKE_FORCE types the force byte as an ordinary character, so a force
#: has to be a byte that is *harmless when typed* -- which means inside the
#: wheel's own range. Above it lies the motion block and then the commands, and
#: seven of those would eat the byte after the force and take the rest of the
#: job with them.
#:
#: This machine does honour the command, so on it the bound is belt-and-braces
#: -- and it turns out to cost nothing anyway, because the force scale saturates
#: at 0x5F, below this. It stays because it is the only thing standing between a
#: mistyped force and half an hour of arbitrary motion on a machine, or a wheel,
#: that answers differently.
MAX_FORCE = MAX_GLYPH_CODE


def is_glyph_code(value: int) -> bool:
    """Could `value` be a key on the wheel rather than a command?

    A range check, not a lookup. The unused slots inside the range are harmless
    -- at worst the machine types nothing -- whereas every byte outside it is a
    motion or a command, and seven of those consume the byte that follows.
    ``glyph_for_code`` is the exact answer where an exact one is wanted.
    """
    return MIN_GLYPH_CODE <= value <= MAX_GLYPH_CODE


def is_usable_force(value: int) -> bool:
    """Can `value` be sent as a force byte without risking anything worse?

    A machine that ignores SET_STRIKE_FORCE types the force byte instead, so the
    value has to be inert as a character. Inside the wheel's range it types a
    glyph and the plan's own position model already accounts for the advance
    (the probe sheet is built to be readable either way). Outside it, the value
    is a motion at best and a command that eats the following byte at worst.

    0 is allowed although it is below the wheel: no key has that code, so a
    machine ignoring the command types nothing rather than a stray character.
    On this one it is a force like any other, and the hardest -- see
    FULL_STRIKE_FORCE.
    """
    return 0 <= value <= MAX_FORCE


# --------------------------------------------------------------------------
# Direct step control
# --------------------------------------------------------------------------
# 0xA5 and 0xA6 drive the carriage and the platen by a count of motor steps
# rather than by a keystroke's worth of movement, and the step is an absolute
# fraction of an inch rather than a fraction of whatever the slide switches are
# set to. That is what makes them worth having: SPACE and the half-step key move
# the carriage by a share of the current pitch, and nothing in a print job can
# check what the pitch switch is actually set to.
#
# None of this has been on paper yet. `python -m erika.pipeline codes` is the
# sheet that answers it.
CARRIAGE_STEPS = 0xA5  #: next byte is a signed count of 1/120" carriage steps
PLATEN_STEPS = 0xA6  #: next byte is a signed count of 1/240" platen steps
WHEEL_STEPS = 0xA7  #: next byte is a signed count of 3.6 deg wheel steps
RIBBON_STEPS = 0xA8  #: next byte is a count of 10 deg ribbon steps

# 0xAA is the one code on this interface that makes a sound instead of a mark,
# and the only output the machine has that does not cost paper or ribbon. The
# next byte is a length, "etwa 20 ms je Einheit".
#
# What the table does *not* contain anywhere is a frequency. The signal
# generator has one pitch and one parameter, so the only musical dimension it
# has is time -- see erika/melody.py, which is the whole instrument.
BELL = 0xAA

#: Measured, on this machine, with `erika.pipeline melody --probe` and a
#: stopwatch: 255 units ran 2.3 s and 200 units 1.8 s, which is 9.02 and 9.00 ms
#: per unit, and a straight line through the pair has a zero offset -- so the
#: length is proportional to the operand with no fixed cost. 127 units timed
#: 1.3 s, or 10.2, which is what hand-stopping does to a shorter interval; the
#: two long readings are the trustworthy ones.
#:
#: **The table says "etwa 20 ms je Einheit" and on the Sigma it is about half
#: that.** That is the S3004 figure, and this is the second thing on this
#: interface to differ from the table rather than the first -- worth remembering
#: the next time a number here is taken on the table's word. A tune written
#: against 20 sounded its notes for 45% of what was asked, which reads as
#: over-articulated rather than as broken, and is exactly the kind of wrongness
#: nothing but a measurement finds.
BELL_UNIT_MS = 9

#: The whole operand, because the sweep settled it: 160, 200 and 255 each came
#: out longer than the one before, so the byte is a plain unsigned length and
#: the high bit is *not* a sign here as it is on 0xA5, 0xA6 and 0xA7. The cap
#: was 127 while that was unread, which cost half the range for nothing.
#:
#: 255 units is 2.3 s, and that is the longest single beep this machine has.
MAX_BELL_UNITS = 255

#: One inch, in each mechanism's own steps.
CARRIAGE_STEPS_PER_INCH = 120
PLATEN_STEPS_PER_INCH = 240

#: Platen step counts the table marks as forbidden ("Die Schritte 3, 4, 5, 6
#: sind verboten!"). 1 and 2 are fine and so is anything from 7 up, so a feed
#: that lands on one of these has to be split -- five steps go out as 2 + 2 + 1.
#: No reason is given; treat it as a property of the mechanism.
FORBIDDEN_PLATEN_STEPS = frozenset({3, 4, 5, 6})


def carriage_steps_per_half_step(pitch: int) -> int:
    """1/120" steps to one half of a character cell.

    Six at pitch 10, five at pitch 12, four at pitch 15 -- all whole, which is
    what lets a half-step be expressed as a step count without rounding. A
    quarter of a cell is three steps at pitch 10 and two at pitch 15, but two
    and a half at pitch 12, which is why quarter-cell offsets are not a
    pitch-12 feature.
    """
    steps = CARRIAGE_STEPS_PER_INCH / (2 * pitch)
    if steps != int(steps):
        raise ValueError(f"pitch {pitch} does not divide the carriage's step")
    return int(steps)


#: 1/240" platen steps to half a line, at line spacing 1. The table gives a full
#: line as 40 motor steps, which is the same number from the other direction.
PLATEN_STEPS_PER_HALF_LINE = PLATEN_STEPS_PER_INCH // 12


def encode_step_operand(steps: int) -> int:
    """Signed step count -> the operand byte, as the table spells it.

    "0…127 Schritte vorwärts; 256-(1..127) Schritte rückwärts" -- two's
    complement in a byte, with 128 unreachable in either direction.
    """
    if not -127 <= steps <= 127:
        raise ValueError(f"{steps} steps is outside the operand's range of ±127")
    return steps & 0xFF


def decode_step_operand(byte: int) -> int:
    """The operand byte -> a signed step count."""
    return byte - 256 if byte > 127 else byte


#: The most one command can move, in each mechanism.
MAX_STEPS_PER_COMMAND = 127


# --------------------------------------------------------------------------
# The slide switches
# --------------------------------------------------------------------------
# Pitch (Schriftteilung) and line spacing (Zeilenschaltung) are slide switches
# on the machine, and every plan this package builds assumes both: the pitch
# decides what a SPACE is worth, and LINE_HEIGHT_MM below is Zeilenschaltung 1.
#
# These six codes are the only ones in the table that travel in **both**
# directions. Sent to the machine, each one sets its switch's setting; received
# from it, each one reports that the operator has just moved that switch. What
# does not exist anywhere in 0x71..0xAF is a code that *asks* -- so a switch's
# position is knowable only from a report or from having been set, and at
# power-on it is knowable not at all.
#
# That asymmetry is why a job pins its pitch rather than checking it, and it is
# the firmware that does so: `IMG PREPARE PITCH`, on by default, puts the pitch
# from the job's header and 0x84 in front of the body, and puts them back if the
# machine reports a switch moving mid-print. See buildPreamble() and
# watchSwitches() in erika_ai/src/erika_image.cpp. There is no operand and no
# motion in any of the six, which is what makes them safe to put in front of a
# plan that did not account for them -- test_erika.py checks that.
#
# Mirrors ERIKA_PITCH_* and ERIKA_LINE_SPACING_* in erika_ai/src/erika_image.h.
PITCH_10 = 0x87
PITCH_12 = 0x88
PITCH_15 = 0x89
LINE_SPACING_1 = 0x84
LINE_SPACING_1_5 = 0x85
LINE_SPACING_2 = 0x86

#: Pitch a code selects, or reports the switch as being at.
PITCH_FOR_CODE = {PITCH_10: 10, PITCH_12: 12, PITCH_15: 15}

#: Line spacing a code selects, in tenths of a line so that 1,5 is an integer --
#: which is also how the firmware reports it, for the same reason.
SPACING_FOR_CODE = {LINE_SPACING_1: 10, LINE_SPACING_1_5: 15, LINE_SPACING_2: 20}


def pitch_code(pitch: int) -> int:
    """The code that pins `pitch`, whether as a command or read as a report."""
    for code, value in PITCH_FOR_CODE.items():
        if value == pitch:
            return code
    raise ValueError(f"no pitch code for {pitch} characters per inch")


# --------------------------------------------------------------------------
# Physical geometry
# --------------------------------------------------------------------------
#: Character cell width in mm, by pitch (Schriftteilung 10 / 12).
#:
#: 15 is deliberately absent although PITCH_FOR_CODE has it and the machine
#: answers to it: the .etp header carries pitch as one flag bit, so a job can
#: only ask for 10 or 12. See "What pitch 15 needs" in docs/control-codes.md.
PITCH_WIDTH_MM = {10: 25.4 / 10, 12: 25.4 / 12}

#: Line advance in mm at Zeilenschaltung 1 (6 lines per inch).
LINE_HEIGHT_MM = 25.4 / 6

#: Widest line the carriage can type, by pitch.
#: Mirrors MAX_LINE_LENGTH_10 / MAX_LINE_LENGTH_12 in erika_ai/src/config.h.
MAX_COLUMNS = {10: 65, 12: 78}


def cell_aspect(pitch: int) -> float:
    """Height / width of one character cell at the given pitch."""
    return LINE_HEIGHT_MM / PITCH_WIDTH_MM[pitch]


# --------------------------------------------------------------------------
# Glyphs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Glyph:
    """One typeable character on the Sigma's daisy wheel."""

    char: str  #: Unicode character, as rendered into the charset sheet
    code: int  #: Byte to send over the serial link
    advances: bool = True  #: False for "ohne Vorschub" dead keys
    name: str = ""  #: Human-readable label for disassembly

    def __post_init__(self):
        if not 0 <= self.code <= 0xFF:
            raise ValueError(f"code out of range for {self.char!r}: {self.code}")


# The blank cell. chop_charset() always prepends a blank tile at index 0, so
# SPACE is handled specially by the planner and is *not* part of GLYPHS.
BLANK = Glyph(" ", SPACE, True, "space")

#: Every printable, advancing key, in the order they appear in
#: erika_char_map.cpp's getErikaChar(). Order defines charset-sheet order,
#: which in turn defines the character indices in choices.json -- so appending
#: is safe, reordering is not (it invalidates previously generated charsets).
GLYPHS: tuple[Glyph, ...] = (
    # Punctuation
    Glyph("!", 0x42),
    Glyph('"', 0x43),
    Glyph("#", 0x41),
    Glyph("$", 0x48),
    Glyph("%", 0x04),
    Glyph("&", 0x02),
    Glyph("'", 0x17),
    Glyph("(", 0x1D),
    Glyph(")", 0x1F),
    Glyph("*", 0x1B),
    Glyph("+", 0x25),
    Glyph(",", 0x64),
    Glyph("-", 0x62),
    Glyph("_", 0x01),
    Glyph(".", 0x63),
    Glyph("/", 0x40),
    # Digits
    Glyph("0", 0x0D),
    Glyph("1", 0x11),
    Glyph("2", 0x10),
    Glyph("3", 0x0F),
    Glyph("4", 0x0E),
    Glyph("5", 0x0C),
    Glyph("6", 0x0B),
    Glyph("7", 0x0A),
    Glyph("8", 0x09),
    Glyph("9", 0x08),
    # More punctuation
    Glyph(":", 0x13),
    Glyph(";", 0x3B),
    Glyph("=", 0x2E),
    Glyph("?", 0x35),
    # Upper case
    Glyph("A", 0x30),
    Glyph("B", 0x18),
    Glyph("C", 0x20),
    Glyph("D", 0x14),
    Glyph("E", 0x34),
    Glyph("F", 0x3E),
    Glyph("G", 0x1C),
    Glyph("H", 0x12),
    Glyph("I", 0x21),
    Glyph("J", 0x32),
    Glyph("K", 0x24),
    Glyph("L", 0x2C),
    Glyph("M", 0x16),
    Glyph("N", 0x2A),
    Glyph("O", 0x1E),
    Glyph("P", 0x2F),
    Glyph("Q", 0x1A),
    Glyph("R", 0x36),
    Glyph("S", 0x33),
    Glyph("T", 0x37),
    Glyph("U", 0x28),
    Glyph("V", 0x22),
    Glyph("W", 0x2D),
    Glyph("X", 0x26),
    Glyph("Y", 0x31),
    Glyph("Z", 0x38),
    # Lower case
    Glyph("a", 0x61),
    Glyph("b", 0x4E),
    Glyph("c", 0x57),
    Glyph("d", 0x53),
    Glyph("e", 0x5A),
    Glyph("f", 0x49),
    Glyph("g", 0x60),
    Glyph("h", 0x55),
    Glyph("i", 0x05),
    Glyph("j", 0x4B),
    Glyph("k", 0x50),
    Glyph("l", 0x4D),
    Glyph("m", 0x4A),
    Glyph("n", 0x5C),
    Glyph("o", 0x5E),
    Glyph("p", 0x5B),
    Glyph("q", 0x52),
    Glyph("r", 0x59),
    Glyph("s", 0x58),
    Glyph("t", 0x56),
    Glyph("u", 0x5D),
    Glyph("v", 0x4F),
    Glyph("w", 0x4C),
    Glyph("x", 0x5F),
    Glyph("y", 0x51),
    Glyph("z", 0x54),
    # Symbols
    Glyph("|", 0x27),
    Glyph("£", 0x06),
    Glyph("§", 0x3D),
    Glyph("°", 0x39),
    Glyph("²", 0x15),
    Glyph("³", 0x23),
    Glyph("µ", 0x07),
    # Umlauts and accented characters
    Glyph("Ä", 0x3F),
    Glyph("Ö", 0x3C),
    Glyph("Ü", 0x3A),
    Glyph("ß", 0x47),
    Glyph("ä", 0x65),
    Glyph("ç", 0x45),
    Glyph("è", 0x46),
    Glyph("é", 0x44),
    Glyph("ö", 0x66),
    Glyph("ü", 0x67),
)

#: "Ohne Vorschub" dead keys: they print a mark but leave the carriage put.
#: Useful for shading (they are the lightest marks the machine can make) but
#: they complicate the position model, so make_charset excludes them unless
#: --dead-keys is passed.
DEAD_KEY_GLYPHS: tuple[Glyph, ...] = (
    Glyph("¨", 0x03, False, "diaeresis"),  # ¨
    Glyph("^", 0x19, False, "circumflex"),
    Glyph("´", 0x29, False, "acute"),  # ´
    Glyph("`", 0x2B, False, "grave"),
)


def all_glyphs(dead_keys: bool = False) -> tuple[Glyph, ...]:
    return GLYPHS + DEAD_KEY_GLYPHS if dead_keys else GLYPHS


_BY_CODE = {g.code: g for g in GLYPHS + DEAD_KEY_GLYPHS}
_BY_CODE[SPACE] = BLANK

_BY_CHAR = {g.char: g for g in GLYPHS + DEAD_KEY_GLYPHS}
_BY_CHAR[" "] = BLANK


def glyph_for_code(code: int) -> Glyph | None:
    return _BY_CODE.get(code)


def glyph_for_char(char: str) -> Glyph | None:
    """Look up a key by the character it types, or None if the Sigma lacks it."""
    return _BY_CHAR.get(char)


#: Names for the interface's control codes, for diagnostics only.
#:
#: The motion block is what the pipeline sends; the rest is here so that a byte
#: which has gone somewhere it should not can be *named* in the error. A message
#: that says 0xA5 has to be read against a table; one that says "direct carriage
#: control, and it takes the next byte with it" explains the failure.
#:
#: Only the codes worth recognising, not the whole table -- see
#: erika_ai/ressources/steuercodes.md for that.
CONTROL_CODE_NAMES = {
    SPACE: "SPACE",
    BACKSPACE: "BACKSPACE",
    HALF_STEP_FORWARD: "HALF_STEP_FWD",
    HALF_STEP_BACK: "HALF_STEP_BACK",
    HALF_LINE_FORWARD: "HALF_LINE_FWD",
    HALF_LINE_BACK: "HALF_LINE_BACK",
    NEWLINE: "NEWLINE",
    CARRIAGE_RETURN: "CR",
    TAB: "TAB",
    MICRO_LINE_FORWARD: "MICRO_FWD",
    MICRO_LINE_BACK: "MICRO_BACK",
    0x7A: "SET_TAB",
    0x7B: "CLEAR_TAB",
    0x7C: "CLEAR_ALL_TABS",
    0x7D: "SET_TAB_GRID",
    0x7E: "SET_LEFT_MARGIN",
    0x7F: "SET_RIGHT_MARGIN",
    0x80: "RELEASE_MARGINS",
    0x83: "FEED_SHEET",
    LINE_SPACING_1: "LINE_SPACING_1",
    LINE_SPACING_1_5: "LINE_SPACING_1_5",
    LINE_SPACING_2: "LINE_SPACING_2",
    PITCH_10: "PITCH_10",
    PITCH_12: "PITCH_12",
    PITCH_15: "PITCH_15",
    0x8B: "CORRECTION_OFF",
    0x8C: "CORRECTION_ON",
    BACKWARD_PRINT_OFF: "BACKWARD_PRINT_OFF",
    BACKWARD_PRINT_ON: "BACKWARD_PRINT_ON",
    0x8F: "MARGIN_RELEASE_ON",
    0x91: "KEYBOARD_OFF",
    0x92: "KEYBOARD_ON",
    0x95: "RESET",
    0x96: "REPORT_WHEN_PRINTED",
    0x9B: "AUTOREPEAT_ON",
    0x9C: "AUTOREPEAT_OFF",
    0x9F: "LINE_FEED",
    0xA1: "SET_BAUD",
    SET_STRIKE_FORCE: "SET_STRIKE_FORCE",
    0xA5: "CARRIAGE_STEPS",
    0xA6: "PLATEN_STEPS",
    0xA7: "WHEEL_STEPS",
    0xA8: "RIBBON_STEPS",
    0xA9: "NO_ADVANCE",
    BELL: "BELL",
}


def describe_code(code: int) -> str:
    """Human-readable label for one raw typewriter byte (for disassembly)."""
    g = _BY_CODE.get(code)
    if g is not None and code not in CONTROL_CODES:
        return g.name or repr(g.char)
    name = CONTROL_CODE_NAMES.get(code)
    if name is None:
        return f"0x{code:02X}"
    if code in OPERAND_CODES:
        return f"{name}+operand"
    return name


def _check_unique():
    seen: dict[int, str] = {}
    for g in GLYPHS + DEAD_KEY_GLYPHS:
        if g.code in CONTROL_CODES:
            raise AssertionError(f"{g.char!r} collides with a control code")
        if not is_glyph_code(g.code):
            # MIN/MAX_GLYPH_CODE is what both this module and the firmware use
            # to decide whether a byte could be a key at all. A glyph outside it
            # would be refused as a command by the very guard meant to protect
            # it, so the bound has to move first.
            raise AssertionError(
                f"{g.char!r} is 0x{g.code:02X}, outside the wheel's range "
                f"0x{MIN_GLYPH_CODE:02X}..0x{MAX_GLYPH_CODE:02X}"
            )
        if g.code in seen:
            raise AssertionError(
                f"duplicate code 0x{g.code:02X}: {seen[g.code]!r} and {g.char!r}"
            )
        seen[g.code] = g.char


_check_unique()
