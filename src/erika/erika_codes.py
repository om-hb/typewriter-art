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

#: Codes that must never appear as a glyph strike.
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
# Physical geometry
# --------------------------------------------------------------------------
#: Character cell width in mm, by pitch (Schriftteilung 10 / 12).
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


def describe_code(code: int) -> str:
    """Human-readable label for one raw typewriter byte (for disassembly)."""
    control = {
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
    }
    if code in control:
        return control[code]
    g = _BY_CODE.get(code)
    if g is None:
        return f"0x{code:02X}"
    return g.name or repr(g.char)


def _check_unique():
    seen: dict[int, str] = {}
    for g in GLYPHS + DEAD_KEY_GLYPHS:
        if g.code in CONTROL_CODES:
            raise AssertionError(f"{g.char!r} collides with a control code")
        if g.code in seen:
            raise AssertionError(
                f"duplicate code 0x{g.code:02X}: {seen[g.code]!r} and {g.char!r}"
            )
        seen[g.code] = g.char


_check_unique()
