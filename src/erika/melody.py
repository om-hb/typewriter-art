"""Tunes on the typewriter's own beeper.

The instrument is one control code. `0xAA` takes a single operand byte -- a
length, about 9 ms per unit on this machine -- and there is no frequency
anywhere in the
interface's table. So the signal generator has exactly **one pitch**, and the
only musical dimension this machine has is time.

That is not the limitation it sounds like. A tune carried entirely by rhythm is
still recognisable -- *shave and a haircut* has no pitch worth speaking of and
nobody mishears it -- but it does decide the shape of everything here: a
"melody" is a sequence of durations, notes differ from each other only in how
long they sound, and the thing that makes two notes two notes rather than one
is the **silence between them**. Articulation is not a nicety on this machine,
it is the only way a rhythm survives at all. Ask for a run of quarter notes
that fill their own slots and the beeper produces one continuous tone.

Three numbers bound what can be played, and all three come from somewhere:

- **A beep is 9 ms per unit** (`ec.BELL_UNIT_MS`), so that is the resolution.
  Measured with `--probe`; the table's own figure is 20 and is wrong here.
- **A note occupies at least 200 ms.** The bell is two bytes and the firmware
  charges a full character delay for each byte it cannot recognise as motion
  (`RAW_BYTE_COST_MS`, mirroring `ERIKA_CHAR_DELAY_MS`). Nothing the host does
  makes a note arrive sooner than that, which puts a hard ceiling on tempo --
  see `max_tempo_for`.
- **A beep is at most 255 units**, 2.3 s (`ec.MAX_BELL_UNITS`) -- the whole
  operand, the sweep having shown the high bit is not a sign here. Longer notes
  hold their slot and simply stop sounding partway through.

All of this has now been heard on the machine, which is why the numbers above
are not the table's. `--probe` timed the unit at 9 ms and showed the operand to
be unsigned, and the gaps between its beeps came out where the pacing model put
them -- 1.55 s measured against 1.62 s predicted for the short ones, which is
the reading that confirms a delay and a byte gap *overlap* rather than add.

One thing the stopwatch could not settle: the sweep as a whole ran 40-45 s
against 38.5 s predicted, and the excess sat in the beeps with the longest
delays -- the ones over 2550 ms, which `delay_ms` splits into several ETP_DELAY
opcodes. A chain of delays looks like it costs a little more than its parts.
It is a few percent and it needs an instrument rather than a wristwatch, so it
is written down here rather than compensated for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from erika import erika_codes as ec
from erika import etp


class MelodyError(ValueError):
    pass


#: What the firmware charges for one byte it has no model for. `ETP_RAW` is
#: paced at `ERIKA_CHAR_DELAY_MS` because a mode switch and an inch of carriage
#: travel look identical from the interpreter's side, and a bell command is two
#: such bytes. Mirrored by hand from `erika_ai/src/erika_image.h`, and guarded
#: by a drift test like every other constant that crosses the repository line.
RAW_BYTE_COST_MS = 100

#: Onset to onset, the shortest a note can be: the command byte, then the
#: operand byte, each with a character delay behind it. A slot shorter than this
#: is not slow playing, it is a promise the device cannot keep -- so it is
#: refused rather than rounded up, which would silently change the rhythm.
MIN_SLOT_MS = 2 * RAW_BYTE_COST_MS

#: The shortest silence that still reads as a gap rather than as one long note.
#: This was two bell units while the unit was believed to be 20 ms, and the
#: measurement pulled it to 18 -- which is not a gap anybody hears. What it was
#: always expressing is an audibility floor, so it says so as a number: the
#: generator's resolution and what an ear resolves are different quantities and
#: only one of them belongs here.
MIN_GAP_MS = 40

#: Milliseconds to one character of `score()`. Not `ec.BELL_UNIT_MS`, which it
#: used to be: at 9 ms a second of tune is 111 characters and a bar wraps three
#: times before it can be read. What the picture wants is a resolution fine
#: enough to show the shortest gap that matters and coarse enough to fit a
#: phrase on a line, and that is a different quantity from what the hardware
#: can resolve -- the same distinction `MIN_GAP_MS` makes.
MS_PER_SCORE_CHAR = 25

#: Fraction of its slot a note sounds for. 0.6 leaves a gap two fifths as long
#: as the note, which is about how a staccato reads; the rest of the slot is
#: what separates this note from the next one.
DEFAULT_GATE = 0.6

#: Quarter notes per minute. Slow, because the floor above is unusually high for
#: an instrument -- at 120 the shortest playable note is an eighth.
DEFAULT_TEMPO = 100


@dataclass(frozen=True)
class Event:
    """One slot in the score: how long it lasts, and how much of it sounds.

    Both are kept, rather than a duration and a flag, because the gap is the
    part that carries the rhythm and it deserves to be visible to whatever
    builds an `Event` -- Morse shapes its own gaps and does not want a gate
    fraction applied on top.
    """

    slot_ms: int  #: onset to the next onset
    sound_ms: int  #: 0 for a rest
    label: str = ""

    @property
    def is_rest(self) -> bool:
        return self.sound_ms == 0

    @property
    def gap_ms(self) -> int:
        return self.slot_ms - self.sound_ms


@dataclass
class Melody:
    name: str = ""
    events: list[Event] = field(default_factory=list)

    @property
    def total_ms(self) -> int:
        return sum(e.slot_ms for e in self.events)

    @property
    def beeps(self) -> int:
        return sum(1 for e in self.events if not e.is_rest)


# ---------------------------------------------------------------------------
# notation
# ---------------------------------------------------------------------------

#: Note values, in quarter notes. The letters are the usual abbreviations for
#: whole/half/quarter/eighth/sixteenth.
NOTE_VALUES = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25}

_TOKEN = re.compile(
    r"""^
    (?P<rest>-)?                     # a leading dash makes the slot silent
    (?: (?P<letter>[whqes])(?P<dots>\.*)   # q, e, h. -- a note value
      | (?P<ms>\d+)ms )              # or a duration in milliseconds
    $""",
    re.X,
)


def quarter_ms(tempo: float) -> float:
    """Milliseconds in one quarter note at `tempo` beats per minute."""
    if tempo <= 0:
        raise MelodyError(f"tempo {tempo} is not a number of beats per minute")
    return 60_000.0 / tempo


def max_tempo_for(value: str = "q") -> float:
    """The fastest tempo at which `value` still clears the 200 ms floor.

    The number worth knowing before writing anything: at the default gate a
    sixteenth note needs a tempo of 75 to be playable at all, which is slower
    than most things are written. Rhythms for this machine want to be written
    in quarters and eighths.
    """
    beats = NOTE_VALUES.get(value)
    if beats is None:
        raise MelodyError(f"{value!r} is not one of {''.join(NOTE_VALUES)}")
    return 60_000.0 * beats / MIN_SLOT_MS


def parse(text: str, tempo: float = DEFAULT_TEMPO,
          gate: float = DEFAULT_GATE, name: str = "") -> Melody:
    """Read a rhythm.

    Whitespace-separated tokens, one per slot:

    | token | slot |
    |---|---|
    | `q` `e` `h` `w` `s` | a note of that value |
    | `q.` | dotted -- one and a half times as long; `q..` doubly dotted |
    | `-q` | a rest of that value |
    | `350ms` | a slot of exactly that long, for anything the values cannot say |
    | `\\|` | bar line, ignored |
    | `# ...` | comment to end of line |

    `gate` is how much of a note's slot sounds. It applies to notes written as
    note values *and* to `350ms` slots, so a rhythm can be retimed by changing
    the tempo alone.
    """
    if not 0 < gate < 1:
        raise MelodyError(
            f"gate {gate} has to be between 0 and 1 -- at 1 a note fills its "
            "whole slot and runs into the next one, and with one pitch that "
            "is a single long beep rather than a rhythm"
        )
    quarter = quarter_ms(tempo)
    events: list[Event] = []
    for raw in re.sub(r"#[^\n]*", " ", text).replace("|", " ").split():
        m = _TOKEN.match(raw)
        if m is None:
            raise MelodyError(
                f"{raw!r} is not a note: expected one of "
                f"{' '.join(sorted(NOTE_VALUES))}, optionally dotted, "
                "optionally led by '-' for a rest, or a count like '350ms'"
            )
        if m["ms"]:
            slot = float(m["ms"])
        else:
            beats = NOTE_VALUES[m["letter"]]
            # A dot adds half of what stands before it, so two dots add a
            # half and then a quarter -- 1.75, not 2.
            slot = beats * quarter * (2 - 0.5 ** len(m["dots"]))
        slot_ms = int(round(slot))
        sound = 0 if m["rest"] else int(round(slot_ms * gate))
        events.append(Event(slot_ms=slot_ms, sound_ms=sound, label=raw))
    if not events:
        raise MelodyError("nothing to play")
    return Melody(name=name, events=events)


# ---------------------------------------------------------------------------
# Morse
# ---------------------------------------------------------------------------

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "-": "-....-",
    "=": "-...-", ":": "---...", "'": ".----.", "!": "-.-.--", "@": ".--.-.",
}

#: Morse's own unit, in milliseconds. It has to clear `MIN_SLOT_MS` on its own,
#: because a dot's slot is one unit of sound and one of silence -- 100 would be
#: exactly the floor, and 120 leaves the arithmetic somewhere to round.
DEFAULT_MORSE_UNIT_MS = 120


def morse(text: str, unit_ms: int = DEFAULT_MORSE_UNIT_MS) -> Melody:
    """Spell `text` out in Morse.

    The one thing a single-pitch signal generator was actually built for, and
    the case where this machine is not making do: Morse *is* a rhythm, so
    nothing is lost in the translation.

    Standard proportions -- dash three units, symbols separated by one, letters
    by three, words by seven -- with the separations folded into each symbol's
    slot so that the last gap of a letter is the letter gap and not one more on
    top of it.
    """
    if unit_ms * 2 < MIN_SLOT_MS:
        raise MelodyError(
            f"a Morse unit of {unit_ms} ms puts a dot's slot at {unit_ms * 2} "
            f"ms, under the {MIN_SLOT_MS} ms the device needs for one beep"
        )
    events: list[Event] = []
    words = text.upper().split()
    for w, word in enumerate(words):
        if w:
            events.append(Event(slot_ms=4 * unit_ms, sound_ms=0, label="word"))
        for c, char in enumerate(word):
            symbols = MORSE.get(char)
            if symbols is None:
                raise MelodyError(f"{char!r} has no Morse code in this table")
            if c:
                events.append(Event(slot_ms=2 * unit_ms, sound_ms=0, label="/"))
            for s, symbol in enumerate(symbols):
                sound = unit_ms * (3 if symbol == "-" else 1)
                events.append(Event(slot_ms=sound + unit_ms, sound_ms=sound,
                                    label=f"{char}{symbol}" if s == 0 else symbol))
    if not events:
        raise MelodyError("nothing to spell")
    return Melody(name=f"morse {text}", events=events)


# ---------------------------------------------------------------------------
# tunes
# ---------------------------------------------------------------------------

#: Rhythms that survive losing their pitch, which is the only kind worth
#: shipping. Each is (notation, tempo) -- the tempo is part of the tune, since
#: a rhythm played at the wrong speed stops being recognisable well before it
#: stops being playable.
TUNES: dict[str, tuple[str, float]] = {
    # The canonical pitchless tune. Nobody has ever needed the notes.
    "shave": ("q  e e q | q  -q | q  q", 132),
    # Stomp stomp clap, and the rest that makes it work.
    "rock-you": ("e e q | -q -q | e e q | -q -q", 100),
    # Two short, one long: the sound of something having finished.
    "tada": ("e e h", 120),
    # A question. Rising in rhythm rather than in pitch: short, short, long.
    "prompt": ("s s q", 75),
    # Three even beats, for "your attention please".
    "alert": ("q q q", 120),
    "sos": ("s s s -s | q q q -s | s s s", 75),
}


def tune(name: str, tempo: float | None = None,
         gate: float = DEFAULT_GATE) -> Melody:
    """One of the built-in rhythms, at its own tempo unless one is given."""
    if name not in TUNES:
        raise MelodyError(
            f"no tune called {name!r} -- have {', '.join(sorted(TUNES))}"
        )
    text, own_tempo = TUNES[name]
    return parse(text, tempo=own_tempo if tempo is None else tempo,
                 gate=gate, name=name)


# ---------------------------------------------------------------------------
# compiling
# ---------------------------------------------------------------------------


def units_for(sound_ms: int) -> int:
    """A note's sounding length, in bell units, as it will actually come out.

    Never rounds to zero: a note quantised out of existence would be a silent
    slot in the middle of a rhythm, which reads as a mistake rather than as a
    short note.
    """
    units = int(round(sound_ms / ec.BELL_UNIT_MS))
    return max(1, min(ec.MAX_BELL_UNITS, units))


def check(melody: Melody) -> list[str]:
    """Everything about a melody that the device will not play as written.

    Returned rather than raised, because they are not all the same kind of
    problem: a slot under the floor is unplayable and a clipped whole note is
    merely not what was asked for. `compile_to` refuses the first and lets the
    second through.
    """
    problems = []
    for i, e in enumerate(melody.events):
        where = f"note {i + 1}" + (f" ({e.label})" if e.label else "")
        if e.slot_ms < MIN_SLOT_MS and not (e.is_rest and e.slot_ms > 0):
            problems.append(
                f"{where}: a {e.slot_ms} ms slot is under the {MIN_SLOT_MS} ms "
                f"the device needs to put out one bell command. Play it slower "
                f"-- {max_tempo_for('q'):.0f} BPM is the fastest a quarter note "
                f"can go, {max_tempo_for('e'):.0f} for an eighth"
            )
            continue
        if e.is_rest:
            continue
        if e.gap_ms < MIN_GAP_MS:
            problems.append(
                f"{where}: only {e.gap_ms} ms of silence after it. With one "
                f"pitch that runs into the next note -- lower the gate"
            )
        if e.sound_ms > ec.MAX_BELL_UNITS * ec.BELL_UNIT_MS:
            problems.append(
                f"{where}: asks for {e.sound_ms} ms of sound and the bell stops "
                f"at {ec.MAX_BELL_UNITS * ec.BELL_UNIT_MS} ms. The slot keeps "
                f"its length; the note just goes quiet early"
            )
    return problems


def compile_to(enc: etp.Encoder, melody: Melody) -> None:
    """Emit a melody into an opcode stream.

    The timing is the whole of this function, so here is where it comes from.
    Taking the moment the bell's command byte goes out as the start of a slot:

    - the operand cannot follow for `RAW_BYTE_COST_MS`, so the beep starts one
      character delay *into* its own slot. Every note is displaced by the same
      amount, which is why this shifts nothing musically;
    - the beep then sounds for its own length, on the machine, while the host
      goes on sending;
    - the firmware takes the longer of the byte's own delay and any `ETP_DELAY`
      standing at the time, and the two are set within a millisecond of each
      other rather than end to end. So a slot of `D` wants a delay of
      `D - RAW_BYTE_COST_MS`, and `D` cannot go below `MIN_SLOT_MS`.

    The silence a listener actually hears between two beeps is therefore
    `slot - sound`, cleanly, with the two hundred milliseconds of lead-in
    cancelling out of the difference.
    """
    problems = [p for p in check(melody) if "under the" in p]
    if problems:
        raise MelodyError(problems[0])
    for e in melody.events:
        if e.is_rest:
            enc.delay_ms(e.slot_ms)
            continue
        enc.raw_command(ec.BELL, units_for(e.sound_ms))
        enc.delay_ms(e.slot_ms - RAW_BYTE_COST_MS)


def to_job(melody: Melody) -> etp.Job:
    """A complete .etp job that types nothing and makes a noise.

    `cols`, `rows` and `strikes` are all zero, which is honest rather than
    awkward: nothing reaches the paper, the firmware's progress report already
    guards a zero strike count, and a job that claimed a column would have the
    device reserve carriage width for a tune.
    """
    enc = etp.Encoder()
    compile_to(enc, melody)
    enc.end()
    return etp.Job(body=enc.body(), cols=0, rows=0, strikes=0)


def score(melody: Melody, width: int = 60) -> str:
    """The rhythm as a picture, at `MS_PER_SCORE_CHAR` to the character.

    A melody job cannot be checked against a mockup the way a print job can --
    there is nothing on the paper to compare. This is the substitute: what the
    machine will do, in the order it will do it, in a form somebody can read
    against the tune in their head before spending the machine's time on it.
    """
    lines = [f"{melody.name or 'melody'}: {len(melody.events)} slots, "
             f"{melody.beeps} beeps, {melody.total_ms / 1000:.2f} s"]
    per_char = MS_PER_SCORE_CHAR
    row = ""
    for e in melody.events:
        sound = units_for(e.sound_ms) * ec.BELL_UNIT_MS if not e.is_rest else 0
        cells = max(1, int(round(e.slot_ms / per_char)))
        on = min(cells, int(round(sound / per_char)))
        row += "#" * on + "." * (cells - on)
    for i in range(0, len(row), width):
        at = (i * per_char) / 1000
        lines.append(f"{at:6.2f}s |{row[i:i + width]}")
    for problem in check(melody):
        lines.append(f"  ! {problem}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

#: Lengths the probe asks for. Doubling up to the high bit, then four values
#: past it -- the second half was the point of the sweep, and it has been
#: answered: 160, 200 and 255 each came out longer than the one before, so the
#: operand is a plain unsigned length and `ec.MAX_BELL_UNITS` is the whole byte.
#:
#: Kept as it is, because it is also the instrument that timed the unit and the
#: only thing that would notice either answer changing -- a machine serviced, a
#: different Sigma, a firmware whose pacing moved.
PROBE_UNITS = (1, 2, 4, 8, 16, 32, 64, 96, 127, 160, 200, 255)

#: Silence between probe beeps. Long enough that a beep and the next one cannot
#: be confused for one sound even if a length overruns wildly, which is exactly
#: the case the sweep is looking for.
PROBE_GAP_MS = 1500


def probe_job(units=PROBE_UNITS) -> etp.Job:
    """A sweep of bell lengths, for timing the unit and finding where it stops.

    Deliberately outside the `Event` machinery, which clamps to
    `ec.MAX_BELL_UNITS`: a clamp would quietly turn part of the sweep into
    identical beeps, and the whole value of it is that every step differs.

    Safe to send. The operand of a bell is a length whatever its value, so an
    unknown one is a wrong-sounding beep and not a desynchronised stream --
    unlike a byte that lands where a command was expected.

    What to listen for: twelve beeps, each longer than the last. Time one of
    the long ones and divide by its unit count -- that is `ec.BELL_UNIT_MS`,
    and it came out 9 rather than the table's 20. Time a gap between two early
    beeps as well; ~1.6 s is the pacing model holding.
    """
    enc = etp.Encoder()
    for n in units:
        enc.raw_command(ec.BELL, n)
        enc.delay_ms(n * ec.BELL_UNIT_MS + PROBE_GAP_MS)
    enc.end()
    return etp.Job(body=enc.body(), cols=0, rows=0, strikes=0)
