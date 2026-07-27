# Sigma SM 8200i back-end

Turns a photograph into something the [erika_ai](../../../erika_ai) firmware can
type on a Sigma SM 8200i electric typewriter.

`optimize.py` produces overlapping layers of characters and a `choices.json`
saying which character goes in which cell of which layer. That is a picture,
not a set of instructions: it says nothing about which *key* to press, and it
assumes a typist who can nudge the paper by half a character. This package
supplies both halves — a charset that carries the key mapping, and a motion
planner that turns the layers into carriage and platen moves.

```
photo ──► optimize.py ──► choices.json ──► planner ──► photo.etp ──► ESP32 ──► paper
              ▲                                │
        charsets/sigma-10                      └──► preview + jitter preview
        (glyph sheet + index→key map)
```

## Quick start

```bash
cd typewriter-art
python3.10 -m venv .venv
.venv/bin/pip install -r src/requirements-erika.txt
cd src

# 1. Build a charset that knows which key makes which mark
../.venv/bin/python -m erika.pipeline charset --pitch 10

# 2. Check the machine actually does what the plan assumes  (see Calibration)
../.venv/bin/python -m erika.pipeline calibrate
../.venv/bin/python -m erika.send results/calibrate.etp --print

# 3. Convert a photo
../.venv/bin/python -m erika.pipeline print -t images/mwdog_crop.png -r 48

# 4. Type it
../.venv/bin/python -m erika.send results/photo.etp --print --watch
```

Step 3 also writes `results/erika_plan.png` (what the plan puts on paper) and
`results/erika_plan_jitter.png` (the same with realistic registration error).
Look at the second one before committing an hour of ribbon.

## Why a new charset

The charsets that ship with typewriter-art are photographs of type from a
Hermes, a Smith-Corona and a Daisywriter. `chop_charset()` slices them into a
grid and refers to each glyph by its position in that grid — so a `choices.json`
entry of `47` means "the 47th surviving tile of that scan", and there is no
record anywhere of which key produces it. Fine for a mockup, useless for
driving a machine.

`make_charset.py` builds a charset where that mapping is the point:

| file | what it is |
|---|---|
| `sigma.png` | the glyph sheet, one cell per typeable character |
| `config.json` | what `chop_charset()` consumes |
| `glyphs.json` | index → `{char, code, advances}` — the actual mapping |
| `preview.png` | labelled contact sheet, each cell tagged `index:code` |

Cell geometry is the machine's real geometry: cell width = one pitch step
(2.54 mm at pitch 10), cell height = one line at Zeilenschaltung 1 (4.23 mm).
That is what makes a `0.5` offset in `layers.json` mean exactly one half-step
or one half-line.

The generator finishes by running the upstream `prep_charset()` and asserting
the tile count it gets back. `chop_charset()` silently drops tiles it judges
blank, which would shift every index after the dropped one — the mapping would
still look plausible and the machine would type nonsense. Better to fail loudly
at build time.

Glyph shapes come from Courier New by default, which is an approximation of the
Sigma's own face. For a faithful charset, type the calibration sheet on the
machine and build from the scan:

```bash
python -m erika.pipeline sheet                    # writes results/charset_sheet.etp
python -m erika.send results/charset_sheet.etp --print
# scan the block of type, cropped square-on to the outermost ink
python -m erika.pipeline charset --name sigma-scanned --from-scan scan.png
python -m erika.pipeline print -c sigma-scanned -t photo.png
```

## What the planner does

The optimizer's coordinates are `(layer, row, col)`. The machine's are "how
far is the carriage from the left margin" and "how far has the platen turned".
`planner.py` converts between them in units of half a cell:

- `x` — half-steps right of the left margin
- `y` — half-lines below the top of the image

A layer at offset `(0.5, 0.5)` is simply `y += 1, x += 1`. All layers flatten
into one list of absolute strikes, blanks dropped.

Ordering matters mechanically:

- **The paper only ever feeds forward.** Strikes are sorted by `y` first.
  Reversing the platen introduces backlash, which shows up as banding.
- **The carriage returns to the left margin for each pass** and steps out to
  the first strike, rather than carrying accumulated position across rows. It
  costs travel time and buys registration. `--no-home` turns this off and
  sweeps serpentine instead, which is faster and less accurate.
- **Whole-line gaps use the line-feed mechanism** (`NEWLINE`) rather than two
  half-line steps, because the detented full-line advance is more repeatable.

Constraints are checked rather than assumed: layer offsets that aren't
multiples of 0.5 are rejected (so `16x1` and `daisy_full` are out — the machine
has no quarter-step), and an image wider than the carriage (65 columns at pitch
10, 78 at pitch 12) is rejected with the `-r` value that would fit.

## The `.etp` print job

A flat opcode stream. All geometry is resolved on the host; the firmware is a
dumb interpreter. See [`etp.py`](etp.py) for the layout — 24-byte header with a
CRC-32 over the body, then one-byte opcodes with at most one operand.

```
$ python -m erika.etp results/photo.etp -n 8
; ETP1  grid 49x41  strikes 5104  pitch 10  home_each_row True
;  offset  col   row  opcode
       0    0.0   0.0  CR
       2    0.0   1.5  DOWN      3
       4    3.0   1.5  RIGHT     6
       6    4.0   1.5  STRIKE    '3'
```

The two position columns are a running model of where the head is, so a plan
can be checked without a typewriter.

## Verification

The chain is long and every stage can silently ruin the picture, so each one is
checked against the stage before it.

**The plan reproduces the optimizer's mockup.** `preview.render()` composites
the planned strikes back into an image using the same glyph tiles
`optimize.py` used. The two are compared automatically at the end of a `print`
run:

```
  plan reproduces optimize.py's mockup exactly
```

Anything less than exact means the flattening, the index mapping or the offset
arithmetic is wrong. Run it on its own with `python -m erika.pipeline verify`.

**A virtual typewriter reproduces the plan.** `emulate.py` expands the opcodes
into the raw bytes the firmware sends, then consumes those bytes the way the
machine does — moving a carriage, turning a platen, recording where each strike
lands. The test suite checks that the marks land exactly where the plan said,
for one layer and four, with and without carriage returns. That covers the
opcode encoding, the half-step expansion and the position bookkeeping.

**The two code tables can't drift.** The Erika byte codes and `.etp` opcodes are
written out by hand in both Python and C++. `test_erika.py` parses
`erika_image.h` and compares. Nothing enforces this at build time, and a
divergence would only surface as a ruined sheet of paper.

```bash
cd src && ../.venv/bin/python -m pytest tests -q
```

## Calibration

One thing here is **not** confirmed by the firmware's own documentation: the
half-step-forward code. `erika_interface.cpp` documents the keycodes the
typewriter emits, and `0x73` is the only gap in the otherwise contiguous
`0x71..0x79` motion block, which is laid out in forward/backward pairs
(`0x71` space / `0x72` backspace, `0x74` half-step back, `0x75`/`0x76` half
line). `0x73` is the obvious candidate for half-step forward, but it is an
inference.

`python -m erika.pipeline calibrate` types a five-part test pattern that settles
it, along with everything else the planner relies on:

| part | what it shows | failure looks like |
|---|---|---|
| 1 | a ruler of `!` at 2-cell pitch | — (reference for part 2) |
| 2 | the same ruler struck again half a step across, so every bar gains a twin | single bars, identical to part 1 |
| 3 | twelve marks in one column: four whole-line gaps, then seven half-line | uneven gaps, or any sideways shift |
| 4 | `O` overstruck with `-` | a hyphen beside the O rather than through it |
| 5 | two `X` struck at the same column via different routes | two X side by side |
| 6 | every typeable glyph, in charset order | any character that differs from the list the tool prints |

If part 2 looks like part 1, the half-step code is wrong: find the right one and
change `HALF_STEP_FORWARD` in [`erika_codes.py`](erika_codes.py) *and*
`ERIKA_HALF_STEP_FWD` in `erika_ai/src/erika_image.h`. The test suite fails if
you change only one.

Part 3 is a pitch comparison, so everything in it shares a column deliberately —
the lower gaps must measure exactly half the upper ones, which is what lets the
planner mix `NEWLINE` for whole-line gaps with half-line steps for odd ones. A
sideways shift there means a line feed is disturbing the carriage.

Part 6 checks something the rest of the pipeline simply assumes: that the type
wheel fitted to the machine matches the byte→glyph table in
[`erika_codes.py`](erika_codes.py) and `erika_ai/src/erika_char_map.cpp`. It
matters more than it looks. The optimizer picks characters by how much ink they
lay down, so a wheel that disagrees with the table does not merely substitute
the odd glyph — it makes every tonal decision in the picture wrong. Compare the
printed rows against the ones the tool prints; if a character differs, fix that
entry in both tables and rebuild the charset.

Wheel layouts vary most in the punctuation and symbol range — `|`, `£`, `§`,
`µ`, `²`, `³` are the usual suspects. That is why the rulers use `!`, which
every wheel carries.

A *single* wrong glyph, with the rest of the same character correct, is a
different fault: a corrupted byte on the wire rather than a wheel mismatch.
That was traced to `EspSoftwareSerial` bit-banging the link with interrupts
enabled, and the firmware now uses a hardware UART instead — see
[Why a hardware UART](../../../erika_ai/README.md#why-a-hardware-uart). If you
see it again, reflash before suspecting the machine.

## Command reference

```
erika.pipeline charset    build the Sigma charset (--pitch, --font, --from-scan)
erika.pipeline print      photo -> optimize -> .etp     (-t, -r, -n, -l)
erika.pipeline plan       re-plan an existing choices.json without re-optimizing
erika.pipeline verify     diff a plan render against optimize.py's mockup
erika.pipeline calibrate  motion-code test pattern
erika.pipeline area       bracket the corners of the printable area (--rows, --columns)
erika.pipeline sheet      type the charset, for scanning back in

erika.send <file.etp>     upload over USB serial   (--port, --print, --watch)
erika.send --diagnose     test the link step by step
erika.send -c STATUS      talk to the device without uploading
erika.send --list-ports   show the serial ports
erika.etp <file.etp>      disassemble a print job
```

## When the upload fails

```bash
python -m erika.send --port COM6 --diagnose
```

That checks the console, then the base64 upload path, then the stored job, and
says which step broke. `--verbose` adds every line in both directions.

Things that have actually bitten:

- **A data line larger than the serial receive buffer.** This one cost a while.
  The core's stock RX buffer is 256 bytes on both `HWCDC` and `HardwareSerial`,
  and the receiver only drains the port between `poll()` calls — so if a line
  does not fit whole in the buffer, the firmware has to keep pace with the wire
  while it arrives. It cannot: writing the previous chunk to SPIFFS stalls for
  tens of milliseconds whenever a flash page needs erasing, which at 115200
  baud is hundreds of bytes lost. The symptom is nasty, because it is
  intermittent and position-random: short transfers always work, long ones drop
  a line somewhere different every time. Fixed at both ends —
  `Serial.setRxBufferSize()` in `setup()`, and a chunk size small enough
  (128 decoded → 172 characters) that a line fits even without it.
- **A failed transfer poisoning the next run.** The firmware kept answering the
  data lines still in flight, and those replies arrived after the next run
  opened the port — so it read a stale error as the answer to its first
  command. The uploader now resyncs on connect, and the firmware explains a
  stray data line once and then goes quiet instead of erroring per line.
- **The board is still in `setup()`.** `loop()` does not run until WiFi
  association finishes, and the uploader only waits two seconds after opening
  the port. If the device answers nothing at all, try `--settle 10`.
- **DTR/RTS.** pyserial asserts both when it opens a port; on boards with a
  USB-serial bridge those lines are wired to EN and GPIO0, so leaving them
  asserted can hold the ESP32 in reset or drop it into the download ROM, where
  it answers nothing. `Link` parks them low before opening — if you write your
  own tool against this protocol, do the same.

Uploads are newline-framed base64 rather than raw binary, which is what makes
them debuggable: you can watch the whole conversation in a serial terminal, and
log output the firmware interleaves on the same port does no harm. A missing
line is caught immediately by the running `ACK` total, and `erika.send` retries
the transfer (`--retries`, default 2) before giving up.

If you port this protocol to another board, the one invariant to preserve is
that **a whole data line fits in the receive buffer** — `test_erika.py` pins
`CHUNK_SIZE` against `IMG_MAX_LINE`, but the buffer size itself is a runtime
property you have to check yourself.

Useful planning flags for `print` and `plan`:

| flag | effect |
|---|---|
| `-r 48` | characters per row; sets the printed size |
| `-l 4x1` | layer scheme; `1x1`…`4x2` are typeable, quarter-cell ones are not |
| `--no-home` | serpentine sweep instead of returning each pass — faster, less accurate |
| `--settle-ms 200` | pause after each paper feed, if the platen needs to settle |
| `--jitter 0.1` | registration error used for the shaky preview |

## Sizing

At pitch 10 a cell is 2.54 × 4.23 mm, so `-r 48` on a 4:5 photo is about
122 × 170 mm — comfortably inside A4 with margins. The carriage limit is 65
columns; at 48 columns and four layers expect roughly 5,000 strikes and,
at the default 100 ms per character, around 25 minutes of typing.

To see that on paper rather than in millimetres, type a print-area sheet:

```bash
python -m erika.pipeline area                     # the whole reachable area
python -m erika.send results/print_area.etp --print
python -m erika.pipeline area --columns 48 --rows 60   # where an -r 48 print lands
```

It brackets the four corners — an `L` of `_` and `!` at each — and captions the
sheet with its own dimensions. Load the paper exactly as you would for a photo:
the top-left bracket lands wherever the head is when the job starts, so the
sheet shows the area *relative to how you feed the sheet*, which is the part no
amount of arithmetic settles.

The two axes are not the same kind of limit, and the sheet is mostly there to
show it. **Width is the machine**: the carriage reaches 65 columns at pitch 10,
78 at pitch 12, and that is the widest `-r` the planner accepts — a right-hand
bracket that comes out short or smeared means this machine stops earlier, and
that column count is the real ceiling. **Height is only paper**: the platen
keeps feeding for as long as it grips the sheet, so `--rows` (60 by default,
254 mm) is a question about the paper, not the typewriter. Whatever number
still prints both bottom brackets cleanly is how tall a print can be.
