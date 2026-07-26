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

`python -m erika.pipeline calibrate` types a five-part test pattern that checks
it, along with everything else the planner relies on. If part 2's bars land on
top of the reference bars instead of between them, the code is wrong — find the
right one and change `HALF_STEP_FORWARD` in [`erika_codes.py`](erika_codes.py)
and `ERIKA_HALF_STEP_FWD` in `erika_ai/src/erika_image.h`. The test suite will
tell you if you only change one.

## Command reference

```
erika.pipeline charset    build the Sigma charset (--pitch, --font, --from-scan)
erika.pipeline print      photo -> optimize -> .etp     (-t, -r, -n, -l)
erika.pipeline plan       re-plan an existing choices.json without re-optimizing
erika.pipeline verify     diff a plan render against optimize.py's mockup
erika.pipeline calibrate  motion-code test pattern
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

Two things that have actually bitten:

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
log output the firmware interleaves on the same port does no harm.

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
