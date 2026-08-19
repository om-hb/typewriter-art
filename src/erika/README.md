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

# 2b. Worth doing once, and it is the largest quality factor there is:
#     find out which strike forces this machine accepts  (see Strike force)
../.venv/bin/python -m erika.pipeline forces
../.venv/bin/python -m erika.send results/strike_forces.etp --print

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
multiples of 0.5 are rejected, because half a cell and half a line are the
finest motions the *keyboard* has; and an image wider than the carriage (65
columns at pitch 10, 78 at pitch 12) is rejected with the `-r` value that would
fit.

`--fine` lifts the first of those. The carriage moves in 1/120" motor steps and
the platen in 1/240", far finer than any key, so an offset is typeable if it
lands on a whole step — which makes `16x1` realisable at pitch 10 (a quarter of
a cell is three carriage steps) and `daisy_full` realisable at pitch 15 (an
eighth of a cell is one step, a fifth of a line is eight). Not at pitch 12,
where a quarter of a cell is two and a half steps and half a motor step does not
exist; the error says so rather than blaming the scheme. It needs `0xA5` and
`0xA6`, which are unconfirmed here — see **The codes the pipeline does not use**.

## The `.etp` print job

A flat opcode stream. All geometry is resolved on the host; the firmware is a
dumb interpreter. See [`etp.py`](etp.py) for the layout — 24-byte header with a
CRC-32 over the body, then one-byte opcodes with at most one operand.

```
$ python -m erika.etp results/photo.etp -n 8
; ETP1  grid 49x41  strikes 3875  pitch 10  home_each_row True
;  offset  col   row  opcode
       0    0.0   0.0  CR
       2    0.0   1.5  DOWN      3
       4    3.0   1.5  RIGHT     6
       6    4.0   1.5  STRIKE    '3'
```

The two position columns are a running model of where the head is, so a plan
can be checked without a typewriter. A strike also carries the force in effect
(`STRIKE 'b' @f3`) when the job sets one, because a picture typed at several is
unreadable without it.

Adding an opcode is **three** edits on the firmware side, not two: the enum, the
`needsOperand` expression in `fetchNext()`, and a `case`. Miss the middle one and
the interpreter reads one byte too few for the rest of the job — every later
opcode taken from an operand and every operand from an opcode — with each byte
still individually legal and the CRC still passing, because the file is intact.
Two tests in `src/tests/test_erika.py` compare that list and the set of handled
cases against `etp.py`; they are the only thing that would notice.

The header carries a version the firmware rejects on mismatch, but a *new* opcode
does not need a version bump: unknown opcodes already fail the job loudly, so
older firmware meeting a newer job stops rather than typing something else.

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
erika.pipeline print      photo -> optimize -> .etp   (-t, -r, -n, -l, -g, --align)
erika.pipeline plan       re-plan an existing choices.json without re-optimizing
erika.pipeline verify     diff a plan render against optimize.py's mockup
erika.pipeline calibrate  motion-code test pattern
erika.pipeline area       bracket the corners of the printable area (--rows, --columns)
erika.pipeline sheet      type the charset, for scanning back in (--forces)
erika.pipeline forces     sweep the strike-force command to see what it takes
                          (--from/--to/--step to widen it and keep it to a sheet)
erika.pipeline codes      probe the control codes the pipeline does not use yet

erika.send <file.etp>     upload over USB serial   (--port, --print, --watch)
erika.send --diagnose     test the link step by step
erika.send -c STATUS      talk to the device without uploading
erika.send --list-ports   show the serial ports
erika.etp <file.etp>      disassemble a print job
```

## Strike force

The paper this pipeline implements says plainly which choice matters most, in
section 5.5: "using a character set that contains variation in strike force is
the largest factor in obtaining a good tonal range in the midtones and
highlights". Its typewriter got that from a human hand pressing harder or
softer, and section 5.7.2 spends a page on how badly that goes -- closing with
"ultimately, an automated typewriter would provide better consistency in strike
force", which is this.

This machine has a command for it: `A3H`, *Anschlagstärke*, followed by one byte
naming the force. It is carried end to end now -- `SET_STRIKE_FORCE` in
`erika_codes.py`, `OP_SET_FORCE` in the `.etp` stream, `ETP_SET_FORCE` in the
firmware, a `force` per glyph in `glyphs.json`. **What no part of it knows is
which forces this machine accepts.** The manual gives the code and says the next
character is the strength; it does not say what strengths exist, nor whether
"character" means a small integer or the ASCII digit for one.

So the first step is a sheet, not a setting:

```bash
python -m erika.pipeline forces
python -m erika.send results/strike_forces.etp --port /dev/cu.usbmodem1101 --print
```

It types a reference row *before* it sends any force command, then one row per
candidate value across both readings.

If neither reading shows anything, widen it -- but coarsely. `--from 0 --to 0xFF`
is 245 rows and several sheets of paper; `--step 16` makes the same span sixteen
rows and one. The stride is over the values rather than over the rows kept, so the
rows stay evenly spaced with a gap wherever it lands on a motion code, and every
row is labelled with the value it was typed at. A coarse pass brackets a
neighbourhood rather than naming a value: sweep the neighbourhood again at
`--step 1` to find where it starts and stops. Rows that come out lighter or darker than
the reference are forces the machine took; a row with a stray character in front
of the glyphs is a value it typed instead, which rules that value out. Every
candidate is a byte the type wheel could have typed on purpose -- on a machine
that ignores the command the force byte arrives as an ordinary character, and
above the wheel's own codes there is no ordinary character: only a *motion*,
which would shift the rest of the line and ruin the sheet exactly where it has
to be read, or a command that swallows the byte after it.

With the answer, hardest first:

```bash
python -m erika.pipeline sheet --forces 0,3,6        # type the set at each
python -m erika.pipeline charset --forces 0,3,6 \
    --name sigma-forces --from-scan /path/to/scan.png
```

The `--forces` lists must match and be in the same order: the scan is sliced to a
grid, and the grid is the only thing that says which tile is which.

One piece of the paper's advice does **not** transfer. Its figure 20 found that
medium plus light beat dark plus medium, because its Smith-Corona could already
reach black and needed help in the highlights. This machine cannot reach black,
so the hardest strike has to stay: dropping it and keeping two lighter forces
measured clearly worse. Add lighter forces *below* full, do not shift the range
down.

A line is typed force by force, hardest first, which is section 5.7.2's own
suggestion. Here the reason is mechanical rather than human: a force change costs
a full character delay, and interleaved they can outnumber the strikes -- a
9,000-strike picture at three forces needs about 200 changes grouped and
thousands ungrouped. The cost is one carriage sweep per force per line, so
whatever error the carriage accumulates lands differently in each group;
`build_plan(group_by_force=False)` trades back the other way.

Every job also restores the hardest force before it ends. Force is state that
outlives the print: left soft, the next thing typed on this machine -- by hand or
by the firmware's chatbot -- comes out faint for no visible reason.

## How much ink a mark carries

The other half of section 5.5, and the part that was quietly wrong for longer.
Glyphs are rendered from an outline font, so their ink was *pure black with hard
edges*. No ribbon produces that, and it is not a cosmetic approximation: the
optimizer scores candidates per **pixel**, so a stroke at grey 0 laid through a
mid-grey cell costs more squared error than leaving the cell empty. The optimizer
therefore declines to mark it. Measured at 40 columns on `mwdog_crop.png`, half
of every cell in a default run came out blank while the picture as a whole was 39
grey levels too light -- pale and blotchy at the same time, which is the
signature.

`--ink` (the grey the densest ink reaches, default 0.10) and `--spread` (point
spread in cell pixels, default 0.6) make the model *possible*. They do not make
it accurate; `--from-scan` does. The paper is explicit that the non-black ink is
a feature and not a defect: "even the darkest typewriter characters are less than
fully black, which yields a greater tonal range when characters are allowed to
overlap".

Which way to err is not symmetric, so it is worth stating. A charset modelled
lighter than the machine makes the optimizer ask for more ink than the paper
needs, and the print comes out dark -- visible, and correctable by asking for
less. Modelled darker, it refuses to mark the midtones, and what is lost is
detail that nothing downstream can put back. Hence defaults that lean light.

## Letting it halftone: `--match-blur`

Per-pixel error forbids dithering, and for a mechanical reason worth spelling
out: a sparse mark that is right *on average* is wrong at every pixel it covers
and every pixel it leaves bare. So the optimizer will not halftone, and the
highlights go to bare paper.

The paper proposes the fix in section 6 -- "the algorithm could be made to
discourage reliance on precise character placement for tone matching, including
by adding blur or reducing resolution during selection". `softmatch.py` is that:

```bash
python -m erika.pipeline print -t photo.jpg -r 40 -l 4x2 --match-blur 0.8
```

Loss becomes `(1-w) x per-pixel error + w x error over block means`, so `w` of 0
changes nothing and 1 scores local tone alone. The charset is **not** softened
and the mockup stays a faithful composite of the glyphs, which is what keeps the
verification step meaningful -- the plan is still diffed against a picture of
exactly what the machine will type.

Off by default, because the trade is real: at 40 columns with `4x2`, per-pixel
RMSE moves 73 to 82 while the same error measured over half a cell -- roughly
what an eye does with a typed sheet at arm's length -- moves 34 to 15. Which of
those you want depends on whether the result is to be looked at or zoomed into.


## Lining the picture up: `--align`

Characters can only land on a fixed grid, so where the picture sits against that
grid decides how well an edge in the photograph can be matched by an edge in the
type. Section 3.4 of the paper searches for the best placement: 64 crops, shifting
by `a`,`b` of a character cell and scaling by `(c + n)/n`, each scored with a quick
greedy approximation and ranked by `SSIM x 4 + PSNR`.

```bash
python -m erika.pipeline print -t photo.jpg -r 20 --align
```

`align.py` is that search as a pre-pass. It picks a crop, writes
`results/target-aligned.png`, and hands the path on -- nothing downstream knows it
happened. It runs *after* `--grey`, because STRESS is spatial and stochastic and
converting each of 64 candidates would be comparing 64 different pictures.

**It uses two greedy cycles per candidate, not the paper's one.** That is not a
refinement, it is what makes the search mean anything. A single cycle's score
carries enough run-to-run noise to swamp the differences it is meant to measure:
at 40 columns the noise range across seeds is 0.20 against a spread of 0.32 across
all 64 candidates, and the search disagrees with itself -- three runs of it pick
two different crops, on scores a thousandth apart. A second cycle re-evaluates
every cell against a settled background and drops the noise range to 0.06, which
is where three runs agree. The noise is not removable from Python: candidate order
comes from numba's per-thread RNG, and greedy selection is order-dependent
whenever two glyphs tie.

Off by default, because the honest numbers are modest and the cost is not.
Measured on `mwdog_crop.png` with the charset as it ships:

| | SSIM | RMSE | search |
|---|---|---|---|
| 20 columns, `4x1` | 0.388 -> 0.402 | 65.2 -> 64.0 | 12 s |
| 20 columns, `4x2` | 0.407 -> 0.419 | 52.4 -> 51.4 | 24 s |
| 40 columns, `4x1` | 0.372 -> 0.378 | 66.5 -> 65.9 | 42 s |
| 40 columns, `4x2` | 0.384 -> 0.389 | 52.9 -> 52.4 | 85 s |

Nothing regresses, and the gain is larger at 20 columns than at 40 -- which is the
one part of section 5.3.1's "one of the dominant factors" claim that carries over
to this machine. The paper's dramatic examples are a checkerboard and a 20-column
portrait, where alignment is the only thing left to get wrong. Here it is not:
alignment buys *shape* matching, and what binds on this machine is ink.

The search costs up to four times the optimizer run it precedes, and scales with
the charset -- a set carrying three strike forces costs about three times the
figures above. `--align-steps` trades accuracy for time, but not below 3: `4x1`
already places layers at 0 and 0.5 of a cell, so a half-cell shift is close to a
relabelling rather than a better fit. The effective period in each axis is half a
cell, the paper's quarters sample it twice over, and `--align-steps 2` probes
little but the scale and the border.


## Black and white first

The optimizer reads a greyscale image, and how a colour photograph became one is
not a detail. `cv2.IMREAD_GRAYSCALE` -- what it does when left alone -- is Rec.
601 luma: one weighted sum, applied identically everywhere. Two areas that
differ in hue but not in brightness therefore become **one flat tone**, and with
about fifty printable greys in a cell, a flat tone is a region with no characters
in it at all. Nothing further down can undo that; by the time the optimizer sees
the picture there is nothing left to tell the two apart.

The paper raises this in section 5.3 -- "the black and white conversion method
has a substantial effect on the result (Figure 12)" -- and declares it out of
scope. `stress.py` is where it is not.

```bash
python -m erika.pipeline print -t photo.jpg -r 48 --grey c2g
```

| `--grey` | what it does |
|---|---|
| `luma` | Rec. 601, the default, and what happens with no flag at all |
| `average` | the flat mean of the channels; Figure 12's left panel |
| `c2g` | STRESS colour to grey: every pixel measured against its own surroundings |
| `stress` | STRESS local enhancement -- a retinex-like white balance -- then luma |

Both spatial conversions come from one framework (Kolås, Farup and Rizzi, 2011;
reference [12] of the paper) and are ports of GEGL's `gegl:c2g` and
`gegl:stress`. Around each pixel they throw a random spray of sample points,
take the local minimum and maximum, and report where the pixel sits between
them. `c2g` turns that into the grey directly, which is what keeps a colour
difference a formula would merge; `stress` uses it to rescale each channel,
which corrects a cast and opens up an unevenly lit picture.

Three flags tune the spray. `--stress-radius` is a *fraction of the longest
side*, not a pixel count, so a photograph converts the same way whatever
resolution it arrives at. `--stress-iterations` trades time against sampling
noise: the default of 20 is where that noise falls below the machine's own tonal
steps once the optimizer has averaged a cell -- per pixel it is still nine grey
levels, per cell it is under one. `--stress-seed` fixes the spray, so a
conversion is repeatable in a way an optimizer run is not.

The converted target is written to `results/target-grey.png` and the optimizer is
pointed at it, since `kword` has no way to be asked for another conversion.

One thing to watch for on paper: `c2g` raises local contrast *everywhere*,
including where the photograph had none. A plain wall has its own faint texture
pulled up with everything else, and that prints as characters where there might
have been paper.

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
| `-l 4x1` | layer scheme; `1x1`…`4x2` are typeable by keystroke, finer ones need `--fine` |
| `--fine` | place strikes with the machine's own motor steps, for offsets between the half-cell grid points (`16x1`, `daisy_full`) |
| `--no-advance` | stack a cell's glyphs with Doppeldruck instead of a backspace between each pair |
| `--no-home` | serpentine sweep instead of returning each pass — faster, less accurate |
| `--settle-ms 200` | pause after each paper feed, if the platen needs to settle |
| `--jitter 0.1` | registration error used for the shaky preview |

## Sizing

At pitch 10 a cell is 2.54 × 4.23 mm, so `-r 48` on a 4:5 photo is about
122 × 170 mm — comfortably inside A4 with margins. The carriage limit is 65
columns; at 48 columns and four layers expect roughly 3,900 strikes and,
at the default 10 head operations per second, around 17 minutes of typing.

### A whole A4 sheet

Three numbers give the size of the job, but only two of them are geometry — the
third is the picture.

A4 is 210 × 297 mm, or 82 × 70 cells at pitch 10. The carriage never reaches 82,
so the width is the machine's limit rather than the sheet's, and the height wants
a margin the platen can keep gripping:

```
columns   min(65, 210 / 2.54)   =  65      165 mm, 22 mm spare each side
rows      254 / 4.23            =  60      254 mm, 21 mm spare top and bottom
layers    -l 4x1                =   4
slots     65 x 60 x 4           =  15,600  one cell of one layer
```

So 15,600 — but that is the number of *opportunities* to strike, not strikes.
The optimizer leaves a cell blank wherever the picture is white and the planner
drops those (`if index:` in `load_choices`), so the characters actually typed
are `slots × ink fraction`. On the sample photo at `-l 4x1` that fraction is
48%, which puts a full sheet at roughly **7,500 characters**. A dark, busy
photograph climbs toward 15,600; a portrait on a white background can halve it
again. Nothing has to be estimated once the job exists, because `print` reports
both halves of the product — a full sheet would read:

```
  grid         65 x 60 cells, 4 layers (15600 slots, 7522 inked)
```

Typing time needs both numbers, not just the strikes: the carriage is charged
for every half-step it crosses, blank cells included, so head operations come to
roughly slots plus strikes — about 20,000, or ~33 min at the default 10/s, and
appreciably less in practice (see below). Halving the layers to `-l 2Hx1` halves
all of it, at the cost of tonal range.

Feeding the sheet right to its edges buys 70 rows instead of 60 — 18,200 slots,
around 8,800 characters — but the last lines are typed on paper the platen has
nearly let go of, which is exactly what the `area` sheet below is for.

That 17-minute estimate is an upper bound. `--ops-per-second` assumes every head
operation costs the full character delay, but the firmware paces each byte by
what the machine actually has to do — a repeated glyph needs no wheel rotation
and a carriage step prints nothing, so both are quicker. On a real job that is
about a third off the figure above. See
[Pacing](../../../erika_ai/README.md#pacing).

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

### What a 124 m ribbon holds

The ribbon advances a fixed step on every strike, so its yield is a division —
and the step is a property of the ribbon, not of the picture:

```
124 m / 2.54 mm   ≈  48,800 characters    film, one pitch step   (pitch 10)
124 m / 2.12 mm   ≈  58,600 characters    film, one pitch step   (pitch 12)
```

Single-strike film is the pessimistic case and the one to plan against: each
strike has to land on ribbon nobody has used, so the feed is a whole character
width. Call it **~48,800 characters, or about 6 full A4 sheets** — roughly 12 of
the `-r 48` photo above.

Two things make that go faster than it reads:

- **Layers multiply it.** Ribbon is spent per strike, not per cell. The 7,500
  characters of a full `-l 4x1` sheet are 7,500 ribbon steps, four of them
  stacked on most cells.
- **A dead key still costs ribbon.** `STRIKE_NA` suppresses the *carriage*, not
  the ribbon feed, so an overstruck accent consumes as much as a letter.

Both figures are arithmetic on an assumed advance step, which is exactly the
kind of assumption this pipeline otherwise refuses to make — measure it instead.
Every job prints its own strike count, so keep a running total across jobs and
note where a fresh ribbon dies; 124 m divided by that total is this machine's
real figure. Until then plan on six sheets, and know the recovery path before
you need it: note the pass number from `IMG STATUS`, change the ribbon, and
resume with `IMG PRINT pass N`, which replays the paper feeds with the strikes
suppressed (see [Typewriter art](../../../erika_ai/README.md#typewriter-art)).


## The codes the pipeline does not use

Eleven control codes drive everything above. The interface answers to about
sixty, published for the Erika S3004 -- `erika_ai/ressources/steuercodes.md`,
read against this pipeline in `docs/control-codes.md` at the workspace root.
The Sigma shares that interface, which is a claim about a family of machines
and not a measurement of this one.

```bash
python -m erika.pipeline codes
python -m erika.send results/control_codes.etp --print
```

Nine sections, each with a reference typed beside it in codes already known to
work, because "did it move a twelfth of an inch" is not answerable by looking
at one mark. In order: the bell (which says whether a command with an operand
is understood at all, without marking the paper), carriage steps and platen
steps at `0xA5`/`0xA6`, the forbidden feed counts, Doppeldruck at `0xA9`,
backward print at `0x8E`, the correction ribbon at `0x8C`, and 15 characters
per inch at `0x89`. The command prints what to look for on each.

The two that would change the most: `0xA5` and `0xA6` move in absolute
1/120" and 1/240" steps rather than in fractions of whatever the slide
switches are set to, which is both the finer grid the quarter-cell layer
schemes need and the end of `PITCH_WIDTH_MM` being an assumption about a
switch nobody can read. `0xA9` overstrikes without spending the escapement's
repeatability on a backspace.

These reach the machine through `OP_RAW`, the one opcode the firmware has no
opinion about. That is deliberate and it is meant to be temporary: a raw byte
is outside the position model, so it is outside the offline verification that
the rest of this package is built around. A code that survives the sheet
should stop being raw and become an opcode that `planner`, `etp.disassemble`
and `emulate` all understand.

`0x96`, the completion report, is not on the sheet. It changes when RTS is
released rather than what is typed, so the firmware's pacing answers it and
ink cannot.
