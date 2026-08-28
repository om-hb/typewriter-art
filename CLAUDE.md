# typewriter-art (Sigma fork)

Fork of [juleskuehn/typewriter-art](https://github.com/juleskuehn/typewriter-art)
(Graphics Interface 2021) — an optimizer that renders a photograph as
overlapping layers of typed characters.

This fork adds `src/erika/`: a back-end that turns the optimizer's output into
a print job for a **Sigma SM 8200i** typewriter driven by the
[`erika_ai`](https://github.com/om-hb/erika_ai) firmware.

## Commands

Python 3.10 (numba misbehaves on 3.11+). Everything below runs from `src/`.

```bash
# setup, from the repository root
python3.10 -m venv .venv
.venv/bin/pip install -r src/requirements-erika.txt      # macOS/Linux
.venv\Scripts\pip install -r src\requirements-erika.txt  # Windows
```

Invoke the interpreter by path — there is no activated venv. On Windows this
project is also driven through `uv run python`. Substitute whichever applies;
`PY` below stands for one of:

| | |
|---|---|
| macOS/Linux | `../.venv/bin/python` |
| Windows | `..\.venv\Scripts\python` |
| uv | `uv run python` |

```bash
cd src
PY -m pytest tests -q                                   # 238 tests
PY -m erika.pipeline charset --pitch 10                 # build the Sigma charset
PY -m erika.pipeline print -t images/mwdog_crop.png -r 48
PY -m erika.pipeline print -t photo.jpg -r 48 --grey c2g   # STRESS, not luma
PY -m erika.pipeline print -t photo.jpg -r 40 -l 4x2 --match-blur 0.8
PY -m erika.pipeline print -t photo.jpg -r 20 --align         # crop search first
PY -m erika.pipeline forces                             # what strike forces exist
PY -m erika.pipeline calibrate                          # machine test pattern
PY -m erika.pipeline area                               # corners of the print area
PY -m erika.pipeline verify                             # plan vs. optimizer mockup
PY -m erika.pipeline melody --list                      # tunes for the beeper
PY -m erika.pipeline melody --morse "READY"             # ...or spell something
PY -m erika.etp results/photo.etp -n 30                 # disassemble a job
PY -m erika.send results/photo.etp --port COM6 --print --watch
PY -m erika.send --port COM6 --diagnose                 # when uploads fail
```

Subcommands: `charset print plan verify calibrate area sheet forces codes
melody`. The last one makes no marks: see `src/erika/melody.py`.

No lint, typecheck or CI configured. `.vscode/settings.json` (upstream)
declares **black** with format-on-save, but black is not in
`requirements-erika.txt` and has not been run over `src/erika/`.

## Project structure

```
src/
  optimize.py utils.py   UPSTREAM optimizer — see Conventions before editing
  layers.json            layer offset schemes
  charsets/              sigma-10 / sigma-12 generated; the rest upstream scans
  images/                sample inputs
  results/               generated output — gitignored, disposable
  tests/                 pytest suite
  erika/                 the Sigma back-end (entirely ours)
    erika_codes.py         keys + motion codes; mirrors erika_char_map.cpp
    make_charset.py        builds a charset that maps glyph index -> key
    stress.py              colour -> black and white, before the optimizer
    planner.py             layers -> ordered strikes + carriage/paper moves
    etp.py                 the .etp print-job container
    preview.py             renders a plan back to an image
    emulate.py             virtual typewriter; mirrors erika_image.cpp
    align.py               optional pre-pass: crop search against the cell grid
    softmatch.py           optional loss term: local tone, not only pixels
    melody.py              the beeper: one pitch, so tunes are rhythm only
    pipeline.py send.py    CLIs
    README.md              the detailed guide — read this first
```

## Architecture

**All geometry is resolved here, not on the device.** The planner flattens the
optimizer's overlapping layers into one ordered list of absolute strikes and
emits every carriage and paper move between them. The firmware is a dumb
opcode interpreter that never knows where the print head is. That split is what
lets the whole plan be verified without a typewriter.

Coordinates are half-cells throughout: `x` = half-steps right of the left
margin, `y` = half-lines below the top. A layer offset of `0.5` is exactly one
half-step or one half-line — which is why only offsets of 0 and 0.5 are
typeable.

**The pipeline checks itself at every stage**, because a subtle error costs
half an hour of typing to discover:

- every `print` run re-renders the finished plan and diffs it against
  `optimize.py`'s own mockup — it should report *"plan reproduces optimize.py's
  mockup exactly"*, and anything less means the flattening, index mapping or
  offset arithmetic is wrong
- `emulate.py` expands opcodes to raw bytes and runs them through a simulated
  carriage and platen; tests assert every strike lands where planned
- `make_charset.py` re-runs the upstream loader and asserts the glyph count,
  because `chop_charset()` silently drops tiles it judges blank — which would
  shift every index after the dropped one

## Shared tables with erika_ai

The Erika motion codes, the `.etp` opcodes and the upload chunk size each exist
twice — here in Python, and in C++ in `erika_ai/src/`. Thirteen tests parse
those headers and compare. **They only run with `erika_ai` checked out**: beside
this repository by default, or wherever `ERIKA_FIRMWARE_SRC` points. If it is
missing the suite prints a loud `drift guards did not run` banner — do not
ignore it, those thirteen tests are the only protection the tables have.

Some of them compare the *shape* of the opcode stream rather than its
constants, and they exist because adding `OP_SET_FORCE` showed how an opcode
goes wrong: the firmware decides how many bytes to read from its own
hand-written list of opcodes that carry an operand, and an opcode missing from
that list leaves the interpreter one byte out of step for the rest of the job.
Every later opcode is then read from an operand and every operand from an
opcode. The device cannot notice — each byte is individually legal — and the CRC
passes, because the file is intact. Nothing but the comparison catches it.

## Reading a scanned sheet

Two things happen to a scan before its grid is sliced, and both exist because the
grid is sliced *axis-aligned* whatever the sheet does.

**It is squared up first** (`deskew.py`). The registration marks say where the
grid starts, which is a different question from which way it points, and nothing
was answering the second one. Measured on a synthetic sheet with type's ink
extents: a degree of skew moves the worst tile by 26 grey levels out of 255, one
and a half by 42, and two degrees fails the build. The dangerous band is the
shallow one, where the charset comes out looking reasonable and every tone in it
is wrong — and the error grows toward the corners, which a spot check of the
middle does not see. The angle is found by maximising the sharpness of the ink's
row and column projections, not from the marks: finding a mark on a crooked sheet
means first solving what the marks are for, since `_find_sheet_marks` identifies
them by projecting the whole image onto one axis and rotation is exactly what
smears that. Below a twentieth of a degree it does nothing, deliberately — a
resample blurs every edge by half a pixel whatever the angle, which on a sheet
whose rows are two dozen pixels tall costs more than the skew did.

**Its blank-cell threshold is measured rather than assumed.** `WHITE_THRESHOLD`
is 0.999 — a tile is blank if its mean brightness is within a tenth of a percent
of white. That is right for a sheet drawn from a font and hopeless for a scan,
where a blank cell is paper: photographic noise of one and a half grey levels is
already several times that, and so is the trace of the row above that any
resampling smears in. Every trailing cell of a sheet that does not end on a full
row then reads as a glyph and `_verify_mapping` refuses the build — and a
`--dead-keys` sheet has seventeen trailing cells. So `scan_white_threshold` puts
the line in the gap between the darkest glyph cell and the palest cell that must
be blank, which the sheet's known typing order identifies for free. If those two
groups do not separate, the grid is not on the glyphs — a wrong `--sheet-cols`,
or a scan cropped into the block — and that is said out loud, because it is the
one failure here that every later stage would accept in silence.

## Strike force, and what the charset models

The paper's section 5.5 is unambiguous about what matters most: "using a
character set that contains variation in strike force is the largest factor in
obtaining a good tonal range in the midtones and highlights". This machine has
the command for it — `A3H`, *Anschlagstärke*, followed by one byte naming the
force — and until recently nothing here knew about it.

It is now carried end to end: `erika_codes.SET_STRIKE_FORCE`, the `.etp` opcode
`OP_SET_FORCE`, a `force` per glyph in `glyphs.json`, `Charset.forces` and
`force_order`, and `ETP_SET_FORCE` in the firmware.

**The probe sheet has now been typed, and the command works.** On the Sigma SM
8200i with a Courier 10 wheel:

| value | what prints |
|---|---|
| `0` | solid — full strike, and nothing prints darker |
| `1`–`39` | no ink at all |
| `40` | first ink: isolated dots, not a character |
| `43` | first legible character — the practical floor |
| `55` | fully formed characters |
| `95`–`103` | saturated; indistinguishable from each other |

Which settles the question the sheet existed for: the operand is a **raw count**,
not the ASCII digit for one — an ASCII scale would mark only at `0x30`–`0x39` and
type strays everywhere else, and the sheet is one continuous ramp. Larger is
harder, with `0` the exception at the top. `erika_codes` carries the full note.

Two things to carry forward. The usable ladder is **43 to 95** and it is
compressed at the bottom — 43 to 55 spans barely-there to fully formed — so
charset forces want picking from the bottom of it, by how the ink looks rather
than by even arithmetic. And the scale saturates at `0x5F`, *below* the `0x67`
that `MAX_FORCE` imposes for an unrelated reason, so that guard costs this
machine no tonal range. All of it is a measurement of one wheel and one ribbon;
re-run the sheet for another.

```bash
PY -m erika.pipeline forces                 # sweeps both readings, on paper
PY -m erika.pipeline forces --from 35 --to 103 --step 1    # the pass worth typing on a known-good command
PY -m erika.pipeline forces --from 35 --to 103 --step 1 --from-scan scan.png   # and read it back
PY -m erika.pipeline sheet --forces 0,60,50,43   # then type the set at each
PY -m erika.pipeline charset --forces 0,60,50,43 --from-scan scan.png
```

`forces --from-scan` is the second half of the sheet, and the reason it exists is
that reading the sheet by eye answers a different question than the charset asks.
By eye you get which values marked and which did not; what a multi-force charset
wants is three or four values spaced evenly *in tone*, and the force value is a
lever position rather than a quantity of ink — between 43 and 55 is as much of
the range as everything above 55. Every row of the sheet is the same glyph struck
the same number of times, so the ink per row is the transfer curve, and the scan
turns choosing forces into arithmetic. It prints a `--forces` list and the
matching `--force-density` for the modelled path.

Two things about that reader. It needs **the same sweep the sheet was typed
with** — the `charset --sheet-cols` failure again, and for the same reason: the
grid is the only thing that says which row is which, and a row misidentified
reads as a plausible curve rather than as an error. And the grid is *computed
from the title line, not detected*: a force below the ink threshold prints
nothing, and its label prints nothing either because the label is typed at the
row's own force, so a reader that looked for rows would skip the blank ones and
renumber every row after. `force_scan.probe_lines` is the one list of rows, and
`cmd_forces` walks it to type the sheet so the two cannot drift.

`--step` is what makes a wide sweep answerable: every value from `0x00` to `0xFF`
is 245 rows and several sheets of paper, and at `--step 16` it is sixteen rows and
one. It strides the *value space* and drops the motion codes afterwards, so the
rows stay evenly spaced with a gap where the stride lands on one -- filtering first
would hand back N usable values at uneven intervals, which is unreadable as a
sweep. A coarse pass brackets a neighbourhood rather than naming a value, and the
printout says so when the step is not 1.

Read `forces`' own printout before spending a sheet. Two things about it are
load-bearing: the first row is typed *before* any force command, so there is a
reference to compare against, and every candidate value is outside `0x71..0x82`,
because a machine that ignores the command types the force byte instead — and a
byte in that range is a *motion*, which would shift the rest of the line and
make the sheet unreadable exactly where it has to be read. `is_usable_force`
enforces that everywhere a force can enter.

Three consequences worth knowing before touching any of this:

- **A charset without forces is untouched by all of it.** `forces` is empty,
  nothing asserts a force, and the byte stream is what it always was. That is
  what made this safe to switch on.
- **A line is typed force by force, hardest first**, which is the paper's own
  advice in 5.7.2. Not for the typist's sake but because a force change costs a
  full character delay, and interleaved they can outnumber the strikes — a
  9,000-strike picture at three forces needs about 200 changes grouped, and
  thousands not. What it costs is one carriage sweep per force per line, so any
  error the carriage accumulates lands differently in each group;
  `build_plan(group_by_force=False)` buys that back.
- **A job restores the hardest force before it ends.** Force is state that
  outlives the job: left soft, the next thing typed on this machine — by hand or
  by the firmware's chatbot — comes out faint for no visible reason.

The other half of section 5.5 is how much ink a mark carries, and the charset
used to get it plainly wrong: glyphs came from an outline font, so ink was *pure
black with hard edges*, which no ribbon produces. That is not a cosmetic
approximation. The optimizer scores candidates per **pixel**, so a stroke at
grey 0 laid through a mid-grey cell costs more squared error than leaving the
cell empty — and the optimizer declines to mark it. Measured at 40 columns on
`images/mwdog_crop.png`, half of every cell in a default run came out blank
while the picture as a whole was 39 grey levels too light.

`--ink` (the grey the densest ink reaches, default 0.10) and `--spread` (point
spread in cell pixels, default 0.6) exist to stop the model being *impossible*.
They do not make it accurate — a scan does. Which way to err, meanwhile, is not
symmetric: modelled lighter than the machine, the optimizer asks for more ink
than the paper needs and the print comes out dark; modelled darker, it declines
to mark midtones and the print comes out blotchy and pale, losing detail nothing
downstream can recover. Hence defaults that lean light.

`_sheet_from_scan` used to undo this in the one place that measures real ink: it
mapped the 1st percentile to 0, taking the darkest ink on the sheet and making
it black. It now normalises the *paper* and leaves the ink where it lies.

**`sheet` and `charset --from-scan` are one number and both have to be told it.**
`--sheet-cols` is the grid the glyphs are typed on and the grid the scan is sliced
back up on, and for one commit only the first of the two took the flag: `charset`
did not have it and `_build_charset` did not forward one, so a sheet typed at
anything but 20 was sliced on a grid of 20 regardless. That failure has no
symptom. Every tile comes back a blend of two neighbouring glyphs, the count is
still right, so `_verify_mapping` passes, the charset loads, the plan verifies and
the print comes out — with every tonal decision in it made against ink no key
produces. Two tests in `test_erika.py` guard it: one that both subcommands take
the flag and default alike, and one that `_build_charset` actually forwards it,
because a flag accepted and dropped looks identical from the command line.

## Typing right to left: `0x8E`

A serpentine plan (`--no-home`) sweeps alternate passes backwards, and the
machine will do that itself: `0x8E` makes a strike step one whole cell *left*
and then mark, so a reverse pass costs one wire byte a cell instead of a glyph
and two backspaces. On by default since section 7 of the control-code sheet came
back reading EDCBA with the A one cell left of where the head started;
`--backspace-sweep` reverts it, and it does nothing at all to a plan that
returns the carriage every row, which has no reverse passes to type.

`planner._backward_runs` finds maximal runs of strikes exactly one cell apart
and descending, and wraps the ones at least three long in `OP_BACKWARD_ON` /
`OP_BACKWARD_OFF`. Three, because the switch costs a byte at each end and two
cells is a tie when the carriage was already standing on the first of them.

**Whether it is worth anything at all depends on the layer scheme, and that is
the thing to know before measuring it.** `0x8E` moves a whole cell, so it helps
only where consecutive strikes in a pass are a whole cell apart. At 24 columns
on the sample photograph, serpentine, in head operations:

| scheme | backspacing | backwards | |
|---|---|---|---|
| `1x1` | 1526 | 650 | −57% |
| `2Vx1` | 3088 | 1290 | −58% |
| `1x2` | 2502 | 2429 | −3% |
| `4x1` | 5885 | 5815 | −1% |

`4x1` places two layers half a cell apart *across*, so its passes step by halves
and never by the cell the mode moves; `1x2` puts both layers in one cell, so its
passes are stacks, and a stack is `0xA9`.

Three things about the restrictions, because they look like timidity and are
not. The sheet typed five plain letters and asked nothing else, so **nothing but
a plain advancing strike goes between the two opcodes**: not a motion (whether
the motion keys invert with the printing direction is unasked), not a `0xA9`
(whether "print where the head stands" still means that is unasked), not a dead
key, and not a force change. Every one of those would put each later mark on the
line in the wrong cell, and none would read as a mode fault on the sheet — they
would read as the carriage slipping. `emulate.Typewriter` refuses all four
rather than modelling a guess the suite would then certify.

It is also *state on the machine*, like strike force: a job that stops between
the two codes leaves the typewriter typing right to left for whoever touches it
next. The firmware's `flushBackwardPrint()` sends `0x8D` on finish, abort and
failure, and a drift test checks all three.

The open question, and the reason it is worth another sheet: if `0x73` still
moves *right* under `0x8E`, a half-cell scheme becomes two bytes a cell instead
of three, and `4x1` — the row that gains nothing above — joins in.

## Halftoning: `--match-blur`

Per-pixel error forbids dithering — a sparse mark that is right on average is
wrong at every pixel — so the optimizer will not halftone, and on this machine
that costs the highlights. `erika/softmatch.py` is the paper's own remedy from
section 6, "adding blur or reducing resolution during selection": loss becomes
`(1-w) * per-pixel AMSE + w * AMSE over block means`.

It installs itself over `optimize.layer_optimization_pass` in `optimize`'s
namespace — the same seam `erika-studio`'s progress and variety passes use, and
for the same reason: `optimize` binds the name at import, so patching `utils`
does nothing. **The charset is not softened and the mockup stays faithful**,
which is what keeps `print`'s own check meaningful.

Off by default because the trade is real, and measurable: at 40 columns with
`4x2`, per-pixel RMSE moves 73 → 82 while the same error measured over half a
cell — roughly what an eye does with a typed sheet at arm's length — moves
34 → 15.

## Lining the picture up: `--align`

Characters land on a fixed grid, so where the picture sits against that grid
decides how well an edge in the photograph can be matched by an edge in the type.
The paper's section 3.4 searches 64 crops — shifts of `a`,`b` ∈ {0,¼,½,¾} of a
cell and scales of `(c + n)/n` — scoring each with a quick greedy approximation
and keeping the best by `SSIM × 4 + PSNR`. `erika/align.py` is that, as a
pre-pass: it picks a crop, writes `results/target-aligned.png`, and hands the path
on. Nothing downstream knows it happened.

It runs *after* `--grey`, for the reason `erika-studio` converts once: STRESS is
spatial and stochastic, so converting each of 64 candidates would be comparing 64
different pictures.

**Two greedy cycles per candidate, not the paper's one**, and this is the
departure that makes the search work rather than a refinement. One cycle's score
carries enough run-to-run noise to swamp the difference between crops — at 40
columns the noise range across seeds is 0.20 against a 0.32 spread across all 64
candidates, and the search *disagrees with itself*: three runs pick two different
crops on scores 0.001 apart. A second cycle drops the noise range to 0.06, where
three runs agree. The noise cannot be removed from Python: candidate order comes
from numba's per-thread RNG and greedy selection is order-dependent on ties.

Off by default, because it is honest about its size. Measured on the sample
photograph with the charset as it ships, nothing regresses and the gain is larger
at 20 columns than at 40 — the one part of §5.3.1's "dominant factor" claim that
carries over:

| | SSIM | RMSE | search |
|---|---|---|---|
| 20 columns, `4x1` | 0.388 → 0.402 | 65.2 → 64.0 | 12 s |
| 20 columns, `4x2` | 0.407 → 0.419 | 52.4 → 51.4 | 24 s |
| 40 columns, `4x1` | 0.372 → 0.378 | 66.5 → 65.9 | 42 s |
| 40 columns, `4x2` | 0.384 → 0.389 | 52.9 → 52.4 | 85 s |

That last row is nearly four times the optimizer run it precedes, and the cost
scales with the charset — three strike forces is three times these numbers. It
buys *shape* matching, and on this machine shape matching has never been the
binding constraint; ink is.

`_greedy_cycles` reimplements the middle of `kword`, which is the one duplication
in `src/erika/` not guarded by comparing two copies. It has to: calling `kword`
64 times would write `choices.json`, four layer PNGs and a matplotlib figure each
time, and clobber the results the real run is about to write. What makes it
tolerable is that it only ranks candidates *against each other* — a divergence
changes the ranking, not the correctness of anything printed — and the hazard that
would matter, wrong offsets or wrong indices, is caught by
`test_align_composites_the_way_the_optimizer_does`, which recomposites the
returned choices independently and requires the same mockup.

## Pre-processing the target

`optimize.py` opens its target with `cv2.IMREAD_GRAYSCALE` and offers no way to
ask for anything else, so the paper's section 5.3 — "the black and white
conversion method has a substantial effect on the result" — is acted on *before*
it runs. `erika/stress.py` converts the photograph, writes
`results/target-grey.png`, and `--grey` on `erika.pipeline print` is what selects
the method. Nothing upstream was touched.

Why it matters more here than in an image editor: luma is a global map, so two
areas differing in hue but not in brightness become one grey — and on a machine
with some fifty printable levels per cell, one grey is a region with no
characters in it. `c2g` and `stress` are ports of GEGL's ops of those names, both
built on the STRESS framework (Kolås, Farup and Rizzi, 2011, reference [12] of
the paper). `src/erika/README.md` has the argument and the flags; the module
docstring has the algorithm and the three places it deliberately differs from
GEGL.

Two properties of the port are load-bearing and tested:

- **The radius is a fraction of the longest side**, not a pixel count. A picture
  converts the same way at any resolution, which is what lets `erika-studio`
  convert a scaled copy and get the answer the original would have given.
- **The spray is seeded per pixel**, so a conversion is bit-reproducible and
  independent of how numba schedules the rows — unlike an optimizer run, whose
  inner loop draws from numba's per-thread RNG.

`GREY_METHODS` and the STRESS defaults are spelled out in `pipeline.py` as well
as in `stress.py`, deliberately: building the parser is what every `--help` and
every subcommand does, and importing the converter would drag numba's two
seconds into a `calibrate` run. `test_stress.py` compares the two copies.

## Gotchas

- **The bundled charsets carry no index → key mapping.** They are photographic
  scans of *other* machines (Hermes, Smith-Corona, Daisywriter); a
  `choices.json` index refers to a tile position in a scan, and nothing records
  which key makes that mark. Only `sigma-10` / `sigma-12`, which ship a
  `glyphs.json`, can drive a typewriter.
- **Layer offsets must be multiples of 0.5.** The machine has no quarter-step,
  so `16x1` and `daisy_full` are rejected by the planner. Typeable schemes:
  `1x1 1x2 1x4 2Hx1 2Hx2 2Vx1 2Vx2 4x1 4x2`.
- **Image width is capped by the carriage** — 65 columns at pitch 10, 78 at
  pitch 12. The planner refuses anything wider rather than truncating it.
- **`results/` is gitignored.** Anything there is regenerable; never treat it
  as input.
- A *single* wrong glyph on a printed sheet is a corrupted byte on the wire,
  not a charset problem — a charset problem corrupts every instance of that
  character. See `erika_ai`'s notes on the typewriter link.
- **A half-cell shift is not a misalignment.** `4x1` places layers at 0 and 0.5 of
  a cell, so the placements are already half-cell periodic and shifting the target
  half a cell maps every layer-0 cell onto a layer-1 cell — a relabelling, not a
  better fit. The effective period in each axis is therefore *half* a cell, the
  paper's quarters sample it twice over, and `--align-steps 2` probes almost
  nothing but the scale and the border. What is left over at a half-cell shift is
  a border effect, which is why it is not exactly a no-op on a photograph.
- **Force blocks in a charset sheet run contiguously, not row-aligned.** A blank
  tile *between* two glyphs is dropped by `chop_charset()` and shifts every index
  after it, so a force block starts wherever the last one ended, part-way through
  a row if that is where it lands. `pipeline sheet` types them in the same order
  for the same reason. `preview.png` labels each tile's force, which is the one
  place the blocks can be told apart by eye.
- **The tonal figures in the two `CLAUDE.md` files that quote them are stale.**
  Regenerating the charsets with the ink model changed the gamut: measured on flat
  patches, `sigma-10` at `4x1` now bottoms out around grey 143 with 17 distinct
  levels (was 132 and 14), and at `4x2` around 92 with 25 (was 79 and 18) — a
  lighter floor, which is what non-black ink means, but a longer usable ramp and
  no bare-paper cliff at `4x2`. `erika-studio`'s `tone.py` derives all of this
  from the charset at run time, so the server is right and only the prose is
  wrong; re-measure before quoting a number.

## Conventions

- **`optimize.py` and `utils.py` are upstream.** Change them only for genuine
  incompatibilities or defects, minimally, marked `FORK FIX` with a comment
  saying why, and with a test in `src/tests/` — a rebase is what undoes these,
  and the test is what notices. Everything else of ours lives in `src/erika/`
  and `src/tests/`, which keeps the fork rebaseable. Two fixes exist:
  - matplotlib ≥ 3.7 rejects scalars in `set_xdata`
  - a single layer was composited against *itself*. `bg` for "the other layers"
    is `layers[(layer_num + 1) % len(layer_offsets)]`, which with one layer
    wraps back round to it — so every candidate glyph was scored as if struck
    twice, and the mockup could not be reproduced by any plan that strikes a
    cell once. `erika.pipeline print` reported a mismatch on every `1x1` job.
    A lone layer now gets bare paper, and `1x1` verifies like the rest.
- Comments explain *why*, not *what*. Much of `src/erika/` documents decisions
  that are not recoverable from the code — preserve that when editing.
- Code, comments and documentation in English.
