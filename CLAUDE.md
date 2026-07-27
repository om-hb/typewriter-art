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
PY -m pytest tests -q                                   # 49 tests
PY -m erika.pipeline charset --pitch 10                 # build the Sigma charset
PY -m erika.pipeline print -t images/mwdog_crop.png -r 48
PY -m erika.pipeline calibrate                          # machine test pattern
PY -m erika.pipeline area                               # corners of the print area
PY -m erika.pipeline verify                             # plan vs. optimizer mockup
PY -m erika.etp results/photo.etp -n 30                 # disassemble a job
PY -m erika.send results/photo.etp --port COM6 --print --watch
PY -m erika.send --port COM6 --diagnose                 # when uploads fail
```

Subcommands: `charset print plan verify calibrate area sheet`.

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
    planner.py             layers -> ordered strikes + carriage/paper moves
    etp.py                 the .etp print-job container
    preview.py             renders a plan back to an image
    emulate.py             virtual typewriter; mirrors erika_image.cpp
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
twice — here in Python, and in C++ in `erika_ai/src/`. Four tests parse those
headers and compare. **They only run with `erika_ai` checked out**: beside this
repository by default, or wherever `ERIKA_FIRMWARE_SRC` points. If it is
missing the suite prints a loud `drift guards did not run` banner — do not
ignore it, those four tests are the only protection the tables have.

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

## Conventions

- **`optimize.py` and `utils.py` are upstream.** Change them only for genuine
  incompatibilities, minimally, with a comment saying why. One such fix exists
  (matplotlib ≥ 3.7 rejects scalars in `set_xdata`). Everything of ours lives
  in `src/erika/` and `src/tests/`, which keeps the fork rebaseable.
- Comments explain *why*, not *what*. Much of `src/erika/` documents decisions
  that are not recoverable from the code — preserve that when editing.
- Code, comments and documentation in English.
