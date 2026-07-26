# Typewriter Projects

A **Sigma SM 8200i** electric typewriter driven by an ESP32. It does two things:

- **[erika_ai](erika_ai/)** — a chatbot terminal. Type a message, press carriage
  return, and the machine types back a reply from Botario or OpenAI.
- **[typewriter-art](typewriter-art/)** — photographs, typed. Thousands of
  overlapping characters on a half-character grid, from
  [Kühn's optimizer](https://github.com/juleskuehn/typewriter-art) (Graphics
  Interface 2021).

```
                  ┌──────────────┐
   keystrokes ───►│              │───► Botario / OpenAI
                  │    ESP32     │
   photo ─┐       │              │
          │       └──────┬───────┘
          ▼              │ serial + RTS
   typewriter-art ──.etp─┘
      pipeline                    ▼
                            print head ──► paper
```

## Printing a photograph

```bash
cd typewriter-art
python3.10 -m venv .venv
.venv/bin/pip install -r src/requirements-erika.txt
cd src

../.venv/bin/python -m erika.pipeline charset --pitch 10       # once
../.venv/bin/python -m erika.pipeline calibrate                # once, see below
../.venv/bin/python -m erika.send results/calibrate.etp --print

../.venv/bin/python -m erika.pipeline print -t images/mwdog_crop.png -r 48
../.venv/bin/python -m erika.send results/photo.etp --print --watch
```

Look at `results/erika_plan_jitter.png` before you start — that is the preview
with realistic registration error, and it is a much better predictor of the
printed sheet than the clean mockup.

Full documentation: **[typewriter-art/src/erika/README.md](typewriter-art/src/erika/README.md)**.

## How the two halves fit together

The optimizer produces a `choices.json` describing which character sits in
which cell of which overlapping layer. That is a picture, not instructions:
it says nothing about which *key* to press, and it assumes a typist who can
nudge the paper by half a character.

The [`erika` package](typewriter-art/src/erika/) supplies both missing pieces:

| | |
|---|---|
| `erika_codes.py` | the Sigma's keys and motion codes, mirroring `erika_char_map.cpp` |
| `make_charset.py` | a charset whose glyph indices map back to actual keys |
| `planner.py` | layers → an ordered list of strikes and the moves between them |
| `etp.py` | the `.etp` print-job container |
| `preview.py` | renders a plan back to an image, clean and jittered |
| `emulate.py` | a virtual typewriter, for testing without paper |
| `pipeline.py` | the CLI |
| `send.py` | uploads to the ESP32 over USB serial |

On the firmware side, [`erika_image.*`](erika_ai/src/erika_image.h) interprets
the opcode stream and [`image_receiver.*`](erika_ai/src/image_receiver.h)
handles upload and the `IMG` command set. All geometry is resolved on the
host, so the firmware never has to know where the print head is.

## Why the pipeline checks itself

Between the photograph and the paper there are six stages, and every one of
them can quietly ruin the picture — a shifted character index, an off-by-one
in the half-step arithmetic, a layer flattened in the wrong order. None of
those fail loudly; they just produce a sheet of plausible-looking noise after
half an hour of typing.

So each stage is checked against the one before it:

- The charset generator re-runs the upstream loader and asserts the glyph
  count, because `chop_charset()` silently drops tiles it judges blank and
  that would shift every index after the dropped one.
- Every `print` run re-renders the finished plan and diffs it against the
  optimizer's own mockup. It should read `plan reproduces optimize.py's mockup
  exactly`.
- The test suite runs plans through a virtual typewriter — expanding opcodes
  to raw bytes, moving a simulated carriage — and checks that every strike
  lands where the plan said.
- The Erika byte codes and `.etp` opcodes are written by hand in both Python
  and C++; a test parses the header and compares, so the two cannot drift.

```bash
cd typewriter-art/src && ../.venv/bin/python -m pytest tests -q
```

## Calibrate first

One motion code is inferred rather than documented: **half-step forward
(`0x73`)**. The firmware's own notes cover the codes the typewriter *emits*,
and `0x73` is the only gap in the contiguous `0x71..0x79` motion block, which
otherwise runs in forward/backward pairs. The inference is well-founded but it
is still an inference, and the whole half-character grid depends on it.

`python -m erika.pipeline calibrate` types a test pattern that settles it in
about a minute of typing, along with checks for platen drift, backspace
registration and carriage repeatability. Do that before your first photograph.

## Physical limits

| | pitch 10 | pitch 12 |
|---|---|---|
| character cell | 2.54 × 4.23 mm | 2.12 × 4.23 mm |
| carriage width | 65 columns | 78 columns |
| `-r 48` on a 4:5 photo | 124 × 171 mm | 103 × 171 mm |

A 48-column picture with four layers is roughly 5,000 strikes — about 20–25
minutes at the default 100 ms per character. Jobs live in SPIFFS and survive a
reboot, and `IMG PRINT pass N` resumes at a given row after a ribbon change.
