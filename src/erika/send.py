"""Send a .etp print job to the ESP32 over USB serial, and drive the print.

    python -m erika.send results/photo.etp --port COM6 --print

Speaks the line-framed base64 protocol implemented by
``erika_ai/src/image_receiver.cpp``::

    ->  IMG UPLOAD 8814
    <-  UPLOAD-READY 128
    ->  D SVRQMQECAAA...
    <-  ACK 128
        ...
    ->  Z
    <-  OK 8814 bytes stored and verified

Everything is newline-framed text, so the payload rides the same reader as
every other command: no modal switch into a raw binary mode that can
desynchronise, and log output the firmware interleaves is harmless. The
running ACK total catches a dropped line at once, and the job's own CRC-32 --
checked by the firmware after the last line -- catches corruption.

Lines are kept short enough to fit whole in the device's serial receive
buffer, so the firmware never has to keep pace with the wire mid-line -- it
cannot, because writing the previous chunk to flash can stall for tens of
milliseconds.

If a transfer fails, `--diagnose` walks the link one step at a time and says
which step broke.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from typing import Protocol

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import etp

#: Decoded bytes per data line. Must match IMG_UPLOAD_CHUNK in image_receiver.h.
#:
#: Deliberately small: a whole data line has to fit in the device's serial
#: receive buffer, whose stock size is 256 bytes on both HWCDC and
#: HardwareSerial. 128 decoded bytes is 172 base64 characters, so the firmware
#: never has to keep pace with the wire in the middle of a line.
CHUNK_SIZE = 128


def _require_serial():
    try:
        import serial  # noqa: F401
        from serial.tools import list_ports  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyserial is not installed. In the project venv:\n"
            "    pip install pyserial"
        )
    return sys.modules["serial"]


def list_ports() -> list[str]:
    _require_serial()
    from serial.tools import list_ports as lp

    return [p.device for p in lp.comports()]


def autodetect_port() -> str:
    def looks_like_a_board(name: str) -> bool:
        low = name.lower()
        # macOS /dev/cu.usbmodem*, Linux /dev/ttyUSB* and /dev/ttyACM*, Windows COMn
        return "usb" in low or "acm" in low or low.startswith("com")

    ports = [p for p in list_ports() if looks_like_a_board(p)]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise SystemExit("no USB serial ports found -- is the ESP32 plugged in?")
    raise SystemExit(
        "several USB serial ports found; pick one with --port:\n  "
        + "\n  ".join(ports)
    )


#: Reply tokens the firmware uses to open a message. Anything else on the wire
#: is log output from another subsystem and gets skipped.
REPLY_PREFIXES = ("OK", "ERR", "ACK", "PONG", "STATE", "INFO", "UPLOAD-READY")


class Console(Protocol):
    """The conversation the transfer needs, so tests can stand in for a port."""

    def send_line(self, text: str) -> None: ...
    def await_reply(self, *prefixes: str, timeout: float = ...) -> str: ...
    def drain(self, seconds: float = ...) -> list[str]: ...
    def resync(self) -> None: ...


class Link:
    """A line-oriented conversation with the firmware's IMG console."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 10.0,
                 verbose: bool = False, reset: bool = True, settle: float = 2.0):
        serial = _require_serial()
        self.verbose = verbose
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = timeout
        self.ser.write_timeout = 10.0
        # pyserial asserts DTR and RTS on open. On boards with a USB-serial
        # bridge those lines are wired to EN and GPIO0, so leaving them
        # asserted can hold the ESP32 in reset or drop it into the download
        # ROM -- where it answers nothing. Park them low before opening.
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.ser.dtr = False
        self.ser.rts = False
        if reset:
            # Boot plus setup() -- WiFi association in particular -- takes a
            # while, and loop() is not running until it finishes.
            time.sleep(settle)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def send_line(self, text: str) -> None:
        if self.verbose:
            print(f"  -> {text if len(text) < 90 else text[:87] + '...'}")
        self.ser.write((text + "\n").encode("ascii"))
        self.ser.flush()

    def read_line(self) -> str:
        raw = self.ser.readline()
        if not raw:
            raise TimeoutError("no reply from the device")
        line = raw.decode(errors="replace").rstrip("\r\n")
        if self.verbose:
            print(f"  <- {line}")
        return line

    def await_reply(self, *prefixes: str, timeout: float = 30.0,
                    errors: str = "raise") -> str:
        """Read until a line starts with one of `prefixes`, or with ERR.

        `errors="skip"` treats an ERR as ordinary noise instead. Only `resync`
        wants that, and for a specific reason -- see there.
        """
        wanted = prefixes or REPLY_PREFIXES
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            try:
                line = self.read_line()
            except TimeoutError:
                continue
            if line.startswith("ERR"):
                if errors != "skip":
                    raise RuntimeError(line)
                seen.append(line)
                continue
            if any(line.startswith(p) for p in wanted):
                return line
            seen.append(line)
        hint = ""
        if seen:
            hint = "\n  device said instead:\n    " + "\n    ".join(seen[-6:])
        raise TimeoutError(
            f"device never replied with {'/'.join(wanted)}{hint}\n"
            "  If it said nothing at all, the firmware may still be in setup() "
            "(WiFi association blocks loop()); try --settle 10.\n"
            "  Check the link with:  python -m erika.send --port <port> --diagnose"
        )

    def command(self, text: str, echo: bool = True) -> str:
        """Send a command and return its first reply line."""
        self.send_line(text)
        line = self.await_reply()
        if echo:
            print(line)
        return line

    def drain(self, seconds: float = 0.4) -> list[str]:
        """Collect any trailing lines, e.g. the INFO block after an OK."""
        old, self.ser.timeout = self.ser.timeout, seconds
        out = []
        try:
            while True:
                try:
                    out.append(self.read_line())
                except TimeoutError:
                    break
        finally:
            self.ser.timeout = old
        return out

    def resync(self) -> None:
        """Get the device to a known-quiet state before trusting a reply.

        A transfer that died leaves the firmware answering the data lines still
        in flight. Those replies arrive after the port is opened, so without
        this the next run reads a stale error as the answer to its first
        command -- which is exactly as confusing as it sounds.

        Two things make draining until silence the wrong way to do it, and the
        first is not an error at all:

        `IMG CANCEL` is answered "ERR no upload in progress" whenever nothing is
        in flight, which is the normal state of a board that has just been
        opened. So this provokes an ERR every time it runs, and its own reply is
        the stale line most likely to be mistaken for the next answer.

        And the board may still be in `setup()`, answering nothing for many
        seconds: WiFi association waits up to 15s and `loop()` does not run until
        it returns. The commands sit in the receive buffer meanwhile and are all
        answered in one pass when it does -- so the reply to `IMG CANCEL` can
        arrive *after* the next command was sent, whatever was drained first.

        Hence a sentinel instead. `IMG STATUS` is answered with "STATE ..." in
        every state the firmware has, so everything ahead of that line is stale
        by definition, whenever it turns up and whatever it says.

        Asked twice, because the other thing a reset does is throw the question
        away: lines that arrive while the boot ROM still holds the port are
        consumed by it and never reach the firmware, and waiting the whole
        timeout out for a reply nothing ever heard is a poor way to spend it.
        """
        # Half the budget each: long enough to outlast association (15s) plus
        # the fixed delays in setup(), and a board silent for both halves is not
        # merely booting.
        for attempt in (1, 2):
            self.send_line("IMG CANCEL")
            self.send_line("IMG STATUS")
            try:
                self.await_reply("STATE", timeout=12.5, errors="skip")
                break
            except TimeoutError:
                if attempt == 2:
                    raise
        self.drain()
        self.ser.reset_input_buffer()


def upload(link: Console, path: str, progress: bool = True, retries: int = 2) -> None:
    """Upload a job, retrying the whole transfer if a line goes missing."""
    job = etp.load(path)  # parse locally first: fail fast on a bad file
    data = open(path, "rb").read()
    print(f"{os.path.basename(path)}: {len(data)} bytes, {job.strikes} strikes, "
          f"{job.cols}x{job.rows} cells")

    for attempt in range(retries + 1):
        try:
            _upload_once(link, data, progress=progress)
            return
        except (RuntimeError, TimeoutError) as exc:
            if attempt == retries:
                raise
            print(f"\n  attempt {attempt + 1} failed ({exc}); retrying")
            link.resync()


def _upload_once(link: Console, data: bytes, progress: bool = True) -> None:
    link.send_line(f"IMG UPLOAD {len(data)}")
    ready = link.await_reply("UPLOAD-READY")
    parts = ready.split()
    chunk = int(parts[1]) if len(parts) > 1 else CHUNK_SIZE

    sent = 0
    start = time.time()
    while sent < len(data):
        block = data[sent : sent + chunk]
        link.send_line("D " + base64.b64encode(block).decode("ascii"))
        sent += len(block)
        acked = int(link.await_reply("ACK").split()[1])
        if acked != sent:
            raise RuntimeError(
                f"device has {acked} bytes, we have sent {sent} -- a data line "
                "was lost or truncated"
            )
        if progress:
            pct = 100 * sent / len(data)
            elapsed = time.time() - start
            print(f"\r  uploading {pct:5.1f}%  {sent}/{len(data)} bytes  "
                  f"{sent / max(elapsed, 1e-6) / 1024:.1f} kB/s", end="", flush=True)
    if progress:
        print()

    link.send_line("Z")
    print(link.await_reply("OK"))
    for extra in link.drain():
        print(f"  {extra}")


def diagnose(link: Console) -> int:
    """Walk the link one step at a time and report where it breaks."""
    print("\n1. console alive?")
    try:
        print(f"   {link.command('IMG PING', echo=False)}")
    except (RuntimeError, TimeoutError) as exc:
        print(f"   FAILED: {exc}")
        print("\n   The firmware is not answering text commands. Check that the")
        print("   board is flashed with this firmware, that --port is right, and")
        print("   that nothing else (a serial monitor, the PlatformIO terminal)")
        print("   holds the port open.")
        return 1

    print("\n2. single-line upload?")
    probe = etp.pack(etp.Job(body=bytes([etp.OP_END]), cols=1, rows=1, strikes=0))
    try:
        _upload_once(link, probe, progress=False)
    except (RuntimeError, TimeoutError) as exc:
        print(f"   FAILED: {exc}")
        return 1

    # A single short line proves almost nothing: the interesting failure is a
    # transfer long enough that the device has to drain the port while writing
    # the previous chunk to flash. Push enough lines to actually provoke it.
    print(f"\n3. multi-line upload? ({CHUNK_SIZE}-byte lines)")
    filler = etp.Encoder()
    for _ in range(1500):
        filler.right(2)
    filler.end()
    big = etp.pack(etp.Job(body=filler.body(), cols=1, rows=1, strikes=0))
    lines = -(-len(big) // CHUNK_SIZE)
    print(f"   {len(big)} bytes over {lines} data lines")
    try:
        _upload_once(link, big, progress=False)
    except (RuntimeError, TimeoutError) as exc:
        print(f"   FAILED: {exc}")
        print("\n   Short transfers work but long ones drop lines. That is the")
        print("   receive buffer overrunning: check that setup() calls")
        print("   Serial.setRxBufferSize() before Serial.begin(), and that the")
        print("   board is running a current build.")
        return 1

    print("\n4. stored job?")
    print(f"   {link.command('IMG INFO', echo=False)}")
    for extra in link.drain():
        print(f"   {extra}")

    print("\nlink is healthy. Note that the probes above replaced any stored job")
    print("-- upload the real one again before printing.")
    return 0


def watch(link: Console, poll: float = 5.0) -> int:
    """Poll IMG STATUS until the job finishes or fails."""
    print("watching progress (Ctrl-C to stop watching; the print continues)")
    try:
        while True:
            link.send_line("IMG STATUS")
            state = link.await_reply("STATE").removeprefix("STATE ").strip()
            print(f"\r  {state}".ljust(100), end="", flush=True)
            if state.startswith("FINISHED"):
                print("\ndone")
                return 0
            if state.startswith("FAILED"):
                print("\nprint failed")
                return 1
            if state.startswith(("IDLE", "READY")):
                print("\nprinter is idle -- job not running")
                return 1
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nstopped watching; the typewriter carries on")
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("file", nargs="?", help="the .etp job to upload")
    p.add_argument("--port", "-p", default=None,
                   help="serial port (autodetected if there is exactly one)")
    p.add_argument("--baud", "-b", type=int, default=115200)
    p.add_argument("--print", "-P", dest="do_print", action="store_true",
                   help="start printing once the upload is verified")
    p.add_argument("--from-pass", type=int, default=0,
                   help="resume at a carriage-return pass (paper feeds are replayed)")
    p.add_argument("--speed", type=int, default=None,
                   help="overall typing speed: ms before a glyph the wheel has "
                        "to turn to (default 100). The firmware scales its "
                        "shorter delays -- a repeated glyph, a carriage step -- "
                        "from this one number")
    p.add_argument("--watch", "-w", action="store_true", help="poll progress until done")
    p.add_argument("--command", "-c", default=None,
                   help="send one IMG command instead of uploading, e.g. 'STATUS'")
    p.add_argument("--diagnose", action="store_true",
                   help="test the link step by step and report where it breaks")
    p.add_argument("--retries", type=int, default=2,
                   help="restart the whole transfer this many times if a data "
                        "line goes missing (default 2)")
    p.add_argument("--settle", type=float, default=2.0,
                   help="seconds to wait for the board to boot after opening the "
                        "port; raise it if setup() is slow (default 2)")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)

    if a.list_ports:
        for port in list_ports():
            print(port)
        return 0

    if not a.file and not a.command and not a.diagnose:
        p.error("give a .etp file to upload, or --command / --diagnose")

    port = a.port or autodetect_port()
    print(f"connecting to {port} at {a.baud} baud")
    link = Link(port, a.baud, verbose=a.verbose, settle=a.settle)
    try:
        # Clear any backlog from a previous run before trusting a reply.
        link.resync()

        if a.diagnose:
            return diagnose(link)
        if a.command:
            link.command(f"IMG {a.command}")
            for extra in link.drain():
                print(f"  {extra}")
            return 0

        upload(link, a.file, retries=a.retries)
        if a.speed is not None:
            link.command(f"IMG SPEED {a.speed}")
        if a.do_print:
            link.command(f"IMG PRINT pass {a.from_pass}"
                         if a.from_pass else "IMG PRINT")
            if a.watch:
                return watch(link)
            print("printing; check progress with:  "
                  f"python -m erika.send --port {port} -c STATUS")
        return 0
    except (RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
