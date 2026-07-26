"""Send a .etp print job to the ESP32 over USB serial, and drive the print.

    python -m erika.send results/photo.etp --port COM6 --print

Speaks the line-framed base64 protocol implemented by
``erika_ai/src/image_receiver.cpp``::

    ->  IMG UPLOAD 8814
    <-  UPLOAD-READY 256
    ->  D SVRQMQECAAA...
    <-  ACK 256
        ...
    ->  Z
    <-  OK 8814 bytes stored and verified

Everything is newline-framed text, so the payload rides the same reader as
every other command: no modal switch into a raw binary mode that can
desynchronise, and log output the firmware interleaves is harmless. The
running ACK total catches a dropped line at once, and the job's own CRC-32 --
checked by the firmware after the last line -- catches corruption.

If a transfer fails, `--diagnose` walks the link one step at a time and says
which step broke.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import etp

#: Decoded bytes per data line. Must match IMG_UPLOAD_CHUNK in image_receiver.h.
CHUNK_SIZE = 256


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

    def await_reply(self, *prefixes: str, timeout: float = 30.0) -> str:
        """Read until a line starts with one of `prefixes`, or with ERR."""
        wanted = prefixes or REPLY_PREFIXES
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            try:
                line = self.read_line()
            except TimeoutError:
                continue
            if line.startswith("ERR"):
                raise RuntimeError(line)
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


def upload(link: Link, path: str, progress: bool = True) -> None:
    job = etp.load(path)  # parse locally first: fail fast on a bad file
    data = open(path, "rb").read()
    print(f"{os.path.basename(path)}: {len(data)} bytes, {job.strikes} strikes, "
          f"{job.cols}x{job.rows} cells")

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
                "was lost. Try a lower --baud."
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


def diagnose(link: Link) -> int:
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

    print("\n2. base64 upload path?")
    probe = etp.pack(etp.Job(body=bytes([etp.OP_END]), cols=1, rows=1, strikes=0))
    try:
        link.send_line(f"IMG UPLOAD {len(probe)}")
        ready = link.await_reply("UPLOAD-READY")
        print(f"   {ready}")
        link.send_line("D " + base64.b64encode(probe).decode("ascii"))
        print(f"   {link.await_reply('ACK')}")
        link.send_line("Z")
        print(f"   {link.await_reply('OK')}")
        for extra in link.drain():
            print(f"   {extra}")
    except (RuntimeError, TimeoutError) as exc:
        print(f"   FAILED: {exc}")
        return 1

    print("\n3. stored job?")
    print(f"   {link.command('IMG INFO', echo=False)}")
    for extra in link.drain():
        print(f"   {extra}")

    print("\nlink is healthy. Note that step 2 replaced any stored job with a")
    print("one-opcode probe -- upload the real one again before printing.")
    return 0


def watch(link: Link, poll: float = 5.0) -> int:
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
                   help="ms between characters on the typewriter (default 100)")
    p.add_argument("--watch", "-w", action="store_true", help="poll progress until done")
    p.add_argument("--command", "-c", default=None,
                   help="send one IMG command instead of uploading, e.g. 'STATUS'")
    p.add_argument("--diagnose", action="store_true",
                   help="test the link step by step and report where it breaks")
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
        if a.diagnose:
            return diagnose(link)
        if a.command:
            link.command(f"IMG {a.command}")
            for extra in link.drain():
                print(f"  {extra}")
            return 0

        upload(link, a.file)
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
