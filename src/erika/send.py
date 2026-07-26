"""Send a .etp print job to the ESP32 over USB serial, and drive the print.

    python -m erika.send results/photo.etp --port /dev/cu.usbmodem1101 --print

Speaks the chunked, acknowledged protocol implemented by
``erika_ai/src/image_receiver.cpp``::

    ->  IMG UPLOAD 8814
    <-  READY 512
    ->  <512 raw bytes>
    <-  ACK 512
        ...
    <-  OK 8814 bytes stored and verified

Integrity is not checked by this protocol -- it is checked by the CRC in the
job's own header, which the firmware verifies after the last chunk lands. A
mangled transfer fails there rather than on paper.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erika import etp

CHUNK_SIZE = 512  # must match IMG_UPLOAD_CHUNK in image_receiver.h


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
    ports = [p for p in list_ports() if "usb" in p.lower() or "ACM" in p]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise SystemExit("no USB serial ports found -- is the ESP32 plugged in?")
    raise SystemExit(
        "several USB serial ports found; pick one with --port:\n  "
        + "\n  ".join(ports)
    )


class Link:
    """A line-oriented conversation with the firmware's IMG console."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 10.0,
                 verbose: bool = False):
        serial = _require_serial()
        self.verbose = verbose
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # The ESP32 resets when the port opens; give it a moment, then drop
        # whatever boot chatter is already in the buffer.
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def send_line(self, text: str) -> None:
        if self.verbose:
            print(f"  -> {text}")
        self.ser.write((text + "\n").encode())
        self.ser.flush()

    def read_line(self) -> str:
        raw = self.ser.readline()
        if not raw:
            raise TimeoutError("no reply from the device")
        line = raw.decode(errors="replace").rstrip("\r\n")
        if self.verbose:
            print(f"  <- {line}")
        return line

    def await_reply(self, *prefixes: str, skip_logs: bool = True) -> str:
        """Read until a line starts with one of `prefixes`, or ERR."""
        deadline = time.time() + 30
        while time.time() < deadline:
            line = self.read_line()
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if any(line.startswith(p) for p in prefixes):
                return line
            if not skip_logs and line:
                print(line)
        raise TimeoutError(f"device never replied with {prefixes}")

    def command(self, text: str, echo: bool = True) -> str:
        """Send a command and return the first non-log reply line."""
        self.send_line(text)
        line = self.await_reply("OK", "ERR", "IDLE", "READY", "PRINTING",
                                "PAUSED", "FINISHED", "FAILED", "grid",
                                "no job", skip_logs=True)
        if echo:
            print(line)
        return line


def upload(link: Link, path: str, progress: bool = True) -> None:
    job = etp.load(path)  # parse locally first: fail fast on a bad file
    data = open(path, "rb").read()
    print(f"{os.path.basename(path)}: {len(data)} bytes, {job.strikes} strikes, "
          f"{job.cols}x{job.rows} cells")

    link.send_line(f"IMG UPLOAD {len(data)}")
    ready = link.await_reply("READY")
    chunk = int(ready.split()[1]) if len(ready.split()) > 1 else CHUNK_SIZE

    sent = 0
    start = time.time()
    while sent < len(data):
        block = data[sent : sent + chunk]
        link.ser.write(block)
        link.ser.flush()
        sent += len(block)
        ack = link.await_reply("ACK")
        acked = int(ack.split()[1])
        if acked != sent:
            raise RuntimeError(f"device acked {acked} bytes, we sent {sent}")
        if progress:
            pct = 100 * sent / len(data)
            elapsed = time.time() - start
            print(f"\r  uploading {pct:5.1f}%  {sent}/{len(data)} bytes  "
                  f"{sent / max(elapsed, 1e-6) / 1024:.1f} kB/s", end="", flush=True)
    if progress:
        print()

    print(link.await_reply("OK"))
    # The firmware follows OK with an INFO block; surface it if it arrives.
    link.ser.timeout = 0.5
    while True:
        try:
            extra = link.read_line()
        except TimeoutError:
            break
        if not extra:
            break
        print(f"  {extra}")
    link.ser.timeout = 10.0


def watch(link: Link, poll: float = 5.0) -> int:
    """Poll IMG STATUS until the job finishes or fails."""
    print("watching progress (Ctrl-C to stop watching; the print continues)")
    try:
        while True:
            link.send_line("IMG STATUS")
            line = link.await_reply("IDLE", "READY", "PRINTING", "PAUSED",
                                    "FINISHED", "FAILED")
            print(f"\r  {line}".ljust(100), end="", flush=True)
            if line.startswith("FINISHED"):
                print("\ndone")
                return 0
            if line.startswith("FAILED"):
                print("\nprint failed")
                return 1
            if line.startswith(("IDLE", "READY")):
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
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)

    if a.list_ports:
        for port in list_ports():
            print(port)
        return 0

    if not a.file and not a.command:
        p.error("give a .etp file to upload, or --command to just talk to the device")

    port = a.port or autodetect_port()
    print(f"connecting to {port} at {a.baud} baud")
    link = Link(port, a.baud, verbose=a.verbose)
    try:
        if a.command:
            link.command(f"IMG {a.command}")
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
