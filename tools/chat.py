"""Chat with the board (firmware/chat) over its USB serial port.

usage: uv run python tools/chat.py /dev/ttyACM0 [--reset] [--log FILE]

Type a message and press Enter; the reply streams as the board generates it, followed by a stats
line ([<new> tok, <tok/s> tok/s] plus notes). Line editing and history (Up/Down) are local
(readline): the board only ever receives whole lines. Ctrl-C while a reply streams stops it (sends
0x03; a second Ctrl-C stops waiting for the board); Ctrl-C at the prompt clears the line; Ctrl-D
exits. /reset starts a new conversation. Empty lines are not sent.

--reset reboots the board first (RTS pulse) and waits for its banner (~7 s: the system prompt is
prefilled at boot). --log FILE writes the session (a new file each run, overwritten; first line a
"# tools/chat.py session" header) as it appeared: received bytes verbatim, each sent line after the
prompt, "^C" where Ctrl-C was sent.

Serial protocol: see the header of firmware/chat/main/chat_main.c.
"""
import argparse
import codecs
import re
import signal
import sys
import time

import serial

try:
    import readline  # noqa: F401  (line editing and history for input())
except ImportError:  # pragma: no cover - not on Linux
    pass

# The board's response to a line ends with one of these lines, then the prompt "> ".
END = re.compile(
    rb"(?:\A|\n)(?:\[\d+ tok, [0-9.]+ tok/s(?:, dropped \d+ earlier turns)?(?:, serial_lost=\d+)?(?:, stack_free=\d+ worker_stack_free=\d+)?\](?: \((?:context full|stopped)\))?"
    rb"|\(new conversation\)|! [^\n]*|ELLM [^\n]*)\n> \Z")
FAIL_LINE = re.compile(rb"(?:\A|\n)ELLM FAIL[^\n]*\n")
PROMPT = b"> "


class Session:
    def __init__(self, port: str, log_path: str | None):
        # default DTR/RTS (both asserted) on open: no reset of the USB-Serial-JTAG chip
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.log = open(log_path, "wb") if log_path else None  # a fresh transcript per session
        if self.log:
            self.log.write(f"# tools/chat.py session {time.strftime('%Y-%m-%d %H:%M:%S')} on {port}\n".encode())
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _show(self, b: bytes) -> None:
        if not b:
            return
        sys.stdout.write(self.dec.decode(b))
        sys.stdout.flush()
        if self.log:
            self.log.write(b)
            self.log.flush()

    def _log(self, b: bytes) -> None:
        if self.log:
            self.log.write(b)
            self.log.flush()

    def reset(self) -> None:
        self.s.dtr = False
        self.s.rts = True
        time.sleep(0.1)
        self.s.rts = False
        self.s.dtr = True

    def wait(self, idle_s: float | None = None, timeout_s: float | None = None) -> bool:
        """Stream the board's output until it shows the prompt (True). With idle_s: also stop
        after idle_s without data (False). Ctrl-C sends 0x03 once, a second one stops waiting."""
        hits = [0]  # Ctrl-C presses while waiting (a flag, so no KeyboardInterrupt mid-update)
        old = signal.signal(signal.SIGINT, lambda *_: hits.__setitem__(0, hits[0] + 1))
        try:
            return self._wait(hits, idle_s, timeout_s)
        finally:
            signal.signal(signal.SIGINT, old)

    def _wait(self, hits: list, idle_s: float | None, timeout_s: float | None) -> bool:
        buf, shown, sent_ctrl_c = b"", 0, False
        last = start = time.monotonic()
        while True:
            if hits[0] and not sent_ctrl_c:
                sent_ctrl_c = True
                self.s.write(b"\x03")
                self._log(b"^C")
            if hits[0] > 1:
                self._show(buf[shown:])
                print("\n(stopped waiting for the board)")
                return False
            d = self.s.read(self.s.in_waiting or 1)
            now = time.monotonic()
            if d:
                buf += d
                last = now
            if END.search(buf):
                self._show(buf[shown:-len(PROMPT)])
                return True
            if FAIL_LINE.search(buf) and now - last > 0.5:
                # a failure without a prompt (the board idles): show it and stop waiting
                self._show(buf[shown:])
                return False
            # show all but a possible start of the final "\n> "
            hold = 3 if buf.endswith(b"\n> ") else 2 if buf.endswith(b"\n>") else 1 if buf.endswith(b"\n") else 0
            self._show(buf[shown:len(buf) - hold])
            shown = len(buf) - hold
            if idle_s is not None and now - last > idle_s and b"ELLM chat: loading" not in buf:
                self._show(buf[shown:])
                return False
            if timeout_s is not None and now - start > timeout_s:
                self._show(buf[shown:])
                return False

    def send(self, line: str) -> None:
        data = line.encode("utf-8", "surrogateescape")
        self._log(data + b"\n")
        self.s.write(data + b"\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("port")
    ap.add_argument("--reset", action="store_true", help="reboot the board first and wait for its banner")
    ap.add_argument("--log", help="write the session transcript to this file (overwritten)")
    a = ap.parse_args()
    sess = Session(a.port, a.log)
    if a.reset:
        sess.reset()
        if not sess.wait(timeout_s=60):
            print("\n(no prompt from the board after reset)")
    else:
        sess.wait(idle_s=0.5)  # whatever the board printed before we connected (banner, loading)
    print("(connected; Ctrl-C stops a reply, Ctrl-D exits)")
    while True:
        try:
            line = input("> ")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            return 0
        if not line:
            continue
        sess._log(PROMPT)
        sess.send(line)
        sess.wait()


if __name__ == "__main__":
    sys.exit(main())
