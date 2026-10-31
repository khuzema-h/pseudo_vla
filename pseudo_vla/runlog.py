"""Save one terminal transcript for each simulator run.

Stdout and stderr are copied at the file-descriptor level, so Python prints
and Isaac Sim's own logs both land in outputs/logs.
"""

from __future__ import annotations

import atexit
import datetime
import os
import pty
import sys
import termios
import threading
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_state: "_RunLog | None" = None


@dataclass
class _Tee:
    target_fd: int
    saved_fd: int
    read_fd: int
    thread: threading.Thread


@dataclass
class _RunLog:
    path: Path
    log_fd: int
    tees: list[_Tee]
    stopping: bool = False


def start_run_log(directory: Path | None = None) -> Path:
    """Begin copying this process's terminal output into a new log file."""
    global _state
    with _LOCK:
        if _state is not None:
            return _state.path
        folder = directory or (ROOT / "outputs" / "logs")
        folder.mkdir(parents=True, exist_ok=True)
        path = _new_path(folder)
        log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        started = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"started {started}\ncommand: {' '.join(sys.argv)}\n"
        _write_all(log_fd, header.encode("utf-8", errors="replace"))
        tees = [_tee(1, log_fd), _tee(2, log_fd)]
        _state = _RunLog(path=path, log_fd=log_fd, tees=tees)
        latest = folder / "latest.log"
        _replace_latest(latest, path.name)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    atexit.register(stop_run_log)
    return path


def append_run_log(message: str) -> None:
    """Write a line that is not printed on the terminal."""
    state = _state
    if state is None or state.stopping:
        return
    data = (message.rstrip("\n") + "\n").encode("utf-8", errors="replace")
    with _LOCK:
        if _state is None or _state.stopping:
            return
        _write_all(state.log_fd, data)


def stop_run_log() -> None:
    """Restore the original terminal streams and finish the transcript."""
    global _state
    with _LOCK:
        state = _state
        if state is None or state.stopping:
            return
        state.stopping = True
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    for tee in state.tees:
        try:
            os.dup2(tee.saved_fd, tee.target_fd)
        except OSError:
            pass
    for tee in state.tees:
        tee.thread.join(timeout=2.0)
        for fd in (tee.saved_fd, tee.read_fd):
            try:
                os.close(fd)
            except OSError:
                pass
    try:
        os.close(state.log_fd)
    except OSError:
        pass
    with _LOCK:
        if _state is state:
            _state = None


def _new_path(folder: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = folder / f"{stamp}.log"
    if not candidate.exists():
        return candidate
    return folder / f"{stamp}-{os.getpid()}.log"


def _replace_latest(latest: Path, name: str) -> None:
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(name)
    except OSError:
        return


def _tee(target_fd: int, log_fd: int) -> _Tee:
    saved_fd = os.dup(target_fd)
    read_fd, slave_fd = pty.openpty()
    _keep_newlines(slave_fd)
    os.dup2(slave_fd, target_fd)
    os.close(slave_fd)
    thread = threading.Thread(
        target=_pump,
        args=(read_fd, saved_fd, log_fd),
        name=f"runlog-{target_fd}",
        daemon=True,
    )
    thread.start()
    return _Tee(target_fd=target_fd, saved_fd=saved_fd, read_fd=read_fd, thread=thread)


def _keep_newlines(slave_fd: int) -> None:
    """Leave newline conversion to the original terminal so the log stays plain text."""
    try:
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] = attrs[1] & ~termios.ONLCR
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    except termios.error:
        return


def _pump(read_fd: int, saved_fd: int, log_fd: int) -> None:
    while True:
        try:
            chunk = os.read(read_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        with _LOCK:
            _write_all(saved_fd, chunk)
            _write_all(log_fd, chunk)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            break
        view = view[written:]
