#!/usr/bin/env python3
"""Bounded resident JSONL product-process transport with replayable wire evidence."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class JsonlSessionError(RuntimeError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class SessionEvent:
    index: int
    received_monotonic_ns: int
    read_index: int
    raw_line: bytes
    value: dict[str, Any]

    def receipt(self) -> dict[str, Any]:
        return {
            "event_index": self.index,
            "received_monotonic_ns": self.received_monotonic_ns,
            "read_index": self.read_index,
            "raw_line_bytes": len(self.raw_line),
            "raw_line_sha256": _sha256(self.raw_line),
            "event": self.value.get("event"),
            "session_id": self.value.get("session_id"),
            "history_epoch": self.value.get("history_epoch"),
            "request_id": self.value.get("request_id"),
            "turn": self.value.get("turn"),
        }


@dataclass(frozen=True)
class SessionCase:
    started_monotonic_ns: int
    finished_monotonic_ns: int
    read_start: int
    read_end: int
    events: tuple[SessionEvent, ...]

    def jsonl_bytes(self, ready: SessionEvent | None = None) -> bytes:
        rows = ([ready] if ready is not None else []) + list(self.events)
        return b"".join(event.raw_line for event in rows)

    def receipt(self) -> dict[str, Any]:
        request_ids = list(
            dict.fromkeys(
                str(event.value["request_id"])
                for event in self.events
                if isinstance(event.value.get("request_id"), str)
            )
        )
        epochs = sorted(
            {
                int(event.value["history_epoch"])
                for event in self.events
                if isinstance(event.value.get("history_epoch"), int)
            }
        )
        return {
            "started_monotonic_ns": self.started_monotonic_ns,
            "finished_monotonic_ns": self.finished_monotonic_ns,
            "duration_sec": (self.finished_monotonic_ns - self.started_monotonic_ns) / 1e9,
            "wire_read_start": self.read_start,
            "wire_read_end": self.read_end,
            "request_ids": request_ids,
            "history_epochs": epochs,
            "events": [event.receipt() for event in self.events],
        }


class JsonlProductSession:
    def __init__(
        self,
        *,
        argv: list[str],
        stdout_path: Path,
        stderr_path: Path,
        timeout_sec: float,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        read_size: int = 4096,
    ) -> None:
        if not argv:
            raise JsonlSessionError("resident JSONL argv is empty")
        if timeout_sec <= 0:
            raise JsonlSessionError("resident JSONL timeout must be positive")
        if read_size <= 0:
            raise JsonlSessionError("resident JSONL read size must be positive")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.argv = list(argv)
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.timeout_sec = timeout_sec
        self.read_size = read_size
        self.started_monotonic_ns = time.monotonic_ns()
        self._stdout_handle = stdout_path.open("wb")
        self._stderr_handle = stderr_path.open("wb")
        self.proc = subprocess.Popen(
            self.argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=False,
        )
        self._condition = threading.Condition()
        self._events: list[SessionEvent] = []
        self._wire_reads: list[dict[str, Any]] = []
        self._cursor = 0
        self._stdout_error: BaseException | None = None
        self._stdout_eof = False
        self._stderr_eof = False
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.ready_event = self._wait_for(
                lambda event: event.value.get("event") == "ready",
                timeout_sec,
                "ready",
            )[-1]
        except Exception:
            self.terminate()
            raise

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        pending = bytearray()
        try:
            while True:
                chunk = os.read(self.proc.stdout.fileno(), self.read_size)
                if not chunk:
                    decoder.decode(b"", final=True)
                    if pending:
                        raise JsonlSessionError("resident JSONL stdout ended without newline")
                    break
                observed_ns = time.monotonic_ns()
                decoder.decode(chunk, final=False)
                buffered, _ = decoder.getstate()
                with self._condition:
                    read_index = len(self._wire_reads)
                    self._wire_reads.append(
                        {
                            "read_index": read_index,
                            "received_monotonic_ns": observed_ns,
                            "byte_count": len(chunk),
                            "sha256": _sha256(chunk),
                            "decoder_buffered_bytes": len(buffered),
                        }
                    )
                self._stdout_handle.write(chunk)
                self._stdout_handle.flush()
                pending.extend(chunk)
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    raw_line = bytes(pending[: newline + 1])
                    del pending[: newline + 1]
                    text = raw_line.decode("utf-8", "strict")
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError as error:
                        raise JsonlSessionError(
                            f"resident stdout line is not JSON: {_sha256(raw_line)}: {error}"
                        ) from error
                    if not isinstance(parsed, dict):
                        raise JsonlSessionError("resident stdout JSONL row must be an object")
                    with self._condition:
                        event = SessionEvent(
                            index=len(self._events),
                            received_monotonic_ns=time.monotonic_ns(),
                            read_index=read_index,
                            raw_line=raw_line,
                            value=parsed,
                        )
                        self._events.append(event)
                        self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._stdout_error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._stdout_eof = True
                self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        try:
            while True:
                chunk = os.read(self.proc.stderr.fileno(), 65536)
                if not chunk:
                    break
                self._stderr_handle.write(chunk)
                self._stderr_handle.flush()
        finally:
            with self._condition:
                self._stderr_eof = True
                self._condition.notify_all()

    def _next_event(self, deadline: float, expected: str) -> SessionEvent:
        with self._condition:
            while True:
                if self._cursor < len(self._events):
                    event = self._events[self._cursor]
                    self._cursor += 1
                    return event
                if self._stdout_error is not None:
                    raise JsonlSessionError(f"resident JSONL stdout failed: {self._stdout_error}")
                if self._stdout_eof:
                    raise JsonlSessionError(
                        f"resident JSONL stdout ended while waiting for {expected}; returncode={self.proc.poll()}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise JsonlSessionError(
                        f"resident JSONL timed out waiting for {expected}; pid={self.proc.pid}"
                    )
                self._condition.wait(remaining)

    def _wait_for(
        self,
        predicate: Callable[[SessionEvent], bool],
        timeout_sec: float,
        expected: str,
    ) -> list[SessionEvent]:
        deadline = time.monotonic() + timeout_sec
        observed: list[SessionEvent] = []
        while True:
            event = self._next_event(deadline, expected)
            observed.append(event)
            if predicate(event):
                return observed

    def send(self, line: str) -> None:
        if "\n" in line or "\r" in line:
            raise JsonlSessionError("resident input must be exactly one logical line")
        if self.proc.poll() is not None:
            raise JsonlSessionError(
                f"resident process exited before input; returncode={self.proc.returncode}"
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write(line.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def run_case(self, prompts: list[str], *, reset: bool = True) -> SessionCase:
        if not prompts:
            raise JsonlSessionError("resident case requires at least one prompt")
        started_ns = time.monotonic_ns()
        with self._condition:
            read_start = len(self._wire_reads)
        observed: list[SessionEvent] = []
        if reset:
            self.send("/clear")
            observed.extend(
                self._wait_for(
                    lambda event: event.value.get("event") == "history_reset",
                    self.timeout_sec,
                    "history_reset",
                )
            )
        for prompt in prompts:
            self.send(prompt)
            request_id: str | None = None

            def terminal(event: SessionEvent) -> bool:
                nonlocal request_id
                kind = event.value.get("event")
                if kind == "user":
                    candidate = event.value.get("request_id")
                    if not isinstance(candidate, str) or not candidate:
                        raise JsonlSessionError("resident user event lacks request_id")
                    if event.value.get("content") != prompt:
                        raise JsonlSessionError("resident user event content differs from input")
                    request_id = candidate
                if kind in {"assistant_delta", "assistant"}:
                    candidate = event.value.get("request_id")
                    if request_id is None or candidate != request_id:
                        raise JsonlSessionError("resident assistant event request binding drift")
                return kind == "assistant" and request_id is not None

            observed.extend(self._wait_for(terminal, self.timeout_sec, "assistant"))
        finished_ns = time.monotonic_ns()
        with self._condition:
            read_end = len(self._wire_reads)
        return SessionCase(
            started_monotonic_ns=started_ns,
            finished_monotonic_ns=finished_ns,
            read_start=read_start,
            read_end=read_end,
            events=tuple(observed),
        )

    def wire_receipt(self) -> dict[str, Any]:
        with self._condition:
            return {
                "read_size": self.read_size,
                "read_count": len(self._wire_reads),
                "reads": list(self._wire_reads),
                "event_count": len(self._events),
                "events": [event.receipt() for event in self._events],
            }

    def stop(self) -> SessionEvent:
        if self.proc.poll() is None:
            self.send("/bye")
            exit_event = self._wait_for(
                lambda event: event.value.get("event") == "exit",
                self.timeout_sec,
                "exit",
            )[-1]
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=self.timeout_sec)
            except subprocess.TimeoutExpired as error:
                self.proc.kill()
                self.proc.wait(timeout=10)
                raise JsonlSessionError("resident process did not exit after /bye") from error
        else:
            raise JsonlSessionError(
                f"resident process exited before controlled stop; returncode={self.proc.returncode}"
            )
        self._stdout_thread.join(timeout=5)
        self._stderr_thread.join(timeout=5)
        self._stdout_handle.close()
        self._stderr_handle.close()
        if self.proc.returncode != 0:
            raise JsonlSessionError(f"resident process exit code is {self.proc.returncode}")
        return exit_event

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for handle in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if handle is not None and not handle.closed:
                handle.close()
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        if not self._stdout_handle.closed:
            self._stdout_handle.close()
        if not self._stderr_handle.closed:
            self._stderr_handle.close()


def _self_test() -> int:
    fake = r'''
import json, sys
session = "fixture-session"
epoch = 0
turn = 0
def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
emit({"schema_version":2,"event":"ready","session_id":session,"history_epoch":epoch})
for raw in sys.stdin:
    line = raw.rstrip("\r\n")
    if line == "/clear":
        epoch += 1
        turn = 0
        emit({"schema_version":2,"event":"history_reset","session_id":session,"history_epoch":epoch,"turn":0})
        continue
    if line == "/bye":
        emit({"schema_version":2,"event":"exit","session_id":session,"history_epoch":epoch,"reason":"bye"})
        break
    request = f"request-{epoch}-{turn}"
    emit({"schema_version":2,"event":"user","session_id":session,"history_epoch":epoch,"request_id":request,"turn":turn,"content":line})
    emit({"schema_version":2,"event":"assistant_delta","session_id":session,"history_epoch":epoch,"request_id":request,"turn":turn,"index":0,"raw_text_delta":"hello"})
    emit({"schema_version":2,"event":"assistant_delta","session_id":session,"history_epoch":epoch,"request_id":request,"turn":turn,"index":1,"raw_text_delta":"\U0001f642"})
    emit({"schema_version":2,"event":"assistant","session_id":session,"history_epoch":epoch,"request_id":request,"turn":turn,"content":"hello\U0001f642","reasoning":None,"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"n_tokens":2,"chunk_count":2,"finish_reason":"stop"})
    turn += 1
'''
    with tempfile.TemporaryDirectory(prefix="ferrum-jsonl-session-") as raw_tmp:
        tmp = Path(raw_tmp)
        session = JsonlProductSession(
            argv=[sys.executable, "-u", "-c", fake],
            stdout_path=tmp / "stdout.jsonl",
            stderr_path=tmp / "stderr.log",
            timeout_sec=5,
            read_size=7,
        )
        first = session.run_case(["first", "second"])
        second = session.run_case(["third"])
        pid = session.proc.pid
        exit_event = session.stop()
        if session.proc.pid != pid or session.proc.returncode != 0:
            raise JsonlSessionError("self-test did not reuse one controlled process")
        if exit_event.value.get("reason") != "bye":
            raise JsonlSessionError("self-test exit event drift")
        if len([event for event in first.events if event.value.get("event") == "assistant"]) != 2:
            raise JsonlSessionError("self-test first case assistant count drift")
        if len([event for event in second.events if event.value.get("event") == "assistant"]) != 1:
            raise JsonlSessionError("self-test second case assistant count drift")
        if first.receipt()["history_epochs"] != [1] or second.receipt()["history_epochs"] != [2]:
            raise JsonlSessionError("self-test reset epochs drift")
        if b"\xf0\x9f\x99\x82" not in (tmp / "stdout.jsonl").read_bytes():
            raise JsonlSessionError("self-test UTF-8 wire bytes were not preserved")
        if session.wire_receipt()["read_count"] <= 2:
            raise JsonlSessionError("self-test did not preserve incremental wire reads")
    print("FERRUM JSONL PRODUCT SESSION SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test is required when running this module directly")
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
