#!/usr/bin/env python3
"""Keep M3 shell wrappers on the shared runner path.

The M3 wrappers are allowed to generate runner config and call
scripts/m3_ab_runner.py, or to delegate to an explicitly allowlisted wrapper.
They must not reintroduce copied server launch, health, bench, or cleanup
logic.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_RUNNER = "scripts/m3_ab_runner.py"
ALLOWED_DELEGATE_WRAPPERS = {
    "scripts/m3_fa2_source_allcells_ab.sh": ("scripts/m3_fa2_direct_ffi_ab.sh",),
}
RETIRED_WRAPPER_MARKERS = {
    "scripts/m3_fa2_source_allcells_ab.sh": "FERRUM_FA2_SOURCE is rejected",
}

FORBIDDEN_PATTERNS = (
    (
        "direct ferrum serve launch",
        re.compile(
            r"(?m)(?:^|[;&|]\s*)"
            r"(?:(?:\"?\$BIN\"?)|\$\{BIN[^}]*\}|"
            r"(?:\./)?target/(?:debug|release)/ferrum|ferrum)\s+serve\b"
        ),
    ),
    ("manual bench-serve invocation", re.compile(r"\bbench-serve\b")),
    ("manual server pid lifecycle", re.compile(r"\bSERVER_PID\b")),
    ("manual health wait loop", re.compile(r"\bwait_health\b")),
    ("manual OpenAI chat curl gate", re.compile(r"/v1/chat/completions")),
    ("manual curl invocation", re.compile(r"(?m)(?:^|\s)curl\s")),
    ("manual cleanup function", re.compile(r"(?m)^\s*cleanup\s*\(\s*\)")),
    ("manual trap handler", re.compile(r"(?m)^\s*trap\b")),
    ("manual process kill", re.compile(r"(?m)^\s*(?:kill|killall|pkill)\b")),
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _active_shell_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def discover_m3_wrappers(root: Path) -> list[Path]:
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return []
    return sorted(scripts_dir.glob("m3_*.sh"))


def check_wrapper(path: Path, root: Path) -> list[str]:
    rel = _relative(path, root)
    text = path.read_text()
    active = _active_shell_text(text)
    errors: list[str] = []

    retired_marker = RETIRED_WRAPPER_MARKERS.get(rel)
    if retired_marker and retired_marker in active:
        return errors

    for description, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(active):
            errors.append(f"{rel}: contains {description}")

    if DIRECT_RUNNER in active:
        return errors

    allowed_delegates = ALLOWED_DELEGATE_WRAPPERS.get(rel, ())
    if allowed_delegates and any(delegate in active for delegate in allowed_delegates):
        return errors

    if allowed_delegates:
        expected = " or ".join(allowed_delegates)
        errors.append(f"{rel}: expected delegated wrapper call to {expected}")
    else:
        errors.append(
            f"{rel}: does not call {DIRECT_RUNNER}; add a runner config or "
            "explicitly allowlist a delegate wrapper"
        )
    return errors


def check_root(root: Path) -> list[str]:
    errors: list[str] = []
    wrappers = discover_m3_wrappers(root)
    if not wrappers:
        errors.append(f"{root}: no scripts/m3_*.sh wrappers found")
        return errors
    for wrapper in wrappers:
        errors.extend(check_wrapper(wrapper, root))
    return errors


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(
            root / "scripts/m3_fa_layout_varlen_ab.sh",
            """#!/usr/bin/env bash
set -euo pipefail
# Commented examples may mention curl or ferrum serve without becoming logic.
python3 scripts/m3_ab_runner.py --config "$CONFIG"
""",
        )
        _write(
            root / "scripts/m3_fa2_source_allcells_ab.sh",
            """#!/usr/bin/env bash
set -euo pipefail
echo "m3_fa2_source_allcells_ab.sh is retired: FERRUM_FA2_SOURCE is rejected until a source-owned FA2 kernel has release evidence." >&2
exit 1
""",
        )
        errors = check_root(root)
        if errors:
            raise AssertionError(f"expected clean wrapper set, got {errors}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(
            root / "scripts/m3_bad_ab.sh",
            """#!/usr/bin/env bash
set -euo pipefail
"$BIN" serve --model "$MODEL_DIR" &
SERVER_PID=$!
wait_health
bench-serve --url http://127.0.0.1:8080
kill "$SERVER_PID"
""",
        )
        errors = check_root(root)
        if len(errors) < 5:
            raise AssertionError(f"expected multiple lifecycle errors, got {errors}")
        expected_fragments = (
            "direct ferrum serve launch",
            "manual bench-serve invocation",
            "manual server pid lifecycle",
            "manual health wait loop",
            "manual process kill",
            f"does not call {DIRECT_RUNNER}",
        )
        for fragment in expected_fragments:
            if not any(fragment in error for error in errors):
                raise AssertionError(f"missing self-test error fragment: {fragment}")

    print("check_m3_runner_wrappers self-test ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    root = args.root.resolve()
    errors = check_root(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"check_m3_runner_wrappers ok: {len(discover_m3_wrappers(root))} wrappers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
