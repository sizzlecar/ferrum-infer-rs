#!/usr/bin/env python3
"""Run the W3 Qwen3.5 c32 diagnostic on a retained Vast instance.

This local orchestrator intentionally does not benchmark vLLM. It starts or
uses one exact 1x4090 Vast instance, uploads the focused remote diagnostic
script, runs it inside tmux, copies back the produced artifact, and stops the
instance unless --keep-running is set.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_INSTANCE_ID = 42216671
DEFAULT_SHA = "8a5f2819110eb9bb876f582c5b41f2f2fe8a1bbe"
DEFAULT_TAG = "mixed_prefill_headroom"
DEFAULT_THROUGHPUT_FLOOR = "600.0"
DEFAULT_MAX_KV_ADMISSION_FAILED = "13"
DEFAULT_MAX_CAPACITY_DEFERRED = "32"
DEFAULT_MIN_MIXED_ITERATIONS = "64"
DEFAULT_MAX_P95_ITL_MS = "25.0"
DEFAULT_REMOTE_SCRIPT = "/workspace/w3_qwen35_c32_diagnostic.sh"
DEFAULT_REMOTE_META_ROOT = "/workspace/artifacts"
DEFAULT_LOCAL_OUT = (
    "docs/goals/model-coverage-2026-06-12/artifacts"
)
DIAG_SCRIPT = "scripts/release/w3_qwen35_c32_diagnostic.sh"


class DiagnosticError(RuntimeError):
    pass


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> CommandResult:
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    result = CommandResult(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if check and proc.returncode != 0:
        raise DiagnosticError(
            f"command failed rc={proc.returncode}: {shlex.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return result


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def vast_request(
    method: str,
    path: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    *,
    attempts: int = 4,
    retry_sleep_sec: float = 5.0,
) -> Any:
    url = f"https://console.vast.ai/api/v0{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code < 500 or attempt >= attempts:
                raise DiagnosticError(
                    f"Vast HTTP {exc.code} for {method} {path}: {detail[:1000]}"
                ) from exc
            last_error = exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= attempts:
                raise DiagnosticError(
                    f"Vast request failed for {method} {path} after {attempts} attempts: {exc}"
                ) from exc
        print(
            json.dumps(
                {
                    "vast_api_retry": {
                        "attempt": attempt,
                        "method": method,
                        "path": path,
                        "reason": str(last_error),
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        time.sleep(retry_sleep_sec)
    raise DiagnosticError(f"Vast request failed for {method} {path}: {last_error}")


def extract_instance(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "instances" in payload:
        value = payload["instances"]
        if isinstance(value, list):
            if not value:
                raise DiagnosticError("Vast response contained no instances")
            return dict(value[0])
        if isinstance(value, dict):
            return dict(value)
    if isinstance(payload, list):
        if not payload:
            raise DiagnosticError("Vast response contained no instances")
        return dict(payload[0])
    if isinstance(payload, dict):
        return dict(payload)
    raise DiagnosticError(f"unexpected Vast response shape: {type(payload).__name__}")


def instance_summary(instance: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "cur_state",
        "actual_status",
        "intended_status",
        "gpu_name",
        "num_gpus",
        "dph_total",
        "cuda_max_good",
        "ssh_host",
        "ssh_port",
        "disk_space",
        "label",
    ]
    return {key: instance.get(key) for key in keys}


def ssh_target(instance: dict[str, Any], user: str) -> tuple[str, int, str]:
    host = str(instance.get("ssh_host") or "")
    port = int(instance.get("ssh_port") or 0)
    if not host or port <= 0:
        raise DiagnosticError(f"instance has no SSH endpoint: {instance_summary(instance)}")
    return host, port, f"{user}@{host}"


def ssh_base(host: str, port: int, identity: str | None) -> list[str]:
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]
    if identity:
        cmd.extend(["-i", identity])
    return cmd


def scp_base(port: int, identity: str | None) -> list[str]:
    cmd = [
        "scp",
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if identity:
        cmd.extend(["-i", identity])
    return cmd


def rsync_ssh(port: int, identity: str | None) -> str:
    parts = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]
    if identity:
        parts.extend(["-i", identity])
    return " ".join(shlex.quote(part) for part in parts)


def wait_for_instance(
    api_key: str,
    instance_id: int,
    *,
    want_running: bool,
    timeout_sec: int,
    poll_sec: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = vast_request("GET", f"/instances/{instance_id}/", api_key)
        last = extract_instance(payload)
        summary = instance_summary(last)
        print(json.dumps({"poll": summary}, sort_keys=True), flush=True)
        if want_running:
            if summary.get("cur_state") == "running" and summary.get("actual_status") == "running":
                return last
        else:
            if summary.get("actual_status") == "exited":
                return last
        time.sleep(poll_sec)
    raise DiagnosticError(f"timed out waiting for instance state; last={instance_summary(last or {})}")


def parse_artifact_path(text: str) -> str:
    pattern = re.compile(r"FERRUM W3 QWEN35 C32 DIAG (?:KEEP|REJECT):\s+(\S+)")
    matches = pattern.findall(text)
    if not matches:
        raise DiagnosticError("could not find diagnostic KEEP/REJECT artifact path in tmux log")
    return matches[-1]


def make_remote_wrapper(args: argparse.Namespace, remote_script: str, remote_meta: str) -> str:
    remote_log_dir = f"{remote_meta}/logs"
    cmd = [
        remote_script,
        "--sha",
        args.sha,
        "--tag",
        args.tag,
        "--throughput-floor",
        args.throughput_floor,
        "--max-kv-admission-failed",
        args.max_kv_admission_failed,
        "--max-capacity-deferred",
        args.max_capacity_deferred,
        "--min-mixed-iterations",
        args.min_mixed_iterations,
        "--max-p95-itl-ms",
        args.max_p95_itl_ms,
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            f"mkdir -p {shlex.quote(remote_log_dir)}",
            f"chmod +x {shlex.quote(remote_script)}",
            "set +e",
            f"{shlex.join(cmd)} > {shlex.quote(remote_log_dir + '/tmux.stdout.log')} "
            f"2> {shlex.quote(remote_log_dir + '/tmux.stderr.log')}",
            "rc=$?",
            "set -e",
            f"echo \"$rc\" > {shlex.quote(remote_meta + '/exit_code')}",
            f"grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' "
            f"{shlex.quote(remote_log_dir + '/tmux.stdout.log')} | tail -n 1 "
            f"| sed -E 's/^.*: //' > {shlex.quote(remote_meta + '/artifact_path')} || true",
            "exit \"$rc\"",
            "",
        ]
    )


def run_self_test() -> None:
    direct = {"id": 1, "cur_state": "running", "actual_status": "running"}
    wrapped_dict = {"instances": direct}
    wrapped_list = {"instances": [direct]}
    assert extract_instance(direct)["id"] == 1
    assert extract_instance(wrapped_dict)["id"] == 1
    assert extract_instance(wrapped_list)["id"] == 1
    keep = "log\nFERRUM W3 QWEN35 C32 DIAG KEEP: /workspace/artifacts/a\n"
    reject = "FERRUM W3 QWEN35 C32 DIAG REJECT: /workspace/artifacts/b\n"
    assert parse_artifact_path(keep) == "/workspace/artifacts/a"
    assert parse_artifact_path(keep + reject) == "/workspace/artifacts/b"
    wrapper = make_remote_wrapper(
        argparse.Namespace(
            sha=DEFAULT_SHA,
            tag=DEFAULT_TAG,
            throughput_floor=DEFAULT_THROUGHPUT_FLOOR,
            max_kv_admission_failed=DEFAULT_MAX_KV_ADMISSION_FAILED,
            max_capacity_deferred=DEFAULT_MAX_CAPACITY_DEFERRED,
            min_mixed_iterations=DEFAULT_MIN_MIXED_ITERATIONS,
            max_p95_itl_ms=DEFAULT_MAX_P95_ITL_MS,
        ),
        "/workspace/diag.sh",
        "/workspace/artifacts/meta",
    )
    assert "--sha" in wrapper and DEFAULT_SHA in wrapper
    assert "artifact_path" in wrapper
    print("W3 QWEN35 VAST C32 DIAGNOSTIC ORCHESTRATOR SELFTEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", type=int, default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--sha", default=DEFAULT_SHA)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--throughput-floor", default=DEFAULT_THROUGHPUT_FLOOR)
    parser.add_argument("--max-kv-admission-failed", default=DEFAULT_MAX_KV_ADMISSION_FAILED)
    parser.add_argument("--max-capacity-deferred", default=DEFAULT_MAX_CAPACITY_DEFERRED)
    parser.add_argument("--min-mixed-iterations", default=DEFAULT_MIN_MIXED_ITERATIONS)
    parser.add_argument("--max-p95-itl-ms", default=DEFAULT_MAX_P95_ITL_MS)
    parser.add_argument("--local-out", default=DEFAULT_LOCAL_OUT)
    parser.add_argument("--remote-script", default=DEFAULT_REMOTE_SCRIPT)
    parser.add_argument("--remote-meta-root", default=DEFAULT_REMOTE_META_ROOT)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    parser.add_argument("--start-timeout-sec", type=int, default=900)
    parser.add_argument("--run-timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--session-name", default="")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    local_diag_script = repo_root / DIAG_SCRIPT
    if not local_diag_script.exists():
        raise DiagnosticError(f"missing local diagnostic script: {local_diag_script}")

    plan = {
        "lane": "W3 Qwen35 c32 mixed-prefill-headroom diagnostic",
        "instance_id": args.instance_id,
        "target_sha": args.sha,
        "correctness_gate": [
            "remote clean SHA",
            "cargo check",
            "CUDA release build",
            "ferrum run smoke",
            "ferrum serve models/chat smoke",
        ],
        "performance_command": "ferrum bench-serve c32 --fail-on-error --seed 9271 --n-repeats 1",
        "keep_thresholds": {
            "output_throughput_floor_tps": args.throughput_floor,
            "max_kv_admission_failed": args.max_kv_admission_failed,
            "max_capacity_deferred": args.max_capacity_deferred,
            "min_mixed_iterations": args.min_mixed_iterations,
            "max_p95_itl_ms": args.max_p95_itl_ms,
        },
        "no_live_vllm": True,
        "stop_condition": "KEEP, REJECT, SSH/CUDA unavailable, timeout, or script failure",
    }
    print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
    if args.plan_only:
        return 0

    load_dotenv(repo_root / ".env.local")
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise DiagnosticError("VAST_API_KEY is missing from environment or .env.local")

    run_stamp = int(time.time())
    session = args.session_name or f"w3_qwen35_c32_{args.sha[:8]}_{run_stamp}"
    remote_meta = f"{args.remote_meta_root}/w3_qwen35_c32_orchestrator_{args.sha[:8]}_{run_stamp}"
    remote_wrapper = f"{remote_meta}/run_remote_diagnostic.sh"
    started = False
    instance = extract_instance(vast_request("GET", f"/instances/{args.instance_id}/", api_key))
    print(json.dumps({"instance_before": instance_summary(instance)}, sort_keys=True))
    try:
        if instance.get("cur_state") != "running" or instance.get("actual_status") != "running":
            response = vast_request("PUT", f"/instances/{args.instance_id}/", api_key, {"state": "running"})
            print(json.dumps({"start_response": response}, sort_keys=True))
            instance = wait_for_instance(
                api_key,
                args.instance_id,
                want_running=True,
                timeout_sec=args.start_timeout_sec,
                poll_sec=args.poll_sec,
            )
            started = True

        host, port, target = ssh_target(instance, args.ssh_user)
        identity = args.ssh_key if args.ssh_key and pathlib.Path(args.ssh_key).exists() else None
        ssh = ssh_base(host, port, identity)
        scp = scp_base(port, identity)

        run_command(ssh + [target, "nvidia-smi && nvcc --version && df -h /workspace"], timeout=60)
        run_command(ssh + [target, f"mkdir -p {shlex.quote(remote_meta)}"], timeout=60)
        run_command(scp + [str(local_diag_script), f"{target}:{args.remote_script}"], timeout=120)

        wrapper_content = make_remote_wrapper(args, args.remote_script, remote_meta)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(wrapper_content)
            tmp_path = tmp.name
        try:
            run_command(scp + [tmp_path, f"{target}:{remote_wrapper}"], timeout=120)
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

        run_command(
            ssh
            + [
                target,
                f"chmod +x {shlex.quote(remote_wrapper)} && "
                f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(remote_wrapper)}",
            ],
            timeout=60,
        )

        deadline = time.time() + args.run_timeout_sec
        exit_code: str | None = None
        while time.time() < deadline:
            check = run_command(
                ssh + [target, f"test -f {shlex.quote(remote_meta + '/exit_code')} && cat {shlex.quote(remote_meta + '/exit_code')} || true"],
                check=False,
                timeout=60,
            )
            if check.stdout.strip():
                exit_code = check.stdout.strip().splitlines()[-1]
                break
            time.sleep(args.poll_sec)
        if exit_code is None:
            raise DiagnosticError(f"remote diagnostic timed out after {args.run_timeout_sec}s")

        path_result = run_command(
            ssh + [target, f"cat {shlex.quote(remote_meta + '/artifact_path')} 2>/dev/null || true"],
            check=False,
            timeout=60,
        )
        remote_artifact = path_result.stdout.strip().splitlines()[-1] if path_result.stdout.strip() else ""

        stdout_result = run_command(
            ssh + [target, f"cat {shlex.quote(remote_meta + '/logs/tmux.stdout.log')} 2>/dev/null || true"],
            check=False,
            timeout=60,
        )
        if not remote_artifact:
            remote_artifact = parse_artifact_path(stdout_result.stdout)

        local_out = pathlib.Path(args.local_out).resolve()
        local_out.mkdir(parents=True, exist_ok=True)
        local_artifact = local_out / pathlib.PurePosixPath(remote_artifact).name
        run_command(
            [
                "rsync",
                "-az",
                "-e",
                rsync_ssh(port, identity),
                f"{target}:{remote_artifact.rstrip('/')}/",
                f"{local_artifact}/",
            ],
            timeout=600,
        )
        local_meta = local_out / pathlib.PurePosixPath(remote_meta).name
        run_command(
            [
                "rsync",
                "-az",
                "-e",
                rsync_ssh(port, identity),
                f"{target}:{remote_meta.rstrip('/')}/",
                f"{local_meta}/",
            ],
            timeout=300,
        )

        print(
            json.dumps(
                {
                    "remote_exit_code": exit_code,
                    "remote_artifact": remote_artifact,
                    "local_artifact": str(local_artifact),
                    "local_orchestrator_meta": str(local_meta),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return int(exit_code)
    finally:
        if not args.keep_running:
            try:
                vast_request("PUT", f"/instances/{args.instance_id}/", api_key, {"state": "stopped"})
                stopped = wait_for_instance(
                    api_key,
                    args.instance_id,
                    want_running=False,
                    timeout_sec=300,
                    poll_sec=args.poll_sec,
                )
                print(json.dumps({"instance_after_stop": instance_summary(stopped)}, sort_keys=True))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to stop instance cleanly: {exc}", file=sys.stderr)
        elif started:
            print("keep-running set; instance was started and left running", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
