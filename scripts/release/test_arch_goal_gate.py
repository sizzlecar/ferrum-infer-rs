#!/usr/bin/env python3
"""Final validator and baseline auditor for the test-architecture goal.

Goal doc: docs/goals/test-architecture-2026-06-10/GOAL.md

Modes:
  --self-test          validate the gate's own logic against synthetic fixtures
  --baseline           recompute coupling/coverage counts from the working tree
                       and write docs/goals/test-architecture-2026-06-10/baseline.json
  --validate OUT_DIR   final goal validation (Gate A/B/C). Prints the goal PASS
                       line only when every check holds.

Required final PASS line:
  TEST_ARCH GOAL PASS: <out_dir>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PASS_PREFIX = "TEST_ARCH GOAL PASS"
SELFTEST_PASS = "TEST_ARCH GOAL SELFTEST PASS"
BASELINE_PREFIX = "TEST_ARCH BASELINE WRITTEN"

GOAL_DIR = Path("docs/goals/test-architecture-2026-06-10")
CONFORMANCE_MANIFEST = GOAL_DIR / "conformance_ops.json"
HISTORICAL_BUGS = GOAL_DIR / "historical_bugs.json"
BASELINE_PATH = GOAL_DIR / "baseline.json"
ALLOWLIST_PATH = Path("scripts/release/test_arch_allowlist.json")
OP_DIFF_DIR = Path("crates/ferrum-testkit/src/op_diff")
MODELS_MANIFEST = Path("scripts/release/models_manifest.json")

MODELS_MAIN_PATH = Path("crates/ferrum-models/src/models")
ENGINE_MAIN_PATH = Path("crates/ferrum-engine/src")
SCOPE_ROOTS = {"models": MODELS_MAIN_PATH, "engine": ENGINE_MAIN_PATH}

# Gate A patterns. A finding outside the allowlist fails final validation.
PATTERNS = {
    # A1: direct env reads in shared LLM main-path code
    "env_var": re.compile(r"env::var"),
    # A2: backend-conditional compilation in shared LLM main-path code
    "cfg_branch": re.compile(
        r'cfg\(feature = "(?:cuda|metal)"\)|target_os = "(?:macos|ios)"'
    ),
    # A3: capability behavior-branches in hot paths (must become trait defaults)
    "supports_branch": re.compile(r"B::supports_[a-z_0-9]+\(\)"),
    # Stage-3 hard constraint: no process-frozen runtime config accessors
    "once_lock": re.compile(r"OnceLock<"),
}
PATTERN_SCOPES = {
    "env_var": ("models", "engine"),
    "cfg_branch": ("models", "engine"),
    "supports_branch": ("models",),
    "once_lock": ("models", "engine"),
}

# Gate C budgets (seconds). May be tightened, never loosened (GOAL.md).
LANE_BUDGETS_SECONDS = {
    "l0": 600,
    "l1_metal": 900,
    "l1_cuda_warm": 1200,
    "l1_cuda_cold": 3600,
}
# Gate C5 stability: lane -> (required_runs, required_green)
STABILITY_REQUIRED = {"l0": (10, 10), "l1_metal": (10, 10), "l1_cuda": (3, 3)}

# Gate B1 scenario tests (stage 1 implements these exact names; the L0 test
# listing captured into <out_dir>/l0_tests.txt must contain every entry).
REQUIRED_SCENARIO_TESTS = [
    "tiny_stack_multi_turn_five_rounds",
    "tiny_stack_eos_terminates",
    "tiny_stack_stop_sequence_composite_token",
    "tiny_stack_stream_chunk_contract",
    "tiny_stack_concurrent_sessions_isolated",
    "tiny_stack_cancel_mid_stream",
    "tiny_stack_repetition_runaway_guard",
    "tiny_stack_guided_tool_constraint",
    "tiny_stack_openai_wire_contract",
    "tiny_stack_kv_capacity_boundary",
]

CPU_KILL_RATIO_MIN = 0.8


class ValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Scanning


def iter_rs_files(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.rs")):
        rel = path.as_posix()
        if "/tests/" in rel or rel.endswith("_test.rs"):
            continue
        yield path


def scan_findings(repo_root: Path) -> list[dict[str, Any]]:
    """Line-level raw scan. Test files are skipped; #[cfg(test)] blocks are
    not parsed out — keep audited code free of these patterns instead."""
    findings: list[dict[str, Any]] = []
    for scope, rel_root in SCOPE_ROOTS.items():
        for path in iter_rs_files(repo_root / rel_root):
            rel = path.relative_to(repo_root).as_posix()
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue
                for name, pattern in PATTERNS.items():
                    if scope not in PATTERN_SCOPES[name]:
                        continue
                    if pattern.search(line):
                        findings.append(
                            {
                                "pattern": name,
                                "scope": scope,
                                "file": rel,
                                "line": lineno,
                                "text": stripped[:160],
                            }
                        )
    return findings


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in PATTERNS:
        rows = [f for f in findings if f["pattern"] == name]
        summary[name] = {
            "total": len(rows),
            "by_scope": {
                scope: sum(1 for f in rows if f["scope"] == scope)
                for scope in SCOPE_ROOTS
                if scope in PATTERN_SCOPES[name]
            },
            "files": sorted({f["file"] for f in rows}),
        }
    return summary


# ---------------------------------------------------------------------------
# Manifests


def load_json(path: Path) -> Any:
    if not path.exists():
        raise ValidationError(f"missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid json in {path}: {exc}") from exc


def validate_conformance(
    manifest: Any, repo_root: Path, require_full: bool
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValidationError("conformance manifest: schema_version must be 1")
    ops = manifest.get("ops")
    if not isinstance(ops, list) or not ops:
        raise ValidationError("conformance manifest: ops must be a non-empty list")
    seen: set[str] = set()
    covered: list[str] = []
    for op in ops:
        op_id = op.get("id")
        if not isinstance(op_id, str) or not re.fullmatch(r"[a-z0-9_]+", op_id):
            raise ValidationError(f"conformance manifest: bad op id {op_id!r}")
        if op_id in seen:
            raise ValidationError(f"conformance manifest: duplicate op id {op_id}")
        seen.add(op_id)
        backends = op.get("backends")
        if not isinstance(backends, dict) or backends.get("cpu") != "reference":
            raise ValidationError(
                f"conformance manifest: op {op_id} must declare cpu as reference"
            )
        if not isinstance(op.get("methods"), list) or not op["methods"]:
            raise ValidationError(f"conformance manifest: op {op_id} missing methods")
        tol = op.get("tolerance")
        if not isinstance(tol, dict) or "rel" not in tol or "abs" not in tol:
            raise ValidationError(f"conformance manifest: op {op_id} missing tolerance")
        if op.get("covered_today"):
            module = repo_root / OP_DIFF_DIR / f"{op_id}.rs"
            if not module.exists():
                raise ValidationError(
                    f"conformance manifest: op {op_id} marked covered but"
                    f" {module.relative_to(repo_root)} does not exist"
                )
            covered.append(op_id)
    if require_full and len(covered) != len(ops):
        missing = sorted(seen - set(covered))
        raise ValidationError(
            f"conformance coverage incomplete: {len(covered)}/{len(ops)} ops covered,"
            f" missing {missing}"
        )
    return {"total_ops": len(ops), "covered_ops": len(covered), "covered_ids": covered}


def validate_bugs(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValidationError("historical bugs: schema_version must be 1")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("historical bugs: entries must be a non-empty list")
    seen: set[str] = set()
    cpu = cuda = 0
    for entry in entries:
        bug_id = entry.get("id")
        if not isinstance(bug_id, str) or not re.fullmatch(r"hb-\d{2}", bug_id):
            raise ValidationError(f"historical bugs: bad id {bug_id!r}")
        if bug_id in seen:
            raise ValidationError(f"historical bugs: duplicate id {bug_id}")
        seen.add(bug_id)
        kind = entry.get("kind")
        if kind not in ("revert-fix", "verify-live"):
            raise ValidationError(f"historical bugs: {bug_id} bad kind {kind!r}")
        reachable = entry.get("reachable")
        if reachable not in ("cpu", "cuda"):
            raise ValidationError(f"historical bugs: {bug_id} bad reachable")
        for sha in entry.get("fix_commits", []):
            if not re.fullmatch(r"[0-9a-f]{8,40}", sha):
                raise ValidationError(f"historical bugs: {bug_id} bad sha {sha!r}")
        status = entry.get("patch_status")
        if kind == "revert-fix" and status not in ("pending", "ready"):
            raise ValidationError(f"historical bugs: {bug_id} bad patch_status")
        if kind == "verify-live" and status != "n/a-live":
            raise ValidationError(
                f"historical bugs: {bug_id} verify-live must be n/a-live"
            )
        if reachable == "cpu":
            cpu += 1
        else:
            cuda += 1
    return {"total": len(entries), "cpu_reachable": cpu, "cuda_only": cuda}


def load_allowlist(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / ALLOWLIST_PATH
    if not path.exists():
        return []
    data = load_json(path)
    entries = data.get("entries", []) if isinstance(data, dict) else None
    if entries is None:
        raise ValidationError("allowlist: top-level entries list required")
    for entry in entries:
        if not entry.get("path") or entry.get("pattern") not in PATTERNS:
            raise ValidationError(f"allowlist: bad entry {entry!r}")
        if not entry.get("reason") or not entry.get("review_condition"):
            raise ValidationError(
                f"allowlist: entry {entry.get('path')} needs reason + review_condition"
            )
    return entries


def finding_allowed(finding: dict[str, Any], allowlist: list[dict[str, Any]]) -> bool:
    return any(
        finding["pattern"] == entry["pattern"]
        and finding["file"].startswith(entry["path"])
        for entry in allowlist
    )


# ---------------------------------------------------------------------------
# Git helpers


def git_output(repo_root: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repo_state(repo_root: Path) -> dict[str, Any]:
    return {
        "sha": git_output(repo_root, ["rev-parse", "HEAD"]),
        "dirty": bool(git_output(repo_root, ["status", "--porcelain"])),
    }


# ---------------------------------------------------------------------------
# Baseline mode


def run_baseline(repo_root: Path, state: dict[str, Any] | None = None) -> Path:
    findings = scan_findings(repo_root)
    conformance = validate_conformance(
        load_json(repo_root / CONFORMANCE_MANIFEST), repo_root, require_full=False
    )
    bugs = validate_bugs(load_json(repo_root / HISTORICAL_BUGS))
    allowlist = load_allowlist(repo_root)
    state = state or repo_state(repo_root)
    baseline = {
        "schema_version": 1,
        "generated_by": "scripts/release/test_arch_goal_gate.py --baseline",
        "base_sha": state["sha"],
        "dirty_worktree": state["dirty"],
        "scan_method": (
            "line-level regex over non-test .rs files under"
            " crates/ferrum-models/src/models and crates/ferrum-engine/src;"
            " comment-only lines skipped; #[cfg(test)] blocks not excluded"
        ),
        "counts": summarize_findings(findings),
        "allowlist_entries": len(allowlist),
        "conformance": conformance,
        "historical_bugs": bugs,
        "lane_budgets_seconds": LANE_BUDGETS_SECONDS,
        "required_scenario_tests": REQUIRED_SCENARIO_TESTS,
    }
    out_path = repo_root / BASELINE_PATH
    out_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Final validation (Gate A/B/C)


def check_gate_a(repo_root: Path, failures: list[str]) -> None:
    allowlist = load_allowlist(repo_root)
    leftovers = [
        f for f in scan_findings(repo_root) if not finding_allowed(f, allowlist)
    ]
    for f in leftovers[:20]:
        failures.append(
            f"gate A: {f['pattern']} at {f['file']}:{f['line']} not allowlisted"
        )
    if len(leftovers) > 20:
        failures.append(f"gate A: ... and {len(leftovers) - 20} more findings")
    try:
        validate_conformance(
            load_json(repo_root / CONFORMANCE_MANIFEST), repo_root, require_full=True
        )
    except ValidationError as exc:
        failures.append(f"gate A: {exc}")


def check_gate_b(repo_root: Path, out_dir: Path, failures: list[str]) -> None:
    try:
        bugs_data = load_json(repo_root / HISTORICAL_BUGS)
        validate_bugs(bugs_data)
    except ValidationError as exc:
        failures.append(f"gate B: {exc}")
        return
    entries = {e["id"]: e for e in bugs_data["entries"]}

    for entry in entries.values():
        if entry["kind"] != "revert-fix":
            continue
        if entry.get("patch_status") != "ready":
            failures.append(f"gate B: {entry['id']} repro patch not ready")
            continue
        patch = entry.get("repro_patch")
        if not patch or not (repo_root / GOAL_DIR / patch).exists():
            failures.append(f"gate B: {entry['id']} repro patch file missing")

    try:
        killrate = load_json(out_dir / "killrate.json")
    except ValidationError as exc:
        failures.append(f"gate B: {exc}")
        killrate = None
    if killrate is not None:
        rows = {r.get("id"): r for r in killrate.get("entries", [])}
        missing = sorted(set(entries) - set(rows))
        if missing:
            failures.append(f"gate B: killrate.json missing entries {missing}")
        cpu_total = cpu_caught = 0
        for bug_id, entry in entries.items():
            row = rows.get(bug_id)
            if row is None:
                continue
            if row.get("exempted"):
                if not row.get("reason"):
                    failures.append(f"gate B: {bug_id} exempted without reason")
                continue
            if entry["reachable"] == "cpu":
                cpu_total += 1
                cpu_caught += 1 if row.get("caught") else 0
            elif not row.get("caught"):
                failures.append(f"gate B: cuda bug {bug_id} not caught")
        if cpu_total and cpu_caught / cpu_total < CPU_KILL_RATIO_MIN:
            failures.append(
                f"gate B: cpu kill rate {cpu_caught}/{cpu_total} below"
                f" {CPU_KILL_RATIO_MIN}"
            )

    tests_file = out_dir / "l0_tests.txt"
    if not tests_file.exists():
        failures.append("gate B: l0_tests.txt missing (cargo test -- --list capture)")
    else:
        listing = tests_file.read_text(encoding="utf-8")
        for name in REQUIRED_SCENARIO_TESTS:
            if name not in listing:
                failures.append(f"gate B: scenario test {name} not in l0 listing")


def check_gate_c(repo_root: Path, out_dir: Path, failures: list[str]) -> None:
    try:
        lanes = load_json(out_dir / "lanes.json")
    except ValidationError as exc:
        failures.append(f"gate C: {exc}")
        lanes = {}
    for lane, budget in LANE_BUDGETS_SECONDS.items():
        value = lanes.get(f"{lane}_seconds")
        if not isinstance(value, (int, float)):
            failures.append(f"gate C: lanes.json missing {lane}_seconds")
        elif value > budget:
            failures.append(f"gate C: {lane} took {value}s > budget {budget}s")

    try:
        stability = load_json(out_dir / "stability.json")
    except ValidationError as exc:
        failures.append(f"gate C: {exc}")
        stability = {}
    for lane, (runs, green) in STABILITY_REQUIRED.items():
        got_runs = stability.get(f"{lane}_runs", 0)
        got_green = stability.get(f"{lane}_green", 0)
        if got_runs < runs or got_green < green:
            failures.append(
                f"gate C: stability {lane} {got_green}/{got_runs} below {green}/{runs}"
            )

    try:
        manifest = load_json(repo_root / MODELS_MANIFEST)
        matrix = load_json(out_dir / "matrix.json")
    except ValidationError as exc:
        failures.append(f"gate C: {exc}")
        return
    matrix_models = {m.get("id"): m for m in matrix.get("models", [])}
    for model in manifest.get("models", []):
        model_id = model.get("id")
        row = matrix_models.get(model_id)
        if row is None:
            failures.append(f"gate C: matrix missing model {model_id}")
            continue
        for platform in ("metal", "cuda"):
            if not model.get(platform):
                continue
            status = (row.get("platforms") or {}).get(platform)
            if status != "PASS":
                failures.append(
                    f"gate C: model {model_id} platform {platform} status"
                    f" {status!r} != PASS"
                )


def run_validate(
    repo_root: Path,
    out_dir: Path,
    state: dict[str, Any] | None = None,
    announce: bool = True,
) -> None:
    failures: list[str] = []
    state = state or repo_state(repo_root)
    if state["dirty"]:
        failures.append("repo: final goal evidence requires a clean worktree")
    if not out_dir.is_dir():
        raise ValidationError(f"out_dir does not exist: {out_dir}")
    check_gate_a(repo_root, failures)
    check_gate_b(repo_root, out_dir, failures)
    check_gate_c(repo_root, out_dir, failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise ValidationError(f"{len(failures)} check(s) failed")
    if announce:
        print(f"validated at sha {state['sha']}")
        print(f"{PASS_PREFIX}: {out_dir}")


# ---------------------------------------------------------------------------
# Self-test


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _selftest_repo(root: Path) -> None:
    """Build a synthetic repo where every final check passes."""
    _write(
        root / MODELS_MAIN_PATH / "clean_model.rs",
        "pub fn forward() {}\n// env::var in comment is skipped\n",
    )
    _write(root / ENGINE_MAIN_PATH / "engine.rs", "pub fn run() {}\n")
    ops = {
        "schema_version": 1,
        "ops": [
            {
                "id": "rms_norm",
                "trait": "Backend",
                "methods": ["rms_norm"],
                "backends": {"cpu": "reference", "metal": True, "cuda": True},
                "tolerance": {"rel": 0.01, "abs": 0.001},
                "covered_today": True,
            },
            {
                "id": "gemm_f16",
                "trait": "Backend",
                "methods": ["gemm"],
                "backends": {"cpu": "reference", "metal": True, "cuda": True},
                "tolerance": {"rel": 0.02, "abs": 0.005},
                "covered_today": True,
            },
        ],
    }
    _write(root / CONFORMANCE_MANIFEST, json.dumps(ops))
    _write(root / OP_DIFF_DIR / "rms_norm.rs", "// parity test\n")
    _write(root / OP_DIFF_DIR / "gemm_f16.rs", "// parity test\n")
    bugs = {
        "schema_version": 1,
        "entries": [
            {
                "id": "hb-01",
                "title": "cpu bug a",
                "kind": "revert-fix",
                "fix_commits": ["abcdef12"],
                "reachable": "cpu",
                "expected_lane": "L0",
                "repro_patch": "patches/hb-01.patch",
                "patch_status": "ready",
            },
            {
                "id": "hb-02",
                "title": "cpu bug b",
                "kind": "revert-fix",
                "fix_commits": ["abcdef13"],
                "reachable": "cpu",
                "expected_lane": "L0",
                "repro_patch": "patches/hb-02.patch",
                "patch_status": "ready",
            },
            {
                "id": "hb-03",
                "title": "cuda live bug",
                "kind": "verify-live",
                "fix_commits": [],
                "reachable": "cuda",
                "expected_lane": "L1-cuda",
                "repro_patch": None,
                "patch_status": "n/a-live",
            },
        ],
    }
    _write(root / HISTORICAL_BUGS, json.dumps(bugs))
    _write(root / GOAL_DIR / "patches" / "hb-01.patch", "--- a\n+++ b\n")
    _write(root / GOAL_DIR / "patches" / "hb-02.patch", "--- a\n+++ b\n")
    _write(
        root / MODELS_MANIFEST,
        json.dumps(
            {"models": [{"id": "qwen3:0.6b", "metal": True, "cuda": True}]}
        ),
    )


def _selftest_out_dir(out_dir: Path) -> None:
    _write(
        out_dir / "killrate.json",
        json.dumps(
            {
                "entries": [
                    {"id": "hb-01", "caught": True},
                    {"id": "hb-02", "caught": False, "exempted": True, "reason": "tiny shapes cannot reproduce"},
                    {"id": "hb-03", "caught": True},
                ]
            }
        ),
    )
    _write(
        out_dir / "lanes.json",
        json.dumps(
            {
                "l0_seconds": 412,
                "l1_metal_seconds": 800,
                "l1_cuda_warm_seconds": 1100,
                "l1_cuda_cold_seconds": 3300,
            }
        ),
    )
    _write(
        out_dir / "stability.json",
        json.dumps(
            {
                "l0_runs": 10,
                "l0_green": 10,
                "l1_metal_runs": 10,
                "l1_metal_green": 10,
                "l1_cuda_runs": 3,
                "l1_cuda_green": 3,
            }
        ),
    )
    _write(
        out_dir / "matrix.json",
        json.dumps(
            {
                "models": [
                    {
                        "id": "qwen3:0.6b",
                        "platforms": {"metal": "PASS", "cuda": "PASS"},
                    }
                ]
            }
        ),
    )
    _write(out_dir / "l0_tests.txt", "\n".join(REQUIRED_SCENARIO_TESTS) + "\n")


def _expect_failure(fn, fragment: str) -> None:
    try:
        fn()
    except ValidationError as exc:
        if fragment not in str(exc):
            raise AssertionError(
                f"expected failure mentioning {fragment!r}, got: {exc}"
            ) from exc
        return
    raise AssertionError(f"expected ValidationError mentioning {fragment!r}")


def run_self_test() -> None:
    fake_state = {"sha": "f" * 40, "dirty": False}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _selftest_repo(root)

        # scan: clean fixture has zero findings
        assert scan_findings(root) == [], "clean fixture must scan to zero findings"

        # scan: seeded patterns are found and attributed
        _write(
            root / MODELS_MAIN_PATH / "dirty_model.rs",
            'let v = std::env::var("FERRUM_X");\n'
            '#[cfg(feature = "cuda")]\n'
            "if B::supports_varlen_qkv() {}\n"
            "static C: OnceLock<u32> = OnceLock::new();\n",
        )
        findings = scan_findings(root)
        got = {f["pattern"] for f in findings}
        assert got == set(PATTERNS), f"expected all patterns, got {got}"
        summary = summarize_findings(findings)
        assert summary["env_var"]["total"] == 1
        assert summary["env_var"]["by_scope"]["models"] == 1

        # allowlist suppresses findings in final gate
        _write(
            root / ALLOWLIST_PATH,
            json.dumps(
                {
                    "entries": [
                        {
                            "path": MODELS_MAIN_PATH.as_posix(),
                            "pattern": name,
                            "reason": "selftest",
                            "owner": "selftest",
                            "review_condition": "never",
                        }
                        for name in PATTERNS
                    ]
                }
            ),
        )
        allowlist = load_allowlist(root)
        assert all(finding_allowed(f, allowlist) for f in scan_findings(root))

        # baseline writes and round-trips
        baseline_path = run_baseline(root, state=fake_state)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert baseline["counts"]["env_var"]["total"] == 1
        assert baseline["conformance"]["covered_ops"] == 2
        assert baseline["historical_bugs"]["total"] == 3

        # manifest validators reject bad shapes
        _expect_failure(
            lambda: validate_conformance({"schema_version": 2}, root, False),
            "schema_version",
        )
        bad_ops = {
            "schema_version": 1,
            "ops": [
                {
                    "id": "ghost_op",
                    "methods": ["x"],
                    "backends": {"cpu": "reference"},
                    "tolerance": {"rel": 0, "abs": 0},
                    "covered_today": True,
                }
            ],
        }
        _expect_failure(
            lambda: validate_conformance(bad_ops, root, False), "marked covered"
        )
        _expect_failure(
            lambda: validate_bugs({"schema_version": 1, "entries": [{"id": "x"}]}),
            "bad id",
        )

        # full validate: happy path passes
        out_dir = root / "out"
        _selftest_out_dir(out_dir)
        run_validate(root, out_dir, state=fake_state, announce=False)

        # full validate: each mutated artifact fails
        _write(
            out_dir / "lanes.json",
            json.dumps(
                {
                    "l0_seconds": 9999,
                    "l1_metal_seconds": 1,
                    "l1_cuda_warm_seconds": 1,
                    "l1_cuda_cold_seconds": 1,
                }
            ),
        )
        _expect_failure(
            lambda: run_validate(root, out_dir, state=fake_state), "check(s) failed"
        )
        _selftest_out_dir(out_dir)  # restore
        run_validate(root, out_dir, state=fake_state, announce=False)

        _write(
            out_dir / "killrate.json",
            json.dumps(
                {
                    "entries": [
                        {"id": "hb-01", "caught": False},
                        {"id": "hb-02", "caught": False},
                        {"id": "hb-03", "caught": True},
                    ]
                }
            ),
        )
        _expect_failure(
            lambda: run_validate(root, out_dir, state=fake_state), "check(s) failed"
        )
        _selftest_out_dir(out_dir)

        # dirty worktree blocks final pass
        _expect_failure(
            lambda: run_validate(
                root, out_dir, state={"sha": "f" * 40, "dirty": True}
            ),
            "check(s) failed",
        )

        # non-allowlisted finding blocks final pass
        (root / ALLOWLIST_PATH).unlink()
        _expect_failure(
            lambda: run_validate(root, out_dir, state=fake_state), "check(s) failed"
        )

    print(SELFTEST_PASS)


# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--validate", metavar="OUT_DIR")
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        elif args.baseline:
            out_path = run_baseline(Path(args.repo_root).resolve())
            print(f"{BASELINE_PREFIX}: {out_path}")
        elif args.validate:
            run_validate(Path(args.repo_root).resolve(), Path(args.validate))
        else:
            print("one of --self-test / --baseline / --validate required")
            return 2
    except (ValidationError, AssertionError) as exc:
        print(f"GATE FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
