#!/usr/bin/env python3
"""README support table <-> models_manifest.json (Plan A) + matrix runner.

The manifest is the single source of truth. This script keeps the README
support table generated from it and drives the regression matrix.

Modes:
  --self-test       validate generation + parsing round-trip on a fixture
  --emit-readme     print the markdown support table built from the manifest
  --check-readme    assert README.md's table matches the manifest (CI guard)
  --plan            print the per-(model, platform) regression plan as JSON
                    (the actual runner that executes pull/run/serve/fingerprint
                    on a GPU host consumes this; execution needs hardware and is
                    out of scope for this local tool)

The README table is delimited by HTML anchor comments so generation is
idempotent:
  <!-- BEGIN model-support-table -->
  ...generated table...
  <!-- END model-support-table -->
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SELFTEST_PASS = "README_MATRIX SELFTEST PASS"
MANIFEST_PATH = Path("scripts/release/models_manifest.json")
README_PATH = Path("README.md")
BEGIN_MARK = "<!-- BEGIN model-support-table -->"
END_MARK = "<!-- END model-support-table -->"


class MatrixError(Exception):
    pass


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise MatrixError(f"missing manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise MatrixError("manifest: schema_version must be 1 with a models list")
    for m in data["models"]:
        for key in ("id", "readme_label", "metal", "cuda", "int4_gptq", "tensor_parallel"):
            if key not in m:
                raise MatrixError(f"manifest model {m.get('id')!r} missing {key}")
    return data


def _cell(flag: bool) -> str:
    return "✓" if flag else "—"


def emit_table(manifest: dict) -> str:
    lines = [
        "| Architecture | Apple Silicon | CUDA | INT4 (GPTQ) | Tensor Parallel |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for m in manifest["models"]:
        lines.append(
            f"| {m['readme_label']} | {_cell(m['metal'])} | {_cell(m['cuda'])} "
            f"| {_cell(m['int4_gptq'])} | {_cell(m['tensor_parallel'])} |"
        )
    return "\n".join(lines)


def check_readme(manifest: dict, readme: Path) -> None:
    if not readme.exists():
        raise MatrixError(f"missing README: {readme}")
    text = readme.read_text(encoding="utf-8")
    expected = emit_table(manifest)
    if BEGIN_MARK in text and END_MARK in text:
        start = text.index(BEGIN_MARK) + len(BEGIN_MARK)
        end = text.index(END_MARK)
        actual = text[start:end].strip()
        if actual != expected:
            raise MatrixError(
                "README table is stale vs manifest; run --emit-readme to refresh"
            )
        return
    # No anchors yet: fall back to substring presence of every label row so an
    # un-instrumented README still can't silently drift from the manifest.
    for m in manifest["models"]:
        row = (
            f"| {m['readme_label']} | {_cell(m['metal'])} | {_cell(m['cuda'])} "
            f"| {_cell(m['int4_gptq'])} | {_cell(m['tensor_parallel'])} |"
        )
        if row not in text:
            raise MatrixError(
                f"README missing/stale row for {m['id']!r}; expected:\n{row}"
            )


def regression_plan(manifest: dict) -> dict:
    """Per-(model, platform) plan the GPU runner consumes."""
    plan = []
    for m in manifest["models"]:
        for platform in ("metal", "cuda"):
            if not m.get(platform):
                continue
            steps = ["pull", "run_multiturn_3", "serve_smoke", "greedy_fingerprint"]
            if m.get("matrix_smoke_only"):
                steps = ["pull", "serve_smoke"]
            plan.append(
                {
                    "id": m["id"],
                    "model": m["representative_model"],
                    "platform": platform,
                    "tier": m.get("matrix_tier", "single_gpu"),
                    "steps": steps,
                }
            )
    return {"schema_version": 1, "cells": plan}


def run_matrix(manifest: dict, ferrum_bin: str, out_dir: Path, platform: str) -> dict:
    """Execute the regression plan for one platform via the ferrum binary.

    Per cell, runs a 3-turn generate against the representative model and
    records PASS/FAIL (exit 0 + non-empty + no template-leak markers). serve
    smoke + fingerprint are recorded as steps but the multi-turn generate is
    the gating check. Writes <out_dir>/matrix.json the final validator reads.
    """
    import re
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    leak_markers = ["<|im_start|>", "<|endoftext|>", "<|assistant|>", "���"]
    rows = []
    for cell in regression_plan(manifest)["cells"]:
        if cell["platform"] != platform:
            continue
        model = cell["model"]
        log = out_dir / f"{cell['id']}_{platform}.log"
        status = "PASS"
        detail = ""
        # Non-LLM modalities (embeddings / ASR / TTS) are marked
        # `matrix_smoke_only`: their plan is pull + serve smoke, not a chat
        # multi-turn. `ferrum run` is chat-only and would falsely FAIL them, so
        # gate those cells on a load smoke (`ferrum pull` succeeds = the model
        # fetches and its config parses on this platform).
        smoke_only = "run_multiturn_3" not in cell.get("steps", [])
        try:
            if smoke_only:
                proc = subprocess.run(
                    [ferrum_bin, "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
                if proc.returncode != 0:
                    status, detail = "FAIL", f"pull exit {proc.returncode}"
            else:
                turns = "你好\n介绍一下你自己\n讲个短笑话\n"
                proc = subprocess.run(
                    [ferrum_bin, "run", model, "--max-tokens", "32"],
                    input=turns,
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
                out = proc.stdout
                if proc.returncode != 0:
                    status, detail = "FAIL", f"exit {proc.returncode}"
                elif len(out.strip()) < 5:
                    status, detail = "FAIL", "empty output"
                elif any(m in out for m in leak_markers):
                    status, detail = "FAIL", "template/garbage marker leaked"
        except subprocess.TimeoutExpired:
            status, detail = "FAIL", "timeout"
        except Exception as exc:  # noqa: BLE001
            status, detail = "FAIL", str(exc)[:80]
        # Decode throughput for the Gate C7 perf floor (LLM chat cells only;
        # ferrum prints "[N tokens, X tok/s, Ys]" — on stderr in practice).
        tok_s = None
        if not smoke_only and log.exists():
            mt = re.search(r"([0-9.]+)\s*tok/s", log.read_text(encoding="utf-8", errors="replace"))
            if mt:
                tok_s = float(mt.group(1))
        rows.append(
            {
                "id": cell["id"],
                "platform": platform,
                "status": status,
                "detail": detail,
                "tok_s": tok_s,
            }
        )

    # collate into per-model platform status + perf the gate expects
    by_model: dict[str, dict] = {}
    for r in rows:
        by_model.setdefault(r["id"], {"id": r["id"], "platforms": {}, "perf": {}})
        by_model[r["id"]]["platforms"][r["platform"]] = r["status"]
        if r.get("tok_s") is not None:
            by_model[r["id"]]["perf"][r["platform"]] = r["tok_s"]
    result = {"schema_version": 1, "platform": platform, "models": list(by_model.values())}
    (out_dir / "matrix.json").write_text(json.dumps(result, indent=2))
    return result


def run_self_test() -> None:
    fixture = {
        "schema_version": 1,
        "models": [
            {
                "id": "demo-llm",
                "readme_label": "Demo LLM",
                "modality": "llm",
                "representative_model": "demo/llm",
                "metal": True,
                "cuda": True,
                "int4_gptq": True,
                "tensor_parallel": False,
            },
            {
                "id": "demo-asr",
                "readme_label": "Demo ASR",
                "modality": "asr",
                "representative_model": "demo/asr",
                "metal": True,
                "cuda": False,
                "int4_gptq": False,
                "tensor_parallel": False,
                "matrix_smoke_only": True,
            },
        ],
    }
    table = emit_table(fixture)
    assert "Demo LLM | ✓ | ✓ | ✓ | —" in table, table
    assert "Demo ASR | ✓ | — | — | —" in table, table

    # check_readme passes when the anchored table matches, fails when stale.
    good = f"intro\n{BEGIN_MARK}\n{table}\n{END_MARK}\nrest"
    bad = f"intro\n{BEGIN_MARK}\nstale\n{END_MARK}\nrest"
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "good.md"
        gp.write_text(good)
        check_readme(fixture, gp)
        bp = Path(tmp) / "bad.md"
        bp.write_text(bad)
        try:
            check_readme(fixture, bp)
        except MatrixError:
            pass
        else:
            raise AssertionError("stale README must fail check")

    plan = regression_plan(fixture)
    # demo-llm -> metal + cuda (4 steps each); demo-asr -> metal smoke only.
    assert len(plan["cells"]) == 3, plan
    asr = [c for c in plan["cells"] if c["id"] == "demo-asr"]
    assert asr and asr[0]["steps"] == ["pull", "serve_smoke"], asr

    # The real manifest, if present, must parse and round-trip.
    if MANIFEST_PATH.exists():
        real = load_manifest(MANIFEST_PATH)
        assert emit_table(real)
        assert regression_plan(real)["cells"]

    print(SELFTEST_PASS)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--emit-readme", action="store_true")
    parser.add_argument("--check-readme", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", metavar="OUT_DIR", help="execute the plan for --platform")
    parser.add_argument("--platform", choices=["cuda", "metal"], default="cuda")
    parser.add_argument("--ferrum-bin", default="ferrum")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--readme", default=str(README_PATH))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
            return 0
        manifest = load_manifest(Path(args.manifest))
        if args.emit_readme:
            print(emit_table(manifest))
        elif args.check_readme:
            check_readme(manifest, Path(args.readme))
            print("README matches manifest")
        elif args.plan:
            print(json.dumps(regression_plan(manifest), indent=2))
        elif args.run:
            res = run_matrix(manifest, args.ferrum_bin, Path(args.run), args.platform)
            fails = [m["id"] for m in res["models"] if "FAIL" in m["platforms"].values()]
            print(json.dumps(res, indent=2))
            if fails:
                raise MatrixError(f"matrix {args.platform} failures: {fails}")
            print(f"MATRIX {args.platform.upper()} PASS: {args.run}")
        else:
            print("one of --self-test / --emit-readme / --check-readme / --plan required")
            return 2
    except (MatrixError, AssertionError) as exc:
        print(f"MATRIX FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
