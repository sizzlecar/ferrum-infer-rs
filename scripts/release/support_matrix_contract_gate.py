#!/usr/bin/env python3
"""Validate README/models support claims against onboarding contracts.

This is the bridge between `models_manifest.json` / README support rows and the
model onboarding contract gate. It rejects support claims that are not backed by
a passing contract artifact from the same git SHA.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_MANIFEST = REPO_ROOT / "scripts/release/models_manifest.json"
PASS_LINE = "SUPPORT MATRIX CONTRACT PASS"
SELFTEST_PASS_LINE = "SUPPORT MATRIX CONTRACT SELFTEST PASS"
SCHEMA_VERSION = 1
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SUPPORT_KEYS = ["metal", "cuda", "int4_gptq", "tensor_parallel"]


class SupportMatrixError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SupportMatrixError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SupportMatrixError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SupportMatrixError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def head_sha() -> str:
    value = git_value(["rev-parse", "HEAD"])
    if not GIT_SHA_RE.match(value):
        raise SupportMatrixError(f"current HEAD is not a git SHA: {value!r}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SupportMatrixError(message)


def require_git_sha(data: dict[str, Any], label: str, expected_sha: str) -> str:
    value = data.get("git_sha")
    require(isinstance(value, str) and GIT_SHA_RE.match(value), f"{label}.git_sha must be a 40-character SHA")
    require(value.lower() == expected_sha.lower(), f"{label}.git_sha {value} is stale vs HEAD {expected_sha}")
    return value


def require_pass_line(data: dict[str, Any], label: str, prefix: str) -> str:
    value = data.get("pass_line")
    require(isinstance(value, str) and value.startswith(f"{prefix}:"), f"{label}.pass_line must start with {prefix}:")
    pass_prefix = value.split(":", 1)[0].upper()
    require("SELFTEST" not in pass_prefix and "SELF-TEST" not in pass_prefix, f"{label}.pass_line must not be selftest evidence")
    return value


def resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def load_models_manifest(path: Path) -> dict[str, Any]:
    data = read_json(path)
    require(data.get("schema_version") == SCHEMA_VERSION, "models_manifest.schema_version must be 1")
    models = data.get("models")
    require(isinstance(models, list) and models, "models_manifest.models must be a non-empty list")
    for index, model in enumerate(models):
        require(isinstance(model, dict), f"models_manifest.models[{index}] must be an object")
        for key in ["id", "readme_label", *SUPPORT_KEYS]:
            require(key in model, f"models_manifest.models[{index}] missing {key}")
    return data


def load_contract_gate(root: Path, expected_sha: str) -> dict[str, Any]:
    manifest = read_json(root / "gate.manifest.json")
    require(manifest.get("status") == "pass", f"{root}/gate.manifest.json status must be pass")
    require_pass_line(manifest, "contract_gate.manifest", "MODEL ONBOARDING CONTRACT PASS")
    require_git_sha(manifest, "contract_gate.manifest", expected_sha)
    summary_path = None
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        summary_path = outputs.get("summary")
    if not isinstance(summary_path, str) or not summary_path.strip():
        summary_path = root / "model_onboarding_contract_summary.json"
    summary = read_json(resolve_path(str(summary_path), base=root))
    require(summary.get("status") == "pass", f"{summary_path} status must be pass")
    return summary


def load_contracts(contract_gate_roots: list[Path], expected_sha: str) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for root in contract_gate_roots:
        summary = load_contract_gate(root, expected_sha)
        for index, result in enumerate(summary.get("contracts") or []):
            require(isinstance(result, dict), f"{root} contracts[{index}] must be an object")
            require(result.get("status") == "pass", f"{root} contracts[{index}].status must be pass")
            contract_id = result.get("contract_id")
            path_value = result.get("contract")
            require(isinstance(contract_id, str) and contract_id, f"{root} contracts[{index}].contract_id missing")
            require(isinstance(path_value, str) and path_value, f"{root} contracts[{index}].contract missing")
            contract_path = resolve_path(path_value, base=root)
            contract = read_json(contract_path)
            require(contract.get("source_git_sha") == expected_sha, f"{contract_id} source_git_sha is stale")
            contract_model = contract.get("model")
            require(isinstance(contract_model, dict), f"{contract_id} model must be an object")
            contract_model_id = contract_model.get("id")
            require(
                isinstance(contract_model_id, str) and contract_model_id.strip(),
                f"{contract_id} model.id missing",
            )
            summary_model_id = result.get("model_id")
            if summary_model_id is not None:
                require(
                    summary_model_id == contract_model_id,
                    f"{contract_id} summary model_id {summary_model_id!r} does not match contract model.id {contract_model_id!r}",
                )
            contracts[contract_id] = {
                "contract_id": contract_id,
                "path": str(contract_path),
                "data": contract,
                "model_id": contract_model_id,
            }
    return contracts


def claim_enabled(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "false", "no", "unsupported", "not yet validated", "not_validated", "—", "-"}
    return False


def contract_backends(contract: dict[str, Any]) -> set[str]:
    backends: set[str] = set()
    for item in contract.get("backend_support") or []:
        if isinstance(item, dict) and item.get("status") == "supported" and isinstance(item.get("backend"), str):
            backends.add(item["backend"])
    return backends


def contract_weight_format(contract: dict[str, Any]) -> str:
    model = contract.get("model")
    if isinstance(model, dict) and isinstance(model.get("weight_format"), str):
        return model["weight_format"].lower()
    return ""


def validate_tensor_parallel_claim(model: dict[str, Any], contract_id: str) -> str:
    artifact = model.get("tensor_parallel_artifact")
    require(
        isinstance(artifact, dict),
        f"model {model.get('id')} tensor_parallel=true requires tensor_parallel_artifact",
    )
    require(artifact.get("status") == "pass", f"model {model.get('id')} tensor_parallel_artifact.status must be pass")
    require_pass_line(artifact, f"model {model.get('id')}.tensor_parallel_artifact", str(artifact.get("pass_prefix") or "TENSOR PARALLEL"))
    require(
        artifact.get("contract_id") == contract_id,
        f"model {model.get('id')} tensor_parallel_artifact.contract_id must match contract_id",
    )
    return str(artifact.get("pass_line"))


def validate_model_row(model: dict[str, Any], contracts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model_id = str(model.get("id"))
    contract_id = model.get("contract_id")
    require(isinstance(contract_id, str) and contract_id.strip(), f"model {model_id} missing contract_id")
    contract_record = contracts.get(contract_id)
    require(contract_record is not None, f"model {model_id} contract_id {contract_id!r} has no passing contract")
    contract = contract_record["data"]
    backends = contract_backends(contract)
    claims = {key: claim_enabled(model.get(key)) for key in SUPPORT_KEYS}
    if claims["metal"]:
        require("metal" in backends, f"model {model_id} claims metal but contract {contract_id} lacks metal backend")
    if claims["cuda"]:
        require("cuda" in backends, f"model {model_id} claims cuda but contract {contract_id} lacks cuda backend")
    if claims["int4_gptq"]:
        require("gptq" in contract_weight_format(contract), f"model {model_id} claims int4_gptq but contract {contract_id} is not GPTQ")
    tensor_parallel_pass_line = None
    if claims["tensor_parallel"]:
        tensor_parallel_pass_line = validate_tensor_parallel_claim(model, contract_id)
    return {
        "model_id": model_id,
        "readme_label": model.get("readme_label"),
        "contract_id": contract_id,
        "contract_model_id": contract_record["model_id"],
        "contract_path": contract_record["path"],
        "claims": claims,
        "contract_backends": sorted(backends),
        "contract_weight_format": contract_weight_format(contract),
        "tensor_parallel_pass_line": tensor_parallel_pass_line,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    expected_sha = head_sha()
    manifest = load_models_manifest(args.models_manifest)
    contracts = load_contracts(args.contract_gate, expected_sha)
    rows = [validate_model_row(model, contracts) for model in manifest["models"]]
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    pass_line = f"{PASS_LINE}: {out}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "support_matrix_contract",
        "git_sha": expected_sha,
        "models_manifest": str(args.models_manifest),
        "contract_gate_count": len(args.contract_gate),
        "row_count": len(rows),
        "rows": rows,
        "pass_line": pass_line,
    }
    write_json(out / "support_matrix_contract_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "support_matrix_contract",
            "status": "pass",
            "repo_root": str(REPO_ROOT),
            "git_sha": expected_sha,
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "outputs": {"summary": str(out / "support_matrix_contract_summary.json")},
            "validation_summary": summary,
        },
    )
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    return summary


def make_contract_gate(root: Path, sha: str) -> Path:
    gate = root / "contract-gate"
    contracts_dir = gate / "contracts"
    contracts_dir.mkdir(parents=True)
    qwen_contract = {
        "schema_version": 1,
        "contract_id": "qwen3-moe-contract",
        "source_git_sha": sha,
        "model": {
            "id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "family": "qwen3_moe",
            "architecture": "qwen3_moe",
            "weight_format": "gptq_int4",
            "source": "hf",
        },
        "backend_support": [
            {"backend": "cuda", "status": "supported"},
        ],
    }
    llama_contract = {
        "schema_version": 1,
        "contract_id": "llama-dense-contract",
        "source_git_sha": sha,
        "model": {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "family": "llama",
            "architecture": "llama_dense",
            "weight_format": "bf16",
            "source": "hf",
        },
        "backend_support": [
            {"backend": "cuda", "status": "supported"},
            {"backend": "metal", "status": "supported"},
        ],
    }
    qwen_path = contracts_dir / "qwen3_moe.json"
    llama_path = contracts_dir / "llama_dense.json"
    write_json(qwen_path, qwen_contract)
    write_json(llama_path, llama_contract)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "gate": "model_onboarding_contract",
        "contracts": [
            {
                "contract": str(qwen_path),
                "contract_id": "qwen3-moe-contract",
                "model_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "status": "pass",
            },
            {
                "contract": str(llama_path),
                "contract_id": "llama-dense-contract",
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "status": "pass",
            },
        ],
        "pass_line": f"MODEL ONBOARDING CONTRACT PASS: {gate}",
    }
    write_json(gate / "model_onboarding_contract_summary.json", summary)
    write_json(
        gate / "gate.manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "artifact_dir": str(gate),
            "git_sha": sha,
            "pass_line": f"MODEL ONBOARDING CONTRACT PASS: {gate}",
            "outputs": {"summary": str(gate / "model_onboarding_contract_summary.json")},
        },
    )
    return gate


def make_manifest(root: Path, models: list[dict[str, Any]]) -> Path:
    path = root / "models_manifest.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "models": models,
        },
    )
    return path


def run_selftest() -> dict[str, Any]:
    sha = head_sha()
    with tempfile.TemporaryDirectory(prefix="ferrum-support-matrix-contract-") as tmp:
        root = Path(tmp)
        contract_gate = make_contract_gate(root, sha)
        valid_manifest = make_manifest(
            root,
            [
                {
                    "id": "qwen3-moe",
                    "readme_label": "Qwen3-MoE",
                    "contract_id": "qwen3-moe-contract",
                    "metal": False,
                    "cuda": True,
                    "int4_gptq": True,
                    "tensor_parallel": False,
                },
                {
                    "id": "llama-family",
                    "readme_label": "LLaMA",
                    "contract_id": "llama-dense-contract",
                    "metal": True,
                    "cuda": True,
                    "int4_gptq": False,
                    "tensor_parallel": False,
                },
            ],
        )
        run_gate(
            argparse.Namespace(
                out=root / "out-valid",
                models_manifest=valid_manifest,
                contract_gate=[contract_gate],
            )
        )
        mismatched_contract_gate = make_contract_gate(root / "mismatched-contract-summary", sha)
        mismatched_summary_path = mismatched_contract_gate / "model_onboarding_contract_summary.json"
        mismatched_summary = read_json(mismatched_summary_path)
        mismatched_summary["contracts"][0]["model_id"] = "Different/Model"
        write_json(mismatched_summary_path, mismatched_summary)
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-mismatched-contract-summary",
                    models_manifest=valid_manifest,
                    contract_gate=[mismatched_contract_gate],
                )
            )
            raise AssertionError("contract summary model_id mismatch unexpectedly passed")
        except SupportMatrixError as exc:
            require("summary model_id" in str(exc), f"unexpected contract summary mismatch failure: {exc}")
        missing_contract_id = make_manifest(
            root / "missing-contract-id",
            [
                {
                    "id": "qwen3-moe",
                    "readme_label": "Qwen3-MoE",
                    "metal": False,
                    "cuda": True,
                    "int4_gptq": True,
                    "tensor_parallel": False,
                }
            ],
        )
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-missing-contract-id",
                    models_manifest=missing_contract_id,
                    contract_gate=[contract_gate],
                )
            )
            raise AssertionError("missing contract_id unexpectedly passed")
        except SupportMatrixError as exc:
            require("contract_id" in str(exc), f"unexpected missing contract failure: {exc}")
        metal_without_backend = make_manifest(
            root / "metal-without-backend",
            [
                {
                    "id": "qwen3-moe",
                    "readme_label": "Qwen3-MoE",
                    "contract_id": "qwen3-moe-contract",
                    "metal": True,
                    "cuda": True,
                    "int4_gptq": True,
                    "tensor_parallel": False,
                }
            ],
        )
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-metal-without-backend",
                    models_manifest=metal_without_backend,
                    contract_gate=[contract_gate],
                )
            )
            raise AssertionError("metal claim without backend unexpectedly passed")
        except SupportMatrixError as exc:
            require("claims metal" in str(exc), f"unexpected metal claim failure: {exc}")
        int4_without_gptq = make_manifest(
            root / "int4-without-gptq",
            [
                {
                    "id": "llama-family",
                    "readme_label": "LLaMA",
                    "contract_id": "llama-dense-contract",
                    "metal": True,
                    "cuda": True,
                    "int4_gptq": True,
                    "tensor_parallel": False,
                }
            ],
        )
        try:
            run_gate(
                argparse.Namespace(
                    out=root / "bad-int4-without-gptq",
                    models_manifest=int4_without_gptq,
                    contract_gate=[contract_gate],
                )
            )
            raise AssertionError("int4 claim without GPTQ contract unexpectedly passed")
        except SupportMatrixError as exc:
            require("int4_gptq" in str(exc), f"unexpected int4 failure: {exc}")
        return {"schema_version": SCHEMA_VERSION, "status": "pass"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--models-manifest", type=Path, default=DEFAULT_MODELS_MANIFEST)
    parser.add_argument("--contract-gate", type=Path, action="append", default=[])
    return parser.parse_args()


def require_normal_args(args: argparse.Namespace) -> None:
    if args.out is None:
        raise SupportMatrixError("--out is required unless --self-test is set")
    if not args.contract_gate:
        raise SupportMatrixError("at least one --contract-gate is required")


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        require_normal_args(args)
        summary = run_gate(args)
        print(summary["pass_line"])
        return 0
    except SupportMatrixError as exc:
        print(f"SUPPORT MATRIX CONTRACT FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
