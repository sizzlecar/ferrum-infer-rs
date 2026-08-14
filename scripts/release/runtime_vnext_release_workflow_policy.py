#!/usr/bin/env python3
"""Validate the v0.8.0 build-once/stage-only release workflow contract."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PASS_PREFIX = "FERRUM RELEASE WORKFLOW POLICY PASS"
SELFTEST_PASS_LINE = "FERRUM RELEASE WORKFLOW POLICY SELFTEST PASS"
PREPROMOTION_READY_PREFIX = "FERRUM PREPROMOTION MANIFEST CONSUMPTION READY"
EXPECTED_VERSION = "0.8.0"
EXPECTED_TAG = "v0.8.0"
EXPECTED_RC_TAG = "v0.8.0-rc.1"
DIAGNOSTICS_TAG = "runtime-vnext-diagnostics-v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MISSING = object()
PREPROMOTION_DEPENDENCY_PASS_PREFIXES = {
    "published_assets": "FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS",
    "crates_io": "FERRUM CRATES IO V0.8.0 PASS",
    "homebrew_metal": "HOMEBREW METAL GATE PASS",
    "homebrew_cuda_fetch": "HOMEBREW CUDA FETCH GATE PASS",
    "workflow_policy": PASS_PREFIX,
}


class PolicyError(RuntimeError):
    """A release workflow violates the frozen v0.8.0 policy."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def parse_workflow_yaml(text: str, label: str) -> dict[str, Any]:
    """Use a real safe YAML parser; Ruby is available on supported build hosts."""
    ruby = r"""
require 'yaml'
require 'json'
text = STDIN.read
walk = lambda do |node|
  if node.is_a?(Psych::Nodes::Mapping)
    seen = {}
    node.children.each_slice(2) do |key, _value|
      next unless key.is_a?(Psych::Nodes::Scalar)
      raise "duplicate YAML key: #{key.value}" if seen[key.value]
      seen[key.value] = true
    end
  end
  Array(node.respond_to?(:children) ? node.children : nil).each { |child| walk.call(child) }
end
walk.call(Psych.parse_stream(text))
value = YAML.safe_load(text, permitted_classes: [], aliases: false)
if value.is_a?(Hash) && value.key?(true) && !value.key?('on')
  value['on'] = value.delete(true)
end
STDOUT.write(JSON.generate(value))
"""
    try:
        completed = subprocess.run(
            ["ruby", "-e", ruby],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PolicyError("Ruby is required to parse workflow YAML safely") from exc
    require(completed.returncode == 0, f"{label}: invalid YAML: {completed.stderr.strip()}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{label}: YAML parser returned invalid JSON: {exc}") from exc
    require(isinstance(parsed, dict), f"{label}: workflow document must be a mapping")
    return parsed


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context} must be a mapping")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    require(isinstance(value, list), f"{context} must be a list")
    return value


def workflow_dispatch_inputs(document: dict[str, Any], label: str) -> dict[str, Any]:
    trigger = require_mapping(document.get("on"), f"{label}.on")
    require(set(trigger) == {"workflow_dispatch"}, f"{label} must only use workflow_dispatch, found {sorted(trigger)}")
    dispatch = trigger["workflow_dispatch"]
    if dispatch in ({}, None):
        return {}
    return require_mapping(require_mapping(dispatch, f"{label}.on.workflow_dispatch").get("inputs", {}), f"{label}.inputs")


def workflow_jobs(document: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    jobs = require_mapping(document.get("jobs"), f"{label}.jobs")
    require(bool(jobs), f"{label}.jobs must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for name, job in jobs.items():
        result[name] = require_mapping(job, f"{label}.jobs.{name}")
    return result


def job_steps(job: dict[str, Any], context: str) -> list[dict[str, Any]]:
    raw = require_list(job.get("steps"), f"{context}.steps")
    result: list[dict[str, Any]] = []
    for index, step in enumerate(raw):
        result.append(require_mapping(step, f"{context}.steps[{index}]"))
    return result


def combined_run_scripts(steps: Iterable[dict[str, Any]]) -> str:
    return "\n".join(str(step.get("run", "")) for step in steps)


def step_uses(steps: Iterable[dict[str, Any]]) -> list[str]:
    return [str(step["uses"]) for step in steps if "uses" in step]


def _validate_permissions(document: dict[str, Any], expected: dict[str, str], label: str) -> None:
    permissions = require_mapping(document.get("permissions"), f"{label}.permissions")
    require(permissions == expected, f"{label}.permissions must be exactly {expected}, found {permissions}")


def _require_dispatch_input(inputs: dict[str, Any], name: str, label: str, *, default: Any = MISSING) -> dict[str, Any]:
    spec = require_mapping(inputs.get(name), f"{label}.inputs.{name}")
    require(spec.get("required") is True, f"{label}.inputs.{name}.required must be true")
    if default is not MISSING:
        require(spec.get("default") == default, f"{label}.inputs.{name}.default must be {default!r}")
    return spec


def _validate_checkout_and_clean_guard(
    steps: list[dict[str, Any]],
    context: str,
    *,
    require_rc_tag: bool = False,
) -> None:
    checkouts = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")]
    require(len(checkouts) == 1, f"{context} must contain exactly one actions/checkout step")
    checkout_with = require_mapping(checkouts[0].get("with"), f"{context}.checkout.with")
    require(checkout_with.get("ref") == "${{ inputs.release_candidate_sha }}", f"{context} checkout must use the exact release_candidate_sha input")
    require(checkout_with.get("fetch-depth") == 0, f"{context} checkout must use fetch-depth 0")
    scripts = combined_run_scripts(steps)
    markers = [
        'expected_sha="${{ inputs.release_candidate_sha }}"',
        'actual_sha="$(git rev-parse HEAD)"',
        'test "$actual_sha" = "$expected_sha"',
        'test -z "$(git status --porcelain)"',
    ]
    if require_rc_tag:
        markers.extend(
            [
                'expected_tag="${{ inputs.release_candidate_tag }}"',
                'test "$(git cat-file -t "$expected_tag")" = tag',
                'test "$(git rev-parse "$expected_tag^{}")" = "$expected_sha"',
            ]
        )
    for marker in markers:
        require(marker in scripts, f"{context} is missing exact-SHA/clean guard marker {marker!r}")


def _validate_staging_workflow(
    document: dict[str, Any],
    label: str,
    expected_jobs: dict[str, tuple[str, str, str]],
) -> None:
    inputs = workflow_dispatch_inputs(document, label)
    require(
        set(inputs)
        == {
            "release_candidate_sha",
            "release_candidate_tag",
            "staging_label",
            "publish_release",
        },
        f"{label} has unexpected workflow inputs: {sorted(inputs)}",
    )
    _require_dispatch_input(inputs, "release_candidate_sha", label)
    _require_dispatch_input(
        inputs,
        "release_candidate_tag",
        label,
        default=EXPECTED_RC_TAG,
    )
    _require_dispatch_input(inputs, "staging_label", label, default="v0.8.0-rc")
    _require_dispatch_input(inputs, "publish_release", label, default=False)
    _validate_permissions(document, {"contents": "read"}, label)
    jobs = workflow_jobs(document, label)
    require(set(jobs) == set(expected_jobs), f"{label} jobs mismatch: expected {sorted(expected_jobs)}, found {sorted(jobs)}")
    for job_name, (asset, target, backend) in expected_jobs.items():
        context = f"{label}.jobs.{job_name}"
        steps = job_steps(jobs[job_name], context)
        _validate_checkout_and_clean_guard(steps, context, require_rc_tag=True)
        uses = step_uses(steps)
        allowed_actions = {
            "actions/checkout@v4",
            "actions/cache@v4",
            "actions/upload-artifact@v4",
            "dtolnay/rust-toolchain@1.91.0",
        }
        require(set(uses).issubset(allowed_actions), f"{context} contains a non-staging action: {sorted(set(uses) - allowed_actions)}")
        scripts = combined_run_scripts(steps)
        require('if [[ "${{ inputs.publish_release }}" != "false" ]]' in scripts, f"{context} does not fail closed when publish_release is true")
        require(scripts.count("cargo build") == 1, f"{context} must invoke cargo build exactly once")
        require(scripts.count("cargo build --release --locked -p ferrum-cli --bin ferrum") == 1, f"{context} must build the release binary exactly once with the canonical command")
        require(f'test "$version" = "{EXPECTED_VERSION}"' in scripts, f"{context} does not lock workspace version {EXPECTED_VERSION}")
        require(asset in scripts, f"{context} does not package expected asset {asset}")
        require(target in scripts, f"{context} does not record expected target triple {target}")
        require(f'BACKEND="{backend}"' in scripts, f"{context} does not record expected backend {backend}")
        for marker in (
            'Path(f"{asset}.sha256")',
            'Path(f"{asset}.binary.sha256")',
            '("version.json", version)',
            '("dependency.json", dependency)',
            '("abi.json", abi)',
        ):
            require(marker in scripts, f"{context} does not generate adjacent manifest marker {marker!r}")
        for marker in (
            "release_candidate_sha",
            "release_candidate_tag",
            "asset_sha256",
            "binary_sha256",
            "workflow_run_id",
        ):
            require(marker in scripts, f"{context} staged manifests omit {marker}")
        uploads = [step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
        require(len(uploads) == 1, f"{context} must upload exactly one immutable staged artifact bundle")
        upload_with = require_mapping(uploads[0].get("with"), f"{context}.upload.with")
        upload_paths = str(upload_with.get("path", ""))
        for suffix in ("", ".sha256", ".binary.sha256", ".version.json", ".dependency.json", ".abi.json"):
            require(asset + suffix in upload_paths, f"{context} upload omits {asset + suffix}")
        lowered = scripts.lower()
        for pattern in (
            r"\bgh\s+release\s+(?:create|edit|upload|delete)\b",
            r"\bgh\s+api\b",
            r"\bcargo\s+publish\b",
            r"\bdocker\s+push\b",
            r"\bgit\s+(?:tag|push)\b",
            r"\b(?:curl|wget)\b[^\n]*api\.github\.com",
        ):
            require(re.search(pattern, lowered) is None, f"{context} contains forbidden publication command matching {pattern}")


def validate_release_workflow(document: dict[str, Any]) -> None:
    _validate_staging_workflow(
        document,
        "release.yml",
        {
            "linux-x86_64": ("ferrum-linux-x86_64.tar.gz", "x86_64-unknown-linux-gnu", "cpu"),
            "macos-aarch64": ("ferrum-macos-aarch64.tar.gz", "aarch64-apple-darwin", "metal"),
        },
    )


def validate_cuda_workflow(document: dict[str, Any]) -> None:
    _validate_staging_workflow(
        document,
        "release-cuda.yml",
        {
            "linux-x86_64-cuda-sm89": (
                "ferrum-linux-x86_64-cuda-sm89.tar.gz",
                "x86_64-unknown-linux-gnu",
                "cuda",
            )
        },
    )
    env = require_mapping(document.get("env"), "release-cuda.yml.env")
    require(str(env.get("CUDA_COMPUTE_CAP")) == "89", "release-cuda.yml must target CUDA sm89")
    jobs = workflow_jobs(document, "release-cuda.yml")
    cuda_job = require_mapping(
        jobs.get("linux-x86_64-cuda-sm89"),
        "release-cuda.yml.jobs.linux-x86_64-cuda-sm89",
    )
    container = require_mapping(
        cuda_job.get("container"),
        "release-cuda.yml.jobs.linux-x86_64-cuda-sm89.container",
    )
    require(
        container.get("image") == "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "release-cuda.yml CUDA build image must match the pinned native operator set toolchain",
    )
    cuda_steps = job_steps(
        cuda_job,
        "release-cuda.yml.jobs.linux-x86_64-cuda-sm89",
    )
    cache_steps = [
        step
        for step in cuda_steps
        if str(step.get("uses", "")).startswith("actions/cache@")
    ]
    require(len(cache_steps) == 1, "release-cuda.yml must contain exactly one cache step")
    cache_with = require_mapping(
        cache_steps[0].get("with"),
        "release-cuda.yml CUDA cache.with",
    )
    require(
        str(cache_with.get("path", "")).splitlines()
        == ["~/.cargo/registry", "~/.cargo/git"],
        "release-cuda.yml must not restore target/ across CUDA/native-set identities",
    )
    cache_prefix = (
        "stage-v0.8.0-linux-x86_64-cuda-sm89-"
        "cuda12.4-native-d229c130-cargo-"
    )
    require(
        cache_with.get("key")
        == cache_prefix + "${{ hashFiles('**/Cargo.lock') }}"
        and cache_with.get("restore-keys") == cache_prefix,
        "release-cuda.yml cache key must bind CUDA 12.4 and native set d229c130",
    )
    require(
        env.get("NATIVE_OPERATOR_SET_ARCHIVE_URL")
        == "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/runtime-vnext-diagnostics-v1/native-operator-set-5503d913.tar.zst",
        "release-cuda.yml native operator set URL is not frozen",
    )
    require(
        env.get("NATIVE_OPERATOR_SET_ARCHIVE_SHA256")
        == "d229c130cbc6bbb3cac86137c29d2e458e8812420d4d57a0d18505c88ca5461e",
        "release-cuda.yml native operator set SHA256 is not frozen",
    )
    scripts = "\n".join(
        combined_run_scripts(job_steps(job, f"release-cuda.yml.jobs.{name}"))
        for name, job in workflow_jobs(document, "release-cuda.yml").items()
    )
    require("--features cuda,vllm-moe-marlin,vllm-paged-attn-v2" in scripts, "release-cuda.yml feature set is incomplete")
    require(
        "validate_native_operator_set" in scripts
        and "FERRUM_NATIVE_OPERATOR_SET_LOCK" in scripts,
        "release-cuda.yml does not validate and bind the pinned native operator set",
    )
    require("python|torch|vllm" in scripts, "release-cuda.yml lacks forbidden runtime-linkage scan")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_docker_workflow(document: dict[str, Any]) -> None:
    inputs = workflow_dispatch_inputs(document, "docker.yml")
    require(inputs == {}, "docker.yml must not accept publishing or tag inputs")
    _validate_permissions(document, {"contents": "read"}, "docker.yml")
    keys = {key.lower() for key in _walk_keys(document)}
    require("push" not in keys and "tags" not in keys, "docker.yml contains a push/tag trigger or job")
    jobs = workflow_jobs(document, "docker.yml")
    for name, job in jobs.items():
        steps = job_steps(job, f"docker.yml.jobs.{name}")
        uses = "\n".join(step_uses(steps)).lower()
        scripts = combined_run_scripts(steps).lower()
        require("docker/" not in uses and "docker." not in uses, "docker.yml invokes a Docker build/login/publish action")
        for pattern in (r"\bdocker\s+(?:push|buildx|tag)\b", r"\bghcr\.io\b", r"\b(?:stable|latest|candidate|0\.8(?:\.0)?)\s*[:=]"):
            require(re.search(pattern, scripts) is None, f"docker.yml contains forbidden Docker publication material matching {pattern}")


def validate_promotion_workflow(document: dict[str, Any]) -> None:
    label = "release-promote.yml"
    inputs = workflow_dispatch_inputs(document, label)
    required_inputs = {
        "release_candidate_sha",
        "release_id",
        "prepromotion_asset_id",
        "prepromotion_archive_sha256",
        "prepromotion_child_sha256",
    }
    require(set(inputs) == required_inputs, f"{label} inputs mismatch: expected {sorted(required_inputs)}, found {sorted(inputs)}")
    for name in required_inputs:
        _require_dispatch_input(inputs, name, label)
    _validate_permissions(document, {"actions": "read", "contents": "write"}, label)
    concurrency = require_mapping(document.get("concurrency"), f"{label}.concurrency")
    require(
        concurrency.get("group") == "promote-validated-prerelease-v0.8.0",
        f"{label} concurrency must serialize the one final release tag",
    )
    require(concurrency.get("cancel-in-progress") is False, f"{label} must serialize rather than cancel promotion attempts")
    jobs = workflow_jobs(document, label)
    require(set(jobs) == {"promote-validated-prerelease"}, f"{label} must have one promotion job")
    job = jobs["promote-validated-prerelease"]
    environment = require_mapping(job.get("env"), f"{label}.jobs.promote-validated-prerelease.env")
    require(environment.get("RELEASE_TAG") == EXPECTED_TAG, f"{label} must fix the final release tag")
    require(environment.get("DIAGNOSTICS_TAG") == DIAGNOSTICS_TAG, f"{label} must fix the diagnostics release tag")
    for key, input_name in (
        ("RC_SHA", "release_candidate_sha"),
        ("RELEASE_ID", "release_id"),
        ("PREPROMOTION_ASSET_ID", "prepromotion_asset_id"),
        ("PREPROMOTION_ARCHIVE_SHA256", "prepromotion_archive_sha256"),
        ("PREPROMOTION_CHILD_SHA256", "prepromotion_child_sha256"),
    ):
        require(
            environment.get(key) == f"${{{{ inputs.{input_name} }}}}",
            f"{label} {key} does not bind inputs.{input_name}",
        )
    steps = job_steps(job, f"{label}.jobs.promote-validated-prerelease")
    uses = step_uses(steps)
    allowed_actions = {
        "actions/checkout@v4",
        "actions/upload-artifact@v4",
    }
    require(set(uses).issubset(allowed_actions), f"{label} contains an action outside the promotion evidence boundary: {sorted(set(uses) - allowed_actions)}")
    _validate_checkout_and_clean_guard(steps, f"{label}.jobs.promote-validated-prerelease")
    scripts = combined_run_scripts(steps)
    for marker in (
        "runtime_vnext_prepromotion_bundle.py verify",
        '--expected-archive-sha256 "$PREPROMOTION_ARCHIVE_SHA256"',
        '--expected-child-sha256 "$PREPROMOTION_CHILD_SHA256"',
        "--validate-prepromotion-manifest",
        "--expected-manifest-sha256",
        "--release-candidate-sha",
        "--expected-tag",
        "--expected-release-id",
        "asset_set_sha256",
        'release["prerelease"] in (True, False)',
        'release["draft"] is False',
        "git/ref/tags/v0.8.0",
        "promotion-consumption.json",
        "CONSUMPTION_CLAIM_NAME",
        "CONSUMPTION_COMPLETE_NAME",
        '"state": "consumed"',
        'claim["promotion"] == {"state": "pending"}',
        'assert not complete, "prepromotion manifest already has a durable complete marker"',
        "resume_claim",
        "ALREADY_PROMOTED",
    ):
        require(marker in scripts, f"{label} is missing promotion guard marker {marker!r}")
    names = [str(step.get("name", "")) for step in steps]
    for expected_name in (
        "Persist consumption claim before release mutation",
        "Flip only prerelease to false",
        "Upload durable promotion completion marker",
    ):
        require(names.count(expected_name) == 1, f"{label} must contain exactly one {expected_name!r} step")
    claim_index = names.index("Persist consumption claim before release mutation")
    mutation_index = names.index("Flip only prerelease to false")
    final_receipt_index = names.index("Upload durable promotion completion marker")
    require(claim_index < mutation_index < final_receipt_index, f"{label} must persist consumption before mutation and final evidence afterward")
    uploads = [step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    require(len(uploads) == 2, f"{label} must upload one pre-mutation claim and one final receipt")
    claim_with = require_mapping(uploads[0].get("with"), f"{label}.consumption-claim.with")
    require(claim_with.get("name") == "${{ env.CONSUMPTION_CLAIM_NAME }}", f"{label} consumption claim name is not deterministic")
    require(str(claim_with.get("path")) == "promotion-consumption.json", f"{label} consumption claim payload mismatch")
    complete_with = require_mapping(uploads[1].get("with"), f"{label}.consumption-complete.with")
    require(
        complete_with.get("name") == "${{ env.CONSUMPTION_COMPLETE_NAME }}",
        f"{label} completion marker name is not deterministic",
    )
    require(
        "promotion-consumption.json" in str(complete_with.get("path", "")),
        f"{label} completion marker omits its canonical receipt",
    )
    patch_lines = [line.strip() for line in scripts.splitlines() if "gh api --method PATCH" in line]
    require(patch_lines == ['gh api --method PATCH "repos/${GITHUB_REPOSITORY}/releases/${RELEASE_ID}" -F prerelease=false > release-after.json'], f"{label} must perform exactly one release mutation and only set prerelease=false")
    api_lines = [line.strip() for line in scripts.splitlines() if line.strip().startswith("gh api")]
    require(
        api_lines
        == [
            'gh api "repos/${GITHUB_REPOSITORY}/releases/tags/${DIAGNOSTICS_TAG}" > prepromotion-diagnostics-release.json',
            'gh api -H "Accept: application/octet-stream" "repos/${GITHUB_REPOSITORY}/releases/assets/${PREPROMOTION_ASSET_ID}" > prepromotion-bundle.zip',
            'gh api "repos/${GITHUB_REPOSITORY}/releases/${RELEASE_ID}" > release-before.json',
            'gh api "repos/${GITHUB_REPOSITORY}/git/ref/tags/v0.8.0" > tag-ref.json',
            'gh api "repos/${GITHUB_REPOSITORY}/git/tags/${tag_object_sha}" > annotated-tag.json',
            'gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${CONSUMPTION_COMPLETE_NAME}&per_page=100" > prior-complete-artifacts.json',
            'gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${CONSUMPTION_CLAIM_NAME}&per_page=100" > prior-pending-artifacts.json',
            'gh api --method PATCH "repos/${GITHUB_REPOSITORY}/releases/${RELEASE_ID}" -F prerelease=false > release-after.json',
        ],
        f"{label} API surface must be the fixed diagnostics reads, consumption reads, and one prerelease-only mutation",
    )
    lowered = scripts.lower()
    require(re.search(r"\bgh\s+release\b", lowered) is None, f"{label} must not create, upload, edit, or delete release assets")
    require(re.search(r"--method\s+(?:post|put|delete)\b", lowered) is None, f"{label} contains a forbidden mutating HTTP method")
    for forbidden in ("-f draft=", "-f tag_name=", "-f target_commitish=", "-f make_latest=", "--clobber"):
        require(forbidden not in lowered, f"{label} contains forbidden mutation {forbidden!r}")
    require(re.search(r"\b(?:curl|wget)\b[^\n]*api\.github\.com", lowered) is None, f"{label} contains an unreviewed GitHub API client")


def validate_workflow_set(texts: dict[str, str]) -> dict[str, dict[str, Any]]:
    expected = {
        "release.yml",
        "release-cuda.yml",
        "docker.yml",
        "release-promote.yml",
    }
    require(
        set(texts) == expected,
        f"release workflow source set mismatch: expected {sorted(expected)}, found {sorted(texts)}",
    )
    parsed = {name: parse_workflow_yaml(text, name) for name, text in texts.items()}
    validate_release_workflow(parsed["release.yml"])
    validate_cuda_workflow(parsed["release-cuda.yml"])
    validate_docker_workflow(parsed["docker.yml"])
    validate_promotion_workflow(parsed["release-promote.yml"])
    return parsed


def _json_mapping(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"{context}: cannot read JSON: {exc}") from exc
    return require_mapping(value, context)


def validate_prepromotion_payload(
    payload: dict[str, Any],
    *,
    release_candidate_sha: str,
    expected_tag: str,
    expected_release_id: str,
) -> None:
    require(SHA_RE.fullmatch(release_candidate_sha) is not None, "expected release candidate SHA is invalid")
    require(expected_tag == EXPECTED_TAG, f"promotion tag must be {EXPECTED_TAG}")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "lane",
        "version",
        "canonical",
        "artifact_dir",
        "manifest_id",
        "release_candidate_sha",
        "pass_line",
        "prepromotion_pass_line",
        "release",
        "consumption",
        "dependencies",
        "created_at",
    }
    require(set(payload) == expected_fields, "prepromotion manifest fields differ")
    require(
        payload.get("schema_version") == 1
        and payload.get("artifact_type") == "runtime_vnext_prepromotion_manifest"
        and payload.get("status") == "pass"
        and payload.get("lane") == "runtime-vnext-prepromotion"
        and payload.get("version") == EXPECTED_VERSION
        and payload.get("canonical") is True,
        "prepromotion manifest schema/identity/status differs",
    )
    require(
        isinstance(payload.get("artifact_dir"), str) and payload["artifact_dir"],
        "prepromotion artifact_dir is missing",
    )
    require(
        isinstance(payload.get("created_at"), str) and payload["created_at"],
        "prepromotion created_at is missing",
    )
    require(payload.get("release_candidate_sha") == release_candidate_sha, "prepromotion manifest release-candidate SHA mismatch")
    manifest_id = str(payload.get("manifest_id", ""))
    require(SHA256_RE.fullmatch(manifest_id) is not None, "prepromotion manifest_id must be a SHA256")
    pass_line = payload.get("pass_line")
    require(
        isinstance(pass_line, str)
        and pass_line.startswith("FERRUM V0.8.0 PREPROMOTION PASS:")
        and payload.get("prepromotion_pass_line") == pass_line,
        "prepromotion child PASS line binding differs",
    )
    release = require_mapping(payload.get("release"), "prepromotion.release")
    require(
        set(release)
        == {
            "id",
            "tag_name",
            "tag_sha",
            "draft",
            "prerelease",
            "asset_set_sha256",
        },
        "prepromotion release fields differ",
    )
    require(
        str(release.get("id")) == str(expected_release_id)
        and release.get("tag_name") == expected_tag
        and release.get("tag_sha") == release_candidate_sha
        and release.get("draft") is False
        and release.get("prerelease") is True,
        "prepromotion release identity/state differs",
    )
    require(SHA256_RE.fullmatch(str(release.get("asset_set_sha256", ""))) is not None, "prepromotion asset_set_sha256 is invalid")
    consumption = require_mapping(payload.get("consumption"), "prepromotion.consumption")
    require(
        set(consumption)
        == {"state", "release_id", "token", "consumed_at", "consumed_by"},
        "prepromotion consumption fields differ",
    )
    require(
        consumption.get("state") == "unconsumed"
        and str(consumption.get("release_id")) == str(expected_release_id)
        and consumption.get("consumed_at") is None
        and consumption.get("consumed_by") is None
        and re.fullmatch(r"[A-Za-z0-9._-]{32,}", str(consumption.get("token", "")))
        is not None,
        "prepromotion consumption state/token differs",
    )
    dependencies = require_mapping(payload.get("dependencies"), "prepromotion.dependencies")
    require(
        set(dependencies) == set(PREPROMOTION_DEPENDENCY_PASS_PREFIXES),
        "prepromotion dependency denominator differs",
    )
    for name, prefix in PREPROMOTION_DEPENDENCY_PASS_PREFIXES.items():
        dependency = require_mapping(
            dependencies.get(name), f"prepromotion.dependencies.{name}"
        )
        require(
            set(dependency) == {"status", "pass_line", "manifest"}
            and dependency.get("status") == "pass"
            and isinstance(dependency.get("pass_line"), str)
            and dependency["pass_line"].startswith(f"{prefix}: "),
            f"prepromotion dependency {name} status/PASS differs",
        )
        reference = require_mapping(
            dependency.get("manifest"),
            f"prepromotion.dependencies.{name}.manifest",
        )
        require(
            set(reference) == {"path", "sha256", "size_bytes"}
            and isinstance(reference.get("path"), str)
            and bool(reference["path"])
            and SHA256_RE.fullmatch(str(reference.get("sha256", ""))) is not None
            and type(reference.get("size_bytes")) is int
            and reference["size_bytes"] >= 0,
            f"prepromotion dependency {name} manifest reference differs",
        )
    identity_payload = {
        "schema_version": 1,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": release_candidate_sha,
        "release": release,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    require(
        canonical_json_sha256(identity_payload) == manifest_id,
        "prepromotion manifest_id does not bind its immutable payload",
    )


def validate_prepromotion_manifest_file(
    path: Path,
    *,
    expected_manifest_sha256: str,
    release_candidate_sha: str,
    expected_tag: str,
    expected_release_id: str,
) -> None:
    require(SHA256_RE.fullmatch(expected_manifest_sha256) is not None, "expected prepromotion manifest SHA256 is invalid")
    require(sha256_file(path) == expected_manifest_sha256, "prepromotion manifest file SHA256 mismatch")
    payload = _json_mapping(path, "prepromotion manifest")
    validate_prepromotion_payload(
        payload,
        release_candidate_sha=release_candidate_sha,
        expected_tag=expected_tag,
        expected_release_id=expected_release_id,
    )


def _replace_once(text: str, old: str, new: str, fixture: str) -> str:
    require(text.count(old) == 1, f"self-test fixture {fixture}: expected one marker {old!r}, found {text.count(old)}")
    return text.replace(old, new, 1)


def _replace_first(text: str, old: str, new: str, fixture: str) -> str:
    require(old in text, f"self-test fixture {fixture}: missing marker {old!r}")
    return text.replace(old, new, 1)


def _expect_policy_failure(label: str, callback: Any) -> None:
    try:
        callback()
    except PolicyError:
        return
    raise PolicyError(f"negative fixture {label} unexpectedly passed")


def _synthetic_prepromotion_payload() -> dict[str, Any]:
    rc = "a" * 40
    release_id = "12345"
    dependencies = {
        name: {
            "status": "pass",
            "pass_line": f"{prefix}: /tmp/{name}",
            "manifest": {
                "path": f"/tmp/{name}/gate.manifest.json",
                "sha256": f"{index + 1:064x}",
                "size_bytes": 1024 + index,
            },
        }
        for index, (name, prefix) in enumerate(
            PREPROMOTION_DEPENDENCY_PASS_PREFIXES.items()
        )
    }
    release = {
        "id": release_id,
        "tag_name": EXPECTED_TAG,
        "tag_sha": rc,
        "draft": False,
        "prerelease": True,
        "asset_set_sha256": "c" * 64,
    }
    consumption = {
        "state": "unconsumed",
        "release_id": release_id,
        "token": "fixture-consumption-token-0123456789abcdef",
        "consumed_at": None,
        "consumed_by": None,
    }
    identity = {
        "schema_version": 1,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": rc,
        "release": release,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_prepromotion_manifest",
        "status": "pass",
        "lane": "runtime-vnext-prepromotion",
        "version": EXPECTED_VERSION,
        "canonical": True,
        "artifact_dir": "/tmp/prepromotion",
        "manifest_id": canonical_json_sha256(identity),
        "release_candidate_sha": rc,
        "pass_line": "FERRUM V0.8.0 PREPROMOTION PASS: /tmp/prepromotion",
        "prepromotion_pass_line": "FERRUM V0.8.0 PREPROMOTION PASS: /tmp/prepromotion",
        "release": release,
        "consumption": consumption,
        "dependencies": dependencies,
        "created_at": "2026-08-14T00:00:00+00:00",
    }


def run_selftest(texts: dict[str, str]) -> None:
    validate_workflow_set(texts)

    direct_release = dict(texts)
    direct_release["release.yml"] = _replace_once(
        direct_release["release.yml"],
        "      - name: Upload staged CPU asset",
        "      - name: Forbidden direct official release fixture\n        uses: softprops/action-gh-release@v2\n\n      - name: Upload staged CPU asset",
        "direct-official-release",
    )
    _expect_policy_failure("direct-official-release", lambda: validate_workflow_set(direct_release))

    docker_tag = dict(texts)
    docker_tag["docker.yml"] = _replace_once(
        docker_tag["docker.yml"],
        "on:\n  workflow_dispatch:\n",
        "on:\n  push:\n    tags:\n      - v0.8.0\n  workflow_dispatch:\n",
        "docker-tag-trigger",
    )
    _expect_policy_failure("docker-tag-trigger", lambda: validate_workflow_set(docker_tag))

    docker_publish = dict(texts)
    docker_publish["docker.yml"] = _replace_once(
        docker_publish["docker.yml"],
        "    steps:\n      - name: Record unsupported distribution policy",
        "    steps:\n      - name: Forbidden Docker publish fixture\n        uses: docker/build-push-action@v5\n        with:\n          push: true\n          tags: ghcr.io/example/ferrum:latest\n\n      - name: Record unsupported distribution policy",
        "docker-publish-job",
    )
    _expect_policy_failure("docker-publish-job", lambda: validate_workflow_set(docker_publish))

    missing_child = dict(texts)
    missing_child["release-promote.yml"] = _replace_once(
        missing_child["release-promote.yml"],
        "runtime_vnext_prepromotion_bundle.py verify",
        "runtime_vnext_prepromotion_bundle.py pack",
        "missing-prepromotion-child",
    )
    _expect_policy_failure("missing-prepromotion-child", lambda: validate_workflow_set(missing_child))

    sha_mismatch = dict(texts)
    sha_mismatch["release.yml"] = _replace_first(
        sha_mismatch["release.yml"],
        "ref: ${{ inputs.release_candidate_sha }}",
        "ref: ${{ github.sha }}",
        "release-candidate-sha-mismatch",
    )
    _expect_policy_failure("release-candidate-sha-mismatch", lambda: validate_workflow_set(sha_mismatch))

    tag_mismatch = dict(texts)
    tag_mismatch["release-cuda.yml"] = _replace_once(
        tag_mismatch["release-cuda.yml"],
        'test "$(git rev-parse "$expected_tag^{}")" = "$expected_sha"',
        'test "$(git rev-parse "$expected_tag^{}")" = "$GITHUB_SHA"',
        "release-candidate-tag-mismatch",
    )
    _expect_policy_failure(
        "release-candidate-tag-mismatch",
        lambda: validate_workflow_set(tag_mismatch),
    )

    cuda_toolkit_mismatch = dict(texts)
    cuda_toolkit_mismatch["release-cuda.yml"] = _replace_once(
        cuda_toolkit_mismatch["release-cuda.yml"],
        "      image: nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "      image: nvidia/cuda:12.6.0-devel-ubuntu22.04",
        "cuda-toolkit-native-operator-mismatch",
    )
    _expect_policy_failure(
        "cuda-toolkit-native-operator-mismatch",
        lambda: validate_workflow_set(cuda_toolkit_mismatch),
    )

    cuda_cache_mismatch = dict(texts)
    cuda_cache_mismatch["release-cuda.yml"] = _replace_once(
        cuda_cache_mismatch["release-cuda.yml"],
        "cuda12.4-native-d229c130-cargo-${{ hashFiles('**/Cargo.lock') }}",
        "cuda12.6-native-d229c130-cargo-${{ hashFiles('**/Cargo.lock') }}",
        "cuda-cache-toolchain-mismatch",
    )
    _expect_policy_failure(
        "cuda-cache-toolchain-mismatch",
        lambda: validate_workflow_set(cuda_cache_mismatch),
    )

    diagnostics_tag_mismatch = dict(texts)
    diagnostics_tag_mismatch["release-promote.yml"] = _replace_once(
        diagnostics_tag_mismatch["release-promote.yml"],
        f"DIAGNOSTICS_TAG: {DIAGNOSTICS_TAG}",
        "DIAGNOSTICS_TAG: runtime-vnext-diagnostics-v2",
        "diagnostics-tag-mismatch",
    )
    _expect_policy_failure(
        "diagnostics-tag-mismatch",
        lambda: validate_workflow_set(diagnostics_tag_mismatch),
    )

    archive_sha_unbound = dict(texts)
    archive_sha_unbound["release-promote.yml"] = _replace_once(
        archive_sha_unbound["release-promote.yml"],
        '--expected-archive-sha256 "$PREPROMOTION_ARCHIVE_SHA256"',
        '--expected-archive-sha256 "$PREPROMOTION_CHILD_SHA256"',
        "diagnostics-archive-sha-unbound",
    )
    _expect_policy_failure(
        "diagnostics-archive-sha-unbound",
        lambda: validate_workflow_set(archive_sha_unbound),
    )

    child_sha_unbound = dict(texts)
    child_sha_unbound["release-promote.yml"] = _replace_once(
        child_sha_unbound["release-promote.yml"],
        '--expected-child-sha256 "$PREPROMOTION_CHILD_SHA256"',
        '--expected-child-sha256 "$PREPROMOTION_ARCHIVE_SHA256"',
        "diagnostics-child-sha-unbound",
    )
    _expect_policy_failure(
        "diagnostics-child-sha-unbound",
        lambda: validate_workflow_set(child_sha_unbound),
    )

    missing_complete_marker = dict(texts)
    missing_complete_marker["release-promote.yml"] = _replace_once(
        missing_complete_marker["release-promote.yml"],
        'gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${CONSUMPTION_COMPLETE_NAME}&per_page=100" > prior-complete-artifacts.json',
        'gh api "repos/${GITHUB_REPOSITORY}/actions/artifacts?name=${CONSUMPTION_CLAIM_NAME}&per_page=100" > prior-complete-artifacts.json',
        "promotion-complete-marker-missing",
    )
    _expect_policy_failure(
        "promotion-complete-marker-missing",
        lambda: validate_workflow_set(missing_complete_marker),
    )

    broad_promotion = dict(texts)
    broad_promotion["release-promote.yml"] = _replace_once(
        broad_promotion["release-promote.yml"],
        '-F prerelease=false > release-after.json',
        '-F prerelease=false -F draft=false > release-after.json',
        "promotion-mutates-more-than-prerelease",
    )
    _expect_policy_failure("promotion-mutates-more-than-prerelease", lambda: validate_workflow_set(broad_promotion))

    payload = _synthetic_prepromotion_payload()
    validate_prepromotion_payload(payload, release_candidate_sha="a" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345")
    consumed = copy.deepcopy(payload)
    consumed["consumption"]["state"] = "consumed"
    consumed["consumption"]["consumed_at"] = "2026-08-14T00:00:00Z"
    consumed["consumption"]["consumed_by"] = "fixture"
    _expect_policy_failure(
        "prepromotion-manifest-reuse",
        lambda: validate_prepromotion_payload(consumed, release_candidate_sha="a" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345"),
    )
    _expect_policy_failure(
        "prepromotion-release-sha-mismatch",
        lambda: validate_prepromotion_payload(payload, release_candidate_sha="d" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345"),
    )
    manifest_id_mismatch = copy.deepcopy(payload)
    manifest_id_mismatch["manifest_id"] = "f" * 64
    _expect_policy_failure(
        "prepromotion-manifest-id-mismatch",
        lambda: validate_prepromotion_payload(manifest_id_mismatch, release_candidate_sha="a" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345"),
    )
    dependency_missing = copy.deepcopy(payload)
    del dependency_missing["dependencies"]["homebrew_cuda_fetch"]
    _expect_policy_failure(
        "prepromotion-dependency-denominator-mismatch",
        lambda: validate_prepromotion_payload(dependency_missing, release_candidate_sha="a" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345"),
    )
    dependency_failed = copy.deepcopy(payload)
    dependency_failed["dependencies"]["published_assets"]["status"] = "fail"
    _expect_policy_failure(
        "prepromotion-dependency-status-mismatch",
        lambda: validate_prepromotion_payload(dependency_failed, release_candidate_sha="a" * 40, expected_tag=EXPECTED_TAG, expected_release_id="12345"),
    )
    print(SELFTEST_PASS_LINE)


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def write_policy_artifact(root: Path, out_dir: Path, paths: dict[str, Path], texts: dict[str, str]) -> None:
    run_selftest(texts)
    git_sha = _git_value(root, "rev-parse", "HEAD")
    git_tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    dirty = _git_value(root, "status", "--short")
    require(not dirty, "official workflow-policy artifact requires a clean source checkout")
    require(not out_dir.exists(), f"refusing to overwrite existing workflow-policy artifact: {out_dir}")
    out_dir.mkdir(parents=True)
    pass_line = f"{PASS_PREFIX}: {out_dir}"
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "lane": "runtime-vnext-release-workflow-policy",
        "version": EXPECTED_VERSION,
        "git_sha": git_sha,
        "git_tree": git_tree,
        "dirty": False,
        "created_at": now,
        "pass_line": pass_line,
        "negative_fixtures": {
            "direct_official_release": "rejected",
            "docker_tag_trigger": "rejected",
            "docker_publish_job": "rejected",
            "missing_prepromotion_child": "rejected",
            "release_candidate_sha_mismatch": "rejected",
            "release_candidate_tag_mismatch": "rejected",
            "diagnostics_tag_mismatch": "rejected",
            "diagnostics_archive_sha_unbound": "rejected",
            "diagnostics_child_sha_unbound": "rejected",
            "promotion_complete_marker_missing": "rejected",
            "promotion_mutates_more_than_prerelease": "rejected",
            "prepromotion_manifest_reuse": "rejected",
            "prepromotion_release_sha_mismatch": "rejected",
            "prepromotion_manifest_id_mismatch": "rejected",
            "prepromotion_dependency_denominator_mismatch": "rejected",
            "prepromotion_dependency_status_mismatch": "rejected",
        },
        "workflows": {
            name: {
                "path": str(paths[name].relative_to(root)),
                "sha256": sha256_bytes(texts[name].encode("utf-8")),
            }
            for name in sorted(paths)
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / "release_workflow_policy.manifest.json").write_text(encoded, encoding="utf-8")
    (out_dir / "gate.manifest.json").write_text(encoded, encoding="utf-8")
    print(pass_line)


def default_paths(root: Path) -> dict[str, Path]:
    workflow_root = root / ".github" / "workflows"
    return {
        "release.yml": workflow_root / "release.yml",
        "release-cuda.yml": workflow_root / "release-cuda.yml",
        "docker.yml": workflow_root / "docker.yml",
        "release-promote.yml": workflow_root / "release-promote.yml",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="validate workflows and all required negative fixtures")
    parser.add_argument("--out", type=Path, help="write the official clean-checkout workflow-policy artifact")
    parser.add_argument("--validate-prepromotion-manifest", type=Path, metavar="PATH")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--release-candidate-sha")
    parser.add_argument("--expected-tag")
    parser.add_argument("--expected-release-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = default_paths(root)
    modes = int(args.self_test) + int(args.out is not None) + int(args.validate_prepromotion_manifest is not None)
    require(modes == 1, "select exactly one of --self-test, --out, or --validate-prepromotion-manifest")
    if args.validate_prepromotion_manifest is not None:
        for name in ("expected_manifest_sha256", "release_candidate_sha", "expected_tag", "expected_release_id"):
            require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required with --validate-prepromotion-manifest")
        validate_prepromotion_manifest_file(
            args.validate_prepromotion_manifest.resolve(),
            expected_manifest_sha256=args.expected_manifest_sha256,
            release_candidate_sha=args.release_candidate_sha,
            expected_tag=args.expected_tag,
            expected_release_id=args.expected_release_id,
        )
        print(f"{PREPROMOTION_READY_PREFIX}: {args.validate_prepromotion_manifest.resolve()}")
        return 0
    texts = {}
    for name, path in paths.items():
        require(path.is_file(), f"required workflow is missing: {path}")
        texts[name] = path.read_text(encoding="utf-8")
    if args.self_test:
        run_selftest(texts)
        return 0
    write_policy_artifact(root, args.out.resolve(), paths, texts)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PolicyError as exc:
        print(f"release workflow policy error: {exc}", file=sys.stderr)
        raise SystemExit(1)
