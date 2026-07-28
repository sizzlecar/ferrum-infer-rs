#!/usr/bin/env python3
"""Validate native device-work attribution in a Ferrum profile JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PASS_PREFIX = "FERRUM NATIVE WORK ATTRIBUTION PASS"
SELFTEST_PASS_LINE = "FERRUM NATIVE WORK ATTRIBUTION SELFTEST PASS"
NATIVE_WORK_PHASE = "vnext.device_native_work"


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpectedWork:
    operation_id: str
    batching_form: str
    native_op_id: str | None
    min_participants: int
    exact_compute_dispatches: int | None
    compute_dispatch_base: int | None
    compute_dispatches_per_participant: int | None
    exact_transfer_commands: int | None
    min_matching_events: int
    require_all_eligible: bool


def require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} must be an object")
    return value


def require_non_negative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{context} must be a non-negative integer")
    return value


def require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context} must be a non-empty string")
    return value


def is_hex_digest(value: Any, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def event_topology(
    event: dict[str, Any],
    line_number: int,
) -> tuple[str, str, int, int, int, int]:
    context = f"line {line_number}"
    attributes = require_object(event.get("attributes"), f"{context}.attributes")
    shape = require_object(event.get("shape"), f"{context}.shape")
    native_op_id = require_non_empty_string(
        attributes.get("native_op_id"),
        f"{context}.attributes.native_op_id",
    )
    batching_form = require_non_empty_string(
        attributes.get("batching_form"),
        f"{context}.attributes.batching_form",
    )
    participant_count = require_non_negative_int(
        shape.get("participant_count"),
        f"{context}.shape.participant_count",
    )
    token_count = require_non_negative_int(
        shape.get("token_count"),
        f"{context}.shape.token_count",
    )
    compute_dispatches = require_non_negative_int(
        shape.get("physical_compute_dispatch_count"),
        f"{context}.shape.physical_compute_dispatch_count",
    )
    transfer_commands = require_non_negative_int(
        shape.get("physical_transfer_command_count"),
        f"{context}.shape.physical_transfer_command_count",
    )
    return (
        native_op_id,
        batching_form,
        participant_count,
        token_count,
        compute_dispatches,
        transfer_commands,
    )


def mismatch_reasons(
    topology: tuple[str, str, int, int, int, int],
    expected: ExpectedWork,
) -> list[str]:
    (
        native_op_id,
        batching_form,
        participant_count,
        _token_count,
        compute_dispatches,
        transfer_commands,
    ) = topology
    reasons: list[str] = []
    if batching_form != expected.batching_form:
        reasons.append(f"batching_form={batching_form}")
    if expected.native_op_id is not None and native_op_id != expected.native_op_id:
        reasons.append(f"native_op_id={native_op_id}")
    if (
        expected.exact_compute_dispatches is not None
        and compute_dispatches != expected.exact_compute_dispatches
    ):
        reasons.append(f"compute_dispatches={compute_dispatches}")
    if expected.compute_dispatch_base is not None:
        assert expected.compute_dispatches_per_participant is not None
        expected_dispatches = (
            expected.compute_dispatch_base
            + expected.compute_dispatches_per_participant * participant_count
        )
        if compute_dispatches != expected_dispatches:
            reasons.append(
                f"compute_dispatches={compute_dispatches}"
                f"(expected={expected_dispatches},participants={participant_count})"
            )
    if (
        expected.exact_transfer_commands is not None
        and transfer_commands != expected.exact_transfer_commands
    ):
        reasons.append(f"transfer_commands={transfer_commands}")
    return reasons


def validate_profile(path: Path, expected: ExpectedWork) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"profile JSONL does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValidationError(f"profile JSONL is empty: {path}")

    total_events = 0
    native_work_events = 0
    operation_events = 0
    eligible_events = 0
    matching_events = 0
    mismatch_counts: Counter[str] = Counter()
    topology_counts: Counter[tuple[str, str, int, int, int]] = Counter()
    matching_samples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            total_events += 1
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            event = require_object(event, f"line {line_number}")
            if event.get("phase") != NATIVE_WORK_PHASE:
                continue
            native_work_events += 1
            attributes = require_object(event.get("attributes"), f"line {line_number}.attributes")
            if attributes.get("operation_id") != expected.operation_id:
                continue
            operation_events += 1
            topology = event_topology(event, line_number)
            (
                native_op_id,
                batching_form,
                participant_count,
                token_count,
                compute_dispatches,
                transfer_commands,
            ) = topology
            if expected.native_op_id is not None and native_op_id != expected.native_op_id:
                continue
            if participant_count < expected.min_participants:
                continue
            eligible_events += 1
            topology_counts[
                (
                    native_op_id,
                    batching_form,
                    participant_count,
                    compute_dispatches,
                    transfer_commands,
                )
            ] += 1
            reasons = mismatch_reasons(topology, expected)
            if reasons:
                mismatch_counts.update(reasons)
                continue
            matching_events += 1
            if len(matching_samples) < 8:
                matching_samples.append(
                    {
                        "event_id": event.get("event_id"),
                        "correlation_id": event.get("correlation_id"),
                        "native_op_id": native_op_id,
                        "batching_form": batching_form,
                        "participant_count": participant_count,
                        "token_count": token_count,
                        "physical_compute_dispatch_count": compute_dispatches,
                        "physical_transfer_command_count": transfer_commands,
                    }
                )

    if total_events == 0:
        raise ValidationError(f"profile JSONL has no JSON events: {path}")
    if operation_events == 0:
        raise ValidationError(
            f"profile has no {NATIVE_WORK_PHASE} event for {expected.operation_id}"
        )
    if eligible_events == 0:
        native_label = expected.native_op_id or "<any>"
        raise ValidationError(
            "profile has no eligible native-work event for "
            f"operation={expected.operation_id}, native_op={native_label}, "
            f"participants>={expected.min_participants}"
        )
    if matching_events < expected.min_matching_events:
        raise ValidationError(
            f"matching event count {matching_events} is below "
            f"{expected.min_matching_events}; mismatches={dict(mismatch_counts)}"
        )
    if expected.require_all_eligible and matching_events != eligible_events:
        raise ValidationError(
            f"{eligible_events - matching_events}/{eligible_events} eligible events violate "
            f"the expected topology; mismatches={dict(mismatch_counts)}"
        )

    observed_topologies = [
        {
            "native_op_id": topology[0],
            "batching_form": topology[1],
            "participant_count": topology[2],
            "physical_compute_dispatch_count": topology[3],
            "physical_transfer_command_count": topology[4],
            "event_count": count,
        }
        for topology, count in sorted(topology_counts.items())
    ]
    return {
        "schema_version": 1,
        "status": "pass",
        "profile_jsonl": str(path.resolve()),
        "constraints": {
            "operation_id": expected.operation_id,
            "native_op_id": expected.native_op_id,
            "batching_form": expected.batching_form,
            "min_participants": expected.min_participants,
            "exact_compute_dispatches": expected.exact_compute_dispatches,
            "compute_dispatch_base": expected.compute_dispatch_base,
            "compute_dispatches_per_participant": (
                expected.compute_dispatches_per_participant
            ),
            "exact_transfer_commands": expected.exact_transfer_commands,
            "min_matching_events": expected.min_matching_events,
            "require_all_eligible": expected.require_all_eligible,
        },
        "counts": {
            "total_events": total_events,
            "native_work_events": native_work_events,
            "operation_events": operation_events,
            "eligible_events": eligible_events,
            "matching_events": matching_events,
        },
        "observed_topologies": observed_topologies,
        "matching_samples": matching_samples,
    }


def synthetic_event(
    *,
    operation_id: str,
    native_op_id: str,
    batching_form: str,
    participants: int,
    compute_dispatches: int,
    transfer_commands: int,
) -> dict[str, Any]:
    return {
        "phase": NATIVE_WORK_PHASE,
        "event_id": "evt-selftest",
        "correlation_id": "corr-selftest",
        "attributes": {
            "operation_id": operation_id,
            "native_op_id": native_op_id,
            "batching_form": batching_form,
        },
        "shape": {
            "participant_count": participants,
            "token_count": participants,
            "physical_compute_dispatch_count": compute_dispatches,
            "physical_transfer_command_count": transfer_commands,
        },
    }


def run_selftest() -> None:
    expected = ExpectedWork(
        operation_id="operation.gated_delta_recurrent_attention",
        native_op_id="vnext_gated_delta_recurrent_attention",
        batching_form="packed",
        min_participants=2,
        exact_compute_dispatches=10,
        compute_dispatch_base=None,
        compute_dispatches_per_participant=None,
        exact_transfer_commands=2,
        min_matching_events=1,
        require_all_eligible=True,
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-native-work-") as raw_dir:
        root = Path(raw_dir)
        passing = root / "passing.jsonl"
        passing_events = [
            {"phase": "request", "event_id": "unrelated"},
            synthetic_event(
                operation_id=expected.operation_id,
                native_op_id="vnext_gated_delta_recurrent_attention_bindings",
                batching_form="participant_loop",
                participants=4,
                compute_dispatches=0,
                transfer_commands=4,
            ),
            synthetic_event(
                operation_id=expected.operation_id,
                native_op_id=expected.native_op_id or "",
                batching_form="packed",
                participants=4,
                compute_dispatches=10,
                transfer_commands=2,
            ),
        ]
        passing.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in passing_events),
            encoding="utf-8",
        )
        summary = validate_profile(passing, expected)
        if summary["counts"]["matching_events"] != 1:
            raise ValidationError("self-test passing fixture did not produce one match")

        failing = root / "failing.jsonl"
        failing.write_text(
            json.dumps(
                synthetic_event(
                    operation_id=expected.operation_id,
                    native_op_id=expected.native_op_id or "",
                    batching_form="participant_loop",
                    participants=4,
                    compute_dispatches=40,
                    transfer_commands=8,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            validate_profile(failing, expected)
        except ValidationError:
            pass
        else:
            raise ValidationError("self-test failing fixture unexpectedly passed")

        affine = ExpectedWork(
            operation_id="operation.causal_paged_attention",
            native_op_id="vnext.causal_attention.vllm_paged_attention_v1_addressed",
            batching_form="packed",
            min_participants=2,
            exact_compute_dispatches=None,
            compute_dispatch_base=6,
            compute_dispatches_per_participant=3,
            exact_transfer_commands=0,
            min_matching_events=2,
            require_all_eligible=True,
        )
        affine_profile = root / "affine.jsonl"
        affine_profile.write_text(
            "".join(
                json.dumps(event, sort_keys=True) + "\n"
                for event in [
                    synthetic_event(
                        operation_id=affine.operation_id,
                        native_op_id=affine.native_op_id or "",
                        batching_form="packed",
                        participants=4,
                        compute_dispatches=18,
                        transfer_commands=0,
                    ),
                    synthetic_event(
                        operation_id=affine.operation_id,
                        native_op_id=affine.native_op_id or "",
                        batching_form="packed",
                        participants=32,
                        compute_dispatches=102,
                        transfer_commands=0,
                    ),
                ]
            ),
            encoding="utf-8",
        )
        affine_summary = validate_profile(affine_profile, affine)
        if affine_summary["counts"]["matching_events"] != 2:
            raise ValidationError("self-test affine fixture did not produce two matches")

        affine_bad = root / "affine-bad.jsonl"
        affine_bad.write_text(
            json.dumps(
                synthetic_event(
                    operation_id=affine.operation_id,
                    native_op_id=affine.native_op_id or "",
                    batching_form="packed",
                    participants=4,
                    compute_dispatches=19,
                    transfer_commands=0,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            validate_profile(affine_bad, affine)
        except ValidationError:
            pass
        else:
            raise ValidationError("self-test invalid affine fixture unexpectedly passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-jsonl", type=Path)
    parser.add_argument("--operation-id")
    parser.add_argument("--native-op-id")
    parser.add_argument("--batching-form")
    parser.add_argument("--min-participants", type=int, default=2)
    parser.add_argument("--exact-compute-dispatches", type=int)
    parser.add_argument("--compute-dispatch-base", type=int)
    parser.add_argument("--compute-dispatches-per-participant", type=int)
    parser.add_argument("--exact-transfer-commands", type=int)
    parser.add_argument("--min-matching-events", type=int, default=1)
    parser.add_argument("--require-all-eligible", action="store_true")
    parser.add_argument("--source-git-sha")
    parser.add_argument("--binary-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        if args.profile_jsonl is None:
            raise ValidationError("--profile-jsonl is required")
        if not args.operation_id:
            raise ValidationError("--operation-id is required")
        if not args.batching_form:
            raise ValidationError("--batching-form is required")
        if args.out is None:
            raise ValidationError("--out is required")
        if not is_hex_digest(args.source_git_sha, {40, 64}):
            raise ValidationError("--source-git-sha must be a 40- or 64-character hex digest")
        if not is_hex_digest(args.binary_sha256, {64}):
            raise ValidationError("--binary-sha256 must be a 64-character hex digest")
        for label in (
            "min_participants",
            "min_matching_events",
            "exact_compute_dispatches",
            "compute_dispatch_base",
            "compute_dispatches_per_participant",
            "exact_transfer_commands",
        ):
            value = getattr(args, label)
            if value is not None and value < 0:
                raise ValidationError(f"--{label.replace('_', '-')} must be non-negative")
        if args.min_participants == 0:
            raise ValidationError("--min-participants must be at least 1")
        if args.min_matching_events == 0:
            raise ValidationError("--min-matching-events must be at least 1")
        affine_dispatch = (
            args.compute_dispatch_base is not None
            or args.compute_dispatches_per_participant is not None
        )
        if affine_dispatch and (
            args.compute_dispatch_base is None
            or args.compute_dispatches_per_participant is None
        ):
            raise ValidationError(
                "--compute-dispatch-base and "
                "--compute-dispatches-per-participant must be provided together"
            )
        if affine_dispatch and args.exact_compute_dispatches is not None:
            raise ValidationError(
                "--exact-compute-dispatches cannot be combined with affine dispatch constraints"
            )

        expected = ExpectedWork(
            operation_id=args.operation_id,
            native_op_id=args.native_op_id,
            batching_form=args.batching_form,
            min_participants=args.min_participants,
            exact_compute_dispatches=args.exact_compute_dispatches,
            compute_dispatch_base=args.compute_dispatch_base,
            compute_dispatches_per_participant=(
                args.compute_dispatches_per_participant
            ),
            exact_transfer_commands=args.exact_transfer_commands,
            min_matching_events=args.min_matching_events,
            require_all_eligible=args.require_all_eligible,
        )
        summary = validate_profile(args.profile_jsonl, expected)
        summary["source_git_sha"] = args.source_git_sha.lower()
        summary["binary_sha256"] = args.binary_sha256.lower()
        summary["command_line"] = sys.argv
        args.out.mkdir(parents=True, exist_ok=True)
        summary_path = args.out / "native-work-attribution.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{PASS_PREFIX}: {args.out}")
        return 0
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
