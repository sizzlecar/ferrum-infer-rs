#!/usr/bin/env python3
"""Versioned C13 tool-result continuation contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


CONTRACT_ID = "c13-tool-result-continuation-v3"
SAMPLING_CONTRACT_ID = "c13-deterministic-semantic-correctness-v1"
CASE_COUNT = 60
TOOL_NAME = "calculator"
VALID_VARIANTS = frozenset({"tool-result", "soft-think", "soft-no-think"})
DETERMINISTIC_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 9271,
    "stop": [],
}


class C13ContractError(ValueError):
    """Raised when a request or response violates the C13 contract."""


@dataclass(frozen=True)
class C13CaseContract:
    ordinal: int
    variant: str
    expression: str
    expected_result: str
    expected_receipt: str
    tool_call_id: str

    @property
    def case_id(self) -> str:
        return f"c13-{self.ordinal:03d}"

    @property
    def user_prompt(self) -> str:
        prompt = (
            f"Calculate {self.expression} using the calculator. "
            "After the calculator result is provided, your final answer must copy "
            "both values from that result: the calculation result and the complete "
            "opaque receipt. Do not omit or shorten the receipt."
        )
        if self.variant == "soft-think":
            return f"{prompt} /think"
        if self.variant == "soft-no-think":
            return f"{prompt} /no_think"
        return prompt

    def messages(self) -> list[dict[str, Any]]:
        arguments = json.dumps(
            {"expression": self.expression},
            separators=(",", ":"),
        )
        return [
            {"role": "user", "content": self.user_prompt},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": self.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": json.dumps(
                    {
                        "receipt": self.expected_receipt,
                        "result": self.expected_result,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        ]

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": (
                        "Evaluate one arithmetic expression and return a "
                        "result with an opaque receipt"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def evidence(self) -> dict[str, Any]:
        messages = self.messages()
        tools = self.tools()
        return {
            "contract_id": CONTRACT_ID,
            "c13_variant": self.variant,
            "expression": self.expression,
            "expected_tool_result": self.expected_result,
            "expected_tool_receipt": self.expected_receipt,
            "tool_call_id": self.tool_call_id,
            "c13_prompt_sha256": hashlib.sha256(
                self.user_prompt.encode("utf-8")
            ).hexdigest(),
            "c13_messages_sha256": hashlib.sha256(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "c13_tools_sha256": hashlib.sha256(
                json.dumps(
                    tools,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }


def case_contract(ordinal: int, *, variant: str) -> C13CaseContract:
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 1 <= ordinal <= CASE_COUNT
    ):
        raise C13ContractError(
            f"C13 ordinal must be in 1..{CASE_COUNT}, observed {ordinal!r}"
        )
    if variant not in VALID_VARIANTS:
        raise C13ContractError(
            f"C13 variant must be one of {sorted(VALID_VARIANTS)}, "
            f"observed {variant!r}"
        )

    group = (ordinal - 1) // 4 + 1
    operation = (ordinal - 1) % 4
    if operation == 0:
        left = 11 + group
        right = 20 + 2 * group
        expression = f"{left} + {right}"
        result = left + right
    elif operation == 1:
        left = 220 + 3 * group
        right = 60 + group
        expression = f"{left} - {right}"
        result = left - right
    elif operation == 2:
        left = 40 + group
        right = 7
        expression = f"{left} * {right}"
        result = left * right
    else:
        quotient = 400 + group
        divisor = 3 + group % 3
        dividend = quotient * divisor
        expression = f"{dividend} / {divisor}"
        result = quotient

    receipt = hashlib.sha256(
        (
            f"{CONTRACT_ID}\0{ordinal}\0{expression}\0"
            f"{result}\0tool-receipt"
        ).encode("utf-8")
    ).hexdigest()[:24]
    return C13CaseContract(
        ordinal=ordinal,
        variant=variant,
        expression=expression,
        expected_result=str(result),
        expected_receipt=f"calc-receipt-{receipt}",
        tool_call_id=f"call-c13-{ordinal:03d}",
    )


def validate_request(
    request: dict[str, Any],
    *,
    ordinal: int,
    variant: str,
) -> C13CaseContract:
    contract = case_contract(ordinal, variant=variant)
    metadata = request.get("metadata")
    if not isinstance(metadata, dict):
        raise C13ContractError("C13 request metadata must be an object")
    expected_metadata = {
        "g00_case_id": contract.case_id,
        "g00_scenario_id": "C13",
        "g00_ordinal": ordinal,
        "g00_contract_id": CONTRACT_ID,
        "g00_sampling_contract": SAMPLING_CONTRACT_ID,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise C13ContractError(
                f"C13 request metadata {key} mismatch: "
                f"expected {expected!r}, observed {metadata.get(key)!r}"
            )
    observed_variant = metadata.get("g00_variant")
    if observed_variant != variant:
        raise C13ContractError(
            f"C13 request metadata g00_variant mismatch: "
            f"expected {variant!r}, observed {observed_variant!r}"
        )
    if request.get("messages") != contract.messages():
        raise C13ContractError(
            f"{contract.case_id} messages differ from {CONTRACT_ID}"
        )
    if request.get("tools") != contract.tools():
        raise C13ContractError(
            f"{contract.case_id} tools differ from {CONTRACT_ID}"
        )
    if request.get("tool_choice") != "auto":
        raise C13ContractError(
            f"{contract.case_id} tool_choice must be 'auto'"
        )
    for key, expected in DETERMINISTIC_SAMPLING.items():
        observed = request.get(key)
        if isinstance(observed, bool) or observed != expected:
            raise C13ContractError(
                f"{contract.case_id} deterministic sampling {key} mismatch: "
                f"expected {expected!r}, observed {observed!r}"
            )
    frequency_penalty = request.get("frequency_penalty", 0.0)
    if isinstance(frequency_penalty, bool) or frequency_penalty != 0.0:
        raise C13ContractError(
            f"{contract.case_id} deterministic sampling frequency_penalty "
            f"must be zero, observed {frequency_penalty!r}"
        )
    return contract


def request_evidence(
    request: dict[str, Any],
    *,
    contract: C13CaseContract,
) -> dict[str, Any]:
    return {
        **contract.evidence(),
        "sampling_contract_id": SAMPLING_CONTRACT_ID,
        "c13_request_sha256": hashlib.sha256(
            json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def validate_response(
    message: dict[str, Any],
    *,
    finish_reason: Any,
    contract: C13CaseContract,
) -> None:
    if finish_reason not in {"stop", "eos"}:
        raise C13ContractError(
            f"{contract.case_id} finish_reason must be stop/eos, "
            f"observed {finish_reason!r}"
        )
    if message.get("tool_calls"):
        raise C13ContractError(
            f"{contract.case_id} repeated a tool call after receiving its result"
        )
    if message.get("role") != "assistant":
        raise C13ContractError(
            f"{contract.case_id} final message role must be assistant, "
            f"observed {message.get('role')!r}"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise C13ContractError(
            f"{contract.case_id} final answer is empty"
        )
    history_markers = (
        contract.user_prompt,
        contract.tool_call_id,
        f'"expression":"{contract.expression}"',
        '"tool_call_id"',
        '"tool_calls"',
        "<|im_start|>",
        "<|im_end|>",
    )
    if any(marker in content for marker in history_markers):
        raise C13ContractError(
            f"{contract.case_id} final answer polluted by serialized history"
        )
    result_pattern = re.compile(
        rf"(?<![0-9.]){re.escape(contract.expected_result)}"
        rf"(?![0-9]|[.,][0-9])"
    )
    if result_pattern.search(content) is None:
        raise C13ContractError(
            f"{contract.case_id} did not incorporate tool result "
            f"{contract.expected_result}"
        )
    receipt_pattern = re.compile(
        rf"(?<![A-Za-z0-9-]){re.escape(contract.expected_receipt)}"
        rf"(?![A-Za-z0-9-])"
    )
    if receipt_pattern.search(content) is None:
        raise C13ContractError(
            f"{contract.case_id} did not incorporate tool receipt "
            f"{contract.expected_receipt}"
        )


def self_test() -> None:
    contracts = [
        case_contract(
            ordinal,
            variant=(
                "tool-result"
                if ordinal <= 40
                else "soft-think"
                if ordinal <= 50
                else "soft-no-think"
            ),
        )
        for ordinal in range(1, CASE_COUNT + 1)
    ]
    assert len({contract.user_prompt for contract in contracts}) == CASE_COUNT
    assert len({contract.expression for contract in contracts}) == CASE_COUNT
    assert len({contract.expected_result for contract in contracts}) == CASE_COUNT
    assert len({contract.expected_receipt for contract in contracts}) == CASE_COUNT
    assert len({contract.tool_call_id for contract in contracts}) == CASE_COUNT
    for contract in contracts:
        history_without_tool_result = json.dumps(
            contract.messages()[:2],
            sort_keys=True,
        )
        assert contract.expected_receipt not in history_without_tool_result
        assert contract.expected_receipt not in contract.user_prompt
        assert "complete opaque receipt" in contract.user_prompt
        assert "Do not omit or shorten the receipt." in contract.user_prompt
    for invalid_ordinal in (0, CASE_COUNT + 1, True, "1"):
        try:
            case_contract(  # type: ignore[arg-type]
                invalid_ordinal,
                variant="tool-result",
            )
        except C13ContractError:
            pass
        else:
            raise AssertionError(
                f"C13 invalid ordinal was accepted: {invalid_ordinal!r}"
            )
    for variant, suffix in (
        ("tool-result", "receipt."),
        ("soft-think", "/think"),
        ("soft-no-think", "/no_think"),
    ):
        contract = case_contract(1, variant=variant)
        assert contract.user_prompt.endswith(suffix)

    sample = contracts[21]
    request = {
        "metadata": {
            "g00_case_id": sample.case_id,
            "g00_scenario_id": "C13",
            "g00_variant": sample.variant,
            "g00_ordinal": sample.ordinal,
            "g00_contract_id": CONTRACT_ID,
            "g00_sampling_contract": SAMPLING_CONTRACT_ID,
        },
        "messages": sample.messages(),
        "tools": sample.tools(),
        "tool_choice": "auto",
        **DETERMINISTIC_SAMPLING,
    }
    assert (
        validate_request(
            request,
            ordinal=sample.ordinal,
            variant=sample.variant,
        )
        == sample
    )
    validate_response(
        {
            "role": "assistant",
            "content": (
                f"The result is {sample.expected_result}; "
                f"receipt {sample.expected_receipt}."
            ),
        },
        finish_reason="stop",
        contract=sample,
    )

    for field, invalid in (
        ("temperature", 1.0),
        ("presence_penalty", 1.5),
        ("top_k", 20),
        ("g00_sampling_contract", "official-stochastic"),
    ):
        invalid_request = json.loads(json.dumps(request))
        if field == "g00_sampling_contract":
            invalid_request["metadata"][field] = invalid
        else:
            invalid_request[field] = invalid
        try:
            validate_request(
                invalid_request,
                ordinal=sample.ordinal,
                variant=sample.variant,
            )
        except C13ContractError:
            pass
        else:
            raise AssertionError(
                f"C13 invalid sampling field was accepted: {field}"
            )

    for message, finish_reason, marker in (
        (
            {"role": "assistant", "content": "Please provide an expression."},
            "stop",
            "did not incorporate",
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "duplicate"}],
            },
            "stop",
            "repeated a tool call",
        ),
        (
            {
                "role": "tool",
                "content": (
                    f"The result is {sample.expected_result}; "
                    f"receipt {sample.expected_receipt}."
                ),
            },
            "stop",
            "role must be assistant",
        ),
        (
            {
                "role": "assistant",
                "content": (
                    f"The result is {sample.expected_result}.5; "
                    f"receipt {sample.expected_receipt}."
                ),
            },
            "stop",
            "did not incorporate",
        ),
        (
            {
                "role": "assistant",
                "content": (
                    f"{sample.user_prompt} "
                    f"The result is {sample.expected_result}; "
                    f"receipt {sample.expected_receipt}."
                ),
            },
            "stop",
            "polluted by serialized history",
        ),
        (
            {
                "role": "assistant",
                "content": f"The result is {sample.expected_result}.",
            },
            "stop",
            "did not incorporate tool receipt",
        ),
    ):
        try:
            validate_response(
                message,
                finish_reason=finish_reason,
                contract=sample,
            )
        except C13ContractError as error:
            assert marker in str(error)
        else:
            raise AssertionError("C13 negative response fixture was accepted")


if __name__ == "__main__":
    self_test()
    print("FERRUM RUNTIME VNEXT C13 CONTRACT SELFTEST PASS")
