#!/usr/bin/env python3
"""Static guard for Milestone E runtime-snapshot ownership boundaries.

The product-server startup path resolves a typed RuntimeConfigSnapshot and
publishes it in effective_config.json / decision_trace.jsonl. Model startup
must consume that resolved snapshot where available instead of independently
reconstructing M3 defaults from process env.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile


REQUIRED_TOKENS = {
    "crates/ferrum-types/src/config.rs": [
        "pub const RUNTIME_CONFIG_SNAPSHOT_BACKEND_OPTION",
        "self.backend.backend_options.insert(",
        "serde_json::to_value(snapshot)",
    ],
    "crates/ferrum-engine/src/registry.rs": [
        "RUNTIME_CONFIG_SNAPSHOT_BACKEND_OPTION",
        "pub fn runtime_config_snapshot(&self) -> Option<RuntimeConfigSnapshot>",
        "let runtime_config_snapshot = config.runtime_config_snapshot();",
        "runtime_config_snapshot.as_ref()",
    ],
    "crates/ferrum-models/src/models/qwen3_moe_runtime.rs": [
        "use ferrum_types::RuntimeConfigSnapshot;",
        "pub(crate) fn from_runtime_config_snapshot(snapshot: &RuntimeConfigSnapshot) -> Self",
        "Self::from_env_vars(",
        "fn qwen3_moe_runtime_env_can_use_typed_snapshot_without_process_env()",
    ],
    "crates/ferrum-models/src/models/qwen3_moe/load.rs": [
        "pub fn new_safetensors_with_runtime_config_snapshot(",
        "runtime_config: &ferrum_types::RuntimeConfigSnapshot",
        "Qwen3MoeRuntimeEnv::from_runtime_config_snapshot(runtime_config)",
        "fn new_safetensors_with_runtime_env(",
    ],
}


FORBIDDEN_TOKENS = {
    "crates/ferrum-engine/src/registry.rs": [
        "Qwen3MoeModel::<B, K>::new_safetensors(mc, &weight_loader)?;\n        ))",
    ],
}


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AssertionError(f"missing file: {path}") from None


def check_root(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    for rel, tokens in REQUIRED_TOKENS.items():
        text = read(root / rel)
        for token in tokens:
            if token not in text:
                errors.append(f"{rel}: missing required token {token!r}")
    for rel, tokens in FORBIDDEN_TOKENS.items():
        text = read(root / rel)
        for token in tokens:
            if token in text:
                errors.append(f"{rel}: forbidden legacy token present {token!r}")
    return errors


def write_fixture(root: pathlib.Path, *, valid: bool) -> None:
    for rel in REQUIRED_TOKENS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = list(REQUIRED_TOKENS[rel])
        if not valid and rel.endswith("qwen3_moe/load.rs"):
            tokens = [token for token in tokens if "new_safetensors_with_runtime_config_snapshot" not in token]
        path.write_text("\n".join(tokens), encoding="utf-8")


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        write_fixture(root, valid=True)
        errors = check_root(root)
        if errors:
            print("valid fixture failed:", errors, file=sys.stderr)
            return 1
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        write_fixture(root, valid=False)
        errors = check_root(root)
        if not errors:
            print("invalid fixture unexpectedly passed", file=sys.stderr)
            return 1
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    root = pathlib.Path(args.root).resolve()
    errors = check_root(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
