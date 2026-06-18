#!/usr/bin/env python3
"""Generate L0 chat-template golden fixtures from HF transformers.

For each model id, downloads the tokenizer (config files only, no weights),
renders a fixed set of conversation cases with
`tokenizer.apply_chat_template(...)`, and writes fixtures under
`crates/ferrum-server/tests/fixtures/chat_template/<slug>/`:

  template.jinja   the chat template ferrum will render
  meta.json        bos/eos strings + render kwargs + provenance
  cases.json       the exact messages/tools per case (single source of truth
                   shared with the Rust test)
  golden_<case>.txt  the transformers-rendered prompt (byte ground truth)
  generation_config.json / tokenizer_config.json / tokenizer.json
                  HF sidecars useful for release-grade L0 token provenance
  tokenizer_special_tokens.json
                  compact AutoTokenizer special-token ids for checked-in gates

Run (network required; no torch needed):
  uv run --with transformers --with jinja2 --with huggingface-hub \
    python scripts/gen_chat_template_goldens.py [model_id ...]
  # Add `--with socksio` as well when HF traffic goes through a SOCKS proxy.

The Rust side (`crates/ferrum-server/tests/chat_template_golden.rs`) renders
the same cases through ferrum's renderer and asserts byte equality.
"""

import json
import re
import shutil
import sys
from pathlib import Path

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / (
    "crates/ferrum-server/tests/fixtures/chat_template"
)

HF_SIDECARS = ["generation_config.json", "tokenizer_config.json"]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

CASES = {
    "single": {"messages": [{"role": "user", "content": "Hi"}]},
    "system": {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
    },
    "multi_turn": {
        "messages": [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
    },
    "think_history": {
        "messages": [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": "<think>\nreasoning here\n</think>\n\nAnswer1",
            },
            {"role": "user", "content": "Q2"},
        ]
    },
    "tools": {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
    },
}


def template_supports_tools(template: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])tools(?![A-Za-z0-9_])", template) is not None


def copy_hf_sidecars(model_id: str, out_dir: Path) -> dict[str, dict[str, object]]:
    from huggingface_hub import hf_hub_download

    sidecars: dict[str, dict[str, object]] = {}
    for filename in HF_SIDECARS:
        target = out_dir / filename
        try:
            cached = Path(hf_hub_download(repo_id=model_id, filename=filename))
            shutil.copyfile(cached, target)
            sidecars[filename] = {
                "status": "copied",
                "path": str(target),
                "size_bytes": target.stat().st_size,
            }
            print(f"   sidecar {filename} ({target.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001 - record missing/remote errors in meta
            sidecars[filename] = {
                "status": "missing_or_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"   !! sidecar {filename} failed: {exc}")
    return sidecars


def write_tokenizer_special_tokens(tok, out_dir: Path) -> dict[str, object]:
    target = out_dir / "tokenizer_special_tokens.json"
    data = {
        "source": "AutoTokenizer",
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "unk_token": tok.unk_token,
        "unk_token_id": tok.unk_token_id,
        "additional_special_tokens": getattr(tok, "additional_special_tokens", []),
        "additional_special_tokens_ids": getattr(tok, "additional_special_tokens_ids", []),
        "all_special_tokens": getattr(tok, "all_special_tokens", []),
        "all_special_ids": getattr(tok, "all_special_ids", []),
        "special_tokens_map": getattr(tok, "special_tokens_map", {}),
    }
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"   sidecar {target.name} ({target.stat().st_size} bytes)")
    return {
        "status": "generated",
        "path": str(target),
        "size_bytes": target.stat().st_size,
    }


def pin_strftime_now() -> str:
    """Freeze the `strftime_now` clock transformers exposes to templates.

    Mistral-Small-3.2 / Llama-3.x date their system prompts; goldens only
    stay byte-reproducible if generation and the Rust test agree on "now".
    The pinned value is recorded in meta.json and replayed via
    `ChatTemplateOptions::now_override`.
    """
    import datetime as _dt

    import transformers.utils.chat_template_utils as _ctu

    pinned = _dt.datetime(2026, 6, 12, 0, 0, 0)

    class _FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003 - signature parity
            return pinned

    # `strftime_now` is a closure resolving `datetime` from this module's
    # globals at call time, so patching the module attribute is enough.
    _ctu.datetime = _FixedDatetime
    return pinned.strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    from transformers import AutoTokenizer

    pinned_now = pin_strftime_now()
    models = sys.argv[1:] or DEFAULT_MODELS
    for model_id in models:
        slug = model_id.replace("/", "__")
        out_dir = FIXTURE_ROOT / slug
        print(f"== {model_id} -> {out_dir}")
        tok = AutoTokenizer.from_pretrained(model_id)

        template = tok.chat_template
        if template is None:
            print("   !! no chat_template; skipping")
            continue
        if isinstance(template, (list, dict)):
            entries = (
                {e["name"]: e["template"] for e in template}
                if isinstance(template, list)
                else template
            )
            template = entries.get("default") or next(iter(entries.values()))

        kwargs = {}
        if "enable_thinking" in template:
            # ferrum defaults enable_thinking=false for templates that
            # support it (ChatTemplateOptions::default_for_template).
            kwargs["enable_thinking"] = False

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "template.jinja").write_text(template, encoding="utf-8")
        sidecars = copy_hf_sidecars(model_id, out_dir)
        sidecars["tokenizer_special_tokens.json"] = write_tokenizer_special_tokens(tok, out_dir)

        cases_out = {}
        for name, case in CASES.items():
            tools = case.get("tools")
            if tools and not template_supports_tools(template):
                # tools-unaware templates take ferrum's injection path,
                # which has no transformers ground truth — covered by unit
                # tests instead.
                continue
            try:
                rendered = tok.apply_chat_template(
                    case["messages"],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001 - record and continue
                print(f"   !! case {name} failed: {e}")
                continue
            (out_dir / f"golden_{name}.txt").write_text(rendered, encoding="utf-8")
            cases_out[name] = case
            print(f"   ok {name} ({len(rendered)} chars)")

        meta = {
            "model_id": model_id,
            "bos_token": tok.bos_token,
            "eos_token": tok.eos_token,
            "render_kwargs": kwargs,
            "now": pinned_now,
            "transformers_version": __import__("transformers").__version__,
            "sidecars": sidecars,
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out_dir / "cases.json").write_text(
            json.dumps(cases_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
