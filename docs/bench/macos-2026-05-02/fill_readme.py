#!/usr/bin/env python3
"""Read all *.json cells in this directory, format the headline
benchmark numbers, and substitute them into:
  - ../../README.md
  - ../../README_zh.md
  - ./README.md (the bench report)

Each result file is named: <engine>__<model_label>__c<c>.json
Each contains keys output_throughput_tok_s, tpot_ms.{median,p99}, etc.

We replace the _BENCH_<TAG>_ placeholders with the right number.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

# Map (engine, model_label, c) -> placeholder tag
MODEL_TAG = {
    "Llama-3.1-8B": "LLAMA",
    "Qwen3-8B": "Q8",
    "Qwen3-30B-A3B": "MOE",
}
ENGINE_TAG = {
    "ferrum": "FERRUM",
    "llamacpp": "LCPP",
    "mistralrs": "MRS",
}


def read_cell(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return None
    return d


def fmt(d):
    if d is None:
        return "—"
    v = d.get("output_throughput_tok_s")
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.1f}"


def collect():
    """Returns: dict[(engine, model, c)] -> tok/s str"""
    out = {}
    for json_path in HERE.glob("*.json"):
        name = json_path.stem  # engine__model__c<n>
        parts = name.split("__")
        if len(parts) != 3:
            continue
        engine, model, c_str = parts
        if not c_str.startswith("c"):
            continue
        c = int(c_str[1:])
        d = read_cell(json_path)
        out[(engine, model, c)] = fmt(d)
    return out


def fill_file(path, results):
    if not path.exists():
        print(f"skip (missing): {path}")
        return
    text = path.read_text()
    pattern = re.compile(r"_BENCH_(LLAMA|Q8|MOE)_(FERRUM|LCPP|MRS)_C(\d+)_")

    def replace(m):
        model_tag, engine_tag, c = m.group(1), m.group(2), int(m.group(3))
        for (eng, mod, cc), val in results.items():
            if (
                ENGINE_TAG.get(eng) == engine_tag
                and MODEL_TAG.get(mod) == model_tag
                and cc == c
            ):
                return val
        return "—"

    new = pattern.sub(replace, text)
    if new == text:
        print(f"no change: {path}")
        return
    path.write_text(new)
    print(f"updated: {path}")


def fill_bench_report_grid(results):
    """Write the full grid into the bench report as a side-effect."""
    report = HERE / "README.md"
    if not report.exists():
        return
    text = report.read_text()

    # Build the headline c=16 table
    headline_rows = []
    for model_label in ["Llama-3.1-8B", "Qwen3-8B", "Qwen3-30B-A3B"]:
        f_v = results.get(("ferrum", model_label, 16), "—")
        l_v = results.get(("llamacpp", model_label, 16), "—")
        m_v = results.get(("mistralrs", model_label, 16), "—")
        # ratio
        ratio = "—"
        try:
            if f_v != "—" and l_v != "—":
                rf = float(f_v)
                rl = float(l_v)
                ratio = f"{rf / rl:.2f}×"
        except Exception:
            pass
        display = (
            "Qwen3-30B-A3B (MoE)" if model_label == "Qwen3-30B-A3B" else model_label
        )
        headline_rows.append(
            f"| {display} | {f_v} | {l_v} | {m_v} | {ratio} |"
        )
    headline_table = (
        "| Model | ferrum | llama.cpp | mistralrs | ferrum vs llama.cpp |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(headline_rows)
    )

    text = re.sub(
        r"\| Model \| ferrum \| llama\.cpp \| mistralrs \| ferrum vs llama\.cpp \|\n\|[^\n]+\n(\| [^\n]*?\(TBD\)[^\n]*\n)+",
        headline_table + "\n",
        text,
    )

    # Build the full grid
    grid_rows = []
    for model_label in ["Llama-3.1-8B", "Qwen3-8B", "Qwen3-30B-A3B"]:
        for engine in ["ferrum", "llamacpp", "mistralrs"]:
            row_vals = [
                results.get((engine, model_label, 1), "—"),
                results.get((engine, model_label, 4), "—"),
                results.get((engine, model_label, 8), "—"),
                results.get((engine, model_label, 16), "—"),
            ]
            display_engine = {
                "ferrum": "ferrum",
                "llamacpp": "llama.cpp",
                "mistralrs": "mistralrs",
            }[engine]
            display_model = (
                "Qwen3-30B-A3B" if model_label == "Qwen3-30B-A3B" else model_label
            )
            grid_rows.append(
                f"| {display_model} | {display_engine} | "
                + " | ".join(row_vals)
                + " |"
            )

    grid_table = (
        "| Model | Engine | c=1 | c=4 | c=8 | c=16 |\n"
        "|---|---|---:|---:|---:|---:|\n"
        + "\n".join(grid_rows)
    )

    text = re.sub(
        r"\| Model \| Engine \| c=1 \| c=4 \| c=8 \| c=16 \|\n\|[^\n]+\n(\| [^\n]*?\(TBD\)[^\n]*\n)+",
        grid_table + "\n",
        text,
    )

    # Build the TPOT c=16 table
    tpot_rows = []
    for model_label in ["Llama-3.1-8B", "Qwen3-8B", "Qwen3-30B-A3B"]:
        cells = {}
        for engine in ["ferrum", "llamacpp", "mistralrs"]:
            json_path = HERE / f"{engine}__{model_label}__c16.json"
            v = "—"
            if json_path.exists():
                try:
                    d = json.loads(json_path.read_text())
                    t = d.get("tpot_ms", {}).get("median")
                    if isinstance(t, (int, float)):
                        v = f"{t:.0f}"
                except Exception:
                    pass
            cells[engine] = v
        display_model = (
            "Qwen3-30B-A3B (MoE)" if model_label == "Qwen3-30B-A3B" else model_label
        )
        tpot_rows.append(
            f"| {display_model} | {cells['ferrum']} | {cells['llamacpp']} | {cells['mistralrs']} |"
        )

    tpot_table = (
        "| Model | ferrum | llama.cpp | mistralrs |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(tpot_rows)
    )

    text = re.sub(
        r"### TPOT median \(ms\) at c = 16\n\n\| Model \| ferrum \| llama\.cpp \| mistralrs \|\n\|[^\n]+\n(\| [^\n]*?\(TBD\)[^\n]*\n)+",
        "### TPOT median (ms) at c = 16\n\n" + tpot_table + "\n",
        text,
    )

    report.write_text(text)
    print(f"updated bench report grid: {report}")


def main():
    results = collect()
    print(f"collected {len(results)} cells")

    fill_file(ROOT / "README.md", results)
    fill_file(ROOT / "README_zh.md", results)
    fill_bench_report_grid(results)


if __name__ == "__main__":
    main()
