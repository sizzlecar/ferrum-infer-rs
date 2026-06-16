#!/usr/bin/env python3
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


TIME_KEYS = {
    "total",
    "matmul",
    "attn",
    "qkr",
    "norm",
    "other",
    "tail_norm",
    "tail_mlp",
    "tail_gate_up",
    "tail_down",
    "tail_act",
    "tail_resid",
    "qkv",
    "o_proj",
    "gate_up",
    "act",
    "down",
    "resid",
    "final_norm",
    "final_copy",
    "lm_head",
    "readback",
    "generic_matmul",
    "generic_attn",
    "generic_qkr",
    "generic_norm",
    "generic_other",
    "marlin_ws_zero",
    "marlin_gather",
    "marlin_kernel",
    "marlin_qkv_ws",
    "marlin_qkv_gather",
    "marlin_qkv_kernel",
    "marlin_o_ws",
    "marlin_o_gather",
    "marlin_o_kernel",
    "marlin_gate_up_ws",
    "marlin_gate_up_gather",
    "marlin_gate_up_kernel",
    "marlin_down_ws",
    "marlin_down_gather",
    "marlin_down_kernel",
    "marlin_lm_head_ws",
    "marlin_lm_head_gather",
    "marlin_lm_head_kernel",
    "marlin_other_ws",
    "marlin_other_gather",
    "marlin_other_kernel",
    "unwrapped",
}

STAT_KEYS = [
    "total",
    "qkv",
    "qkr",
    "attn",
    "o_proj",
    "norm",
    "gate_up",
    "act",
    "down",
    "resid",
    "final_norm",
    "final_copy",
    "lm_head",
    "readback",
    "generic_matmul",
    "generic_attn",
    "generic_qkr",
    "generic_norm",
    "generic_other",
    "matmul",
    "other",
    "tail_mlp",
    "tail_gate_up",
    "tail_down",
    "tail_norm",
    "tail_act",
    "tail_resid",
    "marlin_ws_zero",
    "marlin_gather",
    "marlin_kernel",
    "unwrapped",
]


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    idx = min(
        len(sorted_values) - 1,
        max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1),
    )
    return sorted_values[idx]


def stat(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "max": vals[-1],
    }


def parse_kv_pairs(text):
    row = {}
    kv_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\d+)(?:us(?:\((\d+)\))?)?")
    for key, val, count in kv_re.findall(text):
        value = int(val)
        if key in TIME_KEYS:
            row[f"{key}_us"] = value
            if count:
                row[f"{key}_calls"] = int(count)
        else:
            row[key] = value
    return row


def parse_profile_rows(log_path):
    batched_re = re.compile(r"\[batched-op-profile\]\s+m=(\d+)\s+total=(\d+)us\s+(.*)$")
    unified_re = re.compile(r"\[unified-op-profile\]\s+call#(\d+)\s+(.*)$")
    rows = []
    formats = defaultdict(int)
    for line in log_path.read_text(errors="replace").splitlines():
        batched = batched_re.search(line)
        if batched:
            row = {
                "format": "batched-op-profile",
                "m": int(batched.group(1)),
                "num_seqs": int(batched.group(1)),
                "total_us": int(batched.group(2)),
            }
            row.update(parse_kv_pairs(batched.group(3)))
            rows.append(row)
            formats[row["format"]] += 1
            continue

        unified = unified_re.search(line)
        if unified:
            row = {"format": "unified-op-profile", "call": int(unified.group(1))}
            row.update(parse_kv_pairs(unified.group(2)))
            if "total_us" not in row:
                continue
            row["m"] = row.get("num_seqs", row.get("m_total", 0))
            rows.append(row)
            formats[row["format"]] += 1
    return rows, dict(formats)


def summarize_group(group):
    entry = {"count": len(group)}
    for key in STAT_KEYS:
        field = "total_us" if key == "total" else f"{key}_us"
        entry[f"{key}_us"] = stat([row.get(field) for row in group])
    total_sum = sum(row.get("total_us", 0) for row in group)
    if total_sum > 0:
        entry["share_of_total"] = {
            key: sum(row.get(f"{key}_us", 0) for row in group) / total_sum
            for key in STAT_KEYS
            if key != "total"
        }
    return entry


def group_by(rows, key, predicate=lambda _row: True):
    groups = defaultdict(list)
    for row in rows:
        if not predicate(row):
            continue
        if row.get(key) is None:
            continue
        groups[str(row[key])].append(row)
    return {
        name: summarize_group(group)
        for name, group in sorted(groups.items(), key=lambda item: int(item[0]))
    }


def metric_mean(bench, name):
    node = bench.get(name)
    return node.get("mean") if isinstance(node, dict) else None


def p95_itl(bench):
    node = bench.get("itl_ms")
    if isinstance(node, dict):
        p95 = node.get("p95")
        if isinstance(p95, dict):
            return p95.get("mean")
    return None


def read_rel(out, path):
    p = out / path
    return p.read_text().strip() if p.exists() else None


def nonzero_quality(bench):
    result = []
    for idx, item in enumerate(bench.get("quality_issues_per_run") or []):
        if isinstance(item, dict):
            nz = {k: v for k, v in item.items() if v}
            if nz:
                result.append({"run": idx, "issues": nz})
    return result


def completed_ok(bench):
    completed = bench.get("completed_per_run") or []
    errors = bench.get("errored_per_run") or []
    return bool(completed) and all(x == 100 for x in completed) and all(x == 0 for x in errors)


def top_total_rows(rows, limit=10):
    fields = [
        "call",
        "format",
        "m_total",
        "num_seqs",
        "prefill",
        "decode",
        "sampled",
        "max_kv",
        "total_us",
        "gate_up_us",
        "down_us",
        "attn_us",
        "lm_head_us",
        "generic_matmul_us",
        "generic_attn_us",
    ]
    return [
        {k: row.get(k) for k in fields if row.get(k) is not None}
        for row in sorted(rows, key=lambda item: item.get("total_us", 0), reverse=True)[:limit]
    ]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: postprocess_tail_profile_c16.py <out_dir>")
    out = Path(sys.argv[1])
    bench_path = out / "perf/bench_ferrum_profile_c16_100x1.json"
    log_path = out / "server/server.log"
    smoke_path = out / "smoke/stream_summary.json"
    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads(bench_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    rows, formats = parse_profile_rows(log_path)
    decode_only = [row for row in rows if row.get("prefill") == 0 and row.get("decode", 0) > 0]
    mixed_or_prefill = [row for row in rows if row.get("prefill", 0) > 0]
    quality_nonzero = nonzero_quality(bench)

    summary = {
        "lane": "w2_tail_latency_profile_c16_samepod",
        "status": "diagnostic_pass",
        "release_gate": False,
        "profile_formats": formats,
        "remote_git_sha": read_rel(out, "env/git_sha.txt"),
        "remote_git_status_short": read_rel(out, "env/git_status_short.txt"),
        "ferrum_binary_sha256": read_rel(out, "env/ferrum.sha256"),
        "dataset_sha256": read_rel(out, "env/dataset.sha256"),
        "model_path": read_rel(out, "env/model_path.txt"),
        "smoke": smoke,
        "bench_completed_per_run": bench.get("completed_per_run"),
        "bench_errored_per_run": bench.get("errored_per_run"),
        "bench_quality_issues_per_run": bench.get("quality_issues_per_run"),
        "bench_quality_nonzero": quality_nonzero,
        "output_token_count_source": bench.get("output_token_count_source"),
        "output_tps_mean": metric_mean(bench, "output_throughput_tps"),
        "itl_p95_ms_mean": p95_itl(bench),
        "profile_rows": len(rows),
        "decode_only_rows": len(decode_only),
        "mixed_or_prefill_rows": len(mixed_or_prefill),
        "decode_only_all": summarize_group(decode_only) if decode_only else None,
        "mixed_or_prefill_all": summarize_group(mixed_or_prefill) if mixed_or_prefill else None,
        "profile_by_num_seqs": group_by(rows, "num_seqs"),
        "decode_only_by_num_seqs": group_by(decode_only, "num_seqs"),
        "decode_only_by_decode_count": group_by(decode_only, "decode"),
        "all_rows_by_decode_count": group_by(rows, "decode"),
        "top_total_rows": top_total_rows(rows),
        "same_pod_reference": {
            "vllm_c16_output_tps_mean": 500.67038762731977,
            "vllm_c16_output_tps_lcb": 478.39462812583776,
            "vllm_c16_p95_itl_ms_mean": 33.06958213333332,
            "ferrum_c16_output_tps_mean": 422.34520497237537,
            "ferrum_c16_output_tps_lcb": 414.59153186899397,
            "ferrum_c16_p95_itl_ms_mean": 52.81935383333333,
        },
    }

    if not completed_ok(bench):
        summary["status"] = "diagnostic_fail"
    if quality_nonzero:
        summary["status"] = "diagnostic_fail"
    if bench.get("output_token_count_source") != "usage":
        summary["status"] = "diagnostic_fail"
    if smoke.get("done_count") != 1 or not smoke.get("usage"):
        summary["status"] = "diagnostic_fail"
    if not rows:
        summary["status"] = "diagnostic_fail"

    (analysis_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    decode_all = summary.get("decode_only_all") or {}
    shares = decode_all.get("share_of_total") or {}
    total = decode_all.get("total_us") or {}
    gate = decode_all.get("gate_up_us") or {}
    down = decode_all.get("down_us") or {}
    attn = decode_all.get("attn_us") or {}
    qkv = decode_all.get("qkv_us") or {}
    lm_head = decode_all.get("lm_head_us") or {}
    generic_matmul = decode_all.get("generic_matmul_us") or {}

    lines = [
        "# W2 Tail Profile c16 Diagnostic",
        "",
        f"- Status: `{summary['status']}`",
        "- Release gate: no",
        f"- Profile formats: `{formats}`",
        f"- Profile rows: `{len(rows)}` total, `{len(decode_only)}` decode-only",
        f"- Bench completed/errors: `{bench.get('completed_per_run')}` / `{bench.get('errored_per_run')}`",
        f"- Output token count source: `{bench.get('output_token_count_source')}`",
        f"- Profile-run output throughput mean: `{summary['output_tps_mean']}` tok/s",
        f"- Profile-run p95 ITL mean: `{summary['itl_p95_ms_mean']}` ms",
        "",
        "## Decode-Only Aggregate",
        "",
        f"- total_us mean/p95/max: `{total.get('mean')}` / `{total.get('p95')}` / `{total.get('max')}`",
        f"- generic_matmul_us mean/p95/max: `{generic_matmul.get('mean')}` / `{generic_matmul.get('p95')}` / `{generic_matmul.get('max')}`",
        f"- gate_up_us mean/p95/max: `{gate.get('mean')}` / `{gate.get('p95')}` / `{gate.get('max')}`",
        f"- down_us mean/p95/max: `{down.get('mean')}` / `{down.get('p95')}` / `{down.get('max')}`",
        f"- qkv_us mean/p95/max: `{qkv.get('mean')}` / `{qkv.get('p95')}` / `{qkv.get('max')}`",
        f"- attn_us mean/p95/max: `{attn.get('mean')}` / `{attn.get('p95')}` / `{attn.get('max')}`",
        f"- lm_head_us mean/p95/max: `{lm_head.get('mean')}` / `{lm_head.get('p95')}` / `{lm_head.get('max')}`",
        f"- generic_matmul share: `{shares.get('generic_matmul')}`",
        f"- gate_up + down share: `{(shares.get('gate_up') or 0) + (shares.get('down') or 0)}`",
        f"- attention share: `{shares.get('attn')}`",
        f"- lm_head share: `{shares.get('lm_head')}`",
        "",
        "This artifact is diagnostic only because profile logging changes runtime cost.",
    ]
    (analysis_dir / "summary.md").write_text("\n".join(lines) + "\n")

    if summary["status"] != "diagnostic_pass":
        raise SystemExit("tail profile diagnostic failed; see analysis/profile_summary.json")
    print(f"W2 TAIL PROFILE C16 DIAGNOSTIC PASS: {out}")


if __name__ == "__main__":
    main()
