#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/logs
chmod +x /workspace/w3_qwen35_c32_diagnostic.sh
set +e
/workspace/w3_qwen35_c32_diagnostic.sh --sha ff5cab92175e3929e70ce5e0a70fcc706d82b8b7 --tag mixed_capacity_backoff --throughput-floor 600.0 --max-kv-admission-failed 13 --max-capacity-deferred 32 --min-mixed-iterations 64 --max-p95-itl-ms 25.0 > /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/logs/tmux.stdout.log 2> /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/logs/tmux.stderr.log
rc=$?
set -e
echo "$rc" > /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/exit_code
grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/logs/tmux.stdout.log | tail -n 1 | sed -E 's/^.*: //' > /workspace/artifacts/w3_qwen35_c32_orchestrator_ff5cab92_1782371771/artifact_path || true
exit "$rc"
