#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/logs
chmod +x /workspace/w3_qwen35_c32_diagnostic.sh
set +e
/workspace/w3_qwen35_c32_diagnostic.sh --sha a6cbb1d0b3f3544c4927c5531cf7331182c91412 --tag active_recompute_shared_slot --throughput-floor 600.0 --max-kv-admission-failed 13 --max-capacity-deferred 32 --min-mixed-iterations 64 --max-p95-itl-ms 25.0 > /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/logs/tmux.stdout.log 2> /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/logs/tmux.stderr.log
rc=$?
set -e
echo "$rc" > /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/exit_code
grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/logs/tmux.stdout.log | tail -n 1 | sed -E 's/^.*: //' > /workspace/artifacts/w3_qwen35_c32_orchestrator_a6cbb1d0_1782365116/artifact_path || true
exit "$rc"
