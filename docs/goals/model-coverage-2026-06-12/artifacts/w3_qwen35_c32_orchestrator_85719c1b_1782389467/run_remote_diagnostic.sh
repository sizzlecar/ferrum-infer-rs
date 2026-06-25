#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/logs
chmod +x /workspace/w3_qwen35_c32_diagnostic.sh
set +e
/workspace/w3_qwen35_c32_diagnostic.sh --sha 85719c1b51f3bac6672e7c842a2f6d20ad9ff024 --tag structured_decode_kv_pressure --throughput-floor 600.0 --max-kv-admission-failed 13 --max-capacity-deferred 32 --min-mixed-iterations 64 --max-p95-itl-ms 25.0 > /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/logs/tmux.stdout.log 2> /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/logs/tmux.stderr.log
rc=$?
set -e
echo "$rc" > /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/exit_code
grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/logs/tmux.stdout.log | tail -n 1 | sed -E 's/^.*: //' > /workspace/artifacts/w3_qwen35_c32_orchestrator_85719c1b_1782389467/artifact_path || true
exit "$rc"
