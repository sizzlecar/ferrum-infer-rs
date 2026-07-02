#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/logs
chmod +x /workspace/w3_qwen35_c32_diagnostic.sh
set +e
/workspace/w3_qwen35_c32_diagnostic.sh --sha 9fda11014c17483b0ceca479e3704b3767db0cbe --tag mixed_prefill_immediate_kv --throughput-floor 458.0662677685816 --max-kv-admission-failed 13 --max-capacity-deferred 32 > /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/logs/tmux.stdout.log 2> /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/logs/tmux.stderr.log
rc=$?
set -e
echo "$rc" > /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/exit_code
grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/logs/tmux.stdout.log | tail -n 1 | sed -E 's/^.*: //' > /workspace/artifacts/w3_qwen35_c32_orchestrator_9fda1101_1782359039/artifact_path || true
exit "$rc"
