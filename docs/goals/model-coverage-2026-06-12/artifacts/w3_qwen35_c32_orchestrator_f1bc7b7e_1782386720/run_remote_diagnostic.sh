#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/logs
chmod +x /workspace/w3_qwen35_c32_diagnostic.sh
set +e
/workspace/w3_qwen35_c32_diagnostic.sh --sha f1bc7b7ea71d525b8ea9572623612552fc6232df --tag decode_defer_waits_for_independent_kv_release --throughput-floor 600.0 --max-kv-admission-failed 13 --max-capacity-deferred 32 --min-mixed-iterations 64 --max-p95-itl-ms 25.0 > /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/logs/tmux.stdout.log 2> /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/logs/tmux.stderr.log
rc=$?
set -e
echo "$rc" > /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/exit_code
grep -E 'FERRUM W3 QWEN35 C32 DIAG (KEEP|REJECT):' /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/logs/tmux.stdout.log | tail -n 1 | sed -E 's/^.*: //' > /workspace/artifacts/w3_qwen35_c32_orchestrator_f1bc7b7e_1782386720/artifact_path || true
exit "$rc"
