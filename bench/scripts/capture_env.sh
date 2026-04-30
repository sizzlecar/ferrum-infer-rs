#!/bin/bash
# Capture macOS memory/swap state + top-RSS processes to a file. Used by
# bench_one_model.sh to record the environment at the start AND end of
# each engine's run, so a regression spotted in numbers can be cross-
# checked against memory pressure (the 32 GB Mac that runs Group A
# benches sits within ~5 GB of the 30B-A3B working set, so paging is a
# real risk that quietly skews results).
#
# Usage:
#   capture_env.sh <label> <output_file>
#
# Appends a fenced block to <output_file> identified by <label>.

LABEL="$1"
OUT="$2"

{
  echo ""
  echo "──────── env snapshot: ${LABEL} @ $(date) ────────"
  echo "## vm_stat (selected)"
  vm_stat | grep -E "page size|Pages free|Pages active|Pages inactive|Pages wired|Pages purgeable|swapouts" | head -10
  echo "## swap"
  sysctl -n vm.swapusage
  echo "## top non-system RSS (>50 MB, excluding kernel/window/etc)"
  ps -axm -o rss,pid,comm | sort -nrk 1 \
    | awk '$1 > 51200 { printf "  %5d MB  %s\n", $1/1024, $3 }' \
    | head -10
  echo "## warning thresholds"
  swap_used_mb=$(sysctl -n vm.swapusage | awk -F'used = ' '{print $2}' | awk '{print int($1)}')
  if [ "${swap_used_mb:-0}" -gt 1024 ]; then
    echo "  ⚠ swap_used = ${swap_used_mb} MB (>1 GB) — bench numbers may be paging-affected"
  else
    echo "  swap_used = ${swap_used_mb:-0} MB (under 1 GB threshold)"
  fi
  echo ""
} >> "$OUT"
