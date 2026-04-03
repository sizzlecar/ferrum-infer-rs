#!/bin/bash
# 远程 GPU 机器操作工具 — 所有长命令都后台 + 日志，绝不阻塞
#
# 用法:
#   remote.sh setup  <target>                # 初始化环境 + 克隆仓库
#   remote.sh sync   <target>                # 推送本地代码到远程 (git push + pull)
#   remote.sh build  <target> [features]     # 后台编译 (默认 cuda,tensor-parallel)
#   remote.sh pull   <target> <model>        # 后台下载模型
#   remote.sh bench  <target> <model> [args] # 后台跑 benchmark
#   remote.sh tp     <target> [model]        # 后台跑 TP=2 benchmark
#   remote.sh test   <target>                # 后台跑 test_gpu.sh
#   remote.sh run    <target> <command>      # 后台跑任意命令
#   remote.sh status <target>                # 一览全局状态
#   remote.sh log    <target> [name] [lines] # 查看日志尾部
#   remote.sh errors <target> [name]         # 只看错误
#   remote.sh poll   <target> <name> [interval] # 轮询日志直到完成或出错
#   remote.sh kill   <target>                # 杀掉远程 cargo/ferrum 进程
#
# 日志约定: 所有后台任务写 /tmp/<name>.log, 完成时追加 __DONE__ 或 __FAIL__

set -euo pipefail

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15"
REMOTE_DIR="/workspace/ferrum-infer-rs"
REMOTE_HF="/workspace/.hf_home"
CMD="${1:-help}"
TARGET="${2:-}"
BRANCH="feat/tensor-parallel"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[!!]${NC} $*"; }
log_err()  { echo -e "${RED}[ERR]${NC} $*"; }

need_target() {
    if [ -z "$TARGET" ]; then
        echo "Usage: $0 $CMD <ssh_target> [args...]"
        exit 1
    fi
}

# 远程后台运行命令，输出写日志，完成追加标记
# $1=log_name  $2=shell_command
remote_bg() {
    local logname="$1"
    local cmd="$2"
    $SSH "$TARGET" "bash -c 'source \$HOME/.cargo/env 2>/dev/null; cd $REMOTE_DIR; export CUDA_HOME=/usr/local/cuda HF_HOME=$REMOTE_HF; nohup bash -c \"($cmd) > /tmp/${logname}.log 2>&1; if [ \\\$? -eq 0 ]; then echo __DONE__ >> /tmp/${logname}.log; else echo __FAIL__ >> /tmp/${logname}.log; fi\" > /dev/null 2>&1 & echo PID=\$!'"
    log_ok "Started → /tmp/${logname}.log"
    echo "  Check:  $0 log $TARGET $logname"
    echo "  Poll:   $0 poll $TARGET $logname"
    echo "  Errors: $0 errors $TARGET $logname"
}

case "$CMD" in
  setup)
    need_target
    $SSH "$TARGET" 'bash -s' << SETUP
      set -e
      echo "[\$(date '+%H:%M:%S')] Installing Rust..."
      if ! command -v rustc &>/dev/null; then
          curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -1
      fi
      source \$HOME/.cargo/env
      echo "[\$(date '+%H:%M:%S')] Installing system deps..."
      apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev git > /dev/null 2>&1 || true
      echo "[\$(date '+%H:%M:%S')] Cloning/updating repo..."
      cd /workspace 2>/dev/null || { mkdir -p /workspace && cd /workspace; }
      if [ -d ferrum-infer-rs ]; then
          cd ferrum-infer-rs && git fetch origin && git checkout $BRANCH && git reset --hard origin/$BRANCH
      else
          git clone https://github.com/sizzlecar/ferrum-infer-rs.git && cd ferrum-infer-rs && git checkout $BRANCH
      fi
      echo "[\$(date '+%H:%M:%S')] GPU info:"
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU"
      echo "SETUP_DONE \$(date)"
SETUP
    ;;

  sync)
    need_target
    echo "Pushing local branch to origin..."
    git push origin "$BRANCH" 2>&1 | tail -3
    echo "Pulling on remote..."
    $SSH "$TARGET" "cd $REMOTE_DIR && git fetch origin && git reset --hard origin/$BRANCH && echo 'Synced: \$(git log --oneline -1)'"
    ;;

  build)
    need_target
    FEATURES="${3:-cuda,tensor-parallel}"
    echo "Building with features=$FEATURES ..."
    remote_bg "build" "cargo build --release -p ferrum-cli --features $FEATURES"
    ;;

  pull)
    need_target
    MODEL="${3:?Usage: $0 pull <target> <model>}"
    LOGNAME="pull_$(echo "$MODEL" | tr '/:' '_')"
    remote_bg "$LOGNAME" "./target/release/ferrum pull $MODEL"
    ;;

  bench)
    need_target
    MODEL="${3:?Usage: $0 bench <target> <model> [extra_args]}"
    shift 3
    EXTRA="$*"
    LOGNAME="bench_$(echo "$MODEL" | tr '/:' '_')"
    remote_bg "$LOGNAME" "./target/release/ferrum bench $MODEL $EXTRA"
    ;;

  tp)
    need_target
    MODEL="${3:-qwen3:4b}"
    EXTRA="${4:-}"
    LOGNAME="tp_bench"
    remote_bg "$LOGNAME" "FERRUM_TP=2 ./target/release/ferrum bench $MODEL --rounds 3 $EXTRA"
    ;;

  test)
    need_target
    remote_bg "test" "bash scripts/test_gpu.sh"
    ;;

  run)
    need_target
    shift 2
    RUNCMD="$*"
    LOGNAME="run_$(date +%H%M%S)"
    remote_bg "$LOGNAME" "$RUNCMD"
    ;;

  status)
    need_target
    $SSH "$TARGET" "bash -s" << 'STATUS'
      echo "=== GPU ==="
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
      echo ""
      echo "=== Processes ==="
      ps aux | grep -E 'cargo|rustc|ferrum' | grep -v grep | awk '{printf "  %-6s %s %s %s\n",$2,$3,$4,$11}' || echo "  (none)"
      echo ""
      echo "=== Logs ==="
      for f in /tmp/*.log; do
          [ -f "$f" ] || continue
          name=$(basename "$f" .log)
          if grep -q '__DONE__' "$f" 2>/dev/null; then
              status="DONE"
          elif grep -q '__FAIL__' "$f" 2>/dev/null; then
              status="FAIL"
          elif grep -q 'error\[E' "$f" 2>/dev/null; then
              status="ERROR"
          else
              status="running"
          fi
          size=$(wc -l < "$f" | tr -d ' ')
          printf "  %-20s %-8s %s lines\n" "$name" "$status" "$size"
      done
      echo ""
      echo "=== Models ==="
      ls -d /workspace/.hf_home/hub/models--* 2>/dev/null | while read d; do
          n=$(basename "$d" | sed 's/models--//')
          sf=$(find "$d" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
          echo "  $n ($sf files)"
      done || echo "  (none)"
STATUS
    ;;

  log)
    need_target
    LOGNAME="${3:-build}"
    LINES="${4:-40}"
    $SSH "$TARGET" "tail -$LINES /tmp/${LOGNAME}.log 2>/dev/null || echo 'No log: /tmp/${LOGNAME}.log'"
    ;;

  errors)
    need_target
    LOGNAME="${3:-build}"
    $SSH "$TARGET" "grep -E 'error\[E|panicked|FAIL|__FAIL__' /tmp/${LOGNAME}.log 2>/dev/null | head -20 || echo 'No errors in $LOGNAME'"
    ;;

  poll)
    need_target
    LOGNAME="${3:?Usage: $0 poll <target> <name> [interval]}"
    INTERVAL="${4:-5}"
    SEEN=0
    echo "Polling /tmp/${LOGNAME}.log every ${INTERVAL}s (Ctrl-C to stop)..."
    while true; do
        TOTAL=$($SSH "$TARGET" "wc -l < /tmp/${LOGNAME}.log 2>/dev/null || echo 0" | tr -d ' ')
        if [ "$TOTAL" -gt "$SEEN" ]; then
            SKIP=$((SEEN + 1))
            $SSH "$TARGET" "sed -n '${SKIP},${TOTAL}p' /tmp/${LOGNAME}.log 2>/dev/null"
            SEEN=$TOTAL
        fi
        # Check completion
        DONE=$($SSH "$TARGET" "tail -1 /tmp/${LOGNAME}.log 2>/dev/null || echo ''")
        if [[ "$DONE" == *"__DONE__"* ]]; then
            log_ok "$LOGNAME completed successfully"
            break
        elif [[ "$DONE" == *"__FAIL__"* ]]; then
            log_err "$LOGNAME failed!"
            $SSH "$TARGET" "grep -E 'error\[E|panicked' /tmp/${LOGNAME}.log 2>/dev/null | tail -5"
            break
        fi
        sleep "$INTERVAL"
    done
    ;;

  kill)
    need_target
    $SSH "$TARGET" "pkill -f 'cargo|rustc|ferrum' && echo 'Killed' || echo 'Nothing to kill'"
    ;;

  help|*)
    cat << 'HELP'
远程 GPU 机器操作工具 — 所有长命令后台运行，绝不阻塞

生命周期:
  setup  <target>                初始化环境 + 克隆仓库
  sync   <target>                推送代码 (git push + remote pull)
  kill   <target>                杀掉远程 cargo/ferrum 进程

后台任务 (启动即返回):
  build  <target> [features]     编译 (默认 cuda,tensor-parallel)
  pull   <target> <model>        下载模型
  bench  <target> <model> [args] 跑 benchmark
  tp     <target> [model]        TP=2 benchmark (默认 qwen3:4b)
  test   <target>                跑 test_gpu.sh
  run    <target> <command>      跑任意命令

监控:
  status <target>                一览: GPU/进程/日志/模型
  log    <target> [name] [lines] 看日志尾部 (默认 build, 40行)
  errors <target> [name]         只看错误
  poll   <target> <name> [sec]   持续轮询直到完成/出错

典型 TP 测试流程:
  $0 sync   gpu          # 同步代码
  $0 build  gpu          # 后台编译
  $0 poll   gpu build    # 等编译完成
  $0 pull   gpu qwen3:4b # 下模型 (可与编译并行)
  $0 tp     gpu          # 跑 TP benchmark
  $0 poll   gpu tp_bench # 看结果
HELP
    ;;
esac
