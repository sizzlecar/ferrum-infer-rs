#!/bin/bash
# 远程 GPU 机器操作工具
# 用法:
#   bash scripts/remote.sh setup <ssh_target>     # 初始化环境
#   bash scripts/remote.sh build <ssh_target>     # 后台编译
#   bash scripts/remote.sh status <ssh_target>    # 检查编译状态
#   bash scripts/remote.sh pull <ssh_target> <model>  # 后台下载模型
#   bash scripts/remote.sh test <ssh_target>      # 跑测试
#   bash scripts/remote.sh log <ssh_target> <logname> # 查看日志

set -e
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
CMD=$1
TARGET=$2

case $CMD in
  setup)
    $SSH $TARGET 'bash -s' << 'SETUP'
      curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -1
      source $HOME/.cargo/env
      apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev git > /dev/null 2>&1
      cd /workspace
      if [ -d ferrum-infer-rs ]; then
        cd ferrum-infer-rs && git fetch origin && git checkout feat/tensor-parallel && git reset --hard origin/feat/tensor-parallel
      else
        git clone https://github.com/sizzlecar/ferrum-infer-rs.git && cd ferrum-infer-rs && git checkout feat/tensor-parallel
      fi
      echo "SETUP_DONE $(date)"
SETUP
    ;;

  build)
    FEATURES=${3:-cuda}
    $SSH $TARGET "bash -c 'source \$HOME/.cargo/env && cd /workspace/ferrum-infer-rs && CUDA_HOME=/usr/local/cuda nohup cargo build --release -p ferrum-cli --features $FEATURES > /tmp/build.log 2>&1 & echo PID=\$!'"
    echo "Build started. Check: bash scripts/remote.sh status $TARGET"
    ;;

  status)
    $SSH $TARGET 'echo "=== Build ===" && (grep -q "Finished" /tmp/build.log 2>/dev/null && echo "DONE: $(tail -1 /tmp/build.log)" || (grep -c "Compiling" /tmp/build.log 2>/dev/null | xargs -I{} echo "Compiling: {} crates" && grep "error\[" /tmp/build.log 2>/dev/null | head -5 || echo "In progress...")) && echo "" && echo "=== Models ===" && ls /workspace/.hf_home/hub/models--*/snapshots/*/*.safetensors 2>/dev/null | wc -l | xargs -I{} echo "{} safetensors files" && echo "" && echo "=== Processes ===" && ps aux | grep -E "cargo|rustc|ferrum" | grep -v grep | wc -l | xargs -I{} echo "{} active processes"'
    ;;

  pull)
    MODEL=$3
    $SSH $TARGET "bash -c 'source \$HOME/.cargo/env && cd /workspace/ferrum-infer-rs && HF_HOME=/workspace/.hf_home HF_TOKEN=${HF_TOKEN:-} nohup ./target/release/ferrum pull $MODEL > /tmp/pull_$(echo $MODEL | tr / _).log 2>&1 & echo \"Pull $MODEL started\"'"
    ;;

  test)
    $SSH $TARGET "bash -c 'source \$HOME/.cargo/env && cd /workspace/ferrum-infer-rs && export HF_HOME=/workspace/.hf_home && bash scripts/test_gpu.sh 2>&1'"
    ;;

  log)
    LOGNAME=${3:-build}
    $SSH $TARGET "tail -20 /tmp/${LOGNAME}.log 2>/dev/null || echo 'No log found'"
    ;;

  errors)
    $SSH $TARGET 'grep "error\[" /tmp/build.log 2>/dev/null | head -10 || echo "No errors"'
    ;;

  *)
    echo "Usage: bash scripts/remote.sh {setup|build|status|pull|test|log|errors} <ssh_target> [args]"
    ;;
esac
