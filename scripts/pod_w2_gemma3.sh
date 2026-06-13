#!/usr/bin/env bash
# W2 Gemma3-27B CUDA gate session (armored, one-shot). Mirrors
# pod_w1_final_armored.sh hardening: setsid retry loops + marker files,
# crates.io reachability probe before choosing the rsproxy mirror.
#
# Phases (all marker-gated, babysitter runs gates when staged):
#   dl_gptq.ok    circulus/gemma-3-27b-it-gptq (classic GPTQ, text-only)
#   dl_gguf.ok    unsloth Q4_K_M (for the llama.cpp same-card comparison)
#   build.ok      ferrum slim (cuda,vllm-paged-attn-v2)
#   llamacpp.ok   llama.cpp CUDA build (llama-bench)
#   gates: L2 known-answer + L3/L4 smoke ladder -> L5 bench-serve sweep
#          -> llama-bench tg128 -> w2_session.done
set -ux
W=/workspace/w2
mkdir -p "$W/gates"
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-pip pkg-config libssl-dev cmake build-essential >/dev/null 2>&1
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null 2>&1
fi
. "$HOME/.cargo/env"
mkdir -p ~/.cargo
if curl -sI --max-time 8 https://index.crates.io/config.json | grep -q "200"; then
  printf '[net]\ngit-fetch-with-cli = true\nretry = 10\n' > ~/.cargo/config.toml
else
  cat > ~/.cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[net]
git-fetch-with-cli = true
retry = 10
EOF
fi

pip install -q -U huggingface_hub 2>&1 | tail -1
# Plain-HTTP path: xet/hf_transfer starved on the W1 Iceland host.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0

dl_armored() {
  local args="$1" tag="$2"
  setsid bash -c "n=0; until [ -f $W/dl_${tag}.ok ] || [ \$n -ge 10 ]; do n=\$((n+1)); if [ \$((n%2)) -eq 0 ]; then export HF_ENDPOINT=https://hf-mirror.com; else unset HF_ENDPOINT; fi; hf download $args > $W/dl_${tag}.log 2>&1 && touch $W/dl_${tag}.ok; sleep 10; done" < /dev/null > /dev/null 2>&1 &
}
dl_armored "circulus/gemma-3-27b-it-gptq" gptq
dl_armored "unsloth/gemma-3-27b-it-GGUF --include gemma-3-27b-it-Q4_K_M.gguf" gguf

# ferrum slim build.
setsid bash -c "n=0; until [ -f $W/build.ok ] || [ \$n -ge 15 ]; do n=\$((n+1)); echo \"=== build attempt \$n\" >> $W/build.log; cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2 >> $W/build.log 2>&1 && touch $W/build.ok; sleep 5; done" < /dev/null > /dev/null 2>&1 &

# llama.cpp CUDA build (llama-bench only).
setsid bash -c "n=0; until [ -f $W/llamacpp.ok ] || [ \$n -ge 8 ]; do n=\$((n+1)); { [ -d /workspace/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp; } >> $W/llamacpp.log 2>&1; cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >> $W/llamacpp.log 2>&1 && cmake --build /workspace/llama.cpp/build --target llama-bench -j 8 >> $W/llamacpp.log 2>&1 && touch $W/llamacpp.ok; sleep 5; done" < /dev/null > /dev/null 2>&1 &

# Babysitter: run the gate ladder when everything is staged.
setsid bash -c "
while true; do
  if [ -f $W/build.ok ] && [ -f $W/dl_gptq.ok ] && [ -f $W/dl_gguf.ok ] && [ -f $W/llamacpp.ok ]; then
    bash scripts/pod_w2_gates.sh > $W/gates.log 2>&1
    touch $W/w2_session.done
    exit 0
  fi
  sleep 60
done" < /dev/null > /dev/null 2>&1 &
echo "w2 armored session launched"
