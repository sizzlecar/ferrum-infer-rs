#!/usr/bin/env bash
# Hardened one-shot for the final W1 micro-session. Survives the
# session-scope process reaping seen on budget Vast hosts: every long
# phase runs under setsid with an auto-retry loop, and cargo uses the
# rsproxy mirror (some hosts cannot reach crates.io). Downloads ONLY the
# three 32B GPTQ repos and builds the slim feature set (dense Marlin is
# auto-enabled by `cuda`; the heavy vllm-moe-marlin unit isn't needed).
set -ux
W=/workspace/w1
mkdir -p "$W"
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-pip pkg-config libssl-dev cmake >/dev/null 2>&1
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null 2>&1
fi
. "$HOME/.cargo/env"
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[net]
git-fetch-with-cli = true
retry = 10
EOF

pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Trio downloads, each in its own setsid retry loop.
dl_armored() {
  local repo="$1" tag="$2"
  setsid bash -c "n=0; until [ -f $W/dl_${tag}.ok ] || [ \$n -ge 8 ]; do n=\$((n+1)); hf download $repo > $W/dl_${tag}.log 2>&1 && touch $W/dl_${tag}.ok; sleep 5; done" < /dev/null > /dev/null 2>&1 &
}
dl_armored OPEA/DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc OPEA
dl_armored JunHowie/Qwen3-32B-GPTQ-Int4 JunHowie
dl_armored Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 Qwen25

# Slim build under setsid retry.
setsid bash -c "n=0; until [ -f $W/build.ok ] || [ \$n -ge 15 ]; do n=\$((n+1)); echo \"=== armored build attempt \$n\" >> $W/build.log; cargo build --release -j 8 -p ferrum-cli --features cuda,vllm-paged-attn-v2 >> $W/build.log 2>&1 && touch $W/build.ok; sleep 5; done" < /dev/null > /dev/null 2>&1 &

# Babysitter: when build + trio are staged, run the L5 trio (also setsid).
setsid bash -c "
while true; do
  if [ -f $W/build.ok ] && [ -f $W/dl_OPEA.ok ] && [ -f $W/dl_JunHowie.ok ] && [ -f $W/dl_Qwen25.ok ]; then
    bash scripts/pod_w1_l5_final.sh > $W/l5final.log 2>&1
    touch $W/session.done
    exit 0
  fi
  sleep 60
done" < /dev/null > /dev/null 2>&1 &
echo "armored session launched"
