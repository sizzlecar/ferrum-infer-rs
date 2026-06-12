#!/usr/bin/env bash
# W1 CUDA gate-batch pod bootstrap. Run from the repo root on a fresh
# nvidia/cuda:12.4.0-devel-ubuntu22.04 Vast instance:
#   bash scripts/pod_w1_bootstrap.sh single   # 1x4090 gate batch
#   bash scripts/pod_w1_bootstrap.sh dual     # 2x4090 70B lane
#
# Builds ferrum (cuda,vllm-moe-marlin) while the model downloads run in
# parallel; everything logs under /workspace/w1/. The gate runs themselves
# are driven over ssh afterwards — this script only stages.
set -ux
ROLE="${1:?usage: pod_w1_bootstrap.sh single|dual}"
W=/workspace/w1
mkdir -p "$W"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-pip pkg-config libssl-dev cmake >/dev/null 2>&1

# Rust
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null 2>&1
fi
. "$HOME/.cargo/env"

# Fast HF downloads
pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Build in the background while models download. vllm-paged-attn-v2 is
# required: the CUDA autosizer defaults FERRUM_USE_VLLM_PAGED_ATTN=1 and
# serve hard-errors if the kernel isn't compiled in.
( cd "$(pwd)" && cargo build --release -p ferrum-cli \
    --features cuda,vllm-moe-marlin,vllm-paged-attn-v2 \
    > "$W/build.log" 2>&1 && touch "$W/build.ok" || touch "$W/build.fail" ) &

dl() { # repo [include-pattern]
  local repo="$1"; shift
  ( hf download "$repo" "$@" > "$W/dl_$(echo "$repo" | tr '/' '_').log" 2>&1 \
      && touch "$W/dl_$(echo "$repo" | tr '/' '_').ok" \
      || touch "$W/dl_$(echo "$repo" | tr '/' '_').fail" ) &
}

if [ "$ROLE" = "single" ]; then
  # L1 representative (BF16) + CUDA smoke
  dl deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
  # W1 GPTQ gate models (Marlin-clean picks, GOAL.md UNVERIFIED #4)
  dl OPEA/DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc
  dl JunHowie/Qwen3-32B-GPTQ-Int4
  dl Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
  dl jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq
  # C7/G0 baseline (M3); M2 baseline handled by bench scripts if time allows
  dl Qwen/Qwen3-30B-A3B-GPTQ-Int4
  # transformers for the L1 reference generation. Pinned: transformers
  # 5.11 + tokenizers 0.22 garble the DeepSeek-R1 tokenizer round-trip
  # (encode loses spaces, decode emits raw byte-level markers); 4.51.3 +
  # tokenizers 0.21 round-trips byte-clean.
  pip install -q torch "transformers==4.51.3" "tokenizers==0.21.4" accelerate 2>&1 | tail -1 &
else
  dl unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF --include "*Q4_K_M*"
  dl deepseek-ai/DeepSeek-R1-Distill-Llama-70B --include "tokenizer*" --include "generation_config.json" --include "*.jinja"
fi

wait
echo "=== bootstrap done; build: $(ls "$W"/build.* 2>/dev/null), downloads:"; ls "$W"/dl_*.ok "$W"/dl_*.fail 2>/dev/null
