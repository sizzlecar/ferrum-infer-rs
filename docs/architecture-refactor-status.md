# Architecture Refactor — 5-Dim Complete-Form Goal & Status

> Started 2026-05-09. Tracks the audit-driven refactor toward a clean
> 5-dimension architecture where new backends / models / quant formats /
> KV precisions can be added independently.

## The 5-dim goal

Inference engines compose 5 orthogonal axes. Naive multiplication is
~7 200 cells (8×5×15×4×3); the right design has each axis as a single
polymorphism point so total maintenance units are ~44 (additive, not
multiplicative).

| # | Dimension | Polymorphism point | Examples |
|---|---|---|---|
| 1 | **Model architecture** | Per-family Rust struct (`LlamaFamilyModel<B>`, `Qwen3MoeModel<B>`, future `DeepseekV3Model<B>`) | Llama / Qwen2 / Qwen3 / Qwen3-MoE / Mistral SW / DeepSeek-V3 MLA / Gemma |
| 2 | **Compute precision** | `Linear<B>` impl | FP32 / FP16 / BF16 / FP8 (E4M3) / INT8 / INT4 |
| 3 | **Weight format** | `WeightLoader<B>` produces format-specific `Linear<B>` impl | safetensors / GPTQ / AWQ / GGUF (Q2K-Q8) / Marlin-packed / HQQ / EXL2 |
| 4 | **Inference device** | `Backend` trait + capability supertraits + bundle aliases | CUDA / Metal / CPU / AMD / Intel / 国产 GPU |
| 5 | **KV cache precision** | `KvDtypeKind` marker + `BackendKvDtype<K>` supertrait | FP16 / BF16 / INT8 / FP8 |

The combinatorial complexity stays additive because each axis only adds
its own units, never multiplies with the others.

### Acceptance for "完全体" (complete form)

Three gates — ALL must pass before the refactor is considered done:

1. **CLI + server use the new architecture** — not just trait切片 in
   ferrum-kernels; CLI commands and HTTP handlers reference the new
   modality-specific traits / Linear impls.
2. **All previously perf-tested models infer correctly** — Qwen3
   (0.6B / 1.7B / 4B / 8B), Qwen2.5-Instruct (FP16 + GPTQ-Int4),
   Qwen3-30B-A3B MoE (Marlin INT4 on CUDA, Q4_K_M GGUF on Metal),
   Qwen3-TTS-0.6B/1.7B, whisper-large-v3-turbo, BERT, CLIP.
3. **Performance does not regress** — Metal smoke for each phase
   relative to per-phase baseline ≤ 3% drift (LTO usually folds trait
   reorganization to identical machine code; thermals are the noise
   floor).

## Progress (as of 2026-05-10, completed dims 1-4 + crate consolidation)

### Dim status

| Dim | Status | Notes |
|---|---|---|
| 1. Model architecture | ✅ done | Architecture v2 (historical, pre-audit) |
| 2. Compute precision | ✅ done | Phase 3e cutover complete (#123 Marlin, #124 GGUF). All quant kernel dispatch lives in `Linear<B>` impls in `ferrum-kernels::quant_linear/`. `BackendQuantMarlin::gemm_gptq*` and `BackendQuantGguf::gemm_quant` deleted. |
| 3. Weight format | ✅ done | `Linear<B>` polymorphism point owns the kernel call; new format = new `Linear<B>` impl in ferrum-kernels (no Backend trait change). |
| 4. **Inference device** | ✅ done | Backend → 5 supertraits + 3 capability bundles (#107-#110) |
| 5. KV cache precision | 🟡 mostly done | Type system (#112, #119): `KvDtypeKind` + `BackendKvDtype<K>` + `KvCache<B, K = KvFp16>`. CUDA INT8 KV kernels + `BackendKvDtype<KvInt8> for CudaBackend` marker landed (`paged_decode_attention_int8`, `int8_kv_cache_append`, host-ref cosine parity 0.99999). Model-side wire-up that flips a path to `KvCache<B, KvInt8>` + the int8 launchers is the only remaining step. FP8 KV is a separate future PR. |

**5 of 5 dims structurally complete.** CUDA INT8 KV kernels are validated against an FP32 host reference (cos sim 0.99999, max abs / max output ≈ 0.36%). Closing the last 5% requires only model integration — not kernel work.

### PRs landed (in order)

| PR | Phase | Effect |
|---|---|---|
| [#106](https://github.com/sizzlecar/ferrum-infer-rs/pull/106) | 1 | Delete 377 lines of dead code (KernelExecutor / DecodeBackend / orphan EngineFactory) |
| [#107](https://github.com/sizzlecar/ferrum-infer-rs/pull/107) | 2.1 + 2.2 | Extract `BackendGraph` + `BackendCollective` supertraits |
| [#108](https://github.com/sizzlecar/ferrum-infer-rs/pull/108) | 2.3 | Extract `BackendQuantMarlin` + `BackendQuantGguf` supertraits |
| [#109](https://github.com/sizzlecar/ferrum-infer-rs/pull/109) | 2.4 | Extract `BackendPagedKv` + `BackendMoeFused` supertraits |
| [#110](https://github.com/sizzlecar/ferrum-infer-rs/pull/110) | 2.5 | `LlmBackend` + `QuantLlmBackend` + `MoeLlmBackend` capability bundles |
| [#111](https://github.com/sizzlecar/ferrum-infer-rs/pull/111) | 3a + 3b | Split `qwen3_moe.rs` (3579 → 2950) + `llama_family.rs` (4305 → 2652); -2282 lines combined in two largest files |
| [#112](https://github.com/sizzlecar/ferrum-infer-rs/pull/112) | 4 | KV-dtype scaffolding (`KvDtypeKind` + `BackendKvDtype<K>` traits, Metal +1.0%) |
| [#113](https://github.com/sizzlecar/ferrum-infer-rs/pull/113) | 3c | `ExpertStack::gemv_*` methods replace 4 `B::gemv_quant_moe_id*` call sites in stacked-decode path (Metal +2.5%) |
| [#114](https://github.com/sizzlecar/ferrum-infer-rs/pull/114) | 5a step 1 | Extract `modality_stubs` helper for the 3 fake engines (-39 lines copy-paste, +59 shared) |
| [#116](https://github.com/sizzlecar/ferrum-infer-rs/pull/116) | 3d | `ExpertStack::gemm_*` + `gemv_*_batched/offset` wrappers replace remaining 16 `B::*_moe_id*` call sites in `moe/forward.rs` + `qwen3_moe.rs` decode path. **Vast 4090: Llama-8B INT4 +0.3%, Qwen3-30B-A3B vLLM Marlin c=4 +3.1% (55.9 vs 54.2 tok/s)**. |
| [#117](https://github.com/sizzlecar/ferrum-infer-rs/pull/117) | 5a step 2 | Split `InferenceEngine` mega-trait into lifecycle base + 4 modality supertraits (`LlmInferenceEngine`/`EmbedEngine`/`TranscribeEngine`/`TtsEngine`). `AppState` 4 fields; handlers 501 when missing. The 3 fake engines now impl only the modality trait their executor serves. |
| [#118](https://github.com/sizzlecar/ferrum-infer-rs/pull/118) | 5b/1 | Delete `DefaultInferenceEngine` (1217 lines net). Builder always returns `ContinuousBatchEngine`. Unblocks the `ferrum-runtime` crate delete (5b/2). |
| [#119](https://github.com/sizzlecar/ferrum-infer-rs/pull/119) | 4 wire-up | `KvCache<B, K = KvFp16>` type-param + PhantomData. **Vast 4090: Qwen3-0.6B 82.9 tok/s (baseline 83.0, perf-neutral ✅)**. |
| [#121](https://github.com/sizzlecar/ferrum-infer-rs/pull/121) | 5b/2 | Delete the `ferrum-runtime` crate; move `CandleBackend` + memory pool into `ferrum-engine/src/backends/`. Drops the legacy CPU `ComputeBackend` impl (no consumers). |
| [#122](https://github.com/sizzlecar/ferrum-infer-rs/pull/122) | 3e/1 | Add concrete `CudaMarlinLinear` + `CudaMarlinStackedExpertLinear` in `ferrum-kernels::quant_linear/cuda_marlin.rs`. Make `marlin_gemm_with_perm` + `launch_vllm_marlin` public so the new types can dispatch the kernel without a trait method. Additive — no behaviour change. |
| [#123](https://github.com/sizzlecar/ferrum-infer-rs/pull/123) | 3e/2 | **Marlin cutover.** `BackendQuantMarlin::load_gptq` returns `Box<dyn Linear<Self>>`; `make_stacked_expert_linear` factory replaces `gemm_gptq_with_offset`. Both `gemm_gptq*` trait methods deleted. `GptqLinear<B>` / `StackedExpertLinear<B>` shrink to delegating wrappers. CPU side gains `load_gptq_stacked` + `make_stacked_expert_linear` for parity tests. |
| [#124](https://github.com/sizzlecar/ferrum-infer-rs/pull/124) | 3e/3 | **GGUF cutover (closes Dim 2/3).** `BackendQuantGguf::load_quant` / `load_quant_fused` return `Box<dyn Linear<Self>>`. `gemm_quant` trait method deleted; the 209-line Metal body extracted into `pub fn metal_gemm_quant_dispatch`. New `MetalGgufLinear` + `CpuGgufLinear` in `ferrum-kernels::quant_linear/`. `QuantLinear<B>` shrinks to wrapper. |
| [#126](https://github.com/sizzlecar/ferrum-infer-rs/pull/126) | Dim 5 cleanup | Move `KvDtypeKind` + the four marker structs out of ferrum-kernels into ferrum-interfaces (no GPU deps). |
| [#127](https://github.com/sizzlecar/ferrum-infer-rs/pull/127) | PR B | Delete the legacy `ComputeBackend` / `KernelOps` / `model_builder` / memory modules from ferrum-interfaces (had no implementations after Phase 3e). Drop the dead `MetalKernelOps` shell, the `ferrum-engine::backends::candle` `ComputeBackend` impl, the dead `custom_backend` builder field, and `ferrum-models::{builder,weights}`. |
| [#128](https://github.com/sizzlecar/ferrum-infer-rs/pull/128) | PR C | Merge `ferrum-attention` into `ferrum-kernels::attention`. The crate's only consumers were `ferrum-kernels::backend::metal` and `ferrum-models` (Qwen3-TTS); merging removes a duplicated metal feature pass-through and the cudarc-0.12 stub. Two-cudarc-version workspace gone. |
| [#129](https://github.com/sizzlecar/ferrum-infer-rs/pull/129) | cleanup | Drop dead `LinearFactory` trait + `DefaultLinearFactory` (zero callers anywhere). |
| [#131](https://github.com/sizzlecar/ferrum-infer-rs/pull/131) | Dim 5 INT8 KV step 1 | CUDA INT8 KV kernels: `paged_decode_attention_int8` (read INT8 + per-token FP16 scale, dequantize on the fly) and `int8_kv_cache_append` (FP16 → per-(token, kv_head) symmetric INT8 + scale). Rust launchers in `ferrum-kernels::int8_kv`. `BackendKvDtype<KvInt8>` marker on `CudaBackend`. Host-reference parity test: cosine 0.99999 vs FP32 ref. |

### Backend trait shrinkage

| Phase | Method count | Δ |
|---|---|---|
| Pre-Phase-2 | 94 | — |
| 2.1 + 2.2 (#107) | ~82 | -12 |
| 2.3 (#108) | ~62 | -20 |
| 2.4 (#109) | ~32 | -30 |
| 3e/2 (#123) | ~30 | -2 (gemm_gptq + gemm_gptq_with_offset; +make_stacked_expert_linear) |
| 3e/3 (#124) | ~29 | -1 (gemm_quant; load_quant return type changed) |

### Remaining work

| Phase | Scope | Risk | Estimate |
|---|---|---|---|
| 4 INT8 KV — model wire-up | Switch a model's `KvCache<B>` to `KvCache<B, KvInt8>` behind an env var (`FERRUM_INT8_KV=1`), call the INT8 launchers from `ferrum-kernels::int8_kv` in the decode/append path, run a parity bench against FP16 (Llama-8B INT4 + INT8 KV vs FP16 KV). | Low (kernels already validated) | 1 day. |
| FP8 KV | Add `paged_decode_attention_fp8` + `fp8_kv_cache_append` mirroring the INT8 pair (use `__nv_fp8_e4m3` storage and per-token f8 scale). Marker impl `BackendKvDtype<KvFp8>` on CudaBackend. | Medium (FP8 needs SM ≥ 8.9) | 1-2 days. |

## GPU testing workflow (Vast.ai 4090)

CUDA-only validation runs on Vast.ai rented GPUs. Mac local handles
Metal + CPU; Vast pod handles CUDA. Cost ~$0.30/hr for an RTX 4090.

### Spin up a pod

Use the Vast HTTP API (the `vastai` CLI has a urllib3 incompatibility on
macOS LibreSSL):

```bash
source .env.local  # exports VAST_API_KEY + HF_TOKEN

# Find cheapest 4090 (filter rentable + verified preferred)
curl -s "https://console.vast.ai/api/v0/bundles/?q=%7B%22gpu_name%22%3A%7B%22eq%22%3A%22RTX%204090%22%7D%2C%22rentable%22%3A%7B%22eq%22%3Atrue%7D%2C%22dph_total%22%3A%7B%22lt%22%3A0.40%7D%7D&order=%5B%5B%22dph_total%22%2C%22asc%22%5D%5D" \
  -H "Authorization: Bearer $VAST_API_KEY" | jq '.offers[0:3] | .[] | {id, dph_total, cuda_max_good, geolocation}'

# Create instance — image must be nvidia/cuda:*-devel-ubuntu22.04
# (NOT pytorch/pytorch:* — that pulls 5-10 GB of unused ML deps)
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
curl -X PUT "https://console.vast.ai/api/v0/asks/<OFFER_ID>/" \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json; print(json.dumps({
    'client_id': 'me',
    'image': 'nvidia/cuda:12.4.0-devel-ubuntu22.04',
    'env': {'-p 22:22': '1'},
    'disk': 60,
    'runtype': 'ssh ssh_proxy',
    'onstart': 'apt-get update -qq && apt-get install -y -qq openssh-server pkg-config libssl-dev git build-essential && mkdir -p /run/sshd /root/.ssh && echo \"$PUBKEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd -D &',
    'label': 'ferrum-cuda-validation',
}))")"
```

Notes:
- ~30% of new pods hit a Vast SSH proxy port-forwarding bug — destroy
  + retry if SSH refuses for >2 min after `actual_status=running`.
- Pod returns `ssh_host` + `ssh_port` (e.g. `ssh1.vast.ai:19862`).

### Bootstrap on the pod

```bash
ssh -p <PORT> root@<HOST>

# One-time: install Rust + clone ferrum
curl -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
mkdir -p /workspace && cd /workspace
git clone https://github.com/sizzlecar/ferrum-infer-rs.git
cd ferrum-infer-rs

# HF token for gated models (Qwen3-30B-A3B-GPTQ-Int4 etc).
# Pipe via stdin (NOT --token CLI arg — that exposes in `ps`):
printf '%s\n' "$HF_TOKEN" | ssh -p <PORT> root@<HOST> \
  'mkdir -p /root/.cache/huggingface && cat > /root/.cache/huggingface/token && chmod 600 /root/.cache/huggingface/token'
```

### Build + bench

```bash
# Build (LTO + codegen-units=1 → ~5 min cold, ~3 min incremental)
cargo build --release --features cuda -p ferrum-cli

# For vLLM Marlin MoE path, add the feature:
cargo build --release --features cuda,vllm-moe-marlin -p ferrum-cli

# Standard smokes
HF_HOME=/workspace/.hf_home ./target/release/ferrum bench Qwen/Qwen3-0.6B \
    --rounds 3 --max-tokens 128

# Marlin INT4 on Llama-8B (no auth needed, hugging-quants mirror)
HF_HOME=/workspace/.hf_home ./target/release/ferrum bench \
    hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --rounds 3 --max-tokens 128

# MoE Marlin (gated; needs HF login above)
HF_HOME=/workspace/.hf_home FERRUM_VLLM_MOE=1 ./target/release/ferrum bench \
    Qwen/Qwen3-30B-A3B-GPTQ-Int4 --rounds 3 --max-tokens 64 --concurrency 4
```

### Background workflow (don't block on cargo build)

```bash
nohup bash -c "
    cargo build --release --features cuda -p ferrum-cli 2>&1 | tail -3
    HF_HOME=/workspace/.hf_home ./target/release/ferrum bench Qwen/Qwen3-0.6B --rounds 2 --max-tokens 64 2>&1 | tail -10
    echo __DONE__
" > /tmp/build_bench.log 2>&1 &
```

Then check periodically with `ssh ... 'tail /tmp/build_bench.log'`.

### Pod cost / cleanup

- Idle pod: ~$0.30/hr; destroy when not actively benching.
- Destroy: `curl -X DELETE "https://console.vast.ai/api/v0/instances/<INSTANCE_ID>/" -H "Authorization: Bearer $VAST_API_KEY"`
- Or via web console.

### Reference baselines (Vast 4090, post-Phase-3a/b main = `fff0759`)

| Model | Path | tok/s e2e | TPOT |
|---|---|---|---|
| Qwen3-0.6B | FP16 (safetensors) | 83.0 | 11.88 ms |
| Llama-3.1-8B-Instruct | GPTQ-Int4 (Marlin) | 60.4 | 16.55 ms |
| Qwen3-30B-A3B | GPTQ-Int4 (vLLM Marlin moe path, c=4) | 54.2 | 69.95 ms |
| Qwen3-30B-A3B | GPTQ-Int4 (IST-DASLab Marlin path, c=4) | 57.6 | 66.65 ms |

Note: c=4 isn't in the historical bench tables (`docs/bench/cuda-rtx4090-2026-05-09-pr-101-102/` only logged c=1/8/16/32). Treat
these as new baselines.

### Post-consolidation rerun (Vast 4090, main @ `44a87d0` after PRs #126-#129)

Acceptance gate 3 — same model, same comparison method, ≤ 3% drift.

| Model | Path | tok/s e2e (now) | Δ vs baseline | TPOT | Δ TPOT |
|---|---|---|---|---|---|
| Qwen3-0.6B | FP16 (safetensors) | 81.8 | -1.4% (within 3%) ✅ | 12.16 ms | +2.4% |
| Llama-3.1-8B-Instruct | GPTQ-Int4 (Marlin) | 63.0 | **+4.3% improvement** ✅ | 15.86 ms | -4.2% |
| Qwen3-30B-A3B | GPTQ-Int4 (vLLM Marlin moe, c=4) | 54.9 | +1.3% ✅ | 69.34 ms | -0.9% |

Marlin INT4 improvement comes from the Phase 3e/2 cutover — LTO inlines
`Box<dyn Linear<B>>` dispatch better than the previous
`#[cfg]`-branched trait method body. Metal locally: Qwen3-0.6B Q4_K_M
60.3 tok/s, matches pre-consolidation reading.

**Acceptance gate 3 passes.** Functionality + performance preserved
through PRs #126-#129. Only Dim 5 INT8 KV kernel work remains.

## References

- Audit memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/project_arch_audit_2026_05_09.md`
- Acceptance memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/project_arch_refactor_done_criteria.md`
- Vast workflow memory: `~/.claude/projects/-Users-chejinxuan-rust-ws-ferrum-infer-rs/memory/feedback_arch_refactor_validation.md`
