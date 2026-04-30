# Group A 性能报告 — Apple Silicon (M1 Max 32 GB)

**日期**: 2026-04-30 (更新, after PR #58) · **GGUF Q4_K_M 同一文件三引擎共用** · **Phase 1**（单请求路径）

> Phase 1 完成：单请求 pp50 / pp512 / tg128 + TTFT 估算。Phase 2（16 并发吞吐）待做。

## 摘要

| | Qwen3-8B Q4_K_M (4.7 GiB) | Llama-3.1-8B Q4_K_M (4.6 GiB) | Qwen3-30B-A3B Q4_K_M (17.3 GiB) |
|---|---:|---:|---:|
| **ferrum tg128** | 27.0 | 28.3 | **38.3** ← +31% (#58) |
| **mistral.rs tg128** | 35.8 | **37.5** | N/A (不支持 MoE GGUF) |
| **llama.cpp tg128** | 30.2 | 31.6 | **52.2** |
| **decode 排序** | mistral > llama > ferrum | mistral > llama > ferrum | llama > **ferrum** |

**一句话总结**：在 M1 Max 上，ferrum prefill 维持 96% × llama.cpp（30B-A3B），decode 经 PR #58 wsum kernel reshape 后 30B-A3B 从 53% 提到 **73%**；8B dense 模型 ferrum 仍在 89% 左右（这条路径 PR #58 不影响）。

## 单请求吞吐 (tok/s, 3 trials 中位数)

| 模型 | 指标 | ferrum | mistral.rs | llama.cpp |
|---|---|---:|---:|---:|
| Qwen3-8B | pp50 | 177.0 | 208.1 | **249.4** |
| Qwen3-8B | pp512 | 301.4 | 283.5 | **360.3** |
| Qwen3-8B | tg128 | 27.0 | **35.8** | 30.2 |
| Llama-3.1-8B | pp50 | 182.1 | 214.7 | **250.8** |
| Llama-3.1-8B | pp512 | 301.3 | 294.6 | **363.2** |
| Llama-3.1-8B | tg128 | 28.3 | **37.5** | 31.6 |
| Qwen3-30B-A3B | pp50 | 118.0 | — | **194.0** |
| Qwen3-30B-A3B | pp512 | 568.9 | — | **596.9** |
| Qwen3-30B-A3B | tg128 | **38.3** | — | **52.2** |

## TTFT 估算 (50-token prompt, ms)

> TTFT(50) = 50 / pp50 + 1 / tg128 (ms)（prefill 完后 + 1 个 decode tick）

| 模型 | ferrum | mistral.rs | llama.cpp |
|---|---:|---:|---:|
| Qwen3-8B | 319.5 | 268.2 | **234.5** |
| Llama-3.1-8B | 309.8 | 259.3 | **231.0** |
| Qwen3-30B-A3B | 449.8 | — | **276.9** |

## 30B-A3B decode 演进史

PR | tg128 (t/s) | vs llama.cpp 52.2 | gap closed
---|---:|---:|---:
PR #50 (zero-copy mmap) baseline | ~27 | 52% | —
PR #57 (decode profile) | 29.2 | 56% | +4 pp
**PR #58 (wsum 32→256 threads)** | **38.3** | **73%** | **+17 pp** in one PR

关键洞察来自 PR #57 的 per-stage profile：cross-layer-fused MoE tail kernel `weighted_sum_residual_norm_stacked` 在 1 TG × 32 threads 配置下把 GPU 闲置 7/8（M1 Max 多个 compute unit 只有 1 个 simdgroup 在跑）。改成 1 TG × 256 threads（8 simdgroups + threadgroup 内 sumsq 跨 simdgroup reduce）后该 kernel 时间 -69%，整体 decode +31%。

## 重点观察

1. **ferrum 30B-A3B prefill 仍接近 llama.cpp**（568.9 vs 596.9 = 95%）— PR #54/#55/#56 累计成果。
2. **30B-A3B decode 从 53% 提到 73% × llama.cpp**（PR #58）。decode 路径不再「无视 active 参数变少的红利」——38.3 t/s 显著超过 dense 8B 的 27-28 t/s，符合 MoE 只激活 3B 参数的预期。
3. **mistral.rs 在 dense 8B 模型 decode 最快**（35.8-37.5 t/s）。其 v0.8.0 GGUF loader 启动 Qwen3-MoE 时 dummy run 崩溃 — 当前对 qwen3moe arch 支持有限。
4. **ferrum 8B dense decode 27-28 t/s**（vs llama.cpp 30-32, mistral.rs 35-37）。这是独立的 dense LlamaFamily 路径，不走 MoE 优化。下一刀的目标。
5. **ferrum pp50 偏弱**（30B 上 118 vs 194，8B 上 177 vs 249）— 50 token 还摊不开 ferrum 的启动开销。pp512 时 overhead 摊销，差距收敛到 84-95%。

## 内存带宽分析（更新）

M1 Max 实测有效内存带宽 ≈ 150 GB/s。

| 模型 | active params | 单 token 读取量 (Q4) | llama.cpp BW 利用 | ferrum BW 利用 |
|---|---:|---:|---:|---:|
| Qwen3-8B (dense) | 8 B | ~4 GB | 30×4 = 121 GB/s (81%) | 27×4 = 108 GB/s (72%) |
| Llama-3.1-8B (dense) | 8 B | ~4 GB | 32×4 = 126 GB/s (84%) | 28×4 = 113 GB/s (75%) |
| Qwen3-30B-A3B (MoE) | 3 B active | ~1.5 GB | 52×1.5 = 78 GB/s (52%) | **38×1.5 = 57 GB/s (38%)** |

**Dense 8B**: ferrum 在 72-75% BW 利用率（vs llama.cpp 81-84%）。差距 ~12%。下一步：dense FFN / o_proj 路径。

**MoE 30B-A3B**: ferrum 从 29% → **38% BW 利用率**。距 llama.cpp 52% 仍差 ~14 pp，gate / up / down GEMV 的 dispatch 调度、KV cache 读取 pattern 是下一战场。

## 方法论

（与 PR #56 一致 — 见 [bench/README.md](README.md)）

- **硬件**: MacBook Pro M1 Max, 32 GB unified memory, macOS 15.1.1 (Build 24B91), Apple7 GPU family (Metal 3，**不支持 Metal 4 tensor matmul**)
- **后端**: Metal (所有引擎，统一 GPU 路径)
- **量化**: Q4_K_M（同一份 GGUF 文件三引擎共用，规避 quantization 实现差异）
- **测试串行**: 每次只有一个推理进程在跑（Python 子进程退出 = 干净内存释放）
- **三个 workload**:
  - **pp50** = 50-token "the the the…" 提示 + 1 token 解码，测 prefill 速率（tok/s）
  - **pp512** = 512-token "the" 重复提示 + 1 token 解码（tok/s）
  - **tg128** = "Once upon a time" 短提示 + 128 tokens 解码，测 decode 速率（tok/s）
- **采样**: `temperature=0.0`（确定性输出），`max_tokens=N` 硬截断
- **每条 3 trials 取中位数**

## 测试环境

```
hardware:
  cpu     : Apple M1 Max (10 cores)
  memory  : 32 GB unified memory
  gpu     : Apple M1 Max (Metal 3, NOT Metal 4)
  os      : macOS 15.1.1 (Build 24B91)

ferrum:
  commit  : eb5b12c (perf(metal): widen weighted_sum_residual_norm_stacked from 32 → 256 threads (#58))
  rustc   : 1.91.0 (f8297e351 2025-10-28)
  build   : cargo build --release -p ferrum-cli --features metal

llama.cpp:
  source  : Homebrew, llama.cpp v8960 (libggml 0.10.0)
  binary  : /opt/homebrew/bin/llama-bench
  note    : "tensor API disabled for pre-M5 and pre-A19 devices" — pre-Metal-4 Apple

mistral.rs:
  source  : PyPI, mistralrs-metal==0.8.0 (cp310-abi3-macosx_15_0_arm64 prebuilt wheel)
  python  : 3.12.12 (uv venv at /tmp/mistral_bench)
  note    : Qwen3-MoE GGUF loader 启动 dummy run 崩溃 — 已知限制
```

## 复现命令

完整步骤见 [`bench/README.md`](README.md)。简版：

```bash
# 装 mistral.rs（其它两个已就绪）
uv venv --python python3.12 /tmp/mistral_bench
uv pip install --python /tmp/mistral_bench/bin/python mistralrs-metal==0.8.0

# 跑 bench（每模型 ~5-15 分钟，串行）
SCRIPT="$(pwd)/bench/scripts/bench_one_model.sh"
OUTDIR="$(pwd)/bench/group-a-results"
$SCRIPT Qwen3-8B-Q4_K_M.gguf                    Qwen3-8B.tokenizer.json                    $OUTDIR
$SCRIPT Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  Meta-Llama-3.1-8B-Instruct.tokenizer.json  $OUTDIR
$SCRIPT Qwen3-30B-A3B-Q4_K_M.gguf               Qwen3-30B-A3B.tokenizer.json               $OUTDIR

# 聚合
/tmp/mistral_bench/bin/python bench/scripts/aggregate_bench.py > bench/group-a-report.md
```

## 待办

- **Phase 2: 16 并发 throughput** — 启 HTTP server，16 路并发 chat completion，测聚合 t/s。30B-A3B 在 32 GB Mac 上 16 路 KV ~12 GB + 模型 17 GB ≈ 内存上限，可能要降到 4 并发。
- **Phase 3: 继续 decode 攻坚** — 30B-A3B 还差 14 pp BW 利用率到 llama.cpp。8B dense 也有 ~10% gap。
  - gate / up / down q4_K_MOE_ID GEMV 的 dispatch 调度（同 kernel，不同 feeding pattern）
  - dense 8B：q4_K_v2 GEMV 的 N_R0 / N_SG 配置 + KV cache 读取
  - 可能值得尝试 Metal counter sample buffer 拿真 GPU 时间（不带 sync 噪声）

## 历史快照

- 2026-04-30 初版 (PR #56): ferrum 30B-A3B tg128 = 29.2 t/s (53% × llama.cpp)
- **2026-04-30 本次更新 (PR #58 后)**: ferrum 30B-A3B tg128 = **38.3 t/s (73% × llama.cpp)**
- 旧 raw output 归档在 `bench/group-a-results.pre-wsum/`

## 原始数据

`bench/group-a-results/` 下每模型三个文件（每条都是 3 trials 的原始输出）：

- `<model>__llamacpp.txt`
- `<model>__ferrum.txt`
- `<model>__mistralrs.txt`

聚合脚本：`bench/scripts/aggregate_bench.py`。
