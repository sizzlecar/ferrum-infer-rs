# Group A 性能报告 — 2026-05-01 (after PR #66 sdpa_vector)

**硬件**: MacBook Pro M1 Max 32 GB / macOS 15.1.1 / Apple7 GPU family
**后端**: Metal（三引擎统一）· **量化**: Q4_K_M（同一份 GGUF 文件三引擎共用）
**ferrum 分支**: `perf/metal-sdpa-vector` (PR #66) · 含累计 PRs: #50 zero-copy, #54-#58 prefill/wsum, #62 prepare warmup, **#66 MLX-style sdpa_vector decode**

> 上一轮报告: [bench/group-a-report.md](group-a-report.md) — 2026-04-30, 不覆盖, 留作对比基线。

## 摘要 — ferrum 在 30B-A3B 三项全部追平/反超 llama.cpp

| 模型 | 指标 | ferrum (本轮) | ferrum (旧) | Δ | mistral.rs | llama.cpp | ferrum vs llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-30B-A3B** | pp50 | **220.4** | 118.0 | **+87%** | — | 202.2 | **109% ✓** |
| **Qwen3-30B-A3B** | pp512 | **647.7** | 568.9 | +14% | — | 620.5 | **104% ✓** |
| **Qwen3-30B-A3B** | tg128 | **53.8** | 38.3 | +40% | — | 54.2 | **99% (tie)** |
| Llama-3.1-8B | pp50 | 229.5 | 182.1 | +26% | 223.7 | 258.6 | 89% |
| Llama-3.1-8B | pp512 | 324.2 | 301.3 | +8% | 308.2 | 379.1 | 86% |
| **Llama-3.1-8B** | tg128 | **34.0** | 28.3 | +20% | 36.1 | 31.8 | **107% ✓** |
| Qwen3-8B | pp50 | 222.3 | 177.0 | +26% | 215.9 | 257.3 | 86% |
| Qwen3-8B | pp512 | 323.0 | 301.4 | +7% | 289.1 | 375.5 | 86% |
| Qwen3-8B | tg128 | 32.3 | 27.0 | +20% | 37.9 | 35.5 | 91% |

**一句话总结**：PR #66（MLX `sdpa_vector` 移植到 ferrum-attention）让 decode m=1 的 attention kernel 从 32 线程（1 simdgroup, 3% GPU 占用）扩到 1024 线程（32 simdgroups, ~满占用），三个模型 tg128 平均 +25%。30B-A3B decode 从 73% 追到 99% × llama.cpp，**8B Llama 反超 llama.cpp**（34.0 vs 31.8）。剩余 gap 在 8B 模型 vs mistral.rs（~5%），prefill vs llama.cpp（~14%）。

## 完整数据 (tok/s, 3 trials median)

| 模型 | 指标 | ferrum | mistral.rs | llama.cpp |
|---|---|---:|---:|---:|
| Qwen3-8B | pp50 | 222.3 | 215.9 | **257.3** |
| Qwen3-8B | pp512 | 323.0 | 289.1 | **375.5** |
| Qwen3-8B | tg128 | 32.3 | **37.9** | 35.5 |
| Llama-3.1-8B | pp50 | 229.5 | 223.7 | **258.6** |
| Llama-3.1-8B | pp512 | 324.2 | 308.2 | **379.1** |
| Llama-3.1-8B | tg128 | 34.0 | **36.1** | 31.8 |
| Qwen3-30B-A3B | pp50 | **220.4** | N/A | 202.2 |
| Qwen3-30B-A3B | pp512 | **647.7** | N/A | 620.5 |
| Qwen3-30B-A3B | tg128 | 53.8 | N/A | **54.2** |

> **mistral.rs 在 Qwen3-30B-A3B 上**：v0.8.0 GGUF loader 加载 30B-A3B MoE 时 dummy run 触发 `Engine dead, rebooting` 后 trial 全部空数据。同上一轮，未修。

## TTFT 估算 (50-token prompt, ms) = 50/pp50 + 1/tg128

| 模型 | ferrum | mistral.rs | llama.cpp |
|---|---:|---:|---:|
| Qwen3-8B | 255.9 | 258.0 | **222.5** |
| Llama-3.1-8B | 247.3 | 251.2 | **224.8** |
| Qwen3-30B-A3B | **245.4** | — | 265.7 |

**Qwen3-30B-A3B TTFT ferrum 反超 llama.cpp** —— prefill 端 +87% pp50 + +40% tg128 双重红利。

## 30B-A3B 演进史（累计）

| PR | tg128 (t/s) | pp50 (t/s) | vs llama.cpp tg128 | gap closed |
|---|---:|---:|---:|---:|
| baseline (PR #50 zero-copy) | ~27 | ~118 | 52% | — |
| PR #57 (decode profile) | 29.2 | — | 56% | +4 pp |
| PR #58 (wsum 32→256 threads) | 38.3 | 118.0 | 73% | +17 pp |
| PR #62 (prepare warmup) | 38.0 | 196.0 | 73% (decode 不变) | pp50 +66% |
| **PR #66 (sdpa_vector)** | **53.8** | **220.4** | **99%** | **+26 pp** |

PR #66 是 PR #58 之后最大的单跳。两者都是同一个发现模式：**单 dispatch 内 GPU 严重欠占用，widen 线程数把闲置 simdgroup 用上**。区别在于：
- PR #58 widen 的是 wsum kernel（1 个 TG，32→256 threads）— 8× ALU 利用
- PR #66 widen 的是 attention kernel（每 head 1 TG，32→1024 threads）— 32× 线程并行

## 关键洞察 — sdpa_vector 是怎么找到的

直接看 mistral.rs 的代码。`mistralrs-core/src/attention/mod.rs:235-254`：

```rust
if [q, k, v].into_iter().all(|x| x.device().is_metal())
    && all_head_dims_match
    && valid_head_dims.contains(&head_dim)
{
    return candle_nn::ops::sdpa(q, k, v, mask.as_ref(), false, ...);
}
```

mistral.rs 走 candle 的 `ops::sdpa`，candle 走的就是 `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal` —— 这是 Apple MLX 框架的 `sdpa_vector` kernel 移植。

对比 ferrum 旧的 `flash_attn_f32` (decode m=1 路径)：
- **TG = 32 threads = 1 simdgroup**，每 (head, batch) 一个 TG
- KV 位置在单个 simdgroup 内**串行**循环

8B Llama 32 heads × 32 threads = **1024 个活跃线程**，M1 Max 能并发 ~32K 线程 → **GPU 利用率 ~3%**。

MLX `sdpa_vector` 设计：
- **TG = 32 simdgroups × 32 threads = 1024 threads**，每 (head, batch) 一个 TG
- 32 个 simdgroup **并行**处理不同 KV 位置（每个 SG 一个 KV，stride=32 循环）
- 跨 simdgroup 通过 threadgroup memory 用 online softmax rescale 合并 (max, sum_exp, output)

8B Llama 32 heads × 1024 threads = **32K 活跃线程**，跟硬件并发能力匹配 → **GPU 利用率 ~100%**。

PR #66 把这个 kernel 移植到 `ferrum-attention/src/metal/shaders/flash_attn.metal::flash_attn_decode_f32`，并在 `flash_attn_v2` 调度器加路由：q_len=1 + head_dim=128 + sliding_window=0 时启用，否则回退老路径。

## 仍然落后 / 未来工作

1. **8B 模型 prefill 落后 llama.cpp 14%**（pp512 324 vs 379）。可能是 candle prefill 路径 vs llama.cpp 自家 simdgroup_matmul 的差异。需要 Metal Frame Capture 定位。
2. **8B Qwen3 tg128 vs mistral.rs 落后 ~15%**（32.3 vs 37.9）。同样 sdpa_vector kernel 都用了，差异可能在 Q4_K GEMV / norm / SiLU 这些非 attention kernel 的细节。Llama 8B 上没这个差距（34.0 vs 36.1 = 6%）—— 可能是 Qwen3 的 QK-norm 额外开销在 ferrum 这边更贵。
3. **mistral.rs Qwen3-30B-A3B GGUF 加载仍然崩**。我们这边没有动力修上游 bug，但可以在 bench 报告里持续标注。
4. **Phase 2: 16 并发吞吐** 仍然 pending（task #18）。

## 内存带宽分析（更新）

M1 Max 实测有效内存带宽 ≈ 150 GB/s。

| 模型 | active params | 单 token Q4 字节 | llama.cpp BW% | ferrum BW% (旧) | ferrum BW% (新, PR #66) |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 8 B | ~4 GB | 95% (35.5) | 72% (27.0) | 86% (32.3) |
| Llama-3.1-8B | 8 B | ~4 GB | 85% (31.8) | 75% (28.3) | **91% (34.0)** |
| Qwen3-30B-A3B | 3 B active | ~1.5 GB | 54% (54.2) | 38% (38.3) | **54% (53.8)** |

**30B-A3B**: ferrum 从 38% → **54%** 带宽利用率，**完全追平 llama.cpp**。
**Llama-3.1-8B**: ferrum 75% → **91%**, **超过 llama.cpp 85%**。
**Qwen3-8B**: 86%, 比 llama.cpp 95% 仍略低 —— 这是上面提到的 Qwen3 特有路径差异。

## 环境监控

bench 全程使用 `bench/scripts/capture_env.sh` (PR #60) 记录 vm_stat / RSS / swap，每个引擎前后各一次。

**Swap delta**:
- Qwen3-8B: baseline 3647 MB → after 3647 MB（**0 MB delta**）
- Llama-3.1-8B: baseline 3647 MB → after 3647 MB（**0 MB delta**）
- Qwen3-30B-A3B: baseline 3647 MB → after 3982 MB（**+335 MB**）—— 主要发生在 mistral.rs 加载 17 GB MoE 文件时。**ferrum 和 llama.cpp 的 30B-A3B 数据在 mistral.rs 之前跑完，不受影响**。

> 上一轮报告也是这个模式：mistral.rs 30B-A3B 的 GGUF 加载会触发 swap，但 ferrum / llama.cpp 已经测完。这是为什么 PR #61 把 swap 检查改成了**增量**而不是绝对值 —— 否则每次都会被旧 swap 误判。

## 复现

```bash
# 完整 Group A 重测
bash bench/scripts/bench_one_model.sh Qwen3-8B-Q4_K_M.gguf Qwen3-8B.tokenizer.json bench/group-a-results-XXX
bash bench/scripts/bench_one_model.sh Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf Meta-Llama-3.1-8B-Instruct.tokenizer.json bench/group-a-results-XXX
bash bench/scripts/bench_one_model.sh Qwen3-30B-A3B-Q4_K_M.gguf Qwen3-30B-A3B.tokenizer.json bench/group-a-results-XXX

# 聚合
RESULTS_DIR=bench/group-a-results-XXX python3 bench/scripts/aggregate_bench.py > bench/group-a-report-XXX.md

# 对比 ferrum 新旧 (回退 PR #66 跑老 SDPA)
FERRUM_FA_DECODE=0 ./target/release/ferrum run model.gguf --tokenizer tok.json \
  --prompt "Once upon a time" --max-tokens 128 --temperature 0.0 --bench-mode
```

环境关键变量：
- `FERRUM_FA_DECODE=0` — 关闭 PR #66 sdpa_vector，回退 32-thread 老路径（用于回归测试 / 对比）
- `FERRUM_FA_LEGACY=1` — 全部用最老的 scalar flash_attn（连 Q-tiled prefill 都关）
- `FERRUM_KV_CAPACITY` — KV cache 槽位数；30B-A3B 测试用 512（短 prompt + 128 decode 足够），prefill 测试用 2048
