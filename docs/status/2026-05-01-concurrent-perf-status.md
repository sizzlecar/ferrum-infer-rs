# Concurrent Serving Performance — Status & Roadmap (2026-05-01)

## 目标 (Target)

让 `ferrum serve` 在 Apple Silicon (M1 Max 32 GB) 上跑 Group A 三个 GGUF
Q4_K_M 模型的并发推理性能 **持平或超越 llama.cpp** —— 这是衡量"我们能不能
在 Mac 上当生产引擎用"的硬指标。

具体的 Group A 模型：
- Llama-3.1-8B-Instruct Q4_K_M (4.9 GB, dense)
- Qwen3-8B Q4_K_M (5.0 GB, dense)
- Qwen3-30B-A3B Q4_K_M (18.6 GB, MoE — 128 experts × top-8)

具体的并发档位：c=1, 4, 8, 16 — 因为 16+ 在 32 GB Mac 上 paged 池就紧张了。

## 测试方法

### Bench harness (`bench/scripts/bench_serving.py`)

vLLM `benchmark_serving.py` 风格 —— 标准化的并发服务基准：

- 输入：N 个并发请求 burst（默认）或者 Poisson rate
- 协议：OpenAI 兼容的 `/v1/chat/completions` SSE 流
- 度量：每 token 到达时间戳 → TTFT / TPOT / ITL p50/p99 + request/output throughput
- 公平性：所有引擎跑完全相同的 prompt 集合（确定性 round-robin），同样
  的 max_tokens、temperature=0.0
- Qwen3 thinking-mode 兼容：把 `delta.reasoning_content` 也算 token，
  让 llama.cpp / mistralrs 的 chain-of-thought 输出和 ferrum 的 inline
  输出 token 数对齐

### 三个对手 / 引擎配置

| 引擎 | 启动命令 | 关键参数 |
|---|---|---|
| ferrum | `ferrum serve <gguf path>` | `FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=N FERRUM_KV_CAPACITY=` |
| llama.cpp | `llama-server` (homebrew ggml 0.10.0) | `--parallel N --batch-size 2048 --jinja` |
| mistralrs | `mistralrs serve text --format gguf` (cargo install mistralrs-cli) | `--max-seqs N`, paged-attn 默认 off on Metal |

## 核心流程 (Inference Flow)

```
HTTP /v1/chat/completions  (ferrum-server::axum_server)
   ↓ tokenize + chat template + InferenceRequest
ContinuousBatchEngine                       (ferrum-engine)
   ↓ run_iteration() 拿 iteration_lock
ContinuousBatchScheduler.next_batch         (ferrum-scheduler)
   ↓ 返回 BatchPlan { prefill_ids, decode_ids } 混合批
process_batch:
  • 新请求 → run_prefill (per-id)
  • 已 prefill 的 → run_batch_decode (M ≥ 2 走 decode_batch)
LlmExecutor                                 (ferrum-models::executor)
   ↓ 全部 decode 一次 lock 进 model
DecoderOnlyLLM::decode_batch                (ferrum-models)
   ↓ 分发到具体模型
LlamaFamilyModel<B>::decode_batch_internal  ← Phase 4b 的 dense 完胜路径
   或 Qwen3MoeModel<B>::decode_batch        ← Phase 4b 的 MoE 部分胜利
   ↓ 每层 batched GEMM + 单 batch attention dispatch + (MoE) 专家分派
Backend<B>                                  (ferrum-kernels)
   ↓ Metal kernels: paged_decode_attention / gemm_q4kw_moe_id / …
返回 [M, vocab] 的 logits → sample → push StreamChunk → SSE
```

## 当前成绩

测试日期：2026-05-01；硬件 M1 Max 32GB；max_tokens=128（8B）/ 64（30B）；
deterministic prompts；temperature=0。

### Dense 8B Q4_K_M @ c=16（输出吞吐 tok/s）

| 模型 | ferrum | llama.cpp | mistralrs |
|---|--:|--:|--:|
| Llama-3.1-8B-Instruct | **104.7** | 74.8 | 21.7 |
| Qwen3-8B | **101.2** | 71.7 | 24.9 |

**Dense 完胜：+40% over llama.cpp，~4–5× over mistralrs。**

完成率：ferrum 32/32（c=16 burst）；llama.cpp 26-28/32（`--parallel`
硬槽位拒收）；mistralrs 32/32 但 TTFT p50 飙到 43,691 ms。

### MoE 30B-A3B Q4_K_M（输出吞吐 tok/s）

| c | ferrum (默认) | ferrum (FERRUM_MOE_BATCHED=1) | llama.cpp |
|--:|--:|--:|--:|
| 1 | 43.7 | 43.7 | 50.6 |
| 4 | 46.6 | 46.6（per-item 走默认） | 63.0 |
| 8 | 49.2 | 49.2（per-item 走默认） | 74.4 |
| 16 | 44.6 | **51.3 (+15% over 我们默认)** | **95.4** |

**MoE 落后：c=16 我们 51.3 vs llama.cpp 95.4 = 我们只有他们 54%。**

## 问题诊断

### 为什么 Dense 完胜，MoE 落后？

**Dense 的瓶颈在 dense GEMM 的 batch 红利**：
- 每 token 全部计算就是 qkv_proj / o_proj / gate_up_proj / down_proj
- m=1 的 gemv 在 c=16 时变成 m=16 的 gemm —— **weight 一次读、batch 维度复用** 16 倍
- Phase 4b 又把 M 次 attention dispatch 合并成一次
- llama.cpp 的 `--parallel` 是固定 slot，每 slot 独立 m=1，**它根本没把 16 个 token 真正 batch 进 m=16 GEMM**
- → 我们的 batch 更彻底，所以赢

**MoE 的瓶颈在 sparse expert routing 的 kernel craft**：
- 每 token = 1/3 dense GEMM + 2/3 MoE FFN
- M=16 个 token 选的 top-8 experts 是不一样的，要 bucket by expert 后 indirect dispatch
- llama.cpp 的 `kernel_mul_mm_id` 是世界级 Metal kernel，调了一年多
- 我们的 `gemm_q4kw_moe_id_f32` 已经是 port 自他们的，但 kernel-craft 上还差 ~2×

### Profile 数据（Qwen3-30B-A3B c=8 batched, FERRUM_DECODE_OP_PROFILE=1）

```
[decode-prof] total=134 ms
  attn = 18 (14%)
  moe  = 89 (66%)  ← 决定性瓶颈
    route=13  gate=16  up=16  silu=10  down=19  wsum=13
  other = 22 (16%)  ← dispatch + host overhead
  lmhead = 3
```

3 个 GEMM（gate / up / down）合计 51 ms = MoE 时间的 60% = 总时间的
38%。这就是 ~2× 差距的物理位置。

### 已经验证的"非问题"

- ✅ Kernel structure 已经 port 自 llama.cpp `kernel_mul_mm_q4_K_f32` —
  相同 NR0=64 / NR1=32 / NK=32，相同 4 simdgroups × 32 threads
- ✅ Threadgroup memory 8 KB（高 occupancy）
- ✅ Indirect dispatch 通过 `compute_ids_tpe_gpu` 已经把 grid 缩到
  `max(tpe[e])` —— 不是 over-dispatch
- ✅ 单 command buffer / sticky encoder 跨整个 forward —— 不是 commit overhead
- ✅ Threshold tuning 已经穷尽（T=8 反而比 T=12 慢，T=16 不变）

### 还没验证、需要 Xcode 才能查的

- ❓ Threadgroup memory bank conflict 模式（不同 lx/ly 索引会有不同的 bank pattern）
- ❓ Register pressure 在 inner-K 的 unroll 16 阶段
- ❓ Metal 编译器对不同 indexing 表达式的优化差异
- ❓ GPU occupancy 在 indirect dispatch 时的实际驻留 simdgroup 数

这些只能用 Xcode Metal Frame Capture 看 instruction-level perf。**这是
本 session 解决不了的**，需要单独开一个多日 session 边 frame capture
边迭代 kernel。

## 解决办法（按 ROI 排）

### Tier 1（中等难度，预期收益大）

#### 1. Offset-aware MoE kernels（消除 per-item path 的 copy_slices）

当前 `gemv_quant_moe_id` API 强制每次读 `ids` / `weights` / activation
都从 offset 0 开始，所以 per-item batched decode 路径每个 item 要做 4
次 `copy_slice`：

```rust
// per item i in batched decode loop:
copy_slice(selected_ids_buf, i*top_k, ids_buf, 0, top_k)     // 1
copy_slice(weights_2d, i*top_k, weights_buf, 0, top_k)        // 2
copy_slice(norm_out, i*h, x_single, 0, h)                     // 3
copy_slice(acc_buf, 0, moe_out, i*h, h)                        // 4
```

c=16 / 48 layers = **3,072 dispatches/token 纯浪费**。每 dispatch
~10 µs CPU + 微秒级 GPU launch overhead。

**修复**：给 kernel API 加 offset 参数，同 buffer + offset 直接读：

```rust
B::gemv_quant_moe_id(
    ctx, x: &Buffer, x_offset: usize,
    weight: &QuantStore,
    ids: &Buffer, ids_offset: usize,
    out: &mut Buffer, out_offset: usize,
    n_selected, src1_stride,
)
```

预期：c=16 推到 ~70 tok/s（从 51.3）。**~2 天工作**，不需要重写 kernel。

#### 2. `weighted_sum_residual_norm_batched`（恢复 batched 模式的 cross-layer fusion）

当前 m=1 路径有这个 fused kernel，把 `weighted_sum_residual` 和下一层
的 `rms_norm` 合并成一个 dispatch。Batched 模式没有，每层多 1 次 dispatch。

48 layers × 1 dispatch/layer × ~10 µs = ~0.5 ms/token saved。小但
免费。**~1 天工作**。

### Tier 2（硬骨头，预期决定性）

#### 3. `mul_mm_id` Metal kernel rewrite 对标 llama.cpp

**这是闭合 30B-A3B c=16 那 46% gap 的唯一路径**。需要：

a. **Xcode Metal Frame Capture**：
   - 抓 ferrum 30B-A3B c=16 一个 token 的完整 GPU trace
   - 抓 llama.cpp 同样工作负载的 trace
   - 在 Xcode "Performance" 面板对比：
     - Per-kernel GPU 时间
     - Stall reasons (memory bank conflict / instruction dependency / register pressure)
     - Occupancy
     - Memory bandwidth

b. **逐项消除差异**：
   - 比如如果 bank conflict 是主因 → 重排 shmem 索引
   - 如果 register pressure 是主因 → 调 NR0/NR1 tile 大小或 unroll 因子
   - 如果 occupancy 低 → 调 threadgroup size / shmem 用量

c. **正确性回归**：每改一次 kernel 都要跑 30B-A3B 测试 token 输出对得上

预期：c=16 推到 90+ tok/s，闭合到 llama.cpp 的水平。**3-7 天专心做 Metal 工作**。

#### 4. Gate+up+silu fusion

把当前三个独立 dispatch（gate gemv, up gemv, silu_mul_stacked）合并成
一个 kernel，读 activation 一次、写最终 silu(gate)·up 直接出。bandwidth
省 1×、launch 省 2×。

预期：c=16 +5-10%。**~2-3 天**。

### Tier 3（已经做完 / 不需要做）

- ✅ Phase 4b dense `LlamaFamilyModel` 全 batched paged dispatch（merged #73，确认 +40%）
- ✅ GGUF 端到端入口（merged #74）
- ✅ Phase 4b MoE 框架 + threshold hybrid（merged #77，c=16 +15% over 默认）
- ✅ vLLM 风格 bench harness（merged #75）
- ⚠️ Concurrent encoder 模式（PR #65 NULL RESULT，CPU CI 已修但不 merge）

## Roadmap 推荐

按 ROI 顺序，每个都是独立 PR：

1. **【现在最高 ROI】Offset-aware MoE kernels** —— 不需要 Xcode，
   ~2 天工作，c=16 从 51 推到 ~70。落地后 ferrum 在 30B-A3B 也接近
   llama.cpp。
2. **`weighted_sum_residual_norm_batched`** —— 1 天，小幅改进，结构
   性收尾。
3. **长上下文 + ShareGPT bench** —— 1-2 天，用 vLLM 标准数据集喂
   harness，把 dense 完胜在更真实负载下展示。
4. **Speculative decoding for Metal** —— 1-2 天，CUDA 已经做过、port
   到 Metal。Qwen3-0.6B 当 draft，4B/8B 当 target，预期 c=1 加速 1.5-2×。
5. **【硬骨头，单独大块时间】`mul_mm_id` 重写** —— 3-7 天 Metal 专心
   工作，需要 Xcode。这才是真正闭合 MoE 30B-A3B 差距的路径。

## 记录的死路（don't go back）

- **threshold tuning 穷尽**：T=8 / T=12 / T=16 都试过，T=12 是最优
- **`moe_forward_batched_prefill_impl` 直接复用**：bucketing overhead 在
  decode-time 小 M 太大，需要 per-item 路径配合
- **lx/ly swap 实验性 port to llama.cpp 的 layout**：单纯 swap 让输出全
  是 `]]]` —— layout 不只 swap 这么简单，整个 simdgroup_load 配套都
  得改
- **Qwen3MoeModel 加 paged KV**：不必要 ——`Qwen3MoeModel::ensure_kv` 是
  contig-only 的，paged 开关不影响它，dispatch 仍正确

## 已合并的 PR 链（按时间顺序）

| PR | 标题 | 主要价值 |
|---|---|---|
| #73 | perf(metal): batched paged-KV decode dispatch (Phase 4b) | dense 8B c=16 +40% |
| #74 | feat: GGUF model support in serve/bench/pull | Group A 模型可服务 |
| #75 | bench: Group A concurrent HTTP report | 三引擎并发对比报告 |
| #76 | bench: Qwen3-30B-A3B Q4_K_M concurrent results | 30B 数据 + 修订 |
| #77 | perf(metal): Phase 4b for Qwen3-MoE | MoE c=16 +15% |

## 附录：bench 复现命令

```bash
# Build
cargo build --release --features metal -p ferrum-cli

# ferrum
FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=16 FERRUM_KV_CAPACITY=1024 \
FERRUM_MAX_BATCH=16 ./target/release/ferrum serve \
  --model ~/ferrum-bench/models/Qwen3-8B-Q4_K_M.gguf --port 8000 &

# llama.cpp
llama-server --model ~/ferrum-bench/models/Qwen3-8B-Q4_K_M.gguf \
  --port 8001 --ctx-size 4096 --parallel 16 --batch-size 2048 --jinja &

# mistralrs
mistralrs serve --port 8002 --max-seqs 16 text --format gguf \
  -m ~/ferrum-bench/models -f Qwen3-8B-Q4_K_M.gguf \
  -t ~/ferrum-bench/tokenizers/Qwen3-8B.tokenizer.json &

# Bench (point --base-url at each engine in turn)
python3 bench/scripts/bench_serving.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3-8B-Q4_K_M \
  --num-prompts 32 --max-concurrency 16 --max-tokens 128 \
  --deterministic-prompts \
  --result-file out.json
```
