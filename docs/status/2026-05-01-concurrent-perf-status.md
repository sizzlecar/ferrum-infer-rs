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

Xcode Metal Frame Capture 能直接看到这些 stall reason / instruction
schedule。**没 Xcode 也能做**——bench-driven 二分迭代每个变量，每改
一次跑 30B-A3B c=16 测 TPOT，收敛慢 5–10× 但能做。Xcode 是加速器
不是必需。

### 经验性更新 (2026-05-01 后续做的实验)

**实验：消除 per-item 路径的 4 个 copy_slice (PR #79)**
- 假设：copy_slice dispatch 是 c=4/8 per-item 路径的瓶颈
- 实施：加 offset-aware kernel API (`gemv_quant_moe_id_offset`,
  `weighted_sum_batched_offset`)，eliminate 4 个 copy_slice 每 iter
- 结果：c=4 41.6 → 42.4 (+1.9%), c=8 46.0 → 46.4 (+0.9%)
- 解读：**dispatch 数量不是真瓶颈**——Metal 的 `set_buffer(offset)`
  本来就快，被消除的 copy_slice 单个 ~µs 级。原始 +30% 估算是错的。
- 启示：**后续优化要 profile-driven**——不能再靠 dispatch count
  back-of-envelope 估算。真瓶颈要么是 per-kernel GPU 时间（kernel
  craft），要么是 memory bandwidth。

**实验：gate+up+silu 三 dispatch 融合 (PR — 待编号)**
- 假设：三个 dispatch 共用 norm_out / ids；中间 buffer
  (`gate_out_stacked`, `up_out_stacked`) 写出再读回 = 4× `[top_k, ffn]`
  intermediate bandwidth 浪费；融合能省 BW，预估 c=16 +5-15%
- 实施：新 Metal kernel `gemv_q4kw_moe_id_gate_up_silu_f32` —— 内层
  Q4_K decode loop 用同一份 `yl/yh` activation register-file，并行算
  gate 和 up 两个 row accumulator，最后 in-register `silu(g) * u` 写
  `silu_stacked`。Backend trait + 能力探针，默认 ON，
  `FERRUM_MOE_FUSED_GATE_UP_SILU=0` 可关
- 单元测试：fused vs unfused 在合成 Q4_K 权重上 **bitwise 一致**
  (max_abs=0.000000)
- 实测 c=4/8/16（30B-A3B Q4_K_M, prewarm cache）：

  | c | unfused | fused | Δ tok/s | TPOT median |
  |--:|--:|--:|--:|--:|
  | 4  | 42.5 | **43.5** | +2.4% | -2.7% (82.1→79.9 ms) |
  | 8  | 49.4 | **49.6** | +0.4% | -1.0% |
  | 16 | 50.8 | **51.2** | +0.8% | -0.5% |

- 解读：**实测 +0-2%，远小于 +5-15% 预期**。三个 dispatch 共用的
  中间 buffer (`gate_out_stacked`, `up_out_stacked`) 加起来才
  ~24 KB × 8 slots × 2 = ~384 KB —— 已经 fit Apple GPU L2/L3，写出
  再读回的 bandwidth 实际上没付 DRAM 代价。Fused 路径"省"的是
  cache→cache traffic，量级太小压不动 TPOT。
- 启示（与 PR #79 同源教训）：**dispatch / intermediate-BW 都不是
  真瓶颈**。剩下唯一没排除的是 per-kernel GPU 时间（kernel craft
  本身——bank conflict / register pressure / instruction schedule），
  这就是 Tier 2 `mul_mm_id` 重写的内容。后续 Tier 1 PR 价值低，
  应直接进 Tier 2。

## 解决办法（按实测 ROI 排）

### Tier 0（已完成）

#### 1. Offset-aware MoE kernels — PR #79

消除 per-item batched decode 路径的 4 个 `copy_slice`：用 offset
参数直接从 M-batched scratch buffer 的对应 row 读写。

实测：c=4 +1.9%, c=8 +0.9%。**比预估的 +30% 小一个量级**——见上面
"经验性更新"。结构性收益保留（kernel API 干净），perf 收益微薄。

#### 2. Gate+up+silu 融合（本 PR）

`gate_gemv + up_gemv + silu_mul` 三个 dispatch 合一，内层 Q4_K
reduction 共享 activation register-file，gate / up 两 row accumulator
并行，结果 in-register `silu(g) * u` 直写 `silu_stacked`。Backend
能力探针（Metal=true，CPU/CUDA=false），`FERRUM_MOE_FUSED_GATE_UP_SILU=0`
可关。

实测 c=4/8/16：+0.4-2.4%（c=4 最大 +2.4%）。**比预估的 +5-15% 小一
个量级**——见上面"经验性更新"。Bitwise correct，结构清爽，但 perf
价值微薄。

### Tier 1（残留候选，低 ROI 优先级）

#### 3. `weighted_sum_residual_norm_batched`

batched 模式恢复 m=1 路径已有的 cross-layer rms_norm fusion——把
`weighted_sum_residual` 和下一层 `rms_norm` 合并成一个 kernel。

预期：48 layers × 1 dispatch saved ≈ ~1% TPOT。小但免费。**~1 天**。

#### 4. NR0 / 大 tile 实验

每个 threadgroup 一次算 64×32 tile (NR0=64)。试 NR0=128：
- 半数的 threadgroup → 半数的 launch overhead
- 但每 TG 用 2× shared memory，可能降 occupancy

bench-driven A/B 测就能知道。**~1 天**。

### Tier 2（硬骨头，预期决定性）

#### 5. `mul_mm_id` / `gemv_quant_moe_id` Metal kernel rewrite

**这是闭合 30B-A3B c=16 那 46% gap 的真路径**。我们的 kernel 内部
速度比 llama.cpp 慢 ~2×（c=16 TPOT 313ms vs 139ms）。结构相同
（NR0=64 NR1=32 NK=32 4 simdgroups），差异在亚结构性的 craft。

**两条路都能走**：

**(a) 有 Xcode Frame Capture 的话**（快路径）：
- 抓 ferrum / llama.cpp 同 workload trace
- Xcode 直接显示 stall reasons / occupancy / bank conflict pattern
- 逐项消除——3-7 天能闭合差距

**(b) 没 Xcode 纯 bench-driven**（慢路径，但能走）：
- 假设清单：bank conflict / register pressure / instruction
  scheduling / shmem layout
- 每改一个变量，跑 30B-A3B c=16 测 TPOT
- 二分式收敛：先确认假设 (是不是 bank conflict?)，再优化
- 估计 1-2 周收敛到 +50%

正确性回归：每改一次跑 token-equivalence test (vs default 路径)。

预期：闭合到 llama.cpp 水平 (c=16 90+ tok/s)。**3-14 天专心做**。

### Tier 3（已合并 / 不必做）

- ✅ Phase 4b dense `LlamaFamilyModel` batched paged dispatch（PR #73，dense 8B c=16 +40%）
- ✅ GGUF 端到端入口（PR #74）
- ✅ Phase 4b MoE 框架 + threshold hybrid（PR #77，MoE c=16 +15% over 默认）
- ✅ vLLM 风格 bench harness + Group A 三引擎报告（PR #75 / #76）
- ✅ Offset-aware MoE kernels（PR #79，+1-2%）
- ✅ Gate+up+silu fusion（本 PR，c=4 +2.4% / c=8 +0.4% / c=16 +0.8%）
- ⚠️ Concurrent encoder 模式（PR #65 NULL RESULT，CPU CI 已修但不 merge）

## Roadmap 推荐（重排 — 实测后）

经过 PR #79 + 本 PR 两次 +1-2% 类的"经验性回收"，**结论已经清晰**：
30B-A3B c=16 那 46% gap 不可能由 host 侧 dispatch / intermediate-BW
优化吃下，必须啃 Tier 2 的 `mul_mm_id` kernel craft。后续顺序：

1. **`mul_mm_id` 内部 craft**（bench-driven 或 Xcode-driven）——
   决定性的 50% gap 闭合。**~3-14 天**取决于工具。**这是唯一会
   动 TPOT 的事**。
2. **长上下文 + ShareGPT bench** —— 1-2 天，用 vLLM 标准数据集
   把 dense 完胜在更真实负载下展示。
3. **Speculative decoding for Metal** —— 1-2 天，Qwen3-0.6B 当
   draft，4B/8B 当 target，预期 c=1 加速 1.5-2×。
4. **NR0 大 tile 实验** —— 1 天 bench-driven，可能 +几%。

## 记录的死路（don't go back）

- **threshold tuning 穷尽**：T=8 / T=12 / T=16 都试过，T=12 是最优
- **`moe_forward_batched_prefill_impl` 直接复用**：bucketing overhead 在
  decode-time 小 M 太大，需要 per-item 路径配合
- **lx/ly swap 实验性 port to llama.cpp 的 layout**：单纯 swap 让输出全
  是 `]]]` —— layout 不只 swap 这么简单，整个 simdgroup_load 配套都
  得改
- **Qwen3MoeModel 加 paged KV**：不必要 ——`Qwen3MoeModel::ensure_kv` 是
  contig-only 的，paged 开关不影响它，dispatch 仍正确
- **Offset-aware kernels = +30% (估算)**：实测只 +1-2%。
  Metal `set_buffer(offset)` 比 copy_slice 省的不多，dispatch
  count 不是 c=4/8 的真瓶颈。后续不要再用"省 N 个 dispatch ⟹ 省
  N×10µs"做 ROI 估算——必须 profile 看实际 GPU time / bandwidth。
- **Gate+up+silu 融合 = +5-15% (估算)**：实测 c=4 +2.4% / c=8 +0.4% /
  c=16 +0.8%。原本指望省下 `gate_out_stacked` + `up_out_stacked` 的
  intermediate-BW，但中间 buffer (~24 KB × 8 slots × 2 ≈ 384 KB)
  fit Apple GPU L2/L3 已经，写出再读回根本没付 DRAM 代价。
  "intermediate-BW 估算"和"dispatch count 估算"是同一类错——
  cache hierarchy 已经把这层 traffic 吃了，只有 DRAM-bound /
  per-kernel GPU 时间是真瓶颈。

## 已合并的 PR 链（按时间顺序）

| PR | 标题 | 主要价值 |
|---|---|---|
| #71 | docs(bench): paged attention Phase 4-5 plan + scaffold | 起步规划 |
| #72 | perf(metal): paged-KV multi-seq shared pool foundation | Phase 4a 基础 |
| #73 | perf(metal): batched paged-KV decode dispatch (Phase 4b) | dense 8B c=16 +40% |
| #74 | feat: GGUF model support in serve/bench/pull | Group A 模型可服务 |
| #75 | bench: Group A concurrent HTTP report | 三引擎并发对比报告 |
| #76 | bench: Qwen3-30B-A3B Q4_K_M concurrent results | 30B 数据 + 修订 |
| #77 | perf(metal): Phase 4b for Qwen3-MoE | MoE c=16 +15% over 默认 |
| #78 | docs: 2026-05-01 concurrent perf status & roadmap | 本文档（旧版） |
| #79 | perf(metal): offset-aware MoE kernels | c=4/8 +1-2% (低于预估) |
| 待编号 | perf(metal): gate+up+silu 融合 MoE GEMV | c=4/8/16 +0.4-2.4% (低于预估) |

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
