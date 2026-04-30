# M1 Max Qwen3-MoE 性能追赶 llama.cpp（2026-04-30 状态）

本文档是 PR #50/#52/#53/#54/#55/#56 之后的状态快照、测试方法、核心实现解释和下一步计划。
读者：下一个会话 / 接手者。

## 1. 目标

在 **Apple M1 Max（32 GB 统一内存）** 上让 ferrum-infer-rs 跑 **Qwen3-30B-A3B-Q4_K_M**（17 GB GGUF）追上 llama.cpp 的性能。

**衡量基准（llama.cpp build 27aef3d (225)，本机实测）**：

```
| qwen3moe 30B.A3B Q4_K - Medium | 17.28 GiB | 30.53 B | MTL,BLAS | 8 |  pp512 |  618.64 ± 11.17 |
| qwen3moe 30B.A3B Q4_K - Medium | 17.28 GiB | 30.53 B | MTL,BLAS | 8 |  tg128 |   51.03 ± 3.32  |
```

- llama.cpp 在 M1 Max 上 **不**走 Metal 4 tensor matmul（`tensor API disabled for pre-M5 and pre-A19 devices`），就是普通 simdgroup_half8x8 + fp32 accum。
- `MUL_MAT_ID` 在 ggml-blas 里**没有** handler，MoE 必走 Metal。所以这是 **纯 Metal 的对决**，不存在 AMX 加速这一层（之前的猜测错了）。

**当前阶段目标**：~~pp512 ≥ **80% × 618 = 494 t/s**~~ → ~~**90% = 556 t/s**~~ — **2026-04-30 PR #56 后达成 567 t/s（91.7%）**。新目标：95% = 587 t/s。

**为什么不要瞎猜 llama.cpp 的"魔法"**：我们已经验证过 kernel 架构和我们一致（NR0=64 NR1=32 NK=32, simdgroup_half8x8 + fp32 accum, 同款 dequant_q4_K, 同款 dispatch grid）。差距不是来自架构，而是来自具体的 kernel 写法（向量化 / 指令调度）和 attention 路径的实现质量。**直接 diff llama.cpp 源码找具体差异**，不要假设。

---

## 2. 当前进展（PR #50/#52/#53 累积）

### 2.1 `BENCHMARKS` — 5× pp512 中位数

| 指标 | 修复前 | PR #50 | PR #52 | PR #53 | PR #54 | PR #55 | **PR #56** | llama.cpp |
|-------|------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 加载时间 | 47-54 s | **1.0 s** | 1.0 s | 1.0 s | 1.0 s | 1.0 s | 1.0 s | ~1.5 s |
| pp512 | swap (≪1 t/s) | 64 t/s | 283 t/s | 282 t/s | 370 t/s | 547 t/s | **567 t/s** | 618 t/s |
| tg128 | 0.1 t/s | 27.6 t/s | 27 t/s | 27 t/s | 29 t/s | 30 t/s | 30 t/s | 51 t/s |
| 内存峰值 | ~34 GB | **17 GB** | 17 GB | 17 GB | 17 GB | 17 GB | 17 GB | 17 GB |

**vs llama.cpp（PR #56 后）**：pp512 = **91.7%** ✓ 越过 90% 目标，tg128 = **59%**，加载 + 内存 = **持平**。

### 2.2 PR 时间线

- **PR #50 `perf(metal): zero-copy GGUF mmap`** — 把 GGUF mmap 包装成 per-tensor `newBufferWithBytesNoCopy` MTLBuffer。修复内存翻倍（之前 `new_buffer_with_data` 拷贝），加载从 47s → 1s，30B 模型从 swap 模式 → 正常。⚠️ **不要再尝试一个大 16 GiB buffer 的方案** — 在 M1 Max 上 GPU 残留检查会让 decode 慢 70×，已验证。
- **PR #51 `feat(metal): prefill profiling + capture hooks`** — 加 `FERRUM_DECODE_OP_PROFILE=1` 的 prefill stage 分解打印，加 `FERRUM_METAL_CAPTURE` 的 Xcode `.gputrace` 抓帧。
- **PR #52 `perf(metal): vectorize MoE GEMM stores`** — `gemm_q4kw_moe_id_f32` 把 8 个标量 half store 改成一个 `half2x4` 向量 store。**单点改动，pp512 64 → 283 t/s（4.4×）**。这是和 llama.cpp 的关键 diff。
- **PR #53 `perf(metal): vectorize dense GEMM + flash_attn QK loads`** — 同样的 vector-store 模式应用到 dense Q4_K/Q6_K GEMM 和 flash_attn 的 QK 点积。本模型上 ~0% 提升（这些不是当前瓶颈），合并主要为一致性。
- **PR #54 `perf(metal): Q-tiled flash_attn with simdgroup_matmul`** — 重写 flash_attn：每个 threadgroup 处理 8 query rows × 128 thread × 4 simdgroup，每内层 32 KV cols 用 `simdgroup_multiply_accumulate` 计算 QK^T 和 P@V。**attn 时间 477→197 ms（-59%），pp512 占比 27%→13%，pp512 303→370 t/s（+22%）**。head_dim=128 / sliding_window=0 / q_len≥8 走新 kernel；其它（包括 decode m=1）走原 scalar kernel；`FERRUM_FA_LEGACY=1` 强制 legacy 用于回归对比。
- **PR #55 `perf(metal): MoE prefill topk + ids/tpe on GPU`** — 替换 `moe_forward_batched_prefill_impl` 里 `B::sync + to_vec(router_logits) + host softmax+topk + bucket-by-expert + write_back` 的整段 host roundtrip。`route_topk_softmax`（已存在）+ 新增 `compute_ids_tpe_gpu` 两个 kernel 在同一 compute encoder 内完成。后者走 single-threadgroup × 256 threads × `atomic_fetch_add` 抢 slot；ids 用 worst-case stride（`tokens * top_k`），消费侧 GEMM 的 `r1 >= tpe[e]` 早退处理多余 TG。**pp512 318→547 t/s（+72%），vs llama.cpp 51%→88.5% — 单 PR 超过 §1 80% 目标**。`FERRUM_MOE_HOST_TOPK=1` 强制 legacy 用于回归对比。
- **PR #56 `perf(metal): indirect-dispatch MoE GEMM via tpe max-reduce`** — 把 `compute_ids_tpe_gpu` 加 phase 3/4：max-reduce `tpe[e]` 然后 thread 0 写 `(grid_x, grid_y, grid_z)` 到 `gate_up_args` / `down_args` 两个 12-byte indirect args buffer。三个 GEMM (gate / up / down) 改用 `dispatch_thread_groups_indirect`，grid 从 worst-case 收紧到 `max(tpe[e]) / 32` 列。kernel 内部 `ids` indexing 仍用 worst-case `tokens*top_k` stride（避免 compaction pass）。**pp512 547→567 t/s（+2.3%），vs llama.cpp 88.5%→91.7% — 越过 90% 目标**。`FERRUM_MOE_DIRECT_DISPATCH=1` 回退 worst-case grid 用于 A/B 对比。

---

## 3. 如何测试 / 复现 bench

### 3.1 关键环境变量

| 变量 | 必须 | 作用 |
|---|---|---|
| `FERRUM_KV_CAPACITY=512` | bench 时 | 限 KV cache 容量。**默认值 = `min(model_max, 4096)`**（对 30B-A3B = 4096，约 786 MB GPU 内存），chat 直接能用。bench pp512 想剥离 KV 影响时设 512；REPL 跑长上下文设 8192-16384；超过此值 `forward_layer` 会 panic 而不是越界写脏数据（PR #57）。 |
| `MTL_CAPTURE_ENABLED=1` | 仅 capture | 启用 Metal 抓帧。系统层面的 gate。 |
| `FERRUM_METAL_CAPTURE=path/out.gputrace` | 仅 capture | 让 prefill 写一个 `.gputrace` 文件。 |
| `FERRUM_DECODE_OP_PROFILE=1` | 仅 profile | 每个 stage（attn/moe/host/gate/up/silu/down/wsum）打印分解时间。**注意会插 `B::sync(ctx)` 拖慢 1.5-2×**——不要用它的绝对数字，只用相对占比。 |
| `FERRUM_MMAP_TRACE=1` | 调试 mmap | 打印 PR #50 的 mmap registry hit/miss。生产关。 |

### 3.2 标准 bench 命令

```bash
# 基础设施
cd /Users/chejinxuan/rust_ws/ferrum-infer-rs
cargo build --release -p ferrum-cli --features metal

# 模型路径
GGUF=/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf
TOK=/Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-30B-A3B.tokenizer.json
```

#### A. **正确性**（每次改 kernel 必跑）

```bash
FERRUM_KV_CAPACITY=512 ./target/release/ferrum run "$GGUF" \
  --tokenizer "$TOK" \
  --prompt "The capital of France is" --max-tokens 16 --temperature 0.0
```

**期望输出**（必须**逐字一致**）：

```
The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

任何 `_Cell_Cell_Cell` / `Dund impe impe` / `, , , ,` 等回归，立刻 git diff 找最近的 kernel 改动，回滚定位。**不要用 `--bench-mode` 跑正确性**，它会吞掉文本。

#### B. **pp512**（prefill 吞吐量）

```bash
PROMPT=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..50})
for i in 1 2 3 4 5; do
  FERRUM_KV_CAPACITY=512 ./target/release/ferrum run "$GGUF" \
    --tokenizer "$TOK" \
    --prompt "$PROMPT" --max-tokens 1 --temperature 0.0 --bench-mode \
    2>&1 | grep "prefill:"
done
```

**取 5 次中位数**。本机方差大（~30%），单次数字不可信。

预期实际 token 数 ≈ 501（tokenizer 拆分）。

#### C. **tg128**（decode 吞吐量）

```bash
for i in 1 2 3 4 5; do
  FERRUM_KV_CAPACITY=512 ./target/release/ferrum run "$GGUF" \
    --tokenizer "$TOK" \
    --prompt "Hi" --max-tokens 128 --temperature 0.0 --bench-mode \
    2>&1 | grep "throughput"
done
```

#### D. **prefill 分解 profile**

```bash
FERRUM_DECODE_OP_PROFILE=1 FERRUM_KV_CAPACITY=512 ./target/release/ferrum run "$GGUF" \
  --tokenizer "$TOK" --prompt "$PROMPT" \
  --max-tokens 1 --temperature 0.0 --bench-mode 2>&1 \
  | grep -E "prefill-profile|attn:|moe:|other:|host|gate|up:|silu|down|wsum"
```

输出例子（current PR #53 后）：

```
[prefill-profile] tokens=501 total=2830 ms (177 t/s)
    attn:  925 ms (32.7%) over 48 calls
     moe: 1640 ms (57.9%) over 48 calls
   other:  265 ms ( 9.4%) over  1 calls
    host:  151 ms ( 5.4%) over 48 calls
    gate:  474 ms (16.7%) over 48 calls
      up:  457 ms (16.2%) over 48 calls
    silu:   19 ms ( 0.7%) over 48 calls
    down:  501 ms (17.7%) over 48 calls
    wsum:   19 ms ( 0.7%) over 48 calls
```

总时长 = 2.83s，但**实际 prefill 是 1.79s**（177 vs 282 t/s）—— `B::sync` 在每个 stage 间强制 GPU flush，破坏了 pipeline 并行性。**用相对百分比，不用绝对 ms**。

#### E. **Metal 抓帧**（深度分析时用）

```bash
rm -f /tmp/ferrum_prefill.gputrace
MTL_CAPTURE_ENABLED=1 FERRUM_METAL_CAPTURE=/tmp/ferrum_prefill.gputrace \
FERRUM_KV_CAPACITY=512 ./target/release/ferrum run "$GGUF" \
  --tokenizer "$TOK" \
  --prompt "$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..50})" \
  --max-tokens 1 --temperature 0.0 --bench-mode

open /tmp/ferrum_prefill.gputrace
```

打开后：
1. 选设备 → 勾 **Profile GPU Trace** → 点 **Replay**（等几分钟）
2. 左侧 **Performance** → **Counters** 标签 → **Performance Limiters** 列
3. 找 cost 最高的 encoder（`gemm_q4kw_moe_id_f32` 或 `flash_attn_f32`）
4. 看：**ALU Limiter / ALU Utilization / F32 Utilization**

**Trace 文件 38 GB**（含所有 MTLBuffer 副本），删完看完别忘清理。

#### F. **llama.cpp baseline**（已编译，不需要重 build）

```bash
/Users/chejinxuan/rust_ws/llama.cpp/build/bin/llama-bench \
  -m "$GGUF" -p 512 -n 128 -r 3
```

---

## 4. 当前核心实现（关键设计点）

### 4.1 内存布局：per-tensor zero-copy MTLBuffer（PR #50）

`crates/ferrum-kernels/src/backend/metal.rs::buffer_for_quant_bytes`

每个 quant tensor 调用 `newBufferWithBytesNoCopy` 包装 mmap 中**该 tensor 所在的页对齐区间**。

```rust
let aligned_start = ptr_addr & !(PAGE - 1);
let aligned_end = (ptr_addr + bytes.len()).div_ceil(PAGE) * PAGE;
let buf = device.new_buffer_with_bytes_no_copy(
    aligned_start as *const c_void,
    (aligned_end - aligned_start) as u64,
    MTLResourceOptions::StorageModeShared,
    None,
);
```

`MetalQuantStore::Q4K { blocks, byte_offset, ... }` 里存 buffer + 字节偏移。所有 kernel dispatch 用 `setBuffer:offset:` 绑定。

**反面教材**：尝试过把整个 mmap 做成一个 16 GiB MTLBuffer（M1 Max max_buffer_length），correctness OK 但 decode 慢 **70×**。Apple GPU 残留追踪在大 buffer 上极贵。**多个小 buffer 才是对的**。

### 4.2 GGUF 注册（一次性）

`crates/ferrum-cli/src/commands/run_gguf.rs` 在 Metal backend 路径里：

```rust
ferrum_kernels::backend::metal::register_gguf_mmap(
    gguf_arc.mmap_bytes(),
    gguf_arc.clone(),  // keeper Arc — 让 mmap 在 MTLBuffer 生命周期内活着
)?;
```

`register_gguf_mmap` 只记录范围 + Arc，**不**预先创建 MTLBuffer。MTLBuffer 在每个 `load_quant*` 调用时按需创建。

### 4.3 MoE 热 kernel（PR #52 优化点）

`crates/ferrum-kernels/src/q4_k_moe_id_gemm.metal::gemm_q4kw_moe_id_f32`

3 个关键 store / load 点：

1. **权重 dequant 后 → shmem**（每 K=32 块 fires K/NK=64 次）
   ```c
   FOR_UNROLL (short i = 0; i < 16; ++i) {
       sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];  // 16 标量 store
   }
   ```
   *和 llama.cpp 一致*，**已 OK**。

2. **激活 → shmem**（PR #52 改动点）
   ```c
   // 改之前：8 个 half 标量 store
   *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
       half2x4(*((device const float2x4 *) y));   // 1 个向量 store
   ```
   **就是这个让 pp512 4.4×**。同样的模式 PR #53 应用到了 q4_k_gemm.metal / q6_k_gemm.metal / q6_k_moe_id_gemm.metal（这次模型上无显著收益但保持一致）。

3. **输出 writeback**（PR #52 也改了）
   ```c
   // float4 vector copy + scalar tail
   for (i = tiisg; i < nr0/4; i += 32) D4[i] = C4[i];
   for (i = 4*(nr0/4)+tiisg; i < nr0; i += 32) D[i] = C[i];
   ```

内层 matmul（4 simdgroup × 8 matmul tile × NK/8 iter）和 llama.cpp 完全一样。

### 4.4 Routing：GPU（PR #55）

PR #55 之前 routing **走 CPU**（`B::sync + to_vec + host softmax+topk + bucket + write_back`）。PR #55 之后**全部 GPU**：

```rust
// crates/ferrum-models/src/models/qwen3_moe.rs::moe_forward_batched_prefill_impl
let max_per_expert = if use_gpu_topk {
    B::route_topk_softmax(
        ctx, &scratch.router_logits,
        &mut scratch.selected_ids_buf, &mut scratch.weights_2d,
        tokens, n_exp, top_k, norm_topk_prob,
    )?;
    B::compute_ids_tpe_gpu(
        ctx, &scratch.selected_ids_buf,
        &mut scratch.tpe_buf, &mut scratch.ids_2d,
        tokens, n_exp, top_k,
    )?;
    tokens * top_k    // worst-case row stride
} else {
    /* legacy host path under FERRUM_MOE_HOST_TOPK=1 */
};
```

**关键点**：
- 没有 `B::sync` — 整个 prefill 一个 command buffer 跑到底。
- `compute_ids_tpe_gpu` 用 single-threadgroup × 256 threads × `atomic_fetch_add(&tpe[e])` 抢 slot 写 `ids[e * row_stride + slot] = pair_idx`。row_stride 用 worst-case `tokens * top_k`（4008 for pp512），消费 GEMM 的 `r1 >= tpe[e]` 早退处理多余 TG，实测 launch overhead 可忽略。
- 不需要把 max(tpe[e]) 同步回 host — worst-case stride + 早退把这块完全跳过。

llama.cpp 用类似的 GPU prepass `kernel_mul_mm_id_map0`，每个 expert 一个线程扫 selected_experts。我们的版本算法一样、调度更激进（256 threads atomic 而不是 num_experts threads）。

### 4.5 Flash Attention：Q-tiled simdgroup_matmul（PR #54）

`crates/ferrum-attention/src/metal/shaders/flash_attn.metal`

两个 kernel 共存：
- `flash_attn_f32`（legacy，仍保留）：1 threadgroup × 1 simdgroup 处理一个 (Q-pos, head, batch)。KV 串行，scalar Q·K 点积。decode m=1 走它，head_dim≠128 / sliding_window>0 也回退到它。
- `flash_attn_q_tiled_f32`（PR #54 主线）：1 threadgroup × 4 simdgroup（128 线程）处理 **Q_TILE=8 个 query rows**，KV 以 C=32 为块走外层循环。每块用 `simdgroup_multiply_accumulate` 算 [8,8] QK^T tile，FlashAttention 风格 online softmax，再用 `simdgroup_matmul` 累加 P@V 到 16 个 [8,8] O-tile（NSG=4 各负责 NO=4 个）。head_dim=128 固定。

```c
// QK^T inner loop（每个 simdgroup 独立产出一个 [8,8] mqk tile）
simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
for (int i = 0; i < FA_DK8; ++i) {                        // 16 次（head_dim/8）
    simdgroup_float8x8 mk, mq;
    simdgroup_load(mk, pk + 8*i, FA_DK, ulong2(0, 0), true);   // K transposed
    simdgroup_load(mq, pq + 8*i, FA_DK);                        // Q
    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);            // [8,8] += [8,8] * [8,8]
}
simdgroup_store(mqk, ss + 8*sgitg, FA_C, ulong2(0, 0), false);
```

**绑定 dispatch 路径**：`pipelines.rs::flash_attn_v2` 在 `head_dim==128 && sliding_window==0 && q_len>=8` 时走新 kernel，否则回退 legacy。`FERRUM_FA_LEGACY=1` 可强制 legacy 用于 A/B 对比。

**实测收益（pp512）**：attn 时间 477 → 197 ms（-59%），prefill 占比 27% → 13%，总吞吐 303 → 370 t/s（+22%）。新瓶颈是 MoE GEMM（占比 75%）。

---

## 5. 下一步计划

### 5.1 ✅ 主线（已完成）：flash_attn 重写 — PR #54

**实测**：attn 477 → 197 ms（-59%），占比 27% → 13%，pp512 303 → 370 t/s（+22%）。NSG=4、Q_TILE=8、C=32、head_dim=128，f32 精度。covered: causal + GQA。未支持: ALiBi / sliding_window / 非 128 head_dim → fall back legacy。

**剩余 attn 余地（次要）**：
- 上 NSG=8（双倍并行 8 simdgroup × 8 Q rows = 64 row tile）— 占用 shmem 翻倍但仍在限内
- f16 Q/K 输入 + f32 累加（llama.cpp 的 FA_TYPES_F32 用法）— 减半 memory traffic 但需要 dtype 转换
- 当前 13% 已接近收益拐点，下一步先攻 MoE 而不是再压 attn。

### 5.2 ✅ MoE topk 上 GPU — 已完成（PR #55）

**实测**：pp512 318 → 547 t/s（+72%），prefill profile 中 host 占比 8.7% → 2.2%（残余是 profile 模式自身的 sync 开销）。**预估 8-10%、实测 +72%** — 因为去掉 sync 后 attn → MoE pipeline 完全 parallel。

### 5.3 下一步：MoE GEMM 进一步压榨（突破 90% 目标）

PR #55 之后 prefill profile：

```
[prefill-profile] tokens=501 total=2958 ms (169 t/s)   # profile mode (with per-stage sync)
    attn:  434 ms (14.7%) over 48 calls
     moe: 2193 ms (74.2%) over 48 calls       ← 主瓶颈
    host:   63 ms ( 2.2%) over 48 calls       ← 已优化（GPU）
    gate:  621 ms (21.0%) over 48 calls
      up:  648 ms (21.9%) over 48 calls
    silu:   30 ms ( 1.0%) over 48 calls
    down:  792 ms (26.8%) over 48 calls       ← 最大单项
    wsum:   22 ms ( 0.8%) over 48 calls
```

bench-mode 实际 prefill 911 ms（547 t/s），MoE 占 ~74% × 911 = 674 ms（去掉 profile sync overhead）。

**剩余 gap**：547 → 618 还差 71 t/s ≈ 13%。MoE 砍 1/8 就够。可能方向：

**5.3a gate + up 双 GEMM 合并**（1 天）：两个 GEMM 共享同一 lhs（`norm_out`）和同一 ids/tpe，输出 stack 到 `[batch, top_k, 2*expert_inter]`。一次 dispatch 替两次，省一半 lhs 加载和一次 dispatch 开销。预估 ~5-8% 提升。

**5.3b down GEMM 优化**（2 天）：down 单项 26.8% 是最大的。lhs 是 `silu_stacked[batch * top_k, expert_inter=768]`，rhs 是 `[batch, top_k, hidden=2048]`。M=2048 比 gate/up 的 M=768 大 2.7×，但每个 (token, slot) 对的 lhs 长度只有 768 vs gate/up 的 hidden=2048。理论上不应该比 gate/up 慢，但实测 792 vs 621 ms。猜测原因：down 的 src1 ne11=top_k 下走 per-slot 索引而不是 broadcast，激活加载模式不一样。需要 frame-capture 看具体瓶颈。

**5.3c NR1 = 32 → 64**（半天）：`gemm_q4kw_moe_id_f32` 当前 NR1=32（每 TG 处理 32 列）。M1 Max shmem 32KB，当前用 ~12KB，能扩到 NR1=64（更大 K-tile 摊销开销）。需要重新设计 simdgroup × tile 布局。

### 5.4 已合并 PR

- ~~5.1~~ Q-tiled flash_attn — PR #54 ✅
- ~~5.2~~ MoE topk + ids_tpe → GPU — PR #55 ✅

### 5.3 Decode 优化（独立路线，不影响 prefill）

`docs/qwen3-moe-decode-status-2026-04-30.md` §4.1 列出的三个 dispatch 融合：
1. **gate + up 双 gemv 融合**（48 dispatches/token）
2. **silu_mul + down gemv 融合**（48/token）
3. **router + topk 融合**（48/token）

合计 144 dispatches/token，理论再砍 ~7 ms TPOT。decode 27.6 → ~35-38 t/s，缩小到 llama.cpp 51 t/s 的 70%。

### 5.4 已知小问题

- **`stub_linear` 浪费 576 MB 内存**：`qwen3_moe.rs::stub_linear` 给 dense 路径的 `gate_up_proj / down_proj` 创建 12 MB f32 0 权重 × 48 layer = 576 MB。MoE 模型从来不用这些 slot。Xcode 把这显示为 "Large Unused Resource"。可以改成 `Option<Box<dyn Linear<B>>>` 或者占位 stub 不分配实际内存。**单独小 PR**。

### 5.5 不要再尝试的方向（已验证负面）

| 方向 | 结果 |
|---|---|
| 一个 16 GiB MTLBuffer 包整个 mmap | decode 70× 退化（GPU 残留追踪开销）。多个小 buffer 是对的。 |
| AMX / Accelerate prefill backend | llama.cpp 在 M1 Max 上不用这个（MUL_MAT_ID 在 ggml-blas 没 handler），跟我们一样纯 Metal。AMX bench 显示对 N=32 的 MoE 工作负载也不是比 GPU 快多少。 |
| Apple Tensor matmul (`mpp::tensor_ops`) | M1 Max pre-M5，不支持。`has_tensor=false`。要等硬件升级。 |
| 拷贝路径 weight upload (`new_buffer_with_data`) | PR #50 之前的版本，内存翻倍 + load 50s。回不去。 |

---

## 6. 其它历史信息

### 6.1 m1 内存约束

32 GB 物理 RAM 是硬约束。Qwen3-30B-A3B Q4_K_M GGUF 17.3 GB。任何同时占用 17 GB+ 多余分配的方案都会触发 swap。**始终 `FERRUM_KV_CAPACITY=512` + 关掉所有大 GUI app（VS Code / Lark / 微信 / DBeaver / Chrome 多 tab）才能稳定 bench。**

vm_stat 监控：

```bash
while true; do
  free=$(vm_stat | awk '/Pages free/ {gsub("\\.","",$3); print $3}')
  swap=$(sysctl -n vm.swapusage | awk '{print $7}')
  free_gb=$(echo "scale=1; $free * 16384 / 1073741824" | bc)
  echo "[$(date +%H:%M:%S)] free=${free_gb}GB swap=${swap}"
  sleep 5
done
```

bench 期间 `Pages free` 跌到 < 200 MB 且 `swap` 持续涨 = 准备废测，结果不可信。

### 6.2 测试机配置

- **MacBook Pro M1 Max**, macOS 15.1.1
- 32 GB unified memory
- GPU family: Apple7 / Metal3 / **NOT Metal4**（重要——决定不能用 tensor matmul）
- maxBufferLength = 16 GiB
- recommendedMaxWorkingSetSize ≈ 22.9 GB

### 6.3 当前未合并 PR

- **PR #54** `perf(metal): Q-tiled flash_attn with simdgroup_matmul` — pp512 303 → 370 t/s（+22%），attn 占比 27% → 13%。本地 commit c456984。
- **PR #55** `perf(metal): MoE prefill topk + ids/tpe on GPU` — pp512 370 → 547 t/s（+48% on top of #54），host 占比 8.7% → 2.2%。本地 commit ed000d8。
- **PR #56** `perf(metal): indirect-dispatch MoE GEMM via tpe max-reduce` — pp512 554 → 567 t/s（+2.3%）；pp512 累计 64 → 567 t/s（vs PR #50 8.9× 总提升）。本次工作。
- ~~PR #53~~ 已合并 (e12a347)。

### 6.4 相关 memory note

- `~/.claude/projects/.../memory/project_metal_zerocopy.md` — PR #50 设计要点
- `~/.claude/projects/.../memory/project_qwen3_moe_prefill.md` — 本文档浓缩版
- `~/.claude/projects/.../memory/project_qwen3_moe_decode_fusion.md` — 早期 decode 工作（PR #46-#49）
- `docs/qwen3-moe-decode-status-2026-04-30.md` — decode 优化的详细分析（dispatch 融合路线）
