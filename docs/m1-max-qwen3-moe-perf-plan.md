# M1 Max Qwen3-MoE 性能追赶 llama.cpp（2026-04-30 状态）

本文档是 PR #50/#52/#53/#54 之后的状态快照、测试方法、核心实现解释和下一步计划。
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

**当前阶段目标**：pp512 ≥ **80% × 618 = 494 t/s**。

**为什么不要瞎猜 llama.cpp 的"魔法"**：我们已经验证过 kernel 架构和我们一致（NR0=64 NR1=32 NK=32, simdgroup_half8x8 + fp32 accum, 同款 dequant_q4_K, 同款 dispatch grid）。差距不是来自架构，而是来自具体的 kernel 写法（向量化 / 指令调度）和 attention 路径的实现质量。**直接 diff llama.cpp 源码找具体差异**，不要假设。

---

## 2. 当前进展（PR #50/#52/#53 累积）

### 2.1 `BENCHMARKS` — 5× pp512 中位数

| 指标 | 修复前 | **PR #50** | **PR #52** | **PR #53** | **PR #54** | llama.cpp |
|-------|------:|---------:|---------:|---------:|---------:|---------:|
| 加载时间 | 47-54 s | **1.0 s** | 1.0 s | 1.0 s | 1.0 s | ~1.5 s |
| pp512 | swap (≪1 t/s) | 64 t/s | 283 t/s | 282 t/s | **370 t/s** | 618 t/s |
| tg128 | 0.1 t/s | 27.6 t/s | 27 t/s | 27 t/s | 29 t/s | 51 t/s |
| 内存峰值 | ~34 GB | **17 GB** | 17 GB | 17 GB | 17 GB | 17 GB |

**vs llama.cpp（PR #54 后）**：pp512 = **60%**，tg128 = **57%**，加载 + 内存 = **持平**。

### 2.2 PR 时间线

- **PR #50 `perf(metal): zero-copy GGUF mmap`** — 把 GGUF mmap 包装成 per-tensor `newBufferWithBytesNoCopy` MTLBuffer。修复内存翻倍（之前 `new_buffer_with_data` 拷贝），加载从 47s → 1s，30B 模型从 swap 模式 → 正常。⚠️ **不要再尝试一个大 16 GiB buffer 的方案** — 在 M1 Max 上 GPU 残留检查会让 decode 慢 70×，已验证。
- **PR #51 `feat(metal): prefill profiling + capture hooks`** — 加 `FERRUM_DECODE_OP_PROFILE=1` 的 prefill stage 分解打印，加 `FERRUM_METAL_CAPTURE` 的 Xcode `.gputrace` 抓帧。
- **PR #52 `perf(metal): vectorize MoE GEMM stores`** — `gemm_q4kw_moe_id_f32` 把 8 个标量 half store 改成一个 `half2x4` 向量 store。**单点改动，pp512 64 → 283 t/s（4.4×）**。这是和 llama.cpp 的关键 diff。
- **PR #53 `perf(metal): vectorize dense GEMM + flash_attn QK loads`** — 同样的 vector-store 模式应用到 dense Q4_K/Q6_K GEMM 和 flash_attn 的 QK 点积。本模型上 ~0% 提升（这些不是当前瓶颈），合并主要为一致性。
- **PR #54 `perf(metal): Q-tiled flash_attn with simdgroup_matmul`** — 重写 flash_attn：每个 threadgroup 处理 8 query rows × 128 thread × 4 simdgroup，每内层 32 KV cols 用 `simdgroup_multiply_accumulate` 计算 QK^T 和 P@V。**attn 时间 477→197 ms（-59%），pp512 占比 27%→13%，pp512 303→370 t/s（+22%）**。head_dim=128 / sliding_window=0 / q_len≥8 走新 kernel；其它（包括 decode m=1）走原 scalar kernel；`FERRUM_FA_LEGACY=1` 强制 legacy 用于回归对比。

---

## 3. 如何测试 / 复现 bench

### 3.1 关键环境变量

| 变量 | 必须 | 作用 |
|---|---|---|
| `FERRUM_KV_CAPACITY=512` | **是** | 限 KV cache 容量。不设默认走 `max_seq_len=32K`，会分配 ~6 GB KV cache 把 32 GB Mac 推到 swap，bench 数字直接归零。**每次都要设。** |
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

### 4.4 Routing：CPU vs GPU

ferrum 当前 routing **走 CPU**：

```rust
// crates/ferrum-models/src/models/qwen3_moe.rs::moe_forward_batched_prefill_impl
B::sync(ctx);  // ← 强制 GPU flush，破坏了 attn → MoE 的 pipeline
let logits_host = B::to_vec(&scratch.router_logits, tokens * n_exp);
let route = crate::moe::router::route(&logits_host, ...);  // CPU softmax + topk
let (tpe_host, ids_host, max_per_expert) = compute_ids_tpe(...);
B::write_i32_into(&mut scratch.tpe_buf, &tpe_host);
B::write_i32_into(&mut scratch.ids_2d, &ids_host);
B::write_f32_into(&mut scratch.weights_2d, &route.expert_weights);
```

llama.cpp 用 **GPU prepass kernel**：`kernel_mul_mm_id_map0`，每个 expert 一个线程，扫所有 token 的 selected_experts 数组，输出 `tpe[expert]` + `ids[expert][slot]`。

prefill profile 显示我们的 host topk 占 5.4%。改成 GPU 会**省掉 sync barrier**，让 attn dispatch 和 routing prepass 在 GPU 上 pipeline。

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

### 5.2 主线（下一步）：MoE GEMM / topk 优化

PR #54 之后 prefill profile：

```
[prefill-profile] tokens=501 total=1513 ms (331 t/s)
    attn:  197 ms (13.0%) over 48 calls       ← 已优化
     moe: 1140 ms (75.4%) over 48 calls       ← 新主瓶颈
    host:  131 ms ( 8.7%) over 48 calls       ← topk roundtrip
    gate:  308 ms (20.4%) over 48 calls
      up:  309 ms (20.4%) over 48 calls
    silu:   17 ms ( 1.2%) over 48 calls
    down:  342 ms (22.6%) over 48 calls
    wsum:   17 ms ( 1.2%) over 48 calls
```

**5.2a MoE topk 上 GPU**（半天）：写 `route_topk_softmax_compute_ids_tpe` kernel，模仿 llama.cpp `kernel_mul_mm_id_map0`。直接省 host 8.7% + 去掉 sync barrier。预估 8-10% 提升。

**5.2b MoE GEMM 进一步压榨**（2-3 天）：gate/up/down 三个 GEMM 占 63%，目前 `gemm_q4kw_moe_id_f32` 已经做了 vec store（PR #52）。下一刀可能是：
- gate + up 双 GEMM 合并（共享 lhs 激活，128 threadgroups → 64）
- 增大 NR1 = 32 → 64（取决于 shmem 空间）
- 用 simdgroup_multiply_accumulate 而非 simdgroup_matmul（潜在 2× ALU 吞吐，需重测）

**目标**：进一步把 pp512 从 370 推到 ≥494（80% × 618）。MoE 砍 1/3 时间就够了。

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

- **PR #54** `perf(metal): Q-tiled flash_attn with simdgroup_matmul` — 本次工作。CI 待开。pp512 303 → 370 t/s（+22%），attn 占比 27% → 13%。
- ~~PR #53~~ 已合并 (e12a347)。

### 6.4 相关 memory note

- `~/.claude/projects/.../memory/project_metal_zerocopy.md` — PR #50 设计要点
- `~/.claude/projects/.../memory/project_qwen3_moe_prefill.md` — 本文档浓缩版
- `~/.claude/projects/.../memory/project_qwen3_moe_decode_fusion.md` — 早期 decode 工作（PR #46-#49）
- `docs/qwen3-moe-decode-status-2026-04-30.md` — decode 优化的详细分析（dispatch 融合路线）
