# Qwen3-MoE 解码现状（2026-04-30 截图）

本文是给下一个会话/接手者的状态快照：当前在做什么、怎么测的、目前 decode
hot path 长什么样、以及我认为的问题。截至 PR #49 合入主干（commit
`22134bf`）。

## 1. 当前目标

**主目标：** 让 ferrum-infer-rs 在 Apple M1 Max（32 GB 统一内存）上单流
解码 **Qwen3-30B-A3B-Q4_K_M** 的 tok/s 追平或超过 llama.cpp 的 **44 t/s**
基线，并且不退化已有模型（Qwen3-0.6B/4B、Llama-3.1-8B、TinyLlama）的质量
和速度。

**当前坐标（PR #45 之后，#46-#49 之前）：** `tg128 = 31.1 t/s`，约为
llama.cpp 的 71%。

**理论坐标（#46-#49 全部合入后，未实测）：** 累计减少 ~383 dispatches /
token × ~50 µs/dispatch CPU 同步开销 = TPOT 砍掉约 19 ms。
32 ms → 13 ms，对应 **~77 t/s 上限**。现实里有融合不完全叠加 + 部分
dispatch 本来就便宜的因素，更可能落在 **35-50 t/s**——大概率 **已经追平
甚至小幅超过 llama.cpp**。

**子目标：**

1. 找一台空闲机器（不是开着 Lark/VSCode/微信的开发机）跑一次干净的
   `tg128 5×128`，把数字写进 `BENCHMARKS.md`。
2. 继续吃掉 decode hot path 上仍然存在的小 dispatch（参见 §4）。
3. Prefill (`pp512`) 目前还有 14× 差距（43.3 vs 596 t/s），暂时不是焦点
   但跟着记录。

## 2. 测试方法

### 2.1 正确性（每个 PR 必跑）

```bash
cargo check --workspace --all-targets
cargo test --workspace --features metal --lib   # CPU + Metal 全部 lib 测
```

**金句校验**（Qwen3-30B-A3B-Q4_K_M）：

```bash
./target/release/ferrum run \
    /Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --tokenizer /Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-30B-A3B.tokenizer.json \
    --prompt "The capital of France is" --max-tokens 16 --temperature 0.0
```

**期望输出：**

```
The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

跟未融合路径**逐字一致**才算通过。任何替换为
`_Cell_Cell_Cell` 或 `Dund impe impe impe` 等重复-token 的回归都意味着
关键 invariant（head_dim、norm 系数、cache_len 偏移等）被改坏，必须立即
回滚定位。**不要**用 `--bench-mode` 跑正确性，它会吞掉 token 输出。

### 2.2 性能（落入 PR 之前）

**理论上**应该跑：

```bash
./target/release/ferrum run <gguf> --tokenizer <tok.json> \
    --prompt <hi> --max-tokens 128 --temperature 0.0 --bench-mode
```

5 次重复，取中位 `d_s`。

**实际上** PR #46-#49 都没拿到干净数字，原因见 §5。本机 32 GB Mac 在 17
GB GGUF mmap + scratch 共存时，会把 mmap 页持续踢出页缓存，触发 swap
抖动 → decode 跑出 0.1 t/s 的虚假数字（实际是 mmap miss 驱动的）。

**绕开方法**（下一次干净 bench 时用）：

1. 重启 Mac，关掉 VS Code/Lark/微信/DBeaver/iTerm 多余 tab。
2. 第一次跑只是把 GGUF 装进页缓存（结果丢弃）。
3. 立刻再跑一次 `--max-tokens 128`，取这次的 `d_s`。
4. 至少要看到 `vm_stat | head -3` 的 `Pages free` 在 1M+（16 GB+）才算
   memory-clean。

## 3. 现状的核心流程

### 3.1 数据流（解码 m=1，post #49）

输入是上一个 token，输出是下一个 token 的 logits。Metal 上一次 forward
做这些事：

```
embedding_lookup(token, embed) → residual                               [1 dispatch]
for li in 0..48:
    if !prev_did_norm_fusion:                                           // layer 0 总走这里
        rms_norm(residual, attn[li].input_ln_w) → norm_out              [1]
    qkv_proj.forward(norm_out) → qkv_out                                [1] (Linear gemv, q4_k)
    split_qkv_norm_rope_kvc(qkv_out, q_norm_w, k_norm_w, ...,           [1]
                            q_head_major, cache.k, cache.v,
                            cache_len)
    flash_attention(q_head_major, cache.k, cache.v, kv_len)             [1]
                            → attn_head_major_out
    o_proj.forward(attn_head_major_out)  // tokens=1 时直接复用,        [1]
                            → o_proj_out  无需 transpose
    fused_add_rms_norm(residual, o_proj_out, post_ln_w)                 [1]
                            → residual + norm_out
    router.forward(norm_out) → router_logits                            [1] (Linear gemv, q4_k)
    route_topk_softmax(router_logits) → ids_buf, weights_buf            [1] (GPU side, no host sync)
    gemv_quant_moe_id(gate_stacked, norm_out, ids_buf,                  [1]
                      broadcast=0) → gate_out_stacked
    gemv_quant_moe_id(up_stacked, norm_out, ids_buf,                    [1]
                      broadcast=0) → up_out_stacked
    silu_mul_stacked(gate_out, up_out) → silu_stacked                   [1]
    gemv_quant_moe_id(down_stacked, silu_stacked, ids_buf,              [1]
                      stride=inter) → down_out_stacked
    if next_layer:
        weighted_sum_residual_norm_stacked(down_out, weights_buf,       [1]
            residual, attn[li+1].input_ln_w, norm_out, eps)
        // residual += Σ_k w[k]·down[k];
        // norm_out = residual · scale · next_norm_w
        // 下一层会把 prev_did_norm_fusion=true 进来，跳过自己的 rms_norm
    else:                                                                // 最后一层
        weighted_sum_residual_stacked(down_out, weights_buf, residual)  [1]

// 退出 layer loop
final_rms_norm(residual, final_norm_w) → last_normed                    [1]
lm_head.forward(last_normed) → logits                                   [1]
B::sync(ctx)                                                            (commit + wait)
to_vec(logits)
```

每层 dispatch 数：

| 层 | 数量 | 说明 |
|----|------|------|
| 0       | **12** | 含 leading rms_norm + 11 op |
| 1..N-2  | **11** | rms_norm 被前层尾巴融合掉 |
| N-1     | **11** | 跳了自己 rms_norm，但尾部没有 next-norm 融合 |

总计 = 12 + 46 × 11 + 11 = **529 dispatches** per decode token，加上
prologue/epilogue 大概 **535 ops**。

### 3.2 关键内核（Metal）

| 内核 | 文件 | 用途 |
|------|------|------|
| `rms_norm_f32` | transformer_ops.metal | 单独 RMS norm（仅 layer 0） |
| `gemv_f32a_q4kw_v2` | gemm_f16w / Q4_K | qkv_proj / o_proj / router |
| `split_qkv_norm_rope_kvc_f32` | norm_rope.metal | **#48 新增**，融合 split + Q/K norm + RoPE + cache 写入 |
| `flash_attn_f32` | flash_attn.metal | 单步 flash attention（GQA） |
| `fused_residual_norm_f32` | transformer_ops.metal | residual_add + post_attn rms_norm |
| `moe_router_topk_softmax_f32` | moe_router.metal | softmax + top-K + 可选 renorm，全 GPU 侧 |
| `gemv_q4kw_moe_id_f32` | q4_k_moe_id_gemv.metal | 单 dispatch 覆盖所有 top-K 的 gate/up/down gemv |
| `silu_mul_stacked_f32` | moe_post_ops.metal | top-K 槽位的 SiLU·gate 一次性算完 |
| `weighted_sum_residual_norm_stacked_f32` | moe_post_ops.metal | **#49 新增**，weighted_sum + residual_add + 下一层 rms_norm |

### 3.3 已经做完的融合（本会话）

- **#46** MoE residual fusion + transpose-skip
- **#47** split_qkv + 3× qk_norm_rope → 1 dispatch
- **#48** QKV 内核直接写入 KV cache（同时覆盖 Llama 单流路径）
- **#49** 跨 layer rms_norm 折入 MoE 尾部

## 4. 我认为的问题

### 4.1 还能薅的 dispatch 融合（按 ROI 排）

1. **gate + up 双 gemv 融合**——把 q4_k_moe_id_gemv 改成两个权重源 / 两个
   累加器 / 两个输出。同一个 activation broadcast 读一次，省一次 dispatch。
   **省 48/token，~7% TPOT。**最大顾虑：Q4_K 解 quant 的 scale 拆解逻辑
   复杂（`sc16 / kmask1/2/3 / sc8` 等），双路并行需要小心 register 压力
   不要爆，否则 occupancy 掉反而变慢。

2. **silu_mul + down gemv 融合**——把 SiLU·gate 算到下一个 gemv 内核里，
   省掉 silu_stacked 中间物化和一次 dispatch。**省 48/token。**比 #1 难，
   因为 silu_stacked 既是输出又是下一个 gemv 的输入，融合要重写 down
   gemv 让它从 gate_out + up_out 读、内嵌 silu。

3. **router + route_topk 融合**——一个专用的 small-quant-gemv 内核，把
   2048 → 128 logits 算完后就地做 softmax+topk+renorm。**省 48/token。**
   工作量类似 #1，但需要为 Q4_K / Q6_K 分别写。

合起来 ~144 dispatches/token。理论再砍 ~7 ms TPOT。

### 4.2 dispatch 不再是瓶颈之后的瓶颈

按公式估：

- 当前每层 11-12 dispatch × 50 µs ≈ 550-600 µs/layer 的 CPU 同步开销
- 每层 GPU 实际计算（gate/up/down × 8 个选中专家）≈ 8 × 4.5 MB 权重读取
  ÷ 400 GB/s = 90 µs/layer 的纯带宽下界
- 所以**目前是 dispatch overhead 主导**（~85%）

吃完 §4.1 的三个融合后会反过来——**变成带宽主导**。再往下要看：

- **flash_attn 的 GQA 分组**：当前 32 个 q_head 各自独立读 4 组 kv_head，
  其实 8 个 q_head 应该共享一组 K/V 加载。改完理论上 attention 部分
  DRAM 流量减 8×。kv_len 还小时收益不明显，长上下文会逐步显出来。
- **Metal Indirect Command Buffer**：把 48 层的 dispatch 序列预先编译成
  ICB，每个 token 只回放。可以摊掉绝大部分 CPU 侧 setup 成本。**最大单
  项收益**，但侵入式改造（要改 MetalContext 和所有 Backend method 的
  encoder 抽象）。

### 4.3 测不准问题（最讨厌的）

**这是这次会话最大的痛点。** 32 GB Mac 跑 17 GB GGUF + 我们的 scratch
（每层 ~500 MB scratch + KV cache）会持续抖 swap：

- 第一次跑：cold mmap，~52 s 装载 + decode 0.1 t/s（mmap page miss）
- 第二次跑（紧接着）：还是 0.1 t/s——上一轮装入的页被这一轮的 scratch
  挤出去了
- `vm_stat` 显示 Pages free 跌到 < 100 MB 时基本就是这个状态

**结果是 PR #46-#49 没有一个有真实的 perf 数字背书**，只验证了
correctness（"Paris. Rome. Madrid." 输出一致）。这是个硬伤。下个会话
**优先要做的事**：

1. 找一台干净环境（重启 Mac、关大软件、`vm_stat` 看到 1M+ 空闲页）
2. 跑 `tg128 5×128` 拿中位数
3. 写进 `BENCHMARKS.md` 标 PR #46-#49
4. 跟 llama.cpp 同环境同 GGUF 重测，做对照

### 4.4 架构层小坑

- `forward_layer` 现在签名变长（`next_layer_idx: Option<usize>` +
  `prev_did_norm_fusion: bool` + 返回 `Result<bool>`），调用方两处
  （`prefill_internal` / `decode_internal`）都要小心维护
  `prev_did_norm_fusion` 的传递。如果将来加 prefill 跨层融合（理论上
  也能做），需要再扩这个状态机。
- LlamaFamilyModel 还**没**用 #49 的 cross-layer rms_norm 融合——它的
  MLP 是 dense 的（`gate_up + silu + down`），尾部不是 weighted_sum，
  需要另写一个 "linear_then_residual_norm" 内核。属于"如果给 Llama 也
  搞 M1 Max 调优再做"的工作。
- `q_buf / k_buf / v_buf` scratch 仍然分配着，但只有 unfused fallback
  会用到（Metal 走融合路径不再触发）。占的内存其实不大（每个
  `[max_tokens × q_dim or kv_dim]` 几 MB），但可以未来通过编译时
  cfg 干掉。

## 5. 下次会话的开局清单

1. **先 bench**：拿到 #46-#49 的真实 tg128 数字，写进 `BENCHMARKS.md`，
   贴上对应 commit hash。
2. 决定下一步方向：
   - 数字漂亮（≥ 44 t/s）→ 优先做 §4.1 #1（gate+up 双 gemv），简单稳。
   - 数字一般（30-40 t/s）→ 先用 `FERRUM_DECODE_OP_PROFILE=1` 看具体
     哪段还在拖，再选融合。
   - 数字很糟（<30 t/s）→ 大概率有正确性外的副作用，回滚到 #45（31.1
     t/s）做 bisect。
3. 长期：评估 Metal Indirect Command Buffer，是不是该投入。

相关 memory note：`project_qwen3_moe_decode_fusion.md`。
