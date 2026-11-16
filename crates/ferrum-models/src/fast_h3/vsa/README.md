# FastH3 VSA：固定基线与最小 block 边界

本模型模块提供 Rust 几何、搬运、池化及注意力前后处理的typed执行组件。独立的
[受管原生入口](../../../../../native-operators/cuda/sage-vsa-fp4/README.md)
已用不可变Host pattern和GPU region producer通过Q=K=0的路由/owner fixture；
实际pooled QK已有Reference执行，CUDA组合资格仍待完成；FP4前处理与完整transformer尚未接通。该特殊算术资格不证明一般GPU数值、
视频质量或性能。

后续动态入口已实现 `F32 pooled scores + packed QKV/SF + immutable region owner`
到同 stream GPU 选块和原 attention consumer，已取得CUDA编译与限定fixture的GPU终态。
三个矩形几何各两次变化scores，5,275,648个BF16输出逐位符合独立均匀attention oracle。它复用
原 Core provider 生命周期，不接受外部 I32 lists。首选块 profile 明确分数降序、
索引升序 tie、NaN 最前、+0 排在 -0 前（固定 Torch F32 radix order）；与 Torch
cutoff tie 的选集差异必须另行记录。
现 CPU 通过不代表真实 pooled QK matmul 精度或 Sage QKV quantization 已接通。

[Key centering](key_center.md) 现在有共享Core/Reference/CUDA合同与实际模型消费者，
显式保留BF16 mean和subtraction两个数值边界；CPU测试已通过，CUDA实现尚未实测。
head-axis pack也已实现BF16位复制及F32→BF16一次舍入的不同合同，以接Untile/Expand。
下文数值边界表保留最初设计约束；具体实现及资格状态以上述段落和各组件文档为准。

固定参考为 FastVideo `866a3cdb91a85aef8561847365fc8564add9e4c8`，模型 revision
`f01e6a23ac6066aa9eb622cf94b6259442e70ec2`。下列源文件行号均指该快照。
本机外部证据位于 `~/ferrum-handoffs/20261113-ferrum-fast-h3/reference/`；部署配置
位于相邻 `20261111-fastvideo-h3-5090/`。这些外部文件不作为代码运行前提。

部署 `start.sh:8–18`、`rtx5090.yaml:8–25` 明确启用 NVFP4 DiT、FP4 video
attention、FP8 prefix、two-level P、K smoothing 和 `MINIMAX_H3_FUSIONS=all`。
`EXACT_KERNELS=all` 同时存在，但 fusion 的分支优先级更高，不能据此选择 eager
BF16 舍入路径。`metadata/fastvideo_inference.json`（仓库相同小 fixture 在
`../fixtures/inference.json`）规定 tile128、目标 sparsity 0.9、reference-video
keep rate 0.1、`p2_multi_region`、8 次 transformer forward / PDD32。模型为
hidden5376、56 个 128 维 head、RoPE96、FFN14336、50 个 transformer block；
基准中的 36L 是 VAE decoder 设置。

最小同配置 block 必须保留以下调用链：

1. AdaLN 按 `3 * timestep_index + modality_tag` 选表，执行 norm/modulate。
   参考 latent 的 denoising 时钟与 target 分开；这不意味着禁止参考 hidden
   states 在 transformer block 中参与更新。target-only flow 是 block 外的
   latent 更新边界，不能变成模态之间互不注意的 mask。
2. 将完整 hidden-width 输入按 tile 布局搬运一次，未使用槽显式清零；Q/K/V
   投影均无 bias。只在输入量化 profile 和 global scale 一致时共享 activation
   quantization；不能仅因同为 NVFP4 就共享。然后做 head RMS 与 partial RoPE。
3. Q/K 按 tile 的真实 valid count 进行 F32 mean，F32 pooled QK scores 决定每个
   batch/head/query-tile 的区域 top-k。prefix queries 看全部 key；所有 queries
   保留全部 prefix key；video queries 分别从每个 reference-video 和 target-video
   region 取 top-k。dense step/layer 覆盖所有区域的稀疏度。
4. prefix queries 独立执行 dense attention：先 K smoothing，再对 Q/K 按
   token/head 做 E4M3 roundtrip，V 保持 BF16；仅 valid key 进入 SDPA。视频查询
   走 FP4 Q/K/V、两级概率 P 量化的 block-sparse kernel。两条分支都必须存在。
5. 恢复 packed row order。若 checkpoint 的 compression gate 非零，使用原
   pooled scores 的全 tile softmax 与 pooled V，产生 BF16 compression output，
   按 row 所属 tile gather，与 gate 做 addcmul，再执行 NVFP4 to_out。该支路不是
   sparse mask 后再 softmax；QAD checkpoint 的 gate 不能按 base H3 的零初始化省略。
6. gated residual + norm/modulate、value-first SwiGLU FFN、最后 gated residual。

源定位：`fastvideo/models/dits/minimax_h3.py:382–412,749–833`；
`models/dits/minimax_h3_vsa_fp4.py:189–239,251–311,371–440`；
`attention/backends/video_sparse_attn_h3.py:257–322,405–495,557–618`；
`attention/backends/video_sparse_attn.py:34–113,165–167`。

两种 P2 是独立契约：`p2_multi_region` 是参考视频的区域选择；`two_level_p` 是
softmax 概率量化，不能以实现其中一种替代另一种。后者使用
`s_P1=rowmax(P)/(448*6)` 再做第二级 FP4 量化，见
`fastvideo-kernel/attn_qat_infer/api.py:380–390` 与
`blackwell/softmax_fused.h:32–40,75–88`。

数值边界不能由现有 P-H3 operation 的名字推定：

| 边界 | 固定基线 | Ferrum 可复用部分与缺口 |
|---|---|---|
| RMS/modulate | F32 norm、gamma、`1+scale`、乘加到最终 BF16；residual/norm2 同时输出 BF16 hidden，但 norm2 使用尚未舍入的 F32 hidden | 现 row RMS BF16 + row-selected BF16 modulation 增加中间舍入；需要显式完整 F32 profile 或新 fused contract，不能改变旧消费者的合约 |
| QK RMS/RoPE | normalized 与旋转前的 partner 均保留 F32，只在最终存储 BF16 | 现 `FUSED_HEAD_RMS_PARTIAL_ROPE_BF16` 在 norm 后有 BF16 边界；不能直接沿用。gamma 及 cos/sin 是实际 BF16 输入，尾部仅归一化 |
| SwiGLU | packed `[value|gate]`，完整表达式 F32，最终 BF16 | 现 eager `SILU_MUL_BF16` 先舍入 SiLU，不同；新增通用 packed SwiGLU profile/provider 正由 FFN 消费者接入 |
| prefix FP8 | K mean/subtraction 按原 BF16 tensor 边界；mean 在 valid-key gather 前。Q/K 每 token/head `amax/448`、零 scale 特判、E4M3→F32 scale→BF16，V 不做该 roundtrip | 现 FP8 weight-linear provider 不是这个操作。需独立 precondition/roundtrip，再复用 cross-attention 的不同 Q/KV 长度与 valid-key 契约；cuDNN fused 数值要独立资格验证 |
| video FP4 | K 在全 tile-padded key 序列上平滑，再量化；Q 不做 group-mean smoothing。Q/K 每连续16通道 scale，V 沿 sequence 方向量化，三种物理布局不同 | 现 NVFP4 GEMM 的 global-scale、128×4 scale swizzle、approx reciprocal profile 不等于此 attention quantizer；不能直接复用其输出 ABI |
| compression | 原 F32 pooled scores 与 pooled V，全 tile softmax，最后转 BF16 再 gate addcmul | 可用现 F32 structured GQA 的 full-dense 数学契约做诊断，不能据此声称共享 scores、算术顺序与 torch matmul/softmax 已一致 |

融合源：`minimax_h3_fusions/modulation.py:55–70,98–122`、
`qknorm_rope.py:49–72`、`swiglu.py:49–54`。实际部署的 SwiGLU 缓存为 Triton3.7.0、
sm120；PTX 使用 `sub.f32`、乘 `0x3fb8aa3b`、`ex2.approx.f32`、`add.f32`、
`div.full.f32`、两次 `mul.f32x2`、`cvt.rn.bf16x2.f32`，这些指令没有 `.ftz`。
外部 `_minimax_h3_swiglu_kernel.ptx` SHA256 为
`efcdd0873113efaa594810426ddfdd396acc0c08504e7f84090434582a585609`。
这是 profile 来源证据，不是以 hash 为正确性测试门槛。

可实现的 runtime operator 分割如下；名字描述职责，不宣称已有注册 provider：

| 操作边界 | 明确输入/输出与资源 |
|---|---|
| tile gather / untile | BF16 hidden 或 QKV、显式 row map → tile-padded BF16 / packed BF16；padding 写零、逆映射唯一、不跨 document。现 row gather 只支持至多32段且 stride0/1，不能以海量单行节点冒充高效3D搬运 |
| masked tile mean | BF16 `[B,L,H,D]`、每 tile valid count → F32 `[B,T,H,D]`；只纳入原 valid rows，不把 tile pad 当有效 key |
| region selection | pooled Q/K、不可变区域范围/预算 → 动态 checked block lists；每 B/head/query 独立，明确 dot 精度和 top-k tie policy。现 structured mask 是静态 causal/bidirectional/valid-key，不能表达该 mask |
| attention preconditioning | 显式 K-mean 域、FP8 prefix profile、FP4 video profile；量化缓存只能按布局/profile/完整输入生命周期复用 |
| FP4 sparse attention | 独立 Q/KV 长度、packed Q/K/V/scales、counts/indices/half-valid → BF16 output；两级 P 是 typed profile。原 kernel headers 可通过现 native operator-set/ABI 机制接入，Torch `empty`/LSE/zero-delta/workspace 必须改为 Core 预算内 buffer，不能在 provider 偷配 |
| prefix dense attention | FP8 roundtrip 后 BF16 Q/K 与 BF16 V；Q 比 KV 短。有效 key 可 compact 后复用 `CROSS_ATTENTION_FUSED_BF16`，必须保留供应者数值差异证据 |
| compression / fused pointwise | pooled score reuse、V mean、softmax/matmul、row gather/gate；以及前述 fused norm/modulate/RoPE/SwiGLU profile |

native 证据：`attn_qat_infer/api.py:203–282,337–364`；
`quantization/fp4_quantization_4d.cu:185–228,315–357`；
`blackwell/api.cu:217–340,371–421`。Kernel 输出与 LSE、FP4 payload/scales、
dynamic list、tile maps、pooled scores/values 都需要算入实际容量；不要只计算 GEMM
权重。FP4 sparse 与 dense BF16 不是同质量路径的等价替换。

当前 Rust 切片的消费关系为 `FastH3VsaLayout::ref2va` 调用已有
`FastH3PackedLayout::ref2va`，再由 `lower_mask` 产生 private-field checked descriptor。
只暴露 tile128，拒绝 checkpoint 的64/256；prefix 必须非空。构造前校验声明的
rows/tiles/metadata 容量、整数与 native int 维度；每个 packed row 精确对应一个
非 padding tile slot；dense segments 不互相混 tile；reference-video regions 和
target-video region 各自连续，target 最后。每 tile 的 valid rows 紧排在开头，
不能保留空间边界中的洞后只用一个 valid count。

`lower_mask` 验证 shape、容量、全 prefix visibility、各 region 预算；输出每行
倒序 key-block list、计数和每64行半块的 valid count。它不计算 pooled scores，
也不证明外部 bool mask 是 score top-k。tile128 没有 quadrant mask；prefix/video
切分以完整128行 block 为界，video Q 更短但 KV 仍是完整文档。不同 batch 可以有
不同 mask，同一个 descriptor 必须共享几何/valid-count。后续 continuous batching
必须按该能力组成 cohort，或另加明确的 per-document descriptor ABI，不能将两个
请求拼成一个 VSA region。CUDA provider 仍是下一消费者，当前 CPU 代码不是 GPU 完成证据。

避免复制整份 P-H3 block 的最小迁移边界：

- 将 `p_h3_image/dit_forward/fragment.rs` 的 stateless family composition 提升为
  models 内共享组件；原 P-H3 refiner/stack/final-layer 当场继续消费，FastH3 block
  拼接 time-conditioning、projection、FFN fragments。保留所有输入映射、namespace、
  weight component、numerical operation 冲突与完整 geometry 检查；仍是一张 graph 和
  一个 Core executor，不引入第二执行循环。
- 从 FastH3 FFN 的 quantize→NVFP4 matmul 提取实际 linear fragment，让 FFN 两层与
  Q/K/V/gate/to_out 共同消费原 Stored operand/source 与 scalar derivation。现
  P-H3 的 Comfy W4A8/W8A8 projection 继续原 profile，不能共享假定同一量化算法的函数。
- row selection 可以共用经过校验的 run/parameter-row 几何；调用者选择具体数值
  operation。P-H3 block 与 final-layer 是现有消费者，但不同舍入边界保持独立。
- packed 顺序、三模态时钟、PDD target-only flow、VSA区域、RoPE坐标、compression
  gate 属 FastH3 family。Qwen 的 norm/attention 结构不能为“共用 block”被强行改写。

下一稳定点应先完成上述元数据 fixture 与边界测试，然后在真实 block 的同一输入上
核对 tile mapping、QKV/量化数据、pooled scores、mask/list、prefix/video 分支、
compression 和最终 hidden。top-k tie 与浮点 profile 的允许差异须提前说明；端到端
质量及同配置 latency/throughput 仍须另行实测。完整模型、run/serve 与 continuous
生命周期验证不由单 block 或编译通过替代。
