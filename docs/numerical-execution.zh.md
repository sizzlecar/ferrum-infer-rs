# 数值执行策略

本文说明开发分支的数值执行接口；具体 profile 的可用性由构建及设备能力决定。

数值 profile 规定模型中间结果、运算和状态的精度合同。例如，Q4 GGUF
描述权重的物理编码，推理过程仍可使用 FP16、FP32 或混合精度。选择 profile
不改写源文件、不重新量化，也不保证某种精度一定更快；性能和误差需要按模型、
后端及实际负载测量。

## 默认与显式选择

`run`、`serve`、`bench` 共用 `NumericalExecutionPolicy`。未配置时使用 `Auto`；
模型 family 声明已验证的候选顺序，实际设备的算子、权重物化和资源合同决定
哪个完整候选可以执行。启动完成后不会在 token 循环中切换策略。

开发构建可显式指定完整 profile ID，例如：

```bash
ferrum run qwen3.5:4b-q4_k_m --backend metal --numerical-profile qwen3_5.f32-master
ferrum serve qwen3.5:4b-q4_k_m --backend metal --numerical-profile auto
```

CLI 参数覆盖配置文件；显式 `--numerical-profile auto` 也会覆盖配置中的固定选择。
在现有 JSON 配置的顶层设置 `"numerical_execution": "auto"`，或设置
`"numerical_execution": {"require": "qwen3_5.f32-master"}`。库入口通过
`EngineConfig.numerical_execution` 使用相同的 `Auto` / `Require(profile_id)`。

显式选择只尝试指定的候选。未知 ID、缺失算子、不兼容的物化方案或不足的规划容量
会拒绝启动；不会换 checkpoint、换后端或回退到 legacy 来满足请求。没有注册
数值 profile 的 legacy 模型不能接受 `Require`。

## 当前声明与验证范围

Qwen3.5 声明 `qwen3_5.f16` 和 `qwen3_5.f32-master`。F32-master 保持混合精度：
主干、残差和 logits 使用 FP32，FFN、KV 和卷积状态保留 FP16，delta 状态沿用
模型的状态精度约束。名称不代表每一步都用 FP32。

`Auto` 保留既有、已验证的编码与 recurrent 参数 ABI 组合。声明两个 profile
不意味着所有 GGUF、SafeTensors、Metal 和 CUDA 组合都可用；新增 GGUF + F16
组合在独立数值验收前不会进入默认候选。其他已迁移 family 先保留原有的单一方案。

CUDA SM80 及以上设备可显式选择 `qwen3_5.f32-master.q8-swiglu`。
它仅对 FFN 中原生 GGUF Q4_K、Q5_K、Q6_K 权重的矩阵乘法量化激活：
每 32 个 FP16 输入使用一个 FP32 scale，转换为有符号 INT8，整数点积后以
FP32 缩放和累加。gate/up、SiLU 结果及 FFN 输出仍有 FP16 舍入边界；
其他权重格式和 attention、输出层沿用原运算，主干保持 FP32。
激活量化会改变数值结果，独立算子正确性不等于模型质量或服务性能提升。

此 profile 不进入 `Auto`。它要求原生权重、negative-rate recurrent ABI、
至少一个满足 K256 分组的 FFN Q4_K/Q5_K/Q6_K 权重，并仅支持非 MoE、
无 Hadamard 旋转和 FP16 KV 的组合。`run` 和 `serve` 使用同一选择：

```bash
ferrum run qwen3.5:4b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.q8-swiglu
ferrum serve qwen3.5:4b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.q8-swiglu
```

激活打包临时空间进入正常显存规划，gate/up 共用一次打包，down 复用空间后
重新打包。无对应设备实现或容量不足时，显式选择按既有规划规则拒绝执行。

同一 CUDA 设备范围还可显式选择
`qwen3_5.f32-master.q8-swiglu-gdn-projections`。它在 Q8 SwiGLU 的基础上，
仅将 Gated DeltaNet 输入投影（逻辑 Q/K/V/Z/b/a 的物理叶子）和输出投影中
符合条件的 Q4_K/Q5_K/Q6_K 矩阵改用同一 INT8 激活、FP32 scale 策略；
其他原生权重叶子保留严格运算。投影输入和结果仍有原 FP16 舍入边界，
主干与残差保持 FP32，卷积状态保持 FP16，递推核心与 delta 状态保持 FP32。
这不量化完整 recurrent state，也不改变 causal attention 或 LM head 的运算。

新 profile 除满足 Q8 SwiGLU 的资格外，实际 GDN 层还必须至少有一个满足
K256 分组及真实 block metadata 的输入或输出投影叶子。只有名字相同的
权重、FFN 中的量化权重或其他格式不构成 GDN 资格。非 MoE、无 Hadamard、
negative-rate ABI 和 FP16 KV 限制继续适用；它只接受显式 Require，不进入
`Auto`，原有 profile 的内容和自动顺序不变。两个产品入口使用同一 ID：

```bash
ferrum run qwen3.5:4b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.q8-swiglu-gdn-projections
ferrum serve qwen3.5:4b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.q8-swiglu-gdn-projections
```

GDN 输入投影的合格叶子共用一次打包；输出投影复用规划空间后重新打包。
独立 Q8 投影近似和不同 batch 映射的 FP32 归约可能产生数值差异；固定已量化
投影输入上的分块递推测试，不等于完整 GDN 在任意 batch 或分块下逐位一致。
已有的同形状 eager/replay 能力语义不因此扩展。小规模算子与状态延续 oracle
不构成模型质量或服务性能结论，仍需按选定 profile 验证实际模型输出与负载。

## 显式 RN-F16 fragment FFN

`qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8` 是独立的显式
profile，不进入 `Auto`。它基于 `qwen3_5.f32-master.gguf-f16-projections`：
合格投影权重由原 GGUF 解码后按 round-to-nearest 舍入为 FP16（RN-F16），
主干与残差保持 FP32。新 profile 为 FFN 增加 fragment MMA 路径；attention、
embedding、输出层、KV 和 recurrent state 沿用该 RN-F16 profile 的合同。

它要求 CUDA SM80 及以上设备，且当前构建实际嵌入的 PTX 目标支持 SM80 及以上
的 MMA 实现；只有设备型号满足条件不足以启用。模型必须是原生 GGUF、非 MoE、
无 Hadamard 旋转，采用 negative-rate recurrent ABI 和 FP16 KV。每一层 FFN 的
gate/up/down 都必须是具有真实 block metadata 的 Q4_K、Q5_K 或 Q6_K，矩阵 K
须为 256 的倍数；同一层 gate/up 的格式和形状必须相同，down 可独立使用上述
任一格式。未满足格式、设备实现或容量合同的显式选择会被拒绝。

`run` 与 `serve` 通过相同的数值策略和物化工厂选择此 profile，例如：

```bash
ferrum run qwen3.5:9b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8
ferrum serve qwen3.5:9b-q4_k_m --backend cuda --numerical-profile qwen3_5.f32-master.gguf-f16-projections.ffn-rn-fragment-m1to8
```

FFN 按一次完整物理执行的 token 行数 M 选择路径：M=1..8 使用 fragment MMA，
M>8 使用既有 dense RN-F16 GEMM。这个边界也适用于小段 prefill；批内某个请求
或子段的行数不单独决定路径。gate/up 输出、SiLU 乘积和 down 输出保留 FP16
舍入点，累加使用 FP32；不同归约实现仍需独立数值验证。

冷启动物化从同一组有序 source 各读取一次，同时生成 dense RN-F16 权重和
fragment packet。Gate/up 合并后按完整 N 打包与尾部补齐。两份执行表示都计入
资源预算，另需计入对齐、KV 和工作区；较小的 packet 不代表省去 dense 权重，
GGUF 文件大小也不能用来推断所需显存。具体容量由完整资源计划检查。

此 profile 保持 `Approximate` 资格。物化时的逐系数 RN-F16 校验和算子正确性
不能代替实际模型的 teacher 数值验收；即使已有其他 RN 或 Q8 profile 的资格，
也需要对此完整 profile 使用独立 teacher 对照。真实模型质量与服务性能目前
仍待测量，不能据算子筛查结果推断输出 TPS、TTFT、TPOT 或 ITL 改善。性能评估
须在同模型、同硬件和固定服务容量下验证完整负载及预声明 SLO。

## 来源、计划和缓存

准备阶段先解析 typed 配置、源 schema、模板与 profile 声明，再用选定 profile
生成完整 program。成功的静态编译结果、实际 registry 和 materializer 一起用于
绑定及初始化，避免预检后重新选择。

源下载缓存可以复用。同源不同 profile 的 program、执行计划和依赖这些合同的
运行身份分别记录；物化缓存依据实际源与执行权重 ABI 判断可复用性。
Prepared family 和 resolved plan 的 wire 版本为 2；缺少数值语义的旧计划必须
重新解析，不能补默认值后直接复用。

`--effective-config-json` 中的 `numerical_execution.requested` 记录请求；
`ResolvedModelPlan.numerical_execution` 记录实际选择、版本、静态计划身份和之前
候选的拒绝原因。反序列化得到的字段不能自行证明可信，重建需要注册的模型定义、
实际静态计划以及外部验证上下文。
