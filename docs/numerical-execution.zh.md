# 数值执行策略

本文说明下一版开发接口；v0.8.9 发布资产尚不提供 `--numerical-profile`。

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

## 分阶段算术合同

`NumericalOperationContract` 的可选 `staged_arithmetic` 用于单一乘法/累加类型
无法表达的混合算术。首个阶段 schema 的版本为 `1`，描述一个量化投影的执行顺序：

| 阶段 | F32-scale Q8 点积示例 |
| --- | --- |
| 激活量化 | F16 输入，每 32 值共享 F32 `max(abs(x))/127`；除法按 F32 最近值舍入，整数 ties-away，夹取到 −127…127。零组写正零 scale/q0；非有限组写 NaN scale/q0。 |
| 局部整数点积 | I8 × I8，每 4 值累加到 I32，再进入重缩放；32 值的量化组不等于 32 值的 I32 累加范围。 |
| 重缩放及 min 校正 | I32 点积转为 F32，激活 scale、权重 scale/min 系数及计算均为 F32。可显式声明对同一局部 q 的 I32 和做仿射 min 校正；无 min 的格式声明 `none`。 |
| 浮点归约 | 重缩放的部分和继续在 F32 累加；具体归约顺序由版本化 operation 定义。 |
| 输出舍入 | F32 转 F16，最近值、ties-to-even；输出类型必须与 program 的声明边界一致。 |

阶段顺序、类型衔接、局部整数溢出界和量化分组均参与验证。允许 FMA 与禁止 FMA
是不同合同值，进入 fingerprint；当前示例使用 `disallowed`。局部 dot4 与完整组
整数点积、ties-away 与 ties-to-even 也不能共用同一合同身份。阶段声明本身不证明
数值正确，也不承诺跨 profile、provider 或 batch 的输出逐位相等；这些需要各自验证。
Schema 1 只接受当前实现的 F16 激活输入；非零 F16 组的 F32 scale 不会下溢为零。
F32/BF16 输入须先版本化有限 scale 下溢规则，不能直接扩大输入类型。F64 尚不是支持的
`ElementType`。`min_correction` 的 `{"kind":"none"}` 是严格空对象，附加字段同样拒绝。

有阶段声明时，旧 `multiplication_type` 和 `accumulation_type` 必须都为 `null`，
不能再用“I8 乘/F32 累加”这一对字段掩盖 I32 局部点积和 F32 重缩放。没有阶段声明的
旧合同仍只接受浮点算术。此 schema 不注册新 operation、provider 或 Auto 候选，
也不会改变 `run`、`serve` 的实际选择；上线仍须由 family、版本化 operation 和真实
provider 完整声明并验证，不能从模型名称推断。

融合操作可改用互斥的 `composite_arithmetic`：引用 strict base 的精确 operation ID/版本，
只覆盖具名 projection 的 weight input ordinal、内部激活输入/输出类型及分阶段叶子。
Composite schema 1 核对标准 SwiGLU 的槽位 1/2 和 F32-master GDN 的槽位 2/7；
内部投影均为 F16，GDN 外部输入/输出仍为 F32。归一化、非线性、状态和残差保留 base 语义，
不能把整个融合操作标为五阶段整数投影。Program 的属性、输入/输出数量及已知边界还须匹配 base。

叶子显式声明 Q4_K/Q5_K/IQ4_XS block ABI 和静态 K/N 条件；整数 partial 必须整除 32 值权重系数组。
不匹配格式/形状、带变换的权重以及未覆盖角色保留 strict 算术。Fallback 是合同的一部分；
eligible 叶子缺 provider 支持仍应拒绝资格验证，不能借 fallback 换算术，也不能按 batch 偷换。
`declared_projection_arithmetic` 只解释传入的静态事实，不检查真实 tensor 或证明叶子执行；
物理 parts、provider、运行证据和 `run`/`serve` 接入仍需另行验证。

兼容规则：旧 JSON 缺少 `staged_arithmetic` / `composite_arithmetic` 时仍按原字段读取；重新序列化省略它们，
旧 profile 的 canonical bytes 和 fingerprint 保持不变。新字段带显式 schema 版本，
未知版本、字段或阶段会被拒绝；旧 reader 同样拒绝新字段，不能悄悄降格为旧单阶段合同。
算术语义变化仍须使用独立 operation 身份或不兼容主版本，并重新生成依赖的 profile/plan；
不能仅把 schema 版本改成已知值来复用旧计划。Rust 结构体调用方需为既有浮点合同显式填写
`staged_arithmetic: None` 和 `composite_arithmetic: None`。

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

### 显式 CUDA FFN Q8act G32 profile

`run` 与 `serve` 共用 `--numerical-profile qwen3_5.f32-master.ffn-iq4xs-q8act-g32`
的 Require 路径；这是实验性 profile，Auto 保持原选择。该 profile 只覆盖 dense SwiGLU 内满足完整 K256
block 条件的原生 IQ4_XS 投影，使用每 32 激活的 Q8/F32 scale 和每 32 值的 I32 dot。
它不是上文 dot4 示例，也不改变 GDN、注意力、状态、残差或 SiLU 的严格算术。
当前 family 资格限于未旋转的 dense native GGUF、negative-rate recurrent ABI 和 F16 KV。

另一个显式选项是 `--numerical-profile qwen3_5.f32-master.ffn-q4k-q5k-iq4xs-q8act-g32`，
在同一范围内覆盖 Q4_K、Q5_K 和 IQ4_XS。Q4_K/Q5_K 的非对称 min 项使用同一组
Q8 激活整数之和作 I32 修正，再按合同分别执行 F32 rescale/min 和累加；不得改用原始
F16 激活之和。旧 IQ4-only profile 的数值声明保持不变，Q4_K/Q5_K 在旧 profile 下仍走 strict。
两个 profile 的导出/设备资格独立核验，新 profile 缺 Q4_K 或 Q5_K 内核不取消旧 profile 的资格。
这些选项均不进入 Auto，也不扩大 KV、MoE、变换或非 FFN 操作的支持范围。
Q8 SwiGLU provider 当前未声明 checkpoint 导出/恢复合同；graph replay 支持不能替代该能力声明。

准备阶段依据真实物理 parts 冻结每个 projection 的 staged/strict 决策，并共同用于
scratch 估算、dispatch 和 replay 身份。不支持的格式、变换或形状按合同保留 strict；
eligible 叶子缺少 G32 导出、设备能力或加载/执行失败时拒绝运行，不静默切回 strict。
临时 pack 按实际 M 和最大 eligible K 占用 `M × K × 9/8` 字节，gate/up 与 down
顺序复用；另保留 fused/transform scratch 和最多 15 字节对齐，不扩展常驻权重。

effective config 与 `/health` 的 `numerical_execution.prepared_projection_numerics`
列出节点/provider/实际 component 的准备决策及 strict 原因。它是静态库存，不是
内核实际调用次数或质量证明。此次接入的模型质量、共享 `run`/`serve` 运行和整模型收益
仍须分别验证；不能从叶子实验或准备成功推断。
