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
