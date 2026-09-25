# 有限未来 owner 审计

`calibrate-slo` 的 `required_future_audit_v2` 模式在真实 cohort 的事前指定波次尝试前，读取同一 live snapshot 并投影有限路径。它记录未来各物理替代需要的 owner/domain、数值 support、pending/Length 联合要求和成本 Unknown。它不训练、发布波次或证明时限可行，也不代表生产 planner 的所有候选已经枚举。

原 `structured_discovery_v2` 仍只发现实际执行输入，不允许导入模型；新审计模式必须导入真实 profile10 或 catalog，以及匹配的 prefill reference。不能把发现 JSON 转成 receipt 或把旧 profile8 改名当 seed。

## 从真实数据开始

固定同一个可执行文件、模型/权重、数值策略、后端、资源容量与请求转换。所有阶段使用现有完整请求 driver；不为了收够样本提前取消请求。配置/源身份不匹配由原 loader 拒绝。

1. **先采固定 reference。** 使用现有 `validation_model.kind=exported_profile` 和顶层 `reference`，执行独立 singleton discovery → 冻结计划 → 新 trials → 原 accepted cut 导出/source join。具体配置见 [SLO 配置中的真实 singleton reference](slo-configuration.md#从真实-singleton-试验发布固定参考)。reference 定义工作评分单位；不要求已有合格 V2 模型。若需要覆盖多个 prompt 长度，使用已声明的 V2 piecewise reference 原生流程；不能靠一条曲线外推其他输入。保存 report 的 `summary.reference.artifact.path` 及 `protocol_sha256`，原来源文件继续保留。这个阶段的 legacy cost profile 不是后面审计导入的 V2 seed。
2. **独立 actual discovery。** 使用 `structured_discovery_v2`、原模型/输入和实际路线；从逐波 raw 的真实输入确定一个窄且可校准的 owner、窗口和挑战。实际输入 Known 不等于成本模型 Known，不将未发现的 Full/pending/终态组合默认为已覆盖。
3. **冻结窄 seed，再采独立三阶段。** 使用现有 `structured_whole_wave_v2`，事前固定 `capture.scope`、`membership_rule`、三阶段成员数和完整 cohort。training/residual/validation 分别是 fit/residual/qualification；成功时产品重放原 source3 才导出 profile10。期望不足或 scope 失败保留失败源，不能从 raw 选一批“好”行重编号。最少样本数不表示 p99 保证。这个 seed 可以只覆盖一个普通 decode owner，其他未来 owner 在审计中仍可明确 WrongDomain。
4. **立即运行新 audit。** 使用 seed 与 reference 的实际路径和协议摘要。保留默认样本 TTL（当前 300 秒）及原 source/phase 时钟；reference 宜先准备好，再启动新 seed 三阶段。若完整三阶段或后续启动已超龄，必须报告 Stale，不能延长 TTL、改文件时间或重写 closed clock。

上述步骤都使用同一既有命令，例如：

```text
ferrum calibrate-slo MODEL --backend metal --startup-usage serve --manifest reference.json --slo-config reference.toml --observations reference.raw.jsonl --out reference.report.json
ferrum calibrate-slo MODEL --backend metal --startup-usage serve --manifest discovery.json --slo-config discovery.toml --observations discovery.raw.jsonl --out discovery.report.json
ferrum calibrate-slo MODEL --backend metal --startup-usage serve --manifest seed.json --slo-config seed.toml --observations seed.raw.jsonl --out seed.report.json
ferrum calibrate-slo MODEL --backend metal --startup-usage serve --manifest audit.json --slo-config audit.toml --observations audit.raw.jsonl --out audit.report.json
```

这里的 MODEL 与文件名是用户提供的实际输入；数值 profile、模型源/模板、runtime memory budget 和产品容量等既有参数必须按实际基准一致传入。CUDA 使用其真实后端与相同 pinned native 配置；这些示例不是已测性能命令。每次输出路径必须全新，不能覆盖输入 artifacts。

## Audit 声明

沿用 schema 1 rendered 或 schema 2 pinned selection manifest 的完整 `training` cohort；它在此模式是诊断 cohort，`validation=[]`、顶层 `reference` 不得出现。不是重新采参考。SLO policy 必须显式 Observe、CompleteRequests、credited、StructuredWholeWaveV2、HostSettledV1，指定 `cost_profile`、`prefill_reference` 和原导入时钟误差声明，禁止 profile export。artifact 内容仍由真实 startup loader 校验，路径声明本身不授资格。

以下片段声明在第 0 个 cohort 的第 0 次重复、第 2 个非空 Wave 尝试之前，对当前完整 frontier 表中的第 0 行检查连续两次 decode：

```json
"validation_model": {
  "kind": "required_future_audit_v2",
  "warmup": [],
  "audit": {
    "triggers": [{
      "case_index": 0,
      "repetition": 0,
      "before_wave_attempt": 2,
      "plan": {
        "paths": [{
          "waves": [
            [{"frontier_index": 0, "action": {"kind": "decode"}}],
            [{"frontier_index": 0, "action": {"kind": "decode"}}]
          ]
        }],
        "limits": {
          "budget_ms": 1000,
          "maximum_queries": 32,
          "maximum_coordinates": 4096
        }
      }
    }]
  }
}
```

这只是语法示例。第 2 次尝试不必已经 decode；若实际上仍在 prefill，原合法动作校验会明确停止。必须在运行前按真实输入长度和声明 chunk 分区确定路径，不能运行后把 trigger 移到首次 Known。Prefill 动作为 `{"kind":"prefill","count":128}`；offset 由 live frontier 推进，不接受任意历史偏移。frontier 的完整身份和顺序写入原报告。

每个 cohort 的尝试号从 1 开始，实际 NotSubmitted 也消耗尝试号。未知、预算耗尽或其他审计错误不会隐式重试。cohort 提前终态而未达到 trigger 时，报告保留未触发索引。原请求仍运行至自己的真实终态并排空输出。

单次最多 8 路径 × 每路径 16 波，每波最多 256 行；独立审计预算最多 30 秒、1024 queries、65536 coordinates。整份 manifest 最多 64 triggers、8192 query 预算与 1048576 coordinate 预算。原 raw 大小和全运行超时同时适用。独立诊断预算不是执行 witness 或默认 2 ms 规划预算的延长。

合法路径仍经过生产 readiness、chunk alignment/端点、总 token cap、mixed 策略与输出 credit 逐波消费。没有模拟服务时间，不能假设 credit 自行恢复。所有路径从同一个真实 snapshot 出发，保留全部物理 alternatives；成本 Unknown 可继续结构需求审计，动作/资源/graph Unknown 则保留原因并停止，不跳过后继障碍。

## 阅读结果与多 owner 后续

raw 事件 `required_future_owner_audit` 保存声明、原 snapshot/frontier、每 path/wave/alternative 的需求以及 input/cost Unknown；配置错误/API 失败保留 error。summary 的 `required_future_audit_v2` 区分已触发、缺失、停止、截断和各 Unknown reason。

`all_declared_requirements_recorded=true` 只表示这些有限声明路径取全。成本 KnownAtRead 仍有其原 model age，不能当提交时可用的时限或执行许可；不发布 `complete feasible plan`。

已有 profile10 catalog 可同时导入多个真实合格 child，并按唯一 owner/domain 选择，禁止 Unknown 后换便宜模型。每个 child 保留自己的完整 source/FIFO/三阶段时钟。当前采集 CLI 仍是**单 owner 三阶段**；本片不提供一次 cohort 同时训练多个 owner，也不能保证依次采完几十个 child 时最早样本仍在 300 秒内。若顺序采集超龄，这是真实阻塞，不是扩大 TTL 的理由。下一步需按本审计得到的必要集合，复用完整 cohort 同时分发给有硬上限的现有单-owner collectors；必须保留每 child 原 FIFO/outside/失败槽及统一 cohort phase 边界，不能按结果或 Known 筛样。这一多-owner 采集接线尚未在本功能中实现。
