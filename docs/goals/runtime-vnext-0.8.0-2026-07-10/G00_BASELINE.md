# G00: Legacy 基线与事实冻结

## 状态与依赖

- 状态：Open
- 父目标：[`GOAL.md`](GOAL.md)
- 依赖：G00F 无；G00M1-M3 随对应模型迁移；G00P 在 G09/G10 前完成
- 下游：G00F -> S0/S1；G00M1-M3 -> 对应模型 parity/legacy 删除；G00P -> G09/G10

## 目标

把当前 `cff4c47765ef3259b8a04890187d99c60da86394` 的代码、产品行为、性能、编译时间和已知
失败冻结成可复现 artifact。完整 G00P 必须在正式性能和发布结论前完成，但不再阻塞 S0 contract
拆分和 S1-S5 production refactor；执行顺序以
[`EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md`](EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md)
为准。G00 不允许通过文档推断或旧 summary 代替真实命令。

## 分层与执行时机

- `G00F`：facts checkpoint。只冻结 source/coupling inventory、model/revision/weight、preset、
  historical catalog 和最小入口事实，是 S0/S1 的唯一前置 baseline。
- `G00M1/G00M2/G00M3`：逐模型 just-in-time baseline。在删除对应 legacy entry 前采集必要的
  correctness、token/resource parity 和 performance smoke；未受影响模型不随之重采。
- `G00P`：本文件后续定义的完整三模型双端、external/legacy、ABBA-BAAB、build timing 和
  artifact authenticity gate。它只阻塞 G09/G10。

现有 G00 `collecting` root 和 partial lane 全部保留为 historical/diagnostic evidence。策略或 source
改变后可以 stale，但不得复制为新 PASS，也不得继续阻塞 S0-S5。

## 工作项

1. 建立机器可读 `runtime_vnext_models.json`，锁定 [`MODEL_MATRIX.md`](MODEL_MATRIX.md)
   的 repo、revision、文件、SHA256、backend、format 和 hardware。
2. 构建冻结 legacy CUDA/Metal product binary，记录 features、binary SHA256 和 build logs；
   legacy `run`/`serve` 执行身份保持 `cff4c477...`。
3. 从独立 G00 collector commit 构建 canonical `bench-serve` client。该 client 只允许增加
   typed `--enable-thinking <bool>`、warmup outcome、`BenchReport.repeat_metrics`、按完整 SSE line
   增量解码的 byte-safe UTF-8 framing，以及 typed per-request ITL provenance/eligibility 证据；不得
   改变被测 server、payload 语义或调度。必须记录 source/tree/binary SHA；A/B 两侧使用同一 client，
   不能冒充 legacy server SHA。
4. 对三个主模型执行六个真实 product lane；当前 unsupported 必须输出明确 BLOCKED artifact。
   每个 `run`/`serve` child process 必须从显式 non-product allowlist 构造环境；不得继承 collector
   进程中的 `FERRUM_*`。完整 sanitized env、argv、PID/PGID 和 process start identity 写入 receipt。
5. 运行 same-host vLLM 与 llama.cpp 外部基线。
6. 固化 Qwen3-Coder empty-answer、run/serve config drift、UTF-8、graph key、资源 rollback 等
   historical bug inputs；为 G02 H01-H15 的每个 concrete case 写 evidence ref，并冻结总数 `M`。
7. 测量 CUDA no-op、Rust model leaf、runtime leaf、core PTX、Marlin TU、clean release
   六种构建场景；model/runtime leaf 必须分开，避免掩盖不同失效域。
8. 生成 current dependency/coupling inventory：trait methods、arch-named APIs、cfg branches、
   env reads、duplicate product decisions、legacy factories、model-specific runners。

## 硬件与成本

- CUDA：exactly 1x RTX 4090。开始 paid lane 前按 `AGENTS.md` 记录 inventory、预计时间/成本、
  correctness command、performance command 和 stop condition。
- Metal：固定本机 `32GB / 24-GPU-core Apple M1 Max`。M2 使用固定的 Q4_K_S 文件，并在
  启动和每个测量 cell 前执行 typed memory preflight；任何 active swap growth 或不足
  `2 GiB` 的实测物理 headroom 都是 REJECT，不能把 Qwen3.5 Metal 从总目标删除。
- 基线采集完成后必须停止或明确 bounded cache-retention，不允许 GPU 空闲计费。

MODEL_MATRIX 为 vNext 最终态定义的 active-concurrency floor 不得倒灌并改写冻结 legacy 行为。
G00 对每个 legacy/external cell 原样保存 typed cap、observed max-active、完整 active timeline 和
eligible-interval duty-cycle；legacy cap 低于最终 floor 是待 G08/G09 修复的 baseline fact，不是
G00 failure，也不能在 artifact 中手填为目标 cap。硬件/format amendment 只改变 reviewed amendment
之后的新 reference lane，并使受影响 G00 external/model lock stale；`cff4...` legacy binary 身份不变。

## 代码量与构建输入分类口径

G00 必须同时保存文件路径、内容 SHA256、语言、生产/测试/generated/vendor 分类和逻辑行数，
后续 Goal 以内容身份而不是新路径重新归类，禁止通过移动、改名或拆文件满足减少指标。

- `production LOC`：Rust/C/C++/CUDA/build script 中非空、非纯注释的逻辑行；Rust 文件中
  `#[cfg(test)]` module 及其后代计入 test，examples/fixtures/generated code 单列，不进入分母。
- `model-specific scaffolding`：由模型/family 模块拥有的 setup、admission、batch orchestration、
  state transition、finalize、cleanup、backend dispatch 和重复 decode loop；权重字段声明、真实数学
  program/op 描述及格式 parser 不计入脚手架。G00 人工审阅边界并保存 symbol allowlist。
- `large third-party native source`：一个 vendored/native source tree 满足任一条件即算一个 build input：
  `>=10,000 production LOC`、源文件总量 `>=5 MiB`，或包含 `>=10` 个 C++/CUDA translation units。
  同一 upstream revision 被拆目录仍按一个 content-root 计数。
- 后续新增未被 G00 分类覆盖的文件，由 validator 按相同规则分类；争议项不能默认为 excluded，
  必须在 artifact 中由 owner、理由和 review commit 显式裁决。

所有 LOC、减少比例和 build-input 数量都必须由同一个 checked-in analyzer 从 G00 inventory 与候选
inventory 计算；手工表格不构成 PASS 证据。

## G00P 完整验收

G00 冻结 legacy 事实，不把 G05/G08 才新增的 v0.8.0 产品合同伪装成 legacy 已有能力。
`runtime_vnext_legacy_correctness_expectations.json` 是逐 model/backend/scenario/variant 的唯一
legacy 期望源：

- `pass` 表示已有证据支持且本次真实执行必须通过；
- `known-fail` 表示本次必须精确复现已锁定 failure class，并分配下游 Goal；unexpected PASS
  同样拒绝，必须先提交 expectation amendment；
- `blocked` 只表示该 legacy lane 没有可执行能力，不能用于掩盖已支持路径的失败；
- `discovery-required` 只允许 discovery 命令产出 observation artifact，不构成正式证据且不得
  打印 PASS。观察结果必须通过独立 commit 固定为前三种状态，再完整重跑 canonical lane；
- collector-only 的单个 versioned scenario contract 发生变化时，允许只对该 scenario 做 scoped
  discovery，但必须执行该 scenario 的完整 case matrix，绑定 checked-in runner/contract/input/output
  oracle SHA，并证明其余 scenario rules 的规范化内容 SHA 未变。禁止任意 case-id 列表、多个 scenario
  拼接或在 canonical 模式使用 scope；amendment 提交后仍必须完整重跑该 model/backend canonical lane；
- 正式 G00 artifact 中 `discovery-required`、现场改写 expectation、skip 和 waiver 数均为 `0`。

因此 G00 的 legacy lane PASS 表示所有已锁定期望都被真实逐 case 执行并完全匹配，而不是声称
legacy 已经实现 C01-C21 的最终 v0.8.0 合同。M3 两个既有 release-critical `run`/`serve`、
stream、multi-turn、usage、终止和坏输出扫描子集仍必须全部真实 PASS；C01-C21 的最终全绿属于
G08/G10。该区分不降低最终目标，只防止基线阶段要求未来功能倒灌到冻结二进制。

- 三模型 x CUDA/Metal 共 6 个 lane 均有 PASS 或真实 BLOCKED artifact；waiver 数 `0`。
- M3 Qwen3-30B 的 CUDA/Metal 两个既有 lane 必须全部 PASS。
- M1/M2 Qwen3.5 Metal 必须准确证明当前 unsupported，不得伪造 baseline。
- 每个存在 Ferrum legacy 与 external 两个可执行实现的 comparable performance cell 使用外层
  `ABBA-BAAB`；每 slot `100 requests x 3 inner repeats`，因此每实现每 cell `1200`
  measured requests、12 个由 canonical `BenchReport.repeat_metrics` 直接保存的 repeat samples；
  warmup outcome 也必须来自同一 `RunRecord`。Ferrum lane BLOCKED 时只采 external standalone
  baseline，并明确标记 `comparable=false`，不能伪造 A/B ratio。
- 外部 vLLM/llama.cpp baseline 与 Ferrum 使用相同模型、format、dataset 和 hardware。
- 正式 artifact 字段完整率 `100%`，空 build/runtime log 数 `0`。
- 每个 HTTP raw report 的 `itl_evidence_per_request` 与 token vectors 都必须为 `3 x 100` 且按
  completion order 对齐；逐 request 的 source/event/usage/interval/transport-coalescing/eligibility
  字段完整率、由原始字段重算 eligibility 与 repeat summary 的一致率均为 `100%`。任一 request
  ineligible 时该 repeat/cell 不得生成正式 ITL ratio；TTFT/TPOT/throughput 事实仍独立保存。
- client transport chunk 合并只能记录 `transport_coalesced_output_chunks` 并令该 request 的 ITL
  ineligible；不得写成 server `stream_bulk_flush`、不得令本来成功的请求失败。server bulk flush
  只能由 G06 server-side token-commit/flush trace 归因。
- 每个 build 场景保存 5 个独立样本；nearest-rank p95=`max(5 samples)`。cache 状态、sentinel
  before/after SHA、edit/fsync 与 timed build 边界、Cargo/rustc/nvcc/link argv、binary smoke receipt
  完整率 `100%`，未声明 cache/background prebuild 数 `0`。
- baseline validator 能拒绝 dirty 未声明、模型 SHA 不同、跨硬件、少 repeat、错误请求、usage
  缺失、event/usage 或 interval 不一致、coalescing 伪装 eligible、ITL summary 伪造、ineligible ITL
  仍进入 ratio，以及 stale artifact。
- child process inherited `FERRUM_*` 数量 `0`；saved sanitized env 与 OS process receipt 一致率
  `100%`。self-test 在 parent 注入 hostile `FERRUM_*` 后证明 child argv/env/behavior 不受影响，并
  拒绝省略 env、只保存 `NO_COLOR` 或 receipt/env SHA 不一致的 artifact。
- 代码 inventory 覆盖 `crates/` 和 product/release scripts `100%`。
- inventory 中路径、内容 SHA256、symbol allowlist 和分类字段完整率 `100%`；未裁决分类数 `0`。
- historical bug family `15/15` 有 source commit/artifact 和最小 reproducer；concrete case
  evidence 完整率 `M/M`，orphan/重复 case `0`。
- 不修改冻结 legacy product binary 或核心推理行为；benchmark client/runner 的纯证据增强必须
  独立提交并冻结 collector source/tree/binary SHA。任何改变 `run`/`serve` 推理结果的修复都必须
  正式 amendment legacy SHA 后重采全部受影响 baseline。
- 正式 HTTP benchmark argv 必须显式使用 typed `--http-connection-mode fresh`，且
  `BenchReport.env.http_connection_mode=fresh` 必须进入 `env_hash`。external 与 Ferrum 必须使用
  同一模式；禁止用自动重试 POST 掩盖 stale keep-alive、重复推理或吞吐污染。

## 产物

以下为相对 canonical `<g00-out>/` 的逻辑布局；`<g00-out>` 必须位于源码 Git 工作树之外：

```text
<g00-out>/
  manifest.json
  model-resolution.json
  models.lock.json
  hardware/<hardware-id>/
  legacy-binaries.json
  correctness/<model>/<backend>/
  performance/<model>/<backend>/
  external-baselines/
  build-timings/
  coupling-inventory.json
  historical-bug-corpus.json
```

`models.lock.json` 必须逐 lane 引用 `model-resolution.json` 的 SHA，并与解析结果中的 immutable
revision、每个权重/semantic/tokenizer 文件 SHA 和 size 完全一致。hardware fingerprint 必须由
checked-in probe 的规范化字段重算，并引用原始命令输出；build timing 的每个样本必须绑定真实
Cargo argv、输入失效动作、起止时间、return code、日志/Cargo message SHA 和输出 binary SHA。
每个 correctness execution manifest 还必须显式保存 `semantic_source_root`，当 tokenizer 来源独立时
保存 `tokenizer_source_root`；两者逐文件绑定 `models.lock.json` 的 revision/path/SHA。实际
`ferrum run`/`ferrum serve` 生效配置必须输出 `resolution_evidence`，并与同一 semantic/tokenizer
来源逐字段一致。GGUF 权重文件所在目录不得被默认当成 config/template/tokenizer 来源。
M2 Metal Q4_K_S 还必须同时等于 catalog 的 `expected_size_bytes=20673845888` 和
`expected_sha256=ee93ceffed5ce4df8b09bcbaf59a286d531025a1ebde9cf204c74e800c47d57e`；只匹配文件名或 size
不能通过。

### 真实性能采集器

六个 performance lane 统一由 checked-in collector 采集，不允许人工拼装 summary：

```text
python3 scripts/release/runtime_vnext_performance_collector.py \
  --artifact-root <g00-out> \
  --config <lane-config.json> \
  --plan-only

python3 scripts/release/runtime_vnext_performance_collector.py \
  --artifact-root <g00-out> \
  --config <lane-config.json> \
  --resume
```

若先执行 `--plan-only`，首次正式采集必须用原 config 加 `--resume` 消费已冻结计划；若跳过
`--plan-only`，首次正式执行不带 `--resume`。进程中断后只能用原 config 和原输入执行 `--resume`。collection
fingerprint 同时绑定 collector、`models.lock.json`、`legacy-binaries.json`、correctness lane、external
binary 内容和规范化 config。任一输入变化必须新建 lane artifact，不允许复用旧 slot。
external version/revision identity probe 固定 `60s` 超时，禁止在付费机器上无限等待。
external/performance summary 必须引用同一不可变 `plan.json` 和 normalized config；final validator
从原始输入重算 collection fingerprint，不接受只填写一个不可验证的摘要 hash。

每个 lane config 至少包含以下 typed 输入；路径必须是实际绝对路径或可解析本机路径，config 和
child env 中禁止 secret-bearing key 与 `FERRUM_*`：

```json
{
  "schema_version": 1,
  "model_key": "m3-qwen3-30b-a3b",
  "backend": "cuda",
  "model_origin_path": "/models/pinned-snapshot",
  "tokenizer_origin_path": "/models/pinned-tokenizer-snapshot",
  "request_model": "Qwen/Qwen3-30B-A3B",
  "typed_active_cap": 32,
  "memory_budget_bytes": 24696061952,
  "server": {
    "host": "127.0.0.1",
    "port": 18080,
    "ready_timeout_sec": 900,
    "shutdown_timeout_sec": 60,
    "command_timeout_sec": 7200
  },
  "benchmark_client": {
    "binary_path": "/build/e6b55457125487a0ccb3d88dd3f81eb460783fad/ferrum",
    "build_log_path": "/build/e6b55457125487a0ccb3d88dd3f81eb460783fad/build.log",
    "source_git_sha": "e6b55457125487a0ccb3d88dd3f81eb460783fad",
    "cargo_features": ["cuda"]
  },
  "external": {
    "engine": "vllm",
    "binary_path": "/venv/bin/vllm",
    "engine_version": "<exact-version-output>",
    "engine_revision": "<40-hex-source-commit>",
    "server_argv": [
      "/venv/bin/vllm", "serve", "{model_origin_path}",
      "--served-model-name", "{request_model}", "--host", "{host}",
      "--port", "{port}", "--max-num-seqs", "{typed_active_cap}"
    ],
    "version_argv": ["/venv/bin/vllm", "--version"],
    "revision_argv": ["/opt/vllm-source-revision"],
    "active_probe": {
      "format": "prometheus",
      "path": "/metrics",
      "selector": "vllm:num_requests_running"
    }
  },
  "legacy": {
    "extra_serve_argv": [],
    "active_probe": {
      "format": "prometheus",
      "path": "/metrics",
      "selector": "<ferrum-active-request-metric>"
    }
  },
  "datasets": {"sharegpt": "/datasets/pinned-sharegpt.jsonl"},
  "goodput_slo": {"ttft": 500.0, "tpot": 50.0, "e2e": 30000.0}
}
```

Metal 将 external engine 改为 `llama.cpp`，server argv 必须用 `--parallel {typed_active_cap}`，真实对话
数据键改为 `real-chat`。collector 对 comparable lane 执行固定 `A,B,B,A,B,A,A,B`，对 blocked
M1/M2 Metal 只执行 A slots `1,4,6,7`；每个 HTTP cell 都启动独立 resource sampler。cold
`ferrum run` 使用同 PID exec barrier，先建立 sampler 再 exec 产品，避免漏掉模型加载初期资源峰值。
所有 server/run process receipt、sanitized env、identity probe、raw report 和 resource JSONL 都由 final
validator 重算。collector 自身的 PASS 只表示单 lane 证据结构有效，不替代下方 canonical G00 PASS。

内部 validator（只能由 canonical lane 调用）：

```text
scripts/release/runtime_vnext_baseline_gate.py
```

M1/M2 Metal legacy unsupported 只能由真实 product collector 生成，不接受手写 `lane.json`：

```text
python3 scripts/release/runtime_vnext_blocked_lane.py \
  --artifact-root <g00-out> \
  --model-key <m1-qwen35-4b|m2-qwen35-35b-a3b> \
  --model-arg <pinned-gguf> \
  --semantic-source-root <pinned-semantic-snapshot>
```

collector 必须保存并由 final validator 重验 sanitized child env、product PID/PGID/start identity、
bounded resource receipt、memory/swap preflight、effective config 和精确 unsupported failure signature。

G00P 有效执行命令继续使用现有 canonical lane：

```text
python3 scripts/release/run_gate.py vnext-g00 --out <out_dir>
```

必需 PASS line：

```text
FERRUM RUNTIME VNEXT G00 BASELINE PASS: <out_dir>
FERRUM GATE vnext-g00 PASS: <out_dir>
```

G00F 和 G00M 的 canonical lane/manifest 在首次需要对应 milestone PASS 前注册到 `run_gate.py`；
不得为了 S0A mechanical split 先建设 G00P collector。它们的必需 line 为：

```text
python3 scripts/release/run_gate.py vnext-g00f \
  --g00a <fresh-external-g00a-gate-manifest> --out <external-g00f-out>
```

G00F 逐字节引用同一 fresh G00a outer/child artifact，不复制其八个事实文件，也不重新执行一套
平行 collector。G00a 或当前 Git SHA/tree/clean 状态变化时，该引用立即 stale。

```text
FERRUM RUNTIME VNEXT G00F FACTS PASS: <out_dir>
FERRUM GATE vnext-g00f PASS: <out_dir>
FERRUM RUNTIME VNEXT G00M MODEL BASELINE PASS: <out_dir>
FERRUM GATE vnext-g00m PASS: <out_dir>
```

BLOCKED artifact 只允许描述当前不存在的 lane，不等于 G00 PASS。G00 PASS 还要求该 blocker
已被分配给明确下游 Goal，并具有实现和验收路径。
