## Ferrum Inference 重构提案（vLLM 借鉴 + Rust 优势）

本文件提出面向“超过 vLLM、成为 LLM 推理事实标准”的架构与核心接口（trait）重构方案，结合 Rust 的性能与安全优势，解决当前 MVP 阶段的分层与职责边界问题，为后续高性能与多后端扩展奠定基础。

### 目标
- 超越 vLLM 的吞吐、延迟稳定性与可扩展性，并提供更强的类型安全与可维护性。
- 建立清晰稳固的分层与接口：可插拔后端、统一采样与分词、可演化的调度与 KV-Cache 体系。
- 立即重构：允许不兼容变更（Breaking Changes），以最短路径完成架构升级（不考虑最小破坏迁移）。

---

### 现状与主要问题（摘自当前代码）
- 抽象重叠/命名冲突：
  - `Scheduler` 在 `ferrum-core` 与 `ferrum-scheduler` 各有定义，语义与方法不同，易混淆。
  - `MemoryManager` 在 `core` 与 `runtime` 均存在但职责不同（KV cache vs 设备内存）。
- 职责耦合：
  - `Model` 同时承担分词（encode/decode）与推理，限制 tokenizer 复用与多模态拓展。
  - `Backend::load_weights -> Box<dyn Model>` 让后端“拥有”模型构造与分词/采样决策，提升耦合度。
  - 采样逻辑位于 Candle 模型实现中，无法被不同后端/模型共享优化。
- 表示层问题：
  - `KVCache` 在 core 以 `Vec<Tensor>` 表示，与后端内部 cache 结构不一致，导致难以零拷贝/句柄复用。
  - `Tensor` 为简化结构（`Vec<f32>`），与真实后端张量割裂，难承载零拷贝与设备信息。
- 扩展性与一致性：
  - 能力发现（capabilities）维度有限，调度/批处理/注意力实现难以基于能力自适应。
  - 流式增量解码逻辑分散于引擎与模型，增量 decode 不够健壮、难统一优化。

---

### 借鉴 vLLM 的关键设计
- 清晰的执行通道拆分：请求层（Admission/Queue）→ 连续批处理调度 → KV-Cache 分配与复用 → 模型执行（prefill/decode）→ 采样与后处理 → 流式输出。
- 明确的 KV-Cache 管理：块表（block table）、页化（paged）KV、前缀复用与换入换出策略明确分层。
- 可插拔的采样链与 logits 处理链（processors）以及并行采样能力。
- 广覆盖的模型执行适配层（model executor），将上层策略与后端实现解耦。

---

### 设计原则（Rust-first）
- 单一职责与稳定边界：接口只表达“需要/保证”的最小集合，避免跨层泄漏实现细节。
- 零拷贝与句柄语义：以 `Handle/Ref` 表示张量与 KV-Cache；核心路径避开多余拷贝与动态分配。
- 能力发现驱动：后端能力 → 调度/批处理/注意力/采样策略自适应。
- 性能优先的 API 形态：热点路径尽量使用具体类型与 `impl Trait` 返回；可插拔处用 `dyn Trait`。
- 可测试与可观测：关键组件可替换 mock；统一指标、trace 与 profile 接口。

---

### 目标架构与 crate 边界（按依赖关系重排）

已完成（稳定基座）：
- `ferrum-types`（值类型、协议对象）
  - 公共 ID、请求/响应、采样参数、错误、设备/数据类型、模型元信息、配置与指标。
  - 无实现依赖，所有 crate 只读依赖它。
- `ferrum-interfaces`（稳定接口）
  - 面向上层的稳定 trait：`InferenceEngine`、`Scheduler`、`Tokenizer`、`Sampler/LogitsProcessor`、`ModelExecutor`、`KvCache*`、`TensorLike/TensorFactory/TensorOps`、`ComputeBackend/WeightLoader`、`DeviceMemoryManager`。
  - 仅依赖 `ferrum-types`。

依赖图（A → B 表示 A 依赖 B）：
- `ferrum-engine` → `ferrum-interfaces`, `ferrum-types`, `ferrum-scheduler`, `ferrum-kv`, `ferrum-runtime`, `ferrum-tokenizer`, `ferrum-sampler`, `ferrum-models`
- `ferrum-runtime` → `ferrum-interfaces`, `ferrum-types`
- `ferrum-kv` → `ferrum-interfaces`, `ferrum-types`（可选依赖 `ferrum-runtime` 的内存/流接口）
- `ferrum-scheduler` → `ferrum-interfaces`, `ferrum-types`
- `ferrum-models` → `ferrum-interfaces`, `ferrum-types`
- `ferrum-tokenizer` → `ferrum-interfaces`, `ferrum-types`
- `ferrum-sampler` → `ferrum-interfaces`, `ferrum-types`
- `ferrum-server` → `ferrum-engine`, `ferrum-types`
- `ferrum-cli` → `ferrum-engine`, `ferrum-types`
- （可选过渡）`ferrum-core` → 仅作为门面 re-export `ferrum-interfaces`/`ferrum-types`，不再存放定义

模块实现细节（建议的公共 API 与内部结构）

- `ferrum-engine`（编排层，强流式）
  - 责任：请求接入→调度→（KV 分配）→ prefill→采样→decode 循环→流式输出；指标/trace。
  - 关键组件：
    - Pipeline：`prefill_batch()`、`decode_step()`；TTFT 优化；并行采样可选。
    - SamplerChain：接入 `ferrum-sampler`；支持温度/top-k/top-p/惩罚与自定义处理。
    - TokenizerManager：`encode() / decode_incremental()`，前缀共享，缓存。
    - KvCoordinator：与 `ferrum-kv` 对接，管理 `KvCacheHandle` 生命周期与 block_table 更新。
    - SchedulerAdapter：对接 `ferrum-scheduler`，拉取/回填请求与状态。
  - 公共 API：实现 `InferenceEngine`；导出构造器（`Engine::new(config, deps...)`）与 `EngineFactory`。
  - 指标：TTFT、inter-token、tokens/s、batch 利用率、KV 命中、采样/调度/内存阶段耗时。

- `ferrum-runtime`（设备运行时/后端）
  - 责任：`TensorFactory/TensorOps`、`KernelExecutor`、`DeviceMemoryManager`、`StreamManager` 的具体实现；后端能力发现。
  - 后端：
    - Candle：首个实现；提供张量创建/搬运，矩阵算子与 softmax 等基本算子；合理的内存池与对齐。
    - Metal：提供 `ComputeBackend` 骨架与 Command Buffer 流；优先实现 attention 路径所需基础算子。
  - 公共 API：`ComputeBackend`、`WeightLoader`（权重 mmap/加载到设备缓冲）。

- `ferrum-kv`（KV-Cache 子系统）
  - 责任：PagedAttention 块池（GPU/CPU 两级）、BlockTable 映射（逻辑→物理）、SwapManager、PrefixCache、EvictionPolicy。
  - 结构：
    - `BlockPool`（GPU/CPU）：固定大小、对齐、引用计数、统计。
    - `KvCacheHandle`：持有 block_table 与设备位置；不可变视图与可变更新接口。
    - `KvCacheManager`：`allocate/resize/deallocate/get_stats`；`can_allocate/suggest_eviction`。
    - 可选：`Compression`（int4/fp8）在线压缩策略，限 MVP 可跳过实现细节。
  - 指标：块利用率、前缀命中、swap 次数/时延、压缩开销。

- `ferrum-scheduler`（调度）
  - 责任：Admission/排队/批构建；支持优先级；可扩展抢占/SLA/公平性策略。
  - MVP：提供 FIFO/优先级两个实现；`next_batch(hint)` 返回兼容 batch 计划；暴露队列指标。
  - 对外：实现 `Scheduler` 与 `RequestQueue`；提供策略枚举与配置结构。

- `ferrum-models`（模型构建）
  - 责任：`ModelDefinition`、`ModelBuilder`；将解析到的结构映射为后端算子/权重；`WeightLoader` 配合加载 safetensors/GGUF。
  - Registry：架构到 builder 的映射；来源解析（HF、本地、URL、S3）。
  - 对外：`ModelBuilder::build(prefill/decode)` 返回 `ModelExecutor` 的具体实现实例。

- `ferrum-tokenizer`（分词）
  - 责任：统一 `Tokenizer` 与工厂；`decode_incremental(prev, next)`；缓存与多模型共享；chat template。
  - 对外：`TokenizerFactory::create_from_source(...)`；能力信息（special tokens、vocab）。

- `ferrum-sampler`（采样/处理）
  - 责任：`LogitsProcessorChain` 与 `Sampler`；常用处理器与 GREEDY/MULTINOMIAL；并行采样接口预留。
  - 对外：`SamplingConfig::from_params`；统计信息与微基准挂钩。

- `ferrum-server`（OpenAI 兼容服务）
  - 责任：HTTP/WS/SSE；流式输出；中间件（鉴权/限流/日志/健康检查）。
  - 对外：与 `InferenceEngine` 对接；OpenAI Chat/Completions 兼容结构。

- `ferrum-cli`（命令行）
  - 责任：启动引擎、推理、serve、benchmark、模型维护；输出人类友好的统计与 JSON 报告。

注意：`ferrum-core`（可选过渡门面）
- 若仍然存在历史依赖，可将其改为只 re-export `ferrum-types`/`ferrum-interfaces`，并用别名承接旧符号（如 `Tensor` → `TensorRef`，`RuntimeConfig` → `ModelConfig`），待全仓迁移完毕后下线。

---

### 核心 trait 重设计（示例草案）

以下为关键接口的“形状”，用于约束分层；实际字段可在实现中扩展。

#### Tensor 抽象（零拷贝/设备感知）
```rust
pub trait TensorLike: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DataType;
    fn device(&self) -> Device;
}

pub type TensorRef = std::sync::Arc<dyn TensorLike>;
```

#### Tokenizer（从 `Model` 中剥离，支持增量解码）
```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String>;
    // 依据 prev tokens 与新 token，返回追加文本，避免全量 decode
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn special_tokens(&self) -> &SpecialTokens;
}
```

#### LogitsProcessor / Sampler（与后端解耦，可并行/可复用）
```rust
pub struct SamplingContext<'a> {
    pub step: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logits: &'a mut [f32],
}

pub trait LogitsProcessor: Send + Sync {
    fn process(&self, ctx: &mut SamplingContext);
}

pub trait Sampler: Send + Sync {
    fn sample(&self, logits: &[f32], rng: &mut dyn rand::RngCore) -> TokenId;
}
```

#### KV-Cache（句柄与块表抽象）
```rust
pub struct BlockTable {
    pub physical: smallvec::SmallVec<[u32; 8]>,
    pub logical_to_physical: smallvec::SmallVec<[u32; 8]>,
    pub seq_len: usize,
}

pub trait KvCacheHandle: Send + Sync {
    fn block_table(&self) -> &BlockTable;
    fn device(&self) -> Device;
    fn num_tokens(&self) -> usize;
}

pub trait KvCacheManager: Send + Sync {
    fn allocate(&self, req: &AllocationRequest) -> Result<BlockTable>;
    fn resize(&self, request: RequestId, new_size: usize) -> Result<BlockTable>;
    fn deallocate(&self, request: RequestId) -> Result<()>;
    fn can_allocate(&self, num_blocks: usize) -> bool;
    fn stats(&self) -> CacheStats;
}
```

#### ModelExecutor（替代“胖”Model；prefill/decode 明确）
```rust
pub struct PrefillInput {
    pub input_ids: TensorRef,   // [B, T]
}

pub struct PrefillOutput {
    pub logits: TensorRef,      // [B, T, V]
    pub kv: std::sync::Arc<dyn KvCacheHandle>,
}

pub struct DecodeInput {
    pub input_ids: TensorRef,   // [B, 1]
    pub kv: std::sync::Arc<dyn KvCacheHandle>,
}

pub struct DecodeOutput {
    pub logits: TensorRef,      // [B, V]
    pub kv: std::sync::Arc<dyn KvCacheHandle>,
}

pub trait ModelExecutor: Send + Sync {
    fn info(&self) -> &ModelInfo;
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput>;
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput>;
    // 可选：保留完整前向，满足非自回归用途
    async fn forward(&self, input: &TensorRef) -> Result<TensorRef> { unimplemented!() }
}
```

#### Backend 拆分（计算后端 vs 权重加载）
```rust
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;
    fn tensor_ops(&self) -> &dyn TensorOps;
    fn kernels(&self) -> &dyn KernelExecutor;
    fn memory(&self) -> &dyn DeviceMemoryManager;
}

pub trait WeightLoader: Send + Sync {
    async fn load_tensor(&self, spec: &TensorSpec, dst: &mut DeviceBuffer) -> Result<()>;
}
```

> 注：模型构建由 `ModelBuilder` 承担，调用 `ComputeBackend` 的原语与 `WeightLoader` 进行权重映射；不再由 `Backend::load_weights` 直接返回 `Model`。

#### 调度与队列（统一到 `ferrum-scheduler`）
```rust
pub trait RequestQueue: Send + Sync {
    async fn enqueue(&self, req: InferenceRequest) -> Result<()>;
    async fn dequeue(&self) -> Result<Option<InferenceRequest>>;
    fn len(&self) -> usize;
}

pub struct BatchHint { pub max_batch: usize, pub max_tokens: usize }

pub struct BatchPlan {
    pub batch_id: BatchId,
    pub requests: Vec<InferenceRequest>,
}

pub trait Scheduler: Send + Sync {
    async fn submit(&self, req: InferenceRequest) -> Result<RequestId>;
    async fn next_batch(&self, hint: BatchHint) -> Option<BatchPlan>;
    async fn complete(&self, req: RequestId, resp: &InferenceResponse) -> Result<()>;
    fn metrics(&self) -> SchedulerMetrics;
}
```

#### 引擎（统一入口、强流式）
```rust
pub trait InferenceEngine: Send + Sync {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse>;
    async fn infer_stream(
        &self,
        req: InferenceRequest,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send + Unpin>>;
    async fn status(&self) -> EngineStatus;
    async fn shutdown(&self) -> Result<()>;
}
```

---

### 数据流与执行路径（对齐 vLLM、发挥 Rust 优势）
1. Admission/Validation：请求验证、优先级与速率控制。
2. Tokenize：使用 `Tokenizer`（支持增量解码）；Prompt 模板化处理。
3. Schedule：`Scheduler` 生成连续批处理 `BatchPlan`。
4. KV 分配：`KvCacheManager.allocate`；前缀复用（可选）。
5. Prefill：`ModelExecutor.prefill` 生成 logits 与初始 KV 句柄。
6. 采样：通用 `LogitsProcessor` 链 + `Sampler`（支持并行采样策略）。
7. Decode 循环：以批为单位步进；重用 KV 句柄；流式增量文本通过 `decode_incremental` 生成输出片段。
8. 完成与回收：`Scheduler.complete`，KV 回收或缓存策略更新。

---

### 与 vLLM 的差异化与超越点
- **类型安全&可组合**：用 trait 将模型执行/采样/分词/缓存完全解耦，组合灵活、编译期校验。
- **零拷贝句柄**：`TensorLike` 与 `KvCacheHandle` 作为统一句柄，避免跨层复制与泄漏。
- **能力发现驱动策略**：调度/注意力/并发度根据 `BackendCapabilities` 自适应，易扩展新硬件。
- **并行采样/高效增量解码**：标准化 `LogitsProcessor` + `Sampler`，让 speculative/多样本在引擎层通用实现。
- **可观测性**：统一 metrics/logging/tracing 钩子，便于线下优化与线上 SLO 保障。

---

### 对 vLLM 的现实不足与 Ferrum 差异化机会
- **性能/运行时开销**：vLLM 受 Python/Ray 边界、GIL、序列化/IPC 成本影响。
  - Ferrum：Rust 单进程执行器、无锁/低锁 MPMC 队列、零拷贝数据面、CUDA Graph/Metal Command Buffer 录制、预分配内存池。
- **KV-Cache 体系**：vLLM 页化 KV 强大，但策略受限于 Python 层 glue。
  - Ferrum：`KvCacheHandle` 句柄化、二级/多级内存（HBM/主存/磁盘）、热度驱动逐出、前缀合并、异步换入换出、KV 压缩（int4/FP8 可选）。
- **采样与 logits 处理**：Python 侧在高 QPS 下可能成为瓶颈。
  - Ferrum：SIMD/向量化 `LogitsProcessor` 链，统一并行采样（speculative/多样本/beam）。
- **编解码与增量 detokenize**：高并发下 CPU 开销较大，增量策略不总最优。
  - Ferrum：Rust tokenizers + LRU 与前缀共享，标准化 `decode_incremental`，仅输出追加片段。
- **调度（SLO/多租户）**：vLLM 连续批处理成熟，但资源感知与强 SLA 可进一步系统化。
  - Ferrum：资源感知+SLO 驱动调度（Admission/Preemption/Reserve），能力自适应批策略，公平性/优先级更强。
- **可观测性与调优闭环**：细粒度阶段指标与内核级 tracing 可更紧密。
  - Ferrum：全链路 tracing（tokenize/prefill/decode/sampling/KV）、内核 profile 对齐、在线自调优。
- **后端与跨硬件**：vLLM 以 CUDA/ROCm 为主，Metal/CPU 非核心。
  - Ferrum：Metal 一等公民（自研 Metal kernels）、CPU（AVX512/AMX）优化，统一 `ComputeBackend` 能力层。
- **确定性与可重复**：Python 并发与 RNG 更易非确定性。
  - Ferrum：可控 RNG（PCG/Philox）与端到端确定性模式，利于回归评测。
- **部署体验**：vLLM 依赖链重，镜像大，冷启动慢。
  - Ferrum：静态二进制、瘦镜像、极快冷启动、离线友好。
- **API 稳定性与扩展机制**：Python API 演进快、定制点分散。
  - Ferrum：稳定 trait/ABI（采样/调度/KV/后端/模型构建/插件），语义版本化。
- **内存规划与碎片**：动态分配与多运行时导致碎片风险。
  - Ferrum：统一内存池、终身期规划、批内复用、全局预算器、OOM 优雅降级。
- **结构化输出与语法约束**：以 Python glue 为主。
  - Ferrum：在采样链中零开销融合语法约束（DFA/regex/PEG/JSON schema masking）。

---

### Ferrum 优先发力方向（可衡量的 KPI）
- **性能**：
  - P50/P99 单 token 解码延迟较 vLLM 降低 15–30%；小 batch 低延迟模式尤显著。
  - 吞吐（等精度/等 batch）提升 10–20%。
  - 峰值显存降低 10–25%（KV 分层/压缩/复用）。
- **能力与体验**：
  - Metal 后端在 M 系列设备上领先 vLLM；CPU 路径稳定可用。
  - 稳定 API：Tokenizer/采样/调度/KV/后端/模型构建 trait 固化，并附最小插件示例。
  - 可观测：内置 tracing/metrics，暴露可操作的策略开关与面板。
- **架构落地**：
  - 句柄化 `TensorLike`/`KvCacheHandle`，消除跨层拷贝。
  - 统一 `Scheduler` + 资源感知与预占/SLA；采样/LogitsProcessor 上移，引擎解码循环向批处理优化。
  - CUDA Graph/Metal pipeline 录制与工作区复用。
- **生态与维护**：
  - 发布静态二进制与瘦容器；semver 与扩展点文档完善。

---

### 评测与对比清单（Evaluator）
- **延迟**：单 token（prefill=0/1）P50/P95/P99；流式 chunk 间隔抖动（jitter）。
- **吞吐**：不同序列长度分布与并发下的 tokens/s；混合长度效率。
- **KV 指标**：块利用率、前缀命中率、swap 频率与时延、压缩开销。
- **采样/处理**：每步 logits 处理与采样耗时；并行采样效率。
- **Tokenizer**：编码/增量解码耗时与 CPU 占用；缓存命中。
- **内存**：峰值显存/主存、碎片率、分配次数；OOM 降级路径验证。
- **启动**：冷/热启动时延、预热时间。
- **后端覆盖**：CUDA/ROCm/Metal/CPU 各维用例通过率与性能曲线。

输出形式建议：`benchmarks/` 脚本 + `docs/benchmarks.md` 指标说明 + 统一 JSON 报告（便于 CI 对比）。

---

### 验收指标汇总（对齐里程碑）
- 解码延迟：P50/P99 较 vLLM 至少 -15%/-20%。
- 吞吐：相同硬件 + 精度 + batch 下 +10–20%。
- 显存：峰值 -10–25%；KV 分层开启下依旧达到同等吞吐。
- 启动：小模型冷启动 < 1s、热启动 < 200ms（目标）。
- 流式：chunk 间隔 P99 抖动 < 20ms（目标）。
- API：稳定 trait 集 + 插件样例（采样/调度/KV/后端/模型构建）。

### 实施计划（MVP 阶段立刻执行）
- 强制重构接口与分层（Breaking）：
  - 删除 `core::Model.encode/decode/generate_next_token`，以 `Tokenizer`+`ModelExecutor`+`Sampler` 取代。
  - 删除 `core::Backend::load_weights -> Model`，以 `ModelBuilder`+`WeightLoader`+`ComputeBackend` 构建执行体。
  - 统一 `Scheduler` 接口到 `ferrum-scheduler`，移除 `ferrum-core` 中重叠定义。
  - 将 `core::KVCache(Vec<Tensor>)` 改为 `KvCacheHandle` 句柄，不再在 core 揭示张量细节。
  - 将 `core::MemoryManager`（KV向）重命名/迁移为 `kv::KvCacheManager`；`runtime::MemoryManager` 更名为 `DeviceMemoryManager`。
- 立即引入通用采样链：
  - 标准化 `LogitsProcessor` 与 `Sampler`，把温度/top-k/top-p/惩罚等从后端模型移到引擎层。
- 立即抽象 `TensorLike/TensorRef`：
  - 核心路径以句柄传递，Candle/Metal 后端提供具体实现，避免 `Vec<f32>` 来回拷贝。
- 引擎重写解码循环：
  - 按批执行 prefill→decode，复用 KV 句柄，流式输出基于 `decode_incremental`。
- 能力驱动：
  - 扩展 `BackendCapabilities`，让调度与注意力实现根据能力自适应选择路径。

---

### 性能与测试策略
- 微基准（criterion）：采样链、增量解码、KV 分配与合并、prefill/decode 单步。
- 端到端基准：吞吐/延迟/P99/P999；多请求并发与混合长度分布；显存/CPU 占用曲线。
- 回归套件：各后端与主流架构（Llama/Mistral/Qwen/...）能正确工作；OpenAI API 兼容测试。
- Profiling：`tracing` + 后端原生 profiler（CUDA/Metal）；火焰图与内存轨迹。

---

### 里程碑与验收（Aggressive）
- W1：完成 trait 强制重构与 crate 边界重排；引擎接入新 `Tokenizer/Sampler` 路径跑通流式。
- W2：KV 句柄化与调度统一；prefill/decode 批处理闭环；基础指标与端到端基准跑通。
- W3+：Backend 拆分/能力自适应优化；持续扩展注意力/并行采样/多后端；完善可观测性。

---

### 附录：旧→新接口映射（草案）
- `core::Model.encode/decode` → `tokenizer::Tokenizer`（Engine 组合调用）
- `core::Model.generate_next_token` → `engine::decode_loop` + `sampler::Sampler` + `model_executor::decode`
- `core::KVCache(Vec<Tensor>)` → `kv::KvCacheHandle`（后端自定义内部结构）
- `core::Scheduler`（旧） → `scheduler::Scheduler`（新），保留 `submit/next_batch/complete` 统一语义
- `core::MemoryManager`（KV 向） → `kv::KvCacheManager`
- `runtime::MemoryManager`（设备向） → `runtime::DeviceMemoryManager`
- `core::Backend.load_weights -> Model` → `models::ModelBuilder + runtime::ComputeBackend + models::WeightLoader`

---

本提案采用强制重构策略：立即抽离 Tokenizer 与 Sampler、句柄化 KV、统一调度与后端职责拆分，并以能力驱动优化热路径。目标是在 MVP 阶段尽快形成高性能、可组合、类型安全的核心，奠定超越 vLLM 的事实标准基础。

---