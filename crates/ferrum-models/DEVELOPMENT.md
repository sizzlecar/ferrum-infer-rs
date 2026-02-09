# ferrum-models 开发文档（MVP / HF-first）

本模块提供“框架无关”的模型抽象与配置加载层，优先面向 Hugging Face (HF) 模型，屏蔽后端实现细节（如 Candle/ONNX）。

## 目标
- HF-first：优先支持从 HF 仓库（或本地缓存）解析、加载配置与分词器。
- 稳定抽象：统一模型架构与超参表示，便于后端实现 `ModelBuilder`。
- 可扩展：为 Mistral/GGUF 等格式预留扩展点，逐步完善。

## 模块划分
- `traits.rs`
  - `Architecture`、`AbstractModelConfig`、`ModelBuilder`、`ModelRegistry`、`Tokenizer`、`ModelSourceResolver`、`ModelConverter`。
- `source.rs`
  - HF 风格的 `DefaultModelSourceResolver`：解析模型 ID/本地路径 → `ResolvedModelSource`（本地路径、格式、缓存信息）。
  - 兼容 HF Hub 的缓存目录结构与 token 解析；支持离线模式。
- `config.rs`
  - `ConfigManager`：从文件或 `ResolvedModelSource` 加载配置，当前重点支持 HF 格式；将结果转换为 `AbstractModelConfig`。
- `tokenizer.rs`
  - `TokenizerWrapper`（基于 `tokenizers`）、`TokenizerFactory`（带缓存与模式）、`CachedTokenizer`。
- `registry.rs`
  - `DefaultModelRegistry`：注册 `ModelBuilder`，管理模型别名与本地发现（扫描目录校验基本文件）。

## HF-first 实现要点
1. 解析
   - 通过 `DefaultModelSourceResolver.resolve(id_or_path, revision)`：
     - 本地路径：直接检测格式（优先 `config.json` → HF）。
     - HF ID：先查找本地缓存（遵循 HF Hub 缓存结构），未命中则下载（依赖 `hf-hub`，可携带 token）。
2. 配置
   - `ConfigManager.load_from_source(source)`：
     - HF：读取 `config.json`，做兼容性处理（RoPE/字段映射等，当前为占位，后续补齐），转换为 `AbstractModelConfig`。
     - 统一填充：激活函数、Norm 类型、注意力配置、KV heads、额外参数（原始 JSON 透传到 `extra_params`）。
3. 分词器
   - `TokenizerFactory.create_from_source(source, mode)`：
     - HF：从 `tokenizer.json` 加载；支持多模式（Auto/Mistral/Custom 占位）。
4. 构建与权重
   - 在后端 crate 注册 `ModelBuilder`（如 Candle 版 Llama）。
   - 通过 `get_builder_arc(&config.architecture)` 获取 builder，执行 `build` 与 `load_weights`。

## 配置分层与 Builder 参数（重要）
为避免“配置”概念混淆，明确两层含义：

- Definition 层（架构/超参）
  - 对应：`AbstractModelConfig`（建议可改名：`ModelDefinition`/`ArchitectureConfig`）。
  - 含义：HF 模型固有属性（隐藏维度、层数、vocab、RoPE、Norm、激活、注意力结构等）。
  - 特性：随权重/架构而变，和设备/并行/量化无关。

- Runtime 层（运行/部署）
  - 对应：`ferrum_core::ModelConfig`（建议可改名：`RuntimeConfig`/`ExecutionConfig`）。
  - 含义：device、dtype、并行度、batch 上限、运行时 `max_sequence_length`、量化、模型加载路径等。
  - 特性：同一模型在不同硬件/策略下可不同，但不改变网络结构。

建议与约束：
- 移除 Runtime 中与 Definition 重叠的语义：
  - `ModelConfig.model_type` 建议移除，统一使用 `AbstractModelConfig.architecture`。
- `max_position_embeddings`（Definition）与 `max_sequence_length`（Runtime）并存：
  - 运行期实际长度 = min(请求、Runtime 限制、Definition 上限)。
- 注意力“实现特性”归到 Runtime：
  - 如 `use_flash_attention`/`use_paged_attention` 归类到运行能力（可与 `ExecutorCapabilities` 协同）。
- Builder 接口建议：
  - 选项 A：保留现状，但将 `load_weights` 的 `weights_path` 换成 `ResolvedModelSource`，减少胶水代码。
  - 选项 B：引入 `BuildContext` 统一传参：
    ```rust
    pub struct BuildContext<'a> {
        pub definition: &'a AbstractModelConfig,     // Definition 层
        pub runtime: &'a ferrum_core::ModelConfig,   // Runtime 层
        pub source: &'a crate::source::ResolvedModelSource,
        pub tokenizer: Option<std::sync::Arc<dyn crate::Tokenizer>>, // 可选
    }

    #[async_trait::async_trait]
    pub trait ModelBuilder {
        async fn build(&self, ctx: &BuildContext<'_>) -> ferrum_core::Result<Box<dyn ferrum_core::Model>>;
    }
    ```


## 近期改造计划（Roadmap）
- 必做（HF-first）：
  - 实现 `apply_hf_compatibility_patches`：
    - RoPE 参数归一化（theta/scaling 类型/命名差异）。
    - KV heads 缺省推断（部分 HF 配置缺失时从 heads/多路并行策略推断）。
    - 激活/Norm 字段别名映射与默认值策略固化。
  - 抽出统一 `detect_format(path)` 工具函数（避免在 `source.rs`/`config.rs`/`registry.rs` 三处重复）。
  - `ModelRegistry`：面向实际使用的 `get_builder_arc` 为主，考虑弃用返回引用的 `get_builder`。
- 可选（增强）：
  - `ModelConverter`：落地 HF/GGUF/SafeTensors → `AbstractModelConfig` 的转换器，实现集中化转换逻辑；`ConfigManager` 仅编排与缓存。
  - GGUF：从文件元数据读取超参（替代当前文件名启发）。
  - Mistral：解析 `params.json` 的真实字段。
  - 权重文件解析策略：在 `ResolvedModelSource` 扩展 `weight_files` 或解析方法，供后端直接使用。

## 典型流程
```rust
let resolver = DefaultModelSourceResolver::with_defaults();
let source = resolver.resolve("meta-llama/Llama-2-7b-hf", Some("main")).await?;

let mut cfg_mgr = ConfigManager::new();
let abstract_cfg = cfg_mgr.load_from_source(&source).await?;

let tokenizer = TokenizerFactory::new()
    .create_from_source(&source, None)
    .await?;

// 从注册表获取后端构建器并实例化模型
// let builder = registry.get_builder_arc(&abstract_cfg.architecture).unwrap();
// let mut model = builder.build(&abstract_cfg, &ferrum_core::ModelConfig::default()).await?;
// builder.load_weights(&mut *model, source.local_path.to_str().unwrap()).await?;
```

## 测试策略
- `config.rs`
  - HF 配置加载基础用例（必测）：
    - 解析 `architectures`/`model_type` → `Architecture`。
    - Norm/Activation 推断与默认值。
    - 缺省字段回退策略。
- `source.rs`
  - 本地路径解析；缓存命中（构造本地缓存目录 + `config.json`）。
  - 格式检测优先级（同时存在 `config.json`/`params.json` 时选择 HF）。
- `registry.rs`
  - 别名解析；目录扫描发现模型的最小完备条件（`config.json`、`tokenizer.json`、权重文件之一）。

## 运行
- 运行测试：
```bash
cargo test -p ferrum-models
```

## 约定与错误处理
- 离线模式：若设置了 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`，解析器仅使用本地缓存；下载会失败并给出修复建议。
- 错误分类：配置解析错误归类为 `Error::configuration`；下载/IO 为 `Error::internal`；无效请求为 `Error::invalid_request`。

---
如需新增后端（例如 Candle）：在后端 crate 实现并注册对应架构的 `ModelBuilder`，确保仅依赖 `AbstractModelConfig`，不要直接解析 HF 配置。