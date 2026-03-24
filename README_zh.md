# Ferrum Infer

纯 Rust 实现的 LLM 推理引擎。从 Hugging Face 加载模型，本地对话或通过 OpenAI 兼容 API 提供服务。单二进制文件，无需 Python，无运行时依赖。

[English](README.md)

## 快速开始

前置条件：Rust stable 工具链。

访问受限模型（如 Llama 3.2）需先设置 Hugging Face token：
```bash
export HF_TOKEN=hf_your_token_here
```

```bash
# 编译
cargo build --release -p ferrum-cli --bin ferrum

# 下载模型
./target/release/ferrum pull qwen3:0.6b

# 对话
./target/release/ferrum run qwen3:0.6b

# 或启动 API 服务
./target/release/ferrum serve --model qwen3:0.6b --port 8000
```

## 支持的模型

| 别名 | 模型 | 架构 |
|------|------|------|
| `qwen3:0.6b` / `1.7b` / `4b` | Qwen3 | Qwen3 |
| `qwen2.5:0.5b` / `1.5b` / `3b` / `7b` | Qwen2.5-Instruct | Qwen2 |
| `llama3.2:1b` / `3b` | Llama-3.2-Instruct | LLaMA |
| `tinyllama` | TinyLlama-1.1B-Chat | LLaMA |

也可直接使用 Hugging Face 模型 ID：
```bash
./target/release/ferrum run Qwen/Qwen3-0.6B
```

## 命令

| 命令 | 说明 |
|------|------|
| `ferrum run <model>` | 交互式对话 |
| `ferrum serve --model <model>` | 启动 OpenAI 兼容 HTTP 服务 |
| `ferrum stop` | 停止服务 |
| `ferrum pull <model>` | 从 Hugging Face 下载模型 |
| `ferrum list` | 查看已缓存模型 |
| `ferrum embed <model>` | 生成 BERT 向量 |

## API 接口

```bash
# Chat completions（OpenAI 兼容）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"你好"}]}'

# 模型列表
curl http://localhost:8000/v1/models

# 健康检查
curl http://localhost:8000/health
```

## 性能测试

测试环境：**NVIDIA RTX 5090**，Qwen3-4B，FP16，解码吞吐量：

| 版本 | 优化内容 | Tokens/s |
|------|---------|----------|
| 基准 | 原始 candle 算子 | ~33.5 |
| +融合内核 | `candle_nn::RmsNorm` + 融合 RoPE（CUDA） | ~88.3 |
| +预分配 KV 缓存 | O(1) `slice_set` 写入，零拷贝 `narrow` 视图 | ~94.4 |
| +融合投影 | QKV 3→1 矩阵乘，gate+up 2→1 矩阵乘 | ~100+ |

**当前**：Qwen3-4B 平均 ~100 tok/s，峰值超过 103 tok/s。

进行中的 CUDA 内核工作（`ferrum-cuda-kernels` crate）：
- `fused_add_rms_norm` — residual add + RMS Norm 合并为 1 个内核（每层节省 1 次 kernel launch + 1 次显存读写）
- `fused_silu_mul` — SiLU 激活 + 逐元素乘合并为 1 个内核

## 当前状态

**v0.2.0 — 可用的 MVP，尚未达到生产级别。**

已完成：
- CLI 对话和 HTTP 服务（支持流式输出）
- Qwen3、Qwen2/2.5、LLaMA 3.x、TinyLlama 架构
- Metal GPU 加速（macOS）、CUDA（NVIDIA）、CPU 跨平台
- 通过 cudarc/nvrtc 实现的融合 CUDA 内核：RmsNorm、RoPE、QKV 投影、SiLU+mul
- CUDA 上的 FlashAttention-2（KV 缓存零拷贝视图）
- Top-k / Top-p / Temperature / 重复惩罚采样
- Hugging Face 模型下载与缓存管理

进行中：
- Paged Attention——生产级 KV 缓存管理
- 连续批处理——并发请求服务
- CUDA Graph——消除解码循环中的 CPU→GPU 调度开销

## 路线图

1. **Paged KV 缓存** — PagedAttention，支持高并发生产服务
2. **连续批处理** — 迭代级调度与抢占
3. **CUDA Graph** — 消除解码步骤的 CPU→GPU 调度延迟
4. **量化支持** — GPTQ/AWQ/GGUF，让消费级硬件跑更大模型

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 编译选项

```bash
# 仅 CPU（默认）
cargo build --release -p ferrum-cli

# 启用 Metal 加速（macOS）
cargo build --release -p ferrum-cli --features metal

# 启用 CUDA 加速（NVIDIA，需要 CUDA Toolkit）
cargo build --release -p ferrum-cli --features cuda
```

前置条件：Rust stable 工具链。

## 项目结构

```
crates/
├── ferrum-types        # 共享类型定义
├── ferrum-interfaces   # 核心 trait 契约（ComputeBackend, KernelOps, ModelExecutor）
├── ferrum-runtime      # 后端实现（Candle, CPU）
├── ferrum-engine       # Metal 内核、模型编排
├── ferrum-models       # 模型架构（LLaMA, Qwen2, Qwen3, BERT）
├── ferrum-tokenizer    # 分词器
├── ferrum-sampler      # 采样策略
├── ferrum-scheduler    # 请求调度
├── ferrum-kv           # KV 缓存管理
├── ferrum-server       # HTTP API 服务
├── ferrum-cli          # CLI 二进制
└── ferrum-testkit      # 测试工具
```

## 许可证

MIT
