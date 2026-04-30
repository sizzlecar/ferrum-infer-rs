# Group A 性能报告 — Apple Silicon (M1 Max 32 GB)

**日期**: 2026-04-30 · **GGUF Q4_K_M 同一文件三引擎共用** · **Phase 1**（单请求路径）

> Phase 1 完成：单请求 pp50 / pp512 / tg128 + TTFT 估算。Phase 2（16 并发吞吐）待做。

## 摘要

| | Qwen3-8B Q4_K_M (4.7 GiB) | Llama-3.1-8B Q4_K_M (4.6 GiB) | Qwen3-30B-A3B Q4_K_M (17.3 GiB) |
|---|---:|---:|---:|
| **ferrum tg128** | 28.0 | 29.4 | 29.2 |
| **mistral.rs tg128** | 37.7 | **39.9** | N/A (不支持 MoE GGUF) |
| **llama.cpp tg128** | 33.7 | 33.3 | **54.6** |
| **decode 排序** | mistral > llama > ferrum | mistral > llama > ferrum | llama > ferrum |

**一句话总结**：在 M1 Max 上，ferrum 的 prefill 已经追到 llama.cpp **96%**（30B-A3B pp512），decode 还差 **47%**（30B-A3B tg128）；这是接下来 §5.4 的攻击目标。

## 单请求吞吐 (tok/s, 3 trials 中位数)

| 模型 | 指标 | ferrum | mistral.rs | llama.cpp |
|---|---|---:|---:|---:|
| Qwen3-8B | pp50 | 186.3 | 212.8 | **258.0** |
| Qwen3-8B | pp512 | 314.8 | 293.3 | **376.1** |
| Qwen3-8B | tg128 | 28.0 | **37.7** | 33.7 |
| Llama-3.1-8B | pp50 | 185.5 | 221.2 | **258.1** |
| Llama-3.1-8B | pp512 | 316.8 | 313.4 | **379.5** |
| Llama-3.1-8B | tg128 | 29.4 | **39.9** | 33.3 |
| Qwen3-30B-A3B | pp50 | 123.4 | — | **202.3** |
| Qwen3-30B-A3B | pp512 | 596.7 | — | **620.7** |
| Qwen3-30B-A3B | tg128 | 29.2 | — | **54.6** |

## TTFT 估算 (50-token prompt, ms)

> TTFT(50) = 50 / pp50 + 1 / tg128 (ms)（prefill 完后 + 1 个 decode tick）

| 模型 | ferrum | mistral.rs | llama.cpp |
|---|---:|---:|---:|
| Qwen3-8B | 304.1 | 261.5 | **223.5** |
| Llama-3.1-8B | 303.6 | 251.1 | **223.7** |
| Qwen3-30B-A3B | 439.4 | — | **265.5** |

## 重点观察

1. **ferrum 在 30B-A3B 上的 prefill 几乎追平 llama.cpp**（596.7 vs 620.7 = 96%）— 这是 PR #54/#55/#56 的累计战果（Q-tiled flash_attn + GPU MoE topk + indirect dispatch）。
2. **ferrum 的 decode 速率对模型规模/类型几乎无关**（28 / 29.4 / 29.2）。这反映出 decode 路径的 per-token GPU 工作量被 dispatch / sync / GEMV 效率限制，没有跟着「active 参数变少」加速。
3. **llama.cpp 的 MoE decode 比 dense 还快**（54.6 > 33.3）— Qwen3-30B-A3B 只激活 3B 参数，比 8B dense 实际少。llama.cpp 享受到这个红利，ferrum 没有。
4. **ferrum pp50 在 30B-A3B 上偏弱**（123 vs 202）— 50 token 还摊不开 ferrum 的启动开销（kernel cache miss、buffer alloc 等）。pp512 时 overhead 摊销，差距收敛到 96%。
5. **mistral.rs 在 dense 8B 模型 decode 最快**（37-40 t/s），但其 GGUF loader 启动 Qwen3-MoE 时 dummy run 崩溃（"Engine dead, rebooting"）— 当前 v0.8.0 对 qwen3moe arch 支持有限。

## 内存带宽分析（解释为什么 decode 是大坑）

M1 Max 实测有效内存带宽 ≈ 150 GB/s（峰值 200 GB/s 的 75%）。每个 decode token 必须把所有「active 权重」从 RAM 读到 GPU SRAM 一次。

| 模型 | active params | 单 token 读取量 (Q4) | llama.cpp BW 利用 | ferrum BW 利用 |
|---|---:|---:|---:|---:|
| Qwen3-8B (dense) | 8 B | ~4 GB | 33×4 = **132 GB/s (88%)** | 28×4 = **112 GB/s (75%)** |
| Llama-3.1-8B (dense) | 8 B | ~4 GB | 33×4 = **132 GB/s (88%)** | 29×4 = **117 GB/s (78%)** |
| Qwen3-30B-A3B (MoE) | 3 B active | ~1.5 GB | 54×1.5 = **82 GB/s (55%)** | 29×1.5 = **44 GB/s (29%)** |

**Dense 模型上 ferrum 已经达到 75-78% BW 利用率**（vs llama.cpp 88%），差距小（gap ~13%）。

**MoE 模型上 ferrum 只有 29% BW 利用率**（vs llama.cpp 55%），差 ~2× —— ferrum 的 MoE decode 路径没把「active 参数变少」的红利吃下来，每 token 仍走 dense-shaped GEMV，做了大量本不必要的工作或读了不必要的内存。这就是 §5.4 要攻的方向。

## 方法论

- **硬件**: MacBook Pro M1 Max, 32 GB unified memory, macOS 15.1.1 (Build 24B91), Apple7 GPU family (Metal 3，**不支持 Metal 4 tensor matmul**)
- **后端**: Metal (所有引擎，统一 GPU 路径)
- **量化**: Q4_K_M（同一份 GGUF 文件三引擎共用，规避 quantization 实现差异）
- **测试串行**: 每次只有一个推理进程在跑（Python 子进程退出 = 干净内存释放，避免内存竞争干扰数字）
- **三个 workload**:
  - **pp50** = 50-token "the the the…" 提示 + 1 token 解码，测 prefill 速率（tok/s）
  - **pp512** = 512-token "the" 重复提示 + 1 token 解码（tok/s）
  - **tg128** = "Once upon a time" 短提示 + 128 tokens 解码，测 decode 速率（tok/s）
- **采样**: `temperature=0.0`（确定性输出），`max_tokens=N` 硬截断
- **每条 3 trials 取中位数**（llama-bench 内部 3 次平均；ferrum/mistralrs 本脚本各跑 3 次）
- **TTFT(50) 估算**: `1000 × (50 / pp50 + 1 / tg128)` ms

## 引擎调用约定

- **llama.cpp**: `llama-bench -m MODEL -p 50,512 -n 128 -r 3` —— 一发出 pp + tg 数据。`brew install llama.cpp` 装的预编译二进制。
- **ferrum**: `./target/release/ferrum run MODEL --tokenizer TOK --prompt PROMPT --max-tokens N --temperature 0.0 --bench-mode`，单次只测一个 op，3 trials 各起一次进程。`FERRUM_KV_CAPACITY=4096`（PR #55 起的新默认）。
- **mistral.rs**: Python `Runner(Which.GGUF(...), max_seqs=1, paged_attn=False)` + `send_completion_request`（用 raw completion 而非 chat completion 跳过 chat template，与 llama-bench / ferrum --prompt 的「裸 prompt 续写」语义一致）。`pip install mistralrs-metal==0.8.0` 装的预编译 wheel。

## 测试环境（版本快照，2026-04-30 跑出本报告时）

```
hardware:
  cpu     : Apple M1 Max (10 cores)
  memory  : 32 GB unified memory
  gpu     : Apple M1 Max (Metal 3, NOT Metal 4)
  os      : macOS 15.1.1 (Build 24B91)

ferrum:
  commit  : c3bde5c (fix(kv): default capacity 32768→4096 + overflow guard for chat REPL (#55))
  rustc   : 1.91.0 (f8297e351 2025-10-28)
  build   : cargo build --release -p ferrum-cli --features metal

llama.cpp:
  source  : Homebrew, llama.cpp v8960 (libggml 0.10.0)
  binary  : /opt/homebrew/bin/llama-bench
  note    : "tensor API disabled for pre-M5 and pre-A19 devices" — pre-Metal-4 Apple

mistral.rs:
  source  : PyPI, mistralrs-metal==0.8.0 (cp310-abi3-macosx_15_0_arm64 prebuilt wheel)
  python  : 3.12.12 (uv venv at /tmp/mistral_bench)
  note    : Qwen3-MoE GGUF loader 启动 dummy run 崩溃 — 已知限制

bench files:
  Qwen3-8B-Q4_K_M.gguf                    4.7 GB
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  4.6 GB
  Qwen3-30B-A3B-Q4_K_M.gguf               17 GB
```

## 复现命令

完整步骤见 [`bench/README.md`](README.md)。简版：

```bash
# 1. 装 mistral.rs（其它两个已就绪）
uv venv --python python3.12 /tmp/mistral_bench
uv pip install --python /tmp/mistral_bench/bin/python mistralrs-metal==0.8.0

# 2. 跑 bench（每模型 ~5-15 分钟，串行）
SCRIPT="$(pwd)/bench/scripts/bench_one_model.sh"
OUTDIR="$(pwd)/bench/group-a-results"
$SCRIPT Qwen3-8B-Q4_K_M.gguf                    Qwen3-8B.tokenizer.json                    $OUTDIR
$SCRIPT Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  Meta-Llama-3.1-8B-Instruct.tokenizer.json  $OUTDIR
$SCRIPT Qwen3-30B-A3B-Q4_K_M.gguf               Qwen3-30B-A3B.tokenizer.json               $OUTDIR

# 3. 聚合
/tmp/mistral_bench/bin/python bench/scripts/aggregate_bench.py > bench/group-a-report.md
```

## 待办

- **Phase 2: 16 并发 throughput** — 启 HTTP server（ferrum `serve`、mistral.rs `mistralrs serve`、llama.cpp `llama-server`），16 路并发 chat completion，测聚合 t/s。30B-A3B 在 32 GB Mac 上 16 路 KV 估算 12 GB，加 17 GB 模型 = 29 GB，会贴近内存上限——可能要降到 8 并发或 4 并发上限。
- **Phase 3: ferrum decode 攻坚** — 目标 30B-A3B decode 29 → ≥ 50 t/s（封 1.85× decode gap，BW 利用率 29% → ≥50%）。看 `gemv_q4kw_moe_id_f32` kernel 的 GPU 时间占比 + frame capture 找内存访问 pattern 改进点。

## 原始数据

`bench/group-a-results/` 下每模型三个文件：

- `<model>__llamacpp.txt` — `llama-bench` 原始输出
- `<model>__ferrum.txt` — `ferrum run --bench-mode` 9 次输出（pp50×3, pp512×3, tg128×3）
- `<model>__mistralrs.txt` — `bench_mistral.py` 输出（含 mistralrs INFO 日志和 3 个 JSON line）

聚合脚本：`bench/scripts/aggregate_bench.py`。
