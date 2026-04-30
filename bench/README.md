# bench/ — 性能对比报告

ferrum vs mistral.rs vs llama.cpp 在 Apple Silicon 上跑同一 GGUF Q4_K_M 文件的性能对比。

## 当前报告

- **[group-a-report.md](group-a-report.md)** — Group A (Apple Silicon 主战场) 三模型 (Qwen3-8B / Llama-3.1-8B / Qwen3-30B-A3B) Phase 1 单请求 pp/tg/TTFT 报告（生成日期 2026-04-30）
- **`group-a-results/*.txt`** — 每 (model, engine) 的原始 bench 输出（保留下来方便复查 / re-aggregate）

## 复现步骤

### 1. 准备 GGUF 模型 + tokenizer

```bash
# 默认路径假定如下；改 GGUF_DIR / TOK_DIR 可换路径
ls /Users/chejinxuan/ferrum-bench/models/
# Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
# Qwen3-30B-A3B-Q4_K_M.gguf
# Qwen3-8B-Q4_K_M.gguf

ls /Users/chejinxuan/ferrum-bench/tokenizers/
# Meta-Llama-3.1-8B-Instruct.tokenizer.json
# Qwen3-30B-A3B.tokenizer.json
# Qwen3-8B.tokenizer.json
```

### 2. 装三个引擎（在 M1/M2/M3/M4 Mac 上）

```bash
# llama.cpp ─ brew 装预编译二进制（不需要本地编译）
brew install llama.cpp

# ferrum ─ 项目自带，release 编译
cargo build --release -p ferrum-cli --features metal

# mistral.rs ─ 用 PyPI 上的 macOS arm64 预编译 wheel（30s 装好，避免本地编译 mistral.rs 源码 30+ 分钟）
uv venv --python python3.12 /tmp/mistral_bench
uv pip install --python /tmp/mistral_bench/bin/python mistralrs-metal==0.8.0
```

### 3. 跑 bench（每模型 ~5-15 分钟，串行）

```bash
cd /path/to/ferrum-infer-rs
OUTDIR="$(pwd)/bench/group-a-results"
SCRIPT="$(pwd)/bench/scripts/bench_one_model.sh"

# 每条命令是「一个模型 × 三个引擎」串行跑完
$SCRIPT Qwen3-8B-Q4_K_M.gguf                Qwen3-8B.tokenizer.json                $OUTDIR
$SCRIPT Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  Meta-Llama-3.1-8B-Instruct.tokenizer.json  $OUTDIR
$SCRIPT Qwen3-30B-A3B-Q4_K_M.gguf           Qwen3-30B-A3B.tokenizer.json           $OUTDIR
```

⚠ 32 GB Mac 跑 30B-A3B 时**关掉浏览器/IDE 等大内存占用**，否则 KV cache + 模型 17 GB 会触 swap 让数字归零。

### 内存监控 / Memory hygiene

每次 bench 运行**自动捕获**机器状态写进结果文件（`vm_stat` 选定行 + `swapusage` + 大 RSS 进程列表），bench 前后各一次。文件里会出现：

```
──────── env snapshot: before ferrum @ 2026-05-01 ... ────────
## vm_stat (selected)
  Pages free: 248247.
  Pages active: 1016049.
  ...
## swap
  total = 5120.00M  used = 3815.38M  free = 1304.62M  (encrypted)
## top non-system RSS (>50 MB)
    901 MB  claude
    642 MB  iTerm
    ...
## warning thresholds
  ⚠ swap_used = 3815 MB (>1 GB) — bench numbers may be paging-affected
```

如果 swap usage > 2 GB（默认门槛），脚本**直接拒绝跑** bench 并告诉你关哪些 app。覆盖：`FERRUM_BENCH_SWAP_THRESHOLD_MB=N`。

**为什么这个改动是必要的**（教训记录）：
2026-05-01 我曾尝试把 gate gemv + up gemv + silu_mul 三个 dispatch 融成一个，理论上应该省 2 个 dispatch overhead × 48 layer ≈ 3 ms。实测 ferrum 38.0 vs baseline 38.2 t/s（差异 < 1%）。
当时 `swap used = 3.82 GB`、Chrome / Spotlight / mds 共 2.4 GB RSS — 测试环境本身在 paging，3 ms 的预期收益完全淹没在 macOS 后台 IO 噪声里。修过来才知道是「环境污染」还是「优化无效」。
完整调查记录：[`bench/notes/2026-05-01-gate-up-silu-fuse-attempt.md`](notes/2026-05-01-gate-up-silu-fuse-attempt.md).

### 4. 聚合到 markdown

```bash
/tmp/mistral_bench/bin/python bench/scripts/aggregate_bench.py > bench/group-a-report.md
```

## 测试方法（与报告 §方法论 同步）

- **硬件**: MacBook Pro M1 Max, 32 GB unified memory, macOS 15.1.1 (Build 24B91)
- **后端**: Metal (所有引擎，统一 GPU 路径)
- **量化**: Q4_K_M（同一份 GGUF 文件三引擎共用，规避 quantization 实现差异）
- **测试串行**: 每次只有一个推理进程在跑（Python 子进程退出 = 干净内存释放，避免内存竞争干扰数字）
- **三个 workload**:

| 标记 | 含义 | 用户体感对应 |
|---|---|---|
| `pp50` | prompt processing 50 tokens (prefill 速率, tok/s) | 短 prompt TTFT 主导项 |
| `pp512` | prompt processing 512 tokens | 长上下文 TTFT、批量 prefill |
| `tg128` | text generation 128 tokens (decode 速率, tok/s) | 流式输出速度（chat 体感） |

- **TTFT(50) 估算**: `1000 × (50 / pp50 + 1 / tg128)` ms — 假设 prefill 完成后第一个 decode tick 等于 1/tg128
- **采样**: `temperature=0.0`（确定性），`max_tokens=N` 硬截断
- **每条 3 trials 取中位数**（llama-bench 内部已 3 次平均；ferrum/mistralrs 本脚本各跑 3 次）

## 引擎调用约定

- **llama.cpp**: `llama-bench -m MODEL -p 50,512 -n 128 -r 3` —— 一发出 pp + tg 数据
- **ferrum**: `./target/release/ferrum run MODEL --tokenizer TOK --prompt PROMPT --max-tokens N --temperature 0.0 --bench-mode`，单次只测一个 op
- **mistral.rs**: Python `Runner(Which.GGUF(...), max_seqs=1, paged_attn=False)` + `send_completion_request`（用 raw completion 而非 chat completion 跳过 chat template，与 llama-bench / ferrum --prompt 的「裸 prompt 续写」语义一致）

## 已知限制

- **mistral.rs Qwen3-30B-A3B**: dummy run 直接挂掉（Engine dead, rebooting）—— mistral.rs v0.8.0 GGUF loader 当前对 qwen3moe arch 支持有限。后续可能用 `Plain(model_id="Qwen/Qwen3-30B-A3B")` + `in_situ_quant` 走非 GGUF 路径绕开。
- **Phase 2 (16 并发吞吐)**: 待做。30B-A3B 在 32 GB Mac 上 16 并发的 KV cache (~12 GB) + 模型 17 GB 接近内存上限，可能要降到 4 并发再补。

## 文件清单

```
bench/
├── README.md                    ← 本文件
├── group-a-report.md            ← 聚合后的 markdown 报告
├── group-a-results/             ← 原始 bench 输出，保留以便复查
│   ├── Qwen3-8B-Q4_K_M__{ferrum,mistralrs,llamacpp}.txt
│   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M__{ferrum,mistralrs,llamacpp}.txt
│   └── Qwen3-30B-A3B-Q4_K_M__{ferrum,mistralrs,llamacpp}.txt
└── scripts/
    ├── bench_one_model.sh       ← 串行跑一个模型 × 三个引擎的 orchestrator
    ├── bench_mistral.py         ← mistral.rs 单次测量 (model, op, n_prompt, n_gen, trials)
    └── aggregate_bench.py       ← 解析 group-a-results/ 输出 markdown 报告
```
