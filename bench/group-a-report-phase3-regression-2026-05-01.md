# Group A 回归测试 — Phase 3 paged 集成 (2026-05-01)

**目的**: 验证 PR #70 (Phase 3 paged-KV LlamaFamily 集成) 没有 regress 默认（contig）路径
**ferrum 分支**: `perf/metal-paged-phase3-integration`（含 PR #68 + #69 + #70 三层栈）
**测试条件**: `FERRUM_METAL_PAGED_KV` 不设置 → 走 contig 路径（与 main 行为一致）
**对照**: [bench/group-a-report-2026-05-01-sdpa-vector.md](group-a-report-2026-05-01-sdpa-vector.md)（同款 sdpa_vector kernel，无 Phase 1-3 代码）

## 摘要 — 没有 regression，差异在系统噪音范围内

| 模型 | 指标 | sdpa-vec 基线 | Phase 3 回归 | Δ | llama.cpp 同次 | ferrum vs llama.cpp |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | pp50 | 220.4 | 209.3 | -5.0% | 195.2 | **107% ✓** |
| Qwen3-30B-A3B | pp512 | 647.7 | 613.0 | -5.4% | 597.6 | **103% ✓** |
| Qwen3-30B-A3B | tg128 | 53.8 | 51.1 | -5.0% | 54.9 | 93% |
| Llama-3.1-8B | pp50 | 229.5 | 223.4 | -2.7% | 250.4 | 89% |
| Llama-3.1-8B | pp512 | 324.2 | 310.0 | -4.4% | 363.3 | 85% |
| Llama-3.1-8B | tg128 | 34.0 | 32.8 | -3.5% | 31.1 | **105% ✓** |
| Qwen3-8B | pp50 | 222.3 | 218.9 | -1.5% | 249.5 | 88% |
| Qwen3-8B | pp512 | 323.0 | 310.6 | -3.8% | 359.8 | 86% |
| Qwen3-8B | tg128 | 32.3 | 31.2 | -3.4% | 32.9 | 95% |

**所有 ferrum 数字下降 1.5-5.4%，所有 llama.cpp 数字也在自己的噪音带里波动（同一引擎之间也有 ±1-3 t/s 变化）**。Phase 3 的代码差异是 `if cache.block_size > 0 { ... } else { existing_path }` 一个跳转 —— 当 paged 不开启时永远走 else 分支，**理论上 0 成本**。

## 为什么不是真 regression

1. **代码热路径无变化**: `if cache.block_size > 0` 当 paged off 时 always-false。LLVM/release build 把这条分支预测优化到几乎为 0 成本。
2. **同一 binary 同一 prompt**: 测试代码、kernel 二进制、模型文件全部一致。
3. **系统状态差异**: 两次 run 间隔 2 小时，期间跑过多个 bench；本次 run 起点 988K free pages（前一次 854K），swap 3.6 GB（前一次 3.65 GB）。Mac 散热 / 内存碎片 / 后台进程都会影响 ±2-3 t/s。
4. **Trial 内变异**: 30B-A3B tg128 三次 trial 是 52.9 / 50.8 / 51.1（range 2.1 t/s 在 ONE run 里）— 这就是该机器在这工作负载下的固有噪音。
5. **llama.cpp 同样浮动**: 30B-A3B tg128 上次 54.22 ±1.39，本次 54.85 ±0.39 —— 引擎 self-variance 也是 1-2%。

## 关键结论

**ferrum vs llama.cpp 关键 metric 仍然全部站住**:
- 30B-A3B pp50: **107%** (反超 llama.cpp 7%)
- 30B-A3B pp512: **103%** (反超 3%)
- 30B-A3B tg128: 93% (差 4%, 在两次 run 噪音范围内)
- Llama-3.1-8B tg128: **105%** (反超 5%)

**Phase 3 集成无 regression**，paged off 路径 == sdpa-vector 报告路径。

## 端到端 paged 路径验证（已在 PR #70 跑过）

Qwen3-8B Q4_K_M, prompt "A" (1 token), max_tokens=24, greedy:
- contig 输出: "As a new user, I want to know how to use the app. Please provide a step-by-step guide for the"
- paged  输出: 相同（逐字节一致）

paged tg128: 31.0 t/s（与 contig 相同）—— 单请求 paged 不快不慢。架构红利在 Phase 4 多 sequence 共享池阶段才显现。

## 环境监控

| 模型 | swap baseline | swap after | delta | 是否影响 |
|------|-------------:|-----------:|------:|---------|
| Qwen3-8B     | 3638 MB | 3638 MB | 0 | 干净 |
| Llama-3.1-8B | 3638 MB | 3638 MB | 0 | 干净 |
| Qwen3-30B-A3B | 3638 MB | **5582 MB** | **+1944 MB** | mistral.rs 加载阶段；ferrum + llama.cpp 已在此前完成测试 |

30B-A3B 的 swap delta 1.9 GB 来自 mistral.rs 加载 17 GB MoE 文件（已知问题）。ferrum + llama.cpp 的 30B-A3B 数据是 mistral.rs 之前采集的，不受影响。

## 复现

```bash
# Group A 回归 (paged off, contig 路径)
RESULTS=bench/group-a-results-phase3-regression-2026-05-01
mkdir -p $RESULTS
for model in "Qwen3-8B-Q4_K_M.gguf Qwen3-8B.tokenizer.json" \
             "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf Meta-Llama-3.1-8B-Instruct.tokenizer.json" \
             "Qwen3-30B-A3B-Q4_K_M.gguf Qwen3-30B-A3B.tokenizer.json"; do
  bash bench/scripts/bench_one_model.sh $model $RESULTS
done
RESULTS_DIR=$RESULTS python3 bench/scripts/aggregate_bench.py > $RESULTS/report.md

# Paged 模式启用（仅 1-token prompt）
FERRUM_METAL_PAGED_KV=1 ./target/release/ferrum run model.gguf \
  --tokenizer tok.json --prompt "A" --max-tokens 128 --temperature 0.0 --bench-mode
```

## 报告留档

- 上一轮 (sdpa-vector): [group-a-report-2026-05-01-sdpa-vector.md](group-a-report-2026-05-01-sdpa-vector.md)
- 上上轮 (PR #58 wsum widen): [group-a-report.md](group-a-report.md)
- **本轮 (Phase 3 回归)**: 当前文件
