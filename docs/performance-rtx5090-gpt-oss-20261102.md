# RTX 5090 GPT-OSS Fix Validation (2026-11-02, UTC+8)

English | [简体中文](performance-rtx5090-gpt-oss-20261102.zh.md)

This report covers **R11, the pre-release optimization candidate for 0.18.4**, `e36e1cc7e`. Two fresh server processes each completed a C8→C16 sweep, compared with Ferrum 0.18.3 and the retained vLLM baseline on the same GPU and the same source MXFP4 weights.

**Both fresh processes reached at least 90% of vLLM throughput at both concurrency levels and passed the predefined latency SLOs.** C8 achieved 890.93/892.30 tok/s, or 108.68%/108.85% of vLLM; C16 achieved 1383.53/1385.86 tok/s, or 96.77%/96.93%. All 128/128 measured requests succeeded, with no errors or rejections. These are two independent startup checks for this workload, not evidence of long-running stability or performance across all models.

## Primary results

Predeclared P99 SLOs: TTFT ≤2000 ms, TPOT ≤50 ms, and non-empty visible SSE text-event ITL ≤100 ms, with 100% measured-request success. The throughput thresholds are 90% of vLLM: **737.79 tok/s at C8** and **1286.78 tok/s at C16**. Each process ran 32 measured requests, 2 client warmup requests, and 1 repetition per concurrency level. Server capacity and configuration stayed fixed across C8/C16. Latencies below are **P50/P99 in ms**.

| C | Engine / fresh process | Successful / offered | Errors / rejected | Output tok/s | TTFT | TPOT | Visible SSE ITL | Latency SLO / throughput threshold |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | Ferrum 0.18.3 | 32/32 | 0/0 | 72.42 | 54648.74/56304.24 | Not streaming evidence | Missing | Fail / fail |
| 8 | R11, process 1 | 32/32 | 0/0 | **890.93** | 155.70/227.27 | 8.17/8.62 | 8.00/14.38 | Pass / pass |
| 8 | R11, process 2 | 32/32 | 0/0 | **892.30** | 157.81/226.31 | 8.21/8.61 | 7.95/13.02 | Pass / pass |
| 8 | vLLM 0.31.0 | 32/32 | 0/0 | 819.77 | 94.75/387.13 | 8.71/8.96 | 8.16/33.00 | Pass / baseline |
| 16 | Ferrum 0.18.3 | 32/32 | 0/0 | 74.30 | 100934.84/109512.78 | Not streaming evidence | Missing | Fail / fail |
| 16 | R11, process 1 | 32/32 | 0/0 | **1383.53** | 346.93/462.27 | 10.00/10.29 | 9.88/11.73 | Pass / pass |
| 16 | R11, process 2 | 32/32 | 0/0 | **1385.86** | 340.35/444.62 | 10.10/10.39 | 9.89/12.27 | Pass / pass |
| 16 | vLLM 0.31.0 | 32/32 | 0/0 | 1429.75 | 174.08/183.29 | 10.14/10.38 | 9.47/34.24 | Pass / baseline |

The old release delivered only one text event per request. Its reported last-visible TPOT of 0 and missing ITL do not mean zero latency. All four R11 cells have continuous text-timing evidence for all 32 requests, with no missing usage or detected transport coalescing. Strict single-token ITL remains ineligible because text-event counts differ from usage-token counts. R11 still has higher C16 first-text latency than vLLM; it does not lead on every latency metric.

GPT-OSS uses natural EOS, default `medium` reasoning effort, and a maximum of 512 tokens per request. TTFT includes the first non-empty reasoning or final text. TPOT uses the first and last visible events and total completion tokens; throughput is successful usage-output tokens divided by measured duration. These are not final-only metrics. Two fresh processes remain a limited sample; vLLM was not rerun, and percentiles were not averaged.

## Changes covered

Incremental Harmony parsing emits confirmed reasoning/content early while retaining control prefixes split across chunks. Tool arguments, JSON, and other strict output constraints still require full validation. Parser state references offsets in the already-accounted raw text instead of rescanning or copying the entire history per token. Shared Chat, Responses, and CLI text-REPL paths were validated.

CUDA attention improves context parallelism and page addressing while preserving window, sink, YaRN, and independent KV-page semantics. Typed provider capabilities avoid repeatedly constructing unsupported static declarations; every wave still validates KV state, request authority, and resource boundaries. The pinned FlashInfer XQA decode path reads existing pages directly. Sink conversion, workspace, and semaphores are budgeted, and single-split execution avoids an unnecessary reset.

XQA is currently integrated only into the GPT-OSS operation's **SM120/F16 decode path, with head dimension 64, 64 query heads, and 8 KV heads**. Other hardware, or builds without the pinned operator, retain the existing path. This does not establish a benefit for every model or GPU.

Default `Auto` now warms, captures, and replays before readiness according to admitted capacity and budgets, while retaining bounded on-demand learning and eviction. Explicit `OnDemand` remains lazy; `Startup` keeps a sealed catalog. Fresh product requests can reuse graphs after synthetic startup owners are released. Current sampling masks and terminal readback remain bound per wave. `run` and `serve` share the preparation barrier and require no new user tuning.

Optional live graph construction is admitted using the host work that recent successful executions could save and the measured graph-preparation cost. Insufficient benefit evidence temporarily defers construction while normal inference and bounded learning continue. The policy does not branch on model names, concurrency levels, or a fixed request count. Declared startup graphs bypass this economic gate; explicit legacy modes preserve their behavior. Cost estimates for unseen shapes still cannot guarantee that no new spike will occur.

## Startup cost and resource usage

Both startups prepared **34 programs/graphs**: decode widths 1–32 and single-request prefill shapes of 1/256 tokens. Catalog capacity was 3400, with no startup rejection or capacity deferral. Active/pending/executing/maintenance counts were all zero after startup, and measured counters were reset. Published resident catalogs remained at 34 before and after the workload, with lane epoch 1; bounded learning remained enabled.

| R11 process | Spawn → first successful HTTP health | Executor preparation report | Startup sampling window | Startup whole-device VRAM peak | Startup process RSS peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 73.746s | 3.065s | 73.756s | 15.060GiB | 13.917GiB |
| 2 | 74.743s | 4.266s | 74.764s | 15.066GiB | 13.917GiB |

Executor preparation is nested within startup and must not be added to it. Executor preparation was ready at the first health response; `text_ready` became true after the arithmetic probe. The historical vLLM engine-init measurement of about 51.25s has different boundaries from spawn→HTTP health, so no startup speed ratio is claimed.

The following independent samples cover the **entire C8+C16 sweep**, including client preparation and warmup. They are not per-concurrency peaks. The configured sampling interval was 250 ms.

| Engine | Sampling window | Whole-device VRAM peak | Window RSS peak | Lifetime RSS high-water mark | GPU / host samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrum 0.18.3 | 452.509s | 15.990GiB | 1.389GiB | 13.482GiB | 1804/1812 |
| R11, process 1 | 37.258s | 15.978GiB | 1.177GiB | 13.915GiB | 132/151 |
| R11, process 2 | 36.764s | 16.165GiB | 1.192GiB | 13.914GiB | 131/149 |
| vLLM | 37.512s | 29.937GiB | 3.145GiB | 5.925GiB | 147/152 |

Sampling errors were zero. Maximum GPU/host gaps in the R11 measured windows were 2016/251 ms and 1488/251 ms; transient peaks can be missed. Device memory includes other resident processes and the driver; WSL did not provide reliable target-process GPU accounting. Ferrum RSS covers its single server process; vLLM RSS covers its EngineCore worker, not the entire process tree. Lifetime high-water marks include loading and are not window peaks.

vLLM reserves KV memory at utilization 0.9, with a pool of 275324 tokens. This differs from Ferrum's on-demand allocation and does not establish a memory-efficiency ratio. Each R11 process completed 69/69 requests: 64 measured, 4 warmup, and 1 arithmetic probe. Active/queued counts were zero at the end; only the experiment's services were stopped.

<details>
<summary>Graph-construction diagnosis and earlier results</summary>

R9's two fresh processes achieved C8 889.90/899.10 and C16 1307.43/1250.71 tok/s. Its second C16 result reached only 87.48% of vLLM and missed the target. The first 16 requests all produced 512 tokens in both runs, yet their mean TTFT rose from about 298.56 to 766.42 ms; natural-EOS length variation cannot explain that difference. R9 results remain archived but are not the final primary table.

In a `basic` diagnostic using the same workload, first versus same-process hot C16 prefill host-encode totals were 538.679/107.130 ms, including 449.824/0.209 ms in `validate_and_prepare`; device-event totals were 626.436/630.879 ms. This diagnostic omitted the primary run's arithmetic probe. It is diagnostic evidence, not a replacement score.

R10 added timing subdivisions without changing inference behavior. One single-sequence, 142-token prefill invocation spent 515.826 ms in host encode, including 512.906 ms in `cuGraphInstantiateWithFlags`. Graph recording, topology attribution, and upload took only 0.591, 0.077, and 0.820 ms; this invocation did not enter input-packet allocation/synchronization. This identifies graph construction blocking the request path, not the CUDA driver's internal cause. A separate resource-preparation peak of 247.037 ms included 215.521 ms in submission-wave preparation. These are distinct stages; nested timings must not be added.

In each final R11 window, the single-sequence 142-token shape had two catalog misses, while the published catalog stayed at 34 resident graphs with lane epoch 1. Primary `profile=off` runs do not collect attempt records or detailed `AmortizationDeferred` counts: zero-valued fields are not proof of zero events. No exact deferral count or graph-construction substage duration is claimed for that shape. CPU and real CUDA tests separately validate the cost controller's state transitions.

</details>

<details>
<summary>Output lengths and validity limits</summary>

The old release, both R11 processes, and vLLM have identical selected samples, per-prompt hashes, output budgets, and raw/server input-token vectors. Input totals per cell are 4440/6584 tokens before/after server templating. Actual natural-EOS output lengths need not match.

| Engine | C | Output tokens | Per-request range | Requests reaching the 512 cap |
| --- | ---: | ---: | --- | ---: |
| Ferrum 0.18.3 | 8 | 15176 | 190–512 | 27/32 |
| R11, process 1 | 8 | 15223 | 190–512 | 27/32 |
| R11, process 2 | 8 | 15126 | 190–512 | 26/32 |
| vLLM | 8 | 15064 | 188–512 | 27/32 |
| Ferrum 0.18.3 | 16 | 14999 | 194–512 | 26/32 |
| R11, process 1 | 16 | 15062 | 190–512 | 27/32 |
| R11, process 2 | 16 | 15148 | 190–512 | 27/32 |
| vLLM | 16 | 15115 | 117–512 | 26/32 |

Between R11 processes, 4/32 output lengths changed at C8, totaling 97 fewer tokens; 3/32 changed at C16, totaling 86 more tokens. No end-to-end bitwise equivalence or same-answer speedup is claimed.

Most requests reached the cap. The reports do not establish complete final-answer coverage for every request. Protocol success and a correct standalone arithmetic probe do not prove that all answers are complete and correct. The earlier `ignore_eos=true` protocol failure remains in the [historical report](performance-rtx5090-popular-models-20261101.md); passing natural EOS does not cover that issue.

</details>

<details>
<summary>Validation and intermediate candidates</summary>

Four real CUDA tests passed on R11: deferring unearned live capture while preserving startup graphs; Auto releasing synthetic owners and serving fresh requests with current decode masks and new shapes; explicit hybrid retaining its warm/capture/replay learning; and sealed mode preserving exact descriptors. An earlier fixture incorrectly required a prefill result to be `GreedyToken`; only that expectation was corrected to the existing `FullLogits` contract. Decode-mask/replay assertions remained strict, and the original failure was retained.

Both serve arithmetic probes on the same R11 binary returned final `4`, HTTP 200, and complete DONE/usage. An independent `run` text-REPL invocation printed `4` and exited normally. An earlier long-answer check produced 72 content events without leaking Harmony control tokens. CPU tests cover early Chat/Responses output, nonduplicated terminal output, and errors that must not masquerade as normal completion.

R11 passed workspace fmt/check/test/Clippy (`-A warnings`) and Metal compilation: 6873 tests passed, 0 failed, and 79 were ignored. The required CUDA CLI feature compilation check also passed. Ignored tests, empty feature-disabled suites, and compilation are not runtime passes; Metal was compile-only.

Real RTX 5090 XQA numerical, graph-replay, and workspace-boundary checks passed. The independent F64 reference tolerance is `1e-3 × max(|reference|, 1)`. Earlier scalar/packed F16 comparisons used elementwise `PartialEq`, which does not distinguish ±0, so they are not described as strictly bitwise. Prepare/YaRN/KV checks compare against the previous implementation, not an independent CPU YaRN oracle.

Intermediate primary C8/C16 throughput was R5 573.32/859.04, R6 865.50/1198.10, R7 862.49/1190.36, and R8 794.78/1176.36 tok/s. The latest candidate was not replaced by selected best scores. R8 once reached 1389.03 tok/s at hot same-process C16; that was diagnostic, did not exclude unmeasured environmental factors, and was not default-product acceptance evidence.

</details>

<details>
<summary>Versions, workload identity, and reproduction commands</summary>

Hardware: the same RTX 5090 (driver reports 32607 MiB; driver version 610.88), Intel Core Ultra 9 285K, Linux/WSL2. Existing resident services stayed unchanged, and tested engines ran sequentially.

| Item | Pinned identity |
| --- | --- |
| R11 candidate server | `e36e1cc7e8b095faaaf72c66e85e5b6d9448bf8b`; unreleased |
| Candidate CLI SHA256 | `8c09ebc1ffddd0869e45c0092065dddbd630fe1abc3135fd84a2950c0d1264a8` |
| Old Ferrum server and benchmark client for all engines | Released 0.18.3, `e991c6955c3022f80d7e96f9d16d9306314054cf` |
| Released client SHA256 | `76094c69c77d0a1e6d675900c0a91b28bdee3d4237c930193abdac709ebee087` |
| vLLM | 0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6`; retained historical baseline |
| Model | `openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`; the same official MXFP4 weights |
| Tokenizer SHA256 | `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3` |
| Harmony template SHA256 | `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146` |
| ShareGPT file SHA256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| Selected-sample SHA256 | `ceec98e3917466c7dc7347fc0522b2ef1626dd35651747d1018b65b05c562616` |

The historical report records individual weight-shard identities and the cached vLLM image identity. That local image ID is not a publicly pullable registry digest; reproduction must verify the actual version and source revision.

Both Ferrum versions automatically selected `gpt_oss.f16-mxfp4-bf16`: F16 outer activations, a BF16 MXFP4 expert path, and F16 KV. vLLM used native BF16 activations/KV, MARLIN MXFP4 MoE, and FlashInfer attention. Identical source weights do not mean identical internal arithmetic. Both used Harmony's default `medium` effort and the system date 2026-11-02 in Asia/Shanghai.

Filters: 4–1024 input tokens, 128–256 reference-answer tokens, and input + requested output cap + 32 template-reserve tokens ≤2048. Samples are filtered, not truncated. Seed is 42; selection pool is 34. Actual server-templated input vectors were also matched per request. Primary profiling was `off`; server capacity stayed at 32 sequences, context 2048, and 2048 batched tokens, with prefix caching disabled.

Startup preparation follows each product's defaults. vLLM warms the model and captures CUDA graphs before accepting requests. R11 prepares declared graphs before readiness, then continues bounded on-demand learning based on measured cost. The client still uses only 2 warmup requests per cell; no extra benchmark-specific warmup hides first-use shape costs.

Candidate server:

```sh
/path/to/candidate-ferrum serve /path/to/gpt-oss-20b \
  --semantic-source /path/to/gpt-oss-20b \
  --tokenizer-source /path/to/gpt-oss-20b \
  --backend cuda --max-model-len 2048 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --kv-dtype fp16 \
  --gpu-memory-utilization 0.9 --disable-prefix-cache \
  --host 127.0.0.1 --port 19081 --served-model-name gpt-oss-20b \
  --effective-config-json /path/to/effective.json
```

Unmodified released 0.18.3 Rust client:

```sh
/path/to/ferrum-0.18.3 bench-serve \
  --base-url http://127.0.0.1:19081 --model gpt-oss-20b \
  --tokenizer /path/to/gpt-oss-20b --target-backend cuda \
  --dataset sharegpt --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-prepared-cache-dir /path/to/gpt-oss-prepared-cache \
  --seed 42 --concurrency-sweep 8,16 \
  --num-prompts 32 --warmup-requests 2 --n-repeats 1 \
  --sharegpt-selection-pool-size 34 \
  --sharegpt-min-input-tokens 4 --sharegpt-max-input-tokens 1024 \
  --sharegpt-min-output-tokens 128 --sharegpt-max-output-tokens 256 \
  --sharegpt-max-total-tokens 2048 --sharegpt-chat-template-reserve-tokens 32 \
  --sharegpt-fixed-output-tokens 512 --temperature 0 --repetition-penalty 1 \
  --latency-slo ttft:2000,tpot:50,itl:100 \
  --latency-slo-fail-on-violation --fail-on-error --max-error-rate 0 \
  --timeout 120 --output json --out /path/to/report.json
```

Here, `--sharegpt-fixed-output-tokens 512` fixes the requested budget. `--ignore-eos` is not enabled, so normal EOS can end a request early. Neither client nor server overrides default thinking/effort. The historical report retains the actual vLLM command, including `--reasoning-parser openai_gptoss`, equivalent capacity/cache settings, and the same timezone.

</details>

<details>
<summary>Raw evidence</summary>

JSON reports, configurations, commands, logs, and resource samples are stored outside the repository under `20261102-gpt-oss`; historical evidence is under `20261101-popular-model-comparison`. Both R11 primary archives and all 51/39 member files were individually SHA256-verified.

| Artifact | SHA256 |
| --- | --- |
| R11 fresh process 1, complete archive | `ae6d62c84d31dda4719d89ce00ee30977329e0ac5ac49677616a46db28a95b2c` |
| R11 fresh process 2, complete archive | `ab2a59c4beb12b3c4529cd729d603c27b47bd7db40cc8d935c065073590af3c4` |
| R11 real CUDA validation archive, 21 members | `adfd853ccac69f7c7bcc13df9963f813266d7361203655f84ba5ca8419d31bc1` |
| R11 independent run validation archive | `f9f19093b3e79b695ce7e10a147eae24e379bd56d877f94607788a45aaee149b` |
| R9 fresh processes 1/2, complete archives | `2b8489199700b12b0bea645c6440bb297f875e25831addd415bb523ffa1ec1fc` / `5d1f3cd4b24fd388970fb7ab01b0c482f13790d21b38c096ce029f79f9693944` |
| R9 basic phased diagnostic, 32 verified members | `af2c9f6805014d11a661bcafa731246b764a7cc077da7f7e71b38c01c8e27921` |
| R10 graph-preparation diagnostic, 43 verified members | `494fa8306c07f5e8025dc604bb5def6c586893c8bd95355856a9d23f1cd6750a` |
| R11 build-source manifest | `18b1da7dec18f8cc348ce44287972c8d5fb41af6a435ca502ef9185df18be7c7` |

</details>

## MoE models measured so far

These are [historical Ferrum 0.18.3 results](performance-rtx5090-popular-models-20261101.md). Only GPT-OSS was rerun for this change.

| Model | C8 tok/s (Ferrum / vLLM) | C16 tok/s (Ferrum / vLLM) | Historical SLO |
| --- | ---: | ---: | --- |
| Qwen3-30B-A3B GPTQ INT4 | 630.5/824.0 | 848.8/1263.7 | Both engines passed both levels and repetitions |
| Qwen3.5-35B-A3B GPTQ INT4 | 717.6/625.6 | 978.3/947.4 | Ferrum passed all; vLLM C16 repetition 1 exceeded ITL |
| GPT-OSS-20B MXFP4 | 72.4/819.8 | 74.3/1429.8 | Ferrum failed both levels with missing ITL; vLLM passed |

Qwen used two repetitions totaling 128 requests per concurrency level; historical GPT-OSS used one repetition of 32. Models, quantization, attention, and numerical paths differ, so there is no cross-model ranking. The 35B C16 winner differed between repetitions; its pooled advantage of about 3.3% does not establish a stable advantage.
