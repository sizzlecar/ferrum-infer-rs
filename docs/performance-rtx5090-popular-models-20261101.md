# RTX 5090 concurrent serving comparison for popular text models (2026-11-01–02, UTC+8)

[中文](performance-rtx5090-popular-models-20261101.zh.md)

This comparison uses the official Ferrum 0.18.3 release against llama.cpp and vLLM on the same RTX 5090. All 7 paired comparisons are complete: in 6 groups, each engine completed 256/256 measured requests successfully; in the shorter GPT-OSS natural-EOS comparison, each completed 64/64, but Ferrum did not meet the streaming latency SLOs. Only paired results with the same source weights and workload are compared; models and quantization formats are not mixed together.

For subsequent GPT-OSS streaming fixes, CUDA optimizations and retesting, see the [fix validation report](performance-rtx5090-gpt-oss-20261102.md). This page preserves the original 0.18.3 results.

The objective is to increase output throughput while meeting TTFT, TPOT and visible-text ITL latency SLOs. Results exceeding any required SLO are retained, but do not count as successful optimizations. [Historical 27B results](performance-popular-models-p0.zh.md) remain separate; they were not retested here, and older-version numbers are not mixed into the new tables.

## Main results

The preregistered P99 SLOs are TTFT ≤ 2000ms, TPOT ≤ 50ms and non-empty SSE text-event ITL ≤ 100ms, with 100% success for measured requests. Pooled throughput is the total successful usage output tokens across both repetitions divided by their total measurement time. Each engine/concurrency cell contains 128 requests; Qwen3.5 produces 25412 tokens per cell and Qwen3-30B produces 25087. Latency quantiles retain their original per-repetition values.

| Model / source weights | Reference | C | Ferrum tok/s | Reference tok/s | Ferrum / reference | SLO across both repetitions |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B Q4_K_M | llama.cpp | 8 | 960.8 | 634.7 | 1.514× | Both pass |
| Qwen3.5-4B Q4_K_M | llama.cpp | 16 | 1294.8 | 729.4 | 1.775× | Ferrum repetition 1 fails ITL; reference passes |
| Qwen3.5-9B Q4_K_M | llama.cpp | 8 | 731.6 | 550.5 | 1.329× | Both pass |
| Qwen3.5-9B Q4_K_M | llama.cpp | 16 | 942.0 | 654.7 | 1.439× | Ferrum repetition 1 fails ITL; reference passes |
| Qwen3.5-4B official BF16 | vLLM | 8 | 840.0 | 684.6 | 1.227× | Both pass |
| Qwen3.5-4B official BF16 | vLLM | 16 | 1306.8 | 1153.9 | 1.132× | Both pass |
| Qwen3.5-9B official BF16 | vLLM | 8 | 563.4 | 508.2 | 1.109× | Both pass |
| Qwen3.5-9B official BF16 | vLLM | 16 | 875.0 | 864.3 | 1.012× | Both pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM | 8 | 630.5 | 824.0 | 0.765× | Both pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM | 16 | 848.8 | 1263.7 | 0.672× | Both pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM | 8 | 717.6 | 625.6 | 1.147× | Both pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM | 16 | 978.3 | 947.4 | 1.033× | Ferrum passes; vLLM repetition 1 fails ITL |

For the official BF16 source-weight comparisons, Ferrum actually uses F16 activations/KV and vLLM uses native BF16 activations/KV. Both are 16-bit formats, but their numerical paths differ. In 4B C16 repetition 1, Ferrum TTFT P99 is 486.18ms, above vLLM's 173.06ms; visible ITL P99 is 38.00ms, below vLLM's 63.88ms. Both satisfy the SLOs above. Higher throughput does not mean lower latency at every quantile.

For 9B official BF16 source weights at C16, the two repetitions are 802.5/953.9 tok/s for Ferrum and 843.8/883.9 for vLLM. The leading engine changes between repetitions, and the pooled difference is only 1.2%; no stable, significant advantage is claimed.

For 30B GPTQ, Ferrum throughput is lower by 23.5%/32.8% at C8/C16. Both engines use F16 activations/KV and satisfy the SLOs. In C16 repetition 1, TTFT P99 is 578.95/109.97ms and TPOT P99 is 21.39/12.37ms (Ferrum/vLLM). Both repetitions with each engine's fixed configuration are included.

Ferrum passes the SLOs in all four 35B GPTQ measurements. vLLM C16 repetition 1 has visible ITL P99 of 136.73ms, exceeding the 100ms limit. The pooled C16 throughput difference is only 3.3%, and the leading engine changes between repetitions, so this does not establish a stable throughput advantage. Ferrum uses F16 activations/KV in this group; vLLM uses native BF16 activations/KV.

GPT-OSS uses a separate short comparison: default medium reasoning, natural EOS, up to 512 tokens per request, 32 requests per cell and only 1 repetition. Throughput includes reasoning tokens and actual output lengths may differ. The ratios below describe raw throughput for this workload, not speedup for identical answers; they also provide no repeated-run stability evidence.

| GPT-OSS-20B MXFP4 | Ferrum tok/s | vLLM tok/s | Ferrum / vLLM | TTFT P99 ms (F/V) | SLO |
| --- | ---: | ---: | ---: | ---: | --- |
| C8 | 72.4 | 819.8 | 0.088× | 56304.24/387.13 | Ferrum fails and lacks ITL; vLLM passes |
| C16 | 74.3 | 1429.8 | 0.052× | 109512.78/183.29 | Ferrum fails and lacks ITL; vLLM passes |

Ferrum's Harmony streaming interface delivers the entire text only after generation ends. TPOT therefore appears as 0 and ITL has no consecutive intervals; this must not be interpreted as zero latency. This observation does not identify the cause of the overall throughput gap. Most requests on both engines reach the 512-token ceiling, and completion of a final answer was not verified for every request.

## Comparison scope

| Model | Source weights / quantization format | Reference | Status |
| --- | --- | --- | --- |
| Qwen3.5-4B | Same Q4_K_M GGUF | Ferrum / llama.cpp | Measurements complete for both |
| Qwen3.5-4B | Same official BF16 weights | Ferrum / vLLM | Measurements complete for both; internal numerical paths differ |
| Qwen3.5-9B | Same Q4_K_M GGUF | Ferrum / llama.cpp | Measurements complete for both |
| Qwen3.5-9B | Same official BF16 weights | Ferrum / vLLM | Measurements complete for both; internal numerical paths differ |
| Qwen3-30B-A3B | Same GPTQ INT4 weights | Ferrum / vLLM | Measurements complete for both |
| Qwen3.5-35B-A3B | Same GPTQ INT4 weights | Ferrum / vLLM | Measurements complete for both; internal numerical paths differ |
| GPT-OSS-20B | Same MXFP4 weights | Ferrum / vLLM | Both complete default medium with natural EOS; Ferrum fails SLOs, vLLM passes; fixed-output protocol failure also retained |

<details>
<summary>Model identities, frozen workload and server configuration</summary>

GGUF is compared separately from BF16/GPTQ/MXFP4. Results from different quantized files are not described as a three-engine comparison using identical weights. BF16 in the table above describes the source-weight format; it does not mean every engine uses BF16 for intermediate activations. Each group preserves the model repository revision and identities of the weight SHA256, config, tokenizer and chat template; the actual numerical profile, activation precision and KV precision are recorded separately.

GGUF byte identities verified on the test host:

Upstream sources follow the [archived GGUF input inventory](next-release-gguf-sources.zh.md#权重与语义来源): the pinned revisions for [4B weights](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523) and [9B weights](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5). SHA256 values of the complete files on the host match that inventory. The separate semantic sources are [Qwen 4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) and [Qwen 9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a). Cache filenames do not replace byte verification.

| Model | Weight bytes | Weight SHA256 | semantic config SHA256 |
| --- | ---: | --- | --- |
| Qwen3.5-4B Q4_K_M | 2740937888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| Qwen3.5-9B Q4_K_M | 5680522464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` |

Both groups use tokenizer JSON SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` and chat template SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. These identify the actual files used, not just model filenames.

The official 4B SafeTensors source is `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its two weight shards total 9319828096 bytes, with SHA256 values `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` and `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`; size and SHA256 were checked for every shard after download. Its config, tokenizer and template match the 4B semantic source above.

The official 9B SafeTensors source is `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Its four weight shards total 19306310880 bytes, with SHA256 values `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`, `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`, `7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec` and `b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`. Actual sizes/SHA256 values of all 13 files were checked. Its config, tokenizer and template match the 9B semantic source above.

The 30B GPTQ source is the [pinned revision of Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/tree/9b534e4318b7ebc3c961a839f13eb18b1833f441), declaring INT4, group size 128, symmetric quantization and `desc_act=false`. The complete weights contain 16933256392 bytes, with SHA256 `1ad50ff713f7be134a479990e9d70472f5c8ab12e0745d824e25d22f389c8495`; tokenizer JSON SHA256 is `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, and the tokenizer config containing the template is `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`.

The 35B GPTQ source is the [pinned revision of Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/tree/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b). Its 14 weight shards total 24419939760 bytes. The inventory verifying actual sizes/SHA256 values of all 25 files has SHA256 `272404f8f1796469cdbdf77cb3e7af69a1f3c0c41be93064f9481ca3bafacf00`. Config SHA256 is `13dd6a453236a33f16873dced7558fe7c9751e09f6c86ccc0cab66e3f23d4154`; the tokenizer and template match the Qwen3.5 tokenizer/template above. Quantization is INT4, group size 128, symmetric, with `desc_act=false`; some attention/shared-expert and other modules are declared unquantized. The source format name does not imply that every module executes in INT4. Source-file integrity and runtime results below are recorded separately.

GPT-OSS uses the [pinned revision of openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee). Its three native MXFP4 weight shards total 13761316904 bytes, with SHA256 values `16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427`, `4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea` and `a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303`. Tokenizer JSON SHA256 is `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3`; the Harmony template is `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`. Source-file integrity verification is recorded separately from runtime results.

## Frozen workload and SLOs

| Item | Setting |
| --- | --- |
| Client concurrency | C8, C16; server capacity fixed throughout the sweep |
| Repetitions | Default: 2 repetitions per cell, each with 10 warmups and 64 measured requests; see the GPT-OSS natural-EOS short-comparison exception below |
| Request timeout | 120 seconds |
| Dataset | ShareGPT_V3_unfiltered_cleaned_split.json |
| Raw file SHA256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| Selection | First human/GPT pair; seed 42; selection pool 74 per repetition; filtering, no truncation |
| Input length | Raw user text: 4–1024 tokens, excluding the server template |
| Output length | Default: actual reference-answer token length, limited to 128–256; see the GPT-OSS 512-token ceiling and natural-EOS exception below |
| Total-length filter | Input + requested output + 32-token template reserve ≤ 2048 |
| Generation policy | `temperature=0`, `repetition_penalty=1`, `ignore_eos=true` |
| Thinking policy | Disabled on both engines where the model template supports the switch; other model policies recorded separately |
| P99 latency SLO | TTFT ≤ 2000ms; TPOT ≤ 50ms; visible SSE text-event ITL ≤ 100ms |
| Success requirement | 100% success for measured requests; warmup failures recorded separately |

For each model, the same tokenizer is used for selection, with exactly the same requests for both engines and both concurrency cells within a repetition. The two repetitions use different deterministic selections. Checks cover `dataset_evidence.repeats[].selection_sha256`, per-request prompt/answer hashes and request-length vectors, and align actual server input and generated token counts. The template reserve is only a filtering budget; it does not replace checking actual template lengths.

GPT-OSS-20B was initially frozen with `reasoning_effort=low`, without `enable_thinking`, retaining the Harmony template. Reference answers were filtered to 128–256 tokens, with requested output fixed at 512 tokens. For this workload, `ignore_eos=true` triggered a Harmony termination protocol error in Ferrum 0.18.3. The original attempt is preserved as a failure and is not used to calculate a throughput advantage.

The first natural-EOS `low` diagnostic repetition completed 64 requests with no request errors and a measured duration of 390.0s, but did not produce a complete matrix. It was subsequently found that the official client placed `low` only in `chat_template_kwargs`, while this vLLM version's internal GPT-OSS Harmony builder reads top-level `reasoning_effort`. It therefore cannot be assumed that both engines actually used the same `low` setting. [vLLM rendering path](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/renderers/online_renderer.py#L459).

A separate short comparison was finally frozen using the default policy: neither engine receives `ignore_eos` or `reasoning_effort`; both use the model default `medium`, `max_tokens=512` per request, and 32 measured requests plus 2 warmups per C8/C16 cell, for 1 repetition. Both the official template and the actual vLLM image were confirmed to default to `medium` for Harmony. Input filtering, the 2048 total-length cap and the SLOs remain unchanged. Here 512 is a ceiling; actual output lengths may differ and their distributions must be reported separately. Selection, inputs and request budgets are checked for equality, but actual output vectors are not required to match. Functional preflight must confirm final visible text, and missing visible timing cannot count as a pass. This short comparison provides no repeated-run stability evidence, is not pooled with the fixed-output groups, and does not supersede the two earlier diagnostics.

GPT-OSS timing is also limited by the client's metric definitions: this version counts non-empty SSE `content`, `reasoning` or `reasoning_content` uniformly as text updates, and the performance report does not retain separate final/reasoning token usage. TTFT may therefore refer to the first streamed reasoning text, while the TPOT denominator remains total `completion_tokens - 1`. These metrics cannot be described as final-only first-text latency or strict per-token speed of the final answer. Final-answer validity is checked separately, and the original timing metrics are retained.

Ferrum 0.18.3 has a specific Harmony streaming limitation: [`chat_stream.rs`](https://github.com/sizzlecar/ferrum-infer-rs/blob/e991c6955c3022f80d7e96f9d16d9306314054cf/crates/ferrum-server/src/axum_server/chat_stream.rs#L99) enables complete-output buffering for this protocol, parsing and publishing content only after the engine finishes. The first visible SSE update may therefore arrive near overall completion. A single text delivery produces TPOT 0 and no consecutive visible ITL intervals; this is not zero latency or smooth token-by-token output. Missing timing samples and SLO failures must remain in the report. This explains visible streaming behavior, but does not prove that buffering also causes the overall throughput gap; the operator-throughput root cause had not been established from this evidence.

The completed Qwen3.5 4B/9B reports all record `enable_thinking=false`. This reasoning-channel limitation is not used to rewrite those measurements with thinking disabled.

## Server capacity and cache definitions

| Setting | Ferrum | llama.cpp | vLLM |
| --- | --- | --- | --- |
| Per-request context limit | `--max-model-len 2048` | unified KV: `--kv-unified-per-slot 2048`, total pool 65536, 32 slots | `--max-model-len 2048` |
| Maximum active sequences | `--max-num-seqs 32` | `--parallel 32` | `--max-num-seqs 32` |
| Logical scheduled-token limit | `--max-num-batched-tokens 2048` | `--batch-size 2048` | `--max-num-batched-tokens 2048` |
| Physical microbatch | Actual engine execution policy | Default `--ubatch-size 512`, disclosed separately | Actual engine execution policy |
| Prefix cache | `--disable-prefix-cache` | `--no-cache-prompt --cache-ram 0 --no-cache-idle-slots` | `--no-enable-prefix-caching` |
| GGUF group KV precision | FP16 | F16 for both K/V | Not applicable |
| Automatic capacity adjustment | Effective config retained | `--fit off --no-context-shift` | Actual startup configuration and KV capacity retained |

These parameters align request capacity and logical budgets; they do not imply identical kernel batch sizes, prefill/decode scheduling or physical VRAM reservation across engines. llama.cpp retains the existing baseline's `--kv-unified --kv-unified-per-slot 2048 --ctx-size 65536 --parallel 32 --ctx-checkpoints 0`, frozen after a short confirmation in this run. Shared-pool capacity and the per-request limit are recorded separately. `--cache-reuse 0` alone does not disable the prompt cache.

Unified KV layout and checkpoint settings are performance configuration: in the pinned version, the hybrid non-unified path requires adjacent sequence IDs within a microbatch. Checkpoints also do not turn off automatically with `cache_prompt=false`. This table uses only the frozen unified, checkpoints=0 baseline above; diagnostics for other configurations are not mixed into the main results. [Adjacent-sequence restriction](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-batch.cpp#L583), [checkpoint condition](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-context.cpp#L3377).

Ferrum/vLLM's 32 sequences and 2048 context are logical limits, not a physical reservation of 65536 KV tokens. Results must include the actual KV dtype, pool capacity, GPU layer placement and attention implementation. GGUF comparisons offload all layers to the GPU; results that do not meet this condition are listed separately and excluded from same-configuration conclusions.

The 4B Ferrum startup receipt confirms effective limits of 32/2048/2048, F16 KV, session cache off, weights on one device and attention policy `cuda_native_adaptive`. The model also has fixed per-sequence state, so complete VRAM use cannot be inferred from KV tokens alone.

Both 4B and 9B llama.cpp startup logs confirm 33/33 layers offloaded to the GPU, 32 slots, total context 65536, context 2048 per slot, Flash Attention enabled and physical microbatch 512. Both allocate a 2048MiB CUDA KV buffer and a 1608MiB recurrent-state buffer. These allocations have a different scope from the sampled whole-GPU peak.

Identical weights do not imply identical internal arithmetic. For both 4B and 9B GGUF, Ferrum automatically selects `qwen3_5.f32-master.gguf-q6-head.q6-f16-mmq-v1`: the main hidden-state boundaries are F32, prepared quantized projections record F16 input/output, the Q6 F16 path is selected by geometry, and KV is F16. The entire graph must not be labeled simply as pure FP16. The 4B llama.cpp log confirms F16 KV and F32 recurrent R/S; this is not used to infer the precision of all its intermediate activations.

For the official 4B BF16 SafeTensors group, Ferrum actually selects `qwen3_5.f16` automatically. Main activation boundaries and KV are F16, and capacity remains 32/2048/2048. Ferrum rows are labeled “BF16 source / F16 execution,” separately from the GGUF group above.

The 4B/9B SafeTensors vLLM baselines use the native BF16 path with `--dtype bfloat16 --kv-cache-dtype auto` and F32 SSM state. Startup preflight confirmed that forcing `--dtype half` disables the fused CUDA GDN path and falls back to Triton. BF16 activations with F16 KV also failed Flash Attention warmup because query/key dtypes differed. Those preflights produced no measured performance results and are not counted as errors in the measured requests. This group aligns official source weights, workload and logical capacity. Ferrum F16 KV and vLLM BF16 KV both use 16-bit storage but have different numerical ranges; identical activation precision or kernel arithmetic is not claimed.

For 30B GPTQ, Ferrum automatically selects `qwen3_moe.f16` with F16 KV. vLLM's `--dtype auto --kv-cache-dtype auto` resolves to F16 activations/KV; startup logs confirm `MarlinLinearKernel` and `MARLIN` WNA16 MoE. Both retain logical capacity 32/2048/2048. This group freezes a new selection using its own tokenizer rather than reusing Qwen3.5 length vectors. Shared F16 types do not imply identical fusion, accumulation or scheduling.

For 35B GPTQ, Ferrum automatically selects `qwen3_5.f16` and F16 KV. vLLM auto resolves to BF16 activations/KV, F32 SSM, fused CUDA GDN decode and the `MARLIN` WNA16 MoE / `MarlinExperts` expert path. Both use the same GPTQ checkpoint, template and logical capacity 32/2048/2048, retaining the differences between their native numerical paths.

For GPT-OSS, Ferrum automatically selects `gpt_oss.f16-mxfp4-bf16`: outer F16 computation, a BF16 MXFP4 expert-compute path and F16 KV. vLLM auto resolves to BF16 activations/KV with `MARLIN` MXFP4 MoE and FlashInfer attention, so the native numerical paths differ. Both use default medium reasoning and a Harmony system date of 2026-11-02 in Asia/Shanghai. The container mounts the same `/etc/localtime` read-only and records `TZ=Asia/Shanghai`; no template changes were made. vLLM's actual reasoning parser is `openai_gptoss`.

</details>

<details>
<summary>Per-repetition latency, errors and sample counts</summary>

## Per-repetition results

For each model/precision, engine and concurrency cell, two rows retain repetitions 1 and 2. The short GPT-OSS default-medium comparison has only one repetition, explicitly labeled.

| Model / precision | Engine | C | Repetition | Successful/offered | Errors/rejections | TTFT P50/P99 ms | TPOT P50/P99 ms | SSE ITL P50/P99 ms | Output tok/s | SLO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 59.18/261.45 | 8.06/10.95 | 5.51/56.44 | 906.7 | Pass |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 8 | 1 | 64/64 | 0/0 | 81.68/370.26 | 12.11/13.62 | 9.23/63.81 | 622.9 | Pass |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 47.12/242.25 | 7.00/8.93 | 5.44/53.29 | 1016.3 | Pass |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 8 | 2 | 64/64 | 0/0 | 71.68/139.30 | 11.89/13.00 | 9.36/65.57 | 645.9 | Pass |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 61.32/813.27 | 11.59/20.04 | 6.67/136.70 | 1103.6 | Fail: ITL |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 16 | 1 | 64/64 | 0/0 | 83.02/387.91 | 20.82/23.15 | 16.79/88.60 | 719.9 | Pass |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 49.47/270.25 | 9.23/11.01 | 6.60/51.99 | 1538.7 | Pass |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 16 | 2 | 64/64 | 0/0 | 76.22/244.57 | 20.39/22.23 | 16.74/82.23 | 738.3 | Pass |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 74.04/500.18 | 10.87/14.95 | 7.17/86.44 | 673.9 | Pass |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 8 | 1 | 64/64 | 0/0 | 80.27/273.54 | 14.03/15.11 | 11.03/65.62 | 543.7 | Pass |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 51.09/268.68 | 9.13/10.89 | 7.11/69.95 | 793.7 | Pass |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 8 | 2 | 64/64 | 0/0 | 70.26/270.60 | 13.62/14.80 | 11.17/71.81 | 556.8 | Pass |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 146.82/1170.01 | 16.05/27.91 | 8.44/182.94 | 804.5 | Fail: ITL |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 16 | 1 | 64/64 | 0/0 | 85.41/642.24 | 22.92/25.25 | 18.71/86.18 | 638.5 | Pass |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 91.98/537.05 | 11.98/16.53 | 8.29/95.13 | 1116.5 | Pass |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 16 | 2 | 64/64 | 0/0 | 69.81/451.34 | 22.11/23.74 | 18.78/75.28 | 670.2 | Pass |
| Qwen3.5-4B BF16 source / F16 execution | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 44.99/242.70 | 8.94/10.55 | 8.00/29.87 | 821.1 | Pass |
| Qwen3.5-4B BF16 source / BF16 execution | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 67.54/167.12 | 11.02/11.75 | 9.19/58.24 | 684.9 | Pass |
| Qwen3.5-4B BF16 source / F16 execution | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.01/98.21 | 8.67/9.10 | 7.96/28.71 | 857.9 | Pass |
| Qwen3.5-4B BF16 source / BF16 execution | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 68.88/112.55 | 10.76/11.78 | 9.27/49.63 | 684.4 | Pass |
| Qwen3.5-4B BF16 source / F16 execution | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 45.54/486.18 | 11.20/14.57 | 8.91/38.00 | 1215.9 | Pass |
| Qwen3.5-4B BF16 source / BF16 execution | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 78.41/173.06 | 12.46/13.85 | 9.68/63.88 | 1144.1 | Pass |
| Qwen3.5-4B BF16 source / F16 execution | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.45/130.76 | 10.23/11.18 | 8.88/31.14 | 1402.6 | Pass |
| Qwen3.5-4B BF16 source / BF16 execution | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 80.28/115.79 | 12.34/13.94 | 9.63/70.85 | 1163.0 | Pass |
| Qwen3.5-9B BF16 source / F16 execution | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 45.31/221.66 | 13.44/14.41 | 12.25/42.86 | 560.1 | Pass |
| Qwen3.5-9B BF16 source / BF16 execution | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 78.13/216.27 | 14.65/16.45 | 12.82/54.36 | 508.9 | Pass |
| Qwen3.5-9B BF16 source / F16 execution | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.11/173.46 | 13.13/13.96 | 12.20/37.22 | 566.5 | Pass |
| Qwen3.5-9B BF16 source / BF16 execution | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 74.28/126.06 | 14.54/15.51 | 12.88/54.35 | 507.5 | Pass |
| Qwen3.5-9B BF16 source / F16 execution | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 49.57/702.47 | 16.98/22.92 | 13.27/72.25 | 802.5 | Pass |
| Qwen3.5-9B BF16 source / BF16 execution | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 92.03/411.67 | 16.59/17.93 | 13.66/76.87 | 843.8 | Pass |
| Qwen3.5-9B BF16 source / F16 execution | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.92/193.56 | 15.09/16.29 | 13.15/44.69 | 953.9 | Pass |
| Qwen3.5-9B BF16 source / BF16 execution | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 94.14/161.30 | 15.98/17.50 | 13.49/77.87 | 883.9 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 46.71/339.73 | 12.18/15.46 | 11.20/44.13 | 601.4 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 45.61/164.53 | 8.93/9.74 | 7.83/40.01 | 814.7 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.56/154.62 | 11.34/12.73 | 9.90/39.86 | 663.3 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 29.57/92.61 | 8.76/10.29 | 7.76/40.81 | 833.7 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 48.00/578.95 | 17.97/21.39 | 14.22/81.90 | 791.2 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 61.37/109.97 | 11.52/12.37 | 9.71/55.88 | 1238.0 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.99/337.25 | 15.56/18.28 | 13.23/50.53 | 916.7 | Pass |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 59.15/93.07 | 11.11/12.28 | 9.48/51.63 | 1291.1 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 47.08/246.26 | 10.92/12.61 | 9.24/42.04 | 693.1 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 103.55/232.90 | 11.75/13.54 | 9.02/89.98 | 627.2 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 44.04/131.38 | 10.15/10.72 | 9.18/35.06 | 741.6 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 100.94/237.23 | 12.00/13.28 | 9.18/89.33 | 624.2 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 48.89/736.40 | 16.29/21.21 | 11.57/75.99 | 890.6 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 137.97/388.13 | 15.50/18.63 | 10.84/136.73 | 921.9 | Fail: ITL |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 50.09/208.18 | 13.58/15.86 | 11.57/48.11 | 1075.2 | Pass |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 115.97/228.37 | 14.79/17.90 | 10.82/93.94 | 972.0 | Pass |
| GPT-OSS-20B MXFP4 / default medium / natural EOS | Ferrum 0.18.3 | 8 | 1 (short comparison) | 32/32 | 0/0 | 54648.74/56304.24 | 0/0 (single event) | N/A (0 intervals) | 72.4 | Fail: TTFT; ITL missing |
| GPT-OSS-20B MXFP4 / default medium / natural EOS | vLLM 0.31.0 | 8 | 1 (short comparison) | 32/32 | 0/0 | 94.75/387.13 | 8.71/8.96 | 8.16/33.00 | 819.8 | Pass |
| GPT-OSS-20B MXFP4 / default medium / natural EOS | Ferrum 0.18.3 | 16 | 1 (short comparison) | 32/32 | 0/0 | 100934.84/109512.78 | 0/0 (single event) | N/A (0 intervals) | 74.3 | Fail: TTFT; ITL missing |
| GPT-OSS-20B MXFP4 / default medium / natural EOS | vLLM 0.31.0 | 16 | 1 (short comparison) | 32/32 | 0/0 | 174.08/183.29 | 10.14/10.38 | 9.47/34.24 | 1429.8 | Pass |

The 4B Ferrum client exited with code 1 because the SLO check failed after completing the measurements, not because requests failed. All 40 warmups and 256 measured requests succeeded. Protocol, invalid-output, HTTP 500, rejection and panic counts in the report are all 0. C16 cannot be described as passing overall merely because repetition 2 passes.

For 4B llama.cpp, all 40 warmups and 256 measured requests also succeeded, all quality/protocol error counts are 0, all four measurements pass the SLOs, and the client exited with code 0. Complete selection records, raw input lengths, server input lengths, output lengths and per-request indices were aligned item by item across the engines. The corresponding normalized evidence SHA256 is `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`.

For 9B, each engine completed all 40 warmups and 256 measured requests successfully, with quality/protocol error counts of 0. Ferrum C16 repetition 1 has visible ITL P99 of 182.94ms; the other three measurements satisfy the SLOs. Exit code 1 likewise comes from the SLO check after measurement completion. llama.cpp satisfies the SLOs in all four measurements and exits with code 0. Complete selections, length vectors and request indices also match for 9B, with the same normalized evidence SHA256 as 4B; the two groups use the same tokenizer and frozen workload.

For 9B, both engines have 64 TTFT and 64 TPOT samples in each of the four measurements. In order of C8 repetitions 1/2, then C16 repetitions 1/2, visible ITL interval counts are 12061/13171/12051/13167 for Ferrum and 12064/13170/12064/13177 for llama.cpp. Counts of requests whose SSE event count differs from usage are 4/11/4/9 and 4/9/4/7, respectively. Detected transport coalescing, missing usage and missing timing are all 0 for both engines.

For official 4B BF16 source weights, each engine completed 256/256 measured requests and 40/40 warmups successfully. All four measurements satisfy the SLOs, both clients exit with code 0, and all quality/protocol error counts are 0. Full selections, the three length vectors and request indices match item by item. The normalized SHA256 is again `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`.

In this group, both engines have 64 TTFT and 64 TPOT samples in each of the four measurements. In the same repetition order, visible ITL interval counts are 12064/13146/12064/13148 for Ferrum and 12054/13145/12050/13151 for vLLM. Counts of requests whose SSE event count differs from usage are 3/8/3/7 and 6/12/7/9, respectively. Detected transport coalescing, missing usage and missing timing are all 0 for both engines.

For official 9B BF16 source weights, Ferrum actually uses `qwen3_5.f16` and F16 KV; vLLM uses BF16 activations/KV, F32 SSM and fused CUDA GDN decode. Both complete 256/256 measured requests and 40/40 warmups successfully, pass all four SLO checks and exit with code 0, with no quality/protocol/request errors. Each measurement has 64 TTFT/TPOT samples. Visible ITL interval counts for Ferrum/vLLM are 12063/13166/12062/13166 and 12022/13160/12028/13159; counts of requests whose SSE event count differs from usage are 3/9/4/9 and 9/10/9/10. Detected transport coalescing, missing usage and missing timing are all 0. Full selections, the three length vectors and request indices match item by item, again with normalized SHA256 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`.

For 30B GPTQ, both engines complete 256/256 measured requests and 40/40 warmups successfully, pass all four SLO checks, and report quality/protocol error counts of 0. Each measurement has 64 TTFT and TPOT samples. In order of C8 repetitions 1/2, then C16 repetitions 1/2, visible ITL interval counts are 12520/12301/12510/12303 for Ferrum and 12527/12297/12543/12300 for vLLM. Counts of requests whose SSE event count differs from usage are 28/26/28/25 and 30/28/27/30, respectively. Detected transport coalescing, missing usage and missing timing are all 0 for both engines.

For 30B, repetition 1 has raw-input/server-input/output totals of 6213/6981/12657 tokens; repetition 2 has 3912/4680/12430, for 25087 output tokens per concurrency cell across both repetitions. Full selections, the three length vectors and request indices match item by item across engines. Normalized evidence SHA256 is `fdffb2ae8cdf0e6b7f88e477d2bf5b526541b91268181913d2b2f7ae29b85146`.

For 35B GPTQ, both engines complete 256/256 measured requests and 40/40 warmups successfully, with no quality/protocol/request errors. Ferrum passes all four SLO checks and exits with code 0. vLLM fails ITL in C16 repetition 1 and exits with code 1 after saving the complete report; the other three measurements pass. Each has 64 TTFT/TPOT samples. In order of C8 repetitions 1/2, then C16 repetitions 1/2, visible ITL interval counts for Ferrum/vLLM are 12018/13190/12019/13181 and 12014/13121/12002/13114; counts of requests whose SSE event count differs from usage are 5/6/5/8 and 11/9/11/11. Detected transport coalescing, missing usage and missing timing are all 0. Full selections, actual input/server-input/output vectors and request indices match item by item, again with normalized SHA256 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`. Separate functional requests return `4` normally on both engines, using 26/2 input/output tokens.

The original GPT-OSS fixed-output attempt stopped after 10 Harmony protocol errors in C8 repetition 1. It has no complete `bench.json`; the controller records client exit code 15, and later repetitions did not run. The error is `terminal control token was followed by duplicate terminal data or trailing garbage`, with missing completion usage. This is an incomplete failed attempt, not a whole-group request success rate or throughput result. In the corresponding source behavior, `ignore_eos` disables stopping at EOS while still allowing Harmony terminal markers to be sampled; subsequent trailing data is then rejected by the strict parser. A separate normal-EOS functional request returns final answer `4`, but does not make the fixed-output workload pass.

In the GPT-OSS default-medium short comparison, Ferrum completes 32/32 measured requests and 2/2 warmups in each cell, with protocol/quality/request error counts of 0. The client exits with code 1 for SLO failure after saving the complete JSON. Each cell has only 32 non-empty text events and 0 consecutive intervals; event counts differ from usage tokens for all 32 requests. There is no missing usage or detected transport coalescing. The 0 in `last_visible_tpot_ms` is the calculation result for a single delivery and is not evidence of per-token latency.

Each cell in this short comparison has the same raw-input/server-input totals of 4440/6584 tokens and selection SHA256 `ceec98e3917466c7dc7347fc0522b2ef1626dd35651747d1018b65b05c562616`. Ferrum C8 produces 15176 tokens, with per-request range 190–512 and mean 474.25; 27/32 requests reach the 512-token ceiling. C16 produces 14999 tokens, range 194–512 and mean 468.72; 26/32 reach the ceiling. Measurement times are 209.551576669s and 201.874913683s. A separate default-medium functional request produces final text `4`, HTTP 200 and a normal termination, using 81/36 input/output tokens.

In the vLLM short comparison, both cells complete 32/32 measured requests and 2/2 warmups, with no quality/protocol/request errors. The client exits with code 0 and both cells pass the SLOs. C8/C16 produce 15064/15115 tokens, with ranges 188–512/117–512 and means 470.75/472.34; 27/32 and 26/32 requests reach the ceiling. Measurement times are 18.375894987s/10.571745072s. Each cell has 32 TTFT/TPOT samples, with 14879/14934 visible text events and 14847/14902 consecutive ITL intervals. Event counts differ from usage for all 32 requests, with no missing usage, missing visible timing or detected transport coalescing. Passing visible-event SLOs therefore does not mean passing strict single-token timing.

Full selections, the 512-token request budget, raw/server input lengths and request indices match item by item across engines. Normalized input-pairing evidence SHA256 is `09f8d53a5bbc40b1804d261dbd09960d8a115763bb63a53d23dc2481a71915e7`; actual output vectors are retained separately under the natural-EOS policy. vLLM's separate functional request also returns final text `4`, HTTP 200 and normal termination, with the same 81/36 input/output tokens. Its functional evidence additionally records 25 reasoning tokens; this single sample is not used to invent a reasoning/final split for the whole performance report.

GPT-OSS throughput counts total `completion_tokens`, including reasoning, rather than final-answer tokens alone. Most requests above reach the output budget and may have truncated answers. The Rust performance report does not separately preserve each request's final text and completeness coverage. Thus neither “32/32 protocol success” nor a correct standalone functional request establishes that all 32 complete final answers are correct. Ceiling hits and output-length distributions remain disclosed; shorter work caused by truncation is not presented as a speed advantage for identical answers.

The two Qwen3.5 selections have different lengths. Repetition 1 has raw-input/server-input/output totals of 8041/8809/12138 tokens per concurrency cell; repetition 2 has 4097/4865/13274. Comparisons across engines must pair the same repetition; differences between repetitions are not attributed solely to warmup or performance variability.

| 4B timing samples | TTFT requests | TPOT requests | Visible ITL intervals | Requests with SSE event count different from usage | Requests with detected transport coalescing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrum C8 repetition 1 | 64 | 64 | 12066 | 4 | 0 |
| llama.cpp C8 repetition 1 | 64 | 64 | 12068 | 2 | 0 |
| Ferrum C8 repetition 2 | 64 | 64 | 13178 | 5 | 0 |
| llama.cpp C8 repetition 2 | 64 | 64 | 13174 | 5 | 0 |
| Ferrum C16 repetition 1 | 64 | 64 | 12068 | 2 | 0 |
| llama.cpp C16 repetition 1 | 64 | 64 | 12063 | 5 | 0 |
| Ferrum C16 repetition 2 | 64 | 64 | 13150 | 5 | 0 |
| llama.cpp C16 repetition 2 | 64 | 64 | 13161 | 5 | 0 |

None of these measurements has missing usage, missing visible timing or inconsistent interval counts. Requests whose event counts differ from token counts remain included in visible SSE ITL and are not treated as strict per-token timing evidence.

Metrics come directly from the Rust client's `client_slo.repeats[]`:

- TTFT: `ttft_ms`, from request start to the first non-empty SSE text update; streamed reasoning counts here when the server emits it.
- TPOT: `last_visible_tpot_ms`, time from the first to the last visible update divided by `usage output tokens - 1`; this excludes waiting for trailing control messages.
- ITL: `visible_sse_itl_ms`, client-observed intervals between consecutive non-empty SSE text updates, including streamed content/reasoning. Role-only, empty and finish-only messages are excluded. Transport-coalescing stalls and their counters are retained.
- Output throughput: `successful_output_throughput_tps`, server usage output tokens from successful requests divided by the whole repetition's measurement time. Time spent on failed work is not subtracted.
- Status and errors: `status`, `completed_requests`, `offered_requests`, `errors`, `rejected_requests`. Rejections include HTTP 429/503; other HTTP errors count toward errors.

TTFT/TPOT request sample counts, ITL interval sample counts, warmup outcomes, transport-coalescing counts, missing usage/timing, invalid outputs, protocol errors and HTTP 500 are also retained. Strict single-token ITL is diagnostic only and does not replace visible SSE ITL.

Averages of per-repetition P50/P99 values are not labeled as pooled quantiles. By default, each repetition has only 64 TTFT/TPOT request samples; the GPT-OSS short comparison has only 32. These results do not establish confidence intervals, and small differences do not support claims of stable superiority. Missing metrics or output-validity failures do not count as an SLO pass.

</details>

<details>
<summary>VRAM, host memory and validity evidence</summary>

## Resource and validity evidence

| Model / engine | Sampling window | Sampling interval and coverage | Peak VRAM | RSS scope and peak | Unavailable evidence |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_K_M / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 253.010s | Configured 250ms; GPU 978, Host 1014 samples; sampling errors 0; maximum gap GPU 1804ms, Host 252ms | Whole GPU 6.387GiB | Ferrum process sampled window peak 1.862GiB | Target-process VRAM unavailable in WSL |
| Qwen3.5-4B Q4_K_M / llama.cpp | Entire C8+C16 sweep, including preparation/warmup, 177.505s | Configured 250ms; GPU 642, Host 712 samples; GPU query failures 4, Host errors 0; maximum gap GPU 2105ms, Host 259ms | Observed whole-GPU peak 8.087GiB | llama.cpp process sampled window peak 1.240GiB | Target-process VRAM unavailable in WSL; GPU sampling gaps |
| Qwen3.5-9B Q4_K_M / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 110.755s | Configured 250ms; GPU 361, Host 445 samples; GPU query failures 3, Host errors 0; maximum gap GPU 2209ms, Host 251ms | Observed whole-GPU peak 9.238GiB | Ferrum process sampled window peak 1.924GiB | Target-process VRAM unavailable in WSL; GPU sampling gaps |
| Qwen3.5-9B Q4_K_M / llama.cpp | Entire C8+C16 sweep, including preparation/warmup, 156.766s | Configured 250ms; GPU 499, Host 629 samples; GPU query failures 8, Host errors 0; maximum gap GPU 2172ms, Host 259ms | Observed whole-GPU peak 10.304GiB | llama.cpp process sampled window peak 1.300GiB | Target-process VRAM unavailable in WSL; GPU sampling gaps |
| Qwen3.5-4B BF16 source / F16 execution / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 96.754s | Configured 250ms; GPU 382, Host 389 samples; sampling errors 0; maximum gap GPU 1066ms, Host 252ms | Observed whole-GPU peak 11.104GiB | Ferrum process sampled window peak 1.455GiB | Target-process VRAM unavailable in WSL |
| Qwen3.5-4B BF16 source / BF16 execution / vLLM | Entire C8+C16 sweep, including preparation/warmup, 156.505s | Configured 250ms; GPU 623, Host 627 samples; sampling errors 0; maximum gap GPU 836ms, Host 271ms | Observed whole-GPU peak 29.384GiB | EngineCore worker sampled window peak 3.253GiB | Target-process VRAM unavailable in WSL; worker RSS excludes API/other processes |
| Qwen3.5-9B BF16 source / F16 execution / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 92.005s | Configured 250ms; GPU 342, Host 370 samples; sampling errors 0; maximum gap GPU 1877ms, Host 251ms | Observed whole-GPU peak 19.987GiB | Ferrum process sampled window peak 1.396GiB | Target-process VRAM unavailable in WSL |
| Qwen3.5-9B BF16 source / BF16 execution / vLLM | Entire C8+C16 sweep, including preparation/warmup, 98.004s | Configured 250ms; GPU 356, Host 394 samples; GPU query failures 1, Host errors 0; maximum gap GPU 2212ms, Host 265ms | Observed whole-GPU peak 28.675GiB | EngineCore worker sampled window peak 3.311GiB | Target-process VRAM unavailable in WSL; GPU sampling gaps; worker RSS excludes other processes |
| Qwen3-30B-A3B GPTQ INT4 / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 170.505s | Configured 250ms; GPU 678, Host 684 samples; sampling errors 0; maximum gap GPU 832ms, Host 252ms | Observed whole-GPU peak 18.668GiB | Ferrum process sampled window peak 1.711GiB | Target-process VRAM unavailable in WSL |
| Qwen3-30B-A3B GPTQ INT4 / vLLM | Entire C8+C16 sweep, including preparation/warmup, 141.755s | Configured 250ms; GPU 558, Host 569 samples; sampling errors 0; maximum gap GPU 1201ms, Host 259ms | Observed whole-GPU peak 30.008GiB | EngineCore worker sampled window peak 3.130GiB | Target-process VRAM unavailable in WSL; worker RSS excludes API/other processes |
| Qwen3.5-35B-A3B GPTQ INT4 / Ferrum | Entire C8+C16 sweep, including preparation/warmup, 75.005s | Configured 250ms; GPU 287, Host 302 samples; sampling errors 0; maximum gap GPU 1244ms, Host 275ms | Observed whole-GPU peak 23.630GiB | Ferrum process sampled window peak 3.066GiB | Target-process VRAM unavailable in WSL |
| Qwen3.5-35B-A3B GPTQ INT4 / vLLM | Entire C8+C16 sweep, including preparation/warmup, 81.005s | Configured 250ms; GPU 284, Host 326 samples; GPU query failures 1, Host errors 0; maximum gap GPU 2186ms, Host 275ms | Observed whole-GPU peak 29.802GiB | EngineCore worker sampled window peak 3.494GiB | Target-process VRAM unavailable in WSL; GPU sampling gaps; worker RSS excludes other processes |
| GPT-OSS-20B default medium / Ferrum | Entire short C8+C16 sweep, including preparation/warmup, 452.509s | Configured 250ms; GPU 1804, Host 1812 samples; sampling errors 0; maximum gap GPU 1273ms, Host 269ms | Observed whole-GPU peak 15.990GiB | Ferrum process sampled window peak 1.389GiB | Target-process VRAM unavailable in WSL |
| GPT-OSS-20B default medium / vLLM | Entire short C8+C16 sweep, including preparation/warmup, 37.512s | Configured 250ms; GPU 147, Host 152 samples; sampling errors 0; maximum gap GPU 744ms, Host 271ms | Observed whole-GPU peak 29.937GiB | EngineCore worker sampled window peak 3.145GiB | Target-process VRAM unavailable in WSL; worker RSS excludes API/other processes |

The existing Rust sampler is reused at a configured 250ms interval. When one monitor covers the entire C8/C16 sweep, peaks are explicitly labeled as whole-sweep values including warmup, not separate peaks for each concurrency cell. Where WSL provides only whole-GPU VRAM, the column and explanation retain that scope. vLLM's API-parent RSS does not represent all workers. Sampling failures, maximum gaps, other resident processes and unavailable metrics are all disclosed.

The test host has an RTX 5090 with driver 610.88, reporting 32607MiB total VRAM and a 600W power limit. The CPU is a Core Ultra 9 285K with 24 visible logical processors, running Ubuntu 26.04.1 / WSL2. Before the 4B functional check, whole-GPU usage was 1181MiB and GPU utilization was 0%; existing service processes were retained. Later whole-GPU VRAM peaks include those resident allocations; not all GPU memory is claimed to belong to the measured process.

The last 4B Ferrum `VmHWM` was 3.389GiB. This is the process-lifetime high-water mark, including allocations before the sampling window, and differs from the window RSS peak in the table. The largest VRAM sampling gap reached 1.804s, so higher instantaneous peaks within gaps cannot be excluded. Raw byte peaks are 6857687040 for the whole GPU, 1999106048 for window RSS and 3639144448 for lifetime RSS.

The 4B llama.cpp lifetime `VmHWM` is 2.919GiB and is also not treated as a window peak. Its GPU queries failed 4 times, so the table's 8.087GiB is only the peak of successful samples, not proof that no higher instantaneous VRAM usage occurred.

For 9B, Ferrum lifetime `VmHWM` is 6.170GiB and llama.cpp is 5.655GiB. The sampling-window scope and GPU-query gaps likewise do not replace continuous memory observation from process startup.

For 4B BF16 source / F16 execution, Ferrum lifetime `VmHWM` is 8.663GiB, including weight-loading allocations before the window. The table's 1.455GiB is only the sampled RSS peak within the measurement window.

For 9B BF16 source / F16 execution, Ferrum lifetime `VmHWM` is 17.589GiB and the window RSS peak is 1.396GiB; lifetime usage includes loading allocations.

For official 9B BF16 source weights, vLLM EngineCore lifetime `VmHWM` is 6.627GiB. The separate container cgroup `memory.peak` is 11133145088 bytes (10.369GiB), with OOM/kill counts of 0. KV memory available under its 0.9 utilization reservation is 9.53GiB, with a pool capacity of 172909 tokens. There is one GPU-query gap.

The 4B vLLM EngineCore lifetime `VmHWM` is 6.451GiB. The separate new container's cgroup `memory.peak` is 8023486464 bytes (7.472GiB), including loading, warmup, all workers and page cache; this is not process RSS. OOM/kill counts in `memory.events` are 0. vLLM reserves KV at 0.9 utilization; startup reports 19.07GiB available KV memory and pool capacity 346112 tokens, with fused CUDA GDN decode. Its 29.384GiB whole-GPU usage includes this reservation. The VRAM difference from Ferrum must not be directly interpreted as a memory-efficiency difference for the same active data.

The 30B Ferrum lifetime `VmHWM` is 16.670GiB, including weight loading. Window RSS of 1.711GiB does not represent the full process-lifetime peak.

The 30B vLLM EngineCore lifetime `VmHWM` is 17.268GiB. Container cgroup `memory.peak` is 22336495616 bytes (20.802GiB), including all workers, loading and page cache, with OOM/kill counts of 0. With KV reserved at 0.9 utilization, startup reports 12.0GiB available KV memory and pool capacity 131056 tokens. The limits remain fixed at 32 active sequences and 2048 per request.

For 35B GPTQ, Ferrum lifetime `VmHWM` is 22.367GiB, including weight loading. The sampled window RSS peak is 3.066GiB; the two are recorded separately.

For 35B GPTQ, vLLM EngineCore lifetime `VmHWM` is 4.531GiB. The separate container cgroup `memory.peak` is 8915484672 bytes (8.303GiB), with OOM/kill counts of 0. KV memory available under the 0.9 utilization reservation is 6.87GiB, with pool capacity 139673 tokens. GPU-query gaps limit conclusions about instantaneous peaks.

GPT-OSS Ferrum lifetime `VmHWM` is 13.482GiB, including loading and earlier separate diagnostics. Active/queued counts were confirmed as 0 before the default-medium short comparison. The window RSS peak is only 1.389GiB; the reused service process's lifetime high-water mark is not presented as this comparison's peak.

GPT-OSS vLLM EngineCore lifetime `VmHWM` is 5.925GiB. The separate container cgroup `memory.peak` is 8277020672 bytes (7.709GiB), with OOM/kill counts of 0. KV memory available under the 0.9 utilization reservation is 12.65GiB, with pool capacity 275324 tokens. Whole-GPU VRAM likewise includes reserved pools and other resident allocations, so it cannot be directly compared with Ferrum's on-demand peak to derive a memory-efficiency ratio.

Functional checks are separate from performance: first verify model loading and a normal answer, then inspect HTTP/SSE, completion markers, usage and lengths for measured requests. Protocol success is not a general answer-quality evaluation. This report makes no model-accuracy claim.

</details>

<details>
<summary>Versions, reproduction commands and raw report SHA256 values</summary>

## Reproduction

Both the Ferrum server and benchmark client are the official 0.18.3 release, source commit `e991c6955c3022f80d7e96f9d16d9306314054cf`, with Linux SM120 CLI SHA256 `76094c69c77d0a1e6d675900c0a91b28bdee3d4237c930193abdac709ebee087`. llama.cpp is b11429 (`d81235049384534c167caea52b85a694f6103d14`). vLLM is 0.31.0 (`db9527a46873454610df6dbedf79a36d6bf1a7f6`), with local cached image ID `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`. The recorded image tag is `vllm/vllm-openai:v0.31.0`, but the available RepoDigest points only to a local cache registry. A publicly pullable registry digest was not archived, so this image ID must not be turned into an official download URL. Reproduction should verify the actual version, source revision and image identity.

GGUF server command templates follow. For 4B/9B, replace only the corresponding model file, semantic directory and output file. Each engine is started, measured and stopped separately; they do not share GPU execution concurrently.

```sh
ferrum serve /path/to/model-Q4_K_M.gguf \
  --semantic-source /path/to/official-model-metadata \
  --tokenizer-source /path/to/official-model-metadata \
  --backend cuda --disable-thinking \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --kv-dtype fp16 --gpu-memory-utilization 0.9 --disable-prefix-cache \
  --host 127.0.0.1 --port 19081 --served-model-name comparison \
  --effective-config-json /path/to/effective.json

llama-server --model /path/to/model-Q4_K_M.gguf --alias comparison \
  --host 127.0.0.1 --port 19082 \
  --ctx-size 65536 --parallel 32 --batch-size 2048 --ubatch-size 512 \
  --cache-type-k f16 --cache-type-v f16 --n-gpu-layers all --flash-attn on \
  --kv-unified --kv-unified-per-slot 2048 --ctx-checkpoints 0 \
  --no-cache-prompt --cache-ram 0 --no-cache-idle-slots --no-context-shift \
  --chat-template-file /path/to/official-model-metadata/chat_template.jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --metrics --fit off --log-verbosity 4
```

For the official Qwen3.5 SafeTensors groups, Ferrum keeps the capacity and cache parameters above, replacing the model, `--semantic-source` and `--tokenizer-source` with the same complete SafeTensors directory. The numerical profile remains at its default auto setting. The corresponding vLLM command follows; its native BF16 numerical difference is disclosed in the main table.

```sh
vllm serve /path/to/official-safetensors \
  --tokenizer /path/to/official-safetensors --served-model-name comparison \
  --host 127.0.0.1 --port 19083 \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching --language-model-only \
  --mamba-ssm-cache-dtype float32 --generation-config vllm \
  --chat-template /path/to/official-safetensors/chat_template.jinja
```

For 30B GPTQ, Ferrum likewise replaces the three model-directory arguments and retains the remaining parameters. vLLM uses the fixed capacity/cache parameters above, changes to `--dtype auto --kv-cache-dtype auto`, and removes Qwen3.5-specific `--mamba-ssm-cache-dtype`, `--language-model-only` and the external `--chat-template`. It uses the tokenizer template supplied in that GPTQ source directory. Its actual activation/KV types are F16.

For 35B GPTQ, Ferrum replaces the three directories with the complete checkpoint and keeps the other parameters. vLLM follows the Qwen3.5 command, including `--language-model-only --mamba-ssm-cache-dtype float32` and that directory's `chat_template.jinja`, changing only `--dtype` to `auto`; it resolves to BF16 activations/KV.

For GPT-OSS, Ferrum uses the complete MXFP4 model directory and removes `--disable-thinking`, retaining the other capacity/cache parameters. vLLM uses the 30B fixed capacity parameters and `--dtype auto --kv-cache-dtype auto`, adding `--reasoning-parser openai_gptoss` and using internal Harmony rendering. Container options `--env TZ=Asia/Shanghai --mount type=bind,source=/etc/localtime,target=/etc/localtime,readonly` align its date with the Ferrum host. Neither engine overrides the default reasoning effort.

The client template follows. Replace the model name, address, tokenizer, dataset and output paths with the saved commands for the corresponding group. Each engine runs one sweep; measurements are not repeated merely to generate Markdown.

```sh
ferrum bench-serve \
  --base-url http://127.0.0.1:19081 --model comparison \
  --tokenizer /path/to/model-tokenizer --target-backend cuda \
  --dataset sharegpt --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-prepared-cache-dir /path/to/same-model-prepared-cache \
  --seed 42 --concurrency-sweep 8,16 \
  --num-prompts 64 --warmup-requests 10 --n-repeats 2 \
  --sharegpt-selection-pool-size 74 \
  --sharegpt-min-input-tokens 4 --sharegpt-max-input-tokens 1024 \
  --sharegpt-min-output-tokens 128 --sharegpt-max-output-tokens 256 \
  --sharegpt-max-total-tokens 2048 --sharegpt-chat-template-reserve-tokens 32 \
  --ignore-eos --temperature 0 --repetition-penalty 1 \
  --latency-slo ttft:2000,tpot:50,itl:100 \
  --latency-slo-fail-on-violation --fail-on-error --max-error-rate 0 \
  --timeout 120 --output json --out /path/to/new-report.json
```

Where supported by the template, add `--enable-thinking false` identically on both engines. The final GPT-OSS default-policy short comparison omits that switch and `--reasoning-effort`, adds `--sharegpt-fixed-output-tokens 512`, removes `--ignore-eos`, and changes to `--num-prompts 32 --warmup-requests 2 --n-repeats 1 --sharegpt-selection-pool-size 34`; its server does not add `--disable-thinking`. Use a separate report path; the Rust client validates the cache against the request policy. This CLI argument fixes the request ceiling. The assertion that actual output must equal 512 runs only when `ignore_eos` is enabled. When an SLO fails, the client saves the complete report before returning nonzero; slower results must not be discarded for that reason. Raw JSON, server commands/effective config, weight and selection identities, logs and sampler receipts are kept outside the repository in the `20261101-popular-model-comparison` evidence directory.

Raw reports and sampling records are kept outside the repository. The SHA256 values below identify the actual originals from this comparison. This page provides reproduction settings and results; raw logs and weights are not separately published.

| Original artifact | SHA256 |
| --- | --- |
| `4b-ferrum-bench.json` | `6c59b6e64f0a0c1eef3e8d41cf58ec1fe2a70cca26048e66bcadcc362cfd25fe` |
| `4b-ferrum-resource.summary.json` | `4c5dce33d6f332f14643d7786f6b396cc018377cc595985f820b744cc8c5266e` |
| `4b-ferrum-evidence.tar.gz` (report, effective config, logs and resource samples) | `92cbbbe7339cf01b3a17c374ca8d045bdcdffbce8f6cae977ae53957e8821b6f` |
| `4b-llama-unified-bench.json` | `cafec6b3474bf748466780846a0888165d831c38b3a9be05c93c9097a1328890` |
| `4b-llama-unified-resource.summary.json` | `a7f9b5f023ed75e5f6fc887373d790ae75210b9fd11a07cd171450865f79b458` |
| `4b-llama-unified-evidence.tar.gz` (primary-baseline report, logs and resource samples) | `178349fdf04105ff541537a0a0fe2e7b2d0ef1c7a824624feaaa45c97cc79c41` |
| `9b-ferrum-bench.json` | `a02a401b59044736c049bcd7b92eb66fd9d70fc66007cbebcbeb45ced8c475be` |
| `9b-ferrum-resource.summary.json` | `0872128eb2c73b592cc436e873ef21cd73187b3072ed1c1cbd75046d8ea2a55a` |
| `9b-ferrum-evidence.tar.gz` (report, configuration, resources and functional checks) | `8b82fd67b566e4006fa9c1d48cd3e9fd60a7e7d2011016d860e29ec8213ae6ce` |
| `9b-llama-unified-bench.json` | `aeaaacbfb07c460b9790d3f5c78cd389e51eb4dbd7d8b1a43661cfddb58e1216` |
| `9b-llama-unified-resource.summary.json` | `2b2050d0f5a813be97aed1a9e8619e77a6ecec7161b8f0147af126a2b57fa174` |
| `9b-llama-unified-evidence.tar.gz` (primary-baseline report, logs and resource samples) | `d1f107c52bfff9c75aeda156066f944f507dcecf9dbd145c0b479fc6baccc4b7` |
| `4b-f16-ferrum-bench.json` | `e482c46cf526782e769f1b90e4e023d639ae485dbe53b87ef9d4914a001a3992` |
| `4b-f16-ferrum-resource.summary.json` | `c14a071f5e99dfd16ac8d3714cc83c312ad0a605cb263ee3e14988d9c2a1afcf` |
| `4b-f16-ferrum-evidence.tar.gz` (report, configuration, resources and functional checks) | `af21c9488b072f2e9889f6561eb4feba575c95af1aa531a08295812f5affe9ea` |
| `4b-native-vllm-bench.json` | `8d0977a59fbb0739d8ecd87f9b27a14c60a24842d63d4fff80f2cd791e3bd4fe` |
| `4b-native-vllm-resource.summary.json` | `b93a68363459d1f31dec4982316e1c162896bcf1c33362a82af1104a2f0912d7` |
| `4b-native-vllm-evidence.tar.gz` (measured primary-baseline report, configuration, resources and functional checks) | `c45d8406646fda6cfadcd51bf83d5a2dc705a87ce1cb8d560234f71f5b1f9b35` |
| `9b-f16-ferrum-bench.json` | `b5a29267ad3119a366c02f1346a5de12b62575105f01c473cabb9a6a687c3a93` |
| `9b-f16-ferrum-monitor.summary.json` | `4e6185b295153a7e174aaa402b94699316ba58987fc3d846e154441e678394a9` |
| `9b-f16-ferrum-evidence.tar.gz` (report, configuration, resources and functional checks) | `c24af9939bf2581f822ddd60ce1faf48d6ae2121102f6e24949202c17b712b02` |
| `9b-native-vllm-bench.json` | `2e8f987a51bdf52ef09e24fa89f2b7146612e7da4d4f2fd292ff0e8e5bfe55c9` |
| `9b-native-vllm-monitor.summary.json` | `d5f54a31f1a75521ac0cc470aeb94fffb068e3beaf08f1d52522bbf99aed7b78` |
| `9b-native-vllm-evidence.tar.gz` (report, configuration, resources and functional checks) | `92d854086f5ea44b0abe1fb8598ff5ca38f4e360967d21d2a7e34c1b9d9e9243` |
| `qwen3-30b-gptq-ferrum-bench.json` | `1d178440ce20401ffcf47ad192e28d3fa68232b78fa0bd1504bf5f94d1299733` |
| `qwen3-30b-gptq-ferrum-resource.summary.json` | `689b70577162cfa9643a0239bd0cf0ef64ae025673c592b62ad582b12d54083f` |
| `qwen3-30b-gptq-ferrum-evidence.tar.gz` (report, configuration, resources and functional checks) | `faf8e50334444db5ff527f7bc658785ea160e1209f4f409da3c889edbe7bedd9` |
| `qwen3-30b-gptq-vllm-bench.json` | `4d419202610b611243dd2bd3b2194ed652b9118f91f6f904bff59482dd742a03` |
| `qwen3-30b-gptq-vllm-resource.summary.json` | `f50f921df94cf62bc05e1304510a1049141add4c89af754be8bc019fce90860d` |
| `qwen3-30b-gptq-vllm-evidence.tar.gz` (report, configuration, resources and functional checks) | `9560230806638acfd29cc3999620940de964405406c9549d2c1b4260fc8ec5b7` |
| `qwen35-35b-gptq-ferrum-bench.json` | `883dffab4471b7347e6cdcef917b03946648fb2b2d3eb400cdcce4bc6164d2c5` |
| `qwen35-35b-gptq-ferrum-monitor.summary.json` | `08686c3df5223b59446cf4b3fd58522803cddcb51dfb5a0be1b9659edf90934e` |
| `qwen35-35b-gptq-ferrum-evidence.tar.gz` (report, configuration, resources and functional checks) | `a958e8c110be9bbaad730d7e6a5c7e82a0113bd111964cf2c307b47ab177c5a1` |
| `qwen35-35b-gptq-vllm-bench.json` | `086a5db4678f8e21abfb99f8bb63d74123dc6d2dc23e229dc0ead88e9ad5b544` |
| `qwen35-35b-gptq-vllm-monitor.summary.json` | `7f0e677666e17b0c5dbef5c8e9ca37690d5194368ccdd84d0688faf21d8e33f9` |
| `qwen35-35b-gptq-vllm-evidence.tar.gz` (report, configuration, resources and functional checks) | `39a35e6700dfad57943d762b0ee13e25a0995e7409e7dae25ab23c11778bfe8d` |
| `gpt-oss-20b-ferrum-fixed-failure-evidence.tar.gz` (interruption record, failure-log snapshot, configuration and separate functional check) | `55d261542160a061f188a36f234f2b5aa709e3a8ab9c6d57267a4fe64ad39071` |
| `gpt-oss-20b-native-eos-ferrum-evidence.tar.gz` (incomplete low natural-EOS diagnostic, interruption record and log snapshot) | `04415aa3a6ef258f1e897bdddb3e0cc49e72333bb42ea5fe4b7014e576078ab6` |
| `gpt-oss-20b-default-medium-ferrum-bench.json` | `4ccf52e8a8311e91373ef028948fb2683d8a1a12d83d0d5ff1e3d97332a8edcc` |
| `gpt-oss-20b-default-medium-ferrum-monitor.summary.json` | `ca71c429f44d28ab170911d0f7ac13d877a1e92da1470bbd278b58da34ad88ae` |
| `gpt-oss-20b-default-medium-ferrum-evidence.tar.gz` (complete short-comparison report, configuration, resources and functional checks) | `1660ea98a399f06366f8e76ac385bdaf9fea60e0f69a223e8360c4626faa9cd3` |
| `gpt-oss-20b-default-medium-vllm-bench.json` | `739474333c8a43b0ba08513e6080ef61800455f8150e474ec7886fa4ed32c16b` |
| `gpt-oss-20b-default-medium-vllm-monitor.summary.json` | `a50f73a450619d1e0845a6b6517c3e71e315f5bebe0e02c954503a6964535ca6` |
| `gpt-oss-20b-default-medium-vllm-evidence.tar.gz` (complete short-comparison report, configuration, resources and functional checks) | `a684d6f62c9e62aa405e353bc5385c3d651b97463b7d0beb3faf6e1d039538e7` |

</details>
