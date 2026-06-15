# ACTIVE Vast instances — MUST be destroyed before goal completion

| instance | config | offer $/hr | created | purpose | status |
|---|---|---|---|---|---|
| ~~40700470~~ | 1x RTX 4090, 220GB | 0.351 | 2026-06-12 ~20:00 | W1 CUDA gate batch (~9h) | DESTROYED 2026-06-13 ~04:40 |
| ~~40700477~~ | 2x RTX 4090 | 0.589 | | broken host (raw cuInit=999) | DESTROYED 2026-06-12 ~20:30 |
| ~~40703915~~ | 2x RTX 4090 | 0.336 | | host never booted (18min None) | DESTROYED |
| ~~40704262~~ | 2x RTX 4090, 160GB | 0.802 | 2026-06-12 ~21:50 | 70B dual lane (~5h) | DESTROYED 2026-06-13 ~02:50 |

Destroy: `curl -X DELETE "https://console.vast.ai/api/v0/instances/{id}/?api_key=$VAST_API_KEY"`
Verify zero: instances list must return 0 before declaring the goal done.

**2026-06-13 04:40 — API verified: 0 instances remaining.** Session GPU spend ~= $7.

| ~~40742315~~ | 1x RTX 4090 | 0.372 | | Quebec host killed cargo/nvcc twice | DESTROYED ~06:35 |
| ~~40744863~~ | 1x RTX 4090 | 0.136 | | Argentina host never booted (cluster issue) | DESTROYED ~06:50 |
| ~~40745272~~ | 1x RTX 4090 | 0.376 | | Korea host: session-scope process reaping killed builds 3x | DESTROYED ~08:20 |
| ~~40748763~~ | 1x RTX 4090 | 0.336 | | UK host never booted | DESTROYED ~08:40 |
| ~~40749191~~ | 1x RTX 4090 | 0.349 | | Iceland host stuck loading 18min | DESTROYED ~09:15 |
| ~~40749797~~ | 1x RTX 4090, 150GB | 0.376 | 2026-06-13 ~09:15 | Taiwan: host `docker_build() error writing dockerfile` | DESTROYED ~11:35 |
| ~~40750195~~ | 1x RTX 4090, 150GB | 0.136 | 2026-06-13 ~09:45 | Argentina racer: never booted (None ~25min) | DESTROYED ~11:50 |
| ~~40751023~~ | 1x RTX 4090, 120GB | 0.376 | 2026-06-13 ~11:40 | FINAL micro-session winner (Iceland host 1647, pytorch image cache-hit, booted <2min). cuInit=804 fixed by removing container compat libcuda 550 (GeForce unsupported by compat libs); host driver 535/CUDA 12.4 OK. Gotchas hit: rsproxy.cn TLS MITM-broken (switch to direct crates.io), hf xet/hf_transfer stalled to 0 MB/s (plain-HTTP relaunch unlocked full 3.6Gbps). Delivered clean L5 trio | DESTROYED 2026-06-13 ~15:25 |

**2026-06-13 15:25 — W1 COMPLETE. API verified: 0 instances remaining (ZERO-VERIFIED).**
| ~~40770078~~ | 1x RTX 4090, 120GB | 0.402 | 2026-06-13 ~17:50 | W2 Gemma3-27B gate session (Iceland host 1647). Reached run 4: fixed 3 issues (smoke kv-pin rejecting 2048-tok default; 16-kv-head capacity math; CUDA scale_inplace flipping f16 residual→f32). Destroyed mid-run-4 on user "pod 销毁停止". 27B GPTQ loads + serves single requests on CUDA; full L2-L5 gate not yet captured. | DESTROYED 2026-06-13 ~18:30 |

**2026-06-13 ~18:30 — W2 pod stopped by user. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40798977~~ | 1x RTX 4090, 120GB | 0.402 | 2026-06-13 ~19:?? | W2 Gemma3-27B CUDA gate retry (Iceland host 1647). Ferrum build + GPTQ/GGUF downloads + llama.cpp build complete; compat libcuda fixed; CUDA GPTQ loads/serves, but smoke correctness fails 0/10 known-answer with empty content/length, tools 400, schema 500. Artifacts copied to `artifacts/w2_gemma3_cuda_failure_2026-06-13/`; no L5/perf run by first-fail rule. | DESTROYED 2026-06-13 ~21:16 |

**2026-06-13 ~21:16 — W2 failure artifacts collected. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40806710~~ | 1x RTX 4090, 120GB | 0.402 | 2026-06-13 ~22:?? | W2 Gemma3-27B CUDA retry after local sampling-mask fix + smoke top-k diagnostics. Offer 9021757, Iceland host 1647, verified, driver 580.159.03, CUDA max 13.0. Ferrum build + GPTQ/GGUF downloads complete; early smoke reached first known-answer prefill and proved logits all NaN (`262208/262208`), so correctness first-stop before L5/perf. Artifacts copied to `artifacts/w2_gemma3_cuda_nan_logits_2026-06-13/`. | DESTROYED 2026-06-13 ~22:02 |

**2026-06-13 ~22:02 — W2 NaN-logits failure artifacts collected. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40812709~~ | 1x RTX 4090, 90GB | 0.356 | 2026-06-13 ~18:54 local | W2 desc_act CUDA micro-diagnostic attempt. Offer 40812254, Vietnam host 55116, driver 580.95.05. Stuck at `actual_status=loading` until startup timeout; no SSH, no diagnostic run. Destroyed and API verified zero. | DESTROYED 2026-06-13 ~19:03 local |

**2026-06-13 ~19:03 local — W2 micro-diagnostic attempt 40812709 boot-timeout. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40813345~~ | 1x RTX 4090, 90GB | 0.402 | 2026-06-13 ~19:04 local | W2 desc_act CUDA micro-diagnostic retry. Offer 7656399, Iceland host 1647, driver 565.77/CUDA max 12.7. Vast reported running/onstart success, but SSH kept rejecting the local key; no diagnostic run. Destroyed and API verified zero. | DESTROYED 2026-06-13 ~19:10 local |

**2026-06-13 ~19:10 local — W2 micro-diagnostic attempt 40813345 SSH bootstrap failure. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40813765~~ | 1x RTX 4090, 90GB | 0.402 | 2026-06-13 ~19:12 local | W2 desc_act CUDA micro-diagnostic retry using documented Vast `runtype=ssh` + attach-key API. Offer 20236601, Iceland host 1647, driver 570.211/CUDA max 12.8. Vast reached running, but proxy SSH port forwarding failed and direct SSH rejected the attached key; no diagnostic run. Logs preserved in `artifacts/w2_desc_act_cuda_parity_2026-06-13/`. Destroyed and API verified zero. | DESTROYED 2026-06-13 ~19:17 local |

**2026-06-13 ~19:17 local — W2 micro-diagnostic attempt 40813765 SSH/proxy failure. API verified: 0 instances remaining (ZERO-VERIFIED).**

| ~~40814425~~ | 1x RTX 4090, 90GB | 0.401 | 2026-06-13 ~19:20 local | W2 desc_act CUDA micro-diagnostic using Vast `runtype=ssh_direct` to bypass failed proxy forwarding. Offer 40215008, Ukraine host 103274, driver 580.95/CUDA max 13.0. Instance reached running and direct port opened, but root/ubuntu/vastai/user all rejected the associated public key; no diagnostic run. Destroyed and API verified zero. | DESTROYED 2026-06-13 ~19:25 local |

**2026-06-13 ~19:25 local — W2 micro-diagnostic attempt 40814425 ssh_direct key failure. API verified: 0 instances remaining (ZERO-VERIFIED).**

| 40826362 | 1x RTX 4090, 120GB request on offer 7657220 (Iceland host 1647) | 0.402 | 2026-06-13 ~?? local | W2 Gemma3 CUDA GPTQ native-CUDA continuation. Correctness fixed with host/F32 residual shadow: smoke PASS, known-answer 10/10, L3/L4 PASS. Full W2 gate then stopped at L5: c=1/4/16 completed 100/100/100 zero errors; required c=32 OOM at kv-capacity 448. Reused stopped/cache-retained instance on 2026-06-14 for c=32 minimal retry after bench prompt exact-length fix: `kv-capacity=400/max-num-seqs=32` starts and enters c=32 repeat 1/3, then runtime CUDA OOM. Reused again for allocator/KV diagnostics: latest evidence shows the failing c=32 allocation is contiguous fp16 KV cache layer buffer (`400 * 16 * 128` elements); kv396 still OOM, kv384 context-fails, paged fp16 still OOM, and int8 KV correctness fails. Reused again for product-path KV-hint diagnostics: unified prefill hook and actual-written-token hint reduced the failing allocation from 400 tokens to 393 tokens, but c=32 still OOMs. Reused again for admission-cap diagnostics: client c=32 with typed product CLI `--max-num-seqs 31` and `--max-num-seqs 30` both still OOM at the same 393-token KV allocation; scheduler short trace recorded first iteration `returning_batch=4`. Final reuse completed W2 L5 c=32 with client concurrency 32 and active admission cap 16: full 100x3 repeats completed zero-error; llama.cpp same-card ratio was 0.500260 PASS. Artifacts copied to `artifacts/w2_host_shadow_cuda_2026-06-13/`, `artifacts/w2_gemma3_cuda_host_shadow_l5_fail_2026-06-13/`, `artifacts/w2_c32_prompt_exact_oom_2026-06-14/`, `artifacts/w2_c32_alloc_diag_oom_2026-06-14/`, `artifacts/w2_c32_kv_allocator_diagnostics_2026-06-14/`, `artifacts/w2_c32_kv_hint_diagnostics_2026-06-14/`, `artifacts/w2_c32_admission_cap_diagnostics_2026-06-14/`, and `artifacts/w2_c32_admission16_l5_pass_2026-06-14/`. | STOPPED 2026-06-14 07:57 CST; API verified `cur_state=stopped`, `actual_status=exited`, cache retained |

**2026-06-14 02:21 CST — W2 40826362 stopped, not destroyed, to avoid reinstall/cache churn. API verified `actual_status=exited`; no GPU compute process was running before stop.**

**2026-06-14 02:56 CST — W2 40826362 stopped again after c=32 minimal retry. API verified `cur_state=stopped`,`actual_status=exited`; no GPU compute process was running before stop.**

**2026-06-14 03:39 CST — W2 40826362 stopped again after allocator diagnostic. API verified `cur_state=stopped`,`actual_status=exited`; no GPU compute process was running before stop.**

**2026-06-14 05:23 CST — W2 40826362 stopped again after KV allocator diagnostics. Artifact `w2_c32_kv_allocator_diagnostics_2026-06-14` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; cache retained for possible next native-CUDA minimal validation.**

**2026-06-14 06:23 CST — W2 40826362 stopped again after KV-hint product-path diagnostics. Artifact `w2_c32_kv_hint_diagnostics_2026-06-14` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; cache retained for possible next native-CUDA minimal validation.**

**2026-06-14 06:51 CST — W2 40826362 stopped again after admission-cap diagnostics. Artifact `w2_c32_admission_cap_diagnostics_2026-06-14` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; no Ferrum process or GPU allocation remained before stop.**

**2026-06-14 07:57 CST — W2 40826362 stopped again after final c32/cap16 L5 and llama.cpp ratio evidence. Artifact `w2_c32_admission16_l5_pass_2026-06-14` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; no Ferrum/llama.cpp process or GPU allocation remained before stop.**

**2026-06-15 07:39 CST — W2 40826362 stopped again after decode-batch-stats probe. Artifact `w2_decode_batch_stats_probe_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; `nvidia-smi` before stop showed no running processes and 1 MiB GPU memory.**

**2026-06-15 07:59 CST — W2 40826362 stopped again after sliding-window batched-attn probe. Artifact `w2_sliding_batched_attn_probe_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; `nvidia-smi` before stop showed no running processes and 1 MiB GPU memory.**

**2026-06-15 08:53 CST — W2 40826362 stopped again after masked-argmax first probe. Artifact `w2_masked_argmax_probe_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; run stopped for greedy-sentinel/full-logits triage after product correctness had passed.**

**2026-06-15 09:09 CST — W2 40826362 stopped again after masked-argmax retry. Artifact `w2_masked_argmax_retry_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; run stopped after c=4 result and forbidden-token warnings during c=16 triage.**

**2026-06-15 09:37 CST — W2 40826362 stopped again after masked-argmax mask diagnostic. Artifact `w2_masked_argmax_maskdiag_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; GPU returned to 1 MiB after manually stopping on the first forbidden-token diagnostic.**

**2026-06-15 10:14 CST — W2 40826362 stopped again after masked-argmax sentinel-fix validation. Artifact `w2_masked_argmax_sentinel_fix_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; c16 diagnostic completed zero-error with no greedy-argmax forbidden-token warnings.**

**2026-06-15 10:26 CST — W2 40826362 stopped again after sentinel-fix c4/c16 diagnostic. Artifact `w2_sentinel_fix_c4_c16_diag_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; c4/c16 completed zero-error with no greedy-argmax warnings, and c4 diagnostic mean crossed the 80% baseline line.**

**2026-06-15 10:38 CST — W2 40826362 stopped again after c4 CI pre-gate. Artifact `w2_c4_ci_pregate_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; c4 `--require-ci --n-repeats 3` completed zero-error and lower bound cleared the 80% baseline line.**

**2026-06-15 11:26 CST — W2 40826362 stopped again after sentinel-fix full Ferrum release-shape matrix. Artifact `w2_sentinel_fix_release_shape_ferrum_ci_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; c=1/4/16/32 completed N=3, 100 prompts/repeat, zero errors, usage token counts, and zero blocker warnings.**

**2026-06-15 11:39 CST — W2 40826362 stopped again after vLLM baseline safety probe. Artifact `w2_baseline_safety_probe_2026-06-15` copied locally; API verified `cur_state=stopped`,`actual_status=exited`; vLLM c16 release-shape rerun reproduced invalid-UTF8 bad output on repeat 3, so c32 was not run.**
