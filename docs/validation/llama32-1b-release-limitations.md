# Pinned Llama 3.2 1B release capability limits

## Decision

The release's small CUDA representative still must load and execute correctly.
It is not a release-blocking oracle for exact identifier copying or calculator
tool semantics. The user approved this separation on 2026-09-16 after independent
reference inference reproduced the failures below.

This exception applies only to
`unsloth/Llama-3.2-1B-Instruct@5a8abab4a5d6f164389b1079fb721cfab8d7126c`
and its supplied template. A changed source requires a new review. The frozen
catalog declares the affected behaviors and this evidence; the planner records
uncovered obligations in `model_limitations_not_run`, never as passed. If an
enabled, compatible, unrestricted representative exists, it owns the obligation
instead. Missing checker implementations are not model capability exceptions.

The combined tool probe covers selection, handoff and continuation. Because its
continuation cannot pass for this sample, the combined probe is excluded for
this representative, rather than falsely certifying its unexecuted remainder.
This does not remove protocol contract tests or other models' tool regressions.
Basic inference, strict JSON, stop/length behavior, numerical/device correctness,
capacity safety, source identity, installation and public startup remain required.
Executed failures remain failures; report verification does not swallow errors.

## Observed failures and reference reproduction

[Diagnostic CI 35029807824](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/35029807824)
tested candidate `26be658470982fd3211b3368d239b39c36d1fd20` on RTX 4050.
It passed 10 cases and failed three. Both state cases produced `COBALT-731`
instead of `cobalt-731`. The tool probe emitted valid `calc` calls with
`{"expression":"123 + 456"}` in both synchronous and streaming modes, then
refused the synchronous continuation after the actual result `579` was replayed.
The streaming continuation was not reached. Strict JSON passed in both modes.

The official artifact is `cuda-model-probe-35029807824-1`, artifact ID
10421717501, 114261 bytes, with GitHub ZIP digest
`755dd49da0d7a37b08d3b6889c8791d95c4ed3d82404e379160acf18094a35bb`.
Raw requests, responses, source identity and binary hashes are retained in that
artifact. The Ferrum binary SHA256 is
`2bc51aada3945236d8f46b407aa3446dd1db50c77b1fe99900d4808e11ca7fb1`.

Direct diagnosis reused that binary and the existing weights, without rebuilding
or redownloading. Greedy requests used seed 7, context 2048, one sequence and a
4 GiB runtime budget. The original natural-language tool request reproduced the
invalid argument `+123456`; the explicit expression request produced the correct
argument, but its continuation reproduced the refusal. These observations do not
claim that the original natural-language tool request is fixed.

Independent inference used the host's existing Transformers 5.16.1 environment,
the exact local 5a8 snapshot, CPU BF16, eager attention, no quantization or
compilation, greedy decoding and seed 7. CPU was used because that installed
PyTorch build did not include CUDA; this is a semantic reference, not a CUDA
numerical or performance comparison.

- Tool continuation: replayed Ferrum's captured prompt-token text through
  Transformers `/v1/completions`. Both received 96 prompt tokens and generated
  39 tokens ending naturally. Both refused to assist with external `calc`, with
  slightly different wording in the final sentence. The reference output limit
  was 96 versus Ferrum's 512; neither reached its limit.
- State: replayed the same first recall conversation through Transformers
  `/v1/chat/completions`; it returned exactly `COBALT-731`, with 89 input and
  6 output tokens, ending naturally. The reference output limit was 32 versus
  Ferrum's 512; neither reached its limit.

For raw prompt reconstruction the input was ASCII. Token spellings came from the
same tokenizer vocabulary and added-token table; byte-BPE space/newline markers
were decoded, and the leading BOS was omitted because this tokenizer adds it.
The reference response's local snapshot model label ends in `@main`, a server
label suffix, not a request to load remote main. The model source was an immutable
local snapshot path, not a different revision.

The reference continuation response SHA256 is
`73db1cebe6f99606730973bbca8491eab9c092a65816c070879ced84b14387ab`;
the reference state response SHA256 is
`3a7615996a7a08de464c6274b4207ce3031229adb7608442d916d6e485156a66`.
Raw diagnostic/reference files are retained outside the repository. These
comparisons establish limitations for the tested model/template and requests,
not a claim that all possible tool requests fail, nor proof of complete backend
equivalence.

## Release workflow rule

Qualify replacement small models with focused requests before a full packaging
run. Diagnose failures by reusing candidate binaries and model caches. Compare
model-dependent failures against an independent implementation before declaring
a limitation. Keep such declarations pinned, visible and separate from passes;
never exempt process, protocol, numerical or installation failures by matching
an error string after execution.
