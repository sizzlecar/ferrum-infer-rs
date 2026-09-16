# Pinned Qwen3.5 0.8B exact-recall limitation

## Decision and coverage boundary

`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`
does not reliably follow the release probe's exact, case-sensitive identifier-only
recall instruction. Independent Transformers inference reproduced both extra
prose and changed capitalization from the Ferrum failure below. The user's
release policy permits confirmed model-capability limitations to be disclosed
without blocking publication; it does not permit treating them as passes.

The committed local Metal and CUDA policies declare only `architecture_state`
for their respective profiles of this exact source. This is a model/template
limitation applied to both backends, not evidence that CUDA runtime passed.
Changing the source requires a new review. The combined state probe includes
recall, history reset and isolated conversations; excluding it does not certify
its remaining assertions. Unrepresented obligations are explicitly listed in
`model_limitations_not_run`. An enabled compatible unrestricted representative
still owns its required obligation instead.

Both 0.8B profiles remain mandatory for real `run` and `serve` basic execution.
Protocol contracts, actual-device numerical and state-boundary tests, capacity
safety, source identity, installation and public startup remain required.
Other models' state probes and every output oracle are unchanged. Existing
failed reports remain failures; no substring or case-insensitive workaround is
used to make their results pass.

## Formal failure

[Release run 35058753081](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/35058753081)
tested candidate `2f2a299ed93a03bfcbc449b3e16b59920f6dba66`.
Metal job `104679023900` failed on the second profile: Qwen0.8B had five passing
cases and failed `run-state` and `serve-state`. The preceding Llama8B profile
passed all thirteen cases; the following Qwen4B profile was not executed.

Official `release-metal-evidence` artifact `10432113565` is 195684 bytes;
its verified ZIP SHA256 is
`871e78dfd014790b6039fbd16db7aa5a983c65711edebd1540132bbb2bcd07bd`.
The Ferrum executable SHA256 is
`9045b94f99063075b064d8c6980b06733910cde95b8c11c049d41490589daa9f`.
`model-metal/report-1` retains the actual requests, responses, CLI records,
source identity and effective configuration. The task used Metal, context 2048,
one sequence, a 10 GiB runtime budget and a 512-token ceiling. State probes
explicitly disabled thinking.

- Cobalt recall returned `The code to remember is cobalt-731.` in both entrypoints,
  rather than only `cobalt-731`.
- HTTP amber recall also added the explanatory sentence. In the CLI conversation
  after a real reset, fresh recall returned `None`, a new amber write returned
  `OK`, and its subsequent recall returned `AMBER-284` instead of `amber-284`.
- HTTP fresh conversations and CLI post-reset controls returned `None`.
  These observations are retained, not promoted into a passing combined probe.
- Responses ended naturally; this was not an observed crash, timeout or
  output-budget truncation.

The same release's later CUDA job `104689026047` independently failed the two
0.8B state cases with `The code to remember is cobalt-731.` as well. Its basic
`run` and `serve` checks passed; the preceding Llama1B profile passed its required
checks, and CUDA Qwen4B was not executed. This corroborates the Metal failure,
not a passing CUDA state result. The overall release remained failed.

## Independent reference reproduction

The existing Panda reference environment used Transformers 5.16.1 and
PyTorch 2.10.0+cpu, CPU BF16, eager attention, two CPU threads and greedy decoding.
Weights and tokenizer/configuration came from the same local `2fc063...` snapshot.
The reference CLI initially returned HTTP 500 before generation because its
processor could not find a standalone chat template. A private metadata overlay
linked the existing snapshot files and supplied the already-cached official
`chat_template.jinja` from the same revision. Its 7755 bytes hash to
`273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`,
identical to Ferrum's recorded template. Neither the template text nor the model
cache was modified; no model weights were downloaded.

The reference overlay's weight, index, configuration and tokenizer files were
also independently hashed against Ferrum's recorded source identity. All matched;
the 1746942600-byte weight SHA256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

Reference requests retained the captured messages, seed 7, temperature zero and
`chat_template_kwargs.enable_thinking=false`. Only model routing changed to the
local overlay, and the output ceiling was 32 instead of 512. All three reference
requests returned HTTP 200 and natural `stop`, below that ceiling:

| Captured conversation | Reference answer | Prompt tokens, Ferrum/reference |
|---|---|---|
| First cobalt recall | `The code to remember is cobalt-731.` | 68 / 68 |
| First HTTP amber recall, SSE | `The code to remember is amber-284.` with two leading newlines | 67 / 67 |
| Actual CLI history after reset, fresh recall and amber write | `AMBER-284` | 107 / 107 |

Reference completion counts were 15, 14 and 9; Ferrum's counters differ, so these
are not claimed to be identical output-token streams or a numerical comparison.
This establishes an instruction-following limitation for the tested source,
template and conversations, not a performance result or proof of full backend
equivalence. The response model label ends in `@main` because of the reference
server's local-directory labeling; it did not fetch a floating remote revision.
Matching prompt-token counts are not a comparison of the actual input token IDs.
The reference SSE response emitted `stop` and usage but no `[DONE]` marker; it is
used only as semantic evidence, not as OpenAI streaming-protocol certification.

Raw reference response SHA256 values:

- Cobalt JSON: `f411d6127a5021ab75802b1766a9bd11ea60d303e9af3ea80115cdfde820251d`.
- Amber SSE: `ef390281475a57c19b1ce76b00c79359b6b336e28c90ca30569e1ec97e100f71`.
- CLI-history amber JSON: `5a6f27cff613f3ff41078634f8ef09e65077a0ebc2d854a259ec70bd7271b71a`.

Original requests, responses and server logs are retained outside the repository.
The earlier processor HTTP 500 and timed-out attempt to install a separate Mac
reference environment are setup failures, not model capability evidence. The
private reference server was stopped after the three successful requests.
