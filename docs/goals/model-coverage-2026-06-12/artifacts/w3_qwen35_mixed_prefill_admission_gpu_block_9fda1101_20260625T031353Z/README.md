# W3 Qwen35 9fda1101 GPU Diagnostic Start Block

Date: 2026-06-25

Purpose: start the c32 diagnostic for `9fda1101 fix(engine): narrow mixed prefill kv admission`.

Result: no GPU diagnostic ran. The retained Vast instance `42216671` did not become available.

Start response:

```text
success=false
msg=Required resources are currently unavailable, state change queued.
```

Poll result summary:

- Polls observed: 9
- Instance id: `42216671`
- GPU: `1x RTX 4090`
- Cost shown by API: `$0.47777777777777775/hr`
- Final observed state: `cur_state=stopped`, `actual_status=exited`, `intended_status=stopped`
- SSH/CUDA check: not run
- Remote build: not run
- `ferrum run` smoke: not run
- `ferrum serve` smoke: not run
- c32 bench: not run
- live vLLM: not run

Decision: do not rent a new machine in this step; avoid environment churn and leave the source candidate ready for a later exact 1x4090 diagnostic.
