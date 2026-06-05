# G1/G3/G4 Release Regression

Status: METAL PASS

Validated locally:
- `ferrum run` one-shot and piped multi-turn JSONL on Metal.
- `ferrum serve` OpenAI non-stream, stream, structured output, tool calling, context-limit 400.
- Prefix cache and session cache product behavior on Metal.
- LoRA adapter-active and base path with LoRA loaded on Metal.
- G1/G3/G4 gate artifacts copied into top-level regression files.

Final CPU included: `False`
Final CUDA included: `False`
