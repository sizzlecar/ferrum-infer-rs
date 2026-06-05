# G4 LoRA Startup Gate

Status: PASS

Model: `qwen3:0.6b`

Validated:
- `cargo test --workspace --all-targets`
- LoRA f32 reference test
- PEFT adapter loader tests
- server adapter model routing tests
- CLI startup LoRA tests
- `/v1/models`, base chat, adapter chat, unknown adapter error
- c=1 bench smoke for base/no-lora, base/with-lora, and adapter model id
