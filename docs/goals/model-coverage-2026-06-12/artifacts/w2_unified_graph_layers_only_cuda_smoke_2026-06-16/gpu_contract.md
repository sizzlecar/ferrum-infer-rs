# GPU Contract

- Lane: W2 Gemma3 27B GPTQ CUDA unified graph layers-only smoke.
- Hardware: existing Vast instance 40826362, 1x RTX 4090.
- Expected runtime/cost: 20-40 minutes, reused cached instance at about USD 0.425/hr while running.
- Stop condition: run+serve smoke PASS, or first build/startup/graph-capture failure with logs collected.
- Correctness gate: product `ferrum run` and `ferrum serve` smoke using typed CLI flags.
- Performance command: no release-grade performance command in this checkpoint; any later c16 bench is diagnostic only unless run with the release policy requirements.
- Cleanup: stopped after evidence; `vast_shutdown/stopped.json` records `actual_status=exited`.
