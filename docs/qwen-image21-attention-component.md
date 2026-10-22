# Qwen Image 2.1 attention component

`QwenImage21Block0AttentionFamily` models the attention branch after normalization
and modulation: Q/K/V projections, learned Q/K head RMS normalization, explicit
adjacent RoPE, structured attention and the output projection. It reuses the Q/K
frontend and retains all nine BF16 node boundaries for diagnostics. Six stored
BF16 tensors occupy 134,218,240 bytes for the 4096-wide checkpoint. Biases,
residuals, conditioning generation and the rest of the transformer are excluded.

`QwenImage21AttentionConfig` contains checked transformer metadata, a typed
segment/key-valid layout and the exact rational encoding of the model's rounded
FP32 `1/sqrt(head_dim)` scale. Providers consume that scale explicitly. Layout
changes alter family and plan identity. `causal_condition` is not used to guess
an attention layout; callers supply the actual sequence segmentation.

The semantic program binds input x to `[B*S,H*D]` BF16 and frequencies to
`[S,D/2,2]` F32. The generic `ModelProgram::with_input_tensor_specs` contract
participates in serialization and fingerprinting and is checked before plan
inference or resource admission. This prevents a different RoPE period that
would independently divide the flattened row count from being accepted. Older
programs without declared constraints retain caller-selected shapes.

`verify_qwen_image21_attention` is a diagnostic Rust example, not an image CLI.
It requires `--config`, `--model-dir`, `--artifact-dir` and an explicit
`--probe single|causal-pair|full-pair`. CUDA additionally requires the explicit
budget options printed by `--help`. Weights enter through the existing managed
static transaction; input/output/staging bytes are Host-accounted, and the plan
is explicitly closed even when output inspection fails. Source mappings and
oracle allocations are excluded from that managed Host budget.

Single uses e0 and identity frequencies to verify weight wiring. Pair probes
use `[e0,-e0]`; causal and one bidirectional image segment exercise different
visibility and nontrivial two-key online softmax. Expectations are prepared
from weight bytes before runtime creation. All channels of the first eight
ports and up to seven predeclared output-projection coordinates per row are
checked. The oracle propagates prior BF16 alternatives through outward FP32
operation enclosures; distinct legal branches are unioned. Unsupported numeric
domains and intervals spanning over 256 BF16 alternatives return unresolved
before execution. The allowance is never widened from observed candidate data.

The declared four-neighbor exponential envelope is a composition acceptance
criterion, not proof of full-domain FP32 libdevice qualification. These sparse
probes do not establish general-input accuracy, Torch/vLLM parity, full-block
correctness, image quality or throughput parity.
