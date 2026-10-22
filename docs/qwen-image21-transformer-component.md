# Qwen Image 2.1 transformer component

`QwenImage21TransformerFamily` describes one complete text-to-image transformer
evaluation. The number of layers comes from the checked transformer config;
the deployed 32-layer configuration uses all 297 transformer weight tensors.
The family produces target velocity. It does not yet supply the text encoder,
initial noise, diffusion scheduler, VAE decoder, image API, or image-edit path.

## Inputs and layout

`QwenImage21T2iLayout` declares a homogeneous batch: padded text length `T`,
single-frame target grid `H × W`, and each sample's valid text length.
Let `I = H × W`, `S = T + I`, and `B` be batch size.

| Input | Type and shape | Meaning |
| --- | --- | --- |
| packed latents | BF16 `[B*I, in_channels]` | Current target latents |
| text | BF16 `[B*T, context_dim]` | Already encoded prompt features |
| time features | BF16 `[B+causal_condition, 256]` | Cos-then-sin timestep features |
| frequencies | F32 `[S, head_dim/2, 2]` | Adjacent complex rotary pairs |

The mask has a causal text prefix and a bidirectional image suffix. Invalid
text keys remain masked separately for each sample. Padding still occupies
rotary positions; target frame positions start at padded `T`. The current
layout accepts 1–16 samples, positive even target axes up to 2048, and text
length below 8192, matching the declared rotary index domain. These are shape
limits, not a promise that every shape fits a device memory budget.

In causal-conditioning mode, exactly one shared `t=0` row follows the `B`
sample timestep rows. Typed row gathers distribute that row to text tokens
and each sample's own row to its image tokens. Batched concatenation and the
final target slice retain sample boundaries when `B > 1`.

## Compilation and execution

The graph contains text/image input projections, a shared timestep and
modulation path, every configured transformer block, and target normalization
and output projection. Only velocity `[B*I, out_channels]` is a completion
output. BF16 numerical boundaries remain declared even when intermediate
values are eligible for plan reclamation.

Call `config.compile_options()` when compiling this family. It supplies the
four input specifications and seven transport output specifications. Transport
shapes depend on explicit row-map attributes; providers validate those shapes
against the actual operands. Generic compilation does not infer model roles.

`operation.row_gather.bf16` and `operation.batched_row_concat.bf16` preserve
all 16 bits, including NaN payloads and signed zero. They use exact byte
extents, checked indices, and nonaliasing output storage. They do not allocate
scratch or perform floating-point arithmetic.

The CUDA tensor runtime exposes `maximum_commands_per_submission` explicitly;
its default remains 64. Whole-transformer callers must select a sufficient
limit including uploads and initialization. The configured limit enters the
runtime fingerprint and is checked before enqueue. Stream/flight/command
record counts are checked for overflow and addressability. Captured closure
heaps and total Host RSS are outside this experimental accounting scope.
Static weights and tensor storage remain subject to the shared runtime ledger.

## Conditioning arithmetic and validation scope

`QwenImage21ModelTimestep::from_scheduler_timestep` preserves the pipeline's
early BF16 cast: `B(RN32(B(t) / 1000))`. Cancelling the divide with the later
multiply changes rounding and is not allowed. `QwenImage21Timestep256` and
`QwenImage21RotaryFrequencies` write into caller-provided buffers. Those buffers
must enter execution through managed input admission.

The conditioning profile explicitly uses libm 0.2.15 with F32 intermediates.
Reference and CUDA arithmetic identities are separate; this component does
not promise Torch or vLLM-Omni bitwise equivalence. Compilation, managed
execution, numerical agreement, generated-image validity, and measured
throughput are separate acceptance results. A finite velocity tensor alone
does not establish end-to-end image correctness or throughput parity.
