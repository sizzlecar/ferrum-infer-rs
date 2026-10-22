# Managed Qwen3-VL text encoder

`Qwen3VLTextEncoderFamily` evaluates the configured text decoder layers as a
fixed tensor program. The deployed image model has 36 layers, 32 query heads,
8 key/value heads and head width 128. Its required schema contains the 396
layer tensors (13,892,143,104 BF16 bytes). Embedding lookup, the vision tower,
the LM head and final RMSNorm are outside this component.

Inputs are embeddings `[B*S, hidden_size]` and BF16 cosine/sine tensors
`[B*S, head_dim]`. The layout supplies explicit valid-key runs for each sample;
it does not infer left or right padding from sequence lengths. Positions are
represented by the supplied frequency tensors. The output is the final layer's
residual hidden state, before final RMSNorm, matching the image pipeline's
conditioning extraction point.

Each layer has 17 fixed tensor operations. Grouped attention maps each query
head to its original KV head without materializing repeated KV storage.
Text RoPE uses half-split rotation and rounds each BF16 multiplication before
addition. Plain residual addition is separate from gated DiT residuals.
All weights and activations use the existing plan/runtime ownership path.

The Reference and CUDA numerical profiles are distinct. Attention uses an
online FP32 softmax profile, not an asserted equivalence to a BF16 eager
probability tensor. The graph and operator tests do not establish real-model
embedding parity or end-to-end image quality. A prompt adapter, managed
embedding lookup, output cropping, and real-weight execution are still needed
to connect this component to the image transformer.
