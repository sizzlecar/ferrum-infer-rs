# Serialized NVFP4 Qwen3-VL text component

`Qwen3VLTextNvfp4Checkpoint::new` checks the selected checkpoint prefix against
the Qwen3-VL architecture and explicit `num_hidden_layers_override`. The
configuration separately records declared, exported and executed layer counts;
no parameter-count label or model name chooses the architecture. The fixed
FastH3 checkpoint declares hidden width 5120, 64 original layers and 50 retained
layers. Callers may select a shorter explicit prefix for qualification.

The family reuses the existing BF16 text graph. Each declared language linear
becomes a standard NVFP4 activation quantizer followed by packed GEMM. Q/K/V
and gate/up share one quantization of their identical immutable input, unit
scale and profile: each layer has four quantizers and seven GEMMs. The
remaining RMSNorm, interleaved mRoPE, causal attention mask, residual and staged
SiLU nodes retain their existing contracts. Input embeddings, cosine and sine
remain actual caller-owned tensors. The output is the last selected decoder
layer before final norm; no P-H3 hidden-state connector or logits head is added.

The source dialect is the pinned FastVideo
`models/encoders/minimax_h3_checkpoint_nvfp4.py`: U8 packed E2M1 weights,
U8 E4M3 scales in whole 128x4 tiles, F32[1] global weight scale, group size 16.
Its `activation_scheme=dynamic` means per-group activation quantization with a
unit global activation scale. It does not request an activation-wide reduction.
The derived GEMM multiplier is `f32(1.0_f64 / f64(global_scale_f32))`.
AWQ pre-quant smoothing and mixed dense projection exclusions are rejected by
this initial consumer instead of silently changing its graph.

`managed_source(host)` allocates one four-byte multiplier per projection and
one four-byte unit scale through the caller's Host `WeightBacking` account.
The quantizer's scale port explicitly admits numerical Stored scalars or an
exact Metadata/F32 physical component; its packed-weight ports remain
exclusive physical-component consumers.
Packed data and block scales retain immutable mmap ownership and preserve their
original roles/encoding through checked physical-component views. Norms use
`ManagedDenseArchiveSource` with the graph's declared BF16 boundary, including
managed conversion when their checkpoint storage is F32/F16. Source inventory
limits are separate from execution admission.

The CUDA numerical profile explicitly combines the qualified quantization
recipe with Ferrum's existing BF16/F32 text arithmetic. It does not establish
numerical equality with FastVideo's SDPA or complete encoder quality. Real
checkpoint inspection, managed CUDA execution and independent output
qualification remain separate checks. Tokenization, embedding lookup and
vision conditioning are separate consumers.

The CUDA HeadRMS extension retains the original export and reduction tree for
power-of-two widths through 4096. Other widths through 8192 use an explicit
adjacent-pair tree that carries an unpaired subtree unchanged; normalization is
rounded to BF16 before multiplying by gamma and rounding to BF16 again. Its
profile names separate RN32 square/add/div/sqrt/reciprocal operations and does
not promise Torch reduction or rsqrt parity. The real hidden width 5120 needs
target qualification of that extension before a complete encoder forward.

`cargo run -p ferrum-models --example compile_qwen3_vl_nvfp4 --features
cuda-cutlass-nvfp4 -- --config PATH` reads the actual checkpoint, validates its
selected Stored components and builds a managed CUDA execution plan. It does
not upload weights or submit a forward. The JSON configuration explicitly
requires `encoder_dir`, `execute_layers`, `sequence_length`, `device`,
`maximum_metadata_bytes`, `maximum_shards`, `maximum_projections`,
`maximum_source_bytes`, `host_bytes`, `device_bytes`, `bootstrap_bytes`,
`stream_bytes` and `native_bytes`. Source metadata and selected weight limits
are independent of Host and CUDA admission budgets. This diagnostic permits
immutable Hugging Face cache symlinks, checks aggregate metadata before archive
parsing, and reports the selected precision/profile without claiming numerical
or performance qualification.
