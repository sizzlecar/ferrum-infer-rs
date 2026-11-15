# FastH3 dense text refiner

This is the cold, timestep-free component between the context projection and
Ref2VA packing. It consumes BF16 `[sum(request_lengths), hidden]` and publishes
BF16 refined text with the same shape. The caller retains or reuses the output
through existing execution owners; this module creates no cache or allocator.
`FastH3InputProjectionPhase::Text` supplies the preceding context projection.
`FastH3InputPackingFamily::REFINED_TEXT_ID` is the subsequent packing operand.
An explicit graph fragment or admitted Host input copy connects these values.

The source is FastVideo revision
`866a3cdb91a85aef8561847365fc8564add9e4c8`,
`fastvideo/models/dits/minimax_h3.py`: `MiniMaxH3TokenRefinerBlock`,
`MiniMaxH3TokenRefiner`, and `_refined_text`. The released transformer declares
two blocks. Each block selects its original ten dense parameters, followed by
the single final RMS gamma. Projection widths use `heads * head_dim`, which
is 7168 for the released model and differs from hidden size 5376.

Each block executes:

1. RMS normalization with a BF16 boundary before and after gamma multiplication.
2. Separate dense BF16 Q/K/V projections and per-head RMS for Q and K.
3. Noncausal, isolated attention for each declared request; dense output
   projection and BF16 residual addition.
4. RMS normalization, value-first packed dense FFN input projection, separate
   value/gate slices, BF16 SiLU followed by BF16 multiplication, dense output
   projection and BF16 residual addition.

The final RMS applies the checkpoint's separate `final_norm_eps`. There is no
RoPE, time modulation, compression gate or sparse attention. The refiner is
excluded from both the VSA backend and the NVFP4 projection-name pattern. Its
constructor does not receive the main blocks' fused SwiGLU flags. Replacing its
eager SiLU/product boundaries with the main blocks' single-round F32 fusion is
numerically different. P-H3's Comfy quantized refiner is a different source and
arithmetic contract.

`dense_bf16_staged.v1` declares storage and intermediate rounding; provider
identities qualify reductions, exponentials and inverse square roots. The
Reference tests use independent FP64 dot products and full softmax formulas,
allowing one BF16 ULP at reduction boundaries. Residuals and the staged SwiGLU
are compared bitwise, including a fixture that distinguishes single-round
fusion. Packed requests are also compared with individual execution and with
another request's inputs changed. All operations execute through real Stored
initialization and `TensorComponentSession` ownership.

The pinned deployment requests VSA globally, but refiner construction excludes
VSA. The selector consequently falls back to automatic dense selection: Flash
when supported and installed, otherwise SDPA. Its `FASTVIDEO_FA4=0` setting
disables explicit FA4 selection; the remaining FA3/FA2 import selection still
needs an actual worker receipt. The CPU qualification is neither an assertion
of that deployed backend nor a Torch/FlashAttention GPU parity result.
