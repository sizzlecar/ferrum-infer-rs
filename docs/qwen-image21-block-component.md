# Qwen Image 2.1 block component

`QwenImage21Block0Family` composes the existing attention and MLP families with
two LayerNorm, scale modulation, tanh gate and gated residual pairs. The complete
block has 21 compute nodes and nine stored BF16 weights (436,208,128 bytes,
416 MiB plus 512 bytes, for the 4096-wide checkpoint). It remains fixed tensor
work with no token state or independent allocation authority.

The caller supplies x, raw scale1, raw gate1, raw scale2, raw gate2 and explicit
RoPE frequencies. Scale-only modulation computes the declared BF16 `1+scale`
and multiplication boundaries. Gates pass through tanh in the graph. The first
residual uses the original x; the second uses the first residual. There are no
shift or learned LayerNorm parameters in this block. Upstream timestep/t0 row
selection is a separate conditioning adapter responsibility.

`QwenImage21Block0Config` binds the attention layout and a typed conditioning
broadcast mode. Elementwise uses `[B*S,H]` for all four conditioning inputs;
Row uses `[1,H]`. The hidden input is `[B*S,H]` BF16, and frequencies are
`[S,D/2,2]` F32. These six shapes are part of the semantic program and checked
before resource admission. Configurations do not infer conditioning layout from
an attention mask.

This component retains all 21 node boundaries for diagnostics. A future full
transformer composition should expose its actual final outputs and use compiler
liveness for intermediates; retaining every block's diagnostic outputs would
unnecessarily increase memory and readback demand. This component alone does
not implement image/text input projection, conditioning generation, multiple
transformer blocks, final output projection, a denoising step or image decoding.
