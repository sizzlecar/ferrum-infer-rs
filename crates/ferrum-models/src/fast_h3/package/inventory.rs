use anyhow::{ensure, Context, Result};
use ferrum_runtime::ElementType;

use super::{FastH3Nvfp4Profile, FastH3TransformerConfig};

pub(super) struct Entry {
    pub name: String,
    pub aliases: Vec<String>,
    pub shape: Vec<usize>,
    pub compute_type: ElementType,
    pub packed: bool,
}

/// Checkpoint logical parameters, before choosing their physical representation.
/// Source: models/dits/minimax_h3.py and configs/models/dits/minimax_h3.py
/// at the module's pinned FastVideo revision. RoPE frequencies are non-persistent.
pub(super) fn entries(
    c: &FastH3TransformerConfig,
    profile: FastH3Nvfp4Profile,
    skip: &[usize],
    maximum: usize,
) -> Result<Vec<Entry>> {
    let count = c
        .layers()
        .checked_mul(13)
        .and_then(|n| {
            c.refiner_layers()
                .checked_mul(10)
                .and_then(|r| n.checked_add(r))
        })
        .and_then(|n| n.checked_add(18 + usize::from(c.adaln_rank().is_some())))
        .context("FastH3 weight count overflow")?;
    ensure!(
        count <= maximum,
        "FastH3 logical weight inventory exceeds capacity"
    );
    let h = c.hidden_size();
    let inner = c
        .attention_heads()
        .checked_mul(c.attention_head_dim())
        .context("attention width overflow")?;
    let ffn2 = c.ffn_dim().checked_mul(2).context("FFN width overflow")?;
    let modulation = h
        .checked_mul(18)
        .context("three-modality AdaLN width overflow")?;
    let norm_out = h.checked_mul(2).context("final AdaLN width overflow")?;
    let adaln_dim = c.adaln_rank().unwrap_or(c.time_embed_dim());
    let adaln_type = if c.adaln_rank().is_some() {
        ElementType::F16
    } else {
        ElementType::Bf16
    };
    let mut out = Vec::with_capacity(count);
    for (name, rows, cols) in [
        ("proj_in", h, c.video_head_width()),
        ("audio_proj_in", h, c.audio_channels()),
        (
            "time_embedder.fc_in",
            c.time_embed_hidden_dim(),
            c.frequency_dim(),
        ),
        (
            "time_embedder.fc_out",
            c.time_embed_dim(),
            c.time_embed_hidden_dim(),
        ),
        ("proj_out", c.video_projection_rows(), h),
        ("audio_proj_out", c.audio_projection_rows(), h),
    ] {
        linear(&mut out, name, rows, cols, ElementType::F32, true, false);
    }
    linear(
        &mut out,
        "context_embedder",
        h,
        c.text_dim(),
        ElementType::Bf16,
        true,
        false,
    );
    linear(
        &mut out,
        "norm_out.linear",
        norm_out,
        adaln_dim,
        adaln_type,
        true,
        false,
    );
    parameter(
        &mut out,
        "norm_out.norm.weight",
        vec![h],
        ElementType::Bf16,
        false,
    );
    parameter(
        &mut out,
        "token_refiner.final_norm.weight",
        vec![h],
        ElementType::Bf16,
        false,
    );
    if let Some(rank) = c.adaln_rank() {
        linear(
            &mut out,
            "adaln_basis",
            rank,
            c.time_embed_dim(),
            ElementType::F16,
            false,
            false,
        );
    }
    for (refiner, layers) in [(false, c.layers()), (true, c.refiner_layers())] {
        for index in 0..layers {
            let prefix = if refiner {
                format!("token_refiner.refiner_blocks.{index}")
            } else {
                format!("transformer_blocks.{index}")
            };
            let quantized = !refiner && skip.binary_search(&index).is_err();
            for suffix in ["norm1.weight", "norm2.weight"] {
                parameter(
                    &mut out,
                    &format!("{prefix}.{suffix}"),
                    vec![h],
                    ElementType::Bf16,
                    false,
                );
            }
            for suffix in ["norm_q.weight", "norm_k.weight"] {
                parameter(
                    &mut out,
                    &format!("{prefix}.attn.{suffix}"),
                    vec![c.attention_head_dim()],
                    ElementType::Bf16,
                    false,
                );
            }
            for (suffix, rows, cols) in [
                ("to_q", inner, h),
                ("to_k", inner, h),
                ("to_v", inner, h),
                ("to_out", h, inner),
            ] {
                linear(
                    &mut out,
                    &format!("{prefix}.attn.{suffix}"),
                    rows,
                    cols,
                    ElementType::Bf16,
                    false,
                    quantized && profile != FastH3Nvfp4Profile::FeedForward,
                );
            }
            for (suffix, rows, cols) in [("fc_in", ffn2, h), ("fc_out", h, c.ffn_dim())] {
                linear(
                    &mut out,
                    &format!("{prefix}.ff.{suffix}"),
                    rows,
                    cols,
                    ElementType::Bf16,
                    false,
                    quantized,
                );
            }
            if !refiner {
                linear(
                    &mut out,
                    &format!("{prefix}.attn.to_gate_compress"),
                    inner,
                    h,
                    ElementType::Bf16,
                    false,
                    quantized && profile == FastH3Nvfp4Profile::DiTWithVsa,
                );
                linear(
                    &mut out,
                    &format!("{prefix}.adaln_proj.linear"),
                    modulation,
                    adaln_dim,
                    adaln_type,
                    true,
                    false,
                );
            }
        }
    }
    debug_assert_eq!(out.len(), count);
    // Check source-addressable products before readers or component builders.
    for entry in &out {
        entry
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= isize::MAX as usize)
            .context("FastH3 dense weight dimensions exceed address space")?;
    }
    Ok(out)
}

fn linear(
    out: &mut Vec<Entry>,
    name: &str,
    rows: usize,
    columns: usize,
    dtype: ElementType,
    bias: bool,
    packed: bool,
) {
    parameter(
        out,
        &format!("{name}.weight"),
        vec![rows, columns],
        dtype,
        packed,
    );
    if bias {
        parameter(out, &format!("{name}.bias"), vec![rows], dtype, false);
    }
}

fn parameter(
    out: &mut Vec<Entry>,
    name: &str,
    shape: Vec<usize>,
    compute_type: ElementType,
    packed: bool,
) {
    // These are the exact released diffusers -> FastVideo import aliases. A
    // package rejects two aliases present simultaneously, rather than guessing.
    let diffusers = name
        .replace("time_embedder.fc_in.", "time_embedder.linear_1.")
        .replace("time_embedder.fc_out.", "time_embedder.linear_2.")
        .replace(".attn.to_out.", ".attn.to_out.0.")
        .replace(".ff.fc_in.", ".ff.net.0.proj.")
        .replace(".ff.fc_out.", ".ff.net.2.");
    let mut aliases = vec![name.to_owned()];
    if diffusers != name {
        aliases.push(diffusers);
    }
    out.push(Entry {
        name: name.to_owned(),
        aliases,
        shape,
        compute_type,
        packed,
    });
}
