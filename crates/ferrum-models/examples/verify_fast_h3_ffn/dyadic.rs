//! Small complete checkpoint fixture and independent exact-domain oracle.
//! This intentionally does not call Ferrum's quantizer, decoder or GPU providers.
use anyhow::{ensure, Result};
use ferrum_models::fast_h3::*;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::{collections::BTreeMap, sync::Arc};

type Tensor = (Dtype, Vec<usize>, Vec<u8>);

pub fn fixture() -> Result<(tempfile::TempDir, Arc<FastH3TransformerPackage>)> {
    let mut config: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../src/fast_h3/fixtures/transformer.json"
    ))?;
    for (name, v) in [
        ("hidden_size", 128),
        ("num_attention_heads", 1),
        ("attention_head_dim", 128),
        ("ffn_dim", 128),
        ("freq_dim", 8),
        ("in_channels", 2),
        ("audio_in_channels", 4),
        ("num_layers", 1),
        ("num_refiner_layers", 1),
        ("text_dim", 128),
        ("time_embed_dim", 16),
        ("time_embed_hidden_dim", 32),
        ("rope_freq_dim", 2),
    ] {
        config[name] = v.into();
    }
    let checkpoint = FastH3Checkpoint::parse(
        include_bytes!("../../src/fast_h3/fixtures/inference.json"),
        &serde_json::to_vec(&config)?,
        include_bytes!("../../src/fast_h3/fixtures/video_scheduler.json"),
        include_bytes!("../../src/fast_h3/fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )?;
    let mut dense = BTreeMap::new();
    let mut packed = BTreeMap::new();
    for (name, n, k) in [
        ("proj_in", 128, 8),
        ("audio_proj_in", 128, 4),
        ("time_embedder.fc_in", 32, 8),
        ("time_embedder.fc_out", 16, 32),
        ("proj_out", 256, 128),
        ("audio_proj_out", 128, 128),
        ("context_embedder", 128, 128),
        ("norm_out.linear", 256, 16),
    ] {
        linear(&mut dense, name, n, k, true);
    }
    for name in ["norm_out.norm.weight", "token_refiner.final_norm.weight"] {
        zero(&mut dense, name, &[128]);
    }
    for prefix in ["transformer_blocks.0", "token_refiner.refiner_blocks.0"] {
        for name in [
            "norm1.weight",
            "norm2.weight",
            "attn.norm_q.weight",
            "attn.norm_k.weight",
        ] {
            zero(&mut dense, &format!("{prefix}.{name}"), &[128]);
        }
        for name in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"] {
            linear(&mut dense, &format!("{prefix}.{name}"), 128, 128, false);
        }
        if prefix == "transformer_blocks.0" {
            projection(&mut packed, "transformer_blocks.0.ff.fc_in", 256, true);
            projection(&mut packed, "transformer_blocks.0.ff.fc_out", 128, false);
            linear(
                &mut dense,
                &format!("{prefix}.adaln_proj.linear"),
                2304,
                16,
                true,
            );
            linear(
                &mut dense,
                &format!("{prefix}.attn.to_gate_compress"),
                128,
                128,
                false,
            );
        } else {
            linear(&mut dense, &format!("{prefix}.ff.fc_in"), 256, 128, false);
            linear(&mut dense, &format!("{prefix}.ff.fc_out"), 128, 128, false);
        }
    }
    let directory = tempfile::tempdir()?;
    for (name, tensors) in [
        ("dense.safetensors", &dense),
        ("nvfp4_weights.safetensors", &packed),
    ] {
        let views = tensors
            .iter()
            .map(|(n, (t, s, b))| Ok((n.as_str(), TensorView::new(*t, s.clone(), b)?)))
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, &None, &directory.path().join(name))?;
    }
    let package = FastH3TransformerPackage::new(
        checkpoint.transformer(),
        SafetensorsArchive::open_single_file(directory.path(), "dense.safetensors")?,
        SafetensorsArchive::open_single_file(directory.path(), "nvfp4_weights.safetensors")?,
        FastH3Nvfp4Profile::FeedForward,
        FastH3WeightLimits {
            maximum_tensors: 128,
            maximum_source_bytes: 8 << 20,
            maximum_skip_metadata_bytes: 1024,
        },
    )?;
    Ok((directory, Arc::new(package)))
}
fn zero(out: &mut BTreeMap<String, Tensor>, name: &str, shape: &[usize]) {
    out.insert(
        name.into(),
        (
            Dtype::BF16,
            shape.to_vec(),
            vec![0; shape.iter().product::<usize>() * 2],
        ),
    );
}
fn linear(out: &mut BTreeMap<String, Tensor>, name: &str, n: usize, k: usize, bias: bool) {
    zero(out, &format!("{name}.weight"), &[n, k]);
    if bias {
        zero(out, &format!("{name}.bias"), &[n]);
    }
}
fn projection(out: &mut BTreeMap<String, Tensor>, name: &str, n: usize, first: bool) {
    let mut bytes = vec![0; n * 64];
    let mut scales = vec![0x18; n * 8]; // E4M3 1/16; n is already padded to128.
    let set = |bytes: &mut [u8], r: usize, k: usize, code: u8| {
        bytes[r * 64 + k / 2] |= code << ((k % 2) * 4)
    };
    for r in 0..n {
        if first && r >= 128 {
            set(&mut bytes, r, 0, 6); // 4 * input[0]=16, SiLU rounds toBF16 16.
            for g in 0..8 {
                let index =
                    (r / 128) * 1024 + (g / 4) * 512 + (r % 32) * 16 + (r % 128 / 32) * 4 + g % 4;
                scales[index] = 0x38;
            }
        } else if first {
            set(&mut bytes, r, r, 2);
        }
        // value=x/16.
        else {
            set(&mut bytes, r, r, 6);
            set(&mut bytes, r, (r + 1) % 128, 10);
        } // x/4 - next/16.
    }
    for (suffix, tensor) in [
        ("_nvfp4_weight", (Dtype::U8, vec![n, 64], bytes)),
        ("_nvfp4_weight_scale", (Dtype::U8, vec![n * 8], scales)),
        (
            "_weight_global_sf",
            (
                Dtype::BF16,
                vec![],
                bf16::ONE.to_bits().to_le_bytes().to_vec(),
            ),
        ),
        (
            "_nvfp4_alpha",
            (Dtype::F32, vec![], 1f32.to_le_bytes().to_vec()),
        ),
        (
            "_nvfp4_input_global_sf",
            (Dtype::F32, vec![], 1f32.to_le_bytes().to_vec()),
        ),
    ] {
        out.insert(format!("{name}::{suffix}"), tensor);
    }
}

pub fn input_value(row: usize, column: usize) -> f32 {
    if column == 0 {
        4.0
    } else {
        [2.0, 4.0, 6.0][(row + column) % 3]
    }
}

/// FP64 dots over independently decoded bytes; all sparse dyadic partial sums
/// are exact in FP32 too. Both activation quantizations stay in a domain where
/// each group has amax6 and E4M3 scale1, so approximate reciprocal cannot move
/// the scale or E2M1 rounding decisions. General random quantization is outside
/// this oracle; it requires the separately pinned FlashInfer reference.
pub fn expected(package: &FastH3TransformerPackage, rows: usize) -> Result<Vec<u16>> {
    ensure!(
        package.config().hidden_size() == 128 && package.config().ffn_dim() == 128,
        "dyadic oracle only supports its bounded tiny fixture"
    );
    let x = (0..rows)
        .flat_map(|r| (0..128).map(move |k| input_value(r, k)))
        .collect::<Vec<_>>();
    exact_quantization_domain(&x)?;
    let up = dot(
        package,
        "transformer_blocks.0.ff.fc_in.weight",
        &x,
        rows,
        256,
    )?;
    let mut activated = Vec::with_capacity(rows * 128);
    for r in 0..rows {
        for k in 0..128 {
            let gate = up[r * 256 + 128 + k];
            ensure!(gate == 16., "fixture gate must stay16");
            let z = (0f32 - gate) * f32::from_bits(0x3fb8aa3b);
            let e = 2f64.powf(z as f64) as f32;
            let sigmoid = 1f32 / (1f32 + e);
            let silu = gate * sigmoid;
            ensure!(
                bf16::from_f32(silu).to_f32() == 16.,
                "independent SiLU rounds to16"
            );
            activated.push(bf16::from_f32(up[r * 256 + k] * silu).to_f32());
        }
    }
    exact_quantization_domain(&activated)?;
    Ok(dot(
        package,
        "transformer_blocks.0.ff.fc_out.weight",
        &activated,
        rows,
        128,
    )?
    .into_iter()
    .map(|v| bf16::from_f32(v).to_bits())
    .collect())
}
fn exact_quantization_domain(x: &[f32]) -> Result<()> {
    for group in x.chunks_exact(16) {
        ensure!(
            group.iter().all(|v| [2., 4., 6.].contains(v)) && group.iter().any(|v| *v == 6.),
            "oracle activation left exact E2M1/amax6 domain"
        );
    }
    Ok(())
}
fn dot(
    package: &FastH3TransformerPackage,
    name: &str,
    x: &[f32],
    rows: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let w = package.nvfp4_matrix(name)?;
    ensure!(
        w.output_multiplier() == 1. && w.input_global_scale() == Some(1.),
        "fixture scalar changed"
    );
    let mut out = vec![0.; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut sum = 0f64;
            for k in 0..128 {
                let code = (w.packed_bytes()[j * 64 + k / 2] >> ((k % 2) * 4)) & 15;
                let g = k / 16;
                let index =
                    (j / 128) * 1024 + (g / 4) * 512 + (j % 32) * 16 + (j % 128 / 32) * 4 + g % 4;
                let scale = w.scale_bytes()[index];
                ensure!([0x18, 0x38].contains(&scale), "fixture scale changed");
                let scale = if scale == 0x18 { 1. / 16. } else { 1. };
                let magnitude = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
                let weight = magnitude * if code & 8 == 0 { scale } else { -scale };
                sum += x[r * 128 + k] as f64 * weight;
            }
            out[r * n + j] = bf16::from_f32(sum as f32).to_f32();
        }
    }
    Ok(out)
}

#[test]
fn independent_dyadic_full_ffn_oracle_matches_closed_form_and_source_inventory() {
    let (_dir, package) = fixture().unwrap();
    let actual = expected(&package, 3).unwrap();
    let closed = (0..3)
        .flat_map(|r| {
            (0..128).map(move |k| {
                bf16::from_f32(input_value(r, k) / 4. - input_value(r, (k + 1) % 128) / 16.)
                    .to_bits()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, closed);
    assert!(actual.iter().any(|bits| *bits != 0));
}

#[test]
fn eager_exact_and_fused_swiglu_are_distinct_outside_this_oracle_domain() {
    let gate = bf16::from_f32(1.).to_f32();
    let value = bf16::from_f32(1.0078125).to_f32();
    let silu = gate / (1. + (-gate).exp());
    let staged = bf16::from_f32(bf16::from_f32(silu).to_f32() * value).to_bits();
    let fused = bf16::from_f32(silu * value).to_bits();
    assert_eq!(staged, 0x3f3c);
    assert_eq!(fused, 0x3f3d);
    assert_ne!(staged, fused);
}
