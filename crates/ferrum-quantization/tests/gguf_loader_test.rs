//! Phase 1C integration tests: synthesize a GGUF that covers every tensor
//! shape a Llama-family decoder layer asks for, build a `GgufLoader<B>`
//! over it, and exercise the `WeightLoader<B>` API the model actually
//! uses (load_tensor / load_linear / qkv fusion / gate_up fusion / bias).
//!
//! Default tests run on `CpuBackend`; feature-gated tests run the same fused
//! bias fixtures on real Metal/CUDA devices. No downloaded model is required.

use std::io::{Cursor, Write};
use std::sync::Arc;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_quantization::gguf::{GgufFile, GgufLoader};
use ferrum_quantization::WeightLoader;

/// Build a tensor of given shape filled with a deterministic ramp so we can
/// later verify which sub-tensor ended up where in a fused weight.
fn ramp_tensor(rows: usize, cols: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = rows * cols;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (rows, cols), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn vec_tensor(n: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.01).collect();
    let t = Tensor::from_vec(raw, n, &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Build a GGUF tempfile containing the full set of Llama-family layer
/// tensors (one decoder layer + embed/output) — minimal but exercising
/// every name-mapping and fusion path.
fn build_full_layer_gguf() -> tempfile::NamedTempFile {
    // Toy dimensions: hidden=4, head_dim=4 (1 head), ffn=8.
    // q/k/v all 4x4 — combined qkv = 12x4.
    // gate/up both 8x4 — combined gate_up = 16x4.
    let token_embd = ramp_tensor(8, 4, 0.0); // vocab=8, hidden=4
    let output_norm = vec_tensor(4, 0.5);
    let output = ramp_tensor(8, 4, 1.0); // lm_head, vocab=8, hidden=4
    let attn_norm = vec_tensor(4, 0.6);
    let attn_q = ramp_tensor(4, 4, 2.0);
    let attn_k = ramp_tensor(4, 4, 3.0);
    let attn_v = ramp_tensor(4, 4, 4.0);
    let attn_output = ramp_tensor(4, 4, 5.0);
    let ffn_norm = vec_tensor(4, 0.7);
    let ffn_gate = ramp_tensor(8, 4, 6.0);
    let ffn_up = ramp_tensor(8, 4, 7.0);
    let ffn_down = ramp_tensor(4, 8, 8.0);

    let arch_v = Value::String("qwen3".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
        ("token_embd.weight", &token_embd),
        ("output_norm.weight", &output_norm),
        ("output.weight", &output),
        ("blk.0.attn_norm.weight", &attn_norm),
        ("blk.0.attn_q.weight", &attn_q),
        ("blk.0.attn_k.weight", &attn_k),
        ("blk.0.attn_v.weight", &attn_v),
        ("blk.0.attn_output.weight", &attn_output),
        ("blk.0.ffn_norm.weight", &ffn_norm),
        ("blk.0.ffn_gate.weight", &ffn_gate),
        ("blk.0.ffn_up.weight", &ffn_up),
        ("blk.0.ffn_down.weight", &ffn_down),
    ];

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();
    tmp
}

fn build_loader(tmp: &tempfile::NamedTempFile) -> GgufLoader<CpuBackend> {
    let gguf = GgufFile::open(tmp.path()).unwrap();
    GgufLoader::<CpuBackend>::from_file(Arc::new(gguf))
}

#[test]
fn has_tensor_translates_ferrum_names() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert!(loader.has_tensor("model.norm.weight"));
    assert!(loader.has_tensor("lm_head.weight"));
    assert!(loader.has_tensor("model.layers.0.input_layernorm.weight"));
    assert!(loader.has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(loader.has_tensor("model.layers.0.mlp.down_proj.weight"));

    // Unknown ferrum name → false (no GGUF mapping)
    assert!(!loader.has_tensor("totally_made_up.weight"));
    // Mapped, but not in this file (no second layer)
    assert!(!loader.has_tensor("model.layers.1.self_attn.q_proj.weight"));
}

#[test]
fn load_tensor_returns_dequantized_values() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    // attn_norm was vec_tensor(4, 0.6) → [0.60, 0.61, 0.62, 0.63]
    let raw = loader
        .load_tensor("model.layers.0.input_layernorm.weight")
        .unwrap();
    assert_eq!(raw.len(), 4);
    let expected: Vec<f32> = (0..4).map(|i| 0.6 + (i as f32) * 0.01).collect();
    assert_eq!(raw, expected);

    // embed table 8x4 = 32 elems, base 0.0 step 0.001
    let embed = loader.load_tensor("model.embed_tokens.weight").unwrap();
    assert_eq!(embed.len(), 32);
    assert!((embed[0] - 0.0).abs() < 1e-6);
    assert!((embed[31] - 0.031).abs() < 1e-5);
}

#[test]
fn load_tensor_unknown_name_errors() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let err = loader
        .load_tensor("model.layers.99.self_attn.q_proj.weight")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not present"), "error mentions missing: {err}");

    let err = loader
        .load_tensor("nonsense.thing")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unrecognised") || err.contains("no GGUF mapping"));
}

#[test]
fn load_linear_direct_path() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let o = loader
        .load_linear("model.layers.0.self_attn.o_proj")
        .unwrap();
    assert_eq!(o.in_features(), 4);
    assert_eq!(o.out_features(), 4);

    // lm_head is 8x4 (vocab=8, hidden=4)
    let lm = loader.load_linear("lm_head").unwrap();
    assert_eq!(lm.in_features(), 4);
    assert_eq!(lm.out_features(), 8);

    let down = loader.load_linear("model.layers.0.mlp.down_proj").unwrap();
    assert_eq!(down.in_features(), 8);
    assert_eq!(down.out_features(), 4);
}

#[test]
fn load_linear_fuses_qkv() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let qkv = loader
        .load_linear("model.layers.0.self_attn.qkv_proj")
        .unwrap();
    // Each part is 4x4; fused = 12x4
    assert_eq!(qkv.in_features(), 4);
    assert_eq!(qkv.out_features(), 12);

    // Forward verification: feed [1,1,1,1] (all-ones), expect each output
    // row = sum of that row's weights. Construct expected from the ramp.
    use ferrum_kernels::backend::Backend;
    let input: Vec<f32> = vec![1.0; 4];
    let mut out: Vec<f32> = vec![0.0; 12];
    let mut ctx = <CpuBackend as Backend>::new_context();
    qkv.forward(&mut ctx, &input, &mut out, 1);

    // q ramp base 2.0, k base 3.0, v base 4.0; each row of 4 has step 0.001.
    // row r within q (r=0..3) sums to 4*(2.0 + 0.001*(4*r)) + 0.001*(0+1+2+3) =
    //   for ramp(rows=4, cols=4, base=B), row r sum = 4*B + 0.001*(4*r*4 + 0+1+2+3)
    // Simpler: just compute expected via the same ramp logic.
    let mut expected = Vec::with_capacity(12);
    for &(rows_so_far, base) in &[(0usize, 2.0_f32), (4, 3.0), (8, 4.0)] {
        for r in 0..4 {
            // row r in this ramp_tensor(4,4,base): elements = base + 0.001*(r*4 + c) for c in 0..4
            let row_sum: f32 = (0..4)
                .map(|c| base + 0.001 * ((r * 4 + c) as f32))
                .sum::<f32>();
            expected.push(row_sum);
            let _ = rows_so_far; // unused; just here to clarify ordering
        }
    }
    for (i, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "qkv row {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn load_linear_fuses_gate_up() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let gu = loader
        .load_linear("model.layers.0.mlp.gate_up_proj")
        .unwrap();
    // gate (8x4) + up (8x4) → 16x4
    assert_eq!(gu.in_features(), 4);
    assert_eq!(gu.out_features(), 16);
}

#[test]
fn load_linear_unknown_returns_err() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);

    let result = loader.load_linear("model.layers.0.unknown_proj");
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("could not load Linear"),
        "error explains failure: {err}"
    );
}

#[test]
fn open_path_constructor_works() {
    let tmp = build_full_layer_gguf();
    let loader = GgufLoader::<CpuBackend>::open(tmp.path()).unwrap();
    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert_eq!(loader.gguf().architecture().unwrap(), "qwen3");
}

#[test]
fn quant_config_returns_none_in_phase_1c() {
    let tmp = build_full_layer_gguf();
    let loader = build_loader(&tmp);
    assert!(loader.quant_config().is_none());
}

mod fused_bias {
    use super::*;
    use ferrum_kernels::backend::{Backend, BackendQuantGguf, BackendQuantMarlin};

    fn fixture(
        parts: &[(&str, usize)],
        cols: usize,
        dtype: GgmlDType,
        bias_shapes: &[Option<Vec<usize>>],
    ) -> tempfile::NamedTempFile {
        let mut tensors = Vec::new();
        for (index, ((stem, rows), bias_shape)) in parts.iter().zip(bias_shapes).enumerate() {
            let values: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.125 + index as f32 * 0.03125)
                .collect();
            let weight = Tensor::from_vec(values, (*rows, cols), &Device::Cpu).unwrap();
            tensors.push((
                format!("{stem}.weight"),
                QTensor::quantize(&weight, dtype).unwrap(),
            ));
            if let Some(shape) = bias_shape {
                let values: Vec<f32> = (0..shape.iter().product::<usize>())
                    .map(|row| {
                        let magnitude = (index + row + 1) as f32 * 0.25;
                        if row % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        }
                    })
                    .collect();
                let bias = Tensor::from_vec(values, shape.as_slice(), &Device::Cpu).unwrap();
                tensors.push((
                    format!("{stem}.bias"),
                    QTensor::quantize(&bias, GgmlDType::F32).unwrap(),
                ));
            }
        }
        let architecture = Value::String("qwen2".to_owned());
        let metadata = [("general.architecture", &architecture)];
        let views = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        gguf_file::write(&mut Cursor::new(&mut bytes), &metadata, &views).unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    fn assert_complete_outputs<B: Backend + BackendQuantGguf + BackendQuantMarlin>(
        file: &tempfile::NamedTempFile,
        fused_name: &str,
        parts: &[&str],
        cols: usize,
        synchronize: impl Fn(&mut B::Context),
    ) {
        let loader = GgufLoader::<B>::from_file(Arc::new(GgufFile::open(file.path()).unwrap()));
        let reference = build_loader(file);
        let fused = loader.load_linear(fused_name).unwrap();
        let batch = 3;
        // The zero row isolates bias; the two signed rows exercise the full
        // matrix product and batch-major concatenation with unequal Q/K/V widths.
        let input: Vec<f32> = (0..batch)
            .flat_map(|m| {
                (0..cols).map(move |col| {
                    if m == 0 {
                        0.0
                    } else {
                        m as f32 * ((col % 7) as f32 - 3.0) * 0.125
                    }
                })
            })
            .collect();
        let total_rows = fused.out_features();
        let device_input = B::from_slice(&input);
        let mut device_output = B::from_slice(&vec![f32::NAN; batch * total_rows]);
        let mut context = B::new_context();
        fused.forward(&mut context, &device_input, &mut device_output, batch);
        synchronize(&mut context);
        let actual = B::to_vec(&device_output, batch * total_rows);
        let mut reference_context = CpuBackend::new_context();
        let mut offset = 0;
        for part in parts {
            // CUDA's legacy backend has no standalone unbiased Q4K route.
            // The independently loaded CPU projections are the reference for
            // every backend, including mixed-bias Q4K eager fusion on CUDA.
            let single = reference.load_linear(part).unwrap();
            let rows = single.out_features();
            let mut separate = vec![f32::NAN; batch * rows];
            single.forward(&mut reference_context, &input, &mut separate, batch);
            let weight_name = format!("{part}.weight");
            let reference_weights = reference.load_tensor(&weight_name).unwrap();
            // Observe the existing backend storage conversion (CUDA uses
            // FP16 here) separately from fused arithmetic. No precision mode
            // is changed to make the device test pass.
            let weights = B::to_vec(&loader.load_tensor(&weight_name).unwrap(), rows * cols);
            let bias_name = format!("{part}.bias");
            let bias = if loader.has_tensor(&bias_name) {
                B::to_vec(&loader.load_tensor(&bias_name).unwrap(), rows)
            } else {
                vec![0.0; rows]
            };
            let reference_bias = if reference.has_tensor(&bias_name) {
                reference.load_tensor(&bias_name).unwrap()
            } else {
                vec![0.0; rows]
            };
            for m in 0..batch {
                for row in 0..rows {
                    let expected = (0..cols)
                        .map(|col| {
                            f64::from(weights[row * cols + col]) * f64::from(input[m * cols + col])
                        })
                        .sum::<f64>()
                        + f64::from(bias[row]);
                    let reference_expected = (0..cols)
                        .map(|col| {
                            f64::from(reference_weights[row * cols + col])
                                * f64::from(input[m * cols + col])
                        })
                        .sum::<f64>()
                        + f64::from(reference_bias[row]);
                    let got = f64::from(actual[m * total_rows + offset + row]);
                    let standalone = f64::from(separate[m * rows + row]);
                    let roundoff = if B::activation_elem_size_bytes() == 2 {
                        1e-3
                    } else {
                        1e-4
                    };
                    let tolerance = roundoff * (1.0 + expected.abs());
                    let reference_tolerance = 1e-4 * (1.0 + reference_expected.abs());
                    assert!(
                        (standalone - reference_expected).abs() <= reference_tolerance,
                        "{part}, batch {m}, row {row}: CPU {standalone} != scalar {reference_expected}"
                    );
                    assert!(
                        (got - expected).abs() <= tolerance,
                        "{fused_name} / {part}, batch {m}, row {row}: {got} != scalar {expected}"
                    );
                    assert!(
                        (got - standalone).abs()
                            <= tolerance
                                + reference_tolerance
                                + (expected - reference_expected).abs(),
                        "{fused_name} / {part}, batch {m}, row {row}: {got} != standalone {standalone}"
                    );
                }
            }
            assert_eq!(&actual[offset..offset + rows], bias.as_slice());
            offset += rows;
        }
        assert_eq!(offset, total_rows);
    }

    fn qkv_cases<B: Backend + BackendQuantGguf + BackendQuantMarlin>(
        synchronize: impl Fn(&mut B::Context) + Copy,
    ) {
        let gguf_parts = [
            ("blk.0.attn_q", 4),
            ("blk.0.attn_k", 2),
            ("blk.0.attn_v", 2),
        ];
        let ferrum_parts = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ];
        for dtype in [GgmlDType::F32, GgmlDType::Q4K] {
            let cols = if dtype == GgmlDType::Q4K { 256 } else { 4 };
            for present in [
                [true, true, true],
                [false, true, false],
                [true, false, true],
            ] {
                let shapes = gguf_parts
                    .iter()
                    .zip(present)
                    .map(|((_, rows), present)| present.then(|| vec![*rows]))
                    .collect::<Vec<_>>();
                let file = fixture(&gguf_parts, cols, dtype, &shapes);
                assert_complete_outputs::<B>(
                    &file,
                    "model.layers.0.self_attn.qkv_proj",
                    &ferrum_parts,
                    cols,
                    synchronize,
                );
            }
        }
    }

    fn gate_up_cases<B: Backend + BackendQuantGguf + BackendQuantMarlin>(
        synchronize: impl Fn(&mut B::Context) + Copy,
    ) {
        let gguf_parts = [("blk.0.ffn_gate", 3), ("blk.0.ffn_up", 5)];
        for shapes in [vec![Some(vec![3]), None], vec![None, Some(vec![5])]] {
            let file = fixture(&gguf_parts, 4, GgmlDType::F32, &shapes);
            assert_complete_outputs::<B>(
                &file,
                "model.layers.0.mlp.gate_up_proj",
                &["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
                4,
                synchronize,
            );
        }
    }

    #[test]
    fn qkv_preserves_all_and_mixed_bias_for_dense_and_quantized_weights() {
        qkv_cases::<CpuBackend>(CpuBackend::sync);
    }

    #[test]
    fn gate_up_preserves_bias_and_zero_fills_either_missing_segment() {
        gate_up_cases::<CpuBackend>(CpuBackend::sync);
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_fused_bias_matches_complete_oracles() {
        use ferrum_kernels::backend::metal::MetalBackend;

        let synchronize = |context: &mut <MetalBackend as Backend>::Context| {
            MetalBackend::sync_checked(context)
                .expect("fused linear must complete real Metal work");
        };
        qkv_cases::<MetalBackend>(synchronize);
        gate_up_cases::<MetalBackend>(synchronize);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_bias_matches_complete_oracles() {
        use ferrum_kernels::backend::cuda::CudaBackend;

        qkv_cases::<CudaBackend>(CudaBackend::sync);
        gate_up_cases::<CudaBackend>(CudaBackend::sync);
    }

    #[test]
    fn qkv_rejects_bias_with_wrong_length_or_rank() {
        let parts = [
            ("blk.0.attn_q", 4),
            ("blk.0.attn_k", 2),
            ("blk.0.attn_v", 2),
        ];
        for invalid_shape in [vec![3], vec![1, 2]] {
            let file = fixture(
                &parts,
                4,
                GgmlDType::F32,
                &[None, Some(invalid_shape), None],
            );
            let loader = build_loader(&file);
            let error = loader
                .load_linear("model.layers.0.self_attn.qkv_proj")
                .err()
                .expect("malformed fused bias must fail during loading");
            assert!(error.to_string().contains("bias shape"), "{error}");
            assert!(error.to_string().contains("blk.0.attn_k.bias"), "{error}");
            assert!(loader
                .load_linear("model.layers.0.self_attn.k_proj")
                .is_err());
        }
    }
}
