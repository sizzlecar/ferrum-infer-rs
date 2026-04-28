//! Phase 1B tests: `GgufLinear<B>` constructed from a candle `QTensor`
//! produces the same forward output as a `DenseLinear<B>` built from
//! the same dequantized weights.
//!
//! Two angles:
//!   1. **Direct**: build a QTensor in memory, hand to `GgufLinear`, compare
//!      against `DenseLinear` built from the same fp32 source.
//!   2. **End-to-end**: synthesize a GGUF tempfile, open via `GgufFile`,
//!      load tensor, build `GgufLinear`, forward — proves the Phase 1A
//!      reader and Phase 1B linear compose correctly.
//!
//! All tests run on `CpuBackend` (always available, no feature flags).

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_kernels::Linear;
use ferrum_quantization::gguf::{GgufFile, GgufLinear};
use ferrum_quantization::DenseLinear;

/// Run a Linear and return the output vector.
fn forward<B: Backend<Buffer = Vec<f32>, Context = ()>>(
    linear: &dyn Linear<B>,
    input: &[f32],
    m: usize,
) -> Vec<f32> {
    let in_buf = input.to_vec();
    let mut out_buf = vec![0.0f32; m * linear.out_features()];
    let mut ctx = ();
    linear.forward(&mut ctx, &in_buf, &mut out_buf, m);
    out_buf
}

#[test]
fn gguf_linear_matches_dense_linear_on_f32_weights() {
    let device = Device::Cpu;

    // 4x3 weight matrix — rows are output neurons, cols are inputs.
    // Output for input [1,1,1] should equal each row's sum.
    #[rustfmt::skip]
    let weights: Vec<f32> = vec![
        1.0, 2.0, 3.0,    // row 0 — sum = 6
        4.0, 5.0, 6.0,    // row 1 — sum = 15
        7.0, 8.0, 9.0,    // row 2 — sum = 24
        10.0, 11.0, 12.0, // row 3 — sum = 33
    ];
    let t = Tensor::from_vec(weights.clone(), (4, 3), &device).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();

    let gguf_linear = GgufLinear::<CpuBackend>::from_qtensor(&qt).unwrap();
    assert_eq!(gguf_linear.in_features(), 3);
    assert_eq!(gguf_linear.out_features(), 4);

    let dense_linear = DenseLinear::<CpuBackend>::from_rows(&weights, 4, 3);

    let input = [1.0_f32, 1.0, 1.0];
    let out_gguf = forward(&gguf_linear, &input, 1);
    let out_dense = forward(&dense_linear, &input, 1);

    assert_eq!(out_gguf, vec![6.0, 15.0, 24.0, 33.0], "row sums");
    assert_eq!(
        out_gguf, out_dense,
        "GgufLinear and DenseLinear should be bit-exact for F32 weights"
    );
}

#[test]
fn gguf_linear_handles_batch_dimension() {
    let device = Device::Cpu;

    // 2x3 weight, batch of 5 inputs
    #[rustfmt::skip]
    let weights: Vec<f32> = vec![
        1.0, 0.0, 0.0,  // row 0 — picks input[0]
        0.0, 1.0, 0.0,  // row 1 — picks input[1]
    ];
    let t = Tensor::from_vec(weights, (2, 3), &device).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();
    let linear = GgufLinear::<CpuBackend>::from_qtensor(&qt).unwrap();

    // 5 batched inputs, each [a, b, ignored]
    let input: Vec<f32> = vec![
        1.0, 7.0, 99.0, // -> [1, 7]
        2.0, 8.0, 99.0, // -> [2, 8]
        3.0, 9.0, 99.0, // -> [3, 9]
        4.0, 10.0, 99.0, // -> [4, 10]
        5.0, 11.0, 99.0, // -> [5, 11]
    ];
    let out = forward(&linear, &input, 5);
    assert_eq!(
        out,
        vec![1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0]
    );
}

#[test]
fn gguf_linear_rejects_non_2d_tensor() {
    let device = Device::Cpu;
    let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2, 1), &device).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();
    let result = GgufLinear::<CpuBackend>::from_qtensor(&qt);
    assert!(result.is_err(), "rank-3 tensor must be rejected");
    let err = result.err().unwrap().to_string();
    assert!(err.contains("2-D"), "error mentions rank constraint: {err}");
}

#[test]
fn gguf_linear_with_bias_adds_bias() {
    let device = Device::Cpu;
    let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let t = Tensor::from_vec(weights, (2, 2), &device).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();

    let bias_v = vec![10.0_f32, -5.0];
    let bias_t = Tensor::from_vec(bias_v, 2, &device).unwrap();
    let bias_qt = QTensor::quantize(&bias_t, GgmlDType::F32).unwrap();

    let linear = GgufLinear::<CpuBackend>::from_qtensor_with_bias(&qt, &bias_qt).unwrap();
    let out = forward(&linear, &[3.0_f32, 7.0], 1);
    // y = W @ x + b = [3, 7] + [10, -5] = [13, 2]
    assert_eq!(out, vec![13.0, 2.0]);
}

#[test]
fn gguf_linear_round_trip_through_gguf_file() {
    // Synthesize a GGUF with one tensor, then load it back via GgufFile +
    // GgufLinear and run a forward — proves Phase 1A and 1B compose.
    let device = Device::Cpu;
    #[rustfmt::skip]
    let weights: Vec<f32> = vec![
        2.0, 0.0,
        0.0, 3.0,
    ];
    let t = Tensor::from_vec(weights, (2, 2), &device).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();

    let arch_v = Value::String("test".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![("blk.0.attn_q.weight", &qt)];

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();

    let gguf = GgufFile::open(tmp.path()).unwrap();
    let loaded = gguf.read_tensor("blk.0.attn_q.weight", &device).unwrap();
    let linear = GgufLinear::<CpuBackend>::from_qtensor(&loaded).unwrap();

    let out = forward(&linear, &[1.0_f32, 1.0], 1);
    // Identity-ish: [1,1] * diag(2,3) = [2, 3]
    assert_eq!(out, vec![2.0, 3.0]);
}
