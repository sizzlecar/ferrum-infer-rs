use ferrum_kernels::backend::{cpu::CpuBackend, Backend, BackendQuantMarlin};
use ferrum_kernels::LinearMetadata;
use ferrum_quantization::{DenseLinear, Linear, NativeSafetensorsLoader, WeightLoader};
use half::{bf16, f16};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

struct Fixture {
    directory: tempfile::TempDir,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

fn encoded(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Dtype::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect(),
        Dtype::BF16 => values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_le_bytes())
            .collect(),
        _ => unreachable!("dense fixture uses F32, F16 or BF16"),
    }
}

fn fixture(
    name: &str,
    weight_shape: &[usize],
    bias_shape: Option<&[usize]>,
    dtype: Dtype,
) -> Fixture {
    // Dyadic values keep source and backend storage conversions exact. This
    // isolates bias loading without selecting a different precision policy.
    let weight: Vec<_> = (0..weight_shape.iter().product::<usize>())
        .map(|index| ((index % 9) as f32 - 4.0) * 0.125)
        .collect();
    let bias: Option<Vec<_>> = bias_shape.map(|shape| {
        (0..shape.iter().product::<usize>())
            .map(|index| {
                let magnitude = (index + 1) as f32 * 0.25;
                if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect()
    });
    let weight_bytes = encoded(&weight, dtype);
    let bias_bytes = bias.as_ref().map(|values| encoded(values, dtype));
    let mut tensors = vec![(
        format!("{name}.weight"),
        TensorView::new(dtype, weight_shape.to_vec(), &weight_bytes).unwrap(),
    )];
    if let (Some(shape), Some(bytes)) = (bias_shape, bias_bytes.as_ref()) {
        tensors.push((
            format!("{name}.bias"),
            TensorView::new(dtype, shape.to_vec(), bytes).unwrap(),
        ));
    }
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    Fixture {
        directory,
        weight,
        bias,
    }
}

fn assert_complete_outputs<B: Backend + BackendQuantMarlin>(
    name: &str,
    rows: usize,
    dtype: Dtype,
    biased: bool,
    synchronize: impl Fn(&mut B::Context),
) {
    let cols = 4;
    let bias_shape = [rows];
    let fixture = fixture(name, &[rows, cols], biased.then_some(&bias_shape), dtype);
    let loader = NativeSafetensorsLoader::<B>::open(fixture.directory.path()).unwrap();
    let mut context = B::new_context(&Default::default()).expect("backend context");
    let linear = loader.load_linear(&mut context, name).unwrap();
    assert_eq!(linear.in_features(), cols);
    assert_eq!(linear.out_features(), rows);
    assert_eq!(linear.metadata(), LinearMetadata::from_name(name));

    // Zero, positive and signed rows check every output and batch stride.
    let input = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, -1.0, 0.5, -1.5, 2.0];
    let batch = 3;
    let device_input = B::from_slice_in_context(&mut context, &input);
    let mut device_output = B::from_slice_in_context(&mut context, &vec![f32::NAN; batch * rows]);
    linear.forward(&mut context, &device_input, &mut device_output, batch);
    synchronize(&mut context);
    let actual = B::to_vec_in_context(&mut context, &device_output, batch * rows);

    // Construct an independent CPU projection from the fixture, bypassing
    // NativeSafetensorsLoader so the reference cannot repeat a dropped bias.
    let reference = if let Some(bias) = fixture.bias.as_ref() {
        DenseLinear::<CpuBackend>::from_rows_with_bias(&fixture.weight, bias, rows, cols)
    } else {
        DenseLinear::<CpuBackend>::from_rows(&fixture.weight, rows, cols)
    };
    let mut reference_context =
        CpuBackend::new_context(&Default::default()).expect("backend context");
    let mut separate = vec![f32::NAN; batch * rows];
    reference.forward(
        &mut reference_context,
        &input.to_vec(),
        &mut separate,
        batch,
    );
    for m in 0..batch {
        for row in 0..rows {
            let bias = fixture.bias.as_ref().map_or(0.0, |values| values[row]);
            let expected = (0..cols)
                .map(|col| {
                    f64::from(fixture.weight[row * cols + col]) * f64::from(input[m * cols + col])
                })
                .sum::<f64>()
                + f64::from(bias);
            let index = m * rows + row;
            // All products, sums and outputs here are exactly representable
            // in FP16 as well as FP32; no backend-dependent tolerance is needed.
            assert_eq!(
                f64::from(actual[index]),
                expected,
                "{name}, {dtype:?}, biased={biased}, batch {m}, row {row}"
            );
            assert_eq!(f64::from(separate[index]), expected);
        }
    }
    let bias = fixture
        .bias
        .as_deref()
        .map_or_else(|| vec![0.0; rows], <[f32]>::to_vec);
    assert_eq!(&actual[..rows], bias.as_slice());
}

fn direct_cases<B: Backend + BackendQuantMarlin>(
    biased: bool,
    synchronize: impl Fn(&mut B::Context) + Copy,
) {
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        for (name, rows) in [
            ("model.layers.0.self_attn.o_proj", 3),
            ("model.layers.0.self_attn.qkv_proj", 5),
        ] {
            assert_complete_outputs::<B>(name, rows, dtype, biased, synchronize);
        }
    }
}

#[test]
fn direct_bias_matches_complete_oracles() {
    direct_cases::<CpuBackend>(true, CpuBackend::sync);
}

#[test]
fn direct_without_bias_matches_complete_oracles() {
    direct_cases::<CpuBackend>(false, CpuBackend::sync);
}

#[test]
fn direct_bias_rejects_wrong_length_or_rank() {
    for name in [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.qkv_proj",
    ] {
        for shape in [vec![2], vec![4], vec![1, 3], vec![]] {
            let fixture = fixture(name, &[3, 4], Some(&shape), Dtype::F32);
            let loader =
                NativeSafetensorsLoader::<CpuBackend>::open(fixture.directory.path()).unwrap();
            let error = match loader.load_linear(
                &mut ferrum_kernels::backend::cpu::CpuContext::default(),
                name,
            ) {
                Ok(_) => panic!("accepted bias shape {shape:?} for [3, 4] weight"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("bias"), "{error}");
        }
    }
}

const FUSED_PROJECTIONS: &[(&str, &[&str])] = &[
    ("qkv_proj", &["q_proj", "k_proj", "v_proj"]),
    ("gate_up_proj", &["gate_proj", "up_proj"]),
    ("in_proj_qkvz", &["in_proj_qkv", "in_proj_z"]),
    ("in_proj_ba", &["in_proj_b", "in_proj_a"]),
];

fn invalid_fused_bias_fixture(parts: &[&str], invalid_shape: &[usize]) -> tempfile::TempDir {
    let weight = encoded(&[0.25; 8], Dtype::F32);
    let bias = encoded(&[0.5; 2], Dtype::F32);
    let invalid_bias = encoded(&vec![0.5; invalid_shape.iter().product()], Dtype::F32);
    let mut tensors = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        tensors.push((
            format!("{part}.weight"),
            TensorView::new(Dtype::F32, vec![2, 4], &weight).unwrap(),
        ));
        let (shape, bytes) = if index + 1 == parts.len() {
            (invalid_shape.to_vec(), &invalid_bias)
        } else {
            (vec![2], &bias)
        };
        tensors.push((
            format!("{part}.bias"),
            TensorView::new(Dtype::F32, shape, bytes).unwrap(),
        ));
    }
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    directory
}

#[test]
fn fused_bias_rejects_wrong_length_or_rank() {
    for &(name, parts) in FUSED_PROJECTIONS {
        for shape in [&[3][..], &[1, 2][..]] {
            let directory = invalid_fused_bias_fixture(parts, shape);
            let loader = NativeSafetensorsLoader::<CpuBackend>::open(directory.path()).unwrap();
            let error = match loader.load_linear(
                &mut ferrum_kernels::backend::cpu::CpuContext::default(),
                name,
            ) {
                Ok(_) => panic!("accepted {name} with invalid split bias {shape:?}"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("bias"), "{error}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires real CUDA capture to prove invalid fused bias is rejected before upload"]
fn cuda_invalid_fused_bias_does_not_enter_device_allocation() {
    use ferrum_kernels::backend::{cuda::CudaBackend, BackendCompletion, BackendGraph};

    let mut context = CudaBackend::new_context(&Default::default()).expect("backend context");
    CudaBackend::begin_execution(&mut context).unwrap();
    let mut capture_output = CudaBackend::alloc_in_context(&mut context, 1);
    CudaBackend::observe_completion(&mut context)
        .into_result()
        .unwrap();
    CudaBackend::begin_execution(&mut context).unwrap();
    unsafe { CudaBackend::begin_graph_capture(&mut context) }.unwrap();
    CudaBackend::zero_buffer(&mut context, &mut capture_output, 1).unwrap();
    // The instance allocation path rejects allocation during capture. A
    // malformed host bias must return its validation error without entering
    // that path. Catch regressions so the actual capture can retire first.
    let results = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut errors = Vec::new();
        for &(name, parts) in FUSED_PROJECTIONS {
            for shape in [&[3][..], &[1, 2][..]] {
                let directory = invalid_fused_bias_fixture(parts, shape);
                let loader =
                    NativeSafetensorsLoader::<CudaBackend>::open(directory.path()).unwrap();
                match loader.load_linear(&mut context, name) {
                    Ok(_) => panic!("accepted {name} with invalid split bias {shape:?}"),
                    Err(error) => errors.push(error.to_string()),
                }
            }
        }
        errors
    }));
    unsafe { CudaBackend::end_graph_capture(&mut context, 0) }.unwrap();
    assert!(matches!(
        CudaBackend::observe_completion(&mut context),
        BackendCompletion::Succeeded
    ));
    CudaBackend::retire_context(&mut context).unwrap();
    for error in results.expect("invalid bias reached the CUDA allocation path") {
        assert!(error.contains("bias"), "{error}");
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn metal_direct_bias_matches_complete_oracles() {
    use ferrum_kernels::backend::metal::MetalBackend;

    let synchronize = |context: &mut <MetalBackend as Backend>::Context| {
        MetalBackend::sync_checked(context).expect("direct linear must complete real Metal work");
    };
    direct_cases::<MetalBackend>(true, synchronize);
    direct_cases::<MetalBackend>(false, synchronize);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_direct_bias_matches_complete_oracles() {
    use ferrum_kernels::backend::cuda::CudaBackend;

    direct_cases::<CudaBackend>(true, CudaBackend::sync);
    direct_cases::<CudaBackend>(false, CudaBackend::sync);
}

fn interleaved_instances_preserve_loaded_weights<B: Backend + BackendQuantMarlin>(
    synchronize: impl Fn(&mut B::Context),
) {
    let name = "model.layers.0.self_attn.o_proj";
    let first = fixture(name, &[3, 4], Some(&[3]), Dtype::F16);
    let second = fixture(name, &[3, 4], None, Dtype::BF16);
    let mut first_context = B::new_context(&Default::default()).expect("backend context");
    let first_loader = NativeSafetensorsLoader::<B>::open(first.directory.path()).unwrap();
    let first_linear = first_loader.load_linear(&mut first_context, name).unwrap();
    drop(first_loader);
    let mut second_context = B::new_context(&Default::default()).expect("backend context");
    let second_loader = NativeSafetensorsLoader::<B>::open(second.directory.path()).unwrap();
    let second_linear = second_loader
        .load_linear(&mut second_context, name)
        .unwrap();
    drop(second_loader);

    let check = |context: &mut B::Context,
                 linear: &dyn Linear<B>,
                 fixture: &Fixture,
                 input: [f32; 4]| {
        let input_buffer = B::from_slice_in_context(context, &input);
        let mut output = B::from_slice_in_context(context, &[f32::NAN; 3]);
        linear.forward(context, &input_buffer, &mut output, 1);
        synchronize(context);
        let actual = B::to_vec_in_context(context, &output, 3);
        for (row, value) in actual.iter().enumerate() {
            let expected = input
                .iter()
                .enumerate()
                .map(|(col, value)| f64::from(*value) * f64::from(fixture.weight[row * 4 + col]))
                .sum::<f64>()
                + fixture
                    .bias
                    .as_ref()
                    .map_or(0.0, |bias| f64::from(bias[row]));
            assert_eq!(f64::from(*value), expected, "row {row}");
        }
    };

    // B is constructed between A's upload and first execution. Every upload,
    // projection and readback must still use the correct instance context.
    check(
        &mut first_context,
        first_linear.as_ref(),
        &first,
        [1.0, -1.0, 0.5, 2.0],
    );
    check(
        &mut second_context,
        second_linear.as_ref(),
        &second,
        [0.5, 1.0, -2.0, 1.0],
    );
    B::retire_context(&mut second_context).expect("peer execution must retire before teardown");
    drop(second_context);
    // The peer's loaded weight owns its storage after its execution context is gone.
    drop(second_linear);
    // Tearing down B cannot invalidate the loaded weights or stream used by A.
    check(
        &mut first_context,
        first_linear.as_ref(),
        &first,
        [-1.0, 0.5, 1.0, -2.0],
    );
    B::retire_context(&mut first_context).expect("surviving execution must retire before teardown");
}

#[test]
fn independent_loader_contexts_survive_peer_teardown() {
    interleaved_instances_preserve_loaded_weights::<CpuBackend>(CpuBackend::sync);
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn metal_independent_loader_contexts_survive_peer_teardown() {
    use ferrum_kernels::backend::metal::MetalBackend;
    interleaved_instances_preserve_loaded_weights::<MetalBackend>(|context| {
        MetalBackend::sync_checked(context).expect("independent loader work must complete");
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_independent_loader_contexts_survive_peer_teardown() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    interleaved_instances_preserve_loaded_weights::<CudaBackend>(CudaBackend::sync);
}
