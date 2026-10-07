use super::super::test_support::Guarded;
use super::tests::matrix;
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use cudarc::driver::CudaContext;
use ferrum_interfaces::vnext::*;
use half::f16;
use weights::{MatrixFormat, MatrixPart};

mod attention;
mod three_format;

fn id<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    s.to_owned().try_into().unwrap()
}

fn binding(ordinal: u32, parts: &[MatrixPart], k: usize, n: usize) -> ResolvedValueBinding {
    let packed = ordinal == 1;
    let shape = if packed {
        vec![2, n as u64, k as u64]
    } else {
        vec![n as u64, k as u64]
    };
    let mut components = Vec::new();
    let mut layouts = Vec::new();
    for part in parts {
        let (encoding, physical_k, role, layout) = match part.format {
            MatrixFormat::DenseF16 => (
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                k,
                WeightComponentRole::Values,
                PhysicalWeightLayout::Dense {
                    component_id: part.component_id.clone(),
                },
            ),
            MatrixFormat::Block(format) => (
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: id(format.format_id()),
                    logical_values_per_block: format.block_values() as u32,
                    bytes_per_block: format.block_bytes() as u32,
                }),
                k / format.block_values(),
                WeightComponentRole::PackedValues,
                PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(
                        part.component_id.clone(),
                    ),
                    block_axis: if packed { 2 } else { 1 },
                    block_padding: PhysicalWeightPadding::Exact,
                },
            ),
        };
        components.push(WeightComponentSpec {
            id: part.component_id.clone(),
            role,
            external_names: vec![part.component_id.to_string()],
            dimensions: if packed {
                vec![1, part.rows as u64, physical_k as u64]
            } else {
                vec![part.rows as u64, physical_k as u64]
            },
            encoding,
            required: true,
        });
        layouts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets: if packed {
                vec![u64::from(part.output_offset) / (n as u64), 0, 0]
            } else {
                vec![0, 0]
            },
            extents: if packed {
                vec![1, n as u64, k as u64]
            } else {
                shape.clone()
            },
        });
    }
    let weight_id: WeightId = id(&format!("weight.{ordinal}"));
    let schema = WeightSchema {
        format_id: id("weight-format.gguf.native-block"),
        layout_id: id("layout.q8act-fixture"),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: weight_id.clone(),
            dimensions: shape.clone(),
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Composite { parts: layouts },
            required: true,
        }],
    };
    let weight = ResolvedWeightBinding::from_schema(&schema, &weight_id).unwrap();
    let storage = ResolvedValueStorage::composite(
        schema
            .components
            .iter()
            .map(|component| {
                ResolvedStorageComponent::new(
                    Some(component.id.clone()),
                    id(&format!("resource.{}", component.id)),
                    0,
                    component.physical_bytes().unwrap(),
                    component.physical_element_type(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    ResolvedValueBinding::new(
        id(&format!("value.{ordinal}")),
        ResolvedValueRole::Input,
        ordinal,
        ResolvedTensorSpec::new(shape, ElementType::F16, ResolvedTensorLayout::Contiguous).unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(weight),
        storage,
    )
    .unwrap()
}

fn projection_oracle(
    input: &[f16],
    weight: &[f32],
    actual: &[f16],
    k: usize,
    n: usize,
    stride: usize,
    offset: usize,
    q8: bool,
) {
    for (row, source) in input.chunks_exact(k).enumerate() {
        let x = if q8 {
            assert_eq!(k % 32, 0);
            source
                .chunks_exact(32)
                .flat_map(|group| {
                    let amax = group
                        .iter()
                        .map(|x| x.to_f32().abs())
                        .fold(0.0_f32, f32::max);
                    let scale = amax / 127.0;
                    group.iter().map(move |x| {
                        if scale == 0.0 {
                            0.0
                        } else {
                            f64::from((x.to_f32() / scale).round().clamp(-127.0, 127.0))
                                * f64::from(scale)
                        }
                    })
                })
                .collect::<Vec<_>>()
        } else {
            source.iter().map(|x| x.to_f64()).collect::<Vec<_>>()
        };
        for column in 0..n {
            let mut sum = 0.0;
            let mut absolute = 0.0;
            for (&x, &w) in x.iter().zip(&weight[column * k..][..k]) {
                let p = x * f64::from(w);
                sum += p;
                absolute += p.abs();
            }
            // Independent CPU block decode + F64 policy dot. K-term FP32
            // rounding bound includes weight reconstruction and rescaling;
            // final F16 rounding and one minimum subnormal cover cancellation.
            let bound = (k as f64 * f32::EPSILON as f64 + 0.0009765625) * absolute
                + f16::from_bits(1).to_f64();
            let got = actual[row * stride + offset + column].to_f64();
            assert!(
                got.is_finite() && (got - sum).abs() <= bound,
                "q8={q8}, row={row} col={column}: {got} vs {sum}, bound={bound}"
            );
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device and the IQ4_XS G32 kernel overlay"]
fn q8act_swiglu_mixed_leaves_match_stage_oracles_for_scalar_tail_and_prefill_on_cuda() {
    let context = CudaContext::new(0).expect("Q8act SwiGLU conformance requires CUDA");
    let stream = context.default_stream();
    let native = CudaNativeBlockKernels::load(&context).unwrap();
    let q8 = Q8ActKernels::load(&context).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let hidden = 256;
    for (intermediate, formats) in [
        (
            256,
            [
                MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
                MatrixFormat::Block(GgufBlockFormat::Q4K),
                MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
            ],
        ),
        (
            17,
            [
                MatrixFormat::Block(GgufBlockFormat::Q4K),
                MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
                MatrixFormat::DenseF16,
            ],
        ),
    ] {
        let part = |i: usize, n: usize, k: usize, offset: usize| MatrixPart {
            component_id: id(&format!("component.{i}")),
            format: formats[i],
            rows: n as u32,
            columns: k as u32,
            output_offset: offset as u32,
            transform: None,
            signs_region: None,
        };
        let gate_up = [
            part(0, intermediate, hidden, 0),
            part(1, intermediate, hidden, intermediate),
        ];
        let down = [part(2, hidden, intermediate, 0)];
        let values = vec![
            binding(1, &gate_up, hidden, intermediate),
            binding(2, &down, intermediate, hidden),
        ];
        let plan = PreparedProjectionNumerics::prepare(
            &dense_swiglu_iq4xs_q8act_g32_arithmetic(),
            &values,
        )
        .unwrap();
        for (role, parts) in [
            (ProjectionRole::SwiGluGateUp, gate_up.as_slice()),
            (ProjectionRole::SwiGluDown, down.as_slice()),
        ] {
            Q8ActKernels::validate_parts(plan.projection(role).unwrap(), parts).unwrap();
        }
        let host_weights = [
            matrix(formats[0], intermediate, hidden, 0),
            matrix(formats[1], intermediate, hidden, 1),
            matrix(formats[2], hidden, intermediate, 2),
        ];
        let gpu_weights = host_weights
            .iter()
            .map(|(bytes, _)| Guarded::new(&stream, bytes, 0xAB_u8))
            .collect::<Vec<_>>();
        let pointers = gpu_weights
            .iter()
            .map(|w| w.pointer(&stream))
            .collect::<Vec<_>>();
        for rows in [1, 3, 8, 9, 1025] {
            let input = (0..rows * hidden)
                .map(|i| f16::from_f32(((i * 13 % 17) as f32 - 8.0) / 32768.0))
                .collect::<Vec<_>>();
            let x = Guarded::new(&stream, &input, f16::from_f32(79.0));
            let bytes = q8act::workspace_per_token(&plan).unwrap() * rows as u64;
            let pack = Guarded::new(&stream, &vec![0xCB_u8; bytes as usize], 0xAB_u8);
            let mut previous = None;
            for _ in 0..2 {
                let guard = f16::from_f32(-12345.0);
                let gates = Guarded::new(&stream, &vec![f16::NAN; rows * 2 * intermediate], guard);
                let act = Guarded::new(&stream, &vec![f16::NAN; rows * intermediate], guard);
                let y = Guarded::new(&stream, &vec![f16::NAN; rows * hidden], guard);
                launch(
                    &native,
                    &silu,
                    &stream,
                    &gate_up,
                    &down,
                    &pointers,
                    x.pointer(&stream),
                    y.pointer(&stream),
                    gates.pointer(&stream),
                    act.pointer(&stream),
                    rows as u32,
                    hidden as u32,
                    intermediate as u32,
                    0,
                    Some(&q8),
                    Some(&plan),
                    pack.pointer(&stream),
                    bytes,
                )
                .unwrap();
                let gate = gates.read(&stream);
                let activation = act.read(&stream);
                let output = y.read(&stream);
                projection_oracle(
                    &input,
                    &host_weights[0].1,
                    &gate,
                    hidden,
                    intermediate,
                    2 * intermediate,
                    0,
                    plan.projections()[0].leaves()[0].is_staged(),
                );
                projection_oracle(
                    &input,
                    &host_weights[1].1,
                    &gate,
                    hidden,
                    intermediate,
                    2 * intermediate,
                    intermediate,
                    plan.projections()[0].leaves()[1].is_staged(),
                );
                for row in 0..rows {
                    for col in 0..intermediate {
                        let g = gate[row * 2 * intermediate + col].to_f64();
                        let u = gate[row * 2 * intermediate + intermediate + col].to_f64();
                        let reference = g / (1.0 + (-g).exp()) * u;
                        let actual = activation[row * intermediate + col].to_f64();
                        assert!(
                            actual.is_finite()
                                && (actual - reference).abs()
                                    <= 0.001 * reference.abs() + f16::from_bits(1).to_f64()
                        );
                    }
                }
                projection_oracle(
                    &activation,
                    &host_weights[2].1,
                    &output,
                    intermediate,
                    hidden,
                    hidden,
                    0,
                    plan.projections()[1].leaves()[0].is_staged(),
                );
                let stages = [gate, activation, output]
                    .map(|values| values.into_iter().map(f16::to_bits).collect::<Vec<_>>());
                if let Some(prior) = &previous {
                    assert_eq!(&stages, prior, "own-route repeat rows={rows}");
                }
                previous = Some(stages);
            }
            x.assert_unchanged(&stream);
            let _ = pack.read(&stream);
        }
        for w in gpu_weights {
            w.assert_unchanged(&stream);
        }
    }
}
