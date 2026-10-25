use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
    TensorComponentPolicy,
};

#[path = "transforms/family.rs"]
pub(super) mod fixture;
#[path = "transforms/oracle.rs"]
pub(super) mod oracle;
use fixture::{Fixture, TransformFamily};

pub(super) const OPERATIONS: [MetalTensorOperation; 9] = [
    MetalTensorOperation::LayerNorm,
    MetalTensorOperation::HeadRmsNorm,
    MetalTensorOperation::ZeroCenteredRmsNorm,
    MetalTensorOperation::AdjacentRope,
    MetalTensorOperation::HalfSplitRope,
    MetalTensorOperation::StructuredAttention,
    MetalTensorOperation::StructuredGqaAttention,
    MetalTensorOperation::RowGather,
    MetalTensorOperation::BatchedRowConcat,
];
pub(super) fn spec(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
pub(super) fn upload_values(
    host: &HostMemoryAccount,
    name: ProgramValueId,
    spec: &ProgramTensorSpec,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            values.len() * spec.element_type.size_bytes() as usize,
        )
        .unwrap();
    for (dst, &value) in bytes
        .as_mut()
        .chunks_exact_mut(spec.element_type.size_bytes() as usize)
        .zip(values)
    {
        match spec.element_type {
            ElementType::Bf16 => {
                dst.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            ElementType::F32 => dst.copy_from_slice(&value.to_le_bytes()),
            _ => panic!("unsupported fixture input"),
        }
    }
    ManagedHostTensor::new(
        name,
        ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
pub(super) fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
pub(super) fn initialization(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        request_id: id(format!("request.init.{name}")),
        run_id: id(format!("run.init.{name}")),
        transaction_id: id(format!("transaction.init.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

pub(super) fn run_case(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
    fixture: &Fixture,
    values: &[Vec<f32>],
    check: impl Fn(&[f32]),
) {
    let prepared = TypedFamilyRegistration::new(TransformFamily)
        .prepare_with_profile(
            &serde_json::to_value(fixture).unwrap(),
            &id(fixture.kind().profile()),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(fixture.specs()).unwrap();
    if let Some(spec) = &fixture.output {
        options
            .insert_tensor_spec(id(OUTPUT), spec.clone())
            .unwrap();
    }
    let component = CompiledTensorComponent::compile(
        prepared,
        registry,
        catalog,
        policy,
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &options,
    )
    .unwrap();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization("metal.transform"),
        )
        .unwrap();
    let mut previous: Option<ManagedHostBytes> = None;
    for repeat in 0..2 {
        let input = fixture
            .inputs
            .iter()
            .zip(values)
            .enumerate()
            .map(|(i, (spec, values))| upload_values(host, input_id(i), spec, values))
            .collect();
        let output = session
            .execute(
                id(format!("run.metal.transform.{repeat}")),
                id(format!("request.metal.transform.{repeat}")),
                input,
            )
            .unwrap();
        assert_eq!(output.len(), 1);
        check(&decode(output[0].bytes().bytes()));
        if let Some(previous) = &previous {
            assert_eq!(output[0].bytes().bytes(), previous.bytes());
        }
        previous = Some(output[0].bytes().clone());
    }
    assert!(matches!(
        session.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    check(&decode(previous.as_ref().unwrap().bytes()));
    drop(previous);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    use MetalTensorOperation::*;
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let epsilon = CanonicalRational::new(1, 1_000_000).unwrap();
    for kind in [LayerNorm, HeadRmsNorm, ZeroCenteredRmsNorm] {
        for width in [1, 3, 6, 31, 128, 257, 4096, 8192] {
            let columns = if kind == HeadRmsNorm {
                3 * width
            } else {
                width
            };
            // Image head counts can exceed the grid cap. Exercise reuse of
            // one threadgroup's reduction storage across several groups.
            let rows = if kind == HeadRmsNorm && width == 3 {
                (65535 + 3) / 3
            } else {
                3
            };
            let mut inputs = vec![spec(vec![rows, columns], ElementType::Bf16)];
            let mut attributes =
                BTreeMap::from([(id("epsilon"), SemanticValue::Rational(epsilon))]);
            if kind != LayerNorm {
                inputs.push(spec(vec![width], ElementType::Bf16));
            }
            if kind != ZeroCenteredRmsNorm {
                attributes.insert(
                    id(if kind == LayerNorm {
                        "hidden_size"
                    } else {
                        "head_dim"
                    }),
                    SemanticValue::Unsigned(width),
                );
            }
            let fixture = Fixture::new(kind, inputs, attributes);
            let x: Vec<_> = (0..rows * columns)
                .map(|i| match (i / width) % 3 {
                    0 => 0.0,
                    1 => 2.0,
                    _ => rounds(((i * 17) % 101) as f32 / 16.0 - 3.0),
                })
                .collect();
            let gamma: Vec<_> = (0..width)
                .map(|i| rounds((i % 17) as f32 / 8.0 - 1.25))
                .collect();
            let values = if kind == LayerNorm {
                vec![x.clone()]
            } else {
                vec![x.clone(), gamma.clone()]
            };
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &fixture,
                &values,
                |actual| oracle::norm(kind, &x, &gamma, width as usize, 1e-6_f32, actual),
            );
            eprintln!("Metal {kind:?} rows={rows} width={width}: FP64 interval oracle, zero variance, repeated execution and retirement passed");
        }
        // A NaN poisons a group; infinity poisons LayerNorm and only the
        // infinite lanes of RMS. Width three also exercises tree padding.
        let mut attributes = BTreeMap::from([(id("epsilon"), SemanticValue::Rational(epsilon))]);
        if kind != ZeroCenteredRmsNorm {
            attributes.insert(
                id(if kind == LayerNorm {
                    "hidden_size"
                } else {
                    "head_dim"
                }),
                SemanticValue::Unsigned(3),
            );
        }
        let mut inputs = vec![spec(vec![2, 3], ElementType::Bf16)];
        if kind != LayerNorm {
            inputs.push(spec(vec![3], ElementType::Bf16));
        }
        let fixture = Fixture::new(kind, inputs, attributes);
        let x = vec![f32::NAN, 1.0, 2.0, f32::INFINITY, -2.0, 3.0];
        let gamma = vec![1.0; 3];
        let values = if kind == LayerNorm {
            vec![x.clone()]
        } else {
            vec![x.clone(), gamma.clone()]
        };
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &fixture,
            &values,
            |actual| oracle::norm(kind, &x, &gamma, 3, 1e-6, actual),
        );
    }
    for kind in [AdjacentRope, HalfSplitRope] {
        for (rows, sequence, heads, head) in [(6, 3, 2, 6), (4, 2, 3, 128), (2, 1, 1, 2)] {
            let columns = heads * head;
            let mut inputs = vec![spec(vec![rows, columns], ElementType::Bf16)];
            if kind == AdjacentRope {
                inputs.push(spec(vec![sequence, head / 2, 2], ElementType::F32));
            } else {
                inputs.extend([
                    spec(vec![rows, head], ElementType::Bf16),
                    spec(vec![rows, head], ElementType::Bf16),
                ]);
            }
            let fixture = Fixture::new(
                kind,
                inputs,
                BTreeMap::from([(id("head_dim"), SemanticValue::Unsigned(head))]),
            );
            let mut x: Vec<_> = (0..rows * columns)
                .map(|i| rounds(((i * 53) % 127) as f32 / 16.0 - 4.0))
                .collect();
            let count = if kind == AdjacentRope {
                sequence * head
            } else {
                rows * head
            };
            let mut cosine: Vec<_> = (0..count)
                .map(|i| {
                    if kind == AdjacentRope {
                        ((i * 13) % 59) as f32 / 17.0 - 1.5
                    } else {
                        rounds(((i * 13) % 59) as f32 / 17.0 - 1.5)
                    }
                })
                .collect();
            let sine: Vec<_> = (0..count)
                .map(|i| rounds(((i * 31) % 71) as f32 / 23.0 - 1.0))
                .collect();
            if kind == AdjacentRope {
                x[0] = bf16::from_bits(0x3f81).to_f32();
                x[1] = 1.0;
                cosine[0] = f32::from_bits(0x3f7fffff);
                cosine[1] = f32::from_bits(0x3f810000);
                // Explicit FMA retains this cancellation residual; two rounded
                // multiplies plus subtraction incorrectly produce zero.
                assert_eq!(
                    bf16::from_f32(libm::fmaf(x[0], cosine[0], -(x[1] * cosine[1]))).to_bits(),
                    0xb381
                );
            }
            let values = if kind == AdjacentRope {
                vec![x.clone(), cosine.clone()]
            } else {
                vec![x.clone(), cosine.clone(), sine.clone()]
            };
            run_case(
                runtime,
                registry,
                catalog,
                &policy,
                host,
                &fixture,
                &values,
                |actual| {
                    oracle::rope(
                        kind,
                        &x,
                        &cosine,
                        &sine,
                        columns as usize,
                        head as usize,
                        sequence as usize,
                        actual,
                    )
                },
            );
            eprintln!("Metal {kind:?} rows={rows} sequence={sequence} heads={heads} head_dim={head}: exact staged arithmetic, repeat and retirement passed");
        }
    }
}
