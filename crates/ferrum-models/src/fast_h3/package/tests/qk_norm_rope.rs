use super::*;
use crate::fast_h3::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use half::bf16;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn qk_limits() -> FastH3QkNormRopeLimits {
    FastH3QkNormRopeLimits {
        maximum_rows: 1024,
        maximum_metadata_bytes: 8192,
        maximum_stored_bytes: 4096,
    }
}
#[test]
fn actual_qk_norm_source_consumes_managed_ref2va_positions_and_preserves_output_owners() {
    let mut f = Fixture::new();
    for (axis, value) in [("q", 2.0), ("k", 3.0)] {
        for b in f
            .dense
            .get_mut(&format!("transformer_blocks.0.attn.norm_{axis}.weight"))
            .unwrap()
            .2
            .chunks_exact_mut(2)
        {
            b.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    let (_dir, p) = f.open(limits());
    let p = p.unwrap();
    let positions = FastH3PositionPlan::ref2va(
        3,
        &[
            FastH3Reference::Image {
                latent_height: 2,
                latent_width: 4,
            },
            FastH3Reference::Audio {
                latents_per_channel: 3,
            },
        ],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        FastH3PositionLimits {
            max_sequence_rows: 1024,
            max_regions: 16,
        },
    )
    .unwrap();
    let c = FastH3QkNormRopeConfig::from_package(
        &p,
        0,
        positions.layout().sequence_rows() as u64,
        qk_limits(),
    )
    .unwrap();
    let source = FastH3QkNormRopeSource::new(&p, &c, qk_limits()).unwrap();
    let prepared = TypedFamilyRegistration::new(FastH3QkNormRopeFamily::new(qk_limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(FastH3QkNormRopeFamily::PROFILE_ID),
        )
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("original Stored norm weights")
    };
    assert_eq!(schema.components.len(), 2);
    for component in &schema.components {
        let payload = source.component(component).unwrap();
        assert_eq!(payload.bytes().len(), c.head_dim() as usize * 2);
        assert!(payload.retained_host_memory().is_some());
    }
    let composition =
        ReferenceVNextComposition::create_with_capacity(id("device.reference.fast_h3.qk"), 4 << 20)
            .unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &c.compile_options().unwrap(),
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            2 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2 << 20,
                transfer_staging_bytes: 2 << 20,
                output_bytes: 2 << 20,
                weight_backing_bytes: 2 << 20,
            },
        )
        .unwrap(),
    );
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.qk.source.init"),
                request_id: id("request.qk.source.init"),
                transaction_id: id("transaction.qk.source.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("QK init failed; owner retained")
        });
    let mut retained = None;
    for trial in 0..2 {
        let (cos, sin) = positions
            .managed_bf16_frequencies(
                p.config(),
                &host,
                id(FastH3QkNormRopeFamily::COSINE_ID),
                id(FastH3QkNormRopeFamily::SINE_ID),
            )
            .unwrap();
        let cosine = decode(cos.bytes().bytes());
        let sine = decode(sin.bytes().bytes());
        let mut inputs = vec![cos, sin];
        for name in [
            FastH3QkNormRopeFamily::QUERY_ID,
            FastH3QkNormRopeFamily::KEY_ID,
        ] {
            let mut bytes = host
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    (c.rows() * c.heads() * c.head_dim() * 2) as usize,
                )
                .unwrap();
            for b in bytes.as_mut().chunks_exact_mut(2) {
                b.copy_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
            }
            inputs.push(
                ManagedHostTensor::new(
                    id(name),
                    ResolvedTensorSpec::new(
                        vec![c.rows(), c.heads() * c.head_dim()],
                        ElementType::Bf16,
                        ResolvedTensorLayout::Contiguous,
                    )
                    .unwrap(),
                    bytes.freeze(),
                )
                .unwrap(),
            );
        }
        let outputs = session
            .execute(
                id(format!("run.qk.source.{trial}")),
                id(format!("request.qk.source.{trial}")),
                inputs,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("QK source execution failed; owner retained")
            });
        for out in &outputs {
            let gamma = if out.value_id().as_str() == FastH3QkNormRopeFamily::QUERY_OUTPUT_ID {
                2.0
            } else {
                assert_eq!(
                    out.value_id().as_str(),
                    FastH3QkNormRopeFamily::KEY_OUTPUT_ID
                );
                3.0
            };
            let inverse = (1.0 / ((1.0f32 + p.config().qk_norm_eps() as f32) as f64).sqrt()) as f32;
            for (i, value) in decode(out.bytes().bytes()).into_iter().enumerate() {
                let row = i / (c.heads() * c.head_dim()) as usize;
                let col = i % c.head_dim() as usize;
                let norm = inverse * gamma;
                let expected = if col < c.rotary_dim() as usize {
                    let table = row * c.rotary_dim() as usize + col;
                    let rotated = if col < (c.rotary_dim() / 2) as usize {
                        (-inverse).mul_add(gamma, 0.0)
                    } else {
                        norm
                    };
                    norm.mul_add(cosine[table], rotated * sine[table])
                } else {
                    norm
                };
                assert_eq!(
                    bf16::from_f32(value).to_bits(),
                    bf16::from_f32(expected).to_bits(),
                    "row{row}, channel{col}"
                );
            }
        }
        retained = Some(outputs);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("QK source close failed; owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let retained = retained.unwrap();
    assert!(retained.iter().all(|o| host.owns(o.bytes())));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
#[test]
fn qk_source_rejects_changed_origin_shape_and_capacity() {
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    let c = FastH3QkNormRopeConfig::from_package(&p, 0, 3, qk_limits()).unwrap();
    for (key, value) in [
        ("block", 1),
        ("head_dim", 8),
        ("heads", 2),
        ("rotary_dim", 6),
    ] {
        let mut wire = serde_json::to_value(&c).unwrap();
        wire[key] = value.into();
        let altered: FastH3QkNormRopeConfig = serde_json::from_value(wire).unwrap();
        assert!(FastH3QkNormRopeSource::new(&p, &altered, qk_limits()).is_err());
    }
    let mut small = qk_limits();
    small.maximum_stored_bytes = 63;
    assert!(FastH3QkNormRopeConfig::from_package(&p, 0, 3, small).is_err());
    small = qk_limits();
    small.maximum_rows = 2;
    assert!(FastH3QkNormRopeConfig::from_package(&p, 0, 3, small).is_err());
    assert!(FastH3QkNormRopeConfig::from_package(&p, p.config().layers(), 3, qk_limits()).is_err());
}
