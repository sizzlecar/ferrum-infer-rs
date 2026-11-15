use super::*;
use crate::fast_h3::vsa::*;
use crate::fast_h3::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
fn limits() -> FastH3PooledAttentionLimits {
    FastH3PooledAttentionLimits {
        maximum_batches: 2,
        maximum_heads: 64,
        maximum_tiles: 128,
        maximum_head_dim: 128,
        maximum_score_bytes: 8 << 20,
        maximum_metadata_bytes: 4096,
    }
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
fn managed(
    account: &HostMemoryAccount,
    name: &str,
    shape: Vec<u64>,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (slot, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(id(name).unwrap(), resolved(shape).unwrap(), bytes.freeze()).unwrap()
}
#[test]
fn actual_pool_geometry_produces_one_shared_scaled_score_boundary_and_checks_capacity() {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let layout = FastH3VsaLayout::ref2va(
        &c,
        3,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 4,
        }],
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 10000,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap();
    let pool = FastH3VsaPoolConfig::from_layout(
        &layout,
        2,
        c.transformer(),
        FastH3VsaPoolLimits {
            maximum_batches: 2,
            maximum_tiles: 128,
            maximum_columns: 16384,
            maximum_count_bytes: 512,
            maximum_metadata_bytes: 4096,
        },
    )
    .unwrap();
    let config = FastH3PooledAttentionConfig::from_pool(&pool, limits()).unwrap();
    assert_eq!(config.pooled_shape(), pool.output_shape());
    let (p, _) = program::build(&config).unwrap();
    let nodes = &p.blocks()[0].nodes;
    assert_eq!(nodes.len(), 4);
    assert_eq!(nodes[2].inputs, nodes[1].outputs);
    assert_eq!(
        nodes[1].outputs[0].as_str(),
        FastH3PooledAttentionFamily::SCORES_ID
    );
    assert_eq!(
        config.scale_bits(),
        ((1f64 / (pool.head_dim() as f64).sqrt()) as f32).to_bits()
    );
    let mut cap = limits();
    cap.maximum_score_bytes = resolved(config.score_shape())
        .unwrap()
        .minimum_storage_bytes()
        .unwrap()
        - 1;
    assert!(FastH3PooledAttentionConfig::from_pool(&pool, cap).is_err());
    let family = FastH3PooledAttentionFamily::new(limits()).unwrap();
    let mut wire = serde_json::to_value(&config).unwrap();
    wire["scale_bits"] = serde_json::json!(1f32.to_bits());
    assert!(family.parse_config(&wire).is_err());
}
#[test]
fn managed_pooled_attention_keeps_heads_batches_scores_and_f32_weighted_values_independent() {
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.pooled_attention").unwrap(),
        1 << 20,
    )
    .unwrap();
    qualify(
        composition.runtime().clone(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
    );
}
pub(super) fn qualify<R: DeviceRuntime>(
    runtime: std::sync::Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    let c = FastH3PooledAttentionConfig::new(2, 3, 5, 7, limits()).unwrap();
    let definition =
        TypedFamilyRegistration::new(FastH3PooledAttentionFamily::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &NumericalProfileId::new(FastH3PooledAttentionFamily::PROFILE_ID).unwrap(),
            )
            .unwrap();
    let descriptor = runtime.descriptor();
    let compiled = CompiledTensorComponent::compile(
        definition,
        registry,
        catalog,
        &TensorComponentPolicy::new(
            descriptor.total_memory_bytes,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap(),
        materializers,
        &c.compile_options(limits()).unwrap(),
    )
    .unwrap();
    let account = host();
    let mut session = compiled
        .initialize(
            runtime.clone(),
            account.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.pooled.init").unwrap(),
                request_id: id("request.pooled.init").unwrap(),
                transaction_id: id("transaction.pooled.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("initialization owner retained")
        });
    let q = (0..210)
        .map(|i| (i % 17) as f32 / 16. - 0.5)
        .collect::<Vec<_>>();
    let k = (0..210)
        .map(|i| (i % 13) as f32 / 32. - 0.25)
        .collect::<Vec<_>>();
    let v = (0..210)
        .map(|i| (i % 19) as f32 / 8. - 1.)
        .collect::<Vec<_>>();
    let mut scores = vec![0f32; 150];
    let mut answer = vec![0f64; 210];
    for bg in 0..6 {
        for m in 0..5 {
            for n in 0..5 {
                let dot = (0..7)
                    .map(|d| {
                        f64::from(q[(bg * 5 + m) * 7 + d]) * f64::from(k[(bg * 5 + n) * 7 + d])
                    })
                    .sum::<f64>();
                scores[(bg * 5 + m) * 5 + n] = (dot as f32) * f32::from_bits(c.scale_bits);
            }
            let row = &scores[(bg * 5 + m) * 5..(bg * 5 + m + 1) * 5];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = row.iter().map(|s| f64::from(*s - max).exp()).sum::<f64>();
            for d in 0..7 {
                answer[(bg * 5 + m) * 7 + d] = (0..5)
                    .map(|n| {
                        f64::from(row[n] - max).exp() / sum * f64::from(v[(bg * 5 + n) * 7 + d])
                    })
                    .sum();
            }
        }
    }
    let mut retained = None;
    for _ in 0..2 {
        let inputs = [
            (FastH3PooledAttentionFamily::QUERY_ID, &q),
            (FastH3PooledAttentionFamily::KEY_ID, &k),
            (FastH3PooledAttentionFamily::VALUE_ID, &v),
        ]
        .into_iter()
        .map(|(n, v)| managed(&account, n, c.pooled_shape(), v))
        .collect();
        let output = session
            .execute(
                id("run.pooled.execute").unwrap(),
                id("request.pooled.execute").unwrap(),
                inputs,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("execution owner retained")
            });
        assert_eq!(output.len(), 2);
        for item in &output {
            let values = item
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            if item.tensor().dimensions() == c.score_shape() {
                assert_eq!(values, scores);
            } else {
                assert_eq!(item.tensor().dimensions(), c.pooled_shape());
                for (a, b) in values.iter().zip(&answer) {
                    assert!((f64::from(*a) - b).abs() < 2e-7, "{a} != {b}");
                }
            }
        }
        retained = Some(output);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("close owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(account.usage().unwrap().allocated_bytes(), (150 + 210) * 4);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}
