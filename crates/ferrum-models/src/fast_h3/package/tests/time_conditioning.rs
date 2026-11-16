use super::*;
use crate::fast_h3::{
    FastH3TimeConditioningConfig as Config, FastH3TimeConditioningFamily as Family,
    FastH3TimeConditioningLimits, FastH3TimeConditioningSource,
};
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

mod table;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn time_limits() -> FastH3TimeConditioningLimits {
    FastH3TimeConditioningLimits {
        maximum_time_rows: 8,
        maximum_metadata_bytes: 16384,
        maximum_stored_bytes: 1 << 20,
    }
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
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
fn populated_fixture() -> Fixture {
    let mut f = Fixture::new();
    for (name, (_, _, bytes)) in &mut f.dense {
        if name.starts_with("time_embedder.")
            || name.starts_with("transformer_blocks.0.adaln_proj.")
        {
            for (i, chunk) in bytes.chunks_exact_mut(2).enumerate() {
                let v = ((i * 7 % 17) as f32 - 8.) / 64.;
                chunk.copy_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
        }
    }
    f
}
fn parameter(f: &Fixture, name: &str) -> Vec<f32> {
    f.dense[name]
        .2
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
fn linear(x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k)
        .zip(bias)
        .map(|(row, &bias)| {
            let dot: f64 = row
                .iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            (dot as f32) + bias
        })
        .collect()
}
fn activate(x: f32) -> f32 {
    x / (1. + ((-f64::from(x)).exp() as f32))
}

#[test]
fn time_conditioning_uses_managed_promotion_and_preserves_f32_then_bf16_boundaries() {
    let fixture = populated_fixture();
    let (_dir, package) = fixture.open(limits());
    let package = package.unwrap();
    let config = Config::from_package(&package, 0, 3, time_limits()).unwrap();
    let family = Family::new(time_limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&config).unwrap() else {
        panic!("Stored time weights")
    };
    let account = host();
    let source =
        FastH3TimeConditioningSource::new(&package, &config, &account, time_limits()).unwrap();
    let payload = source.component(&schema.components[0]).unwrap();
    assert_eq!(payload.element_type(), ElementType::F32);
    assert_eq!(payload.bytes().len(), 32 * 8 * 4);
    assert_eq!(
        f32::from_le_bytes(payload.bytes()[0..4].try_into().unwrap()),
        -0.125
    );
    assert!(payload.retained_host_memory().is_some());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 32 * 8 * 4);
    drop(payload);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);

    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &id(Family::PROFILE_ID),
        )
        .unwrap();
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fast_h3.time")).unwrap();
    let descriptor = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            descriptor.total_memory_bytes,
            descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        &config.compile_options().unwrap(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            account.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.time.init"),
                request_id: id("request.time.init"),
                transaction_id: id("transaction.time.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let features: Vec<_> = (0..24).map(|i| (i as f32 - 9.0) / 32.).collect();
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, features.len() * 4)
        .unwrap();
    for (dst, x) in bytes.as_mut().chunks_exact_mut(4).zip(&features) {
        dst.copy_from_slice(&x.to_le_bytes());
    }
    let outputs = session
        .execute(
            id("run.time.forward"),
            id("request.time.forward"),
            vec![ManagedHostTensor::new(
                id(Family::INPUT_ID),
                ResolvedTensorSpec::new(
                    vec![3, 8],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap()],
        )
        .unwrap();
    let embedding = outputs
        .iter()
        .find(|o| o.value_id().as_str() == Family::EMBEDDING_ID)
        .unwrap();
    let modulation = outputs
        .iter()
        .find(|o| o.value_id().as_str() == Family::OUTPUT_ID)
        .unwrap();
    assert_eq!(embedding.tensor().dimensions(), [3, 16]);
    assert_eq!(modulation.tensor().dimensions(), [3, 576]);
    let actual_embedding: Vec<_> = embedding
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let actual_modulation: Vec<_> = modulation
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    let w1 = parameter(&fixture, "time_embedder.linear_1.weight");
    let b1 = parameter(&fixture, "time_embedder.linear_1.bias");
    let w2 = parameter(&fixture, "time_embedder.linear_2.weight");
    let b2 = parameter(&fixture, "time_embedder.linear_2.bias");
    let wa = parameter(&fixture, "transformer_blocks.0.adaln_proj.linear.weight");
    let ba = parameter(&fixture, "transformer_blocks.0.adaln_proj.linear.bias");
    for (row, x) in features.chunks_exact(8).enumerate() {
        let activated: Vec<_> = linear(x, &w1, &b1).into_iter().map(activate).collect();
        let expected = linear(&activated, &w2, &b2);
        for (&actual, &expected) in actual_embedding[row * 16..(row + 1) * 16]
            .iter()
            .zip(&expected)
        {
            // Independent F64 dot and exp oracle; finite tiny-fixture bound,
            // not a general libm/libdevice/Torch equality qualification.
            assert!((actual - expected).abs() <= 2e-6, "{actual} vs {expected}");
        }
        let adaln_input: Vec<_> = expected
            .into_iter()
            .map(|v| bf16::from_f32(activate(v)).to_f32())
            .collect();
        let expected = linear(&adaln_input, &wa, &ba);
        for (&actual, expected) in actual_modulation[row * 576..(row + 1) * 576]
            .iter()
            .zip(expected)
        {
            assert_eq!(
                actual.to_bits(),
                bf16::from_f32(expected).to_f32().to_bits()
            );
        }
    }
    assert!(actual_embedding
        .iter()
        .any(|&v| bf16::from_f32(v).to_f32() != v));
    drop(outputs);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("{:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

#[test]
fn time_conditioning_rejects_source_substitution_and_true_capacity_boundaries() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let config = Config::from_package(&package, 0, 3, time_limits()).unwrap();
    let family = Family::new(time_limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&config).unwrap() else {
        panic!("Stored")
    };
    let mut bounded = time_limits();
    bounded.maximum_stored_bytes = schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum::<u64>()
        - 1;
    assert!(Config::from_package(&package, 0, 3, bounded).is_err());
    assert!(Config::from_package(&package, 2, 3, time_limits()).is_err());
    assert!(Config::from_package(&package, 0, 0, time_limits()).is_err());
    let account = host();
    let source =
        FastH3TimeConditioningSource::new(&package, &config, &account, time_limits()).unwrap();
    let mut forged = schema.components[0].clone();
    forged.external_names = vec!["other.weight".into()];
    assert!(source.component(&forged).is_err());
    let bytes = schema.components[0].physical_bytes().unwrap() as usize;
    let tiny = HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes - 1,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: bytes - 1,
            },
        )
        .unwrap(),
    );
    let source =
        FastH3TimeConditioningSource::new(&package, &config, &tiny, time_limits()).unwrap();
    assert!(source.component(&schema.components[0]).is_err());
    assert_eq!(tiny.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn conditioning_clock_features_use_the_components_shape_and_original_host_budget() {
    use crate::fast_h3::{
        FastH3ConditioningClock, FastH3ConditioningSchedule, FastH3PackedLayout, FastH3PddPlan,
        FastH3Reference, FastH3VideoGeometry,
    };
    let checkpoint = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 64,
        },
    )
    .unwrap();
    let schedule = FastH3PddPlan::new(&checkpoint).unwrap();
    let layout = FastH3PackedLayout::ref2va(
        7,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        128,
    )
    .unwrap();
    let clock = FastH3ConditioningClock::new(&layout, &schedule.steps()[1]).unwrap();
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let config =
        Config::from_package(&package, 0, clock.timesteps().len() as u64, time_limits()).unwrap();
    let account = host();
    let input = config.managed_features(&account, &clock).unwrap();
    assert_eq!(input.tensor().dimensions(), [3, 8]);
    assert_eq!(input.tensor().element_type(), ElementType::F32);
    assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
    let first = f32::from_le_bytes(input.bytes().bytes()[..4].try_into().unwrap());
    assert!((f64::from(first) - f64::from(clock.timesteps()[0]).cos()).abs() < 1e-7);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 96);
    drop(input);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let all_times = FastH3ConditioningSchedule::all_modalities(&schedule).unwrap();
    let mut all_limits = time_limits();
    all_limits.maximum_time_rows = all_times.timesteps().len() as u64;
    let all_config =
        Config::from_package(&package, 0, all_limits.maximum_time_rows, all_limits).unwrap();
    let all_input = all_config
        .managed_schedule_features(&account, &all_times)
        .unwrap();
    let row_bytes = config.frequency_dim() as usize * 4;
    let input = config.managed_features(&account, &clock).unwrap();
    for (i, t) in clock.timesteps().iter().enumerate() {
        let global = all_times.timesteps().iter().position(|v| v == t).unwrap();
        assert_eq!(
            &input.bytes().bytes()[i * row_bytes..(i + 1) * row_bytes],
            &all_input.bytes().bytes()[global * row_bytes..(global + 1) * row_bytes]
        );
    }
    assert!(config
        .managed_schedule_features(&account, &all_times)
        .is_err());
    drop(input);
    drop(all_input);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let wrong_rows = Config::from_package(&package, 0, 4, time_limits()).unwrap();
    assert!(wrong_rows.managed_features(&account, &clock).is_err());
    let mut wire = serde_json::to_value(&config).unwrap();
    wire["frequency"] = 0.into();
    let malformed: Config = serde_json::from_value(wire).unwrap();
    assert!(malformed.managed_features(&account, &clock).is_err());
    let tiny = HostMemoryAccount::new(
        HostMemoryLimits::new(
            95,
            HostMemoryClassLimits {
                input_preprocess_bytes: 95,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let failure = config.managed_features(&tiny, &clock).err().unwrap();
    assert!(failure.downcast_ref::<HostMemoryError>().is_some());
    assert_eq!(tiny.usage().unwrap().allocated_bytes(), 0);
}
