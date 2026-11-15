//! Actual package -> fused Stored source -> managed F32 affine execution.
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
use half::{bf16, f16};
use std::sync::atomic::{AtomicUsize, Ordering};

type Family = FastH3PddProjectionFamily;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn projection_limits() -> FastH3PddProjectionLimits {
    FastH3PddProjectionLimits {
        maximum_rows: 16,
        maximum_metadata_bytes: 16384,
        maximum_source_bytes: 1 << 20,
        maximum_fused_bytes: 1 << 20,
    }
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
pub(super) fn plan() -> FastH3PddPlan {
    let mut transformer: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../fixtures/transformer.json")).unwrap();
    // Match the independently built tiny package's output channels. Other
    // transformer geometry is unused by the immutable schedule.
    transformer["in_channels"] = 2.into();
    transformer["audio_in_channels"] = 4.into();
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        &serde_json::to_vec(&transformer).unwrap(),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    FastH3PddPlan::new(&c).unwrap()
}
pub(super) fn fixture() -> Fixture {
    let mut f = Fixture::new();
    for (prefix, output, weight_dtype, bias_dtype) in [
        ("proj_out", 8, Dtype::BF16, Dtype::F32),
        ("audio_proj_out", 4, Dtype::F16, Dtype::BF16),
    ] {
        for (suffix, dtype, width) in [("weight", weight_dtype, 32), ("bias", bias_dtype, 1)] {
            let mut bytes = Vec::new();
            for head in 0..32 {
                for channel in 0..output {
                    for column in 0..width {
                        let v = ((head * 23 + channel * 7 + column * 3) % 37) as f32 / 64.0 - 0.25;
                        match dtype {
                            Dtype::F32 => {
                                bytes.extend_from_slice(&(v + 1.0 / 131072.0).to_le_bytes())
                            }
                            Dtype::F16 => {
                                bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes())
                            }
                            Dtype::BF16 => {
                                bytes.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes())
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            let shape = if suffix == "weight" {
                vec![32 * output, width]
            } else {
                vec![32 * output]
            };
            f.dense
                .insert(format!("{prefix}.{suffix}"), (dtype, shape, bytes));
        }
    }
    f
}
pub(super) fn decode(t: &Tensor) -> Vec<f32> {
    match t.0 {
        Dtype::F32 => {
            t.2.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        Dtype::BF16 => {
            t.2.chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect()
        }
        Dtype::F16 => {
            t.2.chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect()
        }
        _ => unreachable!(),
    }
}
/// Independently decode channel-major heads. F64 operations followed by F32
/// narrowing model each declared RN32 boundary without generated buffers.
pub(super) fn fused(
    raw: &[f32],
    per_head: usize,
    heads: std::ops::Range<usize>,
    coeff: &[f32],
) -> Vec<f32> {
    (0..per_head)
        .map(|i| {
            heads.clone().zip(coeff).fold(0f32, |sum, (head, &c)| {
                let p = (f64::from(raw[head * per_head + i]) * f64::from(c)) as f32;
                (f64::from(sum) + f64::from(p)) as f32
            })
        })
        .collect()
}
struct CountedSource {
    inner: FastH3PddProjectionSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for CountedSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
fn managed(account: &HostMemoryAccount, x: &[f32]) -> ManagedHostTensor {
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, x.len() * 4)
        .unwrap();
    for (b, x) in bytes.as_mut().chunks_exact_mut(4).zip(x) {
        b.copy_from_slice(&x.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(Family::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![3, 32],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}

#[test]
fn actual_widened_heads_feed_one_f32_affine_with_independent_oracle_and_original_owners() {
    let fixture = fixture();
    let plan = plan();
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fast_h3.pdd")).unwrap();
    for (modality, evaluation, prefix, output) in [
        (FastH3OutputModality::Video, 1, "proj_out", 8),
        (FastH3OutputModality::Audio, 7, "audio_proj_out", 4),
    ] {
        let (_dir, package) = fixture.open(limits());
        let package = package.unwrap();
        let account = host(1 << 20);
        let (config, source) = FastH3PddProjectionSource::prepare(
            &package,
            &plan,
            evaluation,
            modality,
            3,
            &account,
            projection_limits(),
        )
        .unwrap();
        drop(package); // Prepared parameters retain only admitted Host owners.
        let step = &plan.steps()[evaluation];
        let coeff = match modality {
            FastH3OutputModality::Video => step.video.normalized_head_weights(),
            FastH3OutputModality::Audio => step.audio.normalized_head_weights(),
        };
        let w = fused(
            &decode(&fixture.dense[&format!("{prefix}.weight")]),
            output * 32,
            step.heads.clone(),
            coeff,
        );
        let b = fused(
            &decode(&fixture.dense[&format!("{prefix}.bias")]),
            output,
            step.heads.clone(),
            coeff,
        );
        let prepared = TypedFamilyRegistration::new(Family::new(projection_limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &id(Family::PROFILE_ID),
            )
            .unwrap();
        let nodes = &prepared.program().blocks()[0].nodes;
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0].operation_id.as_str(),
            AFFINE_LINEAR_F32_OPERATION_ID
        );
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("Stored fused parameters")
        };
        assert_eq!(schema.components.len(), 2);
        for component in &schema.components {
            let expected = if component.dimensions.len() == 2 {
                &w
            } else {
                &b
            };
            assert_eq!(component.role, WeightComponentRole::Values);
            assert_eq!(
                component.encoding,
                WeightEncoding::Dense {
                    element_type: ElementType::F32
                }
            );
            let payload = source.component(component).unwrap();
            assert!(payload.retained_host_memory().is_some());
            assert_eq!(
                payload.bytes(),
                expected
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            (output * 32 + output) * 4
        );
        let source = CountedSource {
            inner: source,
            reads: AtomicUsize::new(0),
        };
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
            &config.compile_options().unwrap(),
        )
        .unwrap();
        let mut session = compiled
            .initialize(
                composition.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                ComponentInitializationConfig {
                    run_id: id("run.pdd.init"),
                    request_id: id("request.pdd.init"),
                    transaction_id: id("transaction.pdd.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        assert_eq!(source.reads.load(Ordering::Relaxed), 2);
        let x: Vec<_> = (0..96)
            .map(|i| ((i * 11 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let mut retained = None;
        let mut first = None;
        for repeat in 0..2 {
            let outputs = session
                .execute(
                    id(format!("run.pdd.{repeat}")),
                    id(format!("request.pdd.{repeat}")),
                    vec![managed(&account, &x)],
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].tensor().element_type(), ElementType::F32);
            assert_eq!(outputs[0].tensor().dimensions(), [3, output as u64]);
            let actual: Vec<_> = outputs[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            for (row, x) in x.chunks_exact(32).enumerate() {
                for (col, w) in w.chunks_exact(32).enumerate() {
                    let dot: f64 = x
                        .iter()
                        .zip(w)
                        .map(|(&x, &w)| f64::from(x) * f64::from(w))
                        .sum();
                    let magnitude: f64 = x
                        .iter()
                        .zip(w)
                        .map(|(&x, &w)| (f64::from(x) * f64::from(w)).abs())
                        .sum();
                    let exact = dot + f64::from(b[col]);
                    let u = 2f64.powi(-24);
                    let gamma = 64.0 * u / (1.0 - 64.0 * u);
                    let bound = (1.0 + u) * gamma * magnitude + u * exact.abs();
                    assert!(actual[row * output + col].is_finite());
                    assert!(
                        (f64::from(actual[row * output + col]) - exact).abs() <= bound,
                        "F32 affine oracle bound"
                    );
                }
            }
            assert!(
                actual.iter().any(|&v| bf16::from_f32(v).to_f32() != v),
                "fixture exposes BF16 output casts"
            );
            let bits: Vec<_> = actual.iter().map(|v| v.to_bits()).collect();
            if let Some(first) = &first {
                assert_eq!(&bits, first);
            } else {
                first = Some(bits);
            }
            retained = Some(outputs);
        }
        assert_eq!(source.reads.load(Ordering::Relaxed), 2);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("{:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        drop(source);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 3 * output * 4);
        drop(retained);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
}

#[test]
fn pdd_family_binds_interval_provenance_and_rejects_capacity_before_execution() {
    let (_dir, package) = fixture().open(limits());
    let package = package.unwrap();
    let account = host(1 << 20);
    let plan = plan();
    let (c0, s0) = FastH3PddProjectionSource::prepare(
        &package,
        &plan,
        0,
        FastH3OutputModality::Video,
        3,
        &account,
        projection_limits(),
    )
    .unwrap();
    let (c1, s1) = FastH3PddProjectionSource::prepare(
        &package,
        &plan,
        1,
        FastH3OutputModality::Video,
        3,
        &account,
        projection_limits(),
    )
    .unwrap();
    let family = Family::new(projection_limits()).unwrap();
    let ModelWeights::Stored(w0) = family.weights(&c0).unwrap() else {
        unreachable!()
    };
    let ModelWeights::Stored(w1) = family.weights(&c1).unwrap() else {
        unreachable!()
    };
    assert_ne!(w0.fingerprint().unwrap(), w1.fingerprint().unwrap());
    for c in &w1.components {
        assert!(s0.component(c).is_err());
    }
    let raw = serde_json::to_value(&c0).unwrap();
    for key in ["evaluation", "output", "coefficient_bits"] {
        let mut forged = raw.clone();
        forged[key] = match key {
            "evaluation" => 1.into(),
            "output" => 9.into(),
            _ => serde_json::json!([1, 1, 1, 1]),
        };
        assert!(family.parse_config(&forged).is_err(), "forged {key}");
    }
    let bytes = w0
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum::<u64>();
    let mut short = projection_limits();
    short.maximum_fused_bytes = bytes - 1;
    assert!(Family::new(short).unwrap().weights(&c0).is_err());
    let short_host = host(bytes as usize - 1);
    assert!(FastH3PddProjectionSource::prepare(
        &package,
        &plan,
        0,
        FastH3OutputModality::Video,
        3,
        &short_host,
        projection_limits()
    )
    .is_err());
    assert_eq!(
        short_host.usage().unwrap().committed_bytes(),
        0,
        "second allocation failure releases the first"
    );
    let composition =
        ReferenceVNextComposition::create_with_capacity(id("device.reference.pdd.short"), 64)
            .unwrap();
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(&raw, &id(Family::PROFILE_ID))
        .unwrap();
    let d = composition.runtime().descriptor();
    assert!(CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect()
        )
        .unwrap(),
        composition.weight_materializers(),
        &c0.compile_options().unwrap()
    )
    .is_err());
    assert_eq!(composition.runtime().snapshot().allocations, 0);
    assert_eq!(composition.runtime().snapshot().submissions, 0);
    drop((s0, s1));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
