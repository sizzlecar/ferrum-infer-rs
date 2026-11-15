use super::pdd_projection;
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
use std::sync::atomic::{AtomicUsize, Ordering};

type F = FastH3FinalOutputFamily;
#[path = "final_output/device_state.rs"]
mod device_state;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn final_limits() -> FastH3FinalOutputLimits {
    FastH3FinalOutputLimits {
        conditioning: FastH3OutputConditioningLimits {
            maximum_rows: 4096,
            maximum_time_rows: 64,
            maximum_metadata_bytes: 1 << 20,
            maximum_stored_bytes: 1 << 20,
        },
        projection: FastH3PddProjectionLimits {
            maximum_rows: 4096,
            maximum_metadata_bytes: 1 << 20,
            maximum_source_bytes: 1 << 20,
            maximum_fused_bytes: 1 << 20,
        },
        maximum_metadata_bytes: 1 << 20,
        maximum_plan_nodes: 256,
        maximum_stored_bytes: 1 << 20,
        maximum_pdd_steps: 64,
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
fn layout() -> FastH3PackedLayout {
    FastH3PackedLayout::ref2va(
        3,
        &[
            FastH3Reference::Image {
                latent_height: 2,
                latent_width: 4,
            },
            FastH3Reference::Audio {
                latents_per_channel: 2,
            },
            FastH3Reference::Video {
                latent_frames: 2,
                latent_height: 2,
                latent_width: 2,
                audio_latents_per_channel: Some(1),
            },
        ],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        4096,
    )
    .unwrap()
}
fn fixture() -> Fixture {
    let mut f = pdd_projection::fixture();
    for b in f
        .dense
        .get_mut("norm_out.norm.weight")
        .unwrap()
        .2
        .chunks_exact_mut(2)
    {
        b.copy_from_slice(&bf16::ONE.to_bits().to_le_bytes());
    }
    f
}
fn managed(
    account: &HostMemoryAccount,
    id_str: &str,
    shape: Vec<u64>,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
        .unwrap();
    for (b, &x) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
        b.copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
    }
    ManagedHostTensor::new(
        id(id_str),
        ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
struct Counted<'a> {
    inner: FastH3FinalOutputSource<'a>,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
fn round(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

#[test]
fn service_owned_final_head_keeps_derived_buffers_without_borrowing_the_package_variable() {
    let (_dir, package) = fixture().open(limits());
    let package = std::sync::Arc::new(package.unwrap());
    let h = host(8 << 20);
    let (config, source) = OwnedFastH3FinalOutputSource::prepare(
        package.clone(),
        &pdd_projection::plan(),
        &layout(),
        1,
        &h,
        final_limits(),
    )
    .unwrap();
    let ModelWeights::Stored(schema) = F::new(final_limits()).unwrap().weights(&config).unwrap()
    else {
        panic!("Stored");
    };
    let (_, borrowed) = FastH3FinalOutputSource::prepare(
        &package,
        &pdd_projection::plan(),
        &layout(),
        1,
        &h,
        final_limits(),
    )
    .unwrap();
    for component in &schema.components {
        assert_eq!(
            source.component(component).unwrap().bytes(),
            borrowed.component(component).unwrap().bytes()
        );
    }
    drop(borrowed);
    let retained = std::sync::Arc::downgrade(&package);
    drop(package);
    assert!(retained.upgrade().is_some());
    for component in &schema.components {
        assert!(source
            .component(component)
            .unwrap()
            .retained_host_memory()
            .is_some());
    }
    drop(source);
    assert!(retained.upgrade().is_none());
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn one_device_graph_selects_interleaved_modalities_and_keeps_final_f32_heads_and_owners() {
    let fixture = fixture();
    let (_dir, package) = fixture.open(limits());
    let package = package.unwrap();
    let plan = pdd_projection::plan();
    let layout = layout();
    let h = package.config().hidden_size();
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.final_output"),
        8 << 20,
    )
    .unwrap();
    for evaluation in [1, 7] {
        let account = host(8 << 20);
        let (config, source) = FastH3FinalOutputSource::prepare(
            &package,
            &plan,
            &layout,
            evaluation,
            &account,
            final_limits(),
        )
        .unwrap();
        let source = Counted {
            inner: source,
            reads: AtomicUsize::new(0),
        };
        let prepared = TypedFamilyRegistration::new(F::new(final_limits()).unwrap())
            .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(F::PROFILE_ID))
            .unwrap();
        // Both instances use the same family but keep distinct logical weights
        // and unchanged physical derived-source identities.
        let ModelWeights::Stored(schema) = prepared.weights() else {
            panic!("Stored")
        };
        assert_eq!(schema.components.len(), 5);
        assert_eq!(prepared.program().outputs().len(), 2);
        assert!(prepared
            .program()
            .outputs()
            .iter()
            .all(|o| o.as_str() == F::VIDEO_OUTPUT_ID || o.as_str() == F::AUDIO_OUTPUT_ID));
        for component in &schema.components {
            let mut forged = component.clone();
            forged.dimensions[0] += 1;
            assert!(source.inner.component(&forged).is_err());
        }
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
            &config.compile_options(final_limits()).unwrap(),
        )
        .unwrap();
        let mut session = compiled
            .initialize(
                composition.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                ComponentInitializationConfig {
                    run_id: id("run.final.init"),
                    request_id: id("request.final.init"),
                    transaction_id: id("transaction.final.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        assert_eq!(source.reads.load(Ordering::Relaxed), 5);
        let rows = layout.sequence_rows();
        let x: Vec<_> = (0..rows)
            .flat_map(|row| std::iter::repeat_n(if row % 3 == 0 { -1.0 } else { 1.0 }, h))
            .collect();
        let times = config.normalization().time_values().len();
        let table: Vec<_> = (0..times)
            .flat_map(|t| {
                (0..2 * h).map(move |j| {
                    if j < h {
                        (t as f32 - 8.) / 16. + (j % 3) as f32 / 8.
                    } else if j == h {
                        1. / 256.
                    } else {
                        (t % 4) as f32 / 32.
                    }
                })
            })
            .collect();
        let schedule = FastH3ConditioningSchedule::all_modalities(&plan).unwrap();
        let mut first = None;
        let mut retained = None;
        for repeat in 0..2 {
            let outputs = session
                .execute(
                    id(format!("run.final.{repeat}")),
                    id(format!("request.final.{repeat}")),
                    vec![
                        managed(&account, F::INPUT_ID, vec![rows as u64, h as u64], &x),
                        managed(
                            &account,
                            F::TABLE_ID,
                            vec![times as u64, 2 * h as u64],
                            &table,
                        ),
                    ],
                )
                .unwrap();
            for (output_id, modality, prefix) in [
                (F::VIDEO_OUTPUT_ID, FastH3OutputModality::Video, "proj_out"),
                (
                    F::AUDIO_OUTPUT_ID,
                    FastH3OutputModality::Audio,
                    "audio_proj_out",
                ),
            ] {
                let output = outputs
                    .iter()
                    .find(|o| o.value_id().as_str() == output_id)
                    .unwrap();
                let pc = config.projection(modality);
                let width = pc.output_size() as usize;
                let step = &plan.steps()[evaluation];
                let coeff = match modality {
                    FastH3OutputModality::Video => step.video.normalized_head_weights(),
                    FastH3OutputModality::Audio => step.audio.normalized_head_weights(),
                };
                let w = pdd_projection::fused(
                    &pdd_projection::decode(&fixture.dense[&format!("{prefix}.weight")]),
                    width * h,
                    step.heads.clone(),
                    coeff,
                );
                let b = pdd_projection::fused(
                    &pdd_projection::decode(&fixture.dense[&format!("{prefix}.bias")]),
                    width,
                    step.heads.clone(),
                    coeff,
                );
                let selected: Vec<_> = layout
                    .segments()
                    .iter()
                    .filter(|s| match modality {
                        FastH3OutputModality::Video => matches!(
                            s.kind,
                            FastH3SegmentKind::ReferenceImage
                                | FastH3SegmentKind::ReferenceVideo
                                | FastH3SegmentKind::TargetVideo
                        ),
                        FastH3OutputModality::Audio => matches!(
                            s.kind,
                            FastH3SegmentKind::ReferenceAudio | FastH3SegmentKind::TargetAudio
                        ),
                    })
                    .flat_map(|s| {
                        let time = schedule.segment_time_index(evaluation, s.kind).unwrap();
                        s.packed_rows.clone().map(move |row| (row, time))
                    })
                    .collect();
                assert_eq!(output.tensor().element_type(), ElementType::F32);
                assert_eq!(
                    output.tensor().dimensions(),
                    [selected.len() as u64, width as u64]
                );
                let actual: Vec<_> = output
                    .bytes()
                    .bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                for (row, &(packed, time)) in selected.iter().enumerate() {
                    let normalized: Vec<_> = (0..h)
                        .map(|j| {
                            // Unit-magnitude rows round to +/-1 after BF16 RMS;
                            // each remaining stage has an independent BF16 boundary.
                            let rms = round(
                                (f64::from(x[packed * h + j])
                                    / (1.0 + package.config().final_norm_eps()).sqrt())
                                    as f32,
                            );
                            round(
                                round(rms * round(1.0 + table[time * 2 * h + h + j]))
                                    + table[time * 2 * h + j],
                            )
                        })
                        .collect();
                    for col in 0..width {
                        let dot: f64 = normalized
                            .iter()
                            .zip(&w[col * h..(col + 1) * h])
                            .map(|(&x, &w)| f64::from(x) * f64::from(w))
                            .sum();
                        let magnitude: f64 = normalized
                            .iter()
                            .zip(&w[col * h..(col + 1) * h])
                            .map(|(&x, &w)| (f64::from(x) * f64::from(w)).abs())
                            .sum();
                        let expected = dot + f64::from(b[col]);
                        let u = 2f64.powi(-24);
                        let gamma = (2 * h) as f64 * u / (1.0 - (2 * h) as f64 * u);
                        assert!(actual[row * width + col].is_finite());
                        assert!(
                            (f64::from(actual[row * width + col]) - expected).abs()
                                <= (1.0 + u) * gamma * magnitude + u * expected.abs(),
                            "e{evaluation} {modality:?} row{row} col{col}"
                        );
                    }
                }
                assert!(
                    actual.iter().any(|&v| round(v) != v),
                    "F32 projection survives output boundary"
                );
            }
            let bytes: Vec<_> = outputs.iter().map(|o| o.bytes().bytes().to_vec()).collect();
            if let Some(first) = &first {
                assert_eq!(&bytes, first);
            } else {
                first = Some(bytes);
            }
            retained = Some(outputs);
        }
        assert_eq!(source.reads.load(Ordering::Relaxed), 5);
        assert!(matches!(
            session.close().unwrap_or_else(|e| {
                let reason = format!("{:?}", e.failure());
                std::mem::forget(e);
                panic!("close failed; owner retained: {reason}")
            }),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        drop(source);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
        assert!(retained
            .as_ref()
            .unwrap()
            .iter()
            .all(|o| account.owns(o.bytes())));
        drop(retained);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn final_composition_rejects_changed_partition_interval_and_aggregate_capacity() {
    let (_dir, package) = fixture().open(limits());
    let package = package.unwrap();
    let account = host(1 << 20);
    let plan = pdd_projection::plan();
    let layout = layout();
    let (config, source) =
        FastH3FinalOutputSource::prepare(&package, &plan, &layout, 0, &account, final_limits())
            .unwrap();
    let family = F::new(final_limits()).unwrap();
    let raw = serde_json::to_value(&config).unwrap();
    for key in ["segments", "norm", "audio", "schedule"] {
        let mut wrong = raw.clone();
        match key {
            "segments" => wrong[key][0]["rows"] = 2.into(),
            "norm" => wrong[key]["runs"][0]["time"] = 1.into(),
            "schedule" => wrong[key]["video_shift"] = 11.into(),
            _ => wrong[key]["evaluation"] = 1.into(),
        }
        if key == "norm" {
            let norm: FastH3OutputNormConfig = serde_json::from_value(wrong[key].clone()).unwrap();
            assert!(
                FastH3OutputNormFamily::new(final_limits().conditioning)
                    .unwrap()
                    .weights(&norm)
                    .is_ok(),
                "individually valid routing still belongs to the wrong evaluation"
            );
        }
        assert!(family.parse_config(&wrong).is_err(), "changed {key}");
    }
    let ModelWeights::Stored(schema) = family.weights(&config).unwrap() else {
        unreachable!()
    };
    let stored: u64 = schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum();
    let mut short = final_limits();
    short.maximum_stored_bytes = stored - 1;
    let before = account.usage().unwrap().committed_bytes();
    assert!(
        FastH3FinalOutputSource::prepare(&package, &plan, &layout, 0, &account, short).is_err()
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before,
        "aggregate admission failure releases both derived heads"
    );
    let mut short = final_limits();
    short.maximum_plan_nodes = 1;
    assert!(config.compile_options(short).is_err());
    drop(source);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
