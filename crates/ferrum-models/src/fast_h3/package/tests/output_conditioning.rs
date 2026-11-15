use super::*;
use crate::fast_h3::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy, TensorComponentSession,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_runtime::*;
use half::bf16;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn output_limits() -> FastH3OutputConditioningLimits {
    FastH3OutputConditioningLimits {
        maximum_rows: 4096,
        maximum_time_rows: 64,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
    }
}
fn time_limits() -> FastH3TimeConditioningLimits {
    FastH3TimeConditioningLimits {
        maximum_time_rows: 64,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
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
fn schedule() -> FastH3ConditioningSchedule {
    let c = FastH3Checkpoint::parse(
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
    FastH3ConditioningSchedule::all_modalities(&FastH3PddPlan::new(&c).unwrap()).unwrap()
}
fn prepare<F: ModelFamilyProvider>(
    family: F,
    config: &F::Config,
    profile: &str,
) -> PreparedModelFamily
where
    F::Config: serde::Serialize,
{
    TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(profile))
        .unwrap()
}
fn session(
    composition: &ReferenceVNextComposition,
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: &dyn WeightComponentSource,
    host: &HostMemoryAccount,
) -> TensorComponentSession<ReferenceDeviceRuntime> {
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
        &options,
    )
    .unwrap();
    compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(source),
            ComponentInitializationConfig {
                run_id: id("run.output.init"),
                request_id: id("request.output.init"),
                transaction_id: id("transaction.output.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("output initialization failed; native owner retained")
        })
}
fn close(session: TensorComponentSession<ReferenceDeviceRuntime>) {
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("output close failed; native owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
fn execute(
    session: &mut TensorComponentSession<ReferenceDeviceRuntime>,
    inputs: Vec<ManagedHostTensor>,
) -> Vec<ferrum_execution::TensorExecutionOutput> {
    session
        .execute(
            id("run.output.execute"),
            id("request.output.execute"),
            inputs,
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("output execution failed; native owner retained")
        })
}
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn populated() -> Fixture {
    let mut f = Fixture::new();
    for (name, (_, _, bytes)) in &mut f.dense {
        if name.starts_with("time_embedder.") {
            for (i, b) in bytes.chunks_exact_mut(2).enumerate() {
                b.copy_from_slice(
                    &bf16::from_f32(((i * 7 % 17) as f32 - 8.) / 64.)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
    }
    for b in f
        .dense
        .get_mut("norm_out.norm.weight")
        .unwrap()
        .2
        .chunks_exact_mut(2)
    {
        b.copy_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
    }
    let h = config().hidden_size();
    let e = config().time_embed_dim();
    // One nonzero input coefficient makes projection rounding independently
    // calculable. Channel zero isolates the required BF16 1+scale boundary.
    for row in 0..2 * h {
        let (weight, bias) = if row == 0 {
            (0.0, -1.0)
        } else if row == h {
            (0.0, 1. / 256.)
        } else if row < h {
            (1.0, 0.125)
        } else {
            (0.5, 0.0)
        };
        f.dense.get_mut("norm_out.linear.weight").unwrap().2[2 * row * e..2 * row * e + 2]
            .copy_from_slice(&bf16::from_f32(weight).to_bits().to_le_bytes());
        f.dense.get_mut("norm_out.linear.bias").unwrap().2[2 * row..2 * row + 2]
            .copy_from_slice(&bf16::from_f32(bias).to_bits().to_le_bytes());
    }
    f
}
#[test]
fn live_embedding_to_final_table_to_two_request_evaluations_keeps_bf16_boundaries_and_owners() {
    let (_dir, p) = populated().open(limits());
    let p = p.unwrap();
    let schedule = schedule();
    let account = host(4 << 20);
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.output_conditioning"),
        4 << 20,
    )
    .unwrap();
    let tc = FastH3TimeConditioningConfig::from_package(
        &p,
        0,
        schedule.timesteps().len() as u64,
        time_limits(),
    )
    .unwrap();
    let ts = FastH3TimeConditioningSource::new(&p, &tc, &account, time_limits()).unwrap();
    let mut s = session(
        &composition,
        prepare(
            FastH3TimeConditioningFamily::new(time_limits()).unwrap(),
            &tc,
            FastH3TimeConditioningFamily::PROFILE_ID,
        ),
        tc.compile_options().unwrap(),
        &ts,
        &account,
    );
    let outputs = execute(
        &mut s,
        vec![tc.managed_schedule_features(&account, &schedule).unwrap()],
    );
    close(s);
    let embeddings =
        FastH3ConditioningTable::from_execution(tc, &schedule, outputs, time_limits()).unwrap();
    let c = FastH3OutputTimeConfig::from_package(&p, &schedule, output_limits()).unwrap();
    let source =
        FastH3OutputConditioningSource::for_time_projection(&p, &c, output_limits()).unwrap();
    let mut s = session(
        &composition,
        prepare(
            FastH3OutputTimeFamily::new(output_limits()).unwrap(),
            &c,
            FastH3OutputTimeFamily::PROFILE_ID,
        ),
        c.compile_options().unwrap(),
        &source,
        &account,
    );
    let outputs = execute(
        &mut s,
        vec![c.embedding_input(&embeddings, &account).unwrap()],
    );
    let retained_before_rejection = account.usage().unwrap().committed_bytes();
    let spec = ResolvedTensorSpec::new(
        vec![c.time_values().len() as u64, c.embedding_dim()],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut nonfinite = account
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            spec.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for b in nonfinite.as_mut().chunks_exact_mut(4) {
        b.copy_from_slice(&f32::NAN.to_le_bytes());
    }
    let nonfinite = execute(
        &mut s,
        vec![ManagedHostTensor::new(
            id(FastH3OutputTimeFamily::INPUT_ID),
            spec,
            nonfinite.freeze(),
        )
        .unwrap()],
    );
    assert!(
        FastH3OutputConditioningTable::from_execution(c.clone(), nonfinite, output_limits())
            .is_err()
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        retained_before_rejection
    );
    assert!(
        FastH3OutputConditioningTable::from_execution(c.clone(), vec![], output_limits()).is_err()
    );
    close(s);
    let table =
        FastH3OutputConditioningTable::from_execution(c.clone(), outputs, output_limits()).unwrap();
    let actual = decode(table.output().bytes().bytes());
    let h = c.hidden_size() as usize;
    let embedding = embeddings.outputs()[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = embedding
        .chunks_exact(c.embedding_dim() as usize)
        .flat_map(|r| {
            let activated = round((f64::from(r[0]) / (1.0 + (-f64::from(r[0])).exp())) as f32);
            (0..2 * h).map(move |j| {
                if j == 0 {
                    -1.0
                } else if j == h {
                    1. / 256.
                } else if j < h {
                    round(round(activated) + 0.125)
                } else {
                    round(activated * 0.5)
                }
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "independent scalar SiLU/affine oracle");
    assert!(
        expected
            .chunks_exact(2 * h)
            .map(|r| r[1].to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
    );
    let before = account.usage().unwrap().committed_bytes();
    let small = host(4);
    assert!(c.embedding_input(&embeddings, &small).is_err());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), before);
    drop(embeddings);
    let mut retained = Vec::new();
    for evaluation in [0, schedule.evaluation_count() - 1] {
        let layout = FastH3PackedLayout::ref2va(
            3,
            &[
                FastH3Reference::Image {
                    latent_height: 2,
                    latent_width: 2,
                },
                FastH3Reference::Audio {
                    latents_per_channel: 2,
                },
            ],
            &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
            4096,
        )
        .unwrap();
        let n = FastH3OutputNormConfig::from_package(
            &p,
            &layout,
            &schedule,
            evaluation,
            output_limits(),
        )
        .unwrap();
        assert_eq!(n.time_values(), schedule.timesteps());
        let expanded = n
            .parameter_row_runs()
            .flat_map(|(rows, t)| std::iter::repeat_n(t, rows as usize))
            .collect::<Vec<_>>();
        for segment in layout.segments() {
            assert!(expanded[segment.packed_rows.clone()]
                .iter()
                .all(|&t| Some(t) == schedule.segment_time_index(evaluation, segment.kind)));
        }
        let source =
            FastH3OutputConditioningSource::for_normalization(&p, &n, output_limits()).unwrap();
        let mut s = session(
            &composition,
            prepare(
                FastH3OutputNormFamily::new(output_limits()).unwrap(),
                &n,
                FastH3OutputNormFamily::PROFILE_ID,
            ),
            n.compile_options().unwrap(),
            &source,
            &account,
        );
        let mut bytes = account
            .allocate_zeroed(HostMemoryClass::InputPreprocess, n.rows() as usize * h * 2)
            .unwrap();
        for b in bytes.as_mut().chunks_exact_mut(2) {
            b.copy_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
        }
        let hidden = ManagedHostTensor::new(
            id(FastH3OutputNormFamily::INPUT_ID),
            ResolvedTensorSpec::new(
                vec![n.rows(), h as u64],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            bytes.freeze(),
        )
        .unwrap();
        let outputs = execute(
            &mut s,
            vec![hidden, table.modulation_input(&n, &account).unwrap()],
        );
        close(s);
        assert_eq!(outputs[0].tensor().element_type(), ElementType::Bf16);
        let values = decode(outputs[0].bytes().bytes());
        for (row, &time) in expanded.iter().enumerate() {
            for j in 0..h {
                let rms =
                    round((1.0 / (1.0 + p.config().final_norm_eps() as f32) as f64).sqrt() as f32);
                let factor = round(1.0 + expected[time * 2 * h + h + j]);
                let product = round(rms * factor);
                let want = round(product + expected[time * 2 * h + j]);
                assert_eq!(
                    values[row * h + j].to_bits(),
                    want.to_bits(),
                    "evaluation{evaluation}, row{row}, column{j}"
                );
                if j == 0 {
                    assert_eq!(
                        values[row * h],
                        0.0,
                        "fused F32 modulation would incorrectly retain 1/256"
                    );
                }
            }
        }
        assert!(table.modulation_input(&n, &small).is_err());
        assert_eq!(small.usage().unwrap().committed_bytes(), 0);
        let mut wire = serde_json::to_value(&n).unwrap();
        wire["times"][0] = serde_json::json!(0.000123);
        let wrong: FastH3OutputNormConfig = serde_json::from_value(wire).unwrap();
        let usage = account.usage().unwrap().committed_bytes();
        assert!(table.modulation_input(&wrong, &account).is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), usage);
        retained.extend(outputs);
    }
    assert!(account.owns(table.output().bytes()));
    drop(table);
    assert!(retained.iter().all(|o| account.owns(o.bytes())));
    drop(retained);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn output_sources_reject_wrong_parameters_capacities_and_final_routing() {
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    let schedule = schedule();
    let c = FastH3OutputTimeConfig::from_package(&p, &schedule, output_limits()).unwrap();
    let mut small = output_limits();
    small.maximum_stored_bytes = 1;
    assert!(FastH3OutputTimeConfig::from_package(&p, &schedule, small).is_err());
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["components"][0]["external_names"] = serde_json::json!(["another.weight"]);
    let wrong: FastH3OutputTimeConfig = serde_json::from_value(wire).unwrap();
    assert!(
        FastH3OutputConditioningSource::for_time_projection(&p, &wrong, output_limits()).is_err()
    );
    let layout = FastH3PackedLayout::ref2va(
        1,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        4096,
    )
    .unwrap();
    assert!(FastH3OutputNormConfig::from_package(
        &p,
        &layout,
        &schedule,
        schedule.evaluation_count(),
        output_limits()
    )
    .is_err());
    let c =
        FastH3OutputNormConfig::from_package(&p, &layout, &schedule, 0, output_limits()).unwrap();
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["runs"][0]["time"] = serde_json::json!(schedule.timesteps().len());
    let wrong: FastH3OutputNormConfig = serde_json::from_value(wire).unwrap();
    assert!(
        FastH3OutputConditioningSource::for_normalization(&p, &wrong, output_limits()).is_err()
    );
}
