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
fn modulation_limits() -> FastH3ModulationLimits {
    FastH3ModulationLimits {
        maximum_rows: 512,
        maximum_time_rows: 64,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
    }
}
fn layout(text: usize) -> FastH3PackedLayout {
    FastH3PackedLayout::ref2va(
        text,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        512,
    )
    .unwrap()
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
fn tags() -> Vec<FastH3TextTagRun> {
    vec![
        FastH3TextTagRun {
            rows: 1,
            tag: FastH3TextTokenTag::Vision,
        },
        FastH3TextTagRun {
            rows: 2,
            tag: FastH3TextTokenTag::Text,
        },
    ]
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
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
    )
}
fn managed(
    host: &HostMemoryAccount,
    name: &str,
    rows: u64,
    width: u64,
    value: f32,
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            (rows * width * 2) as usize,
        )
        .unwrap();
    for b in bytes.as_mut().chunks_exact_mut(2) {
        b.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            vec![rows, width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
#[test]
fn actual_norm_source_and_compact_table_preserve_independent_time_and_text_vision_tags() {
    let mut f = Fixture::new();
    for norm in [1, 2] {
        let name = format!("transformer_blocks.0.norm{norm}.weight");
        for b in f.dense.get_mut(&name).unwrap().2.chunks_exact_mut(2) {
            b.copy_from_slice(&bf16::from_f32(2.0).to_bits().to_le_bytes());
        }
    }
    let (_dir, package) = f.open(limits());
    let package = package.unwrap();
    let layout = layout(3);
    let schedule = schedule();
    let evaluation = 3;
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fast_h3.modulation")).unwrap();
    for stage in [
        FastH3ModulationStage::Attention,
        FastH3ModulationStage::FeedForward,
    ] {
        let c = FastH3ModulationConfig::from_package(
            &package,
            0,
            stage,
            &layout,
            &schedule,
            evaluation,
            &tags(),
            modulation_limits(),
        )
        .unwrap();
        let later = FastH3ModulationConfig::from_package(
            &package,
            0,
            stage,
            &layout,
            &schedule,
            evaluation + 2,
            &tags(),
            modulation_limits(),
        )
        .unwrap();
        let expand = |c: &FastH3ModulationConfig| {
            c.parameter_row_runs()
                .flat_map(|(n, p)| std::iter::repeat_n(p, n as usize))
                .collect::<Vec<_>>()
        };
        assert_ne!(
            expand(&c),
            expand(&later),
            "requests occupy different denoising evaluations"
        );
        let batch = FastH3ModulationConfig::concatenate_requests(
            &[c.clone(), later.clone()],
            modulation_limits(),
        )
        .unwrap();
        assert_eq!(expand(&batch), [expand(&c), expand(&later)].concat());
        for c in [c, batch] {
            let runs = c.parameter_row_runs().collect::<Vec<_>>();
            let time = schedule
                .segment_time_index(evaluation, FastH3SegmentKind::Text)
                .unwrap() as u64;
            assert_eq!(runs[0], (1, 3 * time));
            assert_eq!(runs[1], (2, 3 * time + 1));
            let source = FastH3ModulationSource::new(&package, &c, modulation_limits()).unwrap();
            let prepared = TypedFamilyRegistration::new(
                FastH3ModulationFamily::new(modulation_limits()).unwrap(),
            )
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &id(FastH3ModulationFamily::PROFILE_ID),
            )
            .unwrap();
            let ModelWeights::Stored(schema) = prepared.weights() else {
                panic!("stored norm");
            };
            assert_eq!(schema.components.len(), 1);
            assert_eq!(
                source
                    .component(&schema.components[0])
                    .unwrap()
                    .bytes()
                    .len(),
                c.width() as usize * 2
            );
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
            let host = host();
            let mut session = compiled
                .initialize(
                    composition.runtime().clone(),
                    host.clone(),
                    ComponentWeightSource::Stored(&source),
                    ComponentInitializationConfig {
                        run_id: id("run.modulation.init"),
                        request_id: id("request.modulation.init"),
                        transaction_id: id("transaction.modulation.init"),
                        static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                        executor: TensorExecutorConfig::default(),
                    },
                )
                .unwrap();
            let width = c.width() as usize;
            let mut bytes = vec![0u8; schedule.timesteps().len() * 18 * width * 2];
            for row in 0..schedule.timesteps().len() * 3 {
                for field in [0, 3] {
                    for col in 0..width {
                        let off = (row * 6 * width + field * width + col) * 2;
                        bytes[off..off + 2].copy_from_slice(
                            &bf16::from_f32(row as f32 / 16.).to_bits().to_le_bytes(),
                        );
                    }
                }
            }
            let mut inputs = vec![
                managed(
                    &host,
                    FastH3ModulationFamily::INPUT_ID,
                    c.rows(),
                    c.width(),
                    1.0,
                ),
                c.managed_table(&host, &bytes).unwrap(),
            ];
            if stage == FastH3ModulationStage::FeedForward {
                inputs.push(managed(
                    &host,
                    FastH3ModulationFamily::BRANCH_ID,
                    c.rows(),
                    c.width(),
                    0.0,
                ));
            }
            let outputs = session
                .execute(id("run.modulation"), id("request.modulation"), inputs)
                .unwrap();
            let out = outputs.last().unwrap();
            let mut row = 0;
            for (count, parameter) in runs {
                for _ in 0..count {
                    for b in
                        out.bytes().bytes()[row * width * 2..(row + 1) * width * 2].chunks_exact(2)
                    {
                        assert_eq!(
                            u16::from_le_bytes(b.try_into().unwrap()),
                            bf16::from_f32(2.0 + parameter as f32 / 16.).to_bits()
                        );
                    }
                    row += 1;
                }
            }
            let closed = session.close().unwrap_or_else(|failure| {
                std::mem::forget(failure);
                panic!("modulation close failed; original owner retained");
            });
            assert!(matches!(closed, PlanRuntimeCloseOutcome::Closed(_)));
            assert!(host.owns(out.bytes()));
            drop(outputs);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}
#[test]
fn modality_routing_rejects_invalid_tags_row_capacity_and_evaluation() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let schedule = schedule();
    let l = layout(3);
    for tags in [
        vec![],
        vec![FastH3TextTagRun {
            rows: 2,
            tag: FastH3TextTokenTag::Text,
        }],
        vec![
            FastH3TextTagRun {
                rows: 0,
                tag: FastH3TextTokenTag::Text,
            },
            FastH3TextTagRun {
                rows: 3,
                tag: FastH3TextTokenTag::Text,
            },
        ],
    ] {
        assert!(FastH3ModulationConfig::from_package(
            &package,
            0,
            FastH3ModulationStage::Attention,
            &l,
            &schedule,
            0,
            &tags,
            modulation_limits()
        )
        .is_err());
    }
    let long = layout(33);
    let tags = (0..33)
        .map(|i| FastH3TextTagRun {
            rows: 1,
            tag: if i % 2 == 0 {
                FastH3TextTokenTag::Text
            } else {
                FastH3TextTokenTag::Vision
            },
        })
        .collect::<Vec<_>>();
    let mut capacity = modulation_limits();
    capacity.maximum_rows = long.sequence_rows() as u64 - 1;
    assert!(FastH3ModulationConfig::from_package(
        &package,
        0,
        FastH3ModulationStage::Attention,
        &long,
        &schedule,
        0,
        &tags,
        capacity
    )
    .is_err());
    assert!(FastH3ModulationConfig::from_package(
        &package,
        0,
        FastH3ModulationStage::Attention,
        &l,
        &schedule,
        schedule.evaluation_count(),
        &super::modulation::tags(),
        modulation_limits()
    )
    .is_err());
}

#[test]
fn request_batch_modulation_rejects_different_tables_stages_and_real_routing_capacity() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let schedule = schedule();
    let layout = layout(3);
    let make = |stage| {
        FastH3ModulationConfig::from_package(
            &package,
            0,
            stage,
            &layout,
            &schedule,
            3,
            &tags(),
            modulation_limits(),
        )
        .unwrap()
    };
    let first = make(FastH3ModulationStage::Attention);
    let second_stage = make(FastH3ModulationStage::FeedForward);
    assert!(FastH3ModulationConfig::concatenate_requests(&[], modulation_limits()).is_err());
    assert!(FastH3ModulationConfig::concatenate_requests(
        &[first.clone(), second_stage],
        modulation_limits()
    )
    .is_err());
    let mut wire = serde_json::to_value(&first).unwrap();
    wire["times"][1] = serde_json::json!(f32::from_bits(first.time_values()[1].to_bits() + 1));
    let another_table: FastH3ModulationConfig = serde_json::from_value(wire).unwrap();
    assert!(another_table.compile_options().is_ok());
    assert!(FastH3ModulationConfig::concatenate_requests(
        &[first.clone(), another_table],
        modulation_limits()
    )
    .is_err());
    let mut cap = modulation_limits();
    cap.maximum_rows = first.rows() * 2 - 1;
    assert!(
        FastH3ModulationConfig::concatenate_requests(&[first.clone(), first.clone()], cap).is_err()
    );
    let requests = vec![first.clone(); 33];
    cap.maximum_rows = first.rows() * requests.len() as u64;
    let combined = FastH3ModulationConfig::concatenate_requests(&requests, cap).unwrap();
    assert!(combined.parameter_row_runs().count() > 32);
    cap.maximum_metadata_bytes = serde_json::to_vec(&combined).unwrap().len() - 1;
    assert!(FastH3ModulationConfig::concatenate_requests(&requests, cap).is_err());
}

#[test]
fn alternating_video_text_tags_preserve_complete_role_routing_across_requests() {
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let schedule = schedule();
    let layout = layout(67);
    let tags: Vec<_> = (0..67)
        .map(|i| FastH3TextTagRun {
            rows: 1,
            tag: if i % 2 == 0 {
                FastH3TextTokenTag::Vision
            } else {
                FastH3TextTokenTag::Text
            },
        })
        .collect();
    let config = FastH3ModulationConfig::from_conditioning_roles(
        &package,
        0,
        FastH3ModulationStage::Attention,
        &layout,
        &schedule,
        &[&tags, &tags],
        modulation_limits(),
    )
    .unwrap();
    config.compile_options().unwrap();
    let rows = layout.sequence_rows() as u64;
    let mut covered = 0;
    for (count, parameter) in config.parameter_row_runs() {
        let request = covered / rows;
        assert_eq!(parameter / 5, request);
        assert!(count <= (request + 1) * rows - covered);
        covered += count;
    }
    assert_eq!(covered, rows * 2);
    let mut limited = modulation_limits();
    limited.maximum_metadata_bytes = serde_json::to_vec(&config).unwrap().len() - 1;
    assert!(FastH3ModulationConfig::from_conditioning_roles(
        &package,
        0,
        FastH3ModulationStage::Attention,
        &layout,
        &schedule,
        &[&tags, &tags],
        limited,
    )
    .is_err());
}
