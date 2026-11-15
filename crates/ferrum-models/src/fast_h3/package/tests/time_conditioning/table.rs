use super::*;
use crate::fast_h3::*;

fn plan() -> FastH3PddPlan {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../../fixtures/inference.json"),
        include_bytes!("../../../fixtures/transformer.json"),
        include_bytes!("../../../fixtures/video_scheduler.json"),
        include_bytes!("../../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 64,
        },
    )
    .unwrap();
    FastH3PddPlan::new(&c).unwrap()
}

#[test]
fn live_time_outputs_feed_modulation_after_close_with_accounted_independent_inputs() {
    let fixture = populated_fixture();
    let (_dir, package) = fixture.open(limits());
    let package = package.unwrap();
    let plan = plan();
    let schedule = FastH3ConditioningSchedule::all_modalities(&plan).unwrap();
    let time_limits = FastH3TimeConditioningLimits {
        maximum_time_rows: 64,
        ..time_limits()
    };
    let config =
        Config::from_package(&package, 0, schedule.timesteps().len() as u64, time_limits).unwrap();
    let account = host();
    let source =
        FastH3TimeConditioningSource::new(&package, &config, &account, time_limits).unwrap();
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fast_h3.time_table")).unwrap();
    let d = composition.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(Family::new(time_limits).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &id(Family::PROFILE_ID),
        )
        .unwrap();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &policy,
        composition.weight_materializers(),
        &config.compile_options().unwrap(),
    )
    .unwrap();
    let initialization = || ComponentInitializationConfig {
        run_id: id("run.time_table.init"),
        request_id: id("request.time_table.init"),
        transaction_id: id("transaction.time_table.init"),
        static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    };
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            account.clone(),
            ComponentWeightSource::Stored(&source),
            initialization(),
        )
        .unwrap();
    let mut outputs = session
        .execute(
            id("run.time_table.forward"),
            id("request.time_table.forward"),
            vec![config
                .managed_schedule_features(&account, &schedule)
                .unwrap()],
        )
        .unwrap();
    let retained_before_rejection = account.usage().unwrap().committed_bytes();
    let nan_spec = ResolvedTensorSpec::new(
        vec![config.time_rows(), config.frequency_dim()],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut nan_bytes = account
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            nan_spec.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for chunk in nan_bytes.as_mut().chunks_exact_mut(4) {
        chunk.copy_from_slice(&f32::NAN.to_le_bytes());
    }
    let nonfinite = session
        .execute(
            id("run.time_table.nonfinite"),
            id("request.time_table.nonfinite"),
            vec![
                ManagedHostTensor::new(id(Family::INPUT_ID), nan_spec, nan_bytes.freeze()).unwrap(),
            ],
        )
        .unwrap();
    assert!(FastH3ConditioningTable::from_execution(
        config.clone(),
        &schedule,
        nonfinite,
        time_limits,
    )
    .is_err());
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        retained_before_rejection
    );
    let close = session
        .close()
        .unwrap_or_else(|e| panic!("{:?}", e.failure()));
    assert!(matches!(close, PlanRuntimeCloseOutcome::Closed(_)));
    let retained_bytes = account.usage().unwrap().committed_bytes();
    assert!(retained_bytes > 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);

    assert!(FastH3ConditioningTable::from_execution(
        config.clone(),
        &schedule,
        vec![outputs[0].clone(), outputs[0].clone()],
        time_limits,
    )
    .is_err());
    assert!(FastH3ConditioningTable::from_execution(
        config.clone(),
        &schedule,
        vec![outputs[0].batch_item(0).unwrap(), outputs[1].clone()],
        time_limits,
    )
    .is_err());
    outputs.swap(0, 1);
    let table =
        FastH3ConditioningTable::from_execution(config, &schedule, outputs, time_limits).unwrap();
    assert_eq!(account.usage().unwrap().committed_bytes(), retained_bytes);

    let layout = FastH3PackedLayout::ref2va(
        2,
        &[FastH3Reference::Image {
            latent_height: 2,
            latent_width: 2,
        }],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        256,
    )
    .unwrap();
    let tags = [
        FastH3TextTagRun {
            rows: 1,
            tag: FastH3TextTokenTag::Vision,
        },
        FastH3TextTagRun {
            rows: 1,
            tag: FastH3TextTokenTag::Text,
        },
    ];
    let modulation_limits = FastH3ModulationLimits {
        maximum_rows: 256,
        maximum_time_rows: 64,
        maximum_metadata_bytes: 32768,
        maximum_stored_bytes: 1 << 20,
    };
    for stage in [
        FastH3ModulationStage::Attention,
        FastH3ModulationStage::FeedForward,
    ] {
        let make_config = |block, schedule: &FastH3ConditioningSchedule| {
            FastH3ModulationConfig::from_package(
                &package,
                block,
                stage,
                &layout,
                schedule,
                3,
                &tags,
                modulation_limits,
            )
            .unwrap()
        };
        let c = make_config(0, &schedule);
        assert!(table
            .modulation_input(&make_config(1, &schedule), &account)
            .is_err());
        let reduced_schedule = FastH3ConditioningSchedule::new(&layout, &plan).unwrap();
        assert!(table
            .modulation_input(&make_config(0, &reduced_schedule), &account)
            .is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), retained_bytes);
        let input = table.modulation_input(&c, &account).unwrap();
        assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
        assert_eq!(input.bytes().bytes(), table.outputs()[1].bytes().bytes());
        assert_eq!(table.outputs()[1].bytes().class(), HostMemoryClass::Output);
        let table_bytes = input.bytes().len();
        assert_eq!(
            account.usage().unwrap().committed_bytes(),
            retained_bytes + table_bytes
        );

        let limited = HostMemoryAccount::new(
            HostMemoryLimits::new(
                table_bytes - 1,
                HostMemoryClassLimits {
                    input_preprocess_bytes: table_bytes - 1,
                    output_bytes: 0,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        assert!(table.modulation_input(&c, &limited).is_err());
        assert_eq!(limited.usage().unwrap().committed_bytes(), 0);

        let prepared =
            TypedFamilyRegistration::new(FastH3ModulationFamily::new(modulation_limits).unwrap())
                .prepare_with_profile(
                    &serde_json::to_value(&c).unwrap(),
                    &id(FastH3ModulationFamily::PROFILE_ID),
                )
                .unwrap();
        let source = FastH3ModulationSource::new(&package, &c, modulation_limits).unwrap();
        let compiled = CompiledTensorComponent::compile(
            prepared,
            composition.registry(),
            composition.catalog(),
            &policy,
            composition.weight_materializers(),
            &c.compile_options().unwrap(),
        )
        .unwrap();
        let mut session = compiled
            .initialize(
                composition.runtime().clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                initialization(),
            )
            .unwrap();
        let zero_input = |name| {
            let spec = ResolvedTensorSpec::new(
                vec![c.rows(), c.width()],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap();
            let bytes = account
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    spec.minimum_storage_bytes().unwrap() as usize,
                )
                .unwrap()
                .freeze();
            ManagedHostTensor::new(id(name), spec, bytes).unwrap()
        };
        let mut inputs = vec![zero_input(FastH3ModulationFamily::INPUT_ID), input];
        if stage == FastH3ModulationStage::FeedForward {
            inputs.push(zero_input(FastH3ModulationFamily::BRANCH_ID));
        }
        let outputs = session
            .execute(
                id("run.time_table.modulation"),
                id("request.time_table.modulation"),
                inputs,
            )
            .unwrap();
        let out = outputs.last().unwrap().bytes().bytes();
        // Zero hidden/branch means exactly the appropriate learned shift. The
        // first two rows have the same time but distinct vision/text tags.
        let time = schedule
            .segment_time_index(3, FastH3SegmentKind::Text)
            .unwrap();
        let width = c.width() as usize;
        let shift = if stage == FastH3ModulationStage::Attention {
            0
        } else {
            3 * width
        };
        for modality in [0, 1] {
            let offset = ((time * 3 + modality) * 6 * width + shift) * 2;
            assert_eq!(
                &out[modality * width * 2..(modality + 1) * width * 2],
                &table.outputs()[1].bytes().bytes()[offset..offset + width * 2]
            );
        }
        let close = session
            .close()
            .unwrap_or_else(|e| panic!("{:?}", e.failure()));
        assert!(matches!(close, PlanRuntimeCloseOutcome::Closed(_)));
        drop(outputs);
        assert_eq!(account.usage().unwrap().committed_bytes(), retained_bytes);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
    verify_role_reuse(
        &package,
        &table,
        &schedule,
        &layout,
        &tags,
        &account,
        modulation_limits,
    );
    drop(table);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

fn verify_role_reuse(
    package: &FastH3TransformerPackage,
    table: &FastH3ConditioningTable,
    schedule: &FastH3ConditioningSchedule,
    layout: &FastH3PackedLayout,
    tags: &[FastH3TextTagRun],
    host: &HostMemoryAccount,
    limits: FastH3ModulationLimits,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counted<'a> {
        source: FastH3ModulationSource<'a>,
        reads: AtomicUsize,
    }
    impl WeightComponentSource for Counted<'_> {
        fn component<'a>(
            &'a self,
            c: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.source.component(c)
        }
    }
    let c = FastH3ModulationConfig::from_conditioning_roles(
        package,
        0,
        FastH3ModulationStage::Attention,
        layout,
        schedule,
        &[tags, tags],
        limits,
    )
    .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(table.modulation_input(&c, host).is_err());
    assert!(c.managed_role_table(table, schedule, &[0], host).is_err());
    assert!(c
        .managed_role_table(table, schedule, &[0, schedule.evaluation_count()], host)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    let composition =
        ReferenceVNextComposition::create(id("device.reference.fast_h3.role_reuse")).unwrap();
    let d = composition.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(FastH3ModulationFamily::new(limits).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(FastH3ModulationFamily::PROFILE_ID),
        )
        .unwrap();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &policy,
        composition.weight_materializers(),
        &c.compile_options().unwrap(),
    )
    .unwrap();
    let source = Counted {
        source: FastH3ModulationSource::new(package, &c, limits).unwrap(),
        reads: AtomicUsize::new(0),
    };
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.role.init"),
                request_id: id("request.role.init"),
                transaction_id: id("transaction.role.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    let width = c.width() as usize;
    for step in 0..schedule.evaluation_count() {
        let evaluations = [step, schedule.evaluation_count() - 1 - step];
        let input = c
            .managed_role_table(table, schedule, &evaluations, host)
            .unwrap();
        assert_eq!(input.tensor().dimensions(), &[10, 6 * c.width()]);
        let spec = ResolvedTensorSpec::new(
            vec![c.rows(), c.width()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let zero = host
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                spec.minimum_storage_bytes().unwrap() as usize,
            )
            .unwrap()
            .freeze();
        let outputs = session
            .execute(
                id(format!("run.role.step{step}")),
                id(format!("request.role.step{step}")),
                vec![
                    ManagedHostTensor::new(id(FastH3ModulationFamily::INPUT_ID), spec, zero)
                        .unwrap(),
                    input,
                ],
            )
            .unwrap();
        let mut expected = Vec::new();
        for evaluation in evaluations {
            let scheduled = FastH3ModulationConfig::from_package(
                package,
                0,
                FastH3ModulationStage::Attention,
                layout,
                schedule,
                evaluation,
                tags,
                limits,
            )
            .unwrap();
            for (rows, parameter) in scheduled.parameter_row_runs() {
                let start = parameter as usize * 6 * width * 2;
                for _ in 0..rows {
                    expected.extend_from_slice(
                        &table.outputs()[1].bytes().bytes()[start..start + 2 * width],
                    );
                }
            }
        }
        assert_eq!(outputs[0].bytes().bytes(), expected);
        assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("{:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
}
