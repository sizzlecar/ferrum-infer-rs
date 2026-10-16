//! Reuse the real request/sequence admission and completion helpers privately.
//! The legacy checkpoint constructor is deliberately never used by this gate.
#![allow(dead_code)]
include!("../vnext_checkpoint_continuation/runtime.rs");
#[path = "batch.rs"]
mod batch;
pub use batch::{BatchObservation, Path};
#[path = "runtime/decode_segment.rs"]
mod decode_segment;
#[path = "runtime/extra.rs"]
mod extra;
#[path = "runtime/extra_all_rows.rs"]
mod extra_all_rows;
#[path = "runtime/extra_prefill.rs"]
mod extra_prefill;
#[path = "runtime/hybrid.rs"]
mod hybrid;
#[path = "runtime/identity_projection.rs"]
mod identity_projection;
#[path = "runtime/packed_decode_preparation.rs"]
mod packed_decode_preparation;
#[path = "runtime/prefill.rs"]
mod prefill;
#[path = "runtime/q6_f16.rs"]
#[cfg(feature = "cuda-q6-f16-linear")]
mod q6_f16;
#[path = "runtime/replay_bindings.rs"]
mod replay_bindings;
#[path = "runtime/two_streams.rs"]
mod two_streams;
#[path = "runtime/uniform_binding_prefix.rs"]
mod uniform_binding_prefix;

impl Fixture {
    pub fn for_attention(kind: AttentionKind, reusable: bool, participants: u32) -> Self {
        Self::for_family(kind, reusable, participants, Family::new(kind))
    }
    fn for_family(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
    ) -> Self {
        Self::for_family_with_binding_upload(
            kind,
            reusable,
            participants,
            definition,
            ferrum_types::ProgramBindingUploadStrategy::Sparse,
        )
    }

    fn for_family_with_binding_upload(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
    ) -> Self {
        Self::for_family_with_segment_oracle(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            SegmentBindingOracleMode::Disabled,
        )
    }

    fn for_family_with_segment_oracle(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
    ) -> Self {
        Self::for_family_with_segment_owner_views(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            SegmentBindingOwnerViewMode::Legacy,
        )
    }

    fn for_family_with_segment_owner_views(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
    ) -> Self {
        Self::for_family_with_causal_decode_preparation(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            ferrum_types::AttentionExecutionPolicy::Portable,
            ferrum_types::CausalDecodePreparationMode::PerParticipant,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_causal_decode_preparation(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
    ) -> Self {
        Self::for_family_with_decode_attention_grouping(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            attention_policy,
            preparation_mode,
            ferrum_types::CausalDecodeAttentionGrouping::Contiguous,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_decode_attention_grouping(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
        attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
    ) -> Self {
        Self::for_family_with_q4_mmvq_weight_decode(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            attention_policy,
            preparation_mode,
            attention_grouping,
            ferrum_types::Q4MmvqWeightDecode::Baseline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_q4_mmvq_weight_decode(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
        attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
    ) -> Self {
        Self::for_family_with_mmvq_weight_decode(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            attention_policy,
            preparation_mode,
            attention_grouping,
            q4_mmvq_weight_decode,
            ferrum_types::Q5MmvqWeightDecode::Baseline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_mmvq_weight_decode(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
        attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
    ) -> Self {
        Self::for_family_with_execution_context_limit(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            attention_policy,
            preparation_mode,
            attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_execution_context_limit(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
        attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
    ) -> Self {
        Self::for_family_with_q8_projection_weight_preparation(
            kind,
            reusable,
            participants,
            definition,
            upload_strategy,
            oracle,
            owner_mode,
            attention_policy,
            preparation_mode,
            attention_grouping,
            q4_mmvq_weight_decode,
            q5_mmvq_weight_decode,
            execution_context_limit,
            ferrum_types::Q8ProjectionWeightPreparation::Compressed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_family_with_q8_projection_weight_preparation(
        kind: AttentionKind,
        reusable: bool,
        participants: u32,
        definition: Family,
        upload_strategy: ferrum_types::ProgramBindingUploadStrategy,
        oracle: SegmentBindingOracleMode,
        owner_mode: SegmentBindingOwnerViewMode,
        attention_policy: ferrum_types::AttentionExecutionPolicy,
        preparation_mode: ferrum_types::CausalDecodePreparationMode,
        attention_grouping: ferrum_types::CausalDecodeAttentionGrouping,
        q4_mmvq_weight_decode: ferrum_types::Q4MmvqWeightDecode,
        q5_mmvq_weight_decode: ferrum_types::Q5MmvqWeightDecode,
        execution_context_limit: Option<usize>,
        q8_preparation: ferrum_types::Q8ProjectionWeightPreparation,
    ) -> Self {
        let q8_fixture_layout = definition.small_q8_projection_layout();
        let maximum_tokens = definition.maximum_tokens();
        let baseline = definition.is_g32_baseline();
        let attention_arithmetic = definition.attention_arithmetic();
        let swiglu_arithmetic = definition.swiglu_arithmetic();
        let selected = definition.attention_profile();
        let prefill = (selected.prefill() || selected.hybrid()) && maximum_tokens > MAX_TOKENS;
        let states = definition.states();
        let profile_id = definition.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile_id))
            .unwrap();
        let (runtime, registry, materializers, catalog) =
            CudaVNextComposition::create_with_q8_projection_weight_preparation(
                0,
                id(format!("device.cuda.upstream-marker.{kind:?}")),
                attention_policy,
                upload_strategy,
                oracle,
                owner_mode,
                preparation_mode,
                attention_grouping,
                q4_mmvq_weight_decode,
                q5_mmvq_weight_decode,
                execution_context_limit,
                q8_preparation,
            )
            .unwrap()
            .into_parts();
        let materializer = cuda_weight_materializer_selection(&family).unwrap();
        let bucket = reusable.then(|| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("fixture.upstream-marker.decode").unwrap(),
                ReusableExecutionCapacity::new(
                    participants,
                    if prefill {
                        2048
                    } else {
                        u64::from(participants) * 4
                    },
                    if prefill {
                        maximum_tokens.div_ceil(16)
                    } else {
                        u64::from(participants) * 4
                    },
                )
                .unwrap(),
            )
            .unwrap()
        });
        let reusable_bucket = bucket.as_ref().map(|bucket| bucket.bucket_id().clone());
        let policy = ResolvedRuntimePolicy::new(
            "runtime-policy.fixture.checkpoint-continuation",
            ContractVersion::new(1, 0),
            SchedulingDiscipline::FirstReady,
            RuntimeMemoryPolicy {
                checkpoint_capacity: None,
                capacity_bytes: if selected.q6_f16() {
                    512 << 20
                } else {
                    256 << 20
                },
                reserve_bytes: 1 << 20,
                maximum_active_sequences: participants,
                dynamic_storage_profile_order: runtime
                    .descriptor()
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            },
            AdmissionPolicy {
                maximum_queue_depth: participants,
                maximum_scheduled_tokens: if prefill { 2048 } else { MAX_TOKENS },
                sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
                sequence_backing_growth: Default::default(),
                allow_defer: true,
                cancellation_check_interval_steps: 1,
            },
            runtime.attention_execution_policy(),
            if reusable {
                ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay
            } else {
                ExecutionDeterminismRequirement::BitwiseSameRuntime
            },
            bucket.map(|bucket| ReusableExecutionPolicy::new(1, vec![bucket]).unwrap()),
        )
        .unwrap();
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.tokens"),
            ProgramTensorSpec {
                dimensions: vec![maximum_tokens],
                element_type: ElementType::U32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.require_weight_materializer_selection(materializer);
        options.retain_completion_value(id("value.output"));
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            &materializers,
            &options,
        )
        .unwrap();
        let executable = compilation.executable();
        let plan = executable.execution_plan();
        if kind == AttentionKind::Causal {
            assert!(
                matches!(
                    plan.sequence_checkpoint_capability(),
                    SequenceCheckpointCapability::Unsupported(_)
                ),
                "FP16 causal qualification must remain unsupported"
            );
        }
        for (name, arithmetic) in [
            ("node.attention", attention_arithmetic),
            ("node.swiglu", swiglu_arithmetic),
        ] {
            let node = plan
                .payload()
                .nodes()
                .iter()
                .find(|n| n.id().as_str() == name)
                .unwrap();
            let prepared = node
                .provider_resources()
                .projection_numerics()
                .expect("actual provider retains projection decisions");
            assert_eq!(prepared.contract(), &arithmetic);
            if baseline {
                assert!(node.persistent_resource().is_none());
                continue;
            }
            let resource = node
                .persistent_resource()
                .expect("admitted validation flags cannot be a hidden CUDA allocation");
            let flag_bytes = prepared
                .projections()
                .iter()
                .map(|p| p.leaves().len() as u64 * 8)
                .sum::<u64>();
            let requirement = node.provider_resources().persistent().unwrap();
            let allocation = plan
                .payload()
                .memory()
                .static_allocations()
                .iter()
                .find(|a| a.resource_id() == resource)
                .unwrap();
            let q8_leaves: Vec<_> = prepared.projections().iter().flat_map(|projection| {
                projection.leaves().iter().filter_map(|leaf| {
                    matches!(leaf.encoding(), WeightEncoding::BlockQuantized(spec) if spec.format_id.as_str() == "quantization.gguf.q8-0")
                        .then_some((leaf.output_offset(), leaf.output_features(), projection.input_features()))
                })
            }).collect();
            let expected_q8 = if name == "node.attention" {
                q8_fixture_layout.as_slice()
            } else {
                &[]
            };
            assert_eq!(
                q8_leaves
                    .iter()
                    .map(|&(offset, outputs, _)| (offset, outputs))
                    .collect::<Vec<_>>()
                    .as_slice(),
                expected_q8
            );
            assert!(q8_leaves.iter().all(|&(_, _, inputs)| inputs == HIDDEN));
            let mut persistent_bytes = flag_bytes;
            if q8_preparation == ferrum_types::Q8ProjectionWeightPreparation::PredecodedF32 {
                for &(_, outputs) in expected_q8 {
                    persistent_bytes = persistent_bytes.div_ceil(16) * 16 + outputs * HIDDEN * 4;
                }
            }
            assert_eq!(requirement.fixed_bytes(), Some(persistent_bytes));
            assert_eq!(allocation.per_instance_bytes(), persistent_bytes);
            assert_eq!(
                allocation.size_bytes(),
                persistent_bytes.div_ceil(requirement.alignment_bytes())
                    * requirement.alignment_bytes(),
                "Plan admission must charge the complete aligned persistent extent"
            );
            if persistent_bytes == flag_bytes && q8_leaves.is_empty() && !selected.q6_f16() {
                assert_ne!(flag_bytes % requirement.alignment_bytes(), 0);
                assert!(allocation.size_bytes() > flag_bytes);
            }
            assert_eq!(allocation.element_type(), ElementType::U8);
            assert_eq!(allocation.usage(), BufferUsage::Persistent);
            assert_eq!(allocation.lifetime(), AllocationLifetime::Plan);
            for projection in prepared.projections() {
                let formats: BTreeSet<_> = projection
                    .leaves()
                    .iter()
                    .filter_map(|leaf| match leaf.encoding() {
                        WeightEncoding::BlockQuantized(spec) if leaf.is_staged() => {
                            Some(spec.format_id.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                let expected = if selected.q6_f16() {
                    if projection.has_staged_leaf() {
                        BTreeSet::from(["quantization.gguf.q6-k"])
                    } else {
                        BTreeSet::new()
                    }
                } else if selected.extra() {
                    BTreeSet::from(family::extra::FORMATS)
                } else {
                    BTreeSet::from([
                        "quantization.gguf.q4-k",
                        "quantization.gguf.q5-k",
                        "quantization.gguf.iq4-xs",
                    ])
                };
                assert_eq!(formats, expected);
                assert!(projection.leaves().iter().any(|leaf| matches!(
                    leaf.encoding(),
                    WeightEncoding::Dense { .. }
                ) && !leaf.is_staged()));
            }
            println!(
                "{}",
                serde_json::json!({"kind":"marker_provider_plan", "node":name,
                "attention":format!("{kind:?}"), "participants":participants,"reusable":reusable,
                "prepared":prepared,"flag_bytes":flag_bytes,"scratch":node.provider_resources().scratch(),
                "admitted_persistent_bytes":allocation.size_bytes(),
                "checkpoint":format!("{:?}",plan.sequence_checkpoint_capability())})
            );
        }
        let providers = registry.bind_plan(executable).unwrap();
        let provisioned = plan
            .provision_static(
                Arc::clone(&runtime),
                id("request.fixture.checkpoint.provision"),
            )
            .unwrap();
        let permit = match provisioned.into_provisioning() {
            StaticProvisioning::Required(permit) => permit,
            _ => panic!("real provider fixture has static weights"),
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.fixture.checkpoint.provision"),
            id("transaction.fixture.checkpoint.provision"),
        );
        let driver = RuntimeResourceDriver::new(Arc::clone(&runtime)).unwrap();
        let reserved = ResourceTransaction::begin(driver, identity, permit)
            .unwrap()
            .reserve()
            .unwrap();
        let committed = match reserved.commit() {
            Ok(value) => value,
            Err(ResourceCommitTransitionError::Recoverable(error)) => {
                panic!("static commit failed: {:?}", error.failure())
            }
            Err(ResourceCommitTransitionError::Poisoned(error)) => {
                panic!("static commit indeterminate: {:?}", error.failure())
            }
        };
        // Match the product startup: dynamic pools begin cold and foreground
        // admission provisions the actual first work shape. Preallocating each
        // minimum can strand a smaller contiguous extent below a full-width
        // claim even when no execution owns that extent.
        let source = family::q6_f16::FixtureWeights::new(family.weight_schema(), selected.q6_f16());
        let initialized = committed
            .initialize_static(
                &family,
                plan,
                &source,
                StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
            )
            .unwrap();
        let resources = match initialized.into_plan_runtime() {
            Ok(resources) => resources,
            Err(error) => panic!("plan runtime handoff failed: {}", error.error()),
        };
        let cold_status = resources.dynamic_pool_status().unwrap();
        assert!(cold_status.pools().iter().all(|p| p.resident_bytes() == 0));
        println!(
            "{}",
            serde_json::json!({"kind":"attention_provider_cold_pools",
                "attention":format!("{kind:?}"),"participants":participants,
                "reusable":reusable,"resident_bytes":0})
        );
        let lane = resources.create_execution_lane().unwrap();
        if reusable {
            lane.configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
                .unwrap();
        }
        // The registry, runtime, materializers and catalog retain their real
        // composition owners; no provider descriptor is substituted in tests.
        let composition = CompositionParts {
            _runtime: runtime,
            _registry: registry,
            _materializers: materializers,
            _catalog: catalog,
        };
        Self {
            _composition: composition,
            compilation,
            providers,
            resources,
            lane,
            reaper: CompletionReaper::new(),
            states,
            checkpoint_timing_mode: DeviceTimingMode::Off,
            reusable_bucket,
            output_type: kind.activation_type(),
        }
    }
}

#[test]
#[ignore = "requires CUDA native MarkerV2, real Plan Q8 preparation and Core replay"]
fn q8_predecoded_f32_matches_compressed_across_core_m1_m8_replay_and_fresh_requests() {
    verify_q8_predecoded_preparation(&[1, 8], None);
}

#[test]
#[ignore = "requires CUDA native MarkerV2, paired Q8 Plan preparation and Core replay"]
fn q8_predecoded_f32_pair_matches_compressed_across_core_replay_and_offset_fallback() {
    verify_q8_predecoded_preparation(&[1, 2, 8, 32], Some(true));
    verify_q8_predecoded_preparation(&[8], Some(false));
}

fn verify_q8_predecoded_preparation(participant_counts: &[usize], pair_adjacent: Option<bool>) {
    use ferrum_types::{
        AttentionExecutionPolicy, CausalDecodeAttentionGrouping, CausalDecodePreparationMode,
        InvocationPreparationStrategy, ProgramBindingUploadStrategy, Q4MmvqWeightDecode,
        Q5MmvqWeightDecode, Q8ProjectionWeightPreparation,
    };
    struct NoPreparation;
    impl InvocationPreparationSink for NoPreparation {
        fn record_preparation(&self, _: InvocationPreparationStats) {}
    }
    let policies = [
        Q8ProjectionWeightPreparation::Compressed,
        Q8ProjectionWeightPreparation::PredecodedF32,
    ];
    let q8_leaf_count = if pair_adjacent.is_some() { 2 } else { 1 };
    for &participants in participant_counts {
        let fixtures = policies.map(|policy| {
            Fixture::for_family_with_q8_projection_weight_preparation(
                AttentionKind::GatedDelta,
                true,
                participants as u32,
                match pair_adjacent {
                    Some(adjacent) => Family::new(AttentionKind::GatedDelta)
                        .with_small_q8_projection_pair(adjacent),
                    None => Family::new(AttentionKind::GatedDelta).with_small_q8_projection(),
                },
                ProgramBindingUploadStrategy::CompactScatter,
                SegmentBindingOracleMode::CompareReference,
                SegmentBindingOwnerViewMode::Indexed,
                AttentionExecutionPolicy::Portable,
                CausalDecodePreparationMode::PerParticipant,
                CausalDecodeAttentionGrouping::Contiguous,
                Q4MmvqWeightDecode::Baseline,
                Q5MmvqWeightDecode::Baseline,
                None,
                policy,
            )
        });
        let mut programs = [None, None];
        let mut old_authorities = BTreeSet::new();
        for generation in 0..2 {
            let tokens: Vec<Arc<[u32]>> = (0..participants)
                .map(|owner| {
                    (0..6)
                        .map(|position| {
                            ((position * 7 + owner * 3 + generation * 11 + 1) % 32) as u32
                        })
                        .collect()
                })
                .collect();
            let sessions = fixtures.each_ref().map(|fixture| {
                tokens
                    .iter()
                    .enumerate()
                    .map(|(owner, tokens)| {
                        let session = fixture.admit(
                            &format!("q8-predecoded-{participants}-{generation}-{owner}"),
                            Arc::clone(tokens),
                        );
                        assert!(!old_authorities.contains(&session.sequence_authority()));
                        session
                    })
                    .collect::<Vec<_>>()
            });
            for position in 0..6 {
                let path = if position == 0 {
                    Path::Eager
                } else if position < 3 {
                    Path::Warm
                } else {
                    Path::Replay
                };
                let order: Vec<_> = if position >= 4 {
                    (0..participants).rev().collect()
                } else {
                    (0..participants).collect()
                };
                let local_tokens: Vec<_> = order
                    .iter()
                    .map(|&owner| Arc::clone(&tokens[owner]))
                    .collect();
                let observations = std::array::from_fn::<_, 2, _>(|arm| {
                    let fixture = &fixtures[arm];
                    assert_eq!(
                        fixture
                            ._composition
                            ._runtime
                            .q8_projection_weight_preparation(),
                        policies[arm]
                    );
                    let group: Vec<_> = order
                        .iter()
                        .map(|&owner| Arc::clone(&sessions[arm][owner]))
                        .collect();
                    let observation = fixture.execute_participant_ranges_observed(
                        &fixture.lane,
                        &fixture.reaper,
                        &group,
                        &local_tokens,
                        &vec![position..position + 1; participants],
                        path,
                        InvocationPreparationStrategy::default(),
                        &NoPreparation,
                        None,
                        batch::BatchObservationOptions {
                            timing: if position == 0 {
                                DeviceTimingMode::Kernel
                            } else {
                                DeviceTimingMode::Off
                            },
                            ..Default::default()
                        },
                    );
                    if matches!(path, Path::Replay) {
                        let program = observation.reusable_program_id.as_ref().unwrap();
                        if let Some(previous) = &programs[arm] {
                            assert_eq!(previous, program, "fresh authorities and reordered owners reuse the same captured topology");
                        }
                        programs[arm] = Some(program.clone());
                    }
                    observation
                });
                observations[0].assert_same(&observations[1]);
                if position == 0 {
                    let work = observations.each_ref().map(|observation| {
                        observation
                            .attribution
                            .as_ref()
                            .unwrap()
                            .device()
                            .commands()
                            .iter()
                            .filter(|command| command.dependency_wait_count() > 0)
                            .fold((0_u64, 0_u64), |(compute, waits), command| {
                                (
                                    compute + command.compute_dispatch_count(),
                                    waits + command.dependency_wait_count(),
                                )
                            })
                    });
                    assert_eq!(
                        work[1].1,
                        work[0].1 + q8_leaf_count,
                        "each independent Q8 Plan publication is consumed by both fresh request generations"
                    );
                    assert_eq!(
                        work[1].0,
                        work[0].0 + q8_leaf_count * u64::from(generation == 0),
                        "fresh requests reuse each independently retained Q8 preparation"
                    );
                    let attention_work =
                        observations.each_ref().map(|observation| {
                            let commands: Vec<_> = observation
                                .attribution
                                .as_ref()
                                .unwrap()
                                .device()
                                .commands()
                                .iter()
                                .filter(|command| {
                                    command.native_op_id()
                                        == "vnext_gated_delta_recurrent_attention"
                                })
                                .collect();
                            assert_eq!(
                                commands.len(),
                                1,
                                "real Core attention compute must be observed"
                            );
                            assert_eq!(commands[0].participant_count(), participants as u32);
                            if participants > 1 {
                                assert_eq!(commands[0].batching_form(), DeviceBatchingForm::Packed,
                                "multi-participant qualification must execute a physical packed projection");
                            }
                            commands[0].compute_dispatch_count()
                        });
                    assert_eq!(attention_work[0].checked_sub(attention_work[1]), Some(u64::from(pair_adjacent == Some(true))),
                        "adjacent Q8 leaves must actually fuse; separated/single leaves must preserve original launch count");
                    println!(
                        "{}",
                        serde_json::json!({"kind":"q8_preparation_core_work","participants":participants,"generation":generation,"position":position,"pair_adjacent":pair_adjacent,"q8_leaf_count":q8_leaf_count,"attention_compute_dispatches":attention_work,"compressed":work[0],"predecoded_f32":work[1],"scope":"actual conditional dependency work and declared native compute dispatches; full output/state bits equal"})
                    );
                }
            }
            for group in sessions {
                for session in group {
                    old_authorities.insert(session.sequence_authority());
                    session.try_complete().unwrap();
                }
            }
        }
    }
}
