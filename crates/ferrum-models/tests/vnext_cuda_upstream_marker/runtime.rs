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
            CudaVNextComposition::create_with_causal_decode_preparation_mode(
                0,
                id(format!("device.cuda.upstream-marker.{kind:?}")),
                attention_policy,
                upload_strategy,
                oracle,
                owner_mode,
                preparation_mode,
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
                capacity_bytes: 256 << 20,
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
            assert_eq!(requirement.fixed_bytes(), Some(flag_bytes));
            assert_ne!(flag_bytes % requirement.alignment_bytes(), 0);
            assert_eq!(allocation.per_instance_bytes(), flag_bytes);
            assert!(allocation.size_bytes() > flag_bytes);
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
                let expected = if selected.extra() {
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
        let source = family::Weights::new(family.weight_schema());
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
