//! Reuse the real request/sequence admission and completion helpers privately.
//! The legacy checkpoint constructor is deliberately never used by this gate.
#![allow(dead_code)]
include!("../vnext_checkpoint_continuation/runtime.rs");
#[path = "batch.rs"]
mod batch;
pub use batch::{BatchObservation, Path};

impl Fixture {
    pub fn for_attention(kind: AttentionKind, reusable: bool, participants: u32) -> Self {
        let definition = Family::new(kind);
        let states = definition.states();
        let profile_id = definition.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile_id))
            .unwrap();
        let (runtime, registry, materializers, materializer, catalog) = composition(kind, &family);
        let bucket = reusable.then(|| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("fixture.q8act-attention.decode").unwrap(),
                ReusableExecutionCapacity::new(
                    participants,
                    u64::from(participants) * 16,
                    u64::from(participants) * 16,
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
                maximum_scheduled_tokens: MAX_TOKENS,
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
                dimensions: vec![MAX_TOKENS],
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
        let attention = plan
            .payload()
            .nodes()
            .iter()
            .find(|n| n.id().as_str() == "node.attention")
            .unwrap();
        let prepared = attention
            .provider_resources()
            .projection_numerics()
            .expect("actual provider must retain projection decisions");
        let selected = match kind {
            AttentionKind::GatedDelta => Q8ActAttentionProfile::GatedDelta,
            AttentionKind::Causal => Q8ActAttentionProfile::Causal,
            _ => unreachable!(),
        };
        assert_eq!(prepared.contract(), &selected.arithmetic());
        assert_eq!(
            prepared.projections().len(),
            if kind == AttentionKind::Causal { 4 } else { 2 }
        );
        for projection in prepared.projections() {
            assert_eq!(projection.leaves().len(), 4);
            assert_eq!(
                projection.leaves().iter().filter(|p| p.is_staged()).count(),
                3
            );
            assert!(!projection.leaves()[3].is_staged());
        }
        println!(
            "{}",
            serde_json::json!({"kind":"attention_provider_plan","attention":format!("{kind:?}"),
            "participants":participants,"reusable":reusable,"prepared":prepared,
            "checkpoint":format!("{:?}",plan.sequence_checkpoint_capability())})
        );
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
