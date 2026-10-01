//! Real checked canonical actual/future/replay projections; no fabricated recipe hashes.
use super::*;
use sha2::{Digest, Sha256};
use std::num::{NonZeroU32, NonZeroU64};

fn domain(identity: u8) -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [identity; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(1024).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(4096).unwrap(),
            repetition_slot_capacity: 512,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap()
}
fn scoped(w: &CanonicalStructuredWave, domain: &CostWorkloadDomainV1) -> StructuredInputV2 {
    let selected = w.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual_with_domain(
        &w.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        domain,
    )
    .unwrap()
}
fn family(input: &StructuredInputV2) -> (&StructuredOwnerKeyV2, &[u8; 32]) {
    input
        .cost_template_identity(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .unwrap()
}

#[test]
fn numerical_family_normalizes_order_and_multiplicity_but_preserves_exact_binding_and_numeric_work()
{
    let a = ordered_wave(
        &["fixture.a", "fixture.b"],
        ActualWaveGraphState::Disabled,
        [4; 32],
    );
    let b = ordered_wave(
        &["fixture.b", "fixture.a"],
        ActualWaveGraphState::Disabled,
        [4; 32],
    );
    let c = ordered_wave(
        &["fixture.a", "fixture.b", "fixture.a"],
        ActualWaveGraphState::Disabled,
        [4; 32],
    );
    let (x, y, z) = (
        scoped(&a, &domain(9)),
        scoped(&b, &domain(9)),
        scoped(&c, &domain(9)),
    );
    assert_ne!(x.owner(), y.owner());
    assert_ne!(x.owner(), z.owner());
    assert_eq!(family(&x), family(&y));
    assert_eq!(family(&x), family(&z));
    let key = x.numerical_family_key().unwrap();
    assert_eq!(key, y.numerical_family_key().unwrap());
    assert_eq!(key, z.numerical_family_key().unwrap());
    assert_eq!(x.basis, y.basis);
    assert_ne!(x.basis, z.basis);
    assert_ne!(x.support, z.support);
    let selected = a.statistical.as_ref().unwrap();
    let a_recipe = selected.structured_capture().unwrap().unwrap();
    let b_recipe = b
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .unwrap();
    assert!(a_recipe
        .algorithm_work()
        .unwrap()
        .validate_structure(b_recipe)
        .is_err());
    assert!(StructuredInputV2::from_actual(&a.exact, selected, b_recipe).is_err());
    // Byte-compatible legacy domain formula remains unchanged.
    let mut legacy = Sha256::new();
    legacy.update(MODEL_REVISION_V2.as_bytes());
    legacy.update(serde_json::to_vec(x.owner()).unwrap());
    assert_eq!(*x.domain_signature(), <[u8; 32]>::from(legacy.finalize()));
    let ptr = x.basis.as_ptr();
    let old = x.owner().clone();
    let old_domain = *x.domain_signature();
    let new = x
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .unwrap();
    assert_eq!(new.numerical_family_key().unwrap(), key);
    assert_eq!(new.basis.as_ptr(), ptr);
    let restored = new
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::OrderedV1)
        .unwrap();
    assert_eq!(restored.owner(), &old);
    assert_eq!(restored.domain_signature(), &old_domain);
    assert_eq!(restored.basis.as_ptr(), ptr);
    assert_eq!(restored.numerical_family_key().unwrap(), key);
}

#[test]
fn numerical_family_retains_algorithm_kernel_route_policy_and_executor_identity() {
    let a = ordered_wave(
        &["fixture.a", "fixture.b"],
        ActualWaveGraphState::Disabled,
        [4; 32],
    );
    let base = scoped(&a, &domain(9));
    for w in [
        ordered_wave(
            &["fixture.a", "fixture.v2", "fixture.reduce"],
            ActualWaveGraphState::Disabled,
            [4; 32],
        ),
        ordered_wave(
            &["fixture.a", "fixture.b"],
            ActualWaveGraphState::ConfiguredEager,
            [4; 32],
        ),
        ordered_wave(
            &["fixture.a", "fixture.b"],
            ActualWaveGraphState::Disabled,
            [5; 32],
        ),
    ] {
        assert_ne!(family(&base), family(&scoped(&w, &domain(9))));
        assert_ne!(
            base.numerical_family_key().unwrap(),
            scoped(&w, &domain(9)).numerical_family_key().unwrap()
        );
    }
    assert_ne!(family(&base), family(&scoped(&a, &domain(10))));
    assert!(project(&a)
        .unwrap()
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .is_err());
}

#[test]
fn numerical_family_actual_future_and_original_replay_share_checked_projection() {
    let w = ordered_wave(
        &["fixture.b", "fixture.a", "fixture.a"],
        ActualWaveGraphState::Disabled,
        [4; 32],
    );
    let selected = w.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let d = domain(9);
    let actual = scoped(&w, &d);
    let future = StructuredQueryV2::from_future_with_domain(
        &w.exact,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
        &d,
    )
    .unwrap();
    assert_eq!(family(&actual), family(future.input()));
    assert_eq!(
        actual.numerical_family_key().unwrap(),
        future.input().numerical_family_key().unwrap()
    );
    let algorithms = recipe
        .algorithm_work()
        .unwrap()
        .entries()
        .iter()
        .map(|a| (*a.algorithm().signature(), a.kind(), a.commands(), a.work()))
        .collect::<Vec<_>>();
    let (replayed, _) = StructuredInputV2::from_replay_parts(
        &w.exact,
        selected,
        *recipe.device().ordered_template(),
        recipe.device().provider_grouped_template().copied(),
        StructuredProductV2::FullLogits,
        recipe.device().readback(),
        recipe.physical_host_rows(),
        &algorithms,
        None,
        recipe.device().retries(),
    )
    .unwrap();
    assert_eq!(
        replayed.numerical_family_key(),
        Err(StructuredUnknown::WrongDomain)
    );
    let replayed = replayed
        .bind_validated_physical_domain(&w.exact, &d)
        .unwrap();
    assert_eq!(actual, replayed);
    assert_eq!(
        actual.numerical_family_key().unwrap(),
        replayed.numerical_family_key().unwrap()
    );
}

#[test]
fn typed_numerical_key_preserves_both_original_signature_transcripts() {
    let w = ordered_wave(
        &["fixture.b", "fixture.a"],
        ActualWaveGraphState::ConfiguredEager,
        [4; 32],
    );
    let d = domain(9);
    let input = scoped(&w, &d);
    let selected = w.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let key = input.numerical_family_key().unwrap();
    assert_eq!(key.workload_domain_signature(), d.sha256());
    assert_eq!(key.basis_axes(), input.regression_axes().len());
    assert_eq!(key.support_axes(), input.joint_support_coordinates().len());

    let mut original_route = Sha256::new();
    original_route.update(b"ferrum.numerical-family-route.v1\0");
    for value in [
        w.exact.kind as u64,
        w.exact.path as u64,
        w.exact.graph as u64,
        w.exact.row_order as u64,
        w.exact.recurrent_state_bytes,
        u64::from(recipe.device().retries()),
    ] {
        original_route.update(value.to_le_bytes());
    }
    let mut original_template = Sha256::new();
    original_template.update(b"ferrum.installed-algorithm-set-family.v1\0");
    original_template.update(d.sha256());
    original_template.update(original_route.finalize());
    original_template.update(input.owner().algorithm_domain);
    assert_eq!(
        family(&input).0.provider_template,
        StructuredTemplateV2::InstalledAlgorithmSetV1(original_template.finalize().into())
    );

    let mut original_family = Sha256::new();
    original_family.update(b"ferrum.homogeneous-decode-numerical-family.v1\0");
    original_family.update(MODEL_REVISION_V2.as_bytes());
    original_family.update(d.sha256());
    original_family.update(input.owner().algorithm_domain);
    original_family.update(
        serde_json::to_vec(&(
            recipe.physical_host_rows()[0].installed_policy,
            input.owner().product,
            input.owner().readback,
        ))
        .unwrap(),
    );
    for value in [
        w.exact.kind as u64,
        w.exact.path as u64,
        w.exact.graph as u64,
        w.exact.row_order as u64,
        u64::from(recipe.device().retries()),
    ] {
        original_family.update(value.to_le_bytes());
    }
    original_family.update((input.regression_axes().len() as u64).to_le_bytes());
    original_family.update((input.joint_support_coordinates().len() as u64).to_le_bytes());
    let original_signature: [u8; 32] = original_family.finalize().into();
    assert_eq!(key.signature().unwrap(), original_signature);
    let wrapper = NumericalFamilyInputV1::from_actual(&w.exact, selected, recipe, &d).unwrap();
    assert_eq!(*wrapper.family().signature(), original_signature);

    let mut wire = serde_json::to_value(key).unwrap();
    assert_eq!(
        serde_json::from_value::<NumericalFamilyKeyV1>(wire.clone()).unwrap(),
        key
    );
    wire.as_object_mut()
        .unwrap()
        .insert("execution_authority".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<NumericalFamilyKeyV1>(wire).is_err());
}

#[test]
fn numerical_family_contract_omission_preserves_old_bytes_and_policy_is_bound() {
    let old = NonNegativeEnvelopeContractV1 {
        algorithm_universe: None,
        planning_estimator: Default::default(),
        population_policy: Default::default(),
        workload_domain: domain(9),
        settings: EnvelopeSettings::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
        template_policy: Default::default(),
    };
    let bytes = serde_json::to_vec(&old).unwrap();
    assert!(!String::from_utf8(bytes.clone())
        .unwrap()
        .contains("planning_estimator"));
    assert!(!String::from_utf8(bytes.clone())
        .unwrap()
        .contains("template_policy"));
    assert!(!String::from_utf8(bytes.clone())
        .unwrap()
        .contains("population_policy"));
    assert_eq!(
        serde_json::from_slice::<NonNegativeEnvelopeContractV1>(&bytes).unwrap(),
        old
    );
    let mut new = old.clone();
    new.template_policy = StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1;
    assert_ne!(
        serde_json::to_vec(&old).unwrap(),
        serde_json::to_vec(&new).unwrap()
    );
    let mut empirical = old.clone();
    empirical.planning_estimator = NonNegativePlanningEstimatorV1::FittedResidualV1;
    let encoded = serde_json::to_vec(&empirical).unwrap();
    assert!(String::from_utf8(encoded.clone())
        .unwrap()
        .contains("fitted_residual_v1"));
    assert_eq!(
        serde_json::from_slice::<NonNegativeEnvelopeContractV1>(&encoded).unwrap(),
        empirical
    );
}

fn ordered_wave(
    entries: &[&str],
    graph: ActualWaveGraphState,
    policy: [u8; 32],
) -> CanonicalStructuredWave {
    let terminal = 9usize;
    let capture = true;
    let pending = [false, false];
    let exact_policy = policy;
    let numeric_policy = policy;
    let kv_tokens = 64;
    let generated = 2;
    let domain = HostContentDomainV1::PlainTextGreedyV1;
    let mut command = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(2)
    } else {
        SelectedCommandCostBuilderV1::new(2)
    };
    for (entry, work) in entries.iter().map(|entry| (*entry, 8u64)) {
        command
            .kernel(
                SelectedAlgorithmClassV1::new(entry, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: work,
                    padded_units: work,
                    inner_units_per_logical_unit: 2,
                    grid: [1, 1, 1],
                    scratch_bytes: 64,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
    }
    let selected = command.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.structured",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture.provider",
            implementation_fingerprint: "impl-v1",
            operation_fingerprint: "op-v1",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 2,
        token_count: 2,
        batching_form: "packed",
        compute_dispatch_count: entries.len() as u64,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for position in 0..2 {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens },
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: generated,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_policy_signature: exact_policy,
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(domain),
                    categorical_signature: numeric_policy,
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: generated,
                    maximum_output_tokens: if position == terminal || terminal == 2 {
                        generated + 1
                    } else {
                        generated + 18
                    },
                    sampling_history_tokens: generated,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: pending[position],
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    }
    let wave = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            graph,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let structured = wave
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .map(|recipe| recipe.as_ref().clone());
    CanonicalStructuredWave {
        exact: wave.exact,
        statistical: wave.statistical,
        structured,
    }
}
