mod vnext_core_contract;
use vnext_core_contract::*;
#[path = "vnext_sequence_checkpoint_contract/fixture.rs"]
mod fixture;
use fixture::{Fixture, Spec};

#[test]
fn completed_input_capture_requires_every_selected_provider() {
    for (providers, supported) in [
        (vec![], false),
        (vec!["provider.alternative"], false),
        (vec!["provider.selected"], false),
        (vec!["provider.selected", "provider.output"], true),
    ] {
        let fixture = Fixture::build(Spec {
            output_only_input: Some(id("value.selection")),
            unselected_support: true,
            completed_input_providers: providers.into_iter().map(id).collect(),
            ..Spec::default()
        })
        .unwrap();
        let layout = fixture.layout();
        assert_eq!(layout.permits_capture_from(3, 4, 4), supported);
        assert_eq!(
            layout.completed_input_capture().is_unsupported(),
            !supported
        );
        assert!(layout.permits_capture_from(3, 4, 5));
        assert!(!layout.permits_suffix(4, 4));
        let source = layout
            .bind_inputs(&id("value.input"), &[1, 2, 3, 4], &BTreeMap::new())
            .unwrap();
        let target = layout
            .bind_inputs(&id("value.input"), &[1, 2, 3, 4, 5], &BTreeMap::new())
            .unwrap();
        assert_eq!(source.matches_at(&target, 4), supported);
        assert!(!source.matches_at(&source, 4));
        assert_eq!(
            fixture
                .revalidate(&fixture.plan.to_json().unwrap())
                .unwrap(),
            fixture.plan
        );
    }
}

#[test]
fn completed_input_capture_keeps_actual_span_and_future_suffix_constraints() {
    use std::num::NonZeroU64;
    let boundary = CheckpointBoundaryConstraint::new(
        CheckpointTokenSpanConstraint::new(
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(4).unwrap(),
        )
        .unwrap(),
        CheckpointTokenSpanConstraint::new(
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    for dependency in [
        CheckpointInputDependency::ExactTokenPrefix,
        CheckpointInputDependency::EntireTokenInput,
    ] {
        let fixture = Fixture::build(Spec {
            dependency,
            numerics: CheckpointPartitionNumerics::SamePartitionOnly,
            boundaries: boundary,
            completed_input_providers: BTreeSet::from([id("provider.selected")]),
            ..Spec::default()
        })
        .unwrap();
        let layout = fixture.layout();
        assert_eq!(layout.input_dependency(), dependency);
        assert_eq!(
            layout.providers()[0].contract().partition_numerics(),
            CheckpointPartitionNumerics::SamePartitionOnly
        );
        assert!(layout.permits_capture_from(3, 7, 7));
        assert!(layout.permits_capture_from(3, 7, 10));
        for (start, end, input) in [
            (3, 8, 8),
            (7, 7, 7),
            (8, 7, 7),
            (0, 0, 0),
            (3, 7, 6),
            (3, 7, 9),
        ] {
            assert!(!layout.permits_capture_from(start, end, input));
        }
        assert!(layout.permits_suffix(7, 10));
        assert!(!layout.permits_suffix(7, 7));
        assert!(!layout.permits_suffix(7, 9));
        let source = layout
            .bind_inputs(&id("value.input"), &[1, 2, 3, 4], &BTreeMap::new())
            .unwrap();
        let target = layout
            .bind_inputs(&id("value.input"), &[1, 2, 3, 4, 5, 6, 7], &BTreeMap::new())
            .unwrap();
        assert_eq!(
            source.matches_at(&target, 4),
            dependency == CheckpointInputDependency::ExactTokenPrefix
        );
    }
}

#[test]
fn checkpoint_output_only_inputs_require_real_state_independence() {
    for input in ["value.selection", "value.renamed"] {
        let safe = Fixture::build(Spec {
            output_only_input: Some(id(input)),
            ..Spec::default()
        })
        .unwrap();
        assert_eq!(
            safe.layout().inputs().output_only_inputs(),
            &BTreeSet::from([id(input)])
        );
        assert_eq!(
            safe.revalidate(&safe.plan.to_json().unwrap()).unwrap(),
            safe.plan
        );
        // The same family/node/operation/value names now have an actual extra
        // edge from this input into the state-writing operation.
        let unsafe_plan = Fixture::build(Spec {
            output_only_input: Some(id(input)),
            output_only_feeds_state: true,
            ..Spec::default()
        })
        .unwrap();
        assert!(unsafe_plan.reasons().iter().any(|reason| matches!(reason,
            SequenceCheckpointUnsupportedReason::OutputOnlyInputAffectsState { value_id, node_id, .. }
                if value_id == &id(input) && node_id == &id("node.main"))));
        assert!(unsafe_plan.plan.checkpoint_byte_plan(2).is_err());
    }
}

#[test]
fn checkpoint_output_only_roles_do_not_relax_conditioning_or_unknown_operations() {
    let fixture = Fixture::build(Spec {
        output_only_input: Some(id("value.selection")),
        conditioning: true,
        ..Spec::default()
    })
    .unwrap();
    assert!(fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &BTreeMap::new())
        .is_err());
    let content = |byte| {
        BTreeMap::from([(
            id("value.conditioning"),
            CheckpointCanonicalInput::new(resolved_tensor(ElementType::F32), vec![byte; 16])
                .unwrap(),
        )])
    };
    let source = fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &content(0))
        .unwrap();
    let equal = fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 4], &content(0))
        .unwrap();
    let changed = fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &content(1))
        .unwrap();
    assert!(source.matches_at(&equal, 2));
    assert!(!source.matches_at(&changed, 2));
    let unknown = Fixture::build(Spec {
        output_only_input: Some(id("value.selection")),
        output_only_unknown_operation: true,
        ..Spec::default()
    })
    .err()
    .expect("unknown operation must not establish input independence");
    assert!(
        matches!(unknown, VNextError::UnsupportedOperation { operation_id, .. }
        if operation_id == "operation.unknown")
    );
    let undeclared = Fixture::build(Spec {
        output_only_input: Some(id("value.selection")),
        declare_provider: false,
        ..Spec::default()
    })
    .unwrap();
    assert!(undeclared.reasons().iter().any(|reason| matches!(reason,
        SequenceCheckpointUnsupportedReason::ProviderUndeclared { node_id, .. } if node_id == &id("node.output"))));
}

#[test]
fn checkpoint_layout_is_derived_by_real_plan_build_for_distinct_families() {
    for family_id in ["family.checkpoint-first", "family.checkpoint-second"] {
        let fixture = Fixture::build(Spec {
            family_id: id(family_id),
            ..Spec::default()
        })
        .unwrap();
        assert_eq!(fixture.layout().states().len(), 2);
        assert!(fixture.layout().permits_capture_from(0, 3, 4));
        assert!(fixture.layout().permits_capture_from(3, 4, 5));
        assert!(!fixture.layout().permits_capture_from(3, 3, 4));
        assert!(!fixture.layout().permits_suffix(4, 4));
        let bytes = fixture.plan.checkpoint_byte_plan(3).unwrap();
        assert_eq!(bytes.logical_bytes(), 16); // 3 token values plus one boundary summary
        assert_eq!(bytes.resources()[0].ranges()[0].source(), 0..12);
        assert_eq!(bytes.resources()[1].ranges()[0].source(), 0..4);
        assert!(fixture.plan.checkpoint_byte_plan(0).is_err());
        assert!(fixture.plan.checkpoint_byte_plan(65).is_err());
        assert!(fixture.plan.checkpoint_byte_plan(u64::MAX).is_err());
        assert_eq!(
            fixture
                .revalidate(&fixture.plan.to_json().unwrap())
                .unwrap(),
            fixture.plan
        );
    }
}

#[test]
fn checkpoint_missing_declarations_and_hidden_state_fail_closed() {
    let missing_inputs = Fixture::build(Spec {
        declare_inputs: false,
        ..Spec::default()
    })
    .unwrap();
    assert!(missing_inputs.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::InputsUndeclared
    )));
    let missing_selected = Fixture::build(Spec {
        declare_provider: false,
        unselected_support: true,
        ..Spec::default()
    })
    .unwrap();
    assert!(missing_selected.reasons().iter().any(|reason| matches!(reason, SequenceCheckpointUnsupportedReason::ProviderUndeclared { provider_id, .. } if provider_id == &id("provider.selected"))));
    let persistent = Fixture::build(Spec {
        hidden_persistent: true,
        ..Spec::default()
    })
    .unwrap();
    assert!(persistent.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::ProviderPersistentWorkspace { .. }
    )));
    let mut spec = Spec::default();
    spec.states[1].checkpoint = StateCheckpointCapability::Unsupported;
    let missing_state = Fixture::build(spec).unwrap();
    assert!(missing_state.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateUndeclared { .. }
    )));
    let mut spec = Spec::default();
    spec.states[1].lifetime = StateLifetime::Request;
    spec.states[1].initialization = StateInitialization::None;
    spec.states[1].checkpoint = StateCheckpointCapability::Unsupported;
    let request_state = Fixture::build(spec).unwrap();
    assert!(request_state.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateLifetime {
            lifetime: AllocationLifetime::Request,
            ..
        }
    )));
    let readonly = Fixture::build(Spec {
        read_only_state: true,
        ..Spec::default()
    })
    .unwrap();
    assert!(readonly.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateWithoutWriter { .. }
    )));
}

#[test]
fn checkpoint_storage_mapping_requires_explicit_selected_port_abi_and_exact_stride() {
    let missing = Fixture::build(Spec {
        declare_ports: false,
        ..Spec::default()
    })
    .unwrap();
    assert!(missing.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StatePortUndeclared { .. }
    )));
    let wrong_profile = Fixture::build(Spec {
        port_profile: paged_storage_profile(65536),
        ..Spec::default()
    })
    .unwrap();
    assert!(wrong_profile.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StatePortUndeclared { .. }
    )));
    let mut wrong_stride = Spec::default();
    wrong_stride.states[0].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 8,
        maximum_tokens: 64,
    };
    let wrong_stride = Fixture::build(wrong_stride).unwrap();
    assert!(wrong_stride.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateLayout { .. }
    )));
    let mut wrong_contents = Spec::default();
    wrong_contents.layouts[0] = ProviderCheckpointStateLayout::ContiguousBoundaryValue;
    let wrong_contents = Fixture::build(wrong_contents).unwrap();
    assert!(wrong_contents.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateLayout { .. }
    )));
    let paged = Fixture::build(Spec {
        profile: paged_storage_profile(65536),
        port_profile: paged_storage_profile(65536),
        ..Spec::default()
    })
    .unwrap();
    assert_eq!(
        paged.plan.checkpoint_byte_plan(3).unwrap().logical_bytes(),
        16
    );
    assert_ne!(
        paged.layout().fingerprint().unwrap(),
        Fixture::build(Spec::default())
            .unwrap()
            .layout()
            .fingerprint()
            .unwrap()
    );
}

#[test]
fn checkpoint_compaction_omits_holes_and_rejects_undeclared_aliases() {
    let mut spec = Spec::default();
    spec.states[0].capacity_demand = StateCapacityDemand::FixedPerScope;
    spec.states[0].checkpoint = spec.states[1].checkpoint;
    spec.layouts[0] = ProviderCheckpointStateLayout::ContiguousBoundaryValue;
    spec.locations = vec![(id("resource.shared"), 0), (id("resource.shared"), 16)];
    let fixture = Fixture::build(spec).unwrap();
    let bytes = fixture.plan.checkpoint_byte_plan(3).unwrap();
    assert_eq!(bytes.resources().len(), 1);
    assert_eq!(bytes.logical_bytes(), 8);
    assert_eq!(bytes.resources()[0].ranges()[0].source(), 0..4);
    assert_eq!(bytes.resources()[0].ranges()[1].source(), 16..20);
    assert_eq!(bytes.resources()[0].ranges()[1].checkpoint_offset(), 4);
    let mut spec = Spec::default();
    spec.locations = vec![(id("resource.shared"), 0), (id("resource.shared"), 0)];
    assert!(Fixture::build(spec).is_err());
}

#[test]
fn checkpoint_conditioning_and_suffix_dependencies_compare_full_canonical_contents() {
    let fixture = Fixture::build(Spec {
        conditioning: true,
        ..Spec::default()
    })
    .unwrap();
    let empty = BTreeMap::new();
    assert!(fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &empty)
        .unwrap_err()
        .iter()
        .any(|reason| matches!(
            reason,
            SequenceCheckpointUnsupportedReason::MissingInput { .. }
        )));
    let tensor = resolved_tensor(ElementType::F32);
    let content = BTreeMap::from([(
        id("value.conditioning"),
        CheckpointCanonicalInput::new(tensor.clone(), vec![0; 16]).unwrap(),
    )]);
    let source = fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &content)
        .unwrap();
    let target = fixture
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 4, 5], &content)
        .unwrap();
    assert!(source.matches_at(&target, 2));
    let changed = BTreeMap::from([(
        id("value.conditioning"),
        CheckpointCanonicalInput::new(tensor.clone(), vec![1; 16]).unwrap(),
    )]);
    assert!(!source.matches_at(
        &fixture
            .layout()
            .bind_inputs(&id("value.input"), &[1, 2, 3], &changed)
            .unwrap(),
        2
    ));
    assert!(CheckpointCanonicalInput::new(tensor, vec![0; 15]).is_err());
    assert!(fixture
        .layout()
        .bind_inputs(&id("value.wrong"), &[1, 2, 3], &content)
        .is_err());
    let complete = Fixture::build(Spec {
        dependency: CheckpointInputDependency::EntireTokenInput,
        ..Spec::default()
    })
    .unwrap();
    let source = complete
        .layout()
        .bind_inputs(&id("value.input"), &[1, 2, 3], &empty)
        .unwrap();
    assert!(!source.matches_at(
        &complete
            .layout()
            .bind_inputs(&id("value.input"), &[1, 2, 4], &empty)
            .unwrap(),
        2
    ));
    assert!(source.matches_at(
        &complete
            .layout()
            .bind_inputs(&id("value.input"), &[1, 2, 3], &empty)
            .unwrap(),
        2
    ));
}

#[test]
fn checkpoint_oracle_does_not_authorize_unobserved_state_effects() {
    let fixture = Fixture::build(Spec {
        numerics: CheckpointPartitionNumerics::OperationOracle,
        ..Spec::default()
    })
    .unwrap();
    assert!(fixture.reasons().iter().any(|reason| matches!(
        reason,
        SequenceCheckpointUnsupportedReason::StateOracleCoverage { .. }
    )));
    let bitwise = Fixture::build(Spec::default()).unwrap();
    let same = Fixture::build(Spec {
        numerics: CheckpointPartitionNumerics::SamePartitionOnly,
        ..Spec::default()
    })
    .unwrap();
    assert_ne!(
        bitwise.layout().fingerprint().unwrap(),
        same.layout().fingerprint().unwrap()
    );
}

#[test]
fn checkpoint_wire_requires_rebuild_and_old_undeclared_plans_keep_their_wire_shape() {
    let original = plan_fixture(0);
    let old_wire = original.plan.to_json().unwrap();
    let value: Value = serde_json::from_slice(&old_wire).unwrap();
    assert!(value["payload"].get("sequence_checkpoint_layout").is_none());
    let round_trip = ExecutionPlan::from_json_validated(
        &old_wire,
        &original.family,
        &original.catalog,
        &original.policy,
        original.node_resolutions.clone(),
    )
    .unwrap();
    assert_eq!(old_wire, round_trip.to_json().unwrap());
    assert_eq!(original.plan.plan_hash(), round_trip.plan_hash());
    let fixture = Fixture::build(Spec::default()).unwrap();
    let wire: Value = serde_json::from_slice(&fixture.plan.to_json().unwrap()).unwrap();
    assert!(wire["payload"]["sequence_checkpoint_layout"]
        .get("completed_input_capture")
        .is_none());
    let mut upgraded = wire.clone();
    upgraded["payload"]["sequence_checkpoint_layout"]["completed_input_capture"] =
        json!("supported");
    assert!(fixture
        .revalidate(&serde_json::to_vec(&upgraded).unwrap())
        .is_err());
    let mut forged = wire.clone();
    forged["payload"]["sequence_checkpoint_layout"]["states"][0]["offset_bytes"] = json!(16);
    assert!(fixture
        .revalidate(&serde_json::to_vec(&forged).unwrap())
        .is_err());
    let mut future = wire.clone();
    future["payload"]["sequence_checkpoint_layout"]["contract_version"] =
        serde_json::to_value(ContractVersion::new(2, 0)).unwrap();
    assert!(ExecutionPlan::decode_untrusted(&serde_json::to_vec(&future).unwrap()).is_err());
    let mut missing = wire;
    missing["payload"]
        .as_object_mut()
        .unwrap()
        .remove("sequence_checkpoint_layout");
    assert!(fixture
        .revalidate(&serde_json::to_vec(&missing).unwrap())
        .is_err());
}
