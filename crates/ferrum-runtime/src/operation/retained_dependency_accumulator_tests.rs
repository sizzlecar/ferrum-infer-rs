//! Differential retention tests use authorities minted by actual admitted views.
#[path = "../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;

use super::*;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;
use vnext_device_operation_contract::{
    fixture_with_padded_persistent, fixture_with_padded_persistent_nodes, Fixture, TestRuntime,
};
use vnext_device_operation_wave_contract::{
    prepare_wave, setup_with_fixture, teardown, wave_active_bindings,
};

#[derive(Clone, Copy)]
struct Issue {
    node: usize,
    bank: u64,
    source_offset: u64,
    source_length: u64,
    destination_length: u64,
    validation: &'static str,
}

impl Issue {
    fn bank(node: usize, bank: u64) -> Self {
        Self {
            node,
            bank,
            source_offset: 0,
            source_length: 8,
            destination_length: 4,
            validation: "fixture.retention-index.v1",
        }
    }
}

fn with_issuer(run: impl FnOnce(&dyn Fn(&[Issue]) -> Vec<RetainedPlanDependencyAuthority>)) {
    with_fixture_issuer(fixture_with_padded_persistent(4096), run);
}

fn with_fixture_issuer(
    fixture: Fixture,
    run: impl FnOnce(&dyn Fn(&[Issue]) -> Vec<RetainedPlanDependencyAuthority>),
) {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = crate::OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        step.execution_lane(),
    )
    .unwrap();
    let issue = |specs: &[Issue]| {
        let first = specs
            .first()
            .expect("a live issuer has at least one dependency");
        assert!(specs.iter().all(|spec| spec.node == first.node));
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[first.node].node_id())
            .unwrap();
        let node = identity.materialize_node(first.node).unwrap();
        let invocation = BatchedOperationInvocation::from_wave_node::<TestRuntime, _>(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            &identity,
            &node,
            &wave,
            first.node,
            active.iter(),
        )
        .unwrap();
        let component = crate::WeightId::new("weight.component.left").unwrap();
        specs
            .iter()
            .map(|spec| {
                invocation
                    .retained_plan_dependency(RetainedPlanDependencySpec {
                        input_ordinal: 1,
                        component_id: &component,
                        source_offset_bytes: spec.source_offset,
                        source_length_bytes: spec.source_length,
                        persistent_offset_bytes: spec.bank * 8,
                        persistent_length_bytes: spec.destination_length,
                        alignment_bytes: 4,
                        validation_identity: spec.validation,
                    })
                    .unwrap()
            })
            .collect()
    };
    run(&issue);
    drop(identity);
    drop(active);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);
}

// Copies ownership already issued by the real invocation. This test-only
// helper neither constructs identity nor supplies replacement owner evidence.
fn copy_issued(authority: &RetainedPlanDependencyAuthority) -> RetainedPlanDependencyAuthority {
    RetainedPlanDependencyAuthority {
        scope: Arc::clone(&authority.scope),
        identity: authority.identity.clone(),
        _source_retention: authority._source_retention.clone(),
        _destination_retention: authority._destination_retention.clone(),
    }
}

fn retain_reference(
    retained: &mut Vec<RetainedPlanDependencyAuthority>,
    incoming: Vec<RetainedPlanDependencyAuthority>,
) {
    for authority in incoming {
        if !retained
            .iter()
            .any(|old| old.identity == authority.identity)
        {
            retained.push(authority);
        }
    }
}

fn assert_same_authorities(
    actual: &[RetainedPlanDependencyAuthority],
    expected: &[RetainedPlanDependencyAuthority],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.identity, expected.identity);
        assert!(Arc::ptr_eq(&actual.scope, &expected.scope));
        assert!(actual
            ._source_retention
            .same_owners(&expected._source_retention));
        assert!(actual
            ._destination_retention
            .same_owners(&expected._destination_retention));
    }
}

fn append_arm(
    indexed: bool,
    accumulator: &mut RetainedPlanDependencyAccumulator,
    reference: &mut Vec<RetainedPlanDependencyIdentity>,
    scope: &Arc<()>,
    incoming: Vec<EncodedRetainedPlanDependency<u32>>,
    commands: &mut Vec<u32>,
    leases: &mut Vec<RetainedPlanDependencyAuthority>,
) -> Result<(), VNextError> {
    if indexed {
        accumulator.append(scope, incoming, commands, leases)
    } else {
        append_dependencies(scope, incoming, reference, commands, leases)
    }
}

#[test]
fn retained_dependency_accumulator_preserves_first_order_owners_and_scope() {
    with_issuer(|issue| {
        let first = issue(&[Issue::bank(0, 2), Issue::bank(0, 0), Issue::bank(0, 1)]);
        let scope = Arc::clone(&first[0].scope);
        let incoming = [0, 1, 0, 2, 1].map(|i| copy_issued(&first[i]));
        let weak = Arc::downgrade(&scope);
        let mut indexed = RetainedPlanDependencyAccumulator::default();
        let mut reference = Vec::new();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        let mut reference_commands = Vec::new();
        let mut reference_leases = Vec::new();
        for (ordinal, authority) in incoming.into_iter().enumerate() {
            append_dependencies(
                &scope,
                vec![copy_issued(&authority).encode(ordinal as u32)],
                &mut reference,
                &mut reference_commands,
                &mut reference_leases,
            )
            .unwrap();
            indexed
                .append(
                    &scope,
                    vec![authority.encode(ordinal as u32)],
                    &mut commands,
                    &mut leases,
                )
                .unwrap();
        }
        assert_eq!(indexed.as_slice(), reference);
        assert_eq!(commands, [0, 1, 3]);
        assert_eq!(commands, reference_commands);
        assert_same_authorities(&leases, &reference_leases);
        let wrong_scope = issue(&[Issue::bank(0, 2)]).pop().unwrap();
        assert_eq!(wrong_scope.identity, first[0].identity);
        assert!(indexed
            .append(
                &scope,
                vec![wrong_scope.encode(99)],
                &mut commands,
                &mut leases
            )
            .is_err());
        assert_eq!(commands, [0, 1, 3]);
        drop(first);
        drop(scope);
        drop(reference_leases);
        assert!(weak.strong_count() > 0);
        drop(leases);
        assert_eq!(weak.strong_count(), 0);
    });
}

#[test]
fn retained_dependency_accumulator_preserves_overlap_errors_and_successful_prefix() {
    with_issuer(|issue| {
        let mut overlap = Issue::bank(0, 0);
        overlap.destination_length = 8;
        let entries = issue(&[
            Issue::bank(0, 1),
            overlap,
            Issue::bank(0, 0),
            Issue::bank(0, 2),
        ]);
        let scope = Arc::clone(&entries[0].scope);
        let mut indexed = RetainedPlanDependencyAccumulator::default();
        let mut reference = Vec::new();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        let mut reference_commands = Vec::new();
        let mut reference_leases = Vec::new();
        let old_error = append_dependencies(
            &scope,
            entries
                .iter()
                .enumerate()
                .map(|(i, a)| copy_issued(a).encode(i as u32))
                .collect(),
            &mut reference,
            &mut reference_commands,
            &mut reference_leases,
        )
        .unwrap_err();
        let new_error = indexed
            .append(
                &scope,
                entries
                    .into_iter()
                    .enumerate()
                    .map(|(i, a)| a.encode(i as u32))
                    .collect(),
                &mut commands,
                &mut leases,
            )
            .unwrap_err();
        assert_eq!(new_error.to_string(), old_error.to_string());
        assert_eq!(indexed.as_slice(), reference);
        assert_eq!(commands, [0, 1]);
        assert_eq!(commands, reference_commands);
        assert_same_authorities(&leases, &reference_leases);
    });
}

#[test]
fn retained_dependency_accumulator_keeps_transaction_identity_and_separate_ownership() {
    with_issuer(|issue| {
        with_issuer(|other| {
            let one = issue(&[Issue::bank(0, 0)]).pop().unwrap();
            let two = other(&[Issue::bank(0, 0)]).pop().unwrap();
            assert_eq!(
                one.identity.destination.descriptor.resource_id,
                two.identity.destination.descriptor.resource_id
            );
            assert_ne!(
                one.identity.destination.transaction,
                two.identity.destination.transaction
            );
            let mut indexed = RetainedPlanDependencyAccumulator::default();
            let mut commands = Vec::new();
            let mut leases = Vec::new();
            for (i, a) in [one, two].into_iter().enumerate() {
                let scope = Arc::clone(&a.scope);
                indexed
                    .append(&scope, vec![a.encode(i as u32)], &mut commands, &mut leases)
                    .unwrap();
            }
            assert_eq!(indexed.as_slice().len(), 2);
            assert_eq!(commands, [0, 1]);
            assert!(!Arc::ptr_eq(&leases[0].scope, &leases[1].scope));
        });
    });
}

#[test]
fn retained_dependency_accumulator_does_not_partition_conflicts_by_node_or_generation() {
    with_issuer(|issue| {
        let original = issue(&[Issue::bank(0, 0)]).pop().unwrap();
        let scope = Arc::clone(&original.scope);
        let identity = original.identity.clone();
        let mut indexed = RetainedPlanDependencyAccumulator::default();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        indexed
            .append(
                &scope,
                vec![original.encode(0_u32)],
                &mut commands,
                &mut leases,
            )
            .unwrap();
        assert!(indexed.check_identity(&identity).unwrap());
        // Adversarial identity values exercise only the pure conflict lookup;
        // no replacement authority or physical owner is constructed.
        let mut changed_node = identity.clone();
        changed_node.node = NodeId::new("node.other").unwrap();
        let mut changed_generation = identity.clone();
        changed_generation.destination.generation += 1;
        let mut changed_provider = identity.clone();
        changed_provider.provider = ProviderId::new("provider.other").unwrap();
        for changed in [changed_node, changed_generation, changed_provider] {
            assert!(indexed.check_identity(&changed).is_err());
        }
        let mut adjacent = identity.clone();
        adjacent.destination.offset += adjacent.destination.length;
        assert!(!indexed.check_identity(&adjacent).unwrap());
        assert_eq!(commands, [0]);
    });
}

// Dependency-prefix lifecycle, including a mirror of the unchanged final
// Step retain algorithm and its mutex. Provisioning and initial issuance are
// outside timing; the timed input copies retain real, already-issued owners.
fn lifecycle(
    templates: &[Vec<RetainedPlanDependencyAuthority>],
    indexed: bool,
    occurrences: usize,
) -> usize {
    let retained = Mutex::new(Vec::<RetainedPlanDependencyAuthority>::new());
    let mut accumulator = RetainedPlanDependencyAccumulator::default();
    let mut identities = Vec::new();
    let mut commands = Vec::new();
    let mut leases = Vec::new();
    for group in templates {
        let incoming = group
            .iter()
            .enumerate()
            .flat_map(|(i, a)| (0..occurrences).map(move |_| copy_issued(a).encode(i as u32)))
            .collect();
        append_arm(
            indexed,
            &mut accumulator,
            &mut identities,
            &group[0].scope,
            incoming,
            &mut commands,
            &mut leases,
        )
        .unwrap();
    }
    let count = if indexed {
        accumulator.as_slice().len()
    } else {
        identities.len()
    };
    {
        let mut retained = retained.lock().unwrap();
        retain_reference(&mut retained, leases);
        assert_eq!(retained.len(), count);
    }
    black_box(commands.len());
    black_box(count);
    drop(commands);
    drop(identities);
    drop(accumulator);
    drop(retained);
    count
}

#[test]
#[ignore = "CPU full dependency lifecycle diagnostic; no performance pass threshold"]
fn retained_dependency_accumulator_cpu_paired() {
    // Frozen f5 config's 128 prepared projection nodes: 369 staged leaves plus
    // 96 eligible Q8 F32 leaves. This is config/source-derived, not a runtime
    // dependency counter and not multiplied by request concurrency.
    const FANOUT: [usize; 128] = [
        5, 3, 4, 3, 5, 3, 3, 3, 5, 3, 5, 3, 4, 3, 4, 3, 5, 3, 5, 3, 5, 3, 2, 3, 5, 3, 5, 3, 5, 3,
        3, 3, 5, 3, 5, 3, 5, 3, 2, 3, 5, 3, 4, 3, 5, 3, 2, 3, 5, 3, 4, 3, 5, 3, 1, 3, 5, 3, 5, 3,
        5, 3, 1, 3, 5, 3, 5, 3, 5, 3, 2, 3, 5, 3, 5, 3, 5, 3, 3, 3, 5, 3, 5, 3, 5, 3, 4, 3, 5, 3,
        5, 3, 5, 3, 4, 3, 5, 3, 5, 3, 5, 3, 2, 3, 5, 3, 5, 3, 5, 3, 3, 3, 4, 3, 5, 3, 5, 3, 2, 3,
        5, 3, 5, 3, 5, 3, 2, 1,
    ];
    with_fixture_issuer(
        fixture_with_padded_persistent_nodes(48, FANOUT.len()),
        |issue| {
            let templates = FANOUT
                .iter()
                .enumerate()
                .map(|(node, &banks)| {
                    issue(
                        &(0..banks)
                            .map(|bank| Issue::bank(node, bank as u64))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let expected = FANOUT.iter().sum::<usize>();
            for occurrences in [1, 2] {
                assert_eq!(lifecycle(&templates, false, occurrences), expected);
                assert_eq!(lifecycle(&templates, true, occurrences), expected);
                for pair in 0..8 {
                    let order = if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    };
                    let mut elapsed = [0_u128; 2];
                    for indexed in order {
                        let started = Instant::now();
                        let mut count = 0;
                        for _ in 0..16 {
                            count +=
                                black_box(lifecycle(black_box(&templates), indexed, occurrences));
                        }
                        elapsed[usize::from(indexed)] = started.elapsed().as_nanos();
                        assert_eq!(count, expected * 16);
                    }
                    println!(
                        "{}",
                        serde_json::json!({
                                "kind":"retained_dependency_accumulator_cpu_pair",
                                "fixture_nodes":FANOUT.len(),"dependencies":expected,"occurrences":occurrences,
                                "pair":pair,"order":if pair%2==0 {"scan,indexed"} else {"indexed,scan"},
                                "iterations":16,"scan_ns":elapsed[0],"indexed_ns":elapsed[1],
                                "debug_assertions":cfg!(debug_assertions),
                        "scope":"real admitted issuer; includes append, per-wave resource index, authority/identity clones, mutex and mirror of final Step retain algorithm, duplicate release and all final drops",
                                "population":"recorded config plus source inferred465/128; not measured production counter",
                        "excluded":"initial fixture provisioning/issuance, actual Step method call, opaque backend command payload, other Step resources and GPU; duplicate case diagnostic only"
                            })
                    );
                }
            }
        },
    );
}

#[test]
fn resident_dependency_closure_rejects_conflicts_and_a_different_replay_seal() {
    with_issuer(|issue| {
        let authorities = issue(&[Issue::bank(0, 0), Issue::bank(0, 0), Issue::bank(0, 1)]);
        let expected = vec![
            authorities[0].identity.clone(),
            authorities[2].identity.clone(),
        ];
        let sealed = SealedResidentPlanDependencies::seal(authorities.iter(), &expected).unwrap();
        assert_eq!(&*sealed.identities, &expected);
        assert!(SealedResidentPlanDependencies::seal(authorities.iter(), &expected[..1]).is_err());
        let mut conflicting = Issue::bank(0, 0);
        conflicting.validation = "fixture.incompatible-validation.v1";
        let authorities = issue(&[Issue::bank(0, 0), conflicting]);
        let expected = authorities
            .iter()
            .map(|a| a.identity.clone())
            .collect::<Vec<_>>();
        assert!(SealedResidentPlanDependencies::seal(authorities.iter(), &expected).is_err());
    });
}

#[test]
fn resident_dependency_closure_issues_fresh_linear_scopes_and_retains_current_use() {
    with_issuer(|issue| {
        let authorities = issue(&[Issue::bank(0, 0)]);
        let old_scope = Arc::downgrade(&authorities[0].scope);
        let expected = vec![authorities[0].identity.clone()];
        let sealed = SealedResidentPlanDependencies::seal(authorities.iter(), &expected).unwrap();
        drop(authorities);
        assert!(
            old_scope.upgrade().is_none(),
            "cold closure must not retain a past invocation scope"
        );
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let first = sealed.issue(DeviceBufferRetention::plan(Arc::clone(&owner)));
        let first_scope = Arc::clone(first.scope());
        let second = sealed.issue(DeviceBufferRetention::plan(Arc::new(())));
        let second_scope = Arc::clone(second.scope());
        assert!(!Arc::ptr_eq(&first_scope, &second_scope));
        drop(owner);
        let encoded = first.encode(17_u32);
        assert!(
            weak.upgrade().is_some(),
            "the encoded batch must retain the current use"
        );
        assert!(
            encoded.into_command(&second_scope, &sealed).is_err(),
            "another fresh use cannot consume this batch"
        );
        assert!(weak.upgrade().is_none());
        assert_eq!(
            second
                .encode(23_u32)
                .into_command(&second_scope, &sealed)
                .unwrap(),
            23
        );
    });
}
