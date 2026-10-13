//! Differential retention tests use authorities minted by actual admitted views.
#[path = "../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;

use super::*;
use crate::vnext::resource::StepPlanDependencyRetention;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;
use vnext_device_operation_contract::{fixture_with_padded_persistent, TestRuntime};
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
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_padded_persistent(4096));
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = crate::vnext::OperationDispatch::bind_submission_wave_identity(
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
        let component = crate::vnext::WeightId::new("weight.component.left").unwrap();
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

#[test]
fn plan_dependency_retention_preserves_first_authority_order_and_drop_across_calls() {
    with_issuer(|issue| {
        let specifications = [
            Issue::bank(1, 2),
            Issue::bank(0, 0),
            Issue::bank(1, 2),
            Issue::bank(0, 1),
            Issue::bank(0, 0),
        ];
        let mut indexed = StepPlanDependencyRetention::default();
        let mut reference = Vec::new();
        let mut scopes = Vec::new();
        for group in specifications.chunks(2) {
            // Each incoming occurrence comes from a separate live invocation;
            // equal identities can therefore own different issuance scopes.
            let incoming = group
                .iter()
                .flat_map(|spec| issue(&[*spec]))
                .collect::<Vec<_>>();
            scopes.extend(
                incoming
                    .iter()
                    .map(|authority| Arc::downgrade(&authority.scope)),
            );
            retain_reference(&mut reference, incoming.iter().map(copy_issued).collect());
            indexed.retain(incoming);
            assert_same_authorities(indexed.authorities(), &reference);
        }
        assert_eq!(
            indexed
                .authorities()
                .iter()
                .map(|authority| authority.identity.persistent_offset_bytes())
                .collect::<Vec<_>>(),
            [16, 0, 8]
        );
        drop(reference);
        for (ordinal, scope) in scopes.iter().enumerate() {
            assert_eq!(scope.strong_count() > 0, [0, 1, 3].contains(&ordinal));
        }
        drop(indexed);
        assert!(scopes.iter().all(|scope| scope.strong_count() == 0));
    });
}

#[test]
fn plan_dependency_retention_preserves_legal_range_validation_and_transaction_differences() {
    with_issuer(|issue| {
        let base = Issue::bank(0, 0);
        let mut source_range = base;
        source_range.source_offset = 4;
        source_range.source_length = 4;
        let mut destination_range = base;
        destination_range.destination_length = 8;
        let mut validation = base;
        validation.validation = "fixture.retention-index.other-arithmetic";
        let mut indexed = StepPlanDependencyRetention::default();
        let mut reference = Vec::new();
        for spec in [
            base,
            source_range,
            destination_range,
            validation,
            Issue::bank(1, 0),
            base,
        ] {
            let incoming = issue(&[spec]);
            retain_reference(&mut reference, incoming.iter().map(copy_issued).collect());
            indexed.retain(incoming);
        }
        assert_same_authorities(indexed.authorities(), &reference);
        assert_eq!(indexed.authorities().len(), 5);
        with_issuer(|other_issue| {
            let other = other_issue(&[base]);
            assert_ne!(
                indexed.authorities()[0].identity.source.transaction,
                other[0].identity.source.transaction
            );
            retain_reference(&mut reference, other.iter().map(copy_issued).collect());
            indexed.retain(other);
            assert_same_authorities(indexed.authorities(), &reference);
            assert_eq!(indexed.authorities().len(), 6);
            drop(reference);
            drop(indexed);
        });
    });
}

#[test]
fn plan_dependency_retention_identity_keeps_complete_equality_and_live_scope_checks() {
    with_issuer(|issue| {
        let mut issued = issue(&[Issue::bank(0, 0)]);
        let original = issued.pop().unwrap();
        let identity = original.identity.clone();
        // Key-only adversarial values cannot authorize execution or replace
        // the real source/destination retentions used by the store tests.
        let mut changes = Vec::new();
        macro_rules! change {
            ($field:ident $(.$nested:ident)*, $value:expr) => {{
                let mut changed = identity.clone();
                changed.$field$(.$nested)* = $value;
                changes.push(changed);
            }};
        }
        change!(node, NodeId::new("node.other").unwrap());
        change!(
            provider,
            crate::vnext::ProviderId::new("provider.other").unwrap()
        );
        change!(
            component,
            crate::vnext::WeightId::new("weight.other").unwrap()
        );
        change!(input_ordinal, 2);
        change!(validation_identity, "different-validation".to_owned());
        change!(source.generation, identity.source.generation + 1);
        change!(source.offset, identity.source.offset + 1);
        change!(source.length, identity.source.length + 1);
        change!(
            source.descriptor.resource_id,
            crate::vnext::ResourceId::new("resource.other").unwrap()
        );
        change!(
            source.descriptor.size_bytes,
            identity.source.descriptor.size_bytes + 1
        );
        change!(
            source.descriptor.alignment_bytes,
            identity.source.descriptor.alignment_bytes * 2
        );
        change!(source.descriptor.usage, BufferUsage::Transfer);
        change!(source.descriptor.element_type, ElementType::F16);
        change!(destination.generation, identity.destination.generation + 1);
        change!(destination.offset, identity.destination.offset + 1);
        change!(destination.length, identity.destination.length + 1);
        change!(
            destination.descriptor.resource_id,
            crate::vnext::ResourceId::new("persistent.other").unwrap()
        );
        change!(
            destination.descriptor.size_bytes,
            identity.destination.descriptor.size_bytes + 1
        );
        change!(
            destination.descriptor.alignment_bytes,
            identity.destination.descriptor.alignment_bytes * 2
        );
        change!(destination.descriptor.usage, BufferUsage::Transfer);
        change!(destination.descriptor.element_type, ElementType::U32);
        assert!(changes.iter().all(|changed| changed != &identity));

        let stale_scope = Arc::clone(&original.scope);
        let current = issue(&[Issue::bank(0, 0)]).pop().unwrap();
        assert!(!Arc::ptr_eq(&stale_scope, &current.scope));
        let mut identities = Vec::new();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        assert!(append_dependencies(
            &current.scope,
            vec![original.encode(())],
            &mut identities,
            &mut commands,
            &mut leases
        )
        .is_err());
        assert!(identities.is_empty() && commands.is_empty() && leases.is_empty());
        drop(current);
    });
}

enum RetentionArm {
    Scan(Vec<RetainedPlanDependencyAuthority>),
    Indexed(StepPlanDependencyRetention),
}

impl RetentionArm {
    fn retain(&mut self, incoming: Vec<RetainedPlanDependencyAuthority>) {
        match self {
            Self::Scan(authorities) => retain_reference(authorities, incoming),
            Self::Indexed(store) => store.retain(incoming),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Scan(authorities) => authorities.len(),
            Self::Indexed(store) => store.authorities().len(),
        }
    }
}

fn inclusive_lifecycle(
    templates: &[Vec<RetainedPlanDependencyAuthority>],
    calls: usize,
    indexed: bool,
) -> usize {
    let store = Mutex::new(if indexed {
        RetentionArm::Indexed(StepPlanDependencyRetention::default())
    } else {
        RetentionArm::Scan(Vec::new())
    });
    for _ in 0..calls {
        let mut identities = Vec::new();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        for group in templates {
            let encoded = group
                .iter()
                .flat_map(|authority| {
                    [
                        copy_issued(authority).encode(()),
                        copy_issued(authority).encode(()),
                    ]
                })
                .collect();
            append_dependencies(
                &group[0].scope,
                encoded,
                &mut identities,
                &mut commands,
                &mut leases,
            )
            .unwrap();
        }
        store.lock().unwrap().retain(leases);
        black_box(commands.len());
        // The unchanged append identities/commands and duplicate authorities
        // release inside this same measured lifecycle, before the next call.
    }
    let retained = store.lock().unwrap().len();
    black_box(retained);
    drop(store);
    retained
}

#[test]
#[ignore = "paired CPU diagnostic includes unchanged append and complete retention lifecycle; no performance assertion"]
fn plan_dependency_retention_cpu_inclusive_paired() {
    with_issuer(|issue| {
        for (label, nodes, banks, calls) in [
            ("two_node_64_banks", 2, 32, 1),
            ("two_node_192_banks_repeated_calls", 2, 96, 3),
            ("one_node_384_banks_worst_bucket", 1, 384, 3),
        ] {
            let templates = (0..nodes)
                .map(|node| {
                    let specs = (0..banks)
                        .map(|bank| Issue::bank(node, bank))
                        .collect::<Vec<_>>();
                    issue(&specs)
                })
                .collect::<Vec<_>>();
            let expected = templates.iter().map(Vec::len).sum::<usize>();
            assert_eq!(inclusive_lifecycle(&templates, calls, false), expected);
            assert_eq!(inclusive_lifecycle(&templates, calls, true), expected);
            for pair in 0..8 {
                let order = if pair % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                let mut elapsed = [0_u128; 2];
                for indexed in order {
                    let started = Instant::now();
                    let mut total_retained = 0;
                    for _ in 0..16 {
                        total_retained +=
                            black_box(inclusive_lifecycle(black_box(&templates), calls, indexed));
                    }
                    elapsed[usize::from(indexed)] = started.elapsed().as_nanos();
                    assert_eq!(total_retained, expected * 16);
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "kind":"plan_dependency_retention_cpu_inclusive_pair",
                        "geometry":label,"actual_fixture_nodes":nodes,"unique_dependencies":expected,
                        "duplicate_occurrences_per_dependency_per_call":2,"retention_calls":calls,
                        "pair":pair,"order":if pair % 2 == 0 {"scan,indexed"} else {"indexed,scan"},
                        "iterations":16,"scan_ns":elapsed[0],"indexed_ns":elapsed[1],
                        "scope":"synthetic legal Plan banks; includes store init, authority identity/owner clones, unchanged append scope/conflict/duplicate checks, per-call mutex, duplicate release and final retention drop",
                        "excluded":"fixture provisioning and initial live invocation issuance, device work and other Step resources; no measured production dependency count or inference-throughput claim"
                    })
                );
            }
        }
    });
}
