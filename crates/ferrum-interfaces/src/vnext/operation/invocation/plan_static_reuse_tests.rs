use super::super::plan_static_reuse::{
    plan_static_reuse_slot_sizes, with_plan_static_reference, SharedInvocationResource,
};
use super::*;

fn static_buffers(
    invocation: &BatchedOperationInvocation<'_, TestBuffer>,
) -> Vec<Vec<(ResourceId, BufferDescriptor, usize)>> {
    invocation
        .participants()
        .iter()
        .map(|participant| {
            participant
                .views()
                .iter()
                .filter(|view| view.storage_kind() == OperationBufferStorageKind::StaticContiguous)
                .map(|view| {
                    let regions = view.translate(0, view.descriptor().size_bytes).unwrap();
                    let region = regions.iter().next().unwrap();
                    let (buffer, _, _) = region.buffer_and_physical_range();
                    (
                        view.resource_id().clone(),
                        view.descriptor().clone(),
                        std::ptr::from_ref(buffer) as usize,
                    )
                })
                .collect()
        })
        .collect()
}

fn assert_static_and_dynamic_resources_present(
    invocation: &BatchedOperationInvocation<'_, TestBuffer>,
) {
    let static_rows = static_buffers(invocation);
    assert!(
        !static_rows[0].is_empty(),
        "fixture must contain real static slots"
    );
    assert!(static_rows.iter().all(|row| row == &static_rows[0]));
    assert!(invocation.participants()[0]
        .views()
        .iter()
        .any(|view| view.storage_kind() != OperationBufferStorageKind::StaticContiguous));
    if invocation.participants().len() > 1 {
        let first_shared = invocation.participants()[0]
            .views()
            .iter()
            .find_map(|view| view.shared_backing())
            .expect("both arms must retain the existing dynamic backing sharing");
        assert!(invocation.participants()[1]
            .views()
            .iter()
            .filter_map(|view| view.shared_backing())
            .any(|shared| Arc::ptr_eq(first_shared, shared)));
    }
}

#[test]
fn plan_static_reuse_complete_constructor_preserves_views_and_dynamic_sharing() {
    for width in [1, 8, 32] {
        with_live_wave(width, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = |reference| {
                with_plan_static_reference(reference, || {
                    BatchedOperationInvocation::from_resources(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        identity,
                        &node,
                        OperationInvocationResources::Wave {
                            wave,
                            node_index: 0,
                        },
                        active.iter(),
                        false,
                        true,
                    )
                    .unwrap()
                })
            };
            let reference = make(true);
            let reused = make(false);
            assert_static_and_dynamic_resources_present(&reference);
            assert_static_and_dynamic_resources_present(&reused);
            assert_eq!(snapshot(&reference), snapshot(&reused));
            assert_eq!(static_buffers(&reference), static_buffers(&reused));
            assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
            assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
        });
    }
}

#[test]
fn plan_static_reuse_distinct_plans_with_equal_descriptors_keep_their_owners() {
    with_live_wave(8, |first, first_wave, first_identity, first_active| {
        let first_provider = first
            .registry
            .bind(&first.resolved, first_wave.nodes()[0].node_id())
            .unwrap();
        let first_node = first_identity.materialize_node(0).unwrap();
        let first_invocation = with_plan_static_reference(false, || {
            BatchedOperationInvocation::from_resources(
                first.runtime.as_ref(),
                &first.resolved,
                first_provider.dispatch(),
                first_identity,
                &first_node,
                OperationInvocationResources::Wave {
                    wave: first_wave,
                    node_index: 0,
                },
                first_active.iter(),
                false,
                true,
            )
            .unwrap()
        });
        let first_rows = static_buffers(&first_invocation);
        with_live_wave(8, |second, wave, identity, active| {
            assert!(!Arc::ptr_eq(&first.plan_resources, &second.plan_resources));
            let provider = second
                .registry
                .bind(&second.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            for reference in [true, false] {
                let make = |bindings: &[TrustedActiveSequenceBinding]| {
                    with_plan_static_reference(reference, || {
                        BatchedOperationInvocation::from_resources(
                            second.runtime.as_ref(),
                            &second.resolved,
                            provider.dispatch(),
                            identity,
                            &node,
                            OperationInvocationResources::Wave {
                                wave,
                                node_index: 0,
                            },
                            bindings.iter(),
                            false,
                            true,
                        )
                    })
                };
                let current = make(active).unwrap();
                assert_static_and_dynamic_resources_present(&current);
                let rows = static_buffers(&current);
                assert_eq!(rows[0].len(), first_rows[0].len());
                for ((id, descriptor, buffer), (other_id, other_descriptor, other_buffer)) in
                    rows[0].iter().zip(&first_rows[0])
                {
                    assert_eq!((id, descriptor), (other_id, other_descriptor));
                    assert_ne!(
                        buffer, other_buffer,
                        "equal metadata must not substitute another Plan's allocation"
                    );
                }
                assert!(
                    make(first_active).is_err(),
                    "foreign active Plan authority must still be rejected"
                );
            }
        });
        // Both Plans were live throughout; no allocator-address reuse can explain the check.
        assert_eq!(static_buffers(&first_invocation), first_rows);
    });
}

#[test]
fn plan_static_reuse_proof_requires_exact_lease_and_allocation() {
    with_live_wave(1, |first, first_wave, _, _| {
        let first_participant = OperationInvocationResources::Wave {
            wave: first_wave,
            node_index: 0,
        }
        .participant(0)
        .unwrap();
        let first_lease = first_participant.static_provisioning().unwrap();
        let allocation = &first.plan.payload().memory().static_allocations()[0];
        let equal_allocation = allocation.clone();
        let (first_view, proof) = first_lease.checked_plan_static_view(0, allocation).unwrap();
        assert!(proof.reborrow(first_lease, allocation, 0).is_some());
        assert!(proof.reborrow(first_lease, &equal_allocation, 0).is_none());
        assert!(proof.reborrow(first_lease, allocation, 1).is_none());
        with_live_wave(1, |second, wave, _, _| {
            let participant = OperationInvocationResources::Wave {
                wave,
                node_index: 0,
            }
            .participant(0)
            .unwrap();
            let other_lease = participant.static_provisioning().unwrap();
            let other_allocation = &second.plan.payload().memory().static_allocations()[0];
            let other_view = other_lease.plan_static_view(0, other_allocation).unwrap();
            assert_eq!(
                first_view.committed_descriptor(),
                other_view.committed_descriptor()
            );
            assert!(!std::ptr::eq(first_view.buffer(), other_view.buffer()));
            let (_, local_proof) = first_lease.checked_plan_static_view(0, allocation).unwrap();
            assert!(local_proof.reborrow(other_lease, allocation, 0).is_none());

            // Reuse the actual constructor-local table slot across these operands:
            // a mismatch must take a new full lease check, never return the old owner.
            let mut slot = SharedInvocationResource::Vacant;
            slot.plan_static_view(first_lease, 0, allocation).unwrap();
            let current = slot
                .plan_static_view(other_lease, 0, other_allocation)
                .unwrap();
            assert!(std::ptr::eq(current.buffer(), other_view.buffer()));
            let wrong_slot = slot.plan_static_view(other_lease, 1, other_allocation);
            assert!(wrong_slot.is_err());
        });
    });
}

#[test]
fn plan_static_reuse_rejects_initial_and_later_static_getter_drift() {
    for (width, fault_at) in [(1, 1), (8, 1), (8, 2)] {
        for reference in [true, false] {
            with_live_wave(width, |fixture, wave, identity, active| {
                let provider = fixture
                    .registry
                    .bind(&fixture.resolved, wave.nodes()[0].node_id())
                    .unwrap();
                let node = identity.materialize_node(0).unwrap();
                let mut participant = 0;
                let bindings = active.iter().inspect(|_| {
                    participant += 1;
                    if participant == fault_at {
                        fixture
                            .runtime_trace
                            .lock()
                            .unwrap()
                            .tamper_weight_buffer_descriptor = true;
                    }
                });
                let result = with_plan_static_reference(reference, || {
                    BatchedOperationInvocation::from_resources(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        identity,
                        &node,
                        OperationInvocationResources::Wave {
                            wave,
                            node_index: 0,
                        },
                        bindings,
                        false,
                        true,
                    )
                });
                fixture
                    .runtime_trace
                    .lock()
                    .unwrap()
                    .tamper_weight_buffer_descriptor = false;
                let error = result
                    .err()
                    .expect("current static descriptor drift must be rejected");
                assert!(
                    error.to_string().contains("committed static resource"),
                    "{error}"
                );
                let repaired = with_plan_static_reference(reference, || {
                    BatchedOperationInvocation::from_resources(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        identity,
                        &node,
                        OperationInvocationResources::Wave {
                            wave,
                            node_index: 0,
                        },
                        active.iter(),
                        false,
                        true,
                    )
                    .unwrap()
                });
                assert_static_and_dynamic_resources_present(&repaired);
                assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
                assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
            });
        }
    }
}

#[test]
fn plan_static_reuse_does_not_retain_a_previous_constructor_validation() {
    with_live_wave(8, |fixture, wave, identity, active| {
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let make = || {
            with_plan_static_reference(false, || {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0,
                    },
                    active.iter(),
                    false,
                    true,
                )
            })
        };
        let before = make().unwrap();
        assert_static_and_dynamic_resources_present(&before);
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_weight_buffer_descriptor = true;
        let error = make()
            .err()
            .expect("a new constructor must read the current static descriptor");
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_weight_buffer_descriptor = false;
        assert!(
            error.to_string().contains("committed static resource"),
            "{error}"
        );
        assert_eq!(snapshot(&before), snapshot(&make().unwrap()));
    });
}

#[test]
#[ignore = "same-version complete-constructor static-reuse ablation; no speed threshold"]
fn paired_live_wave_plan_static_reuse() {
    for width in [1, 8, 32] {
        with_live_wave(width, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = |reference| {
                with_plan_static_reference(reference, || {
                    BatchedOperationInvocation::from_resources(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        identity,
                        &node,
                        OperationInvocationResources::Wave {
                            wave,
                            node_index: 0,
                        },
                        active.iter(),
                        false,
                        true,
                    )
                    .unwrap()
                })
            };
            assert_static_and_dynamic_resources_present(&make(true));
            assert_static_and_dynamic_resources_present(&make(false));
            let (original_slot_bytes, candidate_slot_bytes) =
                plan_static_reuse_slot_sizes::<TestRuntime>();
            println!(
                "{}",
                serde_json::json!({
                    "kind": "plan_static_reuse_ablation_scope",
                    "participants": width,
                    "original_dynamic_slot_bytes": original_slot_bytes,
                    "candidate_slot_bytes": candidate_slot_bytes,
                    "both_arms_dynamic_sharing": true,
                    "both_arms_candidate_layout": true,
                    "static_resources_per_participant": static_buffers(&make(false))[0].len(),
                })
            );
            // Both arms keep dynamic sharing and the new table layout. Net layout cost
            // requires the separately retained old binary's original shared arm.
            paired_invocation_elapsed("plan_static_reuse_same_version_ablation", width, make);
        });
    }
}
