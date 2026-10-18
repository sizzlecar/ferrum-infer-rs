use super::super::descriptor_agreement::{with_reference_comparison, DeviceDescriptorAgreement};
use super::*;

#[derive(Clone, Copy, Debug)]
enum Change {
    EqualClone,
    Capacity,
    Capabilities,
}

impl Change {
    fn apply(self, descriptor: &mut DeviceDescriptor) {
        match self {
            Self::EqualClone => {}
            Self::Capacity => descriptor.total_memory_bytes += 1,
            Self::Capabilities => {
                descriptor.capabilities.insert(id("capability.alternate"));
            }
        }
    }

    fn equal(self) -> bool {
        matches!(self, Self::EqualClone)
    }
}

#[test]
fn descriptor_agreement_checks_both_current_operands() {
    let original = catalog().device().clone();
    let equal = original.clone();
    for reference in [true, false] {
        with_reference_comparison(reference, || {
            for change in [Change::EqualClone, Change::Capacity, Change::Capabilities] {
                let mut changed = original.clone();
                change.apply(&mut changed);
                assert_eq!(
                    changed.runtime_implementation_fingerprint,
                    original.runtime_implementation_fingerprint
                );
                assert_eq!(changed.id, original.id);
                let mut proof = DeviceDescriptorAgreement::default();
                assert!(proof.matches(&original, &original));
                assert!(proof.matches(&original, &original));
                assert!(proof.matches(&original, &equal));
                assert_eq!(proof.matches(&changed, &equal), change.equal());
                assert_eq!(proof.matches(&original, &changed), change.equal());
                assert!(proof.matches(&equal, &original));
                // The two production comparisons keep independent operands.
                let mut second = DeviceDescriptorAgreement::default();
                assert_eq!(second.matches(&changed, &original), change.equal());
            }
        });
    }
}

#[test]
fn descriptor_agreement_rechecks_runtime_metadata_on_later_participants() {
    for reference in [true, false] {
        with_reference_comparison(reference, || {
            for change in [Change::EqualClone, Change::Capacity, Change::Capabilities] {
                let fixture = fixture_with_runtime_configuration(|runtime| {
                    runtime.alternate_descriptor = runtime.descriptor.clone();
                    change.apply(&mut runtime.alternate_descriptor);
                });
                with_live_wave_fixture(fixture, vec![1; 8], |fixture, wave, identity, active| {
                    let provider = fixture
                        .registry
                        .bind(&fixture.resolved, wave.nodes()[0].node_id())
                        .unwrap();
                    let node = identity.materialize_node(0).unwrap();
                    let make = |switch| {
                        let mut participant = 0;
                        let current = active.iter().inspect(|_| {
                            participant += 1;
                            if switch && participant == 2 {
                                fixture
                                    .runtime
                                    .use_alternate_descriptor
                                    .store(true, Ordering::Release);
                            }
                        });
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
                            current,
                            false,
                            true,
                        )
                    };
                    let expected = snapshot(&make(false).unwrap());
                    let changed = make(true);
                    assert_eq!(changed.is_ok(), change.equal(), "{change:?}");
                    if let Ok(invocation) = changed {
                        assert_eq!(snapshot(&invocation), expected);
                    }
                    fixture
                        .runtime
                        .use_alternate_descriptor
                        .store(false, Ordering::Release);
                    // A fresh constructor on the same live wave starts anew.
                    assert_eq!(snapshot(&make(false).unwrap()), expected);
                    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
                    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
                });
            }
        });
    }
}

struct SwitchingPlan<'a> {
    original: &'a ResolvedModelPlan,
    alternate_device: DeviceDescriptor,
    alternate_catalog: CapabilityCatalog,
    switch: std::cell::Cell<bool>,
    catalog_operand: bool,
}

#[test]
fn descriptor_agreement_rechecks_runtime_between_the_two_comparisons() {
    for reference in [true, false] {
        with_reference_comparison(reference, || {
            for change in [Change::EqualClone, Change::Capacity, Change::Capabilities] {
                for comparison in [1, 2] {
                    let fixture = fixture_with_runtime_configuration(|runtime| {
                        runtime.alternate_descriptor = runtime.descriptor.clone();
                        change.apply(&mut runtime.alternate_descriptor);
                    });
                    with_live_wave_fixture(
                        fixture,
                        vec![1; 2],
                        |fixture, wave, identity, active| {
                            let provider = fixture
                                .registry
                                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                                .unwrap();
                            let node = identity.materialize_node(0).unwrap();
                            let mut participant = 0;
                            let current = active.iter().inspect(|_| {
                                participant += 1;
                                if participant == 2 {
                                    // Participant one populated both proofs. Arm
                                    // drift at either of participant two's first
                                    // two descriptor getters, in their original order.
                                    fixture
                                        .runtime
                                        .descriptor_reads_until_drift
                                        .store(comparison, Ordering::Release);
                                }
                            });
                            let result = BatchedOperationInvocation::from_resources(
                                fixture.runtime.as_ref(),
                                &fixture.resolved,
                                provider.dispatch(),
                                identity,
                                &node,
                                OperationInvocationResources::Wave {
                                    wave,
                                    node_index: 0,
                                },
                                current,
                                false,
                                true,
                            );
                            assert_eq!(
                                result.is_ok(),
                                change.equal(),
                                "{change:?}, comparison={comparison}"
                            );
                            fixture
                                .runtime
                                .use_alternate_descriptor
                                .store(false, Ordering::Release);
                            fixture
                                .runtime
                                .descriptor_reads_until_drift
                                .store(0, Ordering::Release);
                            assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
                            assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
                        },
                    );
                }
            }
        });
    }
}

impl ExecutablePlanView for SwitchingPlan<'_> {
    fn execution_plan(&self) -> &ExecutionPlan {
        self.original.execution_plan()
    }

    fn device(&self) -> &DeviceDescriptor {
        if self.switch.get() && !self.catalog_operand {
            &self.alternate_device
        } else {
            ExecutablePlanView::device(self.original)
        }
    }

    fn capabilities(&self) -> &CapabilityCatalog {
        if self.switch.get() && self.catalog_operand {
            &self.alternate_catalog
        } else {
            self.original.capabilities()
        }
    }
}

#[test]
fn descriptor_agreement_rechecks_each_plan_operand_on_later_participants() {
    for reference in [true, false] {
        with_reference_comparison(reference, || {
            for catalog_operand in [false, true] {
                for change in [Change::EqualClone, Change::Capacity, Change::Capabilities] {
                    with_live_wave(8, |fixture, wave, identity, active| {
                        let provider = fixture
                            .registry
                            .bind(&fixture.resolved, wave.nodes()[0].node_id())
                            .unwrap();
                        let node = identity.materialize_node(0).unwrap();
                        let catalog = fixture.resolved.capabilities();
                        let mut alternate_device = catalog.device().clone();
                        change.apply(&mut alternate_device);
                        let alternate_catalog = CapabilityCatalog::new(
                            alternate_device.clone(),
                            catalog.operations().values().cloned().collect(),
                            catalog.providers().clone(),
                            catalog.engine_providers().values().cloned().collect(),
                        )
                        .unwrap();
                        let resolved = SwitchingPlan {
                            original: &fixture.resolved,
                            alternate_device,
                            alternate_catalog,
                            switch: std::cell::Cell::new(false),
                            catalog_operand,
                        };
                        let mut participant = 0;
                        let current = active.iter().inspect(|_| {
                            participant += 1;
                            if participant == 2 {
                                resolved.switch.set(true);
                            }
                        });
                        let result = BatchedOperationInvocation::from_resources(
                            fixture.runtime.as_ref(),
                            &resolved,
                            provider.dispatch(),
                            identity,
                            &node,
                            OperationInvocationResources::Wave {
                                wave,
                                node_index: 0,
                            },
                            current,
                            false,
                            true,
                        );
                        assert_eq!(
                            result.is_ok(),
                            change.equal(),
                            "{change:?}, catalog={catalog_operand}"
                        );
                        assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
                        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
                    });
                }
            }
        });
    }
}

#[test]
#[ignore = "paired CPU elapsed diagnostic, no speed threshold"]
fn paired_live_wave_descriptor_agreement() {
    for width in [1, 8, 32] {
        with_live_wave(width, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = |reference| {
                with_reference_comparison(reference, || {
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
            paired_invocation_elapsed("descriptor_agreement_pair", width, make);
        });
    }
}
