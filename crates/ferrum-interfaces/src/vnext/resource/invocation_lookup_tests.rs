//! Shared-backing lookup assertions live inside resource ownership so private
//! pool types never escape into operation tests.
use super::*;
#[path = "../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use vnext_device_operation_contract::{fixture_with_device_id, id};
use vnext_device_operation_wave_contract::{prepare_wave, setup_with_fixture, teardown};

fn fixture() -> vnext_device_operation_contract::Fixture {
    fixture_with_device_id(id("device.certified-shared-backing-lookup"))
}

#[test]
fn fresh_certified_shared_backing_lookup_preserves_wave_step_and_invocation_authorities() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    let missing: ResourceId = id("resource.absent.certified-lookup");
    {
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        // Each lookup returns the original authority retained by this wave or
        // its parent Step, rather than a cached validation or cloned claim.
        for authority in wave.claimed_backing().backing_slices() {
            for node_index in 0..wave.node_count() {
                let (_, actual) = wave
                    .backing_source(node_index, authority.resource_id())
                    .unwrap();
                assert_eq!(actual.len(), 1);
                assert!(std::ptr::eq(&actual[0], authority));
            }
        }
        for authority in step.backing_slices() {
            let (_, actual) = step.backing_source(authority.resource_id()).unwrap();
            assert_eq!(actual.len(), 1);
            assert!(std::ptr::eq(&actual[0], authority));
            let (_, inherited) = wave.backing_source(0, authority.resource_id()).unwrap();
            assert!(std::ptr::eq(&inherited[0], authority));
        }
        assert!(wave.backing_source(0, &missing).is_err());
        assert!(wave.backing_source(wave.node_count(), &missing).is_err());
        assert!(step.backing_source(&missing).is_err());
    }
    teardown(fixture, sequence, session, batch, step);

    // A dropped prepared wave retires its topology. Check the separate
    // invocation path with fresh ownership rather than reusing that step.
    let (fixture, sequence, session, batch, step) = setup_with_fixture(self::fixture());
    {
        let node = &fixture.plan.payload().nodes()[0];
        let invocation = vnext_device_operation_contract::admit_single_participant_invocation(
            &fixture.plan_resources,
            &step,
            node.id(),
        );
        for authority in invocation.backing_slices() {
            let (_, actual) = invocation.backing_source(authority.resource_id()).unwrap();
            assert_eq!(actual.len(), 1);
            assert!(std::ptr::eq(&actual[0], authority));
        }
        for authority in step.backing_slices() {
            let (_, inherited) = invocation.backing_source(authority.resource_id()).unwrap();
            assert!(std::ptr::eq(&inherited[0], authority));
        }
        assert!(invocation.backing_source(&missing).is_err());
    }
    teardown(fixture, sequence, session, batch, step);
}
