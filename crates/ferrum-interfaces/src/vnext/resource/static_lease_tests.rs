use super::super::PlanRuntimeStatic;
use super::*;

#[path = "../../../tests/vnext_resource_contract/support.rs"]
mod support;

#[test]
fn plan_static_reuse_initial_generation_and_descriptor_must_validate_before_mint() {
    let plan = support::execution_plan();
    let (driver, _) = support::configured_driver(&plan, &[], &[]);
    let mut resources = support::plan_runtime(&plan, driver, "plan-static-proof");
    let allocation = &plan.payload().memory().static_allocations()[0];
    {
        let resources = Arc::get_mut(&mut resources).expect("fixture exclusively owns its Plan");
        let PlanRuntimeStatic::Static(static_resources) = &mut resources.static_resources else {
            panic!("fixture must own real static provisioning");
        };
        let lease = static_resources.lease.as_mut().unwrap();
        {
            let (_, proof) = lease.checked_plan_static_view(0, allocation).unwrap();
            assert!(proof.reborrow(lease, allocation, 0).is_some());
        }

        let original_generation = lease.slots[0].actual_generation;
        lease.slots[0].actual_generation =
            Some(original_generation.unwrap().checked_add(1).unwrap());
        let original_error = lease.plan_static_view(0, allocation).err().unwrap();
        let mint_error = lease.checked_plan_static_view(0, allocation).err().unwrap();
        assert_eq!(original_error.to_string(), mint_error.to_string());
        lease.slots[0].actual_generation = original_generation;

        let original_size = lease.slots[0].descriptor.as_ref().unwrap().size_bytes;
        lease.slots[0].descriptor.as_mut().unwrap().size_bytes =
            original_size.checked_add(1).unwrap();
        let original_error = lease.plan_static_view(0, allocation).err().unwrap();
        let mint_error = lease.checked_plan_static_view(0, allocation).err().unwrap();
        assert_eq!(original_error.to_string(), mint_error.to_string());
        lease.slots[0].descriptor.as_mut().unwrap().size_bytes = original_size;

        let (view, proof) = lease.checked_plan_static_view(0, allocation).unwrap();
        let reused = proof.reborrow(lease, allocation, 0).unwrap();
        assert_eq!(view.generation(), original_generation.unwrap());
        assert_eq!(view.committed_descriptor(), reused.committed_descriptor());
        assert!(std::ptr::eq(view.buffer(), reused.buffer()));
    }
    support::close_plan_runtime(resources);
}
