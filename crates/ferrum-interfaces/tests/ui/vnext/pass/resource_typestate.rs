use ferrum_interfaces::vnext::*;

fn assert_send_static<T: Send + 'static>() {}
fn assert_sync<T: Sync>() {}

fn owning_resource_chain_is_send_static_and_arc_targets_are_sync<R: DeviceRuntime>() {
    assert_send_static::<PlanRuntimeResources<R>>();
    assert_sync::<PlanRuntimeResources<R>>();
    assert_send_static::<TrustedPlanRuntimeBinding<R>>();
    assert_sync::<TrustedPlanRuntimeBinding<R>>();
    assert_send_static::<AdmittedRequestResources<R>>();
    assert_sync::<AdmittedRequestResources<R>>();
    assert_send_static::<AdmittedSequenceResources<R>>();
    assert_sync::<AdmittedSequenceResources<R>>();
    assert_send_static::<SequenceSession<R>>();
    assert_sync::<SequenceSession<R>>();
    assert_send_static::<ExecutionBatchParticipants<R>>();
    assert_sync::<ExecutionBatchParticipants<R>>();
    assert_send_static::<StepResourceLease<R>>();
    assert_sync::<StepResourceLease<R>>();
    assert_send_static::<InvocationResourceLease<R>>();
}

fn legal_path<D: ResourceTransactionDriver>(
    driver: D,
    identity: ResourceTransactionIdentity,
    permit: StaticProvisioningPermit<D::Runtime>,
) {
    let transaction = ResourceTransaction::begin(driver, identity, permit).unwrap();
    let reserved = transaction.reserve().unwrap();
    let committed = match reserved.commit() {
        Ok(committed) => committed,
        Err(_) => panic!("fixture commit failed"),
    };
    let _released = committed.release().unwrap();
}

fn main() {}
