//! Real Stored Reference installations under a native grant that cannot hold
//! both owners. These are generic retirement tests, not CPU H3/causal inference.
use super::*;
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceVNextComposition,
};
use std::sync::atomic::AtomicU64;

fn bounded() -> (Execution, QwenImage21TransformerConfig, tempfile::TempDir) {
    let (mut original, model, directory, _) = fixture();
    let first = prepared(
        &original,
        &model,
        ImageStage::Denoiser,
        ImageStageAttachment::InitializeShared,
    );
    let memory = first.executable().execution_plan().payload().memory();
    let peak = memory.static_bytes() + memory.minimum_runnable_request_bytes();
    let grant = peak.div_ceil(256) * 256;
    assert!(memory.static_bytes() + peak > grant);
    let mut config = original.config.clone();
    config.device_capacity_bytes = grant;
    let sources = original.sources.clone();
    let host = original.host.clone();
    original
        .shutdown()
        .unwrap_or_else(|e| panic!("empty original owner: {e}"));
    drop(original);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new(format!(
            "device.real.auxiliary.suspension.{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap(),
        grant,
    )
    .unwrap();
    let execution = Execution::new(
        composition.runtime().clone(),
        host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        config,
        sources,
    )
    .unwrap();
    (execution, model, directory)
}

#[test]
fn suspended_auxiliary_real_native_retirement_reinstallation_preserves_input_and_output() {
    let (mut execution, config, _source) = bounded();
    let scope = execution
        .host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64 << 10),
            (HostMemoryClass::Output, 64 << 10),
            (HostMemoryClass::TransferStaging, 64 << 10),
            (HostMemoryClass::WeightBacking, 0),
        ])
        .unwrap();
    execution.execution_host = Some(scope.account());
    let x = inputs(&scope.account(), &config);
    let original_inputs = x
        .iter()
        .map(|x| x.bytes().bytes().to_vec())
        .collect::<Vec<_>>();
    let first = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::InitializeShared,
    );
    let auxiliary = prepared(
        &execution,
        &config,
        ImageStage::TextConnector,
        ImageStageAttachment::MayInitializeStored,
    );
    let auxiliary_peak = auxiliary.plan.static_bytes + auxiliary.plan.minimum_step_bytes;
    let mut receipts = vec![];
    let (before, _) = run(&mut execution, first.clone(), 10, x.clone(), &mut receipts);
    let pin = execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    let previous = pin.weights().evidence().clone();
    assert!(!execution
        .admit_auxiliary_peak(auxiliary_peak, &mut receipts)
        .unwrap_or_else(|e| panic!("pinned admission: {e}")));
    let host_credit = execution.host.usage().unwrap().committed_bytes();
    drop(pin);
    assert!(execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut receipts)
        .unwrap_or_else(|e| panic!("real retirement: {e}")));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(
        execution.host.usage().unwrap().committed_bytes(),
        host_credit
    );
    assert!(execution
        .admit_auxiliary_peak(auxiliary_peak, &mut receipts)
        .unwrap_or_else(|e| panic!("retired admission: {e}")));
    let (auxiliary_output, _) = run(&mut execution, auxiliary, 11, x.clone(), &mut receipts);
    assert_eq!(auxiliary_output, before);
    assert!(execution
        .release_weight_owner(ImageWeightOwnerKey::TextConnector, &mut receipts)
        .unwrap_or_else(|e| panic!("auxiliary retirement: {e}")));
    let (after, _) = run(&mut execution, first, 12, x.clone(), &mut receipts);
    let pin = execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    assert_ne!(
        previous.snapshot_fingerprint(),
        pin.weights().evidence().snapshot_fingerprint()
    );
    assert_eq!(
        previous.weight_plan_fingerprint(),
        pin.weights().evidence().weight_plan_fingerprint()
    );
    assert_eq!(
        after, before,
        "a different native transaction executes the same source math"
    );
    for (actual, expected) in x.iter().zip(original_inputs) {
        assert_eq!(actual.bytes().bytes(), expected);
    }
    drop(pin);
    drop(x);
    execution.execution_host = None;
    drop(scope);
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("final close: {e}"));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn suspended_auxiliary_foreign_pin_refuses_retirement_without_releasing_original_grant() {
    let (mut execution, config, _source) = bounded();
    let scope = execution
        .host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64 << 10),
            (HostMemoryClass::Output, 64 << 10),
            (HostMemoryClass::TransferStaging, 64 << 10),
            (HostMemoryClass::WeightBacking, 0),
        ])
        .unwrap();
    execution.execution_host = Some(scope.account());
    let first = prepared(
        &execution,
        &config,
        ImageStage::Denoiser,
        ImageStageAttachment::InitializeShared,
    );
    let x = inputs(&scope.account(), &config);
    let (before, _) = run(&mut execution, first.clone(), 20, x.clone(), &mut vec![]);
    let foreign_pin = execution
        .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        .unwrap();
    let original = foreign_pin.weights().evidence().clone();
    let committed = execution.host.usage().unwrap().committed_bytes();
    let allocated = execution.runtime.snapshot().live_allocations;
    assert!(!execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut vec![])
        .unwrap_or_else(|e| panic!("refused close: {e}")));
    assert_eq!(execution.runtime.snapshot().live_allocations, allocated);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), committed);
    let (after, _) = run(&mut execution, first, 21, x.clone(), &mut vec![]);
    assert_eq!(before, after);
    assert_eq!(foreign_pin.weights().evidence(), &original);
    drop((foreign_pin, x));
    execution.execution_host = None;
    drop(scope);
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("close: {e}"));
    assert_eq!(execution.runtime.snapshot().live_allocations, 0);
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
}
