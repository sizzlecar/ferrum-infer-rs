use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

#[test]
fn retained_trace_capacity_proof_counts_physical_backing_without_double_reserving() {
    let (mut execution, _, _directory, _) = fixture();
    // This fresh execution has no work or backing. Declare its original Host
    // before installing the collector or reserving any request capacity.
    assert_eq!(execution.host.usage().unwrap().committed_bytes(), 0);
    let alignment = execution.host.allocation_alignment();
    let physical = execution.host.backing_bytes(1).unwrap();
    execution.host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            3 * physical,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2 * physical,
                output_bytes: 2 * physical,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap();
    let host = execution.host.clone();
    let otherwise_fits = [(HostMemoryClass::Output, 2 * physical)];
    execution
        .check_scope_capacity_with_retained_diagnostics(&otherwise_fits)
        .unwrap();
    let (trace, handle) =
        ImageContinuousTraceRecorder::new(&host, ImageContinuousTraceLimits { maximum_bytes: 1 })
            .unwrap();
    execution.continuous_trace = Some(trace);
    assert_eq!(execution.retained_diagnostic_output_bytes(), physical);
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);

    let class_error = execution
        .check_scope_capacity_with_retained_diagnostics(&[(HostMemoryClass::Output, physical + 1)])
        .unwrap_err();
    assert!(!class_error.is_temporary_capacity());
    let total_error = execution
        .check_scope_capacity_with_retained_diagnostics(&[
            (HostMemoryClass::Output, physical),
            (HostMemoryClass::InputPreprocess, physical + 1),
        ])
        .unwrap_err();
    assert!(!total_error.is_temporary_capacity());
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);

    let exact = [
        (HostMemoryClass::Output, physical),
        (HostMemoryClass::InputPreprocess, physical),
    ];
    execution
        .check_scope_capacity_with_retained_diagnostics(&exact)
        .unwrap();
    // The fixed backing is already charged: actual admission and reservation
    // use only the original new-work demand, so this exact boundary succeeds.
    host.check_scope_admission(&exact).unwrap();
    let scope = host.reserve_scope(&exact).unwrap();
    let output = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 1)
        .unwrap();
    let input = scope
        .account()
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 3 * physical);
    drop((output, input, scope));
    execution
        .shutdown()
        .unwrap_or_else(|e| panic!("shutdown: {e}"));
    assert!(handle.status().sealed);
    assert_eq!(execution.retained_diagnostic_output_bytes(), physical);
    drop(execution);
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    drop(handle);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
