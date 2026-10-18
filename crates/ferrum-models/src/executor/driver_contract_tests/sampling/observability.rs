use super::*;
use ferrum_execution::test_support::observability;

#[tokio::test]
async fn observability_cpu_required_terminal_delivery_retains_actual_ended() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = observability::observability_required_terminal_delivery_retains_actual_ended(
        fixture.executor,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn observability_cpu_optional_transport_keeps_outputs_and_following_requests() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        observability::observability_optional_transport_preserves_actual_tokens(fixture.executor)
            .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn observability_cpu_contract_failure_keeps_actual_progress_without_replay() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = observability::observability_fatal_sink_after_terminal_preserves_progress(
        fixture.executor,
        ExecutionEventDeliveryPolicy::OptionalTelemetry,
        ExecutionEventSinkErrorKind::Contract,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn observability_cpu_required_transport_failure_keeps_actual_progress_without_replay() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = observability::observability_fatal_sink_after_terminal_preserves_progress(
        fixture.executor,
        ExecutionEventDeliveryPolicy::RequiredEvidence,
        ExecutionEventSinkErrorKind::Transport,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}
