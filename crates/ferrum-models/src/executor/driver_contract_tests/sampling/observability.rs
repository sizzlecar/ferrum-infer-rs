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

#[tokio::test]
async fn execution_facts_cpu_on_off_same_numeric_tokens_and_terminal_owner() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = ferrum_execution::test_support::execution_facts::execution_facts_on_off_preserve_actual_tokens(fixture.executor).await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_optional_transport_preserves_numeric_progress() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = ferrum_execution::test_support::execution_facts::execution_facts_optional_transport_preserves_actual_tokens(fixture.executor).await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_terminal_contract_no_replay() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_fatal_terminal_preserves_progress(
        fixture.executor,
        FactsFailure::Contract,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_terminal_required_transport_no_replay() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_fatal_terminal_preserves_progress(
        fixture.executor,
        FactsFailure::RequiredTransport,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_caller_drop_late_required_error_original_sink() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_cancelled_caller_preserves_original_sink(
        fixture.executor,
        FactsFailure::RequiredTransport,
        ExecutionFactsCapture::On,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_caller_drop_late_callback_panic_keeps_terminal() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_cancelled_caller_preserves_original_sink(
        fixture.executor,
        FactsFailure::CallbackPanic,
        ExecutionFactsCapture::On,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_caller_drop_late_policy_panic_keeps_terminal() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_cancelled_caller_preserves_original_sink(
        fixture.executor,
        FactsFailure::PolicyPanic,
        ExecutionFactsCapture::On,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_late_attach_does_not_retrofit_disabled_submission() {
    use ferrum_execution::test_support::execution_facts::{self, FactsFailure};
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = execution_facts::execution_facts_cancelled_caller_preserves_original_sink(
        fixture.executor,
        FactsFailure::None,
        ExecutionFactsCapture::Off,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn execution_facts_cpu_capture_query_panic_closes_prepared_original_owner() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let runtime = ferrum_execution::test_support::execution_facts::execution_facts_capture_query_panic_closes_original_owner(fixture.executor).await;
    assert_eq!(runtime.resident_bytes(), 0);
}
