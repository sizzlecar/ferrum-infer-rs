use super::*;

#[tokio::test]
async fn native_resident_capacity_requires_complete_live_incarnations() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let other = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_resident_capacity_requires_complete_live_incarnations(
        &fixture.executor,
        &other.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_partial_final_decode_and_terminal_receipt() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_partial_final_decode_and_terminal_receipt(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_foreign_stale_and_admission_mutation_rejected() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let other = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_foreign_stale_and_admission_mutation_rejected(
        &fixture.executor,
        &other.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_mixed_preserves_original_order_and_frontiers() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_mixed_preserves_original_order_and_frontiers(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_bad_middle_product_preserves_both_successful_peers() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_bad_middle_product_preserves_both_successful_peers(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_extra_output_row_retains_unassigned_products_and_closes_every_owner() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_extra_output_row_retains_unassigned_products_and_closes_every_owner(&fixture.executor).await;
}

#[tokio::test]
async fn native_tokens_cpu_cancel_queued_wave_waits_for_actual_terminal_receipt() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_cancel_queued_wave_waits_for_actual_terminal_receipt(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_cancel_one_shared_participant_preserves_healthy_peer() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_cancel_one_shared_participant_preserves_healthy_peer(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn native_tokens_cpu_deferral_retry_checks_decode_frontier_and_cancellation() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::native_tokens_cpu_deferral_retry_checks_decode_frontier_and_cancellation(
        &fixture.executor,
    )
    .await;
}

#[tokio::test]
async fn mixed_frontiers_reject_the_same_incarnation_in_both_phases_before_state_changes() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::mixed_frontiers_reject_the_same_incarnation_in_both_phases_before_state_changes(&fixture.executor).await;
}

#[tokio::test]
async fn mixed_frontiers_reject_duplicate_decodes_and_duplicate_prefills() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    token_assertions::mixed_frontiers_reject_duplicate_decodes_and_duplicate_prefills(
        &fixture.executor,
    )
    .await;
}
