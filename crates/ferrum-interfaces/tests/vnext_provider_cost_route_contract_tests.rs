mod vnext_device_operation_contract;
use ferrum_interfaces::execution_cost::*;
use std::num::NonZeroU64;
use vnext_device_operation_contract::*;

fn row(offset: u64, count: u64, total: u64) -> OperationCostWorkRow {
    OperationCostWorkRow {
        offset,
        count: NonZeroU64::new(count).unwrap(),
        full_input_tokens: NonZeroU64::new(total).unwrap(),
    }
}

#[test]
fn bound_route_query_projects_numeric_work_without_allocating_or_encoding() {
    let fixture = fixture_tokens();
    let provider = fixture
        .registry
        .bind(&fixture.resolved, &id("node.main"))
        .unwrap();
    let before = {
        let trace = fixture.runtime_trace.lock().unwrap();
        (
            trace.allocation_calls,
            trace.submit_calls,
            trace.readback_calls,
        )
    };
    let rows = [row(7, 1, 8), row(0, 16, 32)];
    assert!(provider
        .eager_cost_route(&fixture.resolved, &rows)
        .unwrap()
        .is_none());
    fixture.provider_trace.lock().unwrap().cost_route_supported = true;
    let route = provider
        .eager_cost_route(&fixture.resolved, &rows)
        .unwrap()
        .unwrap();
    assert_eq!(route.commands()[0].participant_count(), 2);
    assert_eq!(route.commands()[0].token_count(), 17);
    let advanced = [row(8, 1, 9), row(16, 16, 32)];
    provider
        .eager_cost_route(&fixture.resolved, &advanced)
        .unwrap()
        .unwrap();
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(
            trace.cost_route_queries,
            [rows.to_vec(), rows.to_vec(), advanced.to_vec()]
        );
        assert_eq!(trace.encode_calls, 0);
        // The compiled contiguous resource contract proves an aligned origin
        // without allocating backing or constructing a physical invocation.
        assert_eq!(trace.cost_route_input_alignment, [Some(16); 3]);
        assert_eq!(trace.reusable_binding_encode_calls, 0);
        assert_eq!(trace.reusable_topology_calls, 0);
    }
    let trace = fixture.runtime_trace.lock().unwrap();
    assert_eq!(
        (
            trace.allocation_calls,
            trace.submit_calls,
            trace.readback_calls
        ),
        before
    );
}

#[test]
fn invalid_projection_is_rejected_before_provider_and_invalid_route_after_provider() {
    let fixture = fixture_tokens();
    let provider = fixture
        .registry
        .bind(&fixture.resolved, &id("node.main"))
        .unwrap();
    assert!(provider
        .eager_cost_route(&fixture.resolved, &[row(7, 2, 8)])
        .is_err());
    assert!(fixture
        .provider_trace
        .lock()
        .unwrap()
        .cost_route_queries
        .is_empty());
    {
        let mut trace = fixture.provider_trace.lock().unwrap();
        trace.cost_route_supported = true;
        trace.cost_route_extra_participants = 1;
    }
    assert!(provider
        .eager_cost_route(&fixture.resolved, &[row(7, 1, 8)])
        .is_err());
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
}

fn finish(mut builder: CanonicalWaveCostBuilder) -> CanonicalWaveCostShape {
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 7 },
            host_policy_signature: [9; 32],
            host_features: None,
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0_f32.to_bits(),
            },
        })
        .unwrap();
    builder
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}

#[test]
fn selected_plan_route_uses_bound_provider_identity_and_preserves_core_prefix_offset() {
    let fixture = fixture_tokens();
    let selected = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let rows = [row(7, 1, 8)];
    assert!(selected
        .eager_cost_route(&fixture.resolved, &rows, &mut || Ok(()))
        .unwrap()
        .is_none());
    fixture.provider_trace.lock().unwrap().cost_route_supported = true;
    let route = selected
        .eager_cost_route(&fixture.resolved, &rows, &mut || Ok(()))
        .unwrap()
        .unwrap();
    assert_eq!(route.node_count(), selected.len());
    assert_eq!(route.physical_slots(), selected.len() as u32);
    let mut combined = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    assert_eq!(
        route.append_canonical(&mut combined, 3).unwrap(),
        3 + selected.len() as u32
    );
    let mut direct = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    for (node_index, provider) in selected.providers().iter().enumerate() {
        let descriptor = provider.descriptor();
        let identity = CostProviderIdentity {
            provider_id: descriptor.provider_id().as_str(),
            implementation_fingerprint: descriptor.provider_implementation_fingerprint(),
            operation_fingerprint: descriptor.operation_fingerprint(),
        };
        let per_node = provider
            .eager_cost_route(&fixture.resolved, &rows)
            .unwrap()
            .unwrap();
        direct
            .physical_command(
                per_node.commands()[0]
                    .canonical_command(3 + node_index as u32, node_index as u32, identity)
                    .unwrap(),
            )
            .unwrap();
    }
    assert_eq!(finish(combined), finish(direct));
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
}

#[test]
fn selected_plan_query_checks_identity_and_budget_without_publishing_a_partial_route() {
    let fixture = fixture_tokens();
    let selected = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    fixture.provider_trace.lock().unwrap().cost_route_supported = true;
    let rows = [row(7, 1, 8)];
    let other = fixture_with_zero_state(true);
    assert!(selected
        .eager_cost_route(&other.resolved, &rows, &mut || Ok(()))
        .is_err());
    assert!(fixture
        .provider_trace
        .lock()
        .unwrap()
        .cost_route_queries
        .is_empty());
    // Budget expiry immediately after a provider returns must also discard it.
    assert!(selected
        .eager_cost_route(&fixture.resolved, &rows, &mut || {
            if !fixture
                .provider_trace
                .lock()
                .unwrap()
                .cost_route_queries
                .is_empty()
            {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "test query budget exhausted".into(),
                })
            } else {
                Ok(())
            }
        })
        .is_err());
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .cost_route_queries
            .len(),
        1
    );
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
}
