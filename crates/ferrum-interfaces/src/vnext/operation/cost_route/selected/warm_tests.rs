use super::*;
use crate::execution_cost::*;
use crate::vnext::*;

fn identity(n: usize) -> CostProviderIdentity<'static> {
    CostProviderIdentity {
        provider_id: ["first", "second"][n],
        implementation_fingerprint:
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        operation_fingerprint: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    }
}
fn operation(n: usize, phase: DeviceCommandPhase) -> OperationCostCommand {
    let (name, compute, transfer) = match phase {
        DeviceCommandPhase::DynamicBinding => (["dynamic.first", "dynamic.second"][n], 0, 1),
        DeviceCommandPhase::Compute => (["compute.first", "compute.second"][n], 1, 0),
        DeviceCommandPhase::ResultBinding => (["result.first", "result.second"][n], 0, 1),
        _ => unreachable!(),
    };
    OperationCostCommand::new(
        name,
        phase,
        DeviceBatchingForm::Packed,
        0,
        2,
        2,
        compute,
        transfer,
    )
    .unwrap()
}
fn route() -> SelectedEagerCostRoute<'static> {
    SelectedEagerCostRoute {
        nodes: (0..2)
            .map(|n| NodeRoute {
                identity: identity(n),
                route: OperationCostRoute::new(vec![
                    operation(n, DeviceCommandPhase::DynamicBinding),
                    operation(n, DeviceCommandPhase::Compute),
                    operation(n, DeviceCommandPhase::ResultBinding),
                ])
                .unwrap(),
            })
            .collect(),
        physical_slots: 6,
    }
}
fn catalog(changed: bool, uploaded: bool) -> DeviceCostGraphCatalog {
    catalog_bound(changed, uploaded, false)
}
fn catalog_bound(
    changed: bool,
    uploaded: bool,
    structured_capture: bool,
) -> DeviceCostGraphCatalog {
    let lane = ExecutionLaneId::mint().unwrap();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("warm.test").unwrap(),
        ReusableExecutionCapacity::new(2, 2, 1).unwrap(),
    )
    .unwrap();
    let id = DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        "b".repeat(64),
        lane,
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        7,
        2,
        2,
        0,
    )
    .unwrap();
    let capture = DeviceReusableExecutionCapture::new(id, 2, vec![], vec![0, 1]).unwrap();
    let program = DeviceReusableExecutionProgram::new(
        &capture,
        vec![DeviceReusableExecutionSegment::new(0, 0, 2, 2).unwrap()],
        vec![0, 1],
        vec![],
    )
    .unwrap();
    let mut builder = DeviceCostGraphCatalogBuilder::new(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 1, 1, 0).unwrap(),
        DeviceCostGraphCatalogLimits::new(2, 8, 8).unwrap(),
    )
    .unwrap();
    builder.push_program(&program, &mut || Ok(())).unwrap();
    if uploaded {
        let rows = (0..2)
            .map(|n| {
                DeviceReplayedLogicalCommandAttribution::new(
                    n as u32,
                    n as u32,
                    DeviceNativeOperationId::new(
                        operation(n, DeviceCommandPhase::Compute).native_operation(),
                    )
                    .unwrap(),
                    DeviceBatchingForm::Packed,
                    2,
                    if changed { 3 } else { 2 },
                    1,
                    0,
                    if structured_capture { 1 } else { 2 + n as u64 },
                )
                .unwrap()
                .with_captured_replay_template(if structured_capture {
                    Some(
                        SelectedReplayAlgorithmTemplateV1::from_selected(
                            &compute_evidence(n, 7, 32),
                            2,
                            1,
                            0,
                        )
                        .unwrap(),
                    )
                } else {
                    None
                })
            })
            .collect::<Vec<_>>();
        builder
            .push_uploaded_segment(&program.segments()[0], &"e".repeat(64), &rows, &mut || {
                Ok(())
            })
            .unwrap();
    }
    builder.finish(&mut || Ok(())).unwrap()
}
fn finish(mut builder: CanonicalWaveCostBuilder) -> CanonicalWaveCostShape {
    for _ in 0..2 {
        builder
            .row(CanonicalCostRow {
                work: ActualRowWork::Decode { kv_tokens: 1 },
                host_policy_signature: [3; 32],
                host_features: None,
                mask_upload_required: false,
                output: CostRowOutput::Decode {
                    requires_full_logits: false,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1.0_f32.to_bits(),
                },
            })
            .unwrap();
    }
    builder
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Warm,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}
#[test]
fn future_warm_segment_preserves_actual_grouped_binding_order_and_start_node_attribution() {
    let catalog = catalog(false, true);
    let graph = &catalog.programs()[0];
    let mut predicted = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let selected = route();
    let (end, indices) = selected
        .append_canonical_warm_program(
            &mut predicted,
            4,
            &[],
            None,
            graph,
            "test.replay",
            2,
            2,
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(end, 9);
    selected
        .append_warm_logical(&mut predicted, graph, &indices, &mut || Ok(()))
        .unwrap();
    // Core emits bindings for the complete segment around one replay and
    // attributes all three phases to its first plan node, not the leaf node.
    let mut actual = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    for (index, leaf, phase) in [
        (4, 0, DeviceCommandPhase::DynamicBinding),
        (5, 1, DeviceCommandPhase::DynamicBinding),
    ] {
        actual
            .physical_command(
                operation(leaf, phase)
                    .canonical_command(index, 0, identity(0))
                    .unwrap(),
            )
            .unwrap();
    }
    actual
        .physical_command(CostPhysicalCommand {
            native_op_id: "test.replay",
            command_index: 6,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(identity(0)),
            path: CostCommandPath::Replayed,
            participant_start: 0,
            participant_count: 2,
            token_count: 2,
            batching_form: DeviceBatchingForm::ParticipantLoop.as_str(),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: Some(5),
            statistical_evidence: None,
        })
        .unwrap();
    for (index, leaf) in [(7, 0), (8, 1)] {
        actual
            .physical_command(
                operation(leaf, DeviceCommandPhase::ResultBinding)
                    .canonical_command(index, 0, identity(0))
                    .unwrap(),
            )
            .unwrap();
    }
    let uploaded = &graph.uploaded_segments()[0];
    actual
        .replay_segment(6, uploaded.reusable_executable_fingerprint(), 2)
        .unwrap();
    for row in uploaded.logical_commands() {
        actual
            .logical_command(CostLogicalCommand::from_attribution(
                row,
                identity(row.node_index() as usize),
            ))
            .unwrap();
    }
    assert_eq!(finish(predicted), finish(actual));
}
#[test]
fn future_warm_rejects_same_catalog_counts_with_different_logical_work_or_missing_upload() {
    for graph in [catalog(true, true), catalog(false, false)] {
        let mut canonical = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        assert!(route()
            .append_canonical_warm_program(
                &mut canonical,
                0,
                &[],
                None,
                &graph.programs()[0],
                "test.replay",
                2,
                2,
                &mut || Ok(())
            )
            .is_err());
    }
}
#[test]
fn future_warm_projection_respects_poll_budget_and_never_fills_missing_evidence() {
    let graph = catalog(false, true);
    let mut canonical = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut calls = 0;
    let result = route().append_canonical_warm_program(
        &mut canonical,
        0,
        &[],
        None,
        &graph.programs()[0],
        "test.replay",
        2,
        2,
        &mut || {
            calls += 1;
            if calls == 2 {
                Err(invalid_operation("budget"))
            } else {
                Ok(())
            }
        },
    );
    assert!(result.is_err());
    assert_eq!(calls, 2);
}

fn row() -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 8 },
        host_policy_signature: [3; 32],
        mask_upload_required: false,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 2,
            repetition_penalty_bits: 1f32.to_bits(),
        },
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [4; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 8,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 2,
                maximum_output_tokens: 10,
                sampling_history_tokens: 2,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
    }
}

fn compute_evidence(n: usize, context: u64, block: u32) -> SelectedCommandCostEvidenceV1 {
    let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(2);
    b.kernel_with_replay_geometry(
        SelectedAlgorithmClassV1::new(["first.kernel", "second.kernel"][n], 1, [1; 32], [2; 32])
            .unwrap(),
        KernelNumericWorkV1 {
            logical_units: 2,
            padded_units: 32,
            inner_units_per_logical_unit: context,
            grid: [1, 1, 1],
            scratch_bytes: 0,
            staged_weight_bytes: 0,
        },
        KernelReplayGeometryV1 {
            block: [block, 1, 1],
            dynamic_shared_bytes: 0,
            fixed_parameters: &[2, 32],
        },
    )
    .unwrap();
    b.finish().unwrap()
}
fn selected_with_current_work(context: u64, block: u32) -> SelectedEagerCostRoute<'static> {
    let mut selected = route();
    for (n, node) in selected.nodes.iter_mut().enumerate() {
        for command in &mut node.route.commands {
            let evidence = if command.phase() == DeviceCommandPhase::Compute {
                compute_evidence(n, context, block)
            } else {
                let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(2);
                b.transfer(
                    SelectedAlgorithmClassV1::new("copy", 1, [1; 32], [2; 32]).unwrap(),
                    StatisticalTransferKindV1::DeviceToDevice,
                    16,
                )
                .unwrap();
                b.finish().unwrap()
            };
            *command = command.clone().with_statistical_evidence(evidence).unwrap();
        }
    }
    selected
}
fn structured(mut builder: CanonicalWaveCostBuilder) -> CanonicalStructuredWave {
    for _ in 0..2 {
        builder.row(row()).unwrap();
    }
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    builder
        .finish_with_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Warm,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}
#[test]
fn future_warm_uses_current_work_and_refuses_changed_resident_launch_geometry() {
    let catalog = catalog_bound(false, true, true);
    let graph = &catalog.programs()[0];
    for (block, known) in [(32, true), (64, false)] {
        let current = selected_with_current_work(31, block);
        let mut builder = CanonicalWaveCostBuilder::new_with_structured_statistics(
            0,
            CostProductOutput::GreedyToken,
        );
        let (_, indices) = current
            .append_canonical_warm_program(
                &mut builder,
                0,
                &[],
                None,
                graph,
                "test.replay",
                2,
                2,
                &mut || Ok(()),
            )
            .unwrap();
        current
            .append_warm_logical(&mut builder, graph, &indices, &mut || Ok(()))
            .unwrap();
        let result = structured(builder);
        if known {
            let recipe = result.structured.unwrap();
            let work = recipe.algorithm_work().unwrap();
            assert_eq!(work.aggregate_work().inner_work_units, 2 * 2 * 31);
            assert_eq!(work.aggregate_work().device_to_device_bytes, 4 * 16);
            assert_eq!(work.selected_command_count(), 6); // 2 logical kernels plus actual 4 transfers.
            assert_eq!(
                recipe.device().replay_work().unwrap().native_graph_nodes(),
                2
            );
            assert_eq!(recipe.device().replay_work().unwrap().logical_commands(), 2);
        } else {
            // Original exact routing still works; the passive numerical sidecar
            // cannot attribute a new eager grid/block to an old resident graph.
            assert!(result.structured.is_err());
            assert!(result.statistical.is_err());
        }
    }
}
