//! Complete eager core + selected-provider route versus real native submission.
//! Numeric host fields are a fixed contract fixture; device work is never a
//! fabricated oracle. Queries happen before creating the actual Step.
use super::*;
use ferrum_interfaces::execution_cost::*;
use std::num::NonZeroU64;

#[path = "full_cost_route/resources.rs"]
mod resources;

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("future-route observation must not enable timing")
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("future-route observation must not enable timing")
    }
}

fn row(count: usize) -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Prefill {
            offset: 0,
            count: count as u32,
            total_prompt_tokens: count as u32,
        },
        host_policy_signature: [3; 32],
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [7; 32],
                decoder_text_bytes_per_token: 8,
                decoder_scratch_bytes_per_token: 16,
                raw_token_bytes_bound: 8,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 0,
                maximum_output_tokens: 8,
                sampling_history_tokens: 0,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
        mask_upload_required: false,
        output: CostRowOutput::Prefill { final_logits: true },
    }
}

fn finish(mut builder: CanonicalWaveCostBuilder, lengths: &[usize]) -> CanonicalStatisticalWave {
    for &length in lengths {
        builder.row(row(length)).unwrap();
    }
    builder
        .finish_with_statistics(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}

pub(super) fn run(
    fixture: &Fixture,
    lengths: &[usize],
    predict: bool,
    staged: bool,
) -> (Vec<Vec<u8>>, CoreReadbackRoute) {
    run_with_bucket(
        fixture,
        lengths,
        predict,
        staged,
        fixture.reusable_bucket.as_ref(),
    )
}

fn run_with_bucket(
    fixture: &Fixture,
    lengths: &[usize],
    predict: bool,
    staged: bool,
    bucket: Option<&ReusableExecutionBucketId>,
) -> (Vec<Vec<u8>>, CoreReadbackRoute) {
    run_with_node_probe(
        fixture,
        lengths,
        predict,
        staged,
        bucket,
        "node.attention",
        None,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn run_with_head_statistics(
    fixture: &Fixture,
    lengths: &[usize],
    require: bool,
) -> Vec<Vec<u8>> {
    run_with_node_probe(
        fixture,
        lengths,
        true,
        false,
        None,
        "node.head",
        Some(("node.head", require)),
    )
    .0
}

/// Real readback from another declared output node; no synthetic range proof.
#[cfg(feature = "cuda")]
pub(super) fn run_with_output(
    fixture: &Fixture,
    lengths: &[usize],
    predict: bool,
    output_node: &str,
) -> (Vec<Vec<u8>>, CoreReadbackRoute) {
    run_with_node_probe(
        fixture,
        lengths,
        predict,
        false,
        fixture.reusable_bucket.as_ref(),
        output_node,
        None,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn run_with_ffn_statistics(
    fixture: &Fixture,
    lengths: &[usize],
    require: bool,
) -> Vec<Vec<u8>> {
    run_with_node_probe(
        fixture,
        lengths,
        true,
        false,
        None,
        "node.ffn",
        Some(("node.ffn", require)),
    )
    .0
}

#[cfg(feature = "cuda")]
pub(super) fn run_with_gdn_statistics(
    fixture: &Fixture,
    lengths: &[usize],
    require: bool,
) -> Vec<Vec<u8>> {
    run_with_node_probe(
        fixture,
        lengths,
        true,
        false,
        None,
        "node.attention",
        Some(("node.attention", require)),
    )
    .0
}

#[cfg(feature = "cuda")]
pub(super) fn run_with_embedding_statistics(
    fixture: &Fixture,
    lengths: &[usize],
    require: bool,
) -> Vec<Vec<u8>> {
    run_with_node_probe(
        fixture,
        lengths,
        true,
        false,
        None,
        "node.embedding",
        Some(("node.embedding", require)),
    )
    .0
}

fn run_with_node_probe(
    fixture: &Fixture,
    lengths: &[usize],
    predict: bool,
    staged: bool,
    bucket: Option<&ReusableExecutionBucketId>,
    output_node: &str,
    node_probe: Option<(&str, bool)>,
) -> (Vec<Vec<u8>>, CoreReadbackRoute) {
    let tokens = lengths
        .iter()
        .enumerate()
        .map(|(row, &length)| {
            (0..length)
                .map(|index| ((row * 3 + index * 7 + 1) % 32) as u32)
                .collect::<Arc<[u32]>>()
        })
        .collect::<Vec<_>>();
    if predict && node_probe.is_some() {
        // Prediction consumes resident capacity; it cannot authorize cold growth.
        // Prepare and release real backing using disposable owners before taking
        // the read-only view for the independent, unexecuted request population.
        resources::make_resident(fixture, &tokens, bucket);
    }
    let original = tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| fixture.admit(&format!("full-route.{index}"), Arc::clone(tokens)))
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(original.clone()).unwrap();
    let sessions = batch.sessions().to_vec();
    let order = sessions
        .iter()
        .map(|session| {
            original
                .iter()
                .position(|candidate| Arc::ptr_eq(session, candidate))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let tokens = order
        .iter()
        .map(|&index| Arc::clone(&tokens[index]))
        .collect::<Vec<_>>();
    let lengths = tokens.iter().map(|tokens| tokens.len()).collect::<Vec<_>>();
    let executable = fixture.compilation.executable();
    let node = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id().as_str() == output_node)
        .unwrap();
    let output = node
        .values()
        .iter()
        .find(|binding| binding.role() == ResolvedValueRole::Output && binding.ordinal() == 0)
        .unwrap();
    let component = &output.storage().components()[0];
    let uploads = tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| {
            SubmissionWaveInputUpload::new(
                id("node.embedding"),
                index as u32,
                0,
                0,
                HostTransferLayout::new(ElementType::U32, tokens.len() as u64).unwrap(),
                tokens
                    .iter()
                    .flat_map(|token| token.to_le_bytes())
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let readbacks = CompletionReadbackBatchRequest::new(
        tokens
            .iter()
            .enumerate()
            .map(|(index, tokens)| {
                CompletionReadbackRequest::new(
                    node.id().clone(),
                    index as u32,
                    component.resource_id().clone(),
                    component.offset_bytes(),
                    HostTransferLayout::new(
                        fixture.output_type,
                        if node_probe.is_some_and(|(node, _)| node == "node.head") {
                            component.length_bytes() / fixture.output_type.size_bytes()
                        } else {
                            tokens.len() as u64 * HIDDEN
                        },
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let rows = lengths
        .iter()
        .map(|&count| OperationCostWorkRow {
            offset: 0,
            count: NonZeroU64::new(count as u64).unwrap(),
            full_input_tokens: NonZeroU64::new(count as u64).unwrap(),
        })
        .collect::<Vec<_>>();
    let selected_node = node_probe
        .filter(|(_, required)| *required)
        .map(|(name, _)| {
            let index = executable
                .execution_plan()
                .payload()
                .nodes()
                .iter()
                .position(|node| node.id().as_str() == name)
                .unwrap();
            let route = fixture.providers.providers()[index]
                .eager_cost_route(executable, &rows)
                .unwrap()
                .expect("selected node query is fully specified before admission");
            assert!(!route.commands().is_empty());
            (
                index,
                route
                    .commands()
                    .iter()
                    .map(|command| {
                        command
                            .statistical_evidence()
                            .expect("future selected evidence")
                            .clone()
                    })
                    .collect::<Vec<_>>(),
            )
        });
    let predicted = if predict {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let view = loop {
            match fixture.resources.execution_cost_route_view(
                &sessions.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                &vec![0; sessions.len()],
                &fixture.lane,
                ResourcePlanningLimits::default(),
                &mut || true,
            ) {
                ExecutionCostRouteAvailability::Known(view) => break view,
                ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                )) if std::time::Instant::now() < deadline => std::thread::yield_now(),
                result => panic!("quiescent route view unavailable: {result:?}"),
            }
        };
        let initial = view.initial_state();
        let inputs = uploads
            .iter()
            .map(|upload| EagerCoreInputUpload {
                node_id: upload.node_id(),
                input_ordinal: upload.input_ordinal(),
                participant_index: upload.participant_index() as usize,
                logical_offset_bytes: upload.logical_offset_bytes(),
                layout: upload.source_layout(),
            })
            .collect::<Vec<_>>();
        let outputs = readbacks
            .requests()
            .iter()
            .map(|request| EagerCoreReadback {
                node_id: request.node_id(),
                resource_id: request.resource_id(),
                participant_index: request.participant_index() as usize,
                logical_offset_bytes: request.logical_offset_bytes(),
                layout: request.output_layout(),
            })
            .collect::<Vec<_>>();
        let indices = (0..sessions.len()).collect::<Vec<_>>();
        let query = EagerCoreWaveCostQuery {
            rows: &rows,
            participant_indices: &indices,
            uploads: &inputs,
            readbacks: &outputs,
            attempt_staged_readbacks: staged,
            reusable_bucket: bucket,
            token_mask_input: None,
        };
        let before = CapacitySnapshot::observe(&fixture.resources);
        let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
        let next = append_complete_eager_cost_route(
            fixture._composition._runtime.as_ref(),
            &fixture.providers,
            executable,
            &fixture.resources,
            &view,
            &initial,
            &query,
            &mut builder,
            &mut || true,
        )
        .unwrap();
        assert_eq!(initial.projected_waves(), 0);
        assert_eq!(next.projected_waves(), 1);
        let after = CapacitySnapshot::observe(&fixture.resources);
        assert_eq!(before.resident_bytes, after.resident_bytes);
        assert_eq!(before.free_bytes, after.free_bytes);
        assert_eq!(before.budget_claimed_bytes, after.budget_claimed_bytes);
        // A second wave at the old offset cannot borrow the completed first
        // wave's initialization state or replay the same work.
        assert!(matches!(
            append_complete_eager_cost_route(
                fixture._composition._runtime.as_ref(),
                &fixture.providers,
                executable,
                &fixture.resources,
                &view,
                &next,
                &query,
                &mut CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits),
                &mut || true
            ),
            Err(ExecutionCostRouteUnknown::StaleView)
        ));
        Some(finish(builder, &lengths))
    } else {
        None
    };
    let (step, wave) = resources::prepare(fixture, &batch, &tokens, bucket);
    let wave = if staged {
        wave.with_submission_readbacks(readbacks.clone()).unwrap()
    } else {
        wave
    };
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        active.iter(),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let (handle, attribution) = OperationDispatch::encode_and_submit_wave_with_cost_observation(
        fixture.providers.providers(),
        executable,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &uploads,
        SubmissionExecutionPolicy::adaptive(),
        None,
        &NoTiming,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap()
    .into_parts();
    if let Some((node, predicted_head)) = selected_node {
        let commands = attribution
            .as_ref()
            .unwrap()
            .device()
            .commands()
            .iter()
            .filter(|command| command.node_index() == Some(node as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            commands.len(),
            predicted_head.len(),
            "selected physical command population"
        );
        for (command, predicted) in commands.iter().zip(&predicted_head) {
            let observed = command
                .statistical_evidence()
                .expect("actual node must attach evidence");
            assert_eq!(
                observed, predicted,
                "actual selected class/work and future query differ"
            );
            assert_eq!(
                observed.algorithm_work().unwrap().unwrap(),
                predicted.algorithm_work().unwrap().unwrap(),
                "per-algorithm assignment differs despite equal aggregate work"
            );
            SelectedReplayAlgorithmTemplateV1::from_selected(
                predicted,
                command.token_count(),
                command.compute_dispatch_count(),
                command.transfer_command_count(),
            )
            .unwrap()
            .validate_binding(observed)
            .expect("the original selected fixed launch parameters must also match");
            observed
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(observed)
                .unwrap();
        }
    }
    let route = handle.core_readback_route(fixture._composition._runtime.as_ref());
    if let Some(predicted) = predicted {
        let mut actual = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
        actual.core_readback_route(route).unwrap();
        for command in attribution.as_ref().unwrap().device().commands() {
            let provider = command.node_index().map(|index| {
                let descriptor = fixture.providers.providers()[index as usize].descriptor();
                CostProviderIdentity {
                    provider_id: descriptor.provider_id().as_str(),
                    implementation_fingerprint: descriptor.provider_implementation_fingerprint(),
                    operation_fingerprint: descriptor.operation_fingerprint(),
                }
            });
            actual
                .physical_command(CostPhysicalCommand::from_attribution(command, provider))
                .unwrap();
        }
        let actual = finish(actual, &lengths);
        assert_eq!(
            predicted.exact, actual.exact,
            "full actual core/provider/readback route differs"
        );
        #[cfg(all(feature = "metal", not(feature = "cuda"), target_os = "macos"))]
        {
            // This fixture joins real embedding, attention and core commands.
            // Its declared host fields do not measure serving host work, and
            // this check is not coverage of a full Qwen model or its timings.
            let predicted_statistics = predicted
                .statistical
                .expect("the complete projected native wave must have selected work evidence");
            let actual_statistics = actual
                .statistical
                .expect("every submitted native/core command must carry selected work evidence");
            predicted_statistics
                .validate_exact(&predicted.exact)
                .unwrap();
            actual_statistics.validate_exact(&actual.exact).unwrap();
            assert_eq!(
                predicted_statistics, actual_statistics,
                "complete selected algorithms, work and exact bindings differ"
            );
        }
    }
    let CompletionReadbackBatchObservation::Terminal(receipt) =
        handle.wait_with_readbacks(readbacks).unwrap()
    else {
        panic!("real full route did not complete")
    };
    assert_eq!(
        receipt.completion().fence_timing().timing_mode(),
        DeviceTimingMode::Off
    );
    let values = receipt
        .dispositions()
        .iter()
        .map(|disposition| match disposition {
            CompletionReadbackDisposition::Succeeded(output) => output.bytes().to_vec(),
            other => panic!("real output failed: {other:?}"),
        })
        .collect::<Vec<_>>();
    drop((receipt, handle, attribution, identity, active));
    step.try_retire_normal().unwrap();
    for session in sessions {
        session.try_abort_if_quiescent().unwrap();
    }
    let mut ordered = order.into_iter().zip(values).collect::<Vec<_>>();
    ordered.sort_by_key(|(index, _)| *index);
    (ordered.into_iter().map(|(_, bytes)| bytes).collect(), route)
}

#[test]
fn full_future_native_route_matches_actual_pending_initialization_and_packed_uploads() {
    let fixture = Fixture::new(AttentionKind::GatedDelta);
    let baseline = run(&fixture, &[2, 3], false, false).0;
    let actual = run(&fixture, &[2, 3], true, false);
    assert_eq!(actual.0, baseline);
    assert_eq!(actual.1, CoreReadbackRoute::HostSynchronized);
}

#[test]
fn full_future_native_route_distinguishes_actual_staged_and_fallback_readbacks() {
    let fixture = Fixture::new(AttentionKind::GatedDelta);
    let baseline = run(&fixture, &[2, 3], false, false).0;
    fixture
        .lane
        .configure_submission_readback_staging(1 << 20)
        .unwrap();
    let staged = run(&fixture, &[2, 3], true, true);
    assert_eq!(staged.0, baseline);
    assert_eq!(staged.1, CoreReadbackRoute::SubmissionStaged);
    fixture
        .lane
        .configure_submission_readback_staging(0)
        .unwrap();
    let fallback = run(&fixture, &[2, 3], true, true);
    assert_eq!(fallback.0, baseline);
    assert_eq!(
        fallback.1,
        CoreReadbackRoute::SubmissionFallbackSynchronized
    );
}

#[test]
fn full_future_native_route_matches_workspace_causal_binding_prelude_cold_and_reused() {
    let fixture = Fixture::with_workspace_capacity(AttentionKind::Causal, 4);
    let memory = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .memory();
    let reusable = memory.reusable_execution().unwrap();
    assert!(reusable.program_policy().is_none());
    assert!(fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .any(|node| node.binding_resource().is_some()));
    // Establish resident pool capacity using the actual transient path. This
    // performs real work, but creates no retained workspace slot. The query
    // below must project its first bucketed claim without allocating it.
    run_with_bucket(&fixture, &[4], false, false, None);
    let baseline = run_with_bucket(&fixture, &[3], false, false, None).0;
    let first = run(&fixture, &[3], true, false).0;
    assert_eq!(first, baseline);
    // Same bucket, smaller logical prefix and an already resident idle slot.
    let baseline = run_with_bucket(&fixture, &[2], false, false, None).0;
    let reused = run(&fixture, &[2], true, false).0;
    assert_eq!(reused, baseline);
}
