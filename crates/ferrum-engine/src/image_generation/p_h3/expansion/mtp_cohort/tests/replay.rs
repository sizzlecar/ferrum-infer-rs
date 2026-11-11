//! Exercise changing real state, accepted and rejected drafts, and cancellation
//! after B2 work. The CUDA arm additionally requires actual catalog replay.
use super::*;
use std::collections::BTreeMap;

#[derive(Debug, Default, PartialEq, Eq)]
struct Outputs {
    tokens: [Vec<u32>; 2],
    hidden: [BTreeMap<usize, Vec<u8>>; 2],
    b2: bool,
    b1_after_cancel: bool,
}

fn requests(
    fixture: &Fixture,
    device: ferrum_types::Device,
    replay: bool,
) -> Vec<PreparedTextExpansion> {
    [(3, 3), (ANSWER_ID, 5)]
        .into_iter()
        .map(|(token, count)| {
            let mut engine = EngineConfig::default();
            engine.backend.device = device.clone();
            if replay {
                engine.backend.reusable_execution_capture.scope =
                    ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
                engine.backend.reusable_execution_capture.preparation =
                    ferrum_types::ReusableExecutionPreparationMode::OnDemand;
                engine
                    .backend
                    .reusable_execution_capture
                    .maximum_executables_per_lane = Some(128);
            }
            let mut request = fixture
                .assets
                .prepare_text("draw a cat", 32, 32, 16, 2, engine)
                .unwrap();
            request.tokens = vec![TokenId::new(token); count];
            request.engine.runtime.max_model_len = Some(count + 16);
            request
        })
        .collect()
}

fn assert_candidate_gdn_segments<R: DeviceRuntime>(
    program: &CausalProgramSession<R>,
    participants: usize,
) {
    let plan = program.plan().execution_plan();
    let candidate_nodes = plan
        .payload()
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            node.operation_id().as_str()
                == ferrum_runtime::GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID
        })
        .map(|(index, _)| index as u32)
        .collect::<Vec<_>>();
    assert!(
        !candidate_nodes.is_empty(),
        "fixture must execute candidate GDN"
    );
    let lane = program.execution_lane().unwrap();
    let catalog = lane
        .try_reusable_execution_catalog()
        .unwrap()
        .expect("completed verify must leave its original lane quiescent");
    let id = program
        .last_successful_reusable_execution_test_program()
        .expect("successful direct verify must identify the exact replayed program");
    assert_eq!(id.plan_hash(), plan.plan_hash());
    assert_eq!(id.immediate_sequences() as usize, participants);
    assert_eq!(id.immediate_tokens(), participants as u64 * 2);
    let entry = catalog
        .programs()
        .iter()
        .find(|entry| entry.program_id() == &id)
        .expect("the completed verify must retain its exact resident program");
    // Require positive coverage in the actual replayed program. Absence from
    // eager boundaries alone could instead mean a rejected capture or a gap.
    for &node in &candidate_nodes {
        assert!(
            entry
                .segments()
                .iter()
                .any(|segment| segment.contains_node(node)),
            "candidate GDN node {node} is not in a resident segment: {entry:?}"
        );
    }
}

async fn run<R: DeviceRuntime>(
    fixture: &Fixture,
    composition: &RuntimeComposition<R>,
    device: ferrum_types::Device,
    replay: bool,
) -> (Outputs, [u64; 2]) {
    let host = host();
    let mut cohort =
        fixture.machine_with_requests(composition, &host, requests(fixture, device, replay));
    let mut observed = Outputs::default();
    let mut direct = [0; 2];
    let mut candidate_replayed_b2 = false;
    let mut candidate_replayed_after_cancel = false;
    let mut replay_snapshots = [serde_json::Value::Null, serde_json::Value::Null];
    let mut cancelled = [false; 2];
    let result = tokio::time::timeout(
        Duration::from_secs(45),
        std::future::poll_fn(|cx| {
            for (index, program) in [cohort.programs.main(), cohort.programs.draft()]
                .into_iter()
                .enumerate()
            {
                if let Ok(program) = program {
                    if replay {
                        program.enable_reusable_execution_test_diagnostics();
                    }
                    let mut stats = program.reusable_execution_snapshot();
                    stats.as_object_mut().unwrap().remove("workspace_buckets");
                    stats["plan"] =
                        serde_json::to_value(program.plan().execution_plan().plan_hash()).unwrap();
                    let previous_direct = direct[index];
                    direct[index] = direct[index].max(stats["direct_waves"].as_u64().unwrap());
                    if replay && index == 0 && direct[index] > previous_direct {
                        if let Some(settling) = &cohort.settling {
                            if settling.kind == WaveKind::Verify {
                                assert_candidate_gdn_segments(program, settling.indices.len());
                                candidate_replayed_b2 |= settling.indices.len() == 2;
                                candidate_replayed_after_cancel |=
                                    cancelled[0] && settling.indices == [1];
                            }
                        }
                    }
                    if replay && direct[index] > 0 {
                        assert!(
                            stats["eager_boundary_operations"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|operation| operation.as_str()
                                    == Some(ferrum_runtime::TAIL_TWO_GREEDY_F16_OPERATION_ID)),
                            "terminal selection must stay outside captured compute segments"
                        );
                    }
                    replay_snapshots[index] = stats;
                }
            }
            if let Some(settling) = &cohort.settling {
                if settling.kind == WaveKind::Verify {
                    observed.b2 |= settling.indices.len() == 2;
                    observed.b1_after_cancel |= cancelled[0] && settling.indices == [1];
                }
            }
            let progress = cohort.advance(&cancelled, cx.waker()).unwrap();
            for (index, cursor) in cohort.cursors.iter().enumerate() {
                let Some(cursor) = cursor else { continue };
                observed.tokens[index] = cursor.output.ids().to_vec();
                if let Some(hidden) = cursor.accepted_hidden_bytes(&host).unwrap() {
                    if let Some(previous) =
                        observed.hidden[index].insert(cursor.accepted_frontier(), hidden.clone())
                    {
                        assert_eq!(
                            previous, hidden,
                            "accepted state changed without a new frontier"
                        );
                    }
                }
            }
            // Retire one actual participant first, then its survivor. These are
            // bounded correctness requests, deliberately cancelled before length.
            cancelled[0] |= observed.tokens[0].len() >= 10;
            cancelled[1] |= observed.tokens[1].len() >= 12;
            match progress {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(results) => Poll::Ready(results),
            }
        }),
    )
    .await
    .expect("real causal owners must wake and retire");
    assert!(result
        .into_iter()
        .all(|result| matches!(result, ExpansionParticipantResult::Cancelled)));
    assert!(observed.b2 && observed.b1_after_cancel);
    assert_ne!(observed.tokens[0][0], observed.tokens[1][0]);
    let advances = |index: usize, prompt: usize| {
        observed.hidden[index]
            .keys()
            .copied()
            .filter(|&position| position >= prompt)
            .collect::<Vec<_>>()
    };
    assert!(
        advances(0, 3).windows(2).any(|pair| pair[1] - pair[0] == 2),
        "real accepted draft required"
    );
    assert!(
        advances(1, 5).windows(2).any(|pair| pair[1] - pair[0] == 1),
        "real rejected draft required"
    );
    assert!(
        observed.hidden[0]
            .values()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|pair| pair[0] != pair[1]),
        "nonzero recurrent/attention state must change the accepted hidden rows"
    );
    if replay {
        assert!(
            candidate_replayed_b2 && candidate_replayed_after_cancel,
            "candidate GDN must replay in actual B2 verification and the surviving B1 verification"
        );
        eprintln!(
            "{}",
            serde_json::json!({
                "diagnostic":"mtp_replay_catalogs",
                "main":replay_snapshots[0],
                "draft":replay_snapshots[1],
                "candidate_gdn_replayed_b2":candidate_replayed_b2,
                "candidate_gdn_replayed_after_cancel":candidate_replayed_after_cancel,
            })
        );
    }
    assert!(cohort.try_close().unwrap());
    drop(cohort);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    (observed, direct)
}

#[tokio::test]
async fn mtp_nonzero_continuations_accept_reject_and_cancel_real_b2_state() {
    let fixture = Fixture::nonzero_continuations();
    let (composition, runtime) = composition();
    let (_, direct) = run(&fixture, &composition, ferrum_types::Device::CPU, false).await;
    assert_eq!(direct, [0, 0]);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires exclusive CUDA; real MTP state/cancellation/replay correctness, not model speed"]
async fn cuda_mtp_replay_preserves_nonzero_state_and_retires_original_lanes() {
    let fixture = Fixture::nonzero_continuations();
    let mut results = Vec::new();
    for replay in [false, true] {
        let (composition, runtime) =
            crate::image_generation::p_h3::expansion::replay_tests::cuda::composition(replay);
        let weak = Arc::downgrade(&runtime);
        let (observed, direct) = run(
            &fixture,
            &composition,
            ferrum_types::Device::CUDA(0),
            replay,
        )
        .await;
        if replay {
            assert!(
                direct.into_iter().all(|count| count > 0),
                "both main and draft must actually replay: {direct:?}"
            );
        } else {
            assert_eq!(direct, [0, 0]);
        }
        eprintln!(
            "{}",
            serde_json::json!({"diagnostic":"mtp_causal_replay", "replay":replay, "direct_waves":direct, "outputs":observed.tokens})
        );
        results.push(observed);
        drop(composition);
        drop(runtime);
        assert!(
            weak.upgrade().is_none(),
            "closed main/draft lanes retained their runtime"
        );
    }
    assert_eq!(
        results[0], results[1],
        "accepted hidden rows and every committed token must match eager bitwise"
    );
}
