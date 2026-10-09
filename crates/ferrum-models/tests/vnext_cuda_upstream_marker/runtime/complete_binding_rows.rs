use super::*;
use ferrum_types::{InvocationPreparationStrategy, ProgramBindingUploadStrategy};

struct LazyPreparation;
impl InvocationPreparationSink for LazyPreparation {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        assert!(stats.projected_identities > 0);
        assert_eq!(stats.parts_materialized, 0);
    }
}

fn snapshot(fixture: &Fixture) -> DeviceProgramBindingUploadSnapshot {
    fixture
        ._composition
        ._runtime
        .program_binding_upload_snapshot()
        .unwrap()
}

fn assert_causal_rows(observation: &BatchObservation, ranges: &[Range<usize>], complete: bool) {
    let rows = observation.binding_rows.as_ref().unwrap();
    assert_eq!(rows.len(), ranges.len());
    for (participant, (row, range)) in rows.iter().zip(ranges).enumerate() {
        // This real fixture selects the FP16 VllmBlocks16 ABI: six i32
        // controls followed by one u64 per live KV block. Address values are
        // allocation-specific and must not be compared between runtimes.
        let controls: Vec<i32> = row[..24]
            .chunks_exact(4)
            .map(|v| i32::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        let entries = range.end.div_ceil(16);
        assert_eq!(
            controls,
            [
                entries as i32,
                range.start as i32,
                1,
                range.end as i32,
                participant as i32,
                0
            ]
        );
        let live_end = 24 + entries * 8;
        assert!(row[24..live_end]
            .chunks_exact(8)
            .all(|v| u64::from_ne_bytes(v.try_into().unwrap()) != 0));
        if complete {
            assert!(
                row[live_end..].iter().all(|&v| v == 0),
                "only explicitly authorized inactive logical-row bytes are zeroed"
            );
        }
    }
}

fn compare(kind: AttentionKind) {
    let modes = [
        ProgramBindingUploadStrategy::Sparse,
        ProgramBindingUploadStrategy::CompleteRows,
    ];
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_binding_upload(kind, true, 2, Family::new(kind), mode)
    });
    // Advance one causal participant through a full 64-KiB physical KV page
    // before joint decode (64 tokens for this real 1024-byte/token fixture).
    // Joint waves use two physical pages versus one, as well as unequal live
    // VllmBlocks16 address-table lengths.
    let lead = if kind == AttentionKind::Causal { 64 } else { 0 };
    let joint_waves = 17;
    let tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|participant| {
            (0..if participant == 0 {
                lead + joint_waves
            } else {
                joint_waves
            })
                .map(|position| ((position * 7 + participant * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, tokens)| {
                fixture.admit_with_ceiling(
                    &format!("complete-binding-rows-{p}"),
                    Arc::from(&tokens[..1]),
                    tokens.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    for position in 0..lead {
        let mut observations = Vec::new();
        for (fixture, group) in fixtures.iter().zip(&sessions) {
            fixture.extend(&group[0], Arc::from(&tokens[0][..=position]));
            observations.push(fixture.execute_participants_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                &group[..1],
                &tokens[..1],
                position..position + 1,
                Path::Warm,
                InvocationPreparationStrategy::IdentityProjection,
                &LazyPreparation,
            ));
        }
        observations[0].assert_same(&observations[1]);
    }
    let baselines = fixtures.each_ref().map(snapshot);
    // The fixture's maximum 160 tokens gives ten u64 block addresses. Their
    // 80-byte area is already aligned to the provider's 16-byte alignment.
    let row_bytes = (kind == AttentionKind::Causal).then_some(24 + 10 * 8);
    for position in 0..joint_waves {
        let ranges = [lead + position..lead + position + 1, position..position + 1];
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let mut observations = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&sessions).enumerate() {
            for ((session, tokens), range) in group.iter().zip(&tokens).zip(&ranges) {
                fixture.extend(session, Arc::from(&tokens[..range.end]));
            }
            let physical_kv_pages = if kind == AttentionKind::Causal {
                let attention = fixture
                    .compilation
                    .executable()
                    .execution_plan()
                    .payload()
                    .nodes()
                    .iter()
                    .find(|n| n.id().as_str() == "node.attention")
                    .unwrap();
                let kv_resource = attention
                    .values()
                    .iter()
                    .find(|v| v.role() == ResolvedValueRole::Input && v.ordinal() == 8)
                    .unwrap()
                    .storage()
                    .components()[0]
                    .resource_id();
                let pages: Vec<_> = group.iter().zip(&tokens).zip(&ranges).map(|((session, tokens), range)| {
                    let current_tokens: Arc<[u32]> = Arc::from(&tokens[..range.end]);
                    let work = ResourceWorkShape::single(token_span(current_tokens, 0..range.end)).unwrap();
                    let request = SequenceResourceExtensionRequest::new(work, AdmissionPressureAction::WaitForRelease).unwrap();
                    let SequenceResourceExtensionDecision::Current(backing) = session.try_ensure_backing_covers(request).unwrap()
                        else { panic!("completed fixture extension must return its actual current snapshot") };
                    let bytes: u64 = backing.backing_slices().iter().filter(|s| s.resource_id() == kv_resource)
                        .flat_map(|s| s.evidence().segments()).map(|s| s.length_bytes()).sum();
                    assert_eq!(bytes % 65_536, 0);
                    bytes / 65_536
                }).collect();
                assert_eq!(
                    pages,
                    vec![2, 1],
                    "actual captured physical KV extents must differ"
                );
                Some(pages)
            } else {
                None
            };
            let before = snapshot(fixture);
            let observation = fixture.execute_participant_ranges_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                group,
                &tokens,
                &ranges,
                path,
                InvocationPreparationStrategy::IdentityProjection,
                &LazyPreparation,
                row_bytes,
            );
            let uploads = snapshot(fixture).checked_since(before).unwrap();
            assert!(uploads.attempted_preludes > 0);
            assert_eq!(uploads.failed_preludes, 0);
            assert_eq!(uploads.attempted_preludes, uploads.succeeded_preludes);
            assert_eq!(
                uploads.successful_upload_bytes,
                uploads.planned_upload_bytes
            );
            assert!(uploads.successful_1d_copies + uploads.successful_2d_copies > 0);
            if kind == AttentionKind::Causal {
                assert_causal_rows(&observation, &ranges, arm == 1);
                if arm == 1 {
                    assert!(uploads.planned_upload_bytes > uploads.live_payload_bytes);
                }
            } else {
                assert_eq!(
                    uploads.planned_upload_bytes, uploads.live_payload_bytes,
                    "dense GDN payload is unchanged by the causal-row policy"
                );
            }
            observation.dump(kind, 2, ranges[0].clone(), path);
            println!(
                "{}",
                serde_json::json!({
                    "kind":"complete_binding_rows_wave", "attention":format!("{kind:?}"),
                    "strategy":modes[arm], "participant_ranges":ranges,
                    "uploads":uploads, "captured_physical_kv_pages":physical_kv_pages, "logical_tail_zero_checked":kind == AttentionKind::Causal && arm == 1,
                    "causal_controls_and_status_zero_checked":kind == AttentionKind::Causal,
                })
            );
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
    }
    let totals = [
        snapshot(&fixtures[0]).checked_since(baselines[0]).unwrap(),
        snapshot(&fixtures[1]).checked_since(baselines[1]).unwrap(),
    ];
    if kind == AttentionKind::GatedDelta {
        assert_eq!(
            totals[0], totals[1],
            "GDN upload behavior is the unchanged control"
        );
    }
    println!(
        "{}",
        serde_json::json!({"kind":"complete_binding_rows_actual_cuda",
        "attention":format!("{kind:?}"), "participants":2, "joint_waves":joint_waves,
        "lead_in_waves":lead, "full_output_and_state_equal":true,
        "sparse":totals[0], "complete_rows":totals[1]})
    );
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, actual causal/FFN providers and resident replay"]
fn complete_binding_rows_causal_matches_sparse_with_mixed_kv_lengths() {
    compare(AttentionKind::Causal);
}

#[test]
#[ignore = "requires exclusive CUDA, actual GDN/FFN providers and resident replay"]
fn complete_binding_rows_gdn_matches_sparse_across_committed_frontiers() {
    compare(AttentionKind::GatedDelta);
}
