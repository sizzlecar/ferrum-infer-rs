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

fn causal_live_bytes(range: &Range<usize>) -> usize {
    24 + range.end.div_ceil(16) * 8
}

fn assert_causal_rows(observation: &BatchObservation, ranges: &[Range<usize>], uniform: bool) {
    let rows = observation.binding_rows.as_ref().unwrap();
    assert_eq!(rows.len(), ranges.len());
    let maximum_live_bytes = ranges.iter().map(causal_live_bytes).max().unwrap();
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
                i32::try_from(observation.caller_to_canonical_participant[participant]).unwrap(),
                0
            ]
        );
        let live_end = 24 + entries * 8;
        assert!(row[24..live_end]
            .chunks_exact(8)
            .all(|v| u64::from_ne_bytes(v.try_into().unwrap()) != 0));
        if uniform {
            assert!(
                row[live_end..maximum_live_bytes].iter().all(|&v| v == 0),
                "only padding to this wave's maximum live prefix is zeroed"
            );
        }
        // Bytes beyond maximum_live_bytes are not part of this upload. They
        // need not be zero and are not compared between independent runtimes.
    }
}

fn compare(kind: AttentionKind, alternative: ProgramBindingUploadStrategy) {
    let modes = [ProgramBindingUploadStrategy::Sparse, alternative];
    let compact = alternative == ProgramBindingUploadStrategy::CompactScatter;
    // Keep the existing numerical profile while making unused logical table
    // capacity much larger than this fixture's real frontiers. The two arms
    // must upload live data, not this 8192-token maximum's complete table.
    let maximum_tokens = if kind == AttentionKind::Causal {
        8192
    } else {
        MAX_TOKENS
    };
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_binding_upload(
            kind,
            true,
            2,
            Family::new(kind).with_maximum_tokens(maximum_tokens),
            mode,
        )
    });
    // A physical KV page holds 64 tokens for this real 1024-byte/token fixture.
    // Joint waves exercise unequal live VllmBlocks16 address-table lengths
    // and eventually two physical pages versus one.
    // Compact crosses 64 -> 65 after resident replay has already begun;
    // retain the original Uniform fixture's lead and execution order.
    let lead = if kind == AttentionKind::Causal {
        if compact {
            60
        } else {
            64
        }
    } else {
        0
    };
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
                    &format!("uniform-binding-prefix-{p}"),
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
    // F16 VllmBlocks16 has one u64 per logical block and a 16-byte-aligned
    // address table after six controls. This gives a 4120-byte logical row
    // while the joint frontiers require only 32..72 bytes per live payload.
    let maximum_address_bytes = (maximum_tokens.div_ceil(16) * 8).div_ceil(16) * 16;
    let row_bytes = if kind == AttentionKind::Causal {
        Some(24 + maximum_address_bytes as usize)
    } else {
        compact.then_some(16)
    };
    for position in 0..joint_waves {
        let caller_order = if compact && position >= joint_waves - 2 {
            [1, 0]
        } else {
            [0, 1]
        };
        let original_ranges = [lead + position..lead + position + 1, position..position + 1];
        let ranges = caller_order.map(|p| original_ranges[p].clone());
        let wave_tokens = caller_order.map(|p| Arc::clone(&tokens[p]));
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let mut observations = Vec::new();
        let mut wave_uploads = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&sessions).enumerate() {
            let group = caller_order.map(|p| Arc::clone(&group[p]));
            for ((session, tokens), range) in group.iter().zip(&wave_tokens).zip(&ranges) {
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
                let pages: Vec<_> = group.iter().zip(&wave_tokens).zip(&ranges).map(|((session, tokens), range)| {
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
                    ranges
                        .iter()
                        .map(|range| range.end.div_ceil(64) as u64)
                        .collect::<Vec<_>>(),
                    "actual captured physical KV extents must cover the current frontiers"
                );
                Some(pages)
            } else {
                None
            };
            let before = snapshot(fixture);
            let observation = fixture.execute_participant_ranges_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                &group,
                &wave_tokens,
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
            let is_compact = modes[arm] == ProgramBindingUploadStrategy::CompactScatter;
            let is_uniform = modes[arm] == ProgramBindingUploadStrategy::UniformLivePrefix;
            if is_compact {
                assert_eq!(uploads.compact_scatter_preludes, uploads.succeeded_preludes);
                assert_eq!(uploads.compact_scatter_sparse_fallback_preludes, 0);
                assert_eq!(
                    uploads.successful_scatter_dispatches,
                    uploads.succeeded_preludes
                );
                assert_eq!(uploads.successful_1d_copies, uploads.attempted_preludes);
                assert_eq!(uploads.successful_2d_copies, 0);
                assert!(uploads.planned_upload_bytes > uploads.live_payload_bytes);
                assert_eq!(
                    (uploads.planned_upload_bytes - uploads.live_payload_bytes) % 40,
                    0,
                    "each compact descriptor adds five u64 fields to live payload bytes"
                );
            } else {
                assert_eq!(uploads.compact_scatter_preludes, 0);
                assert_eq!(uploads.compact_scatter_sparse_fallback_preludes, 0);
                assert_eq!(uploads.successful_scatter_dispatches, 0);
            }
            if kind == AttentionKind::Causal {
                assert_causal_rows(&observation, &ranges, is_uniform);
                let live_bytes = ranges.iter().map(causal_live_bytes).sum::<usize>() as u64;
                let maximum_live_bytes = ranges.iter().map(causal_live_bytes).max().unwrap() as u64;
                let planned_bytes = if is_uniform {
                    ranges.len() as u64 * maximum_live_bytes
                } else {
                    live_bytes
                };
                // These counters count typed enqueue attempts, not waves.
                // In this fixture the only binding payload is this node's
                // causal table, so live bytes are exact. Compact adds its
                // independently validated descriptor headers to the plan.
                assert_eq!(
                    uploads.live_payload_bytes,
                    uploads.attempted_preludes * live_bytes
                );
                if !is_compact {
                    assert_eq!(
                        uploads.planned_upload_bytes,
                        uploads.attempted_preludes * planned_bytes
                    );
                }
                assert_eq!(
                    uploads.logical_arena_bytes,
                    uploads.attempted_preludes * ranges.len() as u64 * row_bytes.unwrap() as u64
                );
                assert!(maximum_live_bytes < row_bytes.unwrap() as u64);
                assert!(uploads.planned_upload_bytes < uploads.logical_arena_bytes);
            } else {
                if !is_compact {
                    assert_eq!(
                        uploads.planned_upload_bytes, uploads.live_payload_bytes,
                        "dense GDN payload is unchanged by the causal-row policy"
                    );
                }
                if compact {
                    let rows = observation.binding_rows.as_ref().unwrap();
                    assert_eq!(rows.len(), ranges.len());
                    for row in rows {
                        assert_eq!(row.len(), 16);
                        assert!(row
                            .chunks_exact(8)
                            .all(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()) != 0));
                    }
                }
            }
            observation.dump(kind, 2, ranges[0].clone(), path);
            println!(
                "{}",
                serde_json::json!({
                    "kind":if compact { "compact_scatter_wave" } else { "uniform_binding_prefix_wave" }, "attention":format!("{kind:?}"),
                    "strategy":modes[arm], "participant_ranges":ranges,
                    "caller_order":caller_order,
                    "uploads":uploads, "captured_physical_kv_pages":physical_kv_pages,
                    "logical_row_bytes":row_bytes,
                    "maximum_live_prefix_bytes":(kind == AttentionKind::Causal).then(|| ranges.iter().map(causal_live_bytes).max().unwrap()),
                    "uniform_prefix_zero_checked":kind == AttentionKind::Causal && is_uniform,
                    "causal_controls_and_status_zero_checked":kind == AttentionKind::Causal,
                    "gdn_state_addresses_checked":kind == AttentionKind::GatedDelta && compact,
                })
            );
            wave_uploads.push(uploads);
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
        if compact {
            assert_eq!(
                wave_uploads[0].live_payload_bytes,
                wave_uploads[1].live_payload_bytes
            );
            assert_eq!(
                wave_uploads[0].logical_arena_bytes,
                wave_uploads[1].logical_arena_bytes
            );
            assert_eq!(
                wave_uploads[0].attempted_preludes,
                wave_uploads[1].attempted_preludes
            );
        }
    }
    let totals = [
        snapshot(&fixtures[0]).checked_since(baselines[0]).unwrap(),
        snapshot(&fixtures[1]).checked_since(baselines[1]).unwrap(),
    ];
    if kind == AttentionKind::GatedDelta && !compact {
        assert_eq!(
            totals[0], totals[1],
            "GDN upload behavior is the unchanged control"
        );
    }
    let mut summary = serde_json::json!({"kind":if compact { "compact_scatter_actual_cuda" } else { "uniform_binding_prefix_actual_cuda" },
        "attention":format!("{kind:?}"), "participants":2, "joint_waves":joint_waves,
        "lead_in_waves":lead, "maximum_context_tokens":maximum_tokens,
        "logical_row_bytes":row_bytes, "full_output_and_state_equal":true,
        "counter_scope":"typed binding prelude attempts during joint waves; lead excluded",
        "sparse":totals[0]});
    summary[if compact {
        "compact_scatter"
    } else {
        "uniform_live_prefix"
    }] = serde_json::to_value(totals[1]).unwrap();
    println!("{summary}");
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, actual causal/FFN providers and resident replay"]
fn uniform_binding_prefix_causal_matches_sparse_with_mixed_kv_lengths() {
    compare(
        AttentionKind::Causal,
        ProgramBindingUploadStrategy::UniformLivePrefix,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual GDN/FFN providers and resident replay"]
fn uniform_binding_prefix_gdn_matches_sparse_across_committed_frontiers() {
    compare(
        AttentionKind::GatedDelta,
        ProgramBindingUploadStrategy::UniformLivePrefix,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual causal/FFN providers and resident replay"]
fn compact_scatter_causal_matches_sparse_across_kv_growth_and_reorder() {
    compare(
        AttentionKind::Causal,
        ProgramBindingUploadStrategy::CompactScatter,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual GDN/FFN providers and resident replay"]
fn compact_scatter_gdn_matches_sparse_across_committed_frontiers_and_reorder() {
    compare(
        AttentionKind::GatedDelta,
        ProgramBindingUploadStrategy::CompactScatter,
    );
}
