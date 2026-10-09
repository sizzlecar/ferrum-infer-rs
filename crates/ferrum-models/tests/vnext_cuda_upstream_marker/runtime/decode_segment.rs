//! Independent Sparse histories exercise actual whole-segment publication,
//! fresh requests and a physical KV extent change after an observed hot wave.
use super::*;
use ferrum_types::{InvocationPreparationStrategy, ProgramBindingUploadStrategy};
use std::cell::RefCell;

#[derive(Default)]
struct Preparation(RefCell<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.borrow_mut().push(stats);
    }
}
impl Preparation {
    fn last(&self) -> InvocationPreparationStats {
        *self
            .0
            .borrow()
            .last()
            .expect("dispatch must report actual preparation")
    }
}

fn physical_pages(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    tokens: Arc<[u32]>,
    end: usize,
) -> u64 {
    let attention = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id().as_str() == "node.attention")
        .unwrap();
    let resource = attention
        .values()
        .iter()
        .find(|binding| binding.role() == ResolvedValueRole::Input && binding.ordinal() == 8)
        .unwrap()
        .storage()
        .components()[0]
        .resource_id();
    let work = ResourceWorkShape::single(token_span(tokens, 0..end)).unwrap();
    let request =
        SequenceResourceExtensionRequest::new(work, AdmissionPressureAction::WaitForRelease)
            .unwrap();
    let SequenceResourceExtensionDecision::Current(backing) =
        session.try_ensure_backing_covers(request).unwrap()
    else {
        panic!("fixture extension must expose its actual committed backing")
    };
    assert!(backing.committed_tokens() >= end as u64);
    let bytes: u64 = backing
        .backing_slices()
        .iter()
        .filter(|slice| slice.resource_id() == resource)
        .flat_map(|slice| slice.evidence().segments())
        .map(|segment| segment.length_bytes())
        .sum();
    assert!(bytes > 0 && bytes % 65_536 == 0);
    bytes / 65_536
}

fn check_binding_controls(
    kind: AttentionKind,
    observations: &[BatchObservation],
    ranges: &[Range<usize>],
) {
    for observation in observations {
        let rows = observation.binding_rows.as_ref().unwrap();
        for (participant, (row, range)) in rows.iter().zip(ranges).enumerate() {
            if kind == AttentionKind::Causal {
                let controls = row[..24]
                    .chunks_exact(4)
                    .map(|v| i32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    controls,
                    [
                        range.end.div_ceil(16) as i32,
                        range.start as i32,
                        1,
                        range.end as i32,
                        i32::try_from(observation.caller_to_canonical_participant[participant])
                            .unwrap(),
                        0
                    ]
                );
                let live_end = 24 + range.end.div_ceil(16) * 8;
                assert!(row[24..live_end]
                    .chunks_exact(8)
                    .all(|v| u64::from_ne_bytes(v.try_into().unwrap()) != 0));
            } else {
                assert_eq!(row.len(), 16);
                assert!(row
                    .chunks_exact(8)
                    .all(|v| u64::from_le_bytes(v.try_into().unwrap()) != 0));
            }
        }
    }
    if kind == AttentionKind::Causal {
        for (left, right) in observations[0]
            .binding_rows
            .as_ref()
            .unwrap()
            .iter()
            .zip(observations[1].binding_rows.as_ref().unwrap())
        {
            // The participant index belongs to each runtime's own canonical
            // layout and was checked above. Compare the other five controls,
            // including numerical status, without rewriting either raw row.
            assert_eq!(&left[..16], &right[..16]);
            assert_eq!(&left[20..24], &right[20..24]);
        }
    }
    // Pointer payloads intentionally differ between independent allocations.
    // These are real post-upload row readbacks, not a same-wave host-byte oracle.
}

fn compare(kind: AttentionKind) {
    let modes = [
        InvocationPreparationStrategy::IdentityProjection,
        InvocationPreparationStrategy::DecodeSegment,
    ];
    let maximum_tokens = 8192;
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_segment_oracle(
            kind,
            true,
            2,
            Family::new(kind).with_maximum_tokens(maximum_tokens),
            ProgramBindingUploadStrategy::Sparse,
            if mode == InvocationPreparationStrategy::DecodeSegment {
                SegmentBindingOracleMode::CompareReference
            } else {
                SegmentBindingOracleMode::Disabled
            },
        )
    });
    let lead = if kind == AttentionKind::Causal { 60 } else { 0 };
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
    let mut sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(index, t)| {
                fixture.admit_with_ceiling(
                    &format!("segment-original-{index}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    let sinks = [Preparation::default(), Preparation::default()];
    for position in 0..lead {
        let outputs = fixtures
            .iter()
            .zip(&sessions)
            .enumerate()
            .map(|(arm, (fixture, group))| {
                fixture.extend(&group[0], Arc::from(&tokens[0][..=position]));
                fixture.execute_participants_with_preparation(
                    &fixture.lane,
                    &fixture.reaper,
                    &group[..1],
                    &tokens[..1],
                    position..position + 1,
                    Path::Warm,
                    modes[arm],
                    &sinks[arm],
                )
            })
            .collect::<Vec<_>>();
        outputs[0].assert_same(&outputs[1]);
    }
    let row_bytes = if kind == AttentionKind::Causal {
        24 + (maximum_tokens.div_ceil(16) * 8).div_ceil(16) as usize * 16
    } else {
        16
    };
    let mut hot_before_growth = 0;
    let mut hot_after_growth = 0;
    let mut publications = 0;
    let mut previous_frames: [Vec<ExecutionFrameId>; 2] = [Vec::new(), Vec::new()];
    for position in 0..joint_waves {
        let ranges = [lead + position..lead + position + 1, position..position + 1];
        let changed_extent = kind == AttentionKind::Causal && position == 4;
        let path = if position < 2 || (kind == AttentionKind::Causal && (4..6).contains(&position))
        {
            Path::Warm
        } else {
            Path::Replay
        };
        let mut observations = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&sessions).enumerate() {
            for ((session, t), range) in group.iter().zip(&tokens).zip(&ranges) {
                fixture.extend(session, Arc::from(&t[..range.end]));
            }
            let pages = if kind == AttentionKind::Causal {
                let pages = group
                    .iter()
                    .zip(&tokens)
                    .zip(&ranges)
                    .map(|((session, t), range)| {
                        physical_pages(fixture, session, Arc::from(&t[..range.end]), range.end)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pages, if position < 4 { vec![1, 1] } else { vec![2, 1] });
                Some(pages)
            } else {
                None
            };
            let audited_before = fixture
                ._composition
                ._runtime
                .segment_binding_oracle_audited_nodes();
            let observation = fixture.execute_participant_ranges_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                group,
                &tokens,
                &ranges,
                path,
                modes[arm],
                &sinks[arm],
                Some(row_bytes),
            );
            let stats = sinks[arm].last();
            assert_eq!(
                fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes()
                    - audited_before,
                stats.segment_encoded_nodes,
                "every successful hot node must pass the same-wave byte oracle"
            );
            assert_eq!(stats.parts_materialized, 0);
            if arm == 0 {
                assert_eq!(
                    (
                        stats.segment_hits,
                        stats.segment_misses,
                        stats.segment_encoded_nodes
                    ),
                    (0, 0, 0)
                );
                assert!(!observation.segment_published);
            } else {
                assert!(stats.segment_hits + stats.segment_misses > 0);
                if stats.segment_hits > 0 {
                    assert!(
                        stats.segment_encoded_nodes >= 2,
                        "attention and dependency-only FFN must both use the whole-segment encoder"
                    );
                    assert!(
                        stats.segment_dynamic_resource_requests > 0
                            && stats.segment_unique_physical_buffers > 0
                    );
                }
                if position < 4 {
                    hot_before_growth += stats.segment_hits;
                } else {
                    hot_after_growth += stats.segment_hits;
                }
                publications += u64::from(observation.segment_published);
                if changed_extent {
                    assert_eq!(
                        stats.segment_hits, 0,
                        "new physical KV extent cannot consume the previous exact program recipe"
                    );
                    assert!(stats.segment_misses > 0);
                }
            }
            if !previous_frames[arm].is_empty() {
                assert!(observation
                    .participant_frames
                    .iter()
                    .zip(&previous_frames[arm])
                    .all(|(current, previous)| current > previous));
            }
            previous_frames[arm] = observation.participant_frames.clone();
            println!(
                "{}",
                serde_json::json!({"kind":"decode_segment_wave", "attention":format!("{kind:?}"), "strategy":modes[arm],
                "ranges":ranges, "physical_kv_pages":pages, "published":observation.segment_published,
                "hits":stats.segment_hits, "misses":stats.segment_misses, "encoded_nodes":stats.segment_encoded_nodes,
                "parts_materialized":stats.parts_materialized})
            );
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
        check_binding_controls(kind, &observations, &ranges);
    }
    assert!(
        hot_before_growth > 0,
        "must observe a true hot wave before the page-growth boundary"
    );
    assert!(hot_after_growth > 0);
    assert!(publications >= if kind == AttentionKind::Causal { 2 } else { 1 });

    // Drop the original sessions and admit different requests on the same
    // immutable Plan/lane. No cold recipe may retain their request/state owners.
    let old_authorities = sessions.each_ref().map(|group| {
        group
            .iter()
            .map(|session| session.sequence_authority())
            .collect::<Vec<_>>()
    });
    for group in &mut sessions {
        for session in group.drain(..) {
            session.try_complete().unwrap();
        }
    }
    let fresh_tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|participant| {
            (0..8)
                .map(|position| ((position * 5 + participant + 11) % 32) as u32)
                .collect()
        })
        .collect();
    let fresh = fixtures.each_ref().map(|fixture| {
        fresh_tokens
            .iter()
            .enumerate()
            .map(|(index, t)| {
                fixture.admit_with_ceiling(
                    &format!("segment-fresh-{index}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    let mut fresh_hits = 0;
    for position in 0..8 {
        let mut outputs = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&fresh).enumerate() {
            for (index, (session, t)) in group.iter().zip(&fresh_tokens).enumerate() {
                assert_ne!(session.sequence_authority(), old_authorities[arm][index]);
                fixture.extend(session, Arc::from(&t[..=position]));
            }
            let output = fixture.execute_participants_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                group,
                &fresh_tokens,
                position..position + 1,
                if position < 2 {
                    Path::Warm
                } else {
                    Path::Replay
                },
                modes[arm],
                &sinks[arm],
            );
            let stats = sinks[arm].last();
            assert_eq!(stats.parts_materialized, 0);
            if arm == 1 {
                fresh_hits += stats.segment_hits;
            }
            outputs.push(output);
        }
        outputs[0].assert_same(&outputs[1]);
    }
    assert!(
        fresh_hits > 0,
        "fresh sessions must also reach checked whole-segment reuse"
    );
    for group in fresh {
        for session in group {
            session.try_complete().unwrap();
        }
    }
    assert_eq!(
        fixtures[0]
            ._composition
            ._runtime
            .segment_binding_oracle_audited_nodes(),
        0
    );
    let audited_nodes = fixtures[1]
        ._composition
        ._runtime
        .segment_binding_oracle_audited_nodes();
    assert!(audited_nodes > 0);
    println!(
        "{}",
        serde_json::json!({"kind":"decode_segment_actual_cuda", "attention":format!("{kind:?}"),
        "participants":2, "lead_waves":lead, "joint_waves":joint_waves, "fresh_request_waves":8,
        "hot_before_extent_change":hot_before_growth, "hot_after_extent_change":hot_after_growth,
        "publications":publications, "fresh_request_hits":fresh_hits, "full_output_and_valid_state_equal":true,
        "same_wave_oracle_audited_nodes":audited_nodes, "binding_readback_scope":"exact current control/status bytes and nonzero pointers", "same_wave_oracle_scope":"all encoded patch bytes/owners/ranges/status/attribution and fresh Plan dependency identities; reference never submitted"})
    );
}

#[test]
#[ignore = "requires exclusive CUDA and actual whole-segment publication/replay"]
fn decode_segment_gdn_matches_ip_across_fresh_frames_and_requests() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA and actual whole-segment publication/replay"]
fn decode_segment_causal_matches_ip_across_hot_physical_kv_extension() {
    compare(AttentionKind::Causal);
}
