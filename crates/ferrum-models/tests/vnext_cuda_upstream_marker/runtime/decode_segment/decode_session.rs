//! Same numerical program, with and without cross-step resource reservations.
use super::super::super::batch::{BatchObservationOptions, DecodeSessionFixtureState};
use super::*;

fn compare_decode_sessions(kind: AttentionKind) {
    let maximum_tokens = 8192_u64;
    let fixtures = [false, true].map(|_| {
        Fixture::for_family_with_causal_decode_preparation(
            kind,
            true,
            4,
            Family::new(kind).with_maximum_tokens(maximum_tokens),
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::CompareReference,
            SegmentBindingOwnerViewMode::Indexed,
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ferrum_types::CausalDecodePreparationMode::Packed,
        )
    });
    let mut reservations = [
        DecodeSessionFixtureState::new(false),
        DecodeSessionFixtureState::new(true),
    ];
    let sinks = [Preparation::default(), Preparation::default()];
    // The first two four-participant waves end at 63 and 64. The next wave
    // changes actual physical KV capacity while a session already exists.
    let lead = if kind == AttentionKind::Causal { 62 } else { 0 };
    let tokens: Vec<Arc<[u32]>> = (0..4)
        .map(|p| {
            (0..lead + 12)
                .map(|i| ((i * 7 + p * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, tokens)| {
                fixture.admit_with_ceiling(
                    &format!("decode-session-{p}"),
                    Arc::from(&tokens[..1]),
                    tokens.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    for position in 0..lead {
        let observations = fixtures
            .iter()
            .zip(&sessions)
            .map(|(fixture, group)| {
                fixture.extend(&group[0], Arc::from(&tokens[0][..=position]));
                fixture.execute_participants_with_preparation(
                    &fixture.lane,
                    &fixture.reaper,
                    &group[..1],
                    &tokens[..1],
                    position..position + 1,
                    Path::Warm,
                    InvocationPreparationStrategy::DecodeSegmentProgram,
                    &Preparation::default(),
                )
            })
            .collect::<Vec<_>>();
        observations[0].assert_same(&observations[1]);
    }
    let row_bytes = if kind == AttentionKind::Causal {
        24 + (maximum_tokens.div_ceil(16) * 8).div_ceil(16) as usize * 16
    } else {
        16
    };
    let mut frontiers = [lead, 0, 0, 0];
    let widths = [4, 4, 4, 4, 2, 2, 4, 4];
    let mut previous_frames: [Vec<Option<ExecutionFrameId>>; 2] = [vec![None; 4], vec![None; 4]];
    let mut observed_prepared = Vec::new();
    let mut timing_totals = [[0_u64; 4]; 2];
    let mut fresh_timing_totals = [[0_u64; 4]; 2];
    for (wave, width) in widths.into_iter().enumerate() {
        let ranges: Vec<_> = frontiers[..width].iter().map(|p| *p..*p + 1).collect();
        let path = if matches!(wave, 0 | 1 | 4 | 5) {
            Path::Warm
        } else {
            Path::Replay
        };
        let mut observations: [Option<BatchObservation>; 2] = [None, None];
        let order = if wave % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let fixture = &fixtures[arm];
            let group = &sessions[arm][..width];
            for ((session, t), range) in group.iter().zip(&tokens).zip(&ranges) {
                fixture.extend(session, Arc::from(&t[..range.end]));
            }
            if kind == AttentionKind::Causal {
                assert_eq!(
                    physical_pages(
                        fixture,
                        &group[0],
                        Arc::from(&tokens[0][..ranges[0].end]),
                        ranges[0].end
                    ),
                    if wave < 2 { 1 } else { 2 }
                );
            }
            let audited_before = fixture
                ._composition
                ._runtime
                .segment_binding_oracle_audited_nodes();
            let physical_before = fixture
                ._composition
                ._runtime
                .physical_segment_binding_encodes();
            let uploads_before = fixture
                ._composition
                ._runtime
                .program_binding_upload_snapshot();
            let observation = fixture.execute_participant_ranges_with_decode_session(
                &fixture.lane,
                &fixture.reaper,
                group,
                &tokens[..width],
                &ranges,
                path,
                InvocationPreparationStrategy::DecodeSegmentProgram,
                &sinks[arm],
                Some(row_bytes),
                BatchObservationOptions::default(),
                Some(&mut reservations[arm]),
            );
            let stats = sinks[arm].last();
            check_compiled_upload(fixture, uploads_before, stats);
            assert_eq!(
                fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes()
                    - audited_before,
                stats.segment_encoded_nodes
            );
            assert_eq!(
                fixture
                    ._composition
                    ._runtime
                    .physical_segment_binding_encodes()
                    - physical_before,
                stats.segment_hits
            );
            for (p, frame) in observation.participant_frames.iter().enumerate() {
                if let Some(previous) = previous_frames[arm][p] {
                    assert!(*frame > previous);
                }
                previous_frames[arm][p] = Some(*frame);
            }
            let timing = observation.decode_session.as_ref().unwrap();
            if arm == 0 {
                assert!(!timing.prepared && !timing.reserved && !timing.fallback);
            } else {
                observed_prepared.push(timing.prepared);
                if matches!(wave, 4 | 6) {
                    assert!(timing.fallback, "changed membership must release the old reservation before ordinary preparation");
                }
                if matches!(wave, 2 | 3 | 5 | 7) {
                    assert!(
                        timing.prepared,
                        "this wave must actually use session-backed preparation"
                    );
                }
            }
            let values = [
                timing.resource_prepare_ns,
                timing.prepare_through_submit_ns,
                timing.retire_ns,
                timing.complete_helper_ns,
            ];
            for (total, elapsed) in timing_totals[arm].iter_mut().zip(values) {
                *total += elapsed;
            }
            println!(
                "{}",
                serde_json::json!({"kind":"decode_session_full_transaction", "attention":format!("{kind:?}"),
                "arm":arm,"wave":wave,"order":order,"participants":width,"ranges":ranges,
                "reserved":timing.reserved,"session_prepared":timing.prepared,"fallback":timing.fallback,
                "resource_prepare_ns":timing.resource_prepare_ns,"prepare_through_submit_ns":timing.prepare_through_submit_ns,
                "retire_ns":timing.retire_ns,"complete_helper_ns":timing.complete_helper_ns,
                "compiled_hits":stats.segment_hits,"compiled_misses":stats.segment_misses,"oracle_nodes":stats.segment_encoded_nodes,
                "timing_scope":"resource preparation includes canonical batch, token spans, reservation/ordinary admission and wave preparation; through-submit additionally includes fixture readback descriptor/oracle setup and dispatch; complete-helper includes GPU wait and full state reads; prior Sequence extension and physical-page inspection are excluded; nested values must not be summed"})
            );
            observations[arm] = Some(observation);
        }
        let observations = observations.map(Option::unwrap);
        observations[0].assert_same(&observations[1]);
        check_binding_controls(kind, &observations, &ranges);
        for frontier in &mut frontiers[..width] {
            *frontier += 1;
        }
    }
    assert!(observed_prepared
        .windows(2)
        .any(|pair| pair == [true, true]));
    assert_eq!(reservations[0].prepared_steps, 0);
    let original_prepared = reservations[1].prepared_steps;
    let old_authorities = sessions.each_ref().map(|group| {
        group
            .iter()
            .map(|s| s.sequence_authority())
            .collect::<Vec<_>>()
    });
    for state in &mut reservations {
        state.close();
    }
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }

    // A completed cohort must not keep the reserved lane slots or old request
    // authority alive. Admit different requests and exercise the same path.
    let fresh_tokens: Vec<Arc<[u32]>> = (0..4)
        .map(|p| (0..4).map(|i| ((i * 5 + p + 11) % 32) as u32).collect())
        .collect();
    let fresh = fixtures.each_ref().map(|fixture| {
        fresh_tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("decode-session-fresh-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    for position in 0..4 {
        let mut observations = Vec::new();
        for arm in 0..2 {
            for (p, (session, t)) in fresh[arm].iter().zip(&fresh_tokens).enumerate() {
                assert_ne!(session.sequence_authority(), old_authorities[arm][p]);
                fixtures[arm].extend(session, Arc::from(&t[..=position]));
            }
            let ranges = vec![position..position + 1; 4];
            observations.push(
                fixtures[arm].execute_participant_ranges_with_decode_session(
                    &fixtures[arm].lane,
                    &fixtures[arm].reaper,
                    &fresh[arm],
                    &fresh_tokens,
                    &ranges,
                    if position < 2 {
                        Path::Warm
                    } else {
                        Path::Replay
                    },
                    InvocationPreparationStrategy::DecodeSegmentProgram,
                    &sinks[arm],
                    Some(row_bytes),
                    BatchObservationOptions::default(),
                    Some(&mut reservations[arm]),
                ),
            );
            let timing = observations[arm].decode_session.as_ref().unwrap();
            if arm == 1 && position >= 2 {
                assert!(timing.prepared);
            }
            for (total, elapsed) in fresh_timing_totals[arm].iter_mut().zip([
                timing.resource_prepare_ns,
                timing.prepare_through_submit_ns,
                timing.retire_ns,
                timing.complete_helper_ns,
            ]) {
                *total += elapsed;
            }
            println!(
                "{}",
                serde_json::json!({"kind":"decode_session_fresh_transaction",
                "attention":format!("{kind:?}"),"arm":arm,"position":position,
                "reserved":timing.reserved,"session_prepared":timing.prepared,"fallback":timing.fallback,
                "resource_prepare_ns":timing.resource_prepare_ns,"prepare_through_submit_ns":timing.prepare_through_submit_ns,
                "retire_ns":timing.retire_ns,"complete_helper_ns":timing.complete_helper_ns})
            );
        }
        observations[0].assert_same(&observations[1]);
        check_binding_controls(kind, &observations, &vec![position..position + 1; 4]);
    }
    assert!(reservations[1].prepared_steps > original_prepared);
    for state in &mut reservations {
        state.close();
    }
    for group in fresh {
        for session in group {
            session.try_complete().unwrap();
        }
    }
    println!(
        "{}",
        serde_json::json!({"kind":"decode_session_complete_lifecycle",
        "attention":format!("{kind:?}"),"timing_totals_ns":timing_totals,
        "fresh_timing_totals_ns":fresh_timing_totals,
        "timing_columns":["resource_prepare","prepare_through_submit","retire","complete_helper"],
        "candidate_prepared_steps":reservations[1].prepared_steps,"candidate_reservations":reservations[1].reservations,
        "candidate_fallback_steps":reservations[1].fallback_steps,
        "baseline_close_ns":reservations[0].close_ns,"candidate_close_ns":reservations[1].close_ns,
        "full_output_state_kv_equal":true,"fresh_request_authorities":true,
        "scope":"actual four-participant single-layer attention plus FFN core; first totals cover original 8 waves, fresh totals cover 4 new-request waves; reservation cold/fallback included, model/fixture setup, Sequence extension, physical-page inspection and leading 62 causal waves excluded; close_ns includes fallback close already nested in prepare plus final teardown; never sum nested values; not all session lifecycle costs or a full-model throughput estimate"})
    );
}

#[test]
#[ignore = "requires exclusive CUDA and actual cross-step resource reservations"]
fn decode_session_gdn_matches_fresh_steps() {
    compare_decode_sessions(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA, cross-step reservations and physical KV growth"]
fn decode_session_causal_matches_fresh_steps_and_kv_growth() {
    compare_decode_sessions(AttentionKind::Causal);
}
