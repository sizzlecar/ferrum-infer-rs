//! Actual resident A/B/A programs share a lane while every wave keeps fresh
//! authorization. Both arms retain the existing Sparse numerical policy.
use super::*;

fn compare_resident_programs(kind: AttentionKind) {
    let modes = [
        InvocationPreparationStrategy::IdentityProjection,
        InvocationPreparationStrategy::DecodeSegment,
    ];
    let maximum_tokens = 8192_u64;
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_segment_oracle(
            kind,
            true,
            4,
            Family::new(kind).with_maximum_tokens(maximum_tokens),
            ProgramBindingUploadStrategy::Sparse,
            if mode == InvocationPreparationStrategy::DecodeSegment {
                SegmentBindingOracleMode::CompareReference
            } else {
                SegmentBindingOracleMode::Disabled
            },
        )
    });
    // Return to A at frontier 64, then grow its physical KV on the next wave.
    // This separates the first A/B/A hit from the physical-growth assertion.
    let lead = if kind == AttentionKind::Causal { 53 } else { 0 };
    let widths = [4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4];
    let tokens: Vec<Arc<[u32]>> = (0..4)
        .map(|participant| {
            let joint = if participant < 2 { 14 } else { 10 };
            let count = joint + if participant == 0 { lead } else { 0 };
            (0..count)
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
                    &format!("multi-recipe-original-{index}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    let sinks = [Preparation::default(), Preparation::default()];
    for position in 0..lead {
        let observations = fixtures
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
                    InvocationPreparationStrategy::IdentityProjection,
                    &sinks[arm],
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
    let mut program_a: [Option<DeviceReusableExecutionProgramId>; 2] = [None, None];
    let mut program_b: [Option<DeviceReusableExecutionProgramId>; 2] = [None, None];
    let mut entry_a: [Option<DeviceReusableExecutionEntryIdentity>; 2] = [None, None];
    let mut previous_frames: [Vec<Option<ExecutionFrameId>>; 2] = [vec![None; 4], vec![None; 4]];
    let mut published = [false; 2];
    let mut hot = [false; 2];
    let mut returned_hits = 0;
    for (wave, width) in widths.into_iter().enumerate() {
        let ranges: Vec<_> = frontiers[..width].iter().map(|p| *p..*p + 1).collect();
        let path = if matches!(wave, 0 | 1 | 6 | 7) {
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
                let pages = group[..width]
                    .iter()
                    .zip(&tokens)
                    .zip(&ranges)
                    .map(|((session, t), range)| {
                        physical_pages(fixture, session, Arc::from(&t[..range.end]), range.end)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pages[0], if wave < 11 { 1 } else { 2 });
                assert!(pages[1..].iter().all(|pages| *pages == 1));
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
                &group[..width],
                &tokens[..width],
                &ranges,
                path,
                modes[arm],
                &sinks[arm],
                Some(row_bytes),
            );
            let stats = sinks[arm].last();
            let program = observation.reusable_program_id.as_ref().unwrap();
            assert_eq!(program.immediate_sequences(), width as u32);
            assert_eq!(program.immediate_pages(), 0);
            let which = usize::from(width == 2);
            if wave == 5 {
                program_a[arm] = Some(program.clone());
            }
            if wave == 9 {
                program_b[arm] = Some(program.clone());
                assert_ne!(program_a[arm].as_ref().unwrap(), program);
            }
            if wave >= 10 {
                assert_eq!(program, program_a[arm].as_ref().unwrap());
                assert_ne!(program, program_b[arm].as_ref().unwrap());
            }
            let audited = fixture
                ._composition
                ._runtime
                .segment_binding_oracle_audited_nodes()
                - audited_before;
            assert_eq!(audited, stats.segment_encoded_nodes);
            assert_eq!(stats.parts_materialized, 0);
            if arm == 0 {
                assert_eq!((stats.segment_hits, stats.segment_misses), (0, 0));
                assert!(!observation.segment_published);
            } else {
                published[which] |= observation.segment_published;
                hot[which] |= stats.segment_hits > 0;
                if stats.segment_hits > 0 {
                    assert!(stats.segment_encoded_nodes >= 2);
                    assert!(stats.segment_dynamic_resource_requests > 0);
                    assert!(stats.segment_unique_physical_buffers > 0);
                }
                if wave >= 10 {
                    assert!(published.into_iter().all(|value| value));
                    assert!(hot.into_iter().all(|value| value));
                    assert_eq!((stats.segment_hits, stats.segment_misses), (1, 0));
                    assert!(!observation.segment_published);
                    returned_hits += stats.segment_hits;
                }
            }
            if wave == 5 {
                entry_a[arm] = fixture
                    .lane
                    .reusable_execution_entry_identity(program)
                    .unwrap();
                assert!(entry_a[arm].is_some());
            }
            if wave >= 10 {
                let entry = fixture
                    .lane
                    .reusable_execution_entry_identity(program)
                    .unwrap()
                    .unwrap();
                assert!(entry_a[arm].as_ref().unwrap().same_entry(&entry));
            }
            for (participant, frame) in observation.participant_frames.iter().enumerate() {
                if let Some(previous) = previous_frames[arm][participant] {
                    assert!(*frame > previous);
                }
                previous_frames[arm][participant] = Some(*frame);
            }
            observation.dump(kind, width as u32, ranges[0].clone(), path);
            println!(
                "{}",
                serde_json::json!({"kind":"multi_segment_recipe_wave", "attention":format!("{kind:?}"),
                "wave":wave, "strategy":modes[arm], "ranges":ranges, "physical_kv_pages":pages,
                "program_id":program, "returned_to_a":wave>=10, "published":observation.segment_published,
                "hits":stats.segment_hits, "misses":stats.segment_misses,
                "encoded_nodes":stats.segment_encoded_nodes, "same_wave_oracle_audited_nodes":audited})
            );
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
        check_binding_controls(kind, &observations, &ranges);
        for frontier in &mut frontiers[..width] {
            *frontier += 1;
        }
    }

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
    let fresh_tokens: Vec<Arc<[u32]>> = (0..4)
        .map(|participant| {
            (0..6)
                .map(|p| ((p * 5 + participant + 11) % 32) as u32)
                .collect()
        })
        .collect();
    let fresh = fixtures.each_ref().map(|fixture| {
        fresh_tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("multi-recipe-fresh-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    let mut fresh_hits = 0;
    for position in 0..6 {
        let mut observations = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&fresh).enumerate() {
            for (p, (session, t)) in group.iter().zip(&fresh_tokens).enumerate() {
                assert_ne!(session.sequence_authority(), old_authorities[arm][p]);
                fixture.extend(session, Arc::from(&t[..=position]));
            }
            let before = fixture
                ._composition
                ._runtime
                .segment_binding_oracle_audited_nodes();
            let observation = fixture.execute_participants_with_preparation(
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
            assert_eq!(
                fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes()
                    - before,
                stats.segment_encoded_nodes
            );
            if arm == 1 {
                fresh_hits += stats.segment_hits;
            }
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
    }
    assert!(fresh_hits > 0);
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
    println!(
        "{}",
        serde_json::json!({"kind":"multi_segment_recipe_actual_cuda",
        "attention":format!("{kind:?}"), "participant_widths":widths, "lead_ip_waves":lead,
        "a_and_b_published":published, "a_and_b_hot":hot,
        "return_a_hits_without_publication":returned_hits, "fresh_request_waves":6, "fresh_request_hits":fresh_hits,
        "full_output_and_valid_state_equal":true,
        "same_wave_oracle_audited_nodes":fixtures[1]._composition._runtime.segment_binding_oracle_audited_nodes()})
    );
}

#[test]
#[ignore = "requires exclusive CUDA and actual alternating resident programs"]
fn multi_segment_recipes_gdn_reuses_a_after_b_with_fresh_authority() {
    compare_resident_programs(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA, actual resident programs and physical KV growth"]
fn multi_segment_recipes_causal_reuses_a_after_b_with_fresh_kv() {
    compare_resident_programs(AttentionKind::Causal);
}
