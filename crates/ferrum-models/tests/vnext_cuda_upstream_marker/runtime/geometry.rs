//! Wide, synthetic provider histories. These use the new attention arithmetic,
//! not the Qwen family ID or its Q6-head/full-model quality qualification.
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
        *self.0.borrow().last().expect("actual dispatch preparation")
    }
}

fn assert_selected_routes(fixture: &Fixture, kind: AttentionKind, participants: u32) {
    let selected = Family::m8_geometry(kind).attention_profile();
    let inherited = match kind {
        AttentionKind::GatedDelta => UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
        AttentionKind::Causal => UpstreamMarkerV2Profile::CausalExtraAllRows,
        _ => unreachable!(),
    }
    .arithmetic();
    let node = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|n| n.id().as_str() == "node.attention")
        .unwrap();
    assert_eq!(node.operation_id().as_str(), selected.operation_id());
    let prepared = node.provider_resources().projection_numerics().unwrap();
    assert_eq!(prepared.contract(), &selected.arithmetic());
    let mut mixed_projection = false;
    for projection in prepared.projections() {
        let mut large_formats = BTreeSet::new();
        let mut small_q4 = false;
        for leaf in projection.leaves() {
            let WeightEncoding::BlockQuantized(block) = leaf.encoding() else {
                continue;
            };
            let DeclaredProjectionArithmetic::Staged(staged) = prepared
                .contract()
                .declared_projection_arithmetic(
                    projection.role(),
                    Some(block),
                    projection.input_features(),
                    leaf.output_features(),
                    leaf.has_weight_transform(),
                )
                .unwrap()
            else {
                continue;
            };
            assert!(leaf.is_staged());
            let policy = staged
                .upstream_policy()
                .expect("actual prepared upstream leaf");
            let route = policy
                .select_arithmetic(
                    UpstreamProjectionLayout::Columns,
                    participants,
                    projection.input_features(),
                    leaf.output_features(),
                )
                .unwrap();
            if matches!(
                block.format_id.as_str(),
                "quantization.gguf.q4-k" | "quantization.gguf.q5-k"
            ) {
                let large = projection.input_features() >= 5120 && leaf.output_features() >= 5120;
                let DeclaredProjectionArithmetic::Staged(previous) = inherited
                    .declared_projection_arithmetic(
                        projection.role(),
                        Some(block),
                        projection.input_features(),
                        leaf.output_features(),
                        leaf.has_weight_transform(),
                    )
                    .unwrap()
                else {
                    panic!("geometry must preserve the inherited staged domain")
                };
                let inherited_route = previous
                    .upstream_policy()
                    .unwrap()
                    .select_arithmetic(
                        UpstreamProjectionLayout::Columns,
                        participants,
                        projection.input_features(),
                        leaf.output_features(),
                    )
                    .unwrap();
                assert_eq!(
                    route,
                    if participants == 8 && large {
                        UpstreamProjectionArithmetic::MmqDs4MarkerV2
                    } else {
                        // In particular M7 already selects MMQ for both large
                        // and small leaves. Only eligible M8 changes policy.
                        inherited_route
                    }
                );
                if large {
                    large_formats.insert(block.format_id.as_str());
                }
                if block.format_id.as_str() == "quantization.gguf.q4-k" && !large {
                    small_q4 = true;
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"geometry_prepared_leaf_selection", "attention":format!("{kind:?}"), "physical_component":leaf.component_id(), "format":block.format_id, "m":participants, "k":projection.input_features(), "leaf_n":leaf.output_features(), "projection_n":projection.output_features(), "selected":route,"inherited_selected":inherited_route,
                    "scope":"compiled physical leaf plus shared selector; successful CUDA encode validates native geometry, not an observed launch trace"})
                );
            }
        }
        mixed_projection |= large_formats.len() == 2 && small_q4;
    }
    assert!(
        mixed_projection,
        "one real projection must contain both eligible formats and an ineligible small leaf"
    );
}

fn compare(kind: AttentionKind) {
    let modes = [
        InvocationPreparationStrategy::Full,
        InvocationPreparationStrategy::DecodeSegment,
    ];
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_segment_oracle(
            kind,
            true,
            8,
            Family::m8_geometry(kind),
            ProgramBindingUploadStrategy::Sparse,
            if mode == InvocationPreparationStrategy::DecodeSegment {
                SegmentBindingOracleMode::CompareReference
            } else {
                SegmentBindingOracleMode::Disabled
            },
        )
    });
    let sinks = [Preparation::default(), Preparation::default()];
    let lead = if kind == AttentionKind::Causal { 60 } else { 0 };
    // The width change has its own cold/warm program. Returning to eight must
    // still validate current frames, even if its previous recipe remains cached.
    let widths = [8usize, 8, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8];
    let tokens: Vec<Arc<[u32]>> = (0..8)
        .map(|p| {
            (0..lead + widths.len())
                .map(|position| ((position * 7 + p * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let mut sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("geometry-original-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
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
                    modes[arm],
                    &sinks[arm],
                )
            })
            .collect::<Vec<_>>();
        observations[0].assert_same(&observations[1]);
    }
    let row_bytes = if kind == AttentionKind::Causal {
        24 + (128u64.div_ceil(16) * 8).div_ceil(16) as usize * 16
    } else {
        16
    };
    let mut next_positions = [0usize; 8];
    next_positions[0] = lead;
    let mut width_position = 0;
    let mut last_width = 0;
    let mut hits_by_width = BTreeMap::<usize, u64>::new();
    let mut publications = 0;
    let mut prior_programs: [Option<DeviceReusableExecutionProgramId>; 2] = [None, None];
    let mut prior_frames: [Vec<Option<ExecutionFrameId>>; 2] = [vec![None; 8], vec![None; 8]];
    let mut initial_eight_programs: [Option<DeviceReusableExecutionProgramId>; 2] = [None, None];
    for (position, &width) in widths.iter().enumerate() {
        if width != last_width {
            width_position = 0;
        }
        let ranges = next_positions[..width]
            .iter()
            .map(|&n| n..n + 1)
            .collect::<Vec<_>>();
        let path = if width_position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let mut observations = Vec::new();
        for (arm, (fixture, group)) in fixtures.iter().zip(&sessions).enumerate() {
            assert_selected_routes(fixture, kind, width as u32);
            for ((session, t), range) in group[..width].iter().zip(&tokens).zip(&ranges) {
                fixture.extend(session, Arc::from(&t[..range.end]));
            }
            let pages = if kind == AttentionKind::Causal {
                let pages = group[..width]
                    .iter()
                    .zip(&tokens)
                    .zip(&ranges)
                    .map(|((session, t), range)| {
                        decode_segment::physical_pages(
                            fixture,
                            session,
                            Arc::from(&t[..range.end]),
                            range.end,
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pages[0], if position < 4 { 1 } else { 2 });
                assert!(pages[1..].iter().all(|&p| p == 1));
                Some(pages)
            } else {
                None
            };
            let audited = fixture
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
            let program = observation
                .reusable_program_id
                .as_ref()
                .expect("actual single-token program");
            assert_eq!(program.immediate_pages(), 0);
            if position == 0 {
                initial_eight_programs[arm] = Some(program.clone());
            }
            if position == 6 {
                assert_ne!(Some(program), initial_eight_programs[arm].as_ref());
            }
            if position == 10 {
                assert_eq!(Some(program), initial_eight_programs[arm].as_ref());
            }
            if kind == AttentionKind::Causal && position == 4 {
                assert_eq!(
                    Some(program),
                    prior_programs[arm].as_ref(),
                    "physical KV growth keeps the actual topology key"
                );
                if arm == 1 {
                    assert_eq!(stats.segment_hits, 1);
                    assert_eq!(stats.segment_misses, 0);
                    assert!(!observation.segment_published);
                }
            }
            assert_eq!(
                fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes()
                    - audited,
                stats.segment_encoded_nodes
            );
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
                assert_eq!(stats.parts_materialized, 0);
                assert!(stats.segment_hits + stats.segment_misses > 0);
                *hits_by_width.entry(width).or_default() += stats.segment_hits;
                publications += u64::from(observation.segment_published);
                if width_position >= 3 {
                    assert_eq!(
                        stats.segment_hits, 1,
                        "a fully warmed width must execute the checked segment"
                    );
                    assert!(stats.segment_encoded_nodes >= 2);
                }
            }
            for (p, frame) in observation.participant_frames.iter().enumerate() {
                if let Some(previous) = &prior_frames[arm][p] {
                    assert!(frame > previous);
                }
                prior_frames[arm][p] = Some(frame.clone());
            }
            prior_programs[arm] = Some(program.clone());
            println!(
                "{}",
                serde_json::json!({"kind":"geometry_segment_wave", "attention":format!("{kind:?}"),"strategy":modes[arm],"wave":position,"participants":width,"ranges":ranges,"physical_kv_pages":pages,"program":program,"hits":stats.segment_hits,"misses":stats.segment_misses,"encoded_nodes":stats.segment_encoded_nodes,"published":observation.segment_published})
            );
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
        decode_segment::check_binding_controls(kind, &observations, &ranges);
        for position in &mut next_positions[..width] {
            *position += 1;
        }
        width_position += 1;
        last_width = width;
    }
    assert!(hits_by_width[&8] > 0 && hits_by_width[&7] > 0);
    assert!(
        publications >= 2,
        "distinct seven/eight programs must publish independently"
    );
    let old_authorities = sessions.each_ref().map(|group| {
        group
            .iter()
            .map(|s| s.sequence_authority())
            .collect::<Vec<_>>()
    });
    for group in &mut sessions {
        for session in group.drain(..) {
            session.try_complete().unwrap();
        }
    }
    let fresh_tokens: Vec<Arc<[u32]>> = (0..8)
        .map(|p| (0..6).map(|i| ((i * 5 + p + 11) % 32) as u32).collect())
        .collect();
    let fresh = fixtures.each_ref().map(|fixture| {
        fresh_tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("geometry-fresh-{p}"),
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
            assert_selected_routes(fixture, kind, 8);
            for (p, (session, t)) in group.iter().zip(&fresh_tokens).enumerate() {
                assert_ne!(session.sequence_authority(), old_authorities[arm][p]);
                fixture.extend(session, Arc::from(&t[..=position]));
            }
            let audited = fixture
                ._composition
                ._runtime
                .segment_binding_oracle_audited_nodes();
            let observation = fixture.execute_participant_ranges_with_preparation(
                &fixture.lane,
                &fixture.reaper,
                group,
                &fresh_tokens,
                &vec![position..position + 1; 8],
                if position < 2 {
                    Path::Warm
                } else {
                    Path::Replay
                },
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
                    - audited,
                stats.segment_encoded_nodes
            );
            if arm == 1 {
                fresh_hits += stats.segment_hits;
                assert_eq!(stats.parts_materialized, 0);
            }
            observations.push(observation);
        }
        observations[0].assert_same(&observations[1]);
        decode_segment::check_binding_controls(
            kind,
            &observations,
            &vec![position..position + 1; 8],
        );
    }
    assert!(fresh_hits > 0);
    for group in fresh {
        for session in group {
            session.try_complete().unwrap();
        }
    }
    let audited = fixtures[1]
        ._composition
        ._runtime
        .segment_binding_oracle_audited_nodes();
    assert!(audited > 0);
    assert_eq!(
        fixtures[0]
            ._composition
            ._runtime
            .segment_binding_oracle_audited_nodes(),
        0
    );
    println!(
        "{}",
        serde_json::json!({"kind":"geometry_segment_complete", "attention":format!("{kind:?}"),"fixture_profile":family::geometry::PROFILE,"hidden":family::geometry::WIDE_HIDDEN,"ffn_input_gain":family::geometry::FfnInputGain::OneSixteenth.multiplier(),"widths":widths,"hits_by_width":hits_by_width,"publications":publications,"fresh_hits":fresh_hits,"same_wave_oracle_audited_nodes":audited,"full_output_and_state_equal":true,"qualification_scope":"new attention operation contracts on synthetic provider family; Qwen profile/head eligibility and full-model quality are separate"})
    );
}

#[test]
#[ignore = "requires exclusive CUDA with geometry native declarations and wide stateful plans"]
fn geometry_gdn_m8_mixed_leaves_full_matches_segment_across_widths_and_requests() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA with geometry native declarations and actual KV growth"]
fn geometry_causal_m8_mixed_leaves_full_matches_segment_across_widths_and_kv_growth() {
    compare(AttentionKind::Causal);
}

#[test]
#[ignore = "requires exclusive CUDA; compares first-wave finite values with bounded synthetic FFN input"]
fn geometry_gdn_wide_first_wave_finite_isolation() {
    use family::geometry::{FfnInputGain, WideArithmetic};
    let ffn_input_gain = FfnInputGain::OneSixteenth;
    let kind = AttentionKind::GatedDelta;
    let policies = [
        WideArithmetic::InheritedExtraAllRows,
        WideArithmetic::Geometry,
    ];
    let schemas = policies.map(|policy| {
        Family::wide_arithmetic(kind, policy)
            .weight_schema(&kind)
            .unwrap()
    });
    assert_eq!(
        schemas[0], schemas[1],
        "both arms use the same schema and deterministic weight generator"
    );
    let tokens: Vec<Arc<[u32]>> = (0..8)
        .map(|p| {
            (0..14)
                .map(|position| ((position * 7 + p * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let mut outputs = Vec::new();
    // First compare both declarations under the original liveness plan, then
    // compare the two
    // arithmetic declarations under the same explicitly retained-value plan.
    // If only retained runs are finite, that is an alias/liveness lead, not
    // evidence that scaling or the new arithmetic caused the original failure.
    for (policy, retained) in [
        (WideArithmetic::InheritedExtraAllRows, false),
        (WideArithmetic::Geometry, false),
        (WideArithmetic::InheritedExtraAllRows, true),
        (WideArithmetic::Geometry, true),
    ] {
        let definition = Family::wide_arithmetic(kind, policy).with_ffn_input_gain(ffn_input_gain);
        let fixture = Fixture::for_family_with_segment_oracle(
            kind,
            true,
            8,
            if retained {
                definition.with_intermediate_observation()
            } else {
                definition
            },
            ProgramBindingUploadStrategy::Sparse,
            SegmentBindingOracleMode::Disabled,
        );
        let sessions = tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("wide-isolation-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>();
        let sink = Preparation::default();
        let observation = fixture.execute_participant_ranges_observed(
            &fixture.lane,
            &fixture.reaper,
            &sessions,
            &tokens,
            &vec![0..1; 8],
            Path::Warm,
            InvocationPreparationStrategy::Full,
            &sink,
            None,
            if retained {
                batch::ObservationScope::RetainedIntermediates
            } else {
                batch::ObservationScope::OutputAndState
            },
        );
        println!(
            "{}",
            serde_json::json!({"kind":"geometry_first_wave_finite_isolation","arithmetic":format!("{policy:?}"),
            "participants":8,"range":[0,1],"strategy":"full","ffn_input_gain":ffn_input_gain.multiplier(),"packed_weight_bytes_changed":false,"attention_weight_bytes_changed":false,"retained_intermediate_values":retained,"profile":Family::wide_arithmetic(kind,policy).profile_id(),
            "values":observation.finite_summary()})
        );
        for session in sessions {
            session.try_complete().unwrap();
        }
        outputs.push(observation);
    }
    // All complete readbacks are recorded before a finite assertion can abort.
    // Different declared algorithms are not required to be bitwise equivalent.
    for output in outputs {
        output.assert_same(&output);
    }
}
