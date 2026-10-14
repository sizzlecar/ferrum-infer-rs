//! Actual Q6 provider execution; the primitive F64/marker oracle lives in the
//! native probe. These gates compare one numerical profile across IP and DS.
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
    let family = Family::q6_f16(kind);
    let mut eligible = 0;
    let mut strict = 0;
    let mut middle_routes = 0;
    let mut joined_banks = BTreeSet::new();
    for (name, profile) in [
        ("node.attention", family.attention_profile()),
        ("node.swiglu", family.swiglu_profile()),
    ] {
        let node = fixture
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|n| n.id().as_str() == name)
            .unwrap();
        assert_eq!(node.operation_id().as_str(), profile.operation_id());
        let prepared = node.provider_resources().projection_numerics().unwrap();
        assert_eq!(prepared.contract(), &profile.arithmetic());
        for projection in prepared.projections() {
            for leaf in projection.leaves() {
                let WeightEncoding::BlockQuantized(block) = leaf.encoding() else {
                    continue;
                };
                assert_eq!(block.format_id.as_str(), "quantization.gguf.q6-k");
                assert_eq!(block.bytes_per_block, 210);
                let k = projection.input_features();
                let n = leaf.output_features();
                let declaration = prepared
                    .contract()
                    .declared_projection_arithmetic(
                        projection.role(),
                        Some(block),
                        k,
                        n,
                        leaf.has_weight_transform(),
                    )
                    .unwrap();
                let selected = match declaration {
                    DeclaredProjectionArithmetic::Staged(staged) => {
                        assert!(leaf.is_staged());
                        staged.upstream_policy().unwrap().select_arithmetic(
                            UpstreamProjectionLayout::Columns,
                            participants,
                            k,
                            n,
                        )
                    }
                    DeclaredProjectionArithmetic::StrictBase(_) => {
                        assert!(!leaf.is_staged());
                        None
                    }
                };
                let in_domain = k >= 5120
                    && k % 256 == 0
                    && n >= 1024
                    && (4..=32).contains(&participants)
                    && (participants >= 8 || n >= 5120);
                assert_eq!(
                    selected,
                    in_domain.then_some(UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1)
                );
                if in_domain {
                    eligible += 1;
                } else {
                    strict += 1;
                }
                if k >= 5120 && (1024..5120).contains(&n) {
                    middle_routes += 1;
                    assert_eq!(selected.is_some(), participants >= 8);
                }
                if projection.role() == ProjectionRole::SwiGluGateUp {
                    assert_eq!(
                        projection.output_features(),
                        2 * family::q6_f16::INTERMEDIATE
                    );
                    if n == 5120 {
                        joined_banks.insert(leaf.output_offset() / family::q6_f16::INTERMEDIATE);
                        assert!(selected.is_some(), "large leaf selected at both M7 and M8");
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"q6_f16_prepared_leaf_selection",
                    "attention":format!("{kind:?}"),"node":name,"role":projection.role(),
                    "component":leaf.component_id(),"m":participants,"k":k,"leaf_n":n,
                    "output_offset":leaf.output_offset(),"output_stride":projection.output_features(),
                    "selected":selected,"strict_fallback":selected.is_none(),
                    "scope":"actual prepared physical leaf plus shared selector; CUDA encode validates the native plan, not a launch trace"})
                );
            }
        }
    }
    assert!(eligible > 0 && strict > 0 && middle_routes > 0);
    assert_eq!(joined_banks, BTreeSet::from([0, 1]));
}

fn compare(kind: AttentionKind) {
    compare_family(
        kind,
        Family::q6_f16,
        InvocationPreparationStrategy::IdentityProjection,
        assert_selected_routes,
        "q6_f16",
    );
}

#[test]
#[ignore = "requires actual Q6 F16 providers and full recurrent state"]
fn q6_f16_gdn_joined_ffn_ip_matches_segment_across_widths_and_requests() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires actual Q6 F16 providers and physical FP16 KV growth"]
fn q6_f16_causal_joined_ffn_ip_matches_segment_across_widths_and_kv_growth() {
    compare(AttentionKind::Causal);
}

fn compare_family(
    kind: AttentionKind,
    family: fn(AttentionKind) -> Family,
    reference: InvocationPreparationStrategy,
    check_routes: fn(&Fixture, AttentionKind, u32),
    label: &str,
) {
    let profile = family(kind).profile_id();
    let modes = [
        reference,
        InvocationPreparationStrategy::DecodeSegmentProgram,
    ];
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_causal_decode_preparation(
            kind,
            true,
            8,
            family(kind),
            ProgramBindingUploadStrategy::CompactScatter,
            if mode == InvocationPreparationStrategy::DecodeSegmentProgram {
                SegmentBindingOracleMode::CompareReference
            } else {
                SegmentBindingOracleMode::Disabled
            },
            SegmentBindingOwnerViewMode::Legacy,
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ferrum_types::CausalDecodePreparationMode::Packed,
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
                    &format!("{label}-original-{p}"),
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
            check_routes(fixture, kind, width as u32);
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
                serde_json::json!({"kind":format!("{label}_segment_wave"), "attention":format!("{kind:?}"),"strategy":modes[arm],"wave":position,"participants":width,"ranges":ranges,"physical_kv_pages":pages,"program":program,"hits":stats.segment_hits,"misses":stats.segment_misses,"encoded_nodes":stats.segment_encoded_nodes,"published":observation.segment_published})
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
                    &format!("{label}-fresh-{p}"),
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
            check_routes(fixture, kind, 8);
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
        serde_json::json!({"kind":format!("{label}_segment_complete"), "attention":format!("{kind:?}"),"fixture_profile":profile,"hidden":family::q6_f16::WIDE_HIDDEN,"ffn_input_gain":1.0 / 16.0,"widths":widths,"hits_by_width":hits_by_width,"publications":publications,"fresh_hits":fresh_hits,"same_wave_oracle_audited_nodes":audited,"full_output_and_state_equal":true,"qualification_scope":"declared operation contracts on synthetic provider family; Qwen profile/head eligibility and full-model quality are separate"})
    );
}
