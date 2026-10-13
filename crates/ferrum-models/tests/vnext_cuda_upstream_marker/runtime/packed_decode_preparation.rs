//! Real Core admission and provider submission. The two independent histories
//! differ only in the selected preparation, grouping, execution bound, or Q4/Q5 weight decode; no fixture asserts KV independence
//! on the backend's behalf. Kernel attribution is a submission declaration,
//! not a hardware counter. Ordinary warm/capture/replay remains timing Off.
use super::*;
use batch::BatchObservationOptions;
use ferrum_types::{
    AttentionExecutionPolicy, CausalDecodeAttentionGrouping, CausalDecodePreparationMode,
    InvocationPreparationStrategy, ProgramBindingUploadStrategy, Q4MmvqWeightDecode,
    Q5MmvqWeightDecode,
};
use std::cell::RefCell;

const MAXIMUM_TOKENS: u64 = 1536;
const DECODE_WAVES: usize = 8;
const KV_BYTES_PER_TOKEN: usize = 1024;

#[derive(Default)]
struct Preparation(RefCell<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.borrow_mut().push(stats);
    }
}
impl Preparation {
    fn last(&self) -> InvocationPreparationStats {
        *self.0.borrow().last().unwrap()
    }
}

/// Same-owner comparison includes every previously readable byte, even the
/// invalid token positions inside an already authorized physical block. New
/// allocation padding is never compared between the two independent arms.
fn preserve_existing_kv(before: &[u8], after: &[u8], written: Range<usize>) {
    assert!(after.len() >= before.len());
    let mut writes = vec![false; before.len()];
    for token in written {
        let block = token / 16 * (2 * 2 * 128 * 16);
        for kv in 0..2 {
            for head in 0..2 {
                for dim in 0..128 {
                    let element = block
                        + if kv == 0 {
                            head * 128 * 16 + dim / 8 * 16 * 8 + token % 16 * 8 + dim % 8
                        } else {
                            2 * 128 * 16 + head * 128 * 16 + dim * 16 + token % 16
                        };
                    for byte in element * 2..element * 2 + 2 {
                        if let Some(write) = writes.get_mut(byte) {
                            *write = true;
                        }
                    }
                }
            }
        }
    }
    for (index, ((old, new), written)) in before.iter().zip(after).zip(writes).enumerate() {
        if !written {
            assert_eq!(
                old, new,
                "existing KV byte {index} outside this wave's write extent"
            );
        }
    }
}

fn retain_observation(
    previous: &mut [Vec<u8>],
    observation: &BatchObservation,
    caller_indices: &[usize],
    ranges: &[Range<usize>],
) {
    assert_eq!(observation.raw_kv_blocks.len(), caller_indices.len());
    for (caller, (&owner, range)) in caller_indices.iter().zip(ranges).enumerate() {
        let raw = &observation.raw_kv_blocks[&(caller as u32)];
        assert_eq!(raw.len(), range.end.div_ceil(16) * 16 * KV_BYTES_PER_TOKEN);
        preserve_existing_kv(&previous[owner], raw, range.clone());
        previous[owner].clone_from(raw);
        assert_eq!(
            observation.values[&(caller as u32, "state.kv".into())].len(),
            range.end * KV_BYTES_PER_TOKEN
        );
    }
}

fn physical_pages(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    tokens: &Arc<[u32]>,
    end: usize,
) -> u64 {
    let plan = fixture.compilation.executable().execution_plan();
    let attention = plan
        .payload()
        .nodes()
        .iter()
        .find(|n| n.id().as_str() == "node.attention")
        .unwrap();
    let resource = attention
        .values()
        .iter()
        .find(|v| v.value_id().as_str() == "value.state.kv")
        .unwrap()
        .storage()
        .components()[0]
        .resource_id();
    let work = ResourceWorkShape::single(token_span(Arc::from(&tokens[..end]), 0..end)).unwrap();
    let request =
        SequenceResourceExtensionRequest::new(work, AdmissionPressureAction::WaitForRelease)
            .unwrap();
    let SequenceResourceExtensionDecision::Current(backing) =
        session.try_ensure_backing_covers(request).unwrap()
    else {
        panic!("already extended Sequence must expose its actual committed backing");
    };
    assert!(backing.committed_tokens() >= end as u64);
    let bytes: u64 = backing
        .backing_slices()
        .iter()
        .filter(|s| s.resource_id() == resource)
        .flat_map(|s| s.evidence().segments())
        .map(|s| s.length_bytes())
        .sum();
    assert!(bytes > 0 && bytes % 65_536 == 0);
    bytes / 65_536
}

fn check_bindings(observation: &BatchObservation, ranges: &[Range<usize>]) {
    let rows = observation.binding_rows.as_ref().unwrap();
    assert_eq!(rows.len(), ranges.len());
    for (caller, (row, range)) in rows.iter().zip(ranges).enumerate() {
        let controls: Vec<_> = row[..24]
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(
            controls,
            [
                range.end.div_ceil(16) as i32,
                range.start as i32,
                1,
                range.end as i32,
                observation.caller_to_canonical_participant[caller] as i32,
                0
            ]
        );
        assert!(row[24..24 + range.end.div_ceil(16) * 8]
            .chunks_exact(8)
            .all(|v| u64::from_le_bytes(v.try_into().unwrap()) != 0));
    }
}

fn attention_work(
    observation: &BatchObservation,
    fixture: &Fixture,
    participants: usize,
    mixed: bool,
    grouping: CausalDecodeAttentionGrouping,
) -> DeviceNativeWorkAttribution {
    let node_index = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|n| n.id().as_str() == "node.attention")
        .unwrap() as u32;
    let work: Vec<_> = observation
        .attribution
        .as_ref()
        .expect("Kernel qualification requested attribution")
        .device()
        .commands()
        .iter()
        .filter(|w| {
            w.node_index() == Some(node_index) && w.command_phase() == DeviceCommandPhase::Compute
        })
        .collect();
    assert_eq!(work.len(), 1);
    let work = work[0];
    assert_eq!(work.execution_path(), DeviceExecutionPath::Eager);
    assert_eq!(work.batching_form(), DeviceBatchingForm::Packed);
    assert_eq!(work.participant_count(), participants as u32);
    assert_eq!(work.token_count(), participants as u64);
    assert_eq!(
        work.native_op_id(),
        if matches!(
            grouping,
            CausalDecodeAttentionGrouping::FixedCapacity
                | CausalDecodeAttentionGrouping::FixedExecutionCapacity
        ) {
            "vnext.causal_attention.fixed_capacity_native_paths"
        } else if mixed {
            "vnext.causal_attention.mixed_native_paths"
        } else {
            "vnext.causal_attention.vllm_paged_attention_v1_addressed"
        }
    );
    work.clone()
}

fn verify(leads: &[usize]) {
    verify_modes(leads, false);
}

fn verify_modes(leads: &[usize], indexed_grouping: bool) {
    verify_options(leads, indexed_grouping, [Q4MmvqWeightDecode::Baseline; 2]);
}

fn verify_options(leads: &[usize], indexed_grouping: bool, q4_modes: [Q4MmvqWeightDecode; 2]) {
    verify_options_with_q4_grouping(
        leads,
        indexed_grouping,
        q4_modes,
        CausalDecodeAttentionGrouping::Indexed,
    );
}

fn verify_options_with_q4_grouping(
    leads: &[usize],
    indexed_grouping: bool,
    q4_modes: [Q4MmvqWeightDecode; 2],
    q4_grouping: CausalDecodeAttentionGrouping,
) {
    verify_options_with_weight_decode_grouping(
        leads,
        indexed_grouping,
        q4_modes,
        [Q5MmvqWeightDecode::Baseline; 2],
        q4_grouping,
    );
}

fn verify_options_with_weight_decode_grouping(
    leads: &[usize],
    indexed_grouping: bool,
    q4_modes: [Q4MmvqWeightDecode; 2],
    q5_modes: [Q5MmvqWeightDecode; 2],
    q4_grouping: CausalDecodeAttentionGrouping,
) {
    verify_options_with_execution_capacity(
        leads,
        indexed_grouping,
        q4_modes,
        q5_modes,
        q4_grouping,
        None,
    );
}

fn verify_options_with_execution_capacity(
    leads: &[usize],
    indexed_grouping: bool,
    q4_modes: [Q4MmvqWeightDecode; 2],
    q5_modes: [Q5MmvqWeightDecode; 2],
    q4_grouping: CausalDecodeAttentionGrouping,
    execution_capacity_comparison: Option<usize>,
) {
    let participants = leads.len();
    let mixed = leads.iter().any(|&lead| lead > 512);
    let weight_decode_comparison = q4_modes[0] != q4_modes[1] || q5_modes[0] != q5_modes[1];
    let capacity_comparison = execution_capacity_comparison.is_some();
    let admission_ceiling = execution_capacity_comparison.unwrap_or(MAXIMUM_TOKENS as usize);
    assert!(leads
        .iter()
        .all(|&lead| lead + DECODE_WAVES <= admission_ceiling));
    let modes = if indexed_grouping || weight_decode_comparison || capacity_comparison {
        [CausalDecodePreparationMode::Packed; 2]
    } else {
        [
            CausalDecodePreparationMode::PerParticipant,
            CausalDecodePreparationMode::Packed,
        ]
    };
    let groupings = if capacity_comparison {
        [
            CausalDecodeAttentionGrouping::FixedCapacity,
            CausalDecodeAttentionGrouping::FixedExecutionCapacity,
        ]
    } else if weight_decode_comparison {
        [q4_grouping; 2]
    } else {
        [
            CausalDecodeAttentionGrouping::Contiguous,
            if indexed_grouping {
                CausalDecodeAttentionGrouping::Indexed
            } else {
                CausalDecodeAttentionGrouping::Contiguous
            },
        ]
    };
    let fixtures = std::array::from_fn::<_, 2, _>(|arm| {
        Fixture::for_family_with_execution_context_limit(
            AttentionKind::Causal,
            true,
            participants as u32,
            Family::new(AttentionKind::Causal).with_maximum_tokens(MAXIMUM_TOKENS),
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::CompareReference,
            SegmentBindingOwnerViewMode::Indexed,
            AttentionExecutionPolicy::NativeAdaptive,
            modes[arm],
            groupings[arm],
            q4_modes[arm],
            q5_modes[arm],
            execution_capacity_comparison,
        )
    });
    for (arm, fixture) in fixtures.iter().enumerate() {
        assert_eq!(
            fixture._composition._runtime.attention_execution_policy(),
            AttentionExecutionPolicy::NativeAdaptive
        );
        assert_eq!(
            fixture._composition._runtime.q4_mmvq_weight_decode(),
            q4_modes[arm]
        );
        assert_eq!(
            fixture._composition._runtime.q5_mmvq_weight_decode(),
            q5_modes[arm]
        );
        assert_eq!(
            fixture
                ._composition
                ._runtime
                .segment_binding_owner_view_mode(),
            SegmentBindingOwnerViewMode::Indexed
        );
        assert!(fixture
            .states
            .iter()
            .all(|s| s.lifetime == StateLifetime::Sequence));
    }
    let sinks = [Preparation::default(), Preparation::default()];
    let row_bytes = std::array::from_fn::<_, 2, _>(|arm| {
        let node = fixtures[arm]
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id().as_str() == "node.attention")
            .unwrap();
        let ferrum_interfaces::vnext::ProviderWorkspaceSizeFormula::ActualSequences {
            bytes_per_sequence,
        } = node.provider_resources().binding().unwrap().size_formula()
        else {
            panic!("causal binding must declare its actual per-sequence row stride");
        };
        let maximum_context =
            if groupings[arm] == CausalDecodeAttentionGrouping::FixedExecutionCapacity {
                MAXIMUM_TOKENS.min(admission_ceiling as u64)
            } else {
                MAXIMUM_TOKENS
            };
        assert_eq!(
            *bytes_per_sequence,
            24 + (maximum_context.div_ceil(16) * 8).div_ceil(16) * 16,
            "only FixedExecutionCapacity may tighten the model's binding bound"
        );
        usize::try_from(*bytes_per_sequence).unwrap()
    });
    let mut old_authorities = BTreeSet::new();
    let mut previous_generation_programs = [None, None];
    for generation in 0..2 {
        let tokens: Vec<Arc<[u32]>> = leads
            .iter()
            .enumerate()
            .map(|(owner, &lead)| {
                (0..lead + DECODE_WAVES)
                    .map(|position| {
                        ((position * 7
                            + if position == 0 { 0 } else { owner * 3 }
                            + generation * 11
                            + 1)
                            % 32) as u32
                    })
                    .collect()
            })
            .collect();
        // Assign lengths in each real batch's canonical order. The independent
        // runtimes have different IDs/addresses, but the same per-row lengths.
        let sessions = fixtures.each_ref().map(|fixture| {
            let group = (0..participants)
                .map(|owner| {
                    fixture.admit_with_ceiling(
                        &format!("packed-prepare-{generation}-{owner}"),
                        Arc::from([tokens[owner][0]]),
                        admission_ceiling,
                    )
                })
                .collect();
            ExecutionBatchParticipants::new(group)
                .unwrap()
                .sessions()
                .to_vec()
        });
        for group in &sessions {
            for session in group {
                assert!(
                    !old_authorities.contains(&session.sequence_authority()),
                    "new requests need fresh Sequence authority"
                );
            }
        }
        let mut previous = [
            vec![Vec::new(); participants],
            vec![Vec::new(); participants],
        ];
        let mut positions = vec![0; participants];
        // Genuine eager prefill, outside the decode reusable bucket. It uses
        // the same admitted Sequence resources and both real provider modes.
        while positions != leads {
            let owners: Vec<_> = (0..participants)
                .filter(|&p| positions[p] < leads[p])
                .collect();
            // This profile admits at most 160 scheduled tokens across the
            // whole batch. Six independent 32-token chunks would exceed that
            // bound before reaching either attention implementation.
            let scheduled_limit = fixtures[0]
                .compilation
                .executable()
                .execution_plan()
                .payload()
                .maximum_scheduled_tokens();
            let prefill_chunk =
                32.min(usize::try_from(scheduled_limit / owners.len() as u64).unwrap());
            assert!(prefill_chunk > 0);
            let ranges: Vec<_> = owners
                .iter()
                .map(|&p| positions[p]..(positions[p] + prefill_chunk).min(leads[p]))
                .collect();
            let mut outputs = Vec::new();
            for (arm, fixture) in fixtures.iter().enumerate() {
                let local_sessions: Vec<_> = owners
                    .iter()
                    .map(|&p| Arc::clone(&sessions[arm][p]))
                    .collect();
                let local_tokens: Vec<_> = owners.iter().map(|&p| Arc::clone(&tokens[p])).collect();
                for ((session, t), range) in local_sessions.iter().zip(&local_tokens).zip(&ranges) {
                    fixture.extend(session, Arc::from(&t[..range.end]));
                }
                let output = fixture.execute_participant_ranges_observed(
                    &fixture.lane,
                    &fixture.reaper,
                    &local_sessions,
                    &local_tokens,
                    &ranges,
                    Path::Eager,
                    InvocationPreparationStrategy::DecodeSegment,
                    &sinks[arm],
                    None,
                    BatchObservationOptions {
                        retain_raw_kv: true,
                        use_reusable_bucket: false,
                        ..Default::default()
                    },
                );
                retain_observation(&mut previous[arm], &output, &owners, &ranges);
                outputs.push(output);
            }
            outputs[0].assert_same(&outputs[1]);
            for (&owner, range) in owners.iter().zip(ranges) {
                positions[owner] = range.end;
            }
        }
        let mut stable_programs = [None, None];
        let mut before_transition = [None, None];
        let mut frames = [Vec::new(), Vec::new()];
        let mut hot_hits = [0, 0];
        for wave in 0..DECODE_WAVES {
            // The contiguous/Indexed comparison changes topology at 512.
            // The FixedCapacity Q4 pair keeps its captured program across
            // that boundary; its current lengths select the active nodes.
            let replay_start = if indexed_grouping && wave >= 4 { 6 } else { 3 };
            if indexed_grouping && wave == 4 {
                before_transition = stable_programs.clone();
                stable_programs = [None, None];
            }
            let order: Vec<_> = if wave >= 6 {
                (0..participants).rev().collect()
            } else {
                (0..participants).collect()
            };
            let ranges: Vec<_> = order
                .iter()
                .map(|&p| leads[p] + wave..leads[p] + wave + 1)
                .collect();
            let path = if wave == 0 {
                Path::Eager
            } else if wave < replay_start {
                Path::Warm
            } else {
                Path::Replay
            };
            let mut observations = Vec::new();
            let mut weight_encode_counts = [[0_u64; 2]; 2];
            let mut work = Vec::new();
            for (arm, fixture) in fixtures.iter().enumerate() {
                let group: Vec<_> = order
                    .iter()
                    .map(|&p| Arc::clone(&sessions[arm][p]))
                    .collect();
                let local_tokens: Vec<_> = order.iter().map(|&p| Arc::clone(&tokens[p])).collect();
                let mut pages = Vec::new();
                for ((session, t), range) in group.iter().zip(&local_tokens).zip(&ranges) {
                    fixture.extend(session, Arc::from(&t[..range.end]));
                    pages.push(physical_pages(fixture, session, t, range.end));
                }
                assert_eq!(
                    pages,
                    ranges
                        .iter()
                        .map(|r| (r.end * KV_BYTES_PER_TOKEN).div_ceil(65_536) as u64)
                        .collect::<Vec<_>>()
                );
                let indexed_before = fixture
                    ._composition
                    ._runtime
                    .segment_binding_indexed_encodes();
                let audited_before = fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes();
                let q4_before = fixture
                    ._composition
                    ._runtime
                    .q4_shared_weight_decode_encodes();
                let q5_before = fixture
                    ._composition
                    ._runtime
                    .q5_shared_weight_decode_encodes();
                let observation = fixture.execute_participant_ranges_observed(
                    &fixture.lane,
                    &fixture.reaper,
                    &group,
                    &local_tokens,
                    &ranges,
                    path,
                    InvocationPreparationStrategy::DecodeSegment,
                    &sinks[arm],
                    Some(row_bytes[arm]),
                    BatchObservationOptions {
                        timing: if wave == 0 {
                            DeviceTimingMode::Kernel
                        } else {
                            DeviceTimingMode::Off
                        },
                        retain_raw_kv: true,
                        ..Default::default()
                    },
                );
                let stats = sinks[arm].last();
                let indexed = fixture
                    ._composition
                    ._runtime
                    .segment_binding_indexed_encodes()
                    - indexed_before;
                let audited = fixture
                    ._composition
                    ._runtime
                    .segment_binding_oracle_audited_nodes()
                    - audited_before;
                let q4_encodes = fixture
                    ._composition
                    ._runtime
                    .q4_shared_weight_decode_encodes()
                    - q4_before;
                if q4_modes[arm] == Q4MmvqWeightDecode::Baseline || participants != 8 {
                    assert_eq!(
                        q4_encodes, 0,
                        "baseline and ineligible decode widths must keep the original Q4 route"
                    );
                } else if wave == 0 {
                    assert!(
                        q4_encodes > 0,
                        "the paired physical-M8 candidate must execute the shared-weight Q4 route"
                    );
                }
                let q5_encodes = fixture
                    ._composition
                    ._runtime
                    .q5_shared_weight_decode_encodes()
                    - q5_before;
                if q5_modes[arm] == Q5MmvqWeightDecode::Baseline || participants != 8 {
                    assert_eq!(
                        q5_encodes, 0,
                        "baseline and ineligible decode widths must keep the original Q5 route"
                    );
                } else if wave == 0 {
                    assert!(
                        q5_encodes > 0,
                        "the paired physical-M8 candidate must execute the shared-weight Q5 route"
                    );
                }
                weight_encode_counts[arm] = [q4_encodes, q5_encodes];
                if matches!(path, Path::Replay) {
                    assert_eq!(
                        [q4_encodes, q5_encodes],
                        [0, 0],
                        "hot replay executes the captured route without a new host dot enqueue"
                    );
                }
                assert_eq!(indexed, stats.segment_hits);
                assert_eq!(audited, stats.segment_encoded_nodes);
                if wave == 0 {
                    work.push(attention_work(
                        &observation,
                        fixture,
                        participants,
                        mixed,
                        groupings[arm],
                    ));
                } else {
                    assert!(
                        observation.attribution.is_none(),
                        "ordinary capture/replay must remain Off"
                    );
                }
                if wave >= replay_start {
                    let program = observation.reusable_program_id.as_ref().unwrap();
                    if capacity_comparison {
                        if let Some(previous) = &previous_generation_programs[arm] {
                            assert_eq!(
                                previous, program,
                                "fresh requests reuse the same bounded native topology"
                            );
                        }
                    }
                    if indexed_grouping && wave >= 4 {
                        assert_ne!(
                            before_transition[arm].as_ref().unwrap(),
                            program,
                            "512 -> 513 must select a new exact native topology"
                        );
                    }
                    if let Some(previous) = &stable_programs[arm] {
                        assert_eq!(previous, program);
                    }
                    stable_programs[arm] = Some(program.clone());
                    // Adaptive warm/capture does not supply the resident
                    // program to Core's binding-recipe lookup. The first
                    // direct replay may therefore prepare and publish that
                    // recipe only after successful completion and retirement.
                    if generation == 0 && wave == replay_start {
                        assert_eq!(stats.segment_hits, 0, "first direct replay: {stats:?}");
                    }
                    if wave == replay_start && stats.segment_hits == 0 {
                        assert_eq!(stats.segment_misses, 1, "{stats:?}");
                        assert_eq!(stats.segment_no_cached_recipe, 1, "{stats:?}");
                        assert_eq!(stats.segment_no_resident_program, 0, "{stats:?}");
                        assert_eq!(stats.segment_incomplete_declarations, 0, "{stats:?}");
                        assert_eq!(stats.segment_encoded_nodes, 0, "{stats:?}");
                        assert!(observation.segment_published, "{stats:?}");
                    } else {
                        // This includes the following 64 -> 65 physical-page
                        // growth: the exact program must now encode hot using
                        // the newly authorized KV extent, without republishing.
                        assert_eq!(stats.segment_hits, 1, "{stats:?}");
                        assert_eq!(stats.segment_misses, 0, "{stats:?}");
                        assert!(!observation.segment_published);
                        assert!(audited > 0);
                        hot_hits[arm] += stats.segment_hits;
                    }
                }
                let mut canonical_frames = vec![None; participants];
                for (caller, &owner) in order.iter().enumerate() {
                    canonical_frames[owner] = Some(observation.participant_frames[caller]);
                }
                if !frames[arm].is_empty() {
                    assert!(canonical_frames
                        .iter()
                        .zip(&frames[arm])
                        .all(|(a, b)| a > b));
                }
                frames[arm] = canonical_frames;
                check_bindings(&observation, &ranges);
                retain_observation(&mut previous[arm], &observation, &order, &ranges);
                println!(
                    "{}",
                    serde_json::json!({"kind":"packed_preparation_model_wave", "mode":modes[arm],"grouping":groupings[arm],
                    "q4_mmvq_weight_decode":q4_modes[arm],"q4_shared_host_encodes":q4_encodes,
                    "q5_mmvq_weight_decode":q5_modes[arm],"q5_shared_host_encodes":q5_encodes,
                    "execution_context_limit":execution_capacity_comparison,"binding_row_bytes":row_bytes[arm],
                    "generation":generation,"participants":participants,"ranges":ranges,"caller_order":order,
                    "path":format!("{path:?}"),"physical_64k_pages":pages,"program":observation.reusable_program_id,
                    "hits":stats.segment_hits,"indexed_host_encodes":indexed,"oracle_nodes":audited,
                    "preparation":stats,
                    "published":observation.segment_published,"all_valid_output_kv_and_existing_bytes_checked":true})
                );
                observations.push(observation);
            }
            if q4_modes[0] == q4_modes[1] {
                assert_eq!(
                    weight_encode_counts[0][0], weight_encode_counts[1][0],
                    "Q5 selection must not change Q4 host enqueue counts"
                );
            }
            if q5_modes[0] == q5_modes[1] {
                assert_eq!(
                    weight_encode_counts[0][1], weight_encode_counts[1][1],
                    "Q4 selection must not change Q5 host enqueue counts"
                );
            }
            observations[0].assert_same(&observations[1]);
            if wave == 0 {
                let expected_removed = if weight_decode_comparison || capacity_comparison {
                    0
                } else if indexed_grouping {
                    // V1,V1,V2,V1,V1,V2: four contiguous groups use six
                    // attention kernels; indexed V1 + V2 main/reduce use three.
                    3
                } else {
                    2 * (participants as u64 - 1)
                };
                assert_eq!(work[0].compute_dispatch_count() - work[1].compute_dispatch_count(), expected_removed,
                    "actual selected preparation/grouping must have the declared native work difference");
                println!(
                    "{}",
                    serde_json::json!({"kind":"packed_preparation_model_qualification", "generation":generation,
                    "preparation_modes":modes,"attention_groupings":groupings,
                    "q4_mmvq_weight_decode_modes":q4_modes,"q5_mmvq_weight_decode_modes":q5_modes,
                    "reference":work[0],"candidate":work[1],"scope":"actual submitted attribution; declared dispatch count, not hardware kernel count"})
                );
            }
        }
        assert!(hot_hits.iter().all(|&n| n > 0));
        previous_generation_programs = stable_programs;
        for group in sessions {
            for session in group {
                old_authorities.insert(session.sequence_authority());
                session.try_complete().unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA NativeAdaptive, actual Core Sequence admission and packed prepare/gate providers"]
fn packed_decode_preparation_v1_matches_per_participant_across_core_replay_and_growth() {
    verify(&[60, 60]);
}

#[test]
#[ignore = "requires CUDA NativeAdaptive V1/V2, actual Core Sequence admission and packed prepare/gate providers"]
fn packed_decode_preparation_mixed_v1_v2_matches_per_participant_across_fresh_requests() {
    verify(&[60, 60, 1084, 1084]);
}

#[test]
#[cfg(feature = "cuda-indexed-causal-attention")]
#[ignore = "requires CUDA indexed NativeAdaptive, actual Core Sequence admission and replay"]
fn indexed_decode_attention_matches_contiguous_across_core_replay_boundary_and_fresh_requests() {
    // The 508 rows cross V1/V2 at wave4; the 57 rows grow physical pages on
    // wave7, after the new program is captured and its fresh recipe published.
    verify_modes(&[508, 57, 1084, 57, 508, 1084], true);
}

#[test]
#[cfg(all(
    feature = "cuda-q4-shared-weight-decode",
    feature = "cuda-indexed-causal-attention"
))]
#[ignore = "requires CUDA shared-weight Q4 and indexed NativeAdaptive, actual Core admission and replay"]
fn q4_shared_weight_decode_matches_baseline_across_core_m8_replay_and_fresh_requests() {
    let modes = [Q4MmvqWeightDecode::Baseline, Q4MmvqWeightDecode::SharedM8];
    // Both arms retain the current Packed/Indexed attention configuration.
    // Real M8 activates Q4 sharing; M4 exercises the ineligible original path.
    // Each case mixes native attention widths, grows KV backing, reorders
    // participants, and reuses graphs with fresh request authorities.
    verify_options(&[60, 60, 1084, 1084, 60, 60, 1084, 1084], false, modes);
    verify_options(&[60, 60, 1084, 1084], false, modes);
}

#[test]
#[cfg(all(
    feature = "cuda-q4-shared-weight-decode",
    feature = "cuda-indexed-causal-attention"
))]
#[ignore = "requires CUDA shared-weight Q4 and fixed-capacity NativeAdaptive, actual Core admission and replay"]
fn q4_shared_weight_decode_matches_baseline_with_fixed_capacity_replay_and_growth() {
    let modes = [Q4MmvqWeightDecode::Baseline, Q4MmvqWeightDecode::SharedM8];
    // At wave4 the 508 rows cross V1/V2 without replacing the fixed program.
    // The 57 rows grow physical KV pages at wave7, after capture, publication
    // and caller reordering. The M4 case keeps the original Q4 dot selected.
    for leads in [
        &[508, 57, 1084, 57, 508, 1084, 57, 1084][..],
        &[508, 57, 1084, 57][..],
    ] {
        verify_options_with_q4_grouping(
            leads,
            false,
            modes,
            CausalDecodeAttentionGrouping::FixedCapacity,
        );
    }
}

#[test]
#[cfg(feature = "cuda-q4-shared-weight-decode")]
#[ignore = "requires CUDA shared-weight Q4, actual Core GDN admission and replay"]
fn q4_shared_weight_decode_gated_delta_matches_baseline_across_core_m8_replay_and_fresh_requests() {
    verify_gdn_weight_decode(
        [Q4MmvqWeightDecode::Baseline, Q4MmvqWeightDecode::SharedM8],
        [Q5MmvqWeightDecode::Baseline; 2],
    );
}

fn verify_gdn_weight_decode(q4_modes: [Q4MmvqWeightDecode; 2], q5_modes: [Q5MmvqWeightDecode; 2]) {
    let q5_comparison = q5_modes[0] != q5_modes[1];
    let fixtures = std::array::from_fn::<_, 2, _>(|arm| {
        Fixture::for_family_with_mmvq_weight_decode(
            AttentionKind::GatedDelta,
            true,
            8,
            Family::new(AttentionKind::GatedDelta),
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::CompareReference,
            SegmentBindingOwnerViewMode::Indexed,
            if q5_comparison {
                AttentionExecutionPolicy::NativeAdaptive
            } else {
                AttentionExecutionPolicy::Portable
            },
            if q5_comparison {
                CausalDecodePreparationMode::Packed
            } else {
                CausalDecodePreparationMode::PerParticipant
            },
            if q5_comparison {
                CausalDecodeAttentionGrouping::FixedCapacity
            } else {
                CausalDecodeAttentionGrouping::Contiguous
            },
            q4_modes[arm],
            q5_modes[arm],
        )
    });
    let mut old_authorities = BTreeSet::new();
    let mut programs = [None, None];
    for generation in 0..2 {
        let tokens: Vec<Arc<[u32]>> = (0..8)
            .map(|owner| {
                (0..6)
                    .map(|position| ((position * 7 + owner * 3 + generation * 11 + 1) % 32) as u32)
                    .collect()
            })
            .collect();
        let sessions = fixtures.each_ref().map(|fixture| {
            tokens
                .iter()
                .enumerate()
                .map(|(owner, tokens)| {
                    let session = fixture.admit(
                        &format!("shared-q4-gdn-{generation}-{owner}"),
                        Arc::clone(tokens),
                    );
                    assert!(!old_authorities.contains(&session.sequence_authority()));
                    session
                })
                .collect::<Vec<_>>()
        });
        for position in 0..6 {
            let path = if position == 0 {
                Path::Eager
            } else if position < 3 {
                Path::Warm
            } else {
                Path::Replay
            };
            let order: Vec<_> = if position >= 4 {
                (0..8).rev().collect()
            } else {
                (0..8).collect()
            };
            let local_tokens: Vec<_> = order
                .iter()
                .map(|&owner| Arc::clone(&tokens[owner]))
                .collect();
            let mut weight_encode_counts = [[0_u64; 2]; 2];
            let observations = std::array::from_fn::<_, 2, _>(|arm| {
                let fixture = &fixtures[arm];
                assert_eq!(
                    fixture._composition._runtime.q4_mmvq_weight_decode(),
                    q4_modes[arm]
                );
                assert_eq!(
                    fixture._composition._runtime.q5_mmvq_weight_decode(),
                    q5_modes[arm]
                );
                let group: Vec<_> = order
                    .iter()
                    .map(|&owner| Arc::clone(&sessions[arm][owner]))
                    .collect();
                let before = fixture
                    ._composition
                    ._runtime
                    .q4_shared_weight_decode_encodes();
                let q5_before = fixture
                    ._composition
                    ._runtime
                    .q5_shared_weight_decode_encodes();
                let observation = fixture.execute_participants(
                    &group,
                    &local_tokens,
                    position..position + 1,
                    path,
                );
                let encodes = fixture
                    ._composition
                    ._runtime
                    .q4_shared_weight_decode_encodes()
                    - before;
                if q4_modes[arm] == Q4MmvqWeightDecode::Baseline || matches!(path, Path::Replay) {
                    assert_eq!(
                        encodes, 0,
                        "baseline and hot replay do not enqueue a shared dot on the host"
                    );
                } else if position == 0 {
                    assert!(
                        encodes > 0,
                        "actual GDN M8 must execute the selected shared-weight Q4 route"
                    );
                }
                let q5_encodes = fixture
                    ._composition
                    ._runtime
                    .q5_shared_weight_decode_encodes()
                    - q5_before;
                if q5_modes[arm] == Q5MmvqWeightDecode::Baseline || matches!(path, Path::Replay) {
                    assert_eq!(
                        q5_encodes, 0,
                        "baseline and hot replay do not enqueue a shared dot on the host"
                    );
                } else if position == 0 {
                    assert!(
                        q5_encodes > 0,
                        "actual GDN M8 must execute the selected shared-weight Q5 route"
                    );
                }
                weight_encode_counts[arm] = [encodes, q5_encodes];
                if matches!(path, Path::Replay) {
                    let program = observation.reusable_program_id.as_ref().unwrap();
                    if let Some(previous) = &programs[arm] {
                        assert_eq!(
                            previous, program,
                            "fresh request authorities must reuse the same native topology"
                        );
                    }
                    programs[arm] = Some(program.clone());
                }
                observation
            });
            if q4_modes[0] == q4_modes[1] {
                assert_eq!(
                    weight_encode_counts[0][0], weight_encode_counts[1][0],
                    "GDN Q5 selection must not change Q4 host enqueue counts"
                );
            }
            if q5_modes[0] == q5_modes[1] {
                assert_eq!(
                    weight_encode_counts[0][1], weight_encode_counts[1][1],
                    "GDN Q4 selection must not change Q5 host enqueue counts"
                );
            }
            observations[0].assert_same(&observations[1]);
        }
        for group in sessions {
            for session in group {
                old_authorities.insert(session.sequence_authority());
                session.try_complete().unwrap();
            }
        }
    }
}

#[test]
#[cfg(all(
    feature = "cuda-q5-shared-weight-decode",
    feature = "cuda-indexed-causal-attention"
))]
#[ignore = "requires CUDA Q5 shared-weight, fixed-capacity Core replay and KV growth"]
fn q5_shared_weight_decode_matches_baseline_with_fixed_capacity_replay_and_growth() {
    let q4 = if cfg!(feature = "cuda-q4-shared-weight-decode") {
        Q4MmvqWeightDecode::SharedM8
    } else {
        Q4MmvqWeightDecode::Baseline
    };
    for leads in [
        &[508, 57, 1084, 57, 508, 1084, 57, 1084][..],
        &[508, 57, 1084, 57][..],
    ] {
        verify_options_with_weight_decode_grouping(
            leads,
            false,
            [q4; 2],
            [Q5MmvqWeightDecode::Baseline, Q5MmvqWeightDecode::SharedM8],
            CausalDecodeAttentionGrouping::FixedCapacity,
        );
    }
}

#[test]
#[cfg(all(
    feature = "cuda-q4-shared-weight-decode",
    feature = "cuda-q5-shared-weight-decode",
    feature = "cuda-indexed-causal-attention"
))]
#[ignore = "requires CUDA shared-weight Q4/Q5, actual Core bounded fixed-capacity replay and KV growth"]
fn fixed_execution_capacity_matches_fixed_capacity_across_core_replay_growth_and_fresh_requests() {
    // Same declared model bound and request ceiling in both arms. Only the new
    // grouping may use the execution ceiling to tighten the prepared workspace.
    // M8 exercises both shared projections. Eight decode waves cross V1/V2 at
    // 512, grow physical KV pages after replay starts, reverse caller order,
    // and reach the execution ceiling exactly. A second request generation
    // must preserve the captured topology with fresh Sequence authorities.
    verify_options_with_execution_capacity(
        &[508, 57, 508, 57, 508, 57, 508, 57],
        false,
        [Q4MmvqWeightDecode::SharedM8; 2],
        [Q5MmvqWeightDecode::SharedM8; 2],
        CausalDecodeAttentionGrouping::FixedCapacity,
        Some(516),
    );
}

#[test]
#[cfg(all(
    feature = "cuda-q5-shared-weight-decode",
    feature = "cuda-indexed-causal-attention"
))]
#[ignore = "requires CUDA Q5 shared-weight, actual Core GDN admission and fixed-capacity composition"]
fn q5_shared_weight_decode_gated_delta_matches_baseline_across_core_m8_replay_and_fresh_requests() {
    let q4 = if cfg!(feature = "cuda-q4-shared-weight-decode") {
        Q4MmvqWeightDecode::SharedM8
    } else {
        Q4MmvqWeightDecode::Baseline
    };
    verify_gdn_weight_decode(
        [q4; 2],
        [Q5MmvqWeightDecode::Baseline, Q5MmvqWeightDecode::SharedM8],
    );
}
