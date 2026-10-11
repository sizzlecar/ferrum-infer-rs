//! Real Core admission and provider submission. The two independent histories
//! differ only in causal preparation mode; no fixture asserts KV independence
//! on the backend's behalf. Kernel attribution is a submission declaration,
//! not a hardware counter. Ordinary warm/capture/replay remains timing Off.
use super::*;
use batch::BatchObservationOptions;
use ferrum_types::{
    AttentionExecutionPolicy, CausalDecodePreparationMode, InvocationPreparationStrategy,
    ProgramBindingUploadStrategy,
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
        if mixed {
            "vnext.causal_attention.mixed_native_paths"
        } else {
            "vnext.causal_attention.vllm_paged_attention_v1_addressed"
        }
    );
    work.clone()
}

fn verify(leads: &[usize]) {
    let participants = leads.len();
    let mixed = leads.iter().any(|&lead| lead > 512);
    let modes = [
        CausalDecodePreparationMode::PerParticipant,
        CausalDecodePreparationMode::Packed,
    ];
    let fixtures = modes.map(|mode| {
        Fixture::for_family_with_causal_decode_preparation(
            AttentionKind::Causal,
            true,
            participants as u32,
            Family::new(AttentionKind::Causal).with_maximum_tokens(MAXIMUM_TOKENS),
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::CompareReference,
            SegmentBindingOwnerViewMode::Indexed,
            AttentionExecutionPolicy::NativeAdaptive,
            mode,
        )
    });
    for fixture in &fixtures {
        assert_eq!(
            fixture._composition._runtime.attention_execution_policy(),
            AttentionExecutionPolicy::NativeAdaptive
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
    let row_bytes = 24 + (MAXIMUM_TOKENS.div_ceil(16) * 8).div_ceil(16) as usize * 16;
    let mut old_authorities = BTreeSet::new();
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
        // runtimes have different IDs/addresses, but the same V1,V1,V2,V2 order.
        let sessions = fixtures.each_ref().map(|fixture| {
            let group = (0..participants)
                .map(|owner| {
                    fixture.admit_with_ceiling(
                        &format!("packed-prepare-{generation}-{owner}"),
                        Arc::from([tokens[owner][0]]),
                        MAXIMUM_TOKENS as usize,
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
            let ranges: Vec<_> = owners
                .iter()
                .map(|&p| positions[p]..(positions[p] + 32).min(leads[p]))
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
        let mut frames = [Vec::new(), Vec::new()];
        let mut hot_hits = [0, 0];
        for wave in 0..DECODE_WAVES {
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
            } else if wave < 3 {
                Path::Warm
            } else {
                Path::Replay
            };
            let mut observations = Vec::new();
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
                let observation = fixture.execute_participant_ranges_observed(
                    &fixture.lane,
                    &fixture.reaper,
                    &group,
                    &local_tokens,
                    &ranges,
                    path,
                    InvocationPreparationStrategy::DecodeSegment,
                    &sinks[arm],
                    Some(row_bytes),
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
                assert_eq!(indexed, stats.segment_hits);
                assert_eq!(audited, stats.segment_encoded_nodes);
                if wave == 0 {
                    work.push(attention_work(&observation, fixture, participants, mixed));
                } else {
                    assert!(
                        observation.attribution.is_none(),
                        "ordinary capture/replay must remain Off"
                    );
                }
                if wave >= 3 {
                    assert_eq!(stats.segment_hits, 1);
                    assert_eq!(stats.segment_misses, 0);
                    assert!(!observation.segment_published);
                    assert!(audited > 0);
                    let program = observation.reusable_program_id.as_ref().unwrap();
                    if let Some(previous) = &stable_programs[arm] {
                        assert_eq!(previous, program);
                    }
                    stable_programs[arm] = Some(program.clone());
                    hot_hits[arm] += stats.segment_hits;
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
                    serde_json::json!({"kind":"packed_preparation_model_wave", "mode":modes[arm],
                    "generation":generation,"participants":participants,"ranges":ranges,"caller_order":order,
                    "path":format!("{path:?}"),"physical_64k_pages":pages,"program":observation.reusable_program_id,
                    "hits":stats.segment_hits,"indexed_host_encodes":indexed,"oracle_nodes":audited,
                    "published":observation.segment_published,"all_valid_output_kv_and_existing_bytes_checked":true})
                );
                observations.push(observation);
            }
            observations[0].assert_same(&observations[1]);
            if wave == 0 {
                assert_eq!(work[0].compute_dispatch_count() - work[1].compute_dispatch_count(), 2 * (participants as u64 - 1),
                    "actual selected BatchDecode must collapse both Sequence prepare and enabled gate; common projection/attention work is unchanged");
                println!(
                    "{}",
                    serde_json::json!({"kind":"packed_preparation_model_qualification", "generation":generation,
                    "per_participant":work[0],"packed":work[1],"scope":"actual submitted attribution; declared dispatch count, not hardware kernel count"})
                );
            }
        }
        assert!(hot_hits.iter().all(|&n| n > 0));
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
