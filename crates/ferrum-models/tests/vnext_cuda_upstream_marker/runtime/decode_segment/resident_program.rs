//! Real whole-program uses: ordinary capture/bootstrap, fresh resident frames,
//! original completion/readbacks and explicit retirement. No synthetic fence.
use super::*;

#[derive(Default)]
struct Resident {
    reservation: Option<Arc<ResidentDecodeReservation<Runtime>>>,
    program: Option<Arc<ResidentDecodeProgram<Runtime>>>,
    replay_id: Option<DeviceReusableExecutionProgramId>,
}

struct Quiet;
impl DeviceSubmissionTimingSink for Quiet {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        unreachable!()
    }
}
impl SubmissionWaveDispatchTimingSink for Quiet {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        unreachable!()
    }
}

enum Retire {
    Bootstrap(Arc<StepResourceLease<Runtime>>),
    Resident(ResidentFrameRetirement<Runtime>),
}

fn request(
    fixture: &Fixture,
    batch: &ExecutionBatchParticipants<Runtime>,
    tokens: &[Arc<[u32]>],
    ranges: &[Range<usize>],
) -> StepResourceAdmissionRequest {
    StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(
                tokens
                    .iter()
                    .zip(ranges)
                    .map(|(tokens, range)| token_span(Arc::clone(tokens), range.clone()))
                    .collect(),
            )
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(fixture.reusable_bucket.clone().unwrap())
}

/// None is a genuine pre-enqueue miss. The caller drops the idle reservation
/// before invoking ordinary admission on that same frontier.
fn execute(
    fixture: &Fixture,
    state: &mut Resident,
    sessions: &[Arc<SequenceSession<Runtime>>],
    tokens: &[Arc<[u32]>],
    ranges: &[Range<usize>],
) -> Option<(
    BTreeMap<(u32, String), Vec<u8>>,
    bool,
    Vec<ExecutionFrameId>,
)> {
    let batch = ExecutionBatchParticipants::new(sessions.to_vec()).unwrap();
    let caller: Vec<_> = batch
        .sessions()
        .iter()
        .map(|canonical| {
            sessions
                .iter()
                .position(|s| Arc::ptr_eq(s, canonical))
                .unwrap()
        })
        .collect();
    let tokens: Vec<_> = caller.iter().map(|&i| Arc::clone(&tokens[i])).collect();
    let ranges: Vec<_> = caller.iter().map(|&i| ranges[i].clone()).collect();
    let request = request(fixture, &batch, &tokens, &ranges);
    let lane = &fixture.lane;
    let executable = fixture.compilation.executable();
    let active: Vec<_> = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect();
    let uploads: Vec<_> = tokens
        .iter()
        .zip(&ranges)
        .enumerate()
        .map(|(p, (tokens, range))| {
            SubmissionWaveInputUpload::new(
                id("node.embedding"),
                p as u32,
                0,
                range.start as u64 * 4,
                HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                tokens[range.clone()]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let mut installing = None;
    let hot = state.program.is_some();
    let (profiled, retire) = if let Some(program) = state.program.as_ref() {
        let (use_, retirement) = match program.try_begin_use(&batch, &request).unwrap() {
            ResidentDecodeDecision::Ready { use_, retirement } => (use_, retirement),
            ResidentDecodeDecision::Fallback => return None,
            _ => panic!("unexpected resident request-state deferral"),
        };
        let entry_before = lane
            .reusable_execution_entry_identity(state.replay_id.as_ref().unwrap())
            .unwrap()
            .unwrap();
        let identity =
            OperationDispatch::bind_resident_decode_identity(program, active.iter(), &use_, lane)
                .unwrap();
        let before = fixture
            ._composition
            ._runtime
            .resident_decode_program_encodes();
        let physical_before = fixture
            ._composition
            ._runtime
            .physical_segment_binding_encodes();
        let indexed_before = fixture
            ._composition
            ._runtime
            .segment_binding_indexed_encodes();
        let profiled = OperationDispatch::encode_and_submit_resident_decode_with_inputs(
            program,
            executable,
            &identity,
            &uploads,
            use_,
            lane,
            &fixture.reaper,
            DeviceTimingMode::Off,
            &Quiet,
        )
        .unwrap();
        assert_eq!(
            fixture
                ._composition
                ._runtime
                .resident_decode_program_encodes()
                - before,
            1
        );
        assert_eq!(
            fixture
                ._composition
                ._runtime
                .physical_segment_binding_encodes(),
            physical_before,
            "whole resident hit must bypass the physical program encoder"
        );
        assert_eq!(
            fixture
                ._composition
                ._runtime
                .segment_binding_indexed_encodes(),
            indexed_before,
            "whole resident hit must bypass the ordinary owner/frame encoder"
        );
        let entry_after = lane
            .reusable_execution_entry_identity(state.replay_id.as_ref().unwrap())
            .unwrap()
            .unwrap();
        assert!(
            entry_before.same_entry(&entry_after),
            "append refresh must retain the captured graph entry"
        );
        (profiled, Retire::Resident(retirement))
    } else {
        if state.reservation.is_none() {
            state.reservation = batch.try_reserve_resident_decode(&request, lane).unwrap();
        }
        let reservation = state
            .reservation
            .as_ref()
            .expect("prewarmed fixture has resident capacity");
        let (step, wave) = match reservation.prepare_bootstrap(&request).unwrap() {
            ResidentBootstrapDecision::Prepared { step, wave } => (step, wave),
            _ => panic!("real reserved bootstrap was unavailable"),
        };
        let topology =
            OperationDispatch::compile_submission_wave_identity(executable, lane).unwrap();
        let identity = OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
            &topology,
            active.iter(),
            &wave,
            lane,
            InvocationPreparationStrategy::DecodeSegmentProgram,
        )
        .unwrap();
        let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
            fixture.providers.providers(),
            executable,
            &wave,
            lane,
        )
        .unwrap()
        .unwrap();
        let catalog = lane.reusable_execution_catalog().unwrap();
        let replay = catalog
            .programs()
            .iter()
            .find(|p| p.program_id() == &program_id)
            .filter(|p| p.is_determinism_ready());
        if let Some(replay) = replay {
            installing = OperationDispatch::try_compile_resident_decode(
                Arc::clone(reservation),
                executable,
                &wave,
                &topology,
                replay,
                lane,
            )
            .unwrap();
        }
        if installing.is_some() {
            state.replay_id = Some(program_id.clone());
        }
        println!(
            "{}",
            serde_json::json!({"kind":"resident_bootstrap_preparation",
            "source_start":ranges[0].start,"replay_ready":replay.is_some(),
            "cold_compile_attempted":replay.is_some(),"resident_compiled":installing.is_some()})
        );
        // As in the ordinary fixture's Path::Replay, a ready catalog must use
        // the direct reusable entry: this path publishes the complete segment
        // binding recipe after successful completion and retirement.
        let profiled = if let Some(replay) = replay {
            OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
                fixture.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &uploads,
                replay,
                SubmissionExecutionPolicy::determinism_replayed(0xa5),
                InvocationPreparationStrategy::DecodeSegmentProgram,
                &Quiet,
                &Preparation::default(),
                wave,
                lane,
                &fixture.reaper,
            )
        } else {
            OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
                fixture.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &uploads,
                SubmissionExecutionPolicy::adaptive(),
                InvocationPreparationStrategy::DecodeSegmentProgram,
                &Quiet,
                &Preparation::default(),
                wave,
                lane,
                &fixture.reaper,
            )
        }
        .unwrap();
        (profiled, Retire::Bootstrap(step))
    };
    let (handle, attribution) = profiled.into_parts();
    let bootstrap_step = match &retire {
        Retire::Bootstrap(step) => Some(Arc::downgrade(step)),
        _ => None,
    };
    let publication = handle.take_segment_binding_publication();
    let (readbacks, names) = readbacks(fixture, &ranges);
    let receipt = match handle.wait_with_readback_collection(readbacks).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("resident full-state readback did not terminate: {other:?}"),
    };
    let publication = publication.and_then(|ticket| ticket.bind_completion(receipt.completion()));
    let mut values = BTreeMap::new();
    for disposition in receipt.dispositions() {
        let CompletionReadbackDisposition::Succeeded(result) = disposition else {
            panic!("resident readback failed")
        };
        let name = names[result.request().resource_id()].clone();
        let canonical = result.request().participant_index() as usize;
        if name == "binding_parameters" {
            let row = result.bytes();
            if fixture
                .states
                .iter()
                .any(|state| state.id.as_str() == "state.kv")
            {
                let controls = row[..24]
                    .chunks_exact(4)
                    .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let range = &ranges[canonical];
                assert_eq!(
                    controls,
                    [
                        range.end.div_ceil(16) as i32,
                        range.start as i32,
                        1,
                        range.end as i32,
                        canonical as i32,
                        0
                    ]
                );
                assert!(row[24..24 + range.end.div_ceil(16) * 8]
                    .chunks_exact(8)
                    .all(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()) != 0));
            } else {
                assert_eq!(row.len(), 16);
                assert!(row
                    .chunks_exact(8)
                    .all(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()) != 0));
            }
            continue;
        }
        let bytes = if name == "state.kv" {
            valid_kv(
                result.bytes(),
                ranges[canonical].end,
                fixture
                    .states
                    .iter()
                    .find(|s| s.id.as_str() == "state.kv")
                    .unwrap()
                    .tensor
                    .dimensions[1] as usize,
            )
        } else {
            result.bytes().to_vec()
        };
        assert!(values
            .insert((caller[canonical] as u32, name), bytes)
            .is_none());
    }
    assert_eq!(values.len(), sessions.len() * (names.len() - 1));
    drop((receipt, handle, attribution, active));
    let retirement = match retire {
        Retire::Bootstrap(step) => step.try_retire_normal().unwrap(),
        Retire::Resident(ticket) => ticket.try_retire_normal().unwrap(),
    };
    if let Some(step) = bootstrap_step {
        assert!(
            step.upgrade().is_none(),
            "cold program must not retain the old Step"
        );
    }
    if let Some(publication) = publication {
        assert!(publication.publish(&retirement).unwrap());
    }
    let mut frames = vec![None; caller.len()];
    for (canonical, participant) in retirement.participants().iter().enumerate() {
        frames[caller[canonical]] = Some(participant.assignment().frame_id());
    }
    if let Some(program) = installing {
        state.program = Some(program);
    }
    assert!(
        state.reservation.as_ref().unwrap().is_idle(),
        "retirement must release the fresh use"
    );
    Some((
        values,
        hot,
        frames.into_iter().map(Option::unwrap).collect(),
    ))
}

fn readbacks(
    fixture: &Fixture,
    ranges: &[Range<usize>],
) -> (
    CompletionReadbackCollectionRequest,
    BTreeMap<ResourceId, String>,
) {
    let plan = fixture.compilation.executable().execution_plan();
    let attention = plan
        .payload()
        .nodes()
        .iter()
        .find(|n| n.id().as_str() == "node.attention")
        .unwrap();
    let completion = plan
        .payload()
        .nodes()
        .iter()
        .find(|n| n.id().as_str() == "node.ffn_residual")
        .unwrap();
    let output = completion
        .values()
        .iter()
        .find(|v| v.role() == ResolvedValueRole::Output && v.ordinal() == 0)
        .unwrap();
    let component = &output.storage().components()[0];
    let mut names = BTreeMap::from([(component.resource_id().clone(), "output".to_owned())]);
    let mut batches = vec![CompletionReadbackBatchRequest::new(
        (0..ranges.len())
            .map(|p| {
                CompletionReadbackRequest::new(
                    completion.id().clone(),
                    p as u32,
                    component.resource_id().clone(),
                    component.offset_bytes(),
                    HostTransferLayout::new(
                        ElementType::F32,
                        output.tensor().dimensions().last().copied().unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()];
    for state in &fixture.states {
        let value = attention
            .values()
            .iter()
            .find(|v| v.value_id() == &state.value_id)
            .unwrap();
        let component = &value.storage().components()[0];
        names.insert(component.resource_id().clone(), state.id.to_string());
        batches.push(
            CompletionReadbackBatchRequest::new(
                ranges
                    .iter()
                    .enumerate()
                    .map(|(p, range)| {
                        let bytes = match state.capacity_demand {
                            StateCapacityDemand::FixedPerScope => state.tensor.byte_len().unwrap(),
                            StateCapacityDemand::TokenScaled { .. } => {
                                assert_eq!(state.id.as_str(), "state.kv");
                                range.end.div_ceil(16) as u64
                                    * 16
                                    * state.tensor.byte_len().unwrap()
                            }
                        };
                        CompletionReadbackRequest::new_typed(
                            attention.id().clone(),
                            p as u32,
                            component.resource_id().clone(),
                            BufferUsage::State,
                            component.offset_bytes(),
                            HostTransferLayout::new(
                                state.tensor.element_type,
                                bytes / state.tensor.element_type.size_bytes(),
                            )
                            .unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
        );
    }
    let resource = attention.binding_resource().unwrap().clone();
    names.insert(resource.clone(), "binding_parameters".into());
    let row_bytes = if fixture
        .states
        .iter()
        .any(|state| state.id.as_str() == "state.kv")
    {
        88
    } else {
        16
    };
    batches.push(
        CompletionReadbackBatchRequest::new(
            (0..ranges.len())
                .map(|p| {
                    CompletionReadbackRequest::new_typed(
                        attention.id().clone(),
                        p as u32,
                        resource.clone(),
                        BufferUsage::Binding,
                        p as u64 * row_bytes,
                        HostTransferLayout::new(ElementType::U8, row_bytes).unwrap(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    );
    (
        CompletionReadbackCollectionRequest::new(batches).unwrap(),
        names,
    )
}

fn valid_kv(bytes: &[u8], tokens: usize, heads: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(tokens * 2 * heads * 128 * 2);
    for token in 0..tokens {
        for kv in 0..2 {
            for head in 0..heads {
                for dim in 0..128 {
                    let offset = token / 16 * (2 * heads * 128 * 16)
                        + if kv == 0 {
                            head * 128 * 16 + dim / 8 * 16 * 8 + token % 16 * 8 + dim % 8
                        } else {
                            heads * 128 * 16 + head * 128 * 16 + dim * 16 + token % 16
                        };
                    result.extend_from_slice(&bytes[offset * 2..offset * 2 + 2]);
                }
            }
        }
    }
    result
}

fn compare(kind: AttentionKind) {
    let fixtures = [0, 1].map(|_| {
        Fixture::for_family_with_causal_decode_preparation(
            kind,
            true,
            2,
            {
                let family = Family::new(kind).with_maximum_tokens(128);
                if kind == AttentionKind::Causal {
                    family.with_causal_sixteen_token_pages()
                } else {
                    family
                }
            },
            ProgramBindingUploadStrategy::CompactScatter,
            SegmentBindingOracleMode::Disabled,
            SegmentBindingOwnerViewMode::Indexed,
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ferrum_types::CausalDecodePreparationMode::Packed,
        )
    });
    let tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|p| (0..68).map(|t| ((t * 7 + p * 3 + 1) % 32) as u32).collect())
        .collect();
    if kind == AttentionKind::Causal {
        for fixture in &fixtures {
            let state = fixture
                .states
                .iter()
                .find(|s| s.id.as_str() == "state.kv")
                .unwrap();
            assert_eq!(state.tensor.dimensions, [2, 8, 128]);
            assert_eq!(state.tensor.byte_len().unwrap(), 4096);
            assert_eq!(65_536 / state.tensor.byte_len().unwrap(), 16);
        }
    }
    let sessions = fixtures.each_ref().map(|f| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                f.admit_with_ceiling(
                    &format!("resident-program-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    // Start before a blocks16 boundary: controls and the live address count
    // must advance across genuine 16-token physical pages without a new graph.
    let lead = if kind == AttentionKind::Causal { 10 } else { 2 };
    let sinks = [Preparation::default(), Preparation::default()];
    // Ordinary real work establishes state and allocates the reusable slots.
    for position in 0..lead {
        let observations = fixtures
            .iter()
            .zip(&sessions)
            .enumerate()
            .map(|(arm, (f, group))| {
                for (s, t) in group.iter().zip(&tokens) {
                    f.extend(s, Arc::from(&t[..=position]));
                }
                f.execute_participants_with_preparation(
                    &f.lane,
                    &f.reaper,
                    group,
                    &tokens,
                    position..position + 1,
                    Path::Warm,
                    InvocationPreparationStrategy::DecodeSegmentProgram,
                    &sinks[arm],
                )
            })
            .collect::<Vec<_>>();
        observations[0].assert_same(&observations[1]);
    }
    let mut resident = Resident::default();
    let mut hot_count = 0;
    let mut prior_frames = None;
    let mut prior_values = None;
    for position in lead..36 {
        let ranges = vec![position..position + 1; 2];
        for (f, group) in fixtures.iter().zip(&sessions) {
            for (s, t) in group.iter().zip(&tokens) {
                f.extend(s, Arc::from(&t[..=position]));
            }
        }
        if kind == AttentionKind::Causal && position == lead {
            assert_eq!(
                physical_pages(
                    &fixtures[1],
                    &sessions[1][0],
                    Arc::from(&tokens[0][..=position]),
                    position + 1
                ),
                1
            );
        }
        let reference = fixtures[0].execute_participant_ranges_with_preparation(
            &fixtures[0].lane,
            &fixtures[0].reaper,
            &sessions[0],
            &tokens,
            &ranges,
            Path::Replay,
            InvocationPreparationStrategy::DecodeSegmentProgram,
            &sinks[0],
            None,
        );
        reference.assert_same(&reference);
        let (values, hot, frames) =
            execute(&fixtures[1], &mut resident, &sessions[1], &tokens, &ranges)
                .expect("same cohort and append-only physical growth must remain eligible");
        if kind == AttentionKind::Causal && [16, 32].contains(&position) {
            assert!(hot, "cross-page execution must use the resident path");
            assert_eq!(
                physical_pages(
                    &fixtures[1],
                    &sessions[1][0],
                    Arc::from(&tokens[0][..=position]),
                    position + 1
                ),
                position / 16 + 1
            );
        }
        assert_eq!(
            values, reference.values,
            "complete output/state/KV comparison"
        );
        if let Some(old) = prior_frames {
            assert_ne!(frames, old, "fresh frame IDs must advance");
        }
        if let Some(old) = prior_values {
            assert_ne!(values, old, "fresh inputs must advance actual state/output");
        }
        prior_frames = Some(frames);
        prior_values = Some(values);
        hot_count += usize::from(hot);
        println!(
            "{}",
            serde_json::json!({"kind":"resident_decode_program_correctness", "attention":format!("{kind:?}"),
            "position":position,"participants":2,"resident_hot":hot,"full_output_state_bits_equal":true,
            "resident_encodes":fixtures[1]._composition._runtime.resident_decode_program_encodes()})
        );
        if kind == AttentionKind::GatedDelta && hot_count >= 3 {
            break;
        }
    }
    assert!(
        hot_count >= 2,
        "fixture must actually submit multiple whole-program hot uses"
    );
    let program = resident
        .program
        .as_ref()
        .expect("cold compile must have installed a program");
    assert!(program.is_idle());
    let subset = ExecutionBatchParticipants::new(vec![Arc::clone(&sessions[1][0])]).unwrap();
    let subset_request = request(&fixtures[1], &subset, &tokens[..1], &[0..1]);
    assert!(
        matches!(
            program.try_begin_use(&subset, &subset_request).unwrap(),
            ResidentDecodeDecision::Fallback
        ),
        "changed cohort must not borrow the old complete program's authority"
    );
    assert!(program.is_idle());
    drop(resident);
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA and actual resident whole-program execution"]
fn resident_decode_program_gdn_matches_ordinary_with_fresh_frames() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA, unchanged graph and real append-only KV growth"]
fn resident_decode_program_causal_refreshes_sixteen_token_pages_without_recapture() {
    compare(AttentionKind::Causal);
}
