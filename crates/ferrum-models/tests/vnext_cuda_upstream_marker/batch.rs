use super::*;
use ferrum_types::InvocationPreparationStrategy;

struct NoHostTiming;
impl DeviceSubmissionTimingSink for NoHostTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        unreachable!()
    }
}
impl SubmissionWaveDispatchTimingSink for NoHostTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        unreachable!()
    }
}
struct NoPreparation;
impl InvocationPreparationSink for NoPreparation {
    fn record_preparation(&self, _: InvocationPreparationStats) {
        // Existing fixture callers do not observe preparation diagnostics.
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Path {
    Eager,
    Warm,
    Replay,
    EagerBoundary,
}

pub struct BatchObservation {
    pub values: BTreeMap<(u32, String), Vec<u8>>,
    types: BTreeMap<String, ElementType>,
    pub binding_rows: Option<Vec<Vec<u8>>>,
    /// Observations use caller order; raw rows retain the device's row index.
    pub caller_to_canonical_participant: Vec<u32>,
    pub segment_published: bool,
    pub reusable_program_id: Option<DeviceReusableExecutionProgramId>,
    pub participant_frames: Vec<ExecutionFrameId>,
}

impl BatchObservation {
    pub fn assert_same(&self, other: &Self) {
        assert_eq!(
            self.values, other.values,
            "same-policy output/state/KV bits"
        );
        for ((_, name), bytes) in &self.values {
            let values: Vec<f32> = match self.types[name] {
                ElementType::F16 => bytes
                    .chunks_exact(2)
                    .map(|v| f16::from_le_bytes(v.try_into().unwrap()).to_f32())
                    .collect(),
                ElementType::F32 => bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect(),
                other => panic!("unexpected state dtype: {other:?}"),
            };
            assert!(
                !values.is_empty() && values.iter().all(|v| v.is_finite()),
                "{name}: non-finite/empty observation"
            );
            assert!(
                values.iter().any(|v| *v != 0.0),
                "{name}: fixture must exercise nonzero data"
            );
        }
    }

    pub fn dump(&self, kind: AttentionKind, rows: u32, range: Range<usize>, path: Path) {
        let observations: Vec<_> = self.values.iter().map(|((participant, name), bytes)| {
            serde_json::json!({"participant":participant,"value":name,"bytes":bytes.len(),
                "sha256":format!("{:x}",Sha256::digest(bytes)),
                // Small output dumps preserve actual F32 bit patterns. State
                // hashes cover every compared valid element, not just a sample.
                "output_bits":if name == "output" { Some(bytes.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>()) } else { None }})
        }).collect();
        println!(
            "{}",
            serde_json::json!({"kind":"attention_provider_observation","attention":format!("{kind:?}"),
            "participants":rows,"source_start":range.start,"source_end":range.end,"path":format!("{path:?}"),"values":observations})
        );
    }
}

impl Fixture {
    pub fn execute_participants(
        &self,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
    ) -> BatchObservation {
        self.execute_participants_on_lane(&self.lane, &self.reaper, sessions, tokens, range, path)
    }

    /// A second lane from the same PlanRuntimeResources shares the actual
    /// admitted weights and persistent flags, while retaining its own stream.
    pub fn execute_participants_on_lane(
        &self,
        lane: &Arc<ExecutionLane<Runtime>>,
        reaper: &Arc<CompletionReaper<Runtime>>,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
    ) -> BatchObservation {
        self.execute_participants_with_preparation(
            lane,
            reaper,
            sessions,
            tokens,
            range,
            path,
            InvocationPreparationStrategy::Full,
            &NoPreparation,
        )
    }

    pub fn execute_participants_with_preparation<P: InvocationPreparationSink>(
        &self,
        lane: &Arc<ExecutionLane<Runtime>>,
        reaper: &Arc<CompletionReaper<Runtime>>,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        range: Range<usize>,
        path: Path,
        preparation_strategy: InvocationPreparationStrategy,
        preparation_sink: &P,
    ) -> BatchObservation {
        let ranges = vec![range; sessions.len()];
        self.execute_participant_ranges_with_preparation(
            lane,
            reaper,
            sessions,
            tokens,
            &ranges,
            path,
            preparation_strategy,
            preparation_sink,
            None,
        )
    }

    pub fn execute_participant_ranges_with_preparation<P: InvocationPreparationSink>(
        &self,
        lane: &Arc<ExecutionLane<Runtime>>,
        reaper: &Arc<CompletionReaper<Runtime>>,
        sessions: &[Arc<SequenceSession<Runtime>>],
        tokens: &[Arc<[u32]>],
        ranges: &[Range<usize>],
        path: Path,
        preparation_strategy: InvocationPreparationStrategy,
        preparation_sink: &P,
        binding_row_bytes: Option<usize>,
    ) -> BatchObservation {
        assert_eq!(sessions.len(), tokens.len());
        assert_eq!(sessions.len(), ranges.len());
        let participants = sessions.len() as u32;
        let batch = ExecutionBatchParticipants::new(sessions.to_vec()).unwrap();
        // Admission canonicalizes session order, which can differ from caller
        // order after slots are released and reused. Move each session's token
        // and range inputs together; restore caller order only in observations.
        let caller_indices: Vec<_> = batch
            .sessions()
            .iter()
            .map(|canonical| {
                sessions
                    .iter()
                    .position(|session| Arc::ptr_eq(session, canonical))
                    .expect("batch retains the exact supplied sessions")
            })
            .collect();
        let canonical_tokens: Vec<_> = caller_indices
            .iter()
            .map(|&index| Arc::clone(&tokens[index]))
            .collect();
        let canonical_ranges: Vec<_> = caller_indices
            .iter()
            .map(|&index| ranges[index].clone())
            .collect();
        let sessions = batch.sessions();
        let tokens = canonical_tokens.as_slice();
        let ranges = canonical_ranges.as_slice();
        let spans = tokens
            .iter()
            .zip(ranges)
            .map(|(t, range)| token_span(Arc::clone(t), range.clone()))
            .collect();
        let mut request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(spans).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        if let Some(bucket) = &self.reusable_bucket {
            request = request.with_reusable_execution_bucket(bucket.clone());
        }
        let step = loop {
            match batch.try_begin_step(request.clone(), lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => break step,
                StepResourceAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                    self.resources
                        .maintain_for_admission_deferred(&reason)
                        .unwrap(),
                ),
                StepResourceAdmissionDecision::PermanentRejected(reason) => {
                    panic!("batch admission: {reason:?}")
                }
            }
        };
        let wave = loop {
            match step
                .try_prepare_full_plan_submission_wave(
                    Arc::new(step.work_shape().clone()),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => break wave,
                StepSubmissionWaveAdmissionDecision::BackingDeferred(d) => {
                    require_progress(d.maintain().unwrap())
                }
                StepSubmissionWaveAdmissionDecision::Deferred(reason) => require_progress(
                    self.resources
                        .maintain_for_admission_deferred(&reason)
                        .unwrap(),
                ),
                other => panic!("batch wave rejected: {:?}", std::mem::discriminant(&other)),
            }
        };
        let active: Vec<_> = sessions
            .iter()
            .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
            .collect();
        let executable = self.compilation.executable();
        let plan = executable.execution_plan();
        let identity = match preparation_strategy {
            InvocationPreparationStrategy::Full => {
                OperationDispatch::bind_submission_wave_identity(
                    executable,
                    active.iter(),
                    &wave,
                    lane,
                )
            }
            InvocationPreparationStrategy::IdentityProjection
            | InvocationPreparationStrategy::DecodeSegment => {
                let topology =
                    OperationDispatch::compile_submission_wave_identity(executable, lane).unwrap();
                OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
                    &topology,
                    active.iter(),
                    &wave,
                    lane,
                    preparation_strategy,
                )
            }
        }
        .unwrap();
        let uploads: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(participant, t)| {
                let range = &ranges[participant];
                SubmissionWaveInputUpload::new(
                    id("node.embedding"),
                    participant as u32,
                    0,
                    range.start as u64 * 4,
                    HostTransferLayout::new(ElementType::U32, range.len() as u64).unwrap(),
                    t[range.clone()]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                )
                .unwrap()
            })
            .collect();
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
        let mut readbacks = vec![CompletionReadbackBatchRequest::new(
            (0..participants)
                .map(|p| {
                    CompletionReadbackRequest::new(
                        completion.id().clone(),
                        p,
                        component.resource_id().clone(),
                        component.offset_bytes(),
                        HostTransferLayout::new(
                            ElementType::F32,
                            ranges[p as usize].len() as u64
                                * output.tensor().dimensions().last().copied().unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()];
        let mut types = BTreeMap::from([("output".to_owned(), ElementType::F32)]);
        for state in &self.states {
            let value = attention
                .values()
                .iter()
                .find(|v| v.value_id() == &state.value_id)
                .unwrap();
            let component = &value.storage().components()[0];
            let bytes_for = |range: &Range<usize>| match state.capacity_demand {
                StateCapacityDemand::FixedPerScope => state.tensor.byte_len().unwrap(),
                StateCapacityDemand::TokenScaled { .. } => {
                    assert_eq!(state.id.as_str(), "state.kv");
                    assert_eq!(state.tensor.element_type, ElementType::F16);
                    assert_eq!(state.tensor.dimensions, vec![2, 2, 128]);
                    // The real F16 KV provider uses VllmBlocks16 for these
                    // dimensions. Read complete physical blocks, then select
                    // all valid K/V positions; padding is not numerical state.
                    range.end.div_ceil(16) as u64 * 16 * 2 * 2 * 128 * 2
                }
            };
            names.insert(component.resource_id().clone(), state.id.to_string());
            types.insert(state.id.to_string(), state.tensor.element_type);
            readbacks.push(
                CompletionReadbackBatchRequest::new(
                    (0..participants)
                        .map(|p| {
                            CompletionReadbackRequest::new_typed(
                                attention.id().clone(),
                                p,
                                component.resource_id().clone(),
                                BufferUsage::State,
                                component.offset_bytes(),
                                HostTransferLayout::new(
                                    state.tensor.element_type,
                                    bytes_for(&ranges[p as usize])
                                        / state.tensor.element_type.size_bytes(),
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
        if let Some(row_bytes) = binding_row_bytes {
            let resource = attention
                .binding_resource()
                .expect("binding workspace")
                .clone();
            names.insert(resource.clone(), "binding_parameters".to_owned());
            readbacks.push(
                CompletionReadbackBatchRequest::new(
                    (0..participants)
                        .map(|p| {
                            CompletionReadbackRequest::new_typed(
                                attention.id().clone(),
                                p,
                                resource.clone(),
                                BufferUsage::Binding,
                                u64::from(p) * row_bytes as u64,
                                HostTransferLayout::new(ElementType::U8, row_bytes as u64).unwrap(),
                            )
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap(),
            );
        }
        let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
            self.providers.providers(),
            executable,
            &wave,
            lane,
        )
        .unwrap();
        // Eager boundaries are part of a reusable program, not a missing
        // program ID. Inspect the actual sealed catalog after submission.
        let attention_index = wave
            .nodes()
            .iter()
            .position(|n| n.node_id() == attention.id())
            .unwrap() as u32;
        let handle = if matches!(path, Path::Replay) {
            let program_id = program_id
                .as_ref()
                .expect("single-token attention must authorize reusable topology");
            let catalog = lane.reusable_execution_catalog().unwrap();
            let program = catalog
                .programs()
                .iter()
                .find(|p| p.program_id() == program_id)
                .expect("actual warmed reusable program");
            assert!(
                program.is_determinism_ready(),
                "replay must have complete bindings"
            );
            assert!(
                !program
                    .eager_boundary_node_indices()
                    .contains(&attention_index),
                "single-token attention must belong to a resident segment"
            );
            let bindings: Vec<_> = plan
                .payload()
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, n)| n.binding_resource().is_some())
                .map(|(i, _)| i as u32)
                .collect();
            for binding in bindings {
                assert!(
                    program.per_wave_binding_node_indices().contains(&binding),
                    "real attention bindings cannot disappear"
                );
            }
            // MarkerV2 also has admitted Plan flags initialized by a dynamic
            // prelude. Those nodes need not own a per-sequence binding buffer,
            // so the actual capture catalog may include more binding nodes.
            // It must still cover the full FFN compute segment for ReplayOnly.
            let swiglu_index = wave
                .nodes()
                .iter()
                .position(|n| n.node_id().as_str() == "node.swiglu")
                .unwrap() as u32;
            assert!(!program
                .eager_boundary_node_indices()
                .contains(&swiglu_index));
            for node in ["node.attention", "node.swiglu"] {
                let node_plan = plan
                    .payload()
                    .nodes()
                    .iter()
                    .find(|n| n.id().as_str() == node)
                    .unwrap();
                let arithmetic = node_plan
                    .provider_resources()
                    .projection_numerics()
                    .unwrap()
                    .contract();
                let uses_marker = matches!(
                    arithmetic.schema_version,
                    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM
                        | COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
                        | COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
                );
                if arithmetic.schema_version
                    == COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ
                {
                    assert!(
                        G32MmqPrefillPolicy::g32_partition(
                            step.work_shape().immediate_tokens(),
                            step.work_shape()
                                .participant_token_ranges()
                                .iter()
                                .map(|range| range.immediate_token_range()),
                        )
                        .unwrap(),
                        "this hybrid replay check only authorizes the actual small-row G32 branch"
                    );
                }
                let has_dependency = program
                    .retained_plan_dependencies()
                    .iter()
                    .any(|d| d.node_id().as_str() == node);
                if !uses_marker {
                    assert!(
                        !has_dependency,
                        "actual small-row G32 must not borrow MarkerV2 flag dependencies"
                    );
                    continue;
                }
                assert!(
                    program
                        .retained_plan_dependencies()
                        .iter()
                        .any(|dependency| dependency.node_id().as_str() == node),
                    "selected MarkerV2 compute must retain its exact weight/bank dependencies"
                );
            }
            assert!(program.segments().iter().any(|segment| segment.contains_node(attention_index) && segment.contains_node(swiglu_index)),
                "Plan dependency prefixes must not split otherwise contiguous attention/norm/FFN compute");
            println!(
                "{}",
                serde_json::json!({"kind":"marker_replay_catalog","participants":participants,
                "source_start":ranges[0].start,"source_end":ranges[0].end,"source_ranges":ranges,"bindings":program.per_wave_binding_node_indices(),
                "eager_boundary_nodes":program.eager_boundary_node_indices(),
                "retained_plan_dependencies":program.retained_plan_dependencies().len(),
                "segments":program.segments()})
            );
            OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
                self.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &uploads,
                program,
                SubmissionExecutionPolicy::determinism_replayed(0xa5),
                preparation_strategy,
                &NoHostTiming,
                preparation_sink,
                wave,
                lane,
                reaper,
            )
            .unwrap()
            .into_parts()
            .0
        } else {
            let policy = if matches!(path, Path::Eager) {
                SubmissionExecutionPolicy::determinism_eager(0x5a)
            } else {
                SubmissionExecutionPolicy::adaptive()
            };
            OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
                self.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &uploads,
                policy,
                preparation_strategy,
                &NoHostTiming,
                preparation_sink,
                wave,
                lane,
                reaper,
            )
            .unwrap()
            .into_parts()
            .0
        };
        let segment_publication = handle.take_segment_binding_publication();
        let receipt = match handle
            .wait_with_readback_collection(
                CompletionReadbackCollectionRequest::new(readbacks).unwrap(),
            )
            .unwrap()
        {
            CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
            other => panic!("attention readback did not terminate: {other:?}"),
        };
        let ready_segment =
            segment_publication.and_then(|ticket| ticket.bind_completion(receipt.completion()));
        if matches!(path, Path::EagerBoundary) {
            let program_id = program_id
                .as_ref()
                .expect("bucket-backed multi-token wave has a topology identity");
            let catalog = lane.reusable_execution_catalog().unwrap();
            let program = catalog
                .programs()
                .iter()
                .find(|p| p.program_id() == program_id)
                .expect("actual multi-token catalog evidence");
            assert!(
                program
                    .eager_boundary_node_indices()
                    .contains(&attention_index),
                "multi-token FP16 KV must retain its EagerBoundary"
            );
        }
        let mut values = BTreeMap::new();
        let mut binding_rows = binding_row_bytes.map(|_| vec![Vec::new(); participants as usize]);
        for disposition in receipt.dispositions() {
            let CompletionReadbackDisposition::Succeeded(result) = disposition else {
                panic!("attention readback failed: {disposition:?}")
            };
            let name = names[result.request().resource_id()].clone();
            let participant = result.request().participant_index() as usize;
            let caller_participant = caller_indices[participant];
            if name == "binding_parameters" {
                binding_rows.as_mut().unwrap()[caller_participant] = result.bytes().to_vec();
                continue;
            }
            let bytes = if name == "state.kv" {
                valid_kv_positions(result.bytes(), ranges[participant].end)
            } else {
                result.bytes().to_vec()
            };
            assert!(values
                .insert(
                    (
                        u32::try_from(caller_participant).expect("validated participant count"),
                        name,
                    ),
                    bytes,
                )
                .is_none());
        }
        assert_eq!(
            values.len(),
            participants as usize * (names.len() - usize::from(binding_rows.is_some()))
        );
        drop((receipt, handle, identity, active));
        let retirement = step.try_retire_normal().unwrap();
        let canonical_frames: Vec<_> = retirement
            .participants()
            .iter()
            .map(|participant| participant.assignment().frame_id())
            .collect();
        let mut participant_frames = canonical_frames.clone();
        let mut caller_to_canonical_participant = vec![0; caller_indices.len()];
        for (canonical, &caller) in caller_indices.iter().enumerate() {
            participant_frames[caller] = canonical_frames[canonical];
            caller_to_canonical_participant[caller] =
                u32::try_from(canonical).expect("validated participant count");
        }
        let segment_published = ready_segment
            .map(|ready| ready.publish(&retirement).unwrap())
            .unwrap_or(false);
        BatchObservation {
            values,
            types,
            binding_rows,
            caller_to_canonical_participant,
            segment_published,
            reusable_program_id: program_id,
            participant_frames,
        }
    }
}

/// Canonical token/K-or-V/head/dim ordering from this fixture's real vLLM
/// blocks16 F16 state ABI. Reads every valid position, including later blocks.
fn valid_kv_positions(bytes: &[u8], tokens: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(tokens * 1024);
    for token in 0..tokens {
        let block = token / 16 * (2 * 2 * 128 * 16);
        for kv in 0..2 {
            for head in 0..2 {
                for dim in 0..128 {
                    let offset = block
                        + if kv == 0 {
                            head * 128 * 16 + (dim / 8) * 16 * 8 + (token % 16) * 8 + dim % 8
                        } else {
                            2 * 128 * 16 + head * 128 * 16 + dim * 16 + token % 16
                        };
                    output.extend_from_slice(&bytes[offset * 2..offset * 2 + 2]);
                }
            }
        }
    }
    output
}
