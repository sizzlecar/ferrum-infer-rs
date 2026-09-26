//! Future selected head routes versus actual eager Metal commands and outputs.
use super::*;
use std::num::NonZeroU64;

#[path = "head_cost_route/family.rs"]
mod family;
use family::{Head, HeadFamily, OUTPUTS};

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("head cost observation must not enable device timing");
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("head cost observation must not enable dispatch profiling");
    }
}

fn fixture(head: Head) -> (Fixture, Vec<f32>) {
    let kind = AttentionKind::GatedDelta;
    let definition = HeadFamily {
        base: Family::new(kind),
        head,
    };
    let states = definition.base.states();
    let profile = definition.base.profile_id();
    let prepared = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
        .unwrap();
    // Family preparation canonicalizes component order. The deterministic
    // source depends on that order, so use the exact prepared schema consumed
    // by Fixture::from_prepared_family, not a second unprepared definition.
    let schema = prepared.weight_schema();
    let source = crate::family::Weights::new(schema);
    let component = schema
        .components
        .iter()
        .find(|component| component.id.as_str() == "component.head")
        .unwrap();
    let payload = source.component(component).unwrap();
    let mut weights = vec![0.0; (OUTPUTS * HIDDEN) as usize];
    ferrum_kernels::gguf_blocks::GgufBlockFormat::Q6K
        .decode(payload.bytes(), &mut weights)
        .unwrap();
    (
        Fixture::from_prepared_family_with_composition(
            kind,
            prepared,
            states,
            FixtureExecutionMode::Eager,
            None,
            1,
            BTreeMap::new(),
            crate::composition_with_capture(
                kind,
                ferrum_types::SloStructuredCostCapture::HostSettledV1,
            ),
        ),
        weights,
    )
}

fn capacity(fixture: &Fixture) -> [u64; 4] {
    let snapshot = CapacitySnapshot::observe(&fixture.resources);
    [
        snapshot.resident_bytes,
        snapshot.free_bytes,
        snapshot.pending_growth_bytes,
        snapshot.budget_claimed_bytes,
    ]
}

fn compare(head: Head, prefix_lengths: &[usize]) {
    let (fixture, weights) = fixture(head);
    let original_tokens = prefix_lengths
        .iter()
        .enumerate()
        .map(|(participant, &prefix)| {
            (0..prefix + 1)
                .map(|index| ((index * 7 + participant * 3 + 1) % 32) as u32)
                .collect::<Arc<[u32]>>()
        })
        .collect::<Vec<_>>();
    let admitted = original_tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| fixture.admit(&format!("head.{index}"), Arc::clone(tokens)))
        .collect::<Vec<_>>();
    // ExecutionBatchParticipants canonicalizes identities. Work, uploads and
    // CPU expectations must follow that order, including unequal row lengths.
    let batch = ExecutionBatchParticipants::new(admitted.clone()).unwrap();
    let sessions = batch.sessions().to_vec();
    let tokens = sessions
        .iter()
        .map(|session| {
            let source = admitted
                .iter()
                .position(|value| Arc::ptr_eq(value, session))
                .unwrap();
            Arc::clone(&original_tokens[source])
        })
        .collect::<Vec<_>>();
    let prefixes = tokens
        .iter()
        .map(|tokens| tokens.len() - 1)
        .collect::<Vec<_>>();
    for decode in [false, true] {
        let ranges = prefixes
            .iter()
            .map(|&prefix| {
                if decode {
                    prefix..prefix + 1
                } else {
                    0..prefix
                }
            })
            .collect::<Vec<_>>();
        execute(
            &fixture, &batch, &sessions, &tokens, &ranges, &weights, head,
        );
    }
    drop(batch);
    for session in sessions {
        session.try_abort_if_quiescent().unwrap();
    }
}

fn execute(
    fixture: &Fixture,
    batch: &ExecutionBatchParticipants<Runtime>,
    sessions: &[Arc<SequenceSession<Runtime>>],
    tokens: &[Arc<[u32]>],
    ranges: &[Range<usize>],
    weights: &[f32],
    head: Head,
) {
    let executable = fixture.compilation.executable();
    let rows = ranges
        .iter()
        .zip(tokens)
        .map(|(range, tokens)| OperationCostWorkRow {
            offset: range.start as u64,
            count: NonZeroU64::new(range.len() as u64).unwrap(),
            full_input_tokens: NonZeroU64::new(tokens.len() as u64).unwrap(),
        })
        .collect::<Vec<_>>();
    // The query uses the real bound providers BEFORE Step admission; it must
    // neither allocate backing nor borrow future resource/submission authority.
    let before = capacity(fixture);
    let routes = fixture
        .providers
        .providers()
        .iter()
        .map(|provider| {
            provider
                .eager_cost_route(executable, &rows)
                .unwrap()
                .expect("supported real Q6_K head program")
        })
        .collect::<Vec<_>>();
    assert_eq!(capacity(fixture), before);
    let nodes = executable.execution_plan().payload().nodes();
    let head_index = nodes
        .iter()
        .position(|node| node.id().as_str() == "node.head")
        .unwrap();
    let [predicted_head] = routes[head_index].commands() else {
        panic!("one head command expected")
    };
    assert_eq!(predicted_head.native_operation(), head.label());
    let packed = sessions.len() > 1;
    let decode = ranges.iter().all(|range| range.len() == 1);
    assert_eq!(
        predicted_head.batching(),
        if packed {
            DeviceBatchingForm::Packed
        } else {
            DeviceBatchingForm::Scalar
        }
    );
    assert_eq!(
        predicted_head.transfer_command_count(),
        if packed {
            sessions.len() as u64 * if decode { 1 } else { 2 }
        } else {
            0
        }
    );
    assert_eq!(predicted_head.participant_count() as usize, sessions.len());
    assert_eq!(
        predicted_head.token_count(),
        ranges.iter().map(|range| range.len() as u64).sum::<u64>()
    );
    let work = batch
        .bind_work_shape(
            tokens
                .iter()
                .zip(ranges)
                .map(|(tokens, range)| token_span(Arc::clone(tokens), range.clone()))
                .collect(),
        )
        .unwrap();
    let request = StepResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = loop {
        match batch
            .try_begin_step(request.clone(), &fixture.lane)
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            StepResourceAdmissionDecision::PermanentRejected(reason) => {
                panic!("head step rejected: {reason:?}")
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
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepSubmissionWaveAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            other => panic!(
                "head wave could not be prepared: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    };
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        active.iter(),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let uploads = tokens
        .iter()
        .zip(ranges)
        .enumerate()
        .map(|(participant, (tokens, range))| {
            SubmissionWaveInputUpload::new(
                id("node.embedding"),
                participant as u32,
                0,
                range.start as u64 * ElementType::U32.size_bytes(),
                HostTransferLayout::new(ElementType::U32, range.len() as u64).unwrap(),
                tokens[range.clone()]
                    .iter()
                    .flat_map(|token| token.to_le_bytes())
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (handle, attribution) = OperationDispatch::encode_and_submit_wave_with_cost_observation(
        fixture.providers.providers(),
        executable,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &uploads,
        SubmissionExecutionPolicy::adaptive(),
        None,
        &NoTiming,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap()
    .into_parts();
    let attribution = attribution.expect("actual logical attribution with device timing Off");
    let compute = attribution
        .device()
        .commands()
        .iter()
        .filter(|command| command.command_phase() == DeviceCommandPhase::Compute)
        .collect::<Vec<_>>();
    assert_eq!(
        compute.len(),
        routes.len(),
        "no missing or undeclared compute commands"
    );
    for (node, (actual, route)) in compute.iter().zip(&routes).enumerate() {
        let [predicted] = route.commands() else {
            panic!("one command per selected node")
        };
        assert_eq!(actual.node_index(), Some(node as u32));
        assert_eq!(actual.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(actual.native_op_id(), predicted.native_operation());
        assert_eq!(actual.command_phase(), predicted.phase());
        assert_eq!(actual.participant_start(), predicted.participant_start());
        assert_eq!(actual.participant_count(), predicted.participant_count());
        assert_eq!(actual.token_count(), predicted.token_count());
        assert_eq!(actual.batching_form(), predicted.batching());
        assert_eq!(
            actual.compute_dispatch_count(),
            predicted.compute_dispatch_count()
        );
        assert_eq!(
            actual.transfer_command_count(),
            predicted.transfer_command_count()
        );
        assert_eq!(actual.reusable_graph_node_count(), None);
        if node == head_index {
            crate::assert_metal_algorithm_work(
                actual.statistical_evidence().expect("actual head evidence"),
                predicted
                    .statistical_evidence()
                    .expect("future head evidence"),
                ferrum_types::SloStructuredCostCapture::HostSettledV1,
            );
        }
    }
    // One group per resource, each containing ALL participants. Completion
    // returns participant-local Step spans even after a nonzero decode offset.
    let requests = [("node.attention", HIDDEN), ("node.head", OUTPUTS)]
        .into_iter()
        .map(|(node_id, width)| {
            let node = nodes
                .iter()
                .find(|node| node.id().as_str() == node_id)
                .unwrap();
            let output = node
                .values()
                .iter()
                .find(|binding| binding.role() == ResolvedValueRole::Output)
                .unwrap();
            let component = &output.storage().components()[0];
            CompletionReadbackBatchRequest::new(
                ranges
                    .iter()
                    .enumerate()
                    .map(|(participant, range)| {
                        let rows = if node_id == "node.head" {
                            1
                        } else {
                            range.len() as u64
                        };
                        CompletionReadbackRequest::new(
                            node.id().clone(),
                            participant as u32,
                            component.resource_id().clone(),
                            component.offset_bytes(),
                            HostTransferLayout::new(ElementType::F32, rows * width).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let CompletionReadbackBatchObservation::Terminal(receipt) = handle
        .wait_with_readback_collection(CompletionReadbackCollectionRequest::new(requests).unwrap())
        .unwrap()
    else {
        panic!("head wave did not reach terminal completion")
    };
    assert_eq!(receipt.dispositions().len(), 2 * sessions.len());
    let data = receipt
        .dispositions()
        .iter()
        .map(|result| {
            let CompletionReadbackDisposition::Succeeded(output) = result else {
                panic!("head readback failed: {result:?}")
            };
            (
                (
                    output.request().node_id().as_str(),
                    output.request().participant_index(),
                ),
                output.bytes(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (participant, range) in ranges.iter().enumerate() {
        let input = data[&("node.attention", participant as u32)]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let output = data[&("node.head", participant as u32)]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(input.len(), range.len() * HIDDEN as usize);
        assert_eq!(output.len(), OUTPUTS as usize);
        assert!(input.iter().all(|value| value.is_finite()));
        assert!(input.iter().any(|value| value.abs() > 1e-7));
        let last = &input[(range.len() - 1) * HIDDEN as usize..];
        for (column, &actual) in output.iter().enumerate() {
            let products = last
                .iter()
                .zip(&weights[column * HIDDEN as usize..(column + 1) * HIDDEN as usize])
                .map(|(&input, &weight)| {
                    f64::from(head.operand(input)) * f64::from(head.operand(weight))
                });
            let (expected, magnitude) = products.fold((0.0, 0.0), |(sum, abs), product| {
                (sum + product, abs + product.abs())
            });
            // F64 dot of the operation's declared operands. Scale the F32
            // accumulation tolerance with the actual reduction magnitude.
            let tolerance = 1e-6 + 8.0 * f64::from(f32::EPSILON) * HIDDEN as f64 * magnitude;
            assert!(actual.is_finite() && (f64::from(actual) - expected).abs() <= tolerance,
                "{head:?} participant {participant}, range {range:?}, output {column}: {actual} vs {expected}, tolerance {tolerance}");
        }
    }
    drop(receipt);
    drop(handle);
    drop(attribution);
    drop(identity);
    drop(active);
    step.try_retire_normal().unwrap();
}

#[test]
fn selected_metal_strict_head_route_matches_scalar_and_packed_prefill_decode() {
    compare(Head::Strict, &[3]);
    compare(Head::Strict, &[2, 5]);
}

#[test]
fn selected_metal_f16_operand_head_route_matches_scalar_and_packed_prefill_decode() {
    compare(Head::HalfOperands, &[3]);
    compare(Head::HalfOperands, &[2, 5]);
}
