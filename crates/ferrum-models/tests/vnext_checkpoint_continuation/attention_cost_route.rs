//! Real selected attention routes share canonical batching and semantic checks.
use super::*;
use std::num::NonZeroU64;

#[cfg(any(feature = "cuda", test))]
#[path = "attention_cost_route/kv_readback.rs"]
mod kv_readback;

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("attention cost observation must not enable device timing");
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("attention cost observation must not enable dispatch profiling");
    }
}

fn capacity(fixture: &Fixture) -> [u64; 8] {
    let snapshot = CapacitySnapshot::observe(&fixture.resources);
    [
        snapshot.resident_bytes,
        snapshot.free_bytes,
        snapshot.pending_growth_bytes,
        snapshot.budget_claimed_bytes,
        snapshot.checkpoint_claims,
        snapshot.checkpoint_bytes,
        snapshot.non_checkpoint_claims,
        snapshot.non_checkpoint_bytes,
    ]
}

#[derive(Clone, Copy)]
pub(super) enum ExpectedRoute {
    GatedDelta,
    Causal { int8: bool },
}

pub(super) fn compare(kind: AttentionKind, lengths: &[usize], expected: ExpectedRoute) {
    #[cfg(not(feature = "cuda"))]
    let fixture = Fixture::new(kind);
    #[cfg(feature = "cuda")]
    let fixture = {
        // Cost observation does not require checkpoint support. In particular,
        // CUDA FP16 causal KV has no checkpoint declaration. Keep the original
        // checkpoint fixture's capability assertion intact for its own tests.
        let definition = Family::new(kind);
        let states = definition.states();
        let profile = definition.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
            .unwrap();
        Fixture::from_prepared_family(kind, family, states, false, None, 1, BTreeMap::new())
    };
    let executable = fixture.compilation.executable();
    let node_index = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|node| node.id().as_str() == "node.attention")
        .unwrap();
    let tokens = lengths
        .iter()
        .enumerate()
        .map(|(participant, &count)| {
            (0..count)
                .map(|index| ((index * 7 + participant * 3 + 1) % 32) as u32)
                .collect::<Arc<[u32]>>()
        })
        .collect::<Vec<_>>();
    #[cfg(not(feature = "cuda"))]
    let supported = !kind.hadamard();
    #[cfg(feature = "cuda")]
    let supported = kind != AttentionKind::CausalInt8;
    let observed = execute(&fixture, &tokens, node_index, supported, expected, true);
    let baseline = execute(&fixture, &tokens, node_index, supported, expected, false);
    for (observed, baseline) in observed.iter().zip(&baseline) {
        observed.assert_state_nonzero();
        observed.assert_same(
            baseline,
            "cost attribution keeps the real attention output/state",
        );
    }
}

fn predict(
    fixture: &Fixture,
    node_index: usize,
    lengths: &[usize],
    expected_supported: bool,
    expected: ExpectedRoute,
) -> Option<OperationCostRoute> {
    let executable = fixture.compilation.executable();
    let rows = lengths
        .iter()
        .map(|&count| OperationCostWorkRow {
            offset: 0,
            count: NonZeroU64::new(count as u64).unwrap(),
            full_input_tokens: NonZeroU64::new(count as u64).unwrap(),
        })
        .collect::<Vec<_>>();
    // The canonical request order is known, but no Step or wave exists.
    // The selected provider still receives no invocation or resource authority.
    let before = capacity(fixture);
    let predicted = fixture.providers.providers()[node_index]
        .eager_cost_route(executable, &rows)
        .unwrap();
    assert_eq!(
        before,
        capacity(fixture),
        "query must not change physical claims/backing"
    );
    assert_eq!(predicted.is_some(), expected_supported);
    if let Some(route) = &predicted {
        #[cfg(feature = "cuda")]
        let command = {
            let [host, command] = route.commands() else {
                panic!("CUDA native attention has a binding upload followed by compute")
            };
            assert!(!host.host_only());
            assert_eq!(host.phase(), DeviceCommandPhase::DynamicBinding);
            assert_eq!(host.transfer_command_count(), lengths.len() as u64);
            command
        };
        #[cfg(not(feature = "cuda"))]
        let command = match (expected, route.commands()) {
            (ExpectedRoute::GatedDelta, [command]) => command,
            (ExpectedRoute::Causal { int8 }, [host, command]) => {
                assert!(host.host_only());
                assert_eq!(
                    host.native_operation(),
                    "vnext_causal_paged_attention_bindings"
                );
                assert_eq!(host.phase(), DeviceCommandPhase::DynamicBinding);
                assert_eq!(host.participant_start(), 0);
                assert_eq!(host.participant_count() as usize, lengths.len());
                assert_eq!(host.token_count(), lengths.iter().sum::<usize>() as u64);
                assert_eq!(
                    command.transfer_command_count(),
                    if int8 { lengths.len() as u64 } else { 0 }
                );
                command
            }
            _ => panic!("declared attention command slots differ from the selected contract"),
        };
        assert_eq!(command.participant_count() as usize, lengths.len());
        assert_eq!(command.token_count(), lengths.iter().sum::<usize>() as u64);
        assert_eq!(
            command.batching(),
            if lengths.len() == 1 {
                if cfg!(feature = "cuda") && matches!(expected, ExpectedRoute::GatedDelta) {
                    DeviceBatchingForm::ParticipantLoop
                } else {
                    DeviceBatchingForm::Scalar
                }
            } else {
                DeviceBatchingForm::Packed
            }
        );
    }
    predicted
}

fn execute(
    fixture: &Fixture,
    tokens: &[Arc<[u32]>],
    node_index: usize,
    expected_supported: bool,
    expected: ExpectedRoute,
    observe: bool,
) -> Vec<Observation> {
    let sessions = tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| {
            fixture.admit(
                &format!("attention-cost-{observe}-{index}"),
                Arc::clone(tokens),
            )
        })
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    // The resource contract sorts participant authorities independently of
    // admission order. Use its actual order for work, bindings and uploads.
    let order = batch
        .sessions()
        .iter()
        .map(|canonical| {
            sessions
                .iter()
                .position(|original| Arc::ptr_eq(original, canonical))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let tokens = order
        .iter()
        .map(|&index| Arc::clone(&tokens[index]))
        .collect::<Vec<_>>();
    let sessions = batch.sessions().to_vec();
    let predicted = if observe {
        predict(
            fixture,
            node_index,
            &tokens.iter().map(|tokens| tokens.len()).collect::<Vec<_>>(),
            expected_supported,
            expected,
        )
    } else {
        None
    };
    let work = batch
        .bind_work_shape(
            tokens
                .iter()
                .map(|tokens| token_span(Arc::clone(tokens), 0..tokens.len()))
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
                panic!("attention step rejected: {reason:?}")
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
                "attention wave could not be prepared: {:?}",
                std::mem::discriminant(&other)
            ),
        }
    };
    assert!(
        wave.claimed_backing().program_binding_layout().is_none(),
        "fixture uses adjacent eager binding slots, not coalesced replay prelude slots"
    );
    let executable = fixture.compilation.executable();
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
        .enumerate()
        .map(|(participant, tokens)| {
            SubmissionWaveInputUpload::new(
                id("node.embedding"),
                participant as u32,
                0,
                0,
                HostTransferLayout::new(ElementType::U32, tokens.len() as u64).unwrap(),
                tokens
                    .iter()
                    .flat_map(|token| token.to_le_bytes())
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (handle, attribution) = if observe {
        OperationDispatch::encode_and_submit_wave_with_cost_observation(
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
        .into_parts()
    } else {
        (
            OperationDispatch::encode_and_submit_wave_with_inputs(
                fixture.providers.providers(),
                executable,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &uploads,
                wave,
                &fixture.lane,
                &fixture.reaper,
            )
            .unwrap(),
            None,
        )
    };
    if observe {
        let attribution = attribution
            .as_ref()
            .expect("actual native logical attribution with timing Off");
        #[cfg(feature = "cuda")]
        assert!(attribution
            .device()
            .graph_evidence()
            .unwrap()
            .proves_unconfigured_eager());
        let actual = attribution
            .device()
            .commands()
            .iter()
            .filter(|command| {
                command.node_index() == Some(node_index as u32)
                    && command.command_phase() == DeviceCommandPhase::Compute
            })
            .collect::<Vec<_>>();
        let [actual] = actual.as_slice() else {
            panic!("one real GDN compute command")
        };
        assert_eq!(actual.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(actual.reusable_graph_node_count(), None);
        assert!(actual.compute_dispatch_count() > 0);
        assert_eq!(actual.participant_count() as usize, tokens.len());
        assert_eq!(
            actual.token_count(),
            tokens.iter().map(|tokens| tokens.len()).sum::<usize>() as u64
        );
        if let Some(route) = &predicted {
            #[cfg(not(feature = "cuda"))]
            if matches!(expected, ExpectedRoute::Causal { .. }) {
                let embedding = attribution
                    .device()
                    .commands()
                    .iter()
                    .find(|command| {
                        command.node_index() == Some(0)
                            && command.command_phase() == DeviceCommandPhase::Compute
                    })
                    .expect("the real embedding compute precedes attention");
                assert_eq!(
                    actual.command_index(),
                    embedding.command_index() + route.commands().len() as u32
                );
                let binding_index = embedding.command_index() + 1;
                assert!(
                    attribution
                        .device()
                        .commands()
                        .iter()
                        .all(|command| command.command_index() != binding_index),
                    "the real host binding slot must not become invented GPU work"
                );
            }
            #[cfg(feature = "cuda")]
            {
                let physical = attribution
                    .device()
                    .commands()
                    .iter()
                    .filter(|command| command.node_index() == Some(node_index as u32))
                    .collect::<Vec<_>>();
                assert_eq!(physical.len(), route.commands().len());
                for (actual, predicted) in physical.iter().zip(route.commands()) {
                    assert_eq!(actual.native_op_id(), predicted.native_operation());
                    assert_eq!(actual.command_phase(), predicted.phase());
                    assert_eq!(actual.batching_form(), predicted.batching());
                    assert_eq!(actual.participant_start(), predicted.participant_start());
                    assert_eq!(actual.participant_count(), predicted.participant_count());
                    assert_eq!(actual.token_count(), predicted.token_count());
                    assert_eq!(
                        actual.compute_dispatch_count(),
                        predicted.compute_dispatch_count()
                    );
                    assert_eq!(
                        actual.transfer_command_count(),
                        predicted.transfer_command_count()
                    );
                }
            }
            let predicted = route.commands().last().unwrap();
            #[cfg(not(feature = "cuda"))]
            if matches!(
                expected,
                ExpectedRoute::GatedDelta | ExpectedRoute::Causal { int8: false }
            ) {
                assert!(
                    predicted.statistical_evidence().is_some(),
                    "plain F16 attention has a complete real selected-kernel chain"
                );
                assert_eq!(actual.statistical_evidence(), predicted.statistical_evidence(),
                    "selected PSO/order/numeric work must agree with real native attribution, not only umbrella command counts");
            }
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
        }
    }
    let node = &executable.execution_plan().payload().nodes()[node_index];
    let mut groups = Vec::new();
    let mut names = BTreeMap::new();
    #[cfg(feature = "cuda")]
    let mut vllm_kv_layouts = BTreeMap::new();
    let output = node
        .values()
        .iter()
        .find(|binding| binding.role() == ResolvedValueRole::Output && binding.ordinal() == 0)
        .unwrap();
    let component = &output.storage().components()[0];
    names.insert(component.resource_id().clone(), "output".to_owned());
    groups.push(
        CompletionReadbackBatchRequest::new(
            tokens
                .iter()
                .enumerate()
                .map(|(index, tokens)| {
                    CompletionReadbackRequest::new(
                        node.id().clone(),
                        index as u32,
                        component.resource_id().clone(),
                        component.offset_bytes(),
                        HostTransferLayout::new(fixture.output_type, tokens.len() as u64 * HIDDEN)
                            .unwrap(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    );
    for state in &fixture.states {
        if matches!(expected, ExpectedRoute::GatedDelta) {
            assert_eq!(state.capacity_demand, StateCapacityDemand::FixedPerScope);
        }
        let binding = node
            .values()
            .iter()
            .find(|value| value.value_id() == &state.value_id)
            .unwrap();
        let component = &binding.storage().components()[0];
        assert!(names
            .insert(component.resource_id().clone(), state.id.to_string())
            .is_none());
        #[cfg(feature = "cuda")]
        let vllm_layout = if matches!(expected, ExpectedRoute::Causal { int8: false }) {
            let layout = kv_readback::VllmKvLayout::from_tensor(&state.tensor);
            assert_eq!(component.offset_bytes(), 0);
            vllm_kv_layouts.insert(component.resource_id().clone(), layout);
            Some(layout)
        } else {
            None
        };
        groups.push(
            CompletionReadbackBatchRequest::new(
                (0..tokens.len())
                    .map(|index| {
                        CompletionReadbackRequest::new_typed(
                            node.id().clone(),
                            index as u32,
                            component.resource_id().clone(),
                            BufferUsage::State,
                            component.offset_bytes(),
                            HostTransferLayout::new(
                                state.tensor.element_type,
                                match state.capacity_demand {
                                    StateCapacityDemand::FixedPerScope => {
                                        state.tensor.byte_len().unwrap()
                                    }
                                    StateCapacityDemand::TokenScaled {
                                        bytes_per_token, ..
                                    } => {
                                        #[cfg(feature = "cuda")]
                                        if let Some(layout) = vllm_layout {
                                            assert_eq!(layout.bytes_per_token(), bytes_per_token);
                                            layout.readback_bytes(tokens[index].len())
                                        } else {
                                            bytes_per_token
                                                .checked_mul(tokens[index].len() as u64)
                                                .unwrap()
                                        }
                                        #[cfg(not(feature = "cuda"))]
                                        {
                                            bytes_per_token
                                                .checked_mul(tokens[index].len() as u64)
                                                .unwrap()
                                        }
                                    }
                                } / state.tensor.element_type.size_bytes(),
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
    let CompletionReadbackBatchObservation::Terminal(receipt) = handle
        .wait_with_readback_collection(CompletionReadbackCollectionRequest::new(groups).unwrap())
        .unwrap()
    else {
        panic!("attention execution did not reach terminal readback")
    };
    assert_eq!(
        receipt.completion().fence_timing().timing_mode(),
        DeviceTimingMode::Off
    );
    assert!(matches!(
        receipt.completion().submission_timing(),
        DeviceTimingMeasurement::NotRequested
    ));
    assert!(receipt.readback_timings().is_none());
    let mut values = (0..tokens.len())
        .map(|_| BTreeMap::new())
        .collect::<Vec<_>>();
    for disposition in receipt.dispositions() {
        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
            panic!("attention readback failed: {disposition:?}")
        };
        let request = output.request();
        let name = names.get(request.resource_id()).unwrap();
        let bytes = output.bytes().to_vec();
        #[cfg(feature = "cuda")]
        let bytes = if let Some(layout) = vllm_kv_layouts.get(request.resource_id()) {
            layout.gather(&bytes, tokens[request.participant_index() as usize].len())
        } else {
            bytes
        };
        assert!(values[request.participant_index() as usize]
            .insert(name.clone(), bytes)
            .is_none());
    }
    for values in &values {
        assert_eq!(values.len(), names.len());
    }
    let state_types = fixture
        .states
        .iter()
        .map(|state| (state.id.to_string(), state.tensor.element_type))
        .chain(std::iter::once(("output".to_owned(), fixture.output_type)))
        .collect::<BTreeMap<_, _>>();
    drop((receipt, handle, attribution, identity, active));
    step.try_retire_normal().unwrap();
    for session in sessions {
        session.try_abort_if_quiescent().unwrap();
    }
    let mut ordered = order.into_iter().zip(values).collect::<Vec<_>>();
    ordered.sort_by_key(|(original, _)| *original);
    ordered
        .into_iter()
        .map(|(_, values)| Observation {
            values,
            state_types: state_types.clone(),
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[path = "attention_behavior.rs"]
mod attention_behavior;
