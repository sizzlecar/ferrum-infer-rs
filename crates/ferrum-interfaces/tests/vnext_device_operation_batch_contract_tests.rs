mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn chunked_token_span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[10, 11, 12, 13], 2..3).unwrap()
}

#[test]
fn token_work_separates_current_backing_from_request_fit_ceiling() {
    let span = TokenSpanWork::from_token_ids_with_fit(&[10, 11, 12], 2..3, 128).unwrap();
    assert_eq!(span.immediate_tokens(), 1);
    assert_eq!(span.full_input_tokens(), 3);
    assert_eq!(span.fit_input_tokens(), 128);

    let shape = ResourceWorkShape::single(span).unwrap();
    assert_eq!(shape.immediate_tokens(), 1);
    assert_eq!(shape.fit_tokens(), 128);
    assert!(TokenSpanWork::from_token_ids_with_fit(&[10, 11, 12], 2..3, 2).is_err());
}

fn admit_batch_step(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![chunked_token_span(); batch.len() as usize])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("batch step admission did not converge"),
        }
    }
    unreachable!("bounded batch step admission returns or panics")
}

#[test]
fn profiled_step_admission_reports_completed_phases_without_changing_decision() {
    let Fixture { plan_resources, .. } = fixture();
    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.profiled-step",
        "request.device-operation.profiled-step",
    );
    let session = resources.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![session]).unwrap();
    let lane = plan_resources.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![chunked_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();

    for attempt in 0..=3 {
        let mut phases = Vec::new();
        let decision = batch
            .try_begin_step_profiled(request.clone(), &lane, |phase, _| phases.push(phase))
            .unwrap();
        match decision {
            StepResourceAdmissionDecision::Admitted(_) => {
                assert_eq!(
                    phases,
                    [
                        StepResourceAdmissionProfilePhase::AuthorityAndPolicyValidate,
                        StepResourceAdmissionProfilePhase::DemandEvaluate,
                        StepResourceAdmissionProfilePhase::BackingClaim,
                        StepResourceAdmissionProfilePhase::LogicalCapacityClaim,
                        StepResourceAdmissionProfilePhase::TransactionValidateAndFingerprint,
                        StepResourceAdmissionProfilePhase::FrameCaptureAndLease,
                    ]
                );
                return;
            }
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                assert_eq!(
                    phases,
                    [
                        StepResourceAdmissionProfilePhase::AuthorityAndPolicyValidate,
                        StepResourceAdmissionProfilePhase::DemandEvaluate,
                        StepResourceAdmissionProfilePhase::BackingClaim,
                    ]
                );
                deferred.maintain().unwrap();
            }
            _ => panic!("profiled step admission did not converge"),
        }
    }
    unreachable!("bounded profiled step admission returns or panics")
}

fn admit_batch_invocation(
    _plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![
            chunked_token_span();
            step.participant_count() as usize
        ])
        .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => return invocation,
            InvocationResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("batch invocation admission did not converge"),
        }
    }
    unreachable!("bounded batch invocation admission returns or panics")
}

#[test]
fn participant_buffer_validation_rejects_same_fingerprint_capacity_drift_before_encode() {
    dispatch_with_descriptor_switch_during_first_participant(
        BufferDescriptorRuntimeSwitch::DifferentCapacity,
    );
}

#[test]
fn participant_buffer_validation_accepts_equal_descriptor_at_a_different_address() {
    dispatch_with_descriptor_switch_during_first_participant(
        BufferDescriptorRuntimeSwitch::EqualCopy,
    );
}

fn dispatch_with_descriptor_switch_during_first_participant(switch: BufferDescriptorRuntimeSwitch) {
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture();
    let resources = (0..2)
        .map(|index| {
            logical_resources(
                &plan_resources,
                &format!("run.device-operation.descriptor-switch.{index}"),
                &format!("request.device-operation.descriptor-switch.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = plan_resources.create_execution_lane().unwrap();
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let step = admit_batch_step(&batch, &lane);
    let node = &plan.payload().nodes()[0];
    let invocation = admit_batch_invocation(&plan_resources, &step, node.id());
    let identities = step
        .participant_frames()
        .zip(&active_bindings)
        .enumerate()
        .map(|(index, (frame, active))| {
            operation_identity(
                &plan,
                active,
                frame.frame_id(),
                NodeInvocationId::try_from(index as u64 + 1).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let identity = OperationDispatch::bind_batch_identity(
        &resolved,
        identities,
        &active_bindings,
        &invocation,
        &lane,
    )
    .unwrap();
    let reaper = CompletionReaper::new();

    // Arm only after admission and identity binding. This fixture has no state
    // initialization, so the resource event occurs while building the first
    // participant's views, after invocation-wide descriptor validation. The
    // second participant must inspect the runtime's newly returned object.
    runtime_trace.lock().unwrap().switch_descriptor_on_buffer = Some(switch);
    let outcome = OperationDispatch::encode_and_submit(
        &provider,
        &resolved,
        &identity,
        &active_bindings,
        invocation,
        &lane,
        &reaper,
    );
    assert_eq!(
        runtime_trace.lock().unwrap().switched_descriptor_on_buffer,
        Some(switch)
    );
    assert!(!std::ptr::eq(runtime.descriptor(), &runtime.descriptor));
    assert_eq!(
        runtime.descriptor().runtime_implementation_fingerprint,
        runtime.descriptor.runtime_implementation_fingerprint
    );
    match switch {
        BufferDescriptorRuntimeSwitch::DifferentCapacity => {
            assert_ne!(
                runtime.descriptor().total_memory_bytes,
                runtime.descriptor.total_memory_bytes
            );
            assert!(matches!(outcome, Err(OperationDispatchError::Contract(_))));
            assert_eq!(provider_trace.lock().unwrap().encode_calls, 0);
            assert_eq!(runtime_trace.lock().unwrap().submit_calls, 0);
        }
        BufferDescriptorRuntimeSwitch::EqualCopy => {
            assert_eq!(runtime.descriptor(), &runtime.descriptor);
            let handle = outcome.unwrap();
            assert_eq!(provider_trace.lock().unwrap().encode_calls, 1);
            assert_eq!(provider_trace.lock().unwrap().last_participant_count, 2);
            assert_eq!(runtime_trace.lock().unwrap().submit_calls, 1);
            assert!(matches!(
                handle.wait().unwrap(),
                CompletionObservation::Terminal(_)
            ));
        }
    }
    runtime
        .use_different_capacity_descriptor
        .store(false, Ordering::Release);
    runtime
        .use_equal_descriptor_copy
        .store(false, Ordering::Release);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
}

#[test]
fn thirty_two_participant_dispatch_is_one_physical_submission() {
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture();
    let resources = (0..32)
        .map(|index| {
            logical_resources(
                &plan_resources,
                &format!("run.device-operation.batch32.{index}"),
                &format!("request.device-operation.batch32.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = plan_resources.create_execution_lane().unwrap();
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let step = admit_batch_step(&batch, &lane);
    let node = &plan.payload().nodes()[0];
    let invocation = admit_batch_invocation(&plan_resources, &step, node.id());
    let packed_ranges = invocation.work_shape().participant_token_ranges();
    assert_eq!(packed_ranges.len(), 32);
    for (index, range) in packed_ranges.iter().enumerate() {
        assert_eq!(
            range.participant(),
            invocation.work_shape().participants()[index]
        );
        assert_eq!(
            range.immediate_token_range(),
            index as u64..index as u64 + 1
        );
        assert_eq!(range.immediate_tokens(), 1);
        assert_eq!(range.source_token_range(), 2..3);
        assert_eq!(range.full_input_tokens(), 4);
    }
    let identities = step
        .participant_frames()
        .zip(&active_bindings)
        .enumerate()
        .map(|(index, (frame, active))| {
            operation_identity(
                &plan,
                active,
                frame.frame_id(),
                NodeInvocationId::try_from(index as u64 + 1).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let foreign_lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let provider = registry.bind(&resolved, node.id()).unwrap();
    assert!(OperationDispatch::bind_batch_identity(
        &resolved,
        identities.clone(),
        &active_bindings,
        &invocation,
        &foreign_lane,
    )
    .is_err());
    assert_eq!(provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(runtime_trace.lock().unwrap().submit_calls, 0);
    let batch_identity = OperationDispatch::bind_batch_identity(
        &resolved,
        identities,
        &active_bindings,
        &invocation,
        &lane,
    )
    .unwrap();
    assert_eq!(
        std::mem::size_of::<ExecutionIdentityEnvelope>(),
        std::mem::size_of::<Arc<()>>()
    );
    assert_eq!(
        std::mem::size_of::<BatchOperationParticipantIdentity>(),
        std::mem::size_of::<Arc<()>>()
    );
    assert_eq!(
        std::mem::size_of::<BatchOperationIdentity>(),
        std::mem::size_of::<Arc<()>>()
    );
    let batch_wire = serde_json::to_value(&batch_identity).unwrap();
    assert!(batch_wire.get("batch_step_id").is_some());
    assert!(batch_wire.get("data").is_none());
    assert_eq!(
        batch_wire,
        serde_json::to_value(batch_identity.clone()).unwrap()
    );
    assert_eq!(batch_identity.participants().len(), 32);
    assert_eq!(
        batch_identity
            .single_node()
            .expect("single invocation binds one node")
            .work_shape_fingerprint(),
        invocation.work_shape().fingerprint()
    );
    let handle = OperationDispatch::encode_and_submit(
        &provider,
        &resolved,
        &batch_identity,
        &active_bindings,
        invocation,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(
        std::mem::size_of::<SubmittedOperationReceipt>(),
        std::mem::size_of::<Arc<()>>()
    );
    assert_eq!(
        std::mem::size_of::<SubmittedOperationParticipantReceipt>(),
        std::mem::size_of::<Arc<()>>()
    );
    let submission_wire = serde_json::to_value(handle.receipt()).unwrap();
    assert!(submission_wire.get("slot_id").is_some());
    assert!(submission_wire.get("data").is_none());
    assert_eq!(
        submission_wire,
        serde_json::to_value(handle.receipt().clone()).unwrap()
    );
    assert_eq!(runtime_trace.lock().unwrap().submit_calls, 1);
    let trace = provider_trace.lock().unwrap();
    assert_eq!(trace.encode_calls, 1);
    assert_eq!(trace.last_participant_count, 32);
    assert_eq!(trace.last_work_sequences, 32);
    drop(trace);
    assert_eq!(handle.receipt().participants().len(), 32);
    assert!(handle
        .receipt()
        .participants()
        .iter()
        .all(|participant| participant.batch_submission_fingerprint()
            == handle.receipt().fingerprint()));
    let readbacks = CompletionReadbackBatchRequest::new(
        (0..32)
            .map(|participant_index| {
                CompletionReadbackRequest::new(
                    node.id().clone(),
                    participant_index,
                    id("resource.intermediate"),
                    0,
                    HostTransferLayout::new(ElementType::F32, 4).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let readback = match handle.wait_with_readbacks(readbacks).unwrap() {
        CompletionReadbackBatchObservation::Terminal(readback) => readback,
        other => panic!("32-participant batch did not complete: {other:?}"),
    };
    let completion = readback.completion();
    assert_eq!(completion.participants().len(), 32);
    assert!(completion
        .participants()
        .iter()
        .all(|participant| participant.batch_completion_fingerprint() == completion.fingerprint()));
    assert_eq!(readback.dispositions().len(), 32);
    assert!(readback
        .dispositions()
        .iter()
        .all(|disposition| matches!(disposition, CompletionReadbackDisposition::Succeeded(_))));
    assert_eq!(readback.fingerprint().len(), 64);
    {
        let trace = runtime_trace.lock().unwrap();
        assert_eq!(trace.readback_calls, 32);
        assert_eq!(trace.readback_lengths.iter().sum::<u64>(), 32 * 16);
    }
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(resources);
    drop(reaper);
    drop(foreign_lane);
    drop(lane);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
