use super::*;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
fn close(resources: Arc<PlanRuntimeResources<CpuDeviceRuntime>>) {
    assert!(matches!(
        PlanRuntimeResources::close(resources)
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
}
fn progress(outcome: DynamicDeferredMaintenanceOutcome) {
    assert!(
        !matches!(
            outcome,
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
        ),
        "no outstanding owner can release memory: {outcome:?}"
    );
}

pub(super) fn assert_shared_installation(
    f: &Fixture,
    composition: &CpuVNextComposition,
    union: &Arc<PreparedProductionModel>,
    installed: ProgramPlanCompilation,
    main: ProgramPlanCompilation,
    draft: ProgramPlanCompilation,
    verify: ProgramPlanCompilation,
) {
    let host = host();
    let mapped = union.managed_safetensors().unwrap();
    let source = mapped.weight_source(&host);
    // Canonical BF16 norm +1 is materialized with the original charged adapter.
    let norm = union
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .find(|c| c.external_names == ["mtp.pre_fc_norm_hidden.weight"])
        .unwrap();
    let payload = source.component(norm).unwrap();
    assert_eq!(
        payload.bytes(),
        f16::from_f32(1.125).to_le_bytes().repeat(16)
    );
    drop(payload);
    let plan = installed.executable().execution_plan();
    let StaticProvisioning::Required(permit) = plan
        .provision_static(composition.runtime().clone(), id("request.mtp.install"))
        .unwrap()
        .into_provisioning()
    else {
        panic!("union requires Stored installation")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.mtp.install"),
        id("transaction.mtp.install"),
    );
    let reserved = ResourceTransaction::begin(
        RuntimeResourceDriver::new(composition.runtime().clone()).unwrap(),
        identity,
        permit,
    )
    .unwrap()
    .reserve()
    .unwrap();
    let committed = match reserved.commit() {
        Ok(v) => v,
        Err(ResourceCommitTransitionError::Recoverable(e)) => panic!("commit {:?}", e.failure()),
        Err(ResourceCommitTransitionError::Poisoned(e)) => {
            panic!("commit unknown {:?}", e.failure())
        }
    };
    let initialized = committed
        .initialize_static_managed(
            union.family(),
            plan,
            &source,
            &host,
            StaticInitializationPolicy::new(4096, 16).unwrap(),
        )
        .unwrap();
    let (installer, weights) = initialized
        .into_shared_plan_runtime(plan)
        .unwrap_or_else(|e| panic!("publish {:?}", e.failure()));
    close(installer);
    let installed_bytes = weights.evidence().physical_bytes();
    assert_eq!(composition.runtime().resident_bytes(), installed_bytes);
    let main_resources = weights
        .bind_plan(
            main.executable().execution_plan(),
            composition.runtime().clone(),
            id("request.mtp.main"),
        )
        .unwrap();
    let draft_resources = weights
        .bind_plan(
            draft.executable().execution_plan(),
            composition.runtime().clone(),
            id("request.mtp.draft"),
        )
        .unwrap();
    let verify_resources = weights
        .bind_plan(
            verify.executable().execution_plan(),
            composition.runtime().clone(),
            id("request.mtp.verify"),
        )
        .unwrap();
    assert_eq!(
        composition.runtime().resident_bytes(),
        installed_bytes,
        "attach allocates no second static installation"
    );
    // The two actual compiled graphs keep the root embedding/head components,
    // offsets and resource IDs. A role switch cannot silently repack them.
    let root_weights = union
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .filter(|c| {
            c.external_names == ["model.embed_tokens.weight"]
                || c.external_names == ["lm_head.weight"]
        })
        .count();
    assert_eq!(root_weights, 2);
    let main_output = run(composition, &main, main_resources.clone(), &host, false);
    close(main_resources);
    let verify_output = run(composition, &verify, verify_resources.clone(), &host, false);
    close(verify_resources);
    // The selected-scan and bounded head preserve the ordinary main's last
    // greedy decision. MainVerify also returns the accepted final-norm rows.
    assert_eq!(&verify_output[0].bytes()[4..8], main_output[1].bytes());
    assert_eq!(verify_output[1].len(), 2 * 16 * 2);
    let draft_output = run(composition, &draft, draft_resources.clone(), &host, true);
    close(draft_resources);
    assert!(matches!(
        weights
            .close()
            .unwrap_or_else(|e| panic!("weights close {:?}", e.failure())),
        SharedComponentWeightsCloseOutcome::Closed(_)
    ));
    assert_eq!(composition.runtime().resident_bytes(), 0);
    assert_draft_numeric(&draft_output);
    // Output owners stay readable after both graph and shared-weight retirement.
    assert!(!main_output.is_empty());
    drop((main_output, draft_output, verify_output));
    drop(source);
    drop(mapped);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let _ = f; // The selected checkpoint remains alive for this whole lifecycle.
}

fn input_bytes(name: &str) -> (ElementType, Vec<u8>) {
    let f16s = |xs: Vec<f32>| {
        xs.into_iter()
            .flat_map(|v| f16::from_f32(v).to_le_bytes())
            .collect::<Vec<_>>()
    };
    match name {
        "value.input.token_ids" | model::TOKEN_IDS => (
            ElementType::U32,
            [1u32, 2, 3]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        ),
        model::ACCEPTED_HIDDEN => (
            ElementType::F16,
            f16s(
                (0..48)
                    .map(|i| if i % 16 == 4 { 2. } else { -0.25 })
                    .collect(),
            ),
        ),
        "value.input.visual_rows" => (ElementType::F16, vec![0; 96]),
        "value.input.visual_replace_mask" => (ElementType::U8, vec![0; 3]),
        "value.input.rope_cos" | model::ROPE_COS => (
            ElementType::F16,
            f16s(vec![0.75, 0.5, 0.25, 0.5, 0.75, 0.25]),
        ),
        "value.input.rope_sin" | model::ROPE_SIN => (
            ElementType::F16,
            f16s(vec![0.25, 0.5, 0.75, 0.5, 0.25, 0.75]),
        ),
        "value.input.greedy_token_mask" => (ElementType::U8, vec![1; 32]),
        "value.input.greedy_repetition_token_ids" => {
            (ElementType::U32, 0u32.to_le_bytes().to_vec())
        }
        "value.input.greedy_repetition_offsets" => (ElementType::U32, vec![0; 8]),
        "value.input.greedy_repetition_penalty" => (ElementType::F32, 1f32.to_le_bytes().to_vec()),
        other => panic!("unknown input {other}"),
    }
}

fn run(
    composition: &CpuVNextComposition,
    compilation: &ProgramPlanCompilation,
    resources: Arc<PlanRuntimeResources<CpuDeviceRuntime>>,
    host: &HostMemoryAccount,
    is_draft: bool,
) -> Vec<ManagedHostBytes> {
    let executable = compilation.executable();
    let selected = executable
        .execution_plan()
        .requires_selected_prefix_commit();
    let tokens = [1u32, 2, 3];
    let span = TokenSpanWork::from_token_ids(&tokens, 0..3)
        .unwrap()
        .with_checkpoint_tokens(Arc::from(tokens))
        .unwrap();
    let work = ResourceWorkShape::single(span.clone()).unwrap();
    let binding = resources.trusted_runtime_binding().unwrap();
    let request = loop {
        match binding
            .try_admit_request(
                RequestResourceAdmissionRequest::new(
                    work.clone(),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
                id("run.mtp.execute"),
                id(if selected {
                    "request.mtp.verify.execute"
                } else if is_draft {
                    "request.mtp.draft.execute"
                } else {
                    "request.mtp.main.execute"
                }),
            )
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(r) => break r,
            RequestResourceAdmissionDecision::BackingDeferred(d) => progress(d.maintain().unwrap()),
            RequestResourceAdmissionDecision::Deferred(d) => {
                progress(resources.maintain_for_admission_deferred(&d).unwrap())
            }
            RequestResourceAdmissionDecision::PermanentRejected(d) => {
                panic!("request rejected: {d:?}")
            }
        }
    };
    let sequence = loop {
        match request
            .try_admit_sequence(
                SequenceResourceAdmissionRequest::new(
                    work.clone(),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        {
            SequenceResourceAdmissionDecision::Admitted(r) => break r,
            SequenceResourceAdmissionDecision::BackingDeferred(d) => {
                progress(d.maintain().unwrap())
            }
            SequenceResourceAdmissionDecision::Deferred(d) => {
                progress(resources.maintain_for_admission_deferred(&d).unwrap())
            }
            SequenceResourceAdmissionDecision::PermanentRejected(d) => {
                panic!("sequence rejected: {d:?}")
            }
        }
    };
    let session = sequence.open_session().unwrap();
    if selected {
        let inputs = [
            "value.input.visual_rows",
            "value.input.visual_replace_mask",
            "value.input.rope_cos",
            "value.input.rope_sin",
        ]
        .into_iter()
        .map(|name| {
            let (_, bytes) = input_bytes(name);
            let mut charged = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                .unwrap();
            charged.as_mut().copy_from_slice(&bytes);
            let value = id(name);
            SequenceConditioningInput::new(
                host,
                value,
                compilation.value_tensors()[&id(name)].clone(),
                SequenceConditioningCoverage::Exact,
                vec![SequenceConditioningSegment::new(0, charged.freeze()).unwrap()],
            )
            .unwrap()
        })
        .collect();
        session
            .bind_conditioning_inputs(Arc::new(SequenceConditioningInputs::new(inputs).unwrap()))
            .unwrap();
    }
    let batch = ExecutionBatchParticipants::new(vec![session.clone()]).unwrap();
    let lane = resources.create_execution_lane().unwrap();
    let step_request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = loop {
        match batch.try_begin_step(step_request.clone(), &lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(s) => break s,
            StepResourceAdmissionDecision::BackingDeferred(d) => progress(d.maintain().unwrap()),
            StepResourceAdmissionDecision::Deferred(d) => {
                progress(resources.maintain_for_admission_deferred(&d).unwrap())
            }
            StepResourceAdmissionDecision::PermanentRejected(d) => panic!("step rejected: {d:?}"),
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
            StepSubmissionWaveAdmissionDecision::Prepared(w) => break w,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(d) => {
                progress(d.maintain().unwrap())
            }
            StepSubmissionWaveAdmissionDecision::Deferred(d) => {
                progress(resources.maintain_for_admission_deferred(&d).unwrap())
            }
            StepSubmissionWaveAdmissionDecision::PermanentRejected(d) => {
                panic!("wave rejected: {d:?}")
            }
            _ => panic!("wave has a state hazard"),
        }
    };
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        std::iter::once(&active),
        &wave,
        &lane,
    )
    .unwrap();
    let mut inputs = if is_draft {
        vec![
            (model::TOKEN_IDS, "node.mtp.embedding", 0),
            (model::ACCEPTED_HIDDEN, "node.mtp.norm_hidden", 0),
            (model::ROPE_COS, "node.mtp.attention", 9),
            (model::ROPE_SIN, "node.mtp.attention", 10),
        ]
    } else {
        vec![
            ("value.input.token_ids", "node.embedding", 0),
            ("value.input.visual_rows", "node.visual_embedding_select", 1),
            (
                "value.input.visual_replace_mask",
                "node.visual_embedding_select",
                2,
            ),
            ("value.input.rope_cos", "node.layer.3.attention", 9),
            ("value.input.rope_sin", "node.layer.3.attention", 10),
            ("value.input.greedy_token_mask", "node.greedy_token", 1),
            (
                "value.input.greedy_repetition_token_ids",
                "node.greedy_token",
                2,
            ),
            (
                "value.input.greedy_repetition_offsets",
                "node.greedy_token",
                3,
            ),
            (
                "value.input.greedy_repetition_penalty",
                "node.greedy_token",
                4,
            ),
        ]
    };
    if selected {
        inputs.retain(|(_, node, _)| *node != "node.greedy_token");
    }
    let uploads = inputs
        .iter()
        .map(|(name, node, ordinal)| {
            let (dtype, bytes) = input_bytes(name);
            let mut allocation = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                .unwrap();
            allocation.as_mut().copy_from_slice(&bytes);
            SubmissionWaveInputUpload::managed_range(
                id(*node),
                0,
                *ordinal,
                0,
                HostTransferLayout::new(dtype, bytes.len() as u64 / dtype.size_bytes()).unwrap(),
                allocation.freeze(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let names = if selected {
        vec![model::MAIN_GREEDY_TOKENS, model::MAIN_HIDDEN_TAIL]
    } else if is_draft {
        vec![model::GREEDY_TOKENS, model::HIDDEN_TAIL]
    } else {
        vec!["value.output.logits", "value.output.greedy_token"]
    };
    let readbacks = names
        .iter()
        .map(|name| {
            let readback = executable
                .execution_plan()
                .completion_checkpoint_readback_for_work(&id(*name), 0, &work)
                .unwrap();
            let output = host
                .allocate_zeroed(
                    HostMemoryClass::Output,
                    readback.output_layout().byte_len().unwrap() as usize,
                )
                .unwrap();
            (readback, output)
        })
        .collect();
    let providers = composition.registry().bind_plan(executable).unwrap();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        executable,
        &identity,
        std::iter::once(&active),
        DeviceTimingMode::Off,
        &uploads,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (observation, outputs) = handle
        .wait_with_managed_readbacks(readbacks)
        .unwrap()
        .into_parts();
    let CompletionObservation::Terminal(receipt) = observation else {
        panic!("not terminal")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let outputs = outputs.unwrap().unwrap();
    drop((receipt, handle, identity, active, uploads));
    if selected {
        let step = step.try_retire_normal().unwrap_err().into_step();
        let prepared = step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::All])
            .unwrap();
        let SelectedStateSubmission::Submitted(handle) =
            reaper.submit_selected_state_commit(prepared).unwrap()
        else {
            panic!("selected copy was not submitted");
        };
        let observation = handle.wait().unwrap();
        let CompletionObservation::Terminal(receipt) = observation else {
            panic!("selected copy not terminal");
        };
        assert!(matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        drop((receipt, handle));
        step.try_retire_selected().unwrap();
    } else {
        step.try_retire_normal().unwrap();
    }
    drop(batch);
    session.try_complete().unwrap();
    drop((
        session, sequence, request, binding, lane, reaper, providers, resources,
    ));
    outputs
}

fn assert_draft_numeric(outputs: &[ManagedHostBytes]) {
    assert_eq!(
        outputs[0].bytes(),
        [4u32, 4]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    // Independent F64 formula for this sparse source. Attention-out and FFN
    // down weights are zero; their actual KV/provider work still executes.
    // The two pre-norms, concatenation, one FC, residual and final norm remain
    // observable, including F16 rounding at every declared boundary.
    let round = |v: f64| f16::from_f64(v).to_f64();
    let norm = |v: &[f64]| {
        let rms = (v.iter().map(|x| x * x).sum::<f64>() / 16. + 1e-6).sqrt();
        v.iter().map(|x| round(x / rms * 1.125)).collect::<Vec<_>>()
    };
    let h = norm(
        &(0..16)
            .map(|j| if j == 4 { 2. } else { -0.25 })
            .collect::<Vec<_>>(),
    );
    let expected = [2usize, 3]
        .into_iter()
        .flat_map(|token| {
            let e = norm(
                &(0..16)
                    .map(|j| if j == token { 1. } else { 0. })
                    .collect::<Vec<_>>(),
            );
            norm(
                &e.iter()
                    .zip(&h)
                    .map(|(a, b)| round(a + 2. * b))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    for (actual, expected) in outputs[1].bytes().chunks_exact(2).zip(expected) {
        let actual = f16::from_le_bytes([actual[0], actual[1]]).to_f64();
        assert!(
            (actual - expected).abs() <= 0.003,
            "actual={actual}, expected={expected}"
        );
    }
}
