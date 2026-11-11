//! Product entrypoints over the selected BF16 archive and the real CPU/Core.
//! Unknown native fences remain covered by the separate Core fault-runtime
//! tests; this fixture does not replace CPU execution with a simulated backend.
use super::*;
use ferrum_execution::driver::{
    CausalAcceptedOutput, CausalComputedWave, CausalInputSchema, CausalInputSegment,
    CausalProgramSession, CausalRequestInputs, CausalTailTwoOutput, CausalTokenInput,
    CausalTokenInputSpec, CausalTokenInputStorage, RuntimeComposition, SharedProgramWeights,
    SharedProgramWeightsCloseOutcome,
};
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use std::{
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_ROWS: u64 = 4;
const PHASE: usize = 1 << 20;

struct Registry(TypedFamilyRegistration<model::Qwen35MtpFamilyProvider>);
impl ModelFamilyRegistry for Registry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.0]
    }
}
struct PhaseDrop(Arc<AtomicBool>);
impl Drop for PhaseDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            24 * PHASE,
            HostMemoryClassLimits {
                input_preprocess_bytes: 6 * PHASE,
                transfer_staging_bytes: 6 * PHASE,
                output_bytes: 6 * PHASE,
                weight_backing_bytes: 6 * PHASE,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn phase(host: &HostMemoryAccount) -> (HostMemoryScope, Arc<AtomicBool>) {
    (
        host.reserve_scope(&[
            (HostMemoryClass::InputPreprocess, PHASE),
            (HostMemoryClass::TransferStaging, PHASE),
            (HostMemoryClass::Output, PHASE),
            (HostMemoryClass::WeightBacking, PHASE),
        ])
        .unwrap(),
        Arc::new(AtomicBool::new(false)),
    )
}
fn policy(catalog: &CapabilityCatalog) -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.mtp.product-wave",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            checkpoint_capacity: None,
            capacity_bytes: CAPACITY,
            reserve_bytes: 0,
            maximum_active_sequences: 2,
            dynamic_storage_profile_order: catalog
                .device()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        },
        AdmissionPolicy {
            maximum_queue_depth: 2,
            maximum_scheduled_tokens: MAX_ROWS * 2,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: Default::default(),
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )
    .unwrap()
}
fn options(family: &PreparedModelFamily) -> ProgramPlanCompileOptions {
    let inputs = family
        .program()
        .inputs()
        .iter()
        .map(|input| {
            let (shape, dtype) = match input.as_str() {
                "value.input.token_ids" | model::TOKEN_IDS => (vec![MAX_ROWS], ElementType::U32),
                "value.input.visual_rows" | model::ACCEPTED_HIDDEN => {
                    (vec![MAX_ROWS, 16], ElementType::F16)
                }
                "value.input.visual_replace_mask" => (vec![MAX_ROWS], ElementType::U8),
                "value.input.rope_cos"
                | "value.input.rope_sin"
                | model::ROPE_COS
                | model::ROPE_SIN => (vec![MAX_ROWS, 2], ElementType::F16),
                "value.input.greedy_token_mask" => (vec![32], ElementType::U8),
                "value.input.greedy_repetition_token_ids" => (vec![1], ElementType::U32),
                "value.input.greedy_repetition_offsets" => (vec![2], ElementType::U32),
                "value.input.greedy_repetition_penalty" => (vec![1], ElementType::F32),
                other => panic!("unexpected external input {other}"),
            };
            (input.clone(), tensor_spec(shape, dtype))
        })
        .collect();
    let mut options = ProgramPlanCompileOptions::new(inputs).unwrap();
    model::configure_compile_options(family, &mut options).unwrap();
    options
}

fn initialize(
    fixture: &Fixture,
    composition: &RuntimeComposition<CpuDeviceRuntime>,
    catalog: &CapabilityCatalog,
    root: &HostMemoryAccount,
    role: Role,
    weights: Option<&SharedProgramWeights<CpuDeviceRuntime>>,
) -> (
    CausalProgramSession<CpuDeviceRuntime>,
    Option<SharedProgramWeights<CpuDeviceRuntime>>,
    Arc<AtomicBool>,
) {
    let defined = model::define_from_sources(fixture.sources.clone(), role).unwrap();
    let prepared = Arc::new(defined.prepare(&id(role.numerical_profile())).unwrap());
    let registry = Registry(TypedFamilyRegistration::new(
        model::Qwen35MtpFamilyProvider::new().unwrap(),
    ));
    let (phase, dropped) = phase(root);
    let account = phase.account();
    let mapped = prepared.managed_safetensors().unwrap();
    let source = mapped.weight_source(&account);
    let input = source.execution_input().unwrap();
    let selection = WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID));
    let compiled = if role == Role::InstallUnion {
        composition.compile_causal_weight_installation(
            input,
            policy(catalog),
            options(prepared.family()),
            selection,
        )
    } else {
        let (token, decisions, hidden) = if role == Role::OneDraft {
            (model::TOKEN_IDS, model::GREEDY_TOKENS, model::HIDDEN_TAIL)
        } else {
            (
                "value.input.token_ids",
                model::MAIN_GREEDY_TOKENS,
                model::MAIN_HIDDEN_TAIL,
            )
        };
        composition.compile_causal_program(
            input,
            policy(catalog),
            options(prepared.family()),
            selection,
            CausalTailTwoOutput::new(id(token), id(decisions), id(hidden)).unwrap(),
        )
    }
    .unwrap();
    let resolve = |policy: &ResolvedRuntimePolicy,
                   catalog: &CapabilityCatalog,
                   compilation: &ProgramPlanCompilation| {
        crate::test_support::native_program::resolve_plan(
            &prepared,
            policy,
            catalog,
            compilation,
            &registry,
            defined.definition(),
            ferrum_types::KvStorageFormat::F16,
            StopPolicy {
                maximum_output_tokens: u32::try_from(MAX_ROWS).unwrap(),
                token_ids: BTreeSet::from([31]),
                strings: vec![],
                collision_policy: StopTokenCollisionPolicy::require_distinct(),
            },
        )
    };
    let static_policy = StaticInitializationPolicy::new(4096, 16).unwrap();
    let (program, published) = if let Some(weights) = weights {
        (
            compiled
                .initialize_with_shared_weights_retained(
                    weights,
                    &account,
                    static_policy,
                    (phase, PhaseDrop(dropped.clone())),
                    resolve,
                )
                .unwrap(),
            None,
        )
    } else {
        let (program, weights) = compiled
            .initialize_shared_weights_retained(
                &account,
                static_policy,
                (phase, PhaseDrop(dropped.clone())),
                resolve,
            )
            .unwrap();
        (program, Some(weights))
    };
    (program, published, dropped)
}

struct Harness {
    _fixture: Fixture,
    runtime: Arc<CpuDeviceRuntime>,
    host: HostMemoryAccount,
    program: CausalProgramSession<CpuDeviceRuntime>,
    draft: CausalProgramSession<CpuDeviceRuntime>,
    weights: SharedProgramWeights<CpuDeviceRuntime>,
    phases: [Arc<AtomicBool>; 2],
}
impl Harness {
    fn new() -> Self {
        let fixture = Fixture::new();
        let (runtime, registry, materializers, _, catalog) =
            CpuVNextComposition::create(id("device.mtp.product-wave"), CAPACITY)
                .unwrap()
                .into_parts()
                .unwrap();
        let composition =
            RuntimeComposition::new(runtime.clone(), registry, materializers, catalog.clone());
        let host = host();
        let (install, weights, phase) = initialize(
            &fixture,
            &composition,
            &catalog,
            &host,
            Role::InstallUnion,
            None,
        );
        let weights = weights.unwrap();
        assert!(install.installation_only());
        assert!(matches!(
            install
                .close()
                .unwrap_or_else(|e| panic!("install close {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(phase.load(Ordering::SeqCst));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        let physical = weights.evidence().physical_bytes();
        assert_eq!(runtime.resident_bytes(), physical);
        let (program, none, a) = initialize(
            &fixture,
            &composition,
            &catalog,
            &host,
            Role::MainVerify,
            Some(&weights),
        );
        assert!(none.is_none());
        let (draft, none, b) = initialize(
            &fixture,
            &composition,
            &catalog,
            &host,
            Role::OneDraft,
            Some(&weights),
        );
        assert!(none.is_none());
        assert_eq!(
            runtime.resident_bytes(),
            physical,
            "both actual consumers attach the one installation"
        );
        assert!(program
            .plan()
            .execution_plan()
            .requires_selected_prefix_commit());
        assert!(!draft
            .plan()
            .execution_plan()
            .requires_selected_prefix_commit());
        for p in [&program, &draft] {
            assert_eq!(
                p.shared_weights_receipt().unwrap().evidence(),
                weights.evidence()
            );
        }
        Self {
            _fixture: fixture,
            runtime,
            host,
            program,
            draft,
            weights,
            phases: [a, b],
        }
    }
    fn close(self) {
        let Self {
            _fixture,
            runtime,
            host,
            program,
            draft,
            weights,
            phases,
        } = self;
        for p in [program, draft] {
            assert!(matches!(
                p.close()
                    .unwrap_or_else(|e| panic!("program close {:?}", e.failure())),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
        }
        assert!(phases.iter().all(|p| p.load(Ordering::SeqCst)));
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(
            runtime.resident_bytes(),
            weights.evidence().physical_bytes()
        );
        assert!(matches!(
            weights
                .close()
                .unwrap_or_else(|e| panic!("weight close {:?}", e.failure())),
            SharedProgramWeightsCloseOutcome::Closed(_)
        ));
        assert_eq!(runtime.resident_bytes(), 0);
        drop(_fixture);
    }
}

fn progress(outcome: DynamicDeferredMaintenanceOutcome) {
    assert!(
        !matches!(
            outcome,
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
        ),
        "no outstanding native owner: {outcome:?}"
    );
}
fn span(history: &[u32], range: Range<u64>) -> TokenSpanWork {
    TokenSpanWork::from_token_ids(
        history,
        usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap(),
    )
    .unwrap()
    .with_checkpoint_tokens(Arc::from(history))
    .unwrap()
}
fn bytes(name: &str, history: &[u32]) -> (ElementType, Vec<u8>) {
    let half = |values: Vec<f32>| {
        values
            .into_iter()
            .flat_map(|v| f16::from_f32(v).to_le_bytes())
            .collect()
    };
    match name {
        "value.input.token_ids" | model::TOKEN_IDS => (
            ElementType::U32,
            history.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        "value.input.visual_rows" => (ElementType::F16, vec![0; MAX_ROWS as usize * 32]),
        "value.input.visual_replace_mask" => (ElementType::U8, vec![0; MAX_ROWS as usize]),
        "value.input.rope_cos" | model::ROPE_COS => {
            (ElementType::F16, half(vec![1.; MAX_ROWS as usize * 2]))
        }
        "value.input.rope_sin" | model::ROPE_SIN => {
            (ElementType::F16, half(vec![0.; MAX_ROWS as usize * 2]))
        }
        model::ACCEPTED_HIDDEN => (
            ElementType::F16,
            half(
                (0..MAX_ROWS as usize * 16)
                    .map(|i| if i % 16 == 4 { 2. } else { -0.25 })
                    .collect(),
            ),
        ),
        other => panic!("unexpected executable input {other}"),
    }
}
fn allocate(host: &HostMemoryAccount, data: &[u8]) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
        .unwrap();
    bytes.as_mut().copy_from_slice(data);
    bytes.freeze()
}
fn session(
    program: &CausalProgramSession<CpuDeviceRuntime>,
    history: &[u32],
    name: &str,
) -> Arc<SequenceSession<CpuDeviceRuntime>> {
    let resources = program.resources();
    let work = ResourceWorkShape::single(span(history, 0..history.len() as u64)).unwrap();
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
                id("run.mtp.product"),
                id(format!("request.mtp.{name}")),
            )
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(r) => break r,
            RequestResourceAdmissionDecision::BackingDeferred(d) => progress(d.maintain().unwrap()),
            RequestResourceAdmissionDecision::Deferred(d) => {
                progress(resources.maintain_for_admission_deferred(&d).unwrap())
            }
            RequestResourceAdmissionDecision::PermanentRejected(d) => panic!("request {d:?}"),
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
            SequenceResourceAdmissionDecision::PermanentRejected(d) => panic!("sequence {d:?}"),
        }
    };
    let session = sequence.open_session().unwrap();
    if program
        .plan()
        .execution_plan()
        .requires_selected_prefix_commit()
    {
        let mut specs = Vec::new();
        let inputs = program
            .plan()
            .parts()
            .prepared_family
            .program()
            .checkpoint_inputs()
            .unwrap()
            .conditioning_inputs()
            .iter()
            .map(|value| {
                let (dtype, data) = bytes(value.as_str(), history);
                let row_dimensions = if dtype == ElementType::U8 {
                    vec![]
                } else if value.as_str().contains("visual_rows") {
                    vec![16]
                } else {
                    vec![2]
                };
                specs.push(CausalTokenInputSpec {
                    value_id: value.clone(),
                    element_type: dtype,
                    row_dimensions,
                    storage: CausalTokenInputStorage::Exact,
                });
                CausalTokenInput::new(
                    value.clone(),
                    vec![CausalInputSegment::new(
                        0,
                        MAX_ROWS as usize,
                        allocate(program.host_account(), &data),
                    )
                    .unwrap()],
                )
            })
            .collect();
        let inputs = CausalRequestInputs::new(
            program.host_account(),
            &CausalInputSchema::new(specs).unwrap(),
            MAX_ROWS as usize,
            inputs,
        )
        .unwrap();
        program.bind_conditioning_inputs(&session, &inputs).unwrap();
    }
    session
}

fn prepare(
    program: &CausalProgramSession<CpuDeviceRuntime>,
    sessions: &[Arc<SequenceSession<CpuDeviceRuntime>>],
    histories: &[&[u32]],
    ranges: &[Range<u64>],
) -> (
    ExecutionBatchParticipants<CpuDeviceRuntime>,
    PreparedStepSubmissionWave<CpuDeviceRuntime>,
    Vec<SubmissionWaveInputUpload>,
    Arc<CompletionReaper<CpuDeviceRuntime>>,
) {
    let batch = ExecutionBatchParticipants::new(sessions.to_vec()).unwrap();
    let lane = program.resources().create_execution_lane().unwrap();
    let work = batch
        .bind_work_shape(
            histories
                .iter()
                .zip(ranges)
                .map(|(h, r)| span(h, r.clone()))
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
        match batch.try_begin_step(request.clone(), &lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(s) => break s,
            StepResourceAdmissionDecision::BackingDeferred(d) => progress(d.maintain().unwrap()),
            StepResourceAdmissionDecision::Deferred(d) => progress(
                program
                    .resources()
                    .maintain_for_admission_deferred(&d)
                    .unwrap(),
            ),
            StepResourceAdmissionDecision::PermanentRejected(d) => panic!("step {d:?}"),
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
            StepSubmissionWaveAdmissionDecision::Deferred(d) => progress(
                program
                    .resources()
                    .maintain_for_admission_deferred(&d)
                    .unwrap(),
            ),
            _ => panic!("wave has a state hazard"),
        }
    };
    let inputs = program.plan().parts().prepared_family.program().inputs();
    let mut uploads = Vec::new();
    for node in program.plan().execution_plan().payload().nodes() {
        for input in node
            .values()
            .iter()
            .filter(|v| v.role() == ResolvedValueRole::Input && inputs.contains(v.value_id()))
        {
            for (i, (history, range)) in histories.iter().zip(ranges).enumerate() {
                let (dtype, data) = bytes(input.value_id().as_str(), history);
                let row =
                    input.tensor().dimensions()[1..].iter().product::<u64>() * dtype.size_bytes();
                let offset = range.start * row;
                let length = (range.end - range.start) * row;
                let selected = &data[offset as usize..(offset + length) as usize];
                uploads.push(
                    SubmissionWaveInputUpload::managed_range(
                        node.id().clone(),
                        i as u32,
                        input.ordinal(),
                        offset,
                        HostTransferLayout::new(dtype, length / dtype.size_bytes()).unwrap(),
                        allocate(program.host_account(), selected),
                    )
                    .unwrap(),
                );
            }
        }
    }
    drop((step, lane));
    (batch, wave, uploads, CompletionReaper::new())
}
fn compute(
    program: &CausalProgramSession<CpuDeviceRuntime>,
    sessions: &[Arc<SequenceSession<CpuDeviceRuntime>>],
    histories: &[&[u32]],
    ranges: &[Range<u64>],
) -> CausalComputedWave<CpuDeviceRuntime> {
    let (batch, wave, uploads, reaper) = prepare(program, sessions, histories, ranges);
    // Preserve the original managed owner while exercising valid segmented
    // coverage, including the proposal token and both position tables.
    let uploads = uploads
        .into_iter()
        .flat_map(|upload| {
            let dtype = upload.source_layout().element_type();
            let element = dtype.size_bytes() as usize;
            let bytes = upload.managed_bytes().unwrap();
            let split = (bytes.len() / element / 2) * element;
            if split == 0 {
                return vec![upload];
            }
            [0..split, split..bytes.len()]
                .into_iter()
                .map(|range| {
                    SubmissionWaveInputUpload::managed_range(
                        upload.node_id().clone(),
                        upload.participant_index(),
                        upload.input_ordinal(),
                        upload.logical_offset_bytes() + range.start as u64,
                        HostTransferLayout::new(dtype, (range.len() / element) as u64).unwrap(),
                        bytes.slice(range).unwrap(),
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    let computed = program
        .execute_prepared_wave(&batch, wave, uploads, reaper)
        .unwrap();
    drop(batch);
    computed
}
fn observe(output: &CausalAcceptedOutput, host: &HostMemoryAccount) -> (u32, [u8; 32]) {
    let before = host.usage().unwrap();
    let input = output.copy_hidden_input(host).unwrap();
    assert!(host.owns(&input));
    assert_eq!(input.class(), HostMemoryClass::InputPreprocess);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        before.class_allocated_bytes(HostMemoryClass::InputPreprocess) + input.backing_bytes()
    );
    let result = (
        output.selected_decision(),
        input.bytes().try_into().unwrap(),
    );
    drop(input);
    assert_eq!(host.usage().unwrap(), before);
    result
}
fn finish(session: Arc<SequenceSession<CpuDeviceRuntime>>) {
    session.try_complete().unwrap();
}

#[test]
fn causal_product_wave_selects_n1_and_n2_then_continues_the_same_state() {
    let h = Harness::new();
    let history = [1, 2, 3];
    let full = session(&h.program, &history, "full");
    let result = compute(&h.program, &[full.clone()], &[&history], &[0..2])
        .commit(vec![SelectedPrefix::Two])
        .unwrap()
        .into_outputs();
    let expected = observe(result[0].as_ref().unwrap(), h.program.host_account());
    assert_eq!(result[0].as_ref().unwrap().frontier(), 2);
    drop(result);
    finish(full);
    let one = session(&h.program, &history, "one");
    let computed = compute(&h.program, &[one.clone()], &[&history], &[0..1]);
    let decisions = computed.decisions().next().unwrap();
    assert_eq!(decisions[0], decisions[1]);
    let first = computed
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    let first_expected = observe(first[0].as_ref().unwrap(), h.program.host_account());
    drop(first);
    finish(one);
    let selected = session(&h.program, &history, "selected");
    let first = compute(&h.program, &[selected.clone()], &[&history], &[0..2])
        .commit(vec![SelectedPrefix::One])
        .unwrap()
        .into_outputs();
    assert_eq!(first[0].as_ref().unwrap().frontier(), 1);
    assert_eq!(
        observe(first[0].as_ref().unwrap(), h.program.host_account()),
        first_expected
    );
    drop(first);
    let next = compute(&h.program, &[selected.clone()], &[&history], &[1..2])
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    assert_eq!(
        observe(next[0].as_ref().unwrap(), h.program.host_account()),
        expected
    );
    drop(next);
    finish(selected);
    let draft = session(&h.draft, &history, "draft");
    let result = compute(&h.draft, &[draft.clone()], &[&history], &[0..1])
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    assert_eq!(
        observe(result[0].as_ref().unwrap(), h.draft.host_account()).0,
        4
    );
    drop(result);
    finish(draft);
    h.close();
}

#[test]
fn causal_product_wave_b2_cancellation_keeps_survivor_state_and_hidden_private() {
    let h = Harness::new();
    let a = [1, 2, 3];
    let b = [5, 6, 7];
    let left = session(&h.program, &a, "b2-left");
    let right = session(&h.program, &b, "b2-right");
    let computed = compute(
        &h.program,
        &[left.clone(), right.clone()],
        &[&a, &b],
        &[0..2, 0..2],
    );
    assert_eq!(computed.decisions().len(), 2);
    left.request_cancel().unwrap();
    let outputs = computed
        .commit(vec![SelectedPrefix::One, SelectedPrefix::Two])
        .unwrap()
        .into_outputs();
    assert!(outputs[0].is_none());
    let survivor = outputs[1].as_ref().unwrap();
    assert_eq!(survivor.session(), right.fingerprint());
    assert_eq!(survivor.frontier(), 2);
    let expected = observe(survivor, h.program.host_account());
    assert!(survivor.copy_hidden_input(&host()).is_err());
    drop(outputs);
    left.try_abort_if_quiescent().unwrap();
    drop(left);
    let continued = compute(&h.program, &[right.clone()], &[&b], &[2..3])
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    let continued_expected = observe(continued[0].as_ref().unwrap(), h.program.host_account());
    drop(continued);
    finish(right);
    let alone = session(&h.program, &b, "b1-reference");
    let outputs = compute(&h.program, &[alone.clone()], &[&b], &[0..2])
        .commit(vec![SelectedPrefix::Two])
        .unwrap()
        .into_outputs();
    assert_eq!(
        observe(outputs[0].as_ref().unwrap(), h.program.host_account()),
        expected
    );
    drop(outputs);
    let outputs = compute(&h.program, &[alone.clone()], &[&b], &[2..3])
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    assert_eq!(
        observe(outputs[0].as_ref().unwrap(), h.program.host_account()),
        continued_expected
    );
    drop(outputs);
    finish(alone);
    h.close();
}

#[test]
fn causal_product_wave_verifies_actual_proposal_rejection_and_eos_before_publication() {
    let h = Harness::new();
    // Obtain q0 and its hidden carrier by actually executing the same root.
    // No fabricated completion or caller-supplied acceptance bit is involved.
    let seed_history = [1, 2];
    let reference = session(&h.program, &seed_history, "greedy-reference");
    let outputs = compute(&h.program, &[reference.clone()], &[&seed_history], &[0..1])
        .commit(vec![SelectedPrefix::All])
        .unwrap()
        .into_outputs();
    let root_output = observe(outputs[0].as_ref().unwrap(), h.program.host_account());
    let q0 = root_output.0;
    assert!(q0 < 32);
    drop(outputs);
    finish(reference);

    for (name, proposal, stops, accepted, stopped) in [
        ("accept", q0, BTreeSet::new(), true, false),
        ("reject", (q0 + 1) % 32, BTreeSet::new(), false, false),
        ("eos", q0, BTreeSet::from([q0]), false, true),
    ] {
        let history = [1, proposal];
        let sequence = session(&h.program, &history, name);
        let computed = compute(&h.program, &[sequence.clone()], &[&history], &[0..2]);
        let decisions = computed.decisions().next().unwrap();
        assert_eq!(decisions[0], q0, "the rejected suffix cannot affect q0");
        let mut outputs = computed.commit_one_draft(&[stops], &[2]).unwrap();
        let verified = outputs.pop().unwrap().unwrap();
        assert_eq!(verified.accepted_draft(), accepted);
        assert_eq!(verified.stopped(), stopped);
        assert_eq!(
            verified.tokens(),
            if accepted {
                &decisions[..]
            } else {
                &decisions[..1]
            }
        );
        let state = verified.into_state();
        assert_eq!(state.frontier(), if accepted { 2 } else { 1 });
        let actual = observe(&state, h.program.host_account());
        if !accepted {
            assert_eq!(actual, root_output);
        } else {
            assert_eq!(actual.0, decisions[1]);
        }
        drop(state);
        finish(sequence);
    }
    h.close();
}

#[test]
fn causal_product_wave_rejects_foreign_and_wrong_extent_inputs_before_dispatch() {
    let h = Harness::new();
    let history = [1, 2, 3];
    for mode in ["foreign", "extent", "gap", "overlap"] {
        let sequence = session(&h.program, &history, mode);
        let (batch, wave, mut uploads, reaper) =
            prepare(&h.program, &[sequence.clone()], &[&history], &[0..2]);
        let upload = uploads.remove(0);
        let account = if mode == "foreign" {
            host()
        } else {
            h.program.host_account().clone()
        };
        let source = allocate(&account, upload.managed_bytes().unwrap().bytes());
        let dtype = upload.source_layout().element_type();
        let element = dtype.size_bytes() as usize;
        assert!(source.len() >= 2 * element);
        let selected = if mode == "gap" {
            source.slice(element..source.len()).unwrap()
        } else {
            source.clone()
        };
        uploads.push(
            SubmissionWaveInputUpload::managed_range(
                upload.node_id().clone(),
                upload.participant_index(),
                upload.input_ordinal(),
                upload.logical_offset_bytes()
                    + if matches!(mode, "extent" | "gap") {
                        element as u64
                    } else {
                        0
                    },
                HostTransferLayout::new(dtype, (selected.len() / element) as u64).unwrap(),
                selected,
            )
            .unwrap(),
        );
        if mode == "overlap" {
            uploads.push(
                SubmissionWaveInputUpload::managed_range(
                    upload.node_id().clone(),
                    upload.participant_index(),
                    upload.input_ordinal(),
                    upload.logical_offset_bytes(),
                    HostTransferLayout::new(dtype, 1).unwrap(),
                    source.slice(0..element).unwrap(),
                )
                .unwrap(),
            );
        }
        drop(source);
        drop(upload);
        let failure = match h
            .program
            .execute_prepared_wave(&batch, wave, uploads, reaper.clone())
        {
            Ok(_) => panic!("invalid input executed"),
            Err(error) => error,
        };
        assert!(
            failure.completion().is_none()
                && failure.indeterminate().is_none()
                && failure.dispatch_failure().is_none()
        );
        assert_eq!(reaper.retained_count(), 0);
        failure.try_abort().unwrap();
        drop(batch);
        // An explicitly aborted frame poisons its sequence; use the existing
        // terminal abort, not the open-session/pre-admission abort shortcut.
        assert!(sequence.try_abort_if_quiescent().is_err());
        sequence.try_abort().unwrap();
        drop(sequence);
    }
    h.close();
}
