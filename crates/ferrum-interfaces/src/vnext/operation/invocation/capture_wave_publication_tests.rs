//! A first eager capture supplies declarations, never an executable authority.
//! Publication uses real admitted waves, provider-issued dependencies and fences.
use super::*;
use crate::vnext::operation::retained_dependency::append_dependencies;

struct CapturedSubmission {
    recipe: Arc<CompiledSegmentBindingRecipe>,
    capture: DeviceReusableExecutionCapture,
    handle: CompletionHandle<TestRuntime>,
    ticket: SegmentBindingPublication<TestRuntime>,
}

impl CapturedSubmission {
    fn catalog_row(&self) -> DeviceReusableExecutionProgram {
        DeviceReusableExecutionProgram::new(
            &self.capture,
            vec![DeviceReusableExecutionSegment::new(
                0,
                0,
                self.capture.node_count(),
                self.capture.node_count(),
            )
            .unwrap()],
            self.capture.per_wave_binding_node_indices().to_vec(),
            vec![],
        )
        .unwrap()
    }
}

fn submit_capture(
    fixture: &mut PublicationFixture,
    state: Arc<dyn Any + Send + Sync>,
) -> CapturedSubmission {
    if fixture.step.is_none() {
        fixture.step = Some(begin_single_participant_step_on_lane_with_bucket(
            &fixture.batch,
            &fixture.lane,
            fixture.fixture.reusable_execution_bucket.as_ref(),
        ));
    }
    let mut wave = prepare_wave(
        &fixture.fixture.plan_resources,
        &fixture.fixture.plan,
        fixture.step.as_ref().unwrap(),
    );
    let providers = fixture
        .fixture
        .registry
        .bind_plan(&fixture.fixture.resolved)
        .unwrap();
    let program = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.fixture.resolved,
        &wave,
        &fixture.lane,
    )
    .unwrap()
    .expect("real reusable admission");
    let active = wave_active_bindings(&wave, &fixture.session);
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.fixture.resolved,
        active.iter(),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let mut dependencies = Vec::new();
    let mut dependency_commands = Vec::new();
    let mut dependency_leases = Vec::new();
    let mut operations = Vec::new();
    for (index, provider) in providers.providers().iter().enumerate() {
        let node = identity.materialize_node(index).unwrap();
        let invocation = BatchedOperationInvocation::from_wave_node(
            fixture.fixture.runtime.as_ref(),
            &fixture.fixture.resolved,
            provider.dispatch(),
            &identity,
            node,
            &wave,
            index,
            active.iter(),
        )
        .unwrap();
        let scope = Arc::clone(&invocation.retained_dependency_scope);
        let mut operation = provider.provider().encode_selected(invocation).unwrap();
        append_dependencies(
            &scope,
            operation.take_retained_plan_dependencies(),
            &mut dependencies,
            &mut dependency_commands,
            &mut dependency_leases,
        )
        .unwrap();
        operations.push((index as u32, operation.into_parts()));
    }
    assert!(
        !dependencies.is_empty(),
        "fixture must exercise the real seal"
    );
    wave.step_resources()
        .retain_plan_dependencies(dependency_leases)
        .unwrap();
    let node_count = u32::try_from(providers.providers().len()).unwrap();
    let capture = DeviceReusableExecutionCapture::new(
        program.clone(),
        node_count,
        vec![],
        (0..node_count).collect(),
    )
    .unwrap()
    .with_retained_plan_dependencies(dependencies);
    let recipe = Arc::new(
        CompiledSegmentBindingRecipe::compile(
            &fixture.fixture.resolved,
            program,
            (0..node_count)
                .map(|index| {
                    (
                        index as usize,
                        SegmentBindingDeclaration::new(
                            vec![SegmentBindingRegionRequest {
                                selector: SegmentBindingRegionSelector::Persistent,
                                offset_bytes: 4,
                                extent: SegmentBindingRegionExtent::Exact(4),
                                element_type: ElementType::U8,
                                alignment_bytes: 4,
                            }],
                            vec![],
                            Arc::clone(&state),
                        )
                        .unwrap(),
                    )
                })
                .collect(),
        )
        .unwrap(),
    );
    wave.begin_dispatch().unwrap();
    let mut reservation =
        CompletionReaper::reserve_wave(&fixture.reaper, wave, Arc::clone(&fixture.lane), identity)
            .unwrap();
    reservation
        .set_captured_segment_binding_candidate(
            Arc::clone(&recipe),
            capture.clone(),
            fixture.lane.reusable_execution_epoch(),
        )
        .unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(16);
    reservation
        .encode_backing_initializations(fixture.fixture.runtime.as_ref(), &mut commands)
        .unwrap();
    for command in dependency_commands {
        commands.push_dynamic_binding(command);
    }
    for (index, (bindings, dynamic, compute, results)) in operations {
        for command in bindings {
            commands.push_dynamic_binding(command);
        }
        commands.push_operation_parts(index, dynamic, compute, results);
    }
    commands
        .set_reusable_execution_capture(capture.clone())
        .unwrap();
    let mut enqueue = fixture.lane.reserve_enqueue().unwrap();
    reservation.mark_submission_started();
    let fence = match enqueue.submit(commands) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("first-capture fixture did not submit"),
    };
    drop(enqueue);
    let handle = match reservation.arm(fence, DeviceTimingMode::Off) {
        Ok(handle) => handle,
        Err((error, _)) => panic!("first-capture fixture fence failed: {error}"),
    };
    let ticket = handle.take_segment_binding_publication().unwrap();
    assert!(handle.clone().take_segment_binding_publication().is_none());
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_reusable_captures
            .last(),
        Some(&Some(capture.clone()))
    );
    CapturedSubmission {
        recipe,
        capture,
        handle,
        ticket,
    }
}

fn fresh_catalog(fixture: &PublicationFixture, row: DeviceReusableExecutionProgram) {
    let mut trace = fixture.fixture.runtime_trace.lock().unwrap();
    trace.segment_catalog = vec![row];
    trace.segment_entry = Some(DeviceReusableExecutionEntryIdentity::new());
}

fn terminal(handle: &CompletionHandle<TestRuntime>) -> OperationCompletionReceipt {
    match handle.wait().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("fixture did not terminalize"),
    }
}

#[test]
fn capture_wave_publication_waits_for_completion_and_exact_retirement() {
    let mut fixture = PublicationFixture::new();
    fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
    fixture.fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Pending;
    let submitted = submit_capture(&mut fixture, Arc::new(()));
    assert!(matches!(
        submitted.handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert_eq!(fixture.lane.in_flight_count(), 1);
    assert_eq!(fixture.reaper.retained_count(), 1);
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&submitted.recipe.program_id)
        .unwrap()
        .is_none());
    fresh_catalog(&fixture, submitted.catalog_row());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&submitted.recipe.program_id)
        .unwrap()
        .is_none());
    fixture.fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    let receipt = terminal(&submitted.handle);
    let ready = submitted.ticket.bind_completion(&receipt).unwrap();
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&submitted.recipe.program_id)
        .unwrap()
        .is_none());
    let retirement = fixture.retire();
    assert!(ready.publish(&retirement).unwrap());
    let cached = fixture
        .lane
        .lookup_segment_binding_recipe(&submitted.recipe.program_id)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&cached, &submitted.recipe));
    let enqueue = fixture.lane.reserve_enqueue().unwrap();
    enqueue.validate_segment_binding_recipe(&cached).unwrap();
    drop(enqueue);
    fixture.fixture.runtime_trace.lock().unwrap().segment_entry =
        Some(DeviceReusableExecutionEntryIdentity::new());
    let enqueue = fixture.lane.reserve_enqueue().unwrap();
    assert!(enqueue.validate_segment_binding_recipe(&cached).is_err());
    drop(enqueue);
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&submitted.recipe.program_id)
        .unwrap()
        .is_none());
    drop(cached);
    drop(submitted.recipe);
    drop(submitted.handle);
    fixture.close(false);
}

#[test]
fn capture_wave_publication_missing_warm_catalog_and_cancelled_retirement_are_cold() {
    for cancel in [false, true] {
        let mut fixture = PublicationFixture::new();
        fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
        let submitted = submit_capture(&mut fixture, Arc::new(()));
        let receipt = terminal(&submitted.handle);
        if cancel {
            fresh_catalog(&fixture, submitted.catalog_row());
            fixture.session.request_cancel().unwrap();
        }
        let retirement = fixture.retire();
        let result = submitted
            .ticket
            .bind_completion(&receipt)
            .unwrap()
            .publish(&retirement);
        if cancel {
            assert!(result.is_err());
        } else {
            assert!(
                !result.unwrap(),
                "a warm eager wave has no resident catalog yet"
            );
        }
        assert!(fixture
            .lane
            .lookup_segment_binding_recipe(&submitted.recipe.program_id)
            .unwrap()
            .is_none());
        drop(submitted.recipe);
        drop(submitted.handle);
        fixture.close(cancel);
    }
}

#[test]
fn capture_wave_publication_failed_and_indeterminate_fences_never_authorize() {
    for behavior in [
        FenceBehavior::FailedButQuiescent,
        FenceBehavior::Indeterminate,
    ] {
        let mut fixture = PublicationFixture::new();
        fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
        fixture.fixture.runtime_trace.lock().unwrap().fence_behavior = behavior;
        let submitted = submit_capture(&mut fixture, Arc::new(()));
        fresh_catalog(&fixture, submitted.catalog_row());
        if behavior == FenceBehavior::Indeterminate {
            assert!(matches!(
                submitted.handle.poll().unwrap(),
                CompletionObservation::Indeterminate(_)
            ));
            assert_eq!(fixture.reaper.retained_count(), 1);
            assert!(fixture
                .lane
                .lookup_segment_binding_recipe(&submitted.recipe.program_id)
                .unwrap()
                .is_none());
            fixture.fixture.runtime_trace.lock().unwrap().fence_behavior =
                FenceBehavior::FailedButQuiescent;
        }
        let receipt = terminal(&submitted.handle);
        assert!(submitted.ticket.bind_completion(&receipt).is_none());
        fixture.session.request_cancel().unwrap();
        fixture.retire();
        assert!(fixture
            .lane
            .lookup_segment_binding_recipe(&submitted.recipe.program_id)
            .unwrap()
            .is_none());
        drop(submitted.recipe);
        drop(submitted.handle);
        fixture.close(true);
    }
}

#[derive(Clone, Copy, Debug)]
enum CatalogFault {
    FullProgram,
    NodeCount,
    EagerBoundary,
    MissingMembership,
    RetainedSeal,
    MissingEntry,
    ReplacedDuringCatalog,
    DuplicateFullProgram,
    Epoch,
}

#[test]
fn capture_wave_publication_rejects_changed_catalog_seal_and_lane_epoch() {
    for fault in [
        CatalogFault::FullProgram,
        CatalogFault::NodeCount,
        CatalogFault::EagerBoundary,
        CatalogFault::MissingMembership,
        CatalogFault::RetainedSeal,
        CatalogFault::MissingEntry,
        CatalogFault::ReplacedDuringCatalog,
        CatalogFault::DuplicateFullProgram,
        CatalogFault::Epoch,
    ] {
        let mut fixture = PublicationFixture::new();
        fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
        let submitted = submit_capture(&mut fixture, Arc::new(()));
        let expected_catalog = submitted.catalog_row();
        let mut row = expected_catalog.clone();
        match fault {
            CatalogFault::FullProgram | CatalogFault::NodeCount | CatalogFault::RetainedSeal => {
                let program = if matches!(fault, CatalogFault::FullProgram) {
                    submitted
                        .capture
                        .program_id()
                        .clone()
                        .with_topology_fingerprint(
                            DeviceReusableExecutionTopologyFingerprint::from_sha256([9; 32]),
                        )
                } else {
                    submitted.capture.program_id().clone()
                };
                let count = submitted.capture.node_count()
                    + u32::from(matches!(fault, CatalogFault::NodeCount));
                let mut changed = DeviceReusableExecutionCapture::new(
                    program,
                    count,
                    vec![],
                    submitted.capture.per_wave_binding_node_indices().to_vec(),
                )
                .unwrap();
                if !matches!(fault, CatalogFault::RetainedSeal) {
                    changed = changed.with_retained_plan_dependencies(
                        submitted.capture.retained_plan_dependencies().to_vec(),
                    );
                }
                row = DeviceReusableExecutionProgram::new(
                    &changed,
                    vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
                    changed.per_wave_binding_node_indices().to_vec(),
                    vec![],
                )
                .unwrap();
            }
            CatalogFault::EagerBoundary => {
                let changed = DeviceReusableExecutionCapture::new(
                    submitted.capture.program_id().clone(),
                    submitted.capture.node_count(),
                    vec![0],
                    submitted.capture.per_wave_binding_node_indices().to_vec(),
                )
                .unwrap()
                .with_retained_plan_dependencies(
                    submitted.capture.retained_plan_dependencies().to_vec(),
                );
                row = DeviceReusableExecutionProgram::new(
                    &changed,
                    vec![],
                    vec![],
                    (1..changed.node_count())
                        .map(|index| {
                            DeviceReusableExecutionProgramGap::new(
                                index,
                                DeviceReusableExecutionProgramGapReason::WarmupRequired,
                            )
                        })
                        .collect(),
                )
                .unwrap();
            }
            CatalogFault::MissingMembership => {
                row = DeviceReusableExecutionProgram::new(
                    &submitted.capture,
                    row.segments().to_vec(),
                    vec![],
                    vec![],
                )
                .unwrap();
            }
            CatalogFault::MissingEntry
            | CatalogFault::ReplacedDuringCatalog
            | CatalogFault::DuplicateFullProgram
            | CatalogFault::Epoch => {}
        }
        fresh_catalog(&fixture, row);
        let receipt = terminal(&submitted.handle);
        let ready = submitted.ticket.bind_completion(&receipt).unwrap();
        let retirement = fixture.retire();
        if matches!(fault, CatalogFault::MissingEntry) {
            fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
        }
        if matches!(fault, CatalogFault::ReplacedDuringCatalog) {
            fixture
                .fixture
                .runtime_trace
                .lock()
                .unwrap()
                .segment_catalog_replaces_entry = true;
        }
        if matches!(fault, CatalogFault::DuplicateFullProgram) {
            fixture
                .fixture
                .runtime_trace
                .lock()
                .unwrap()
                .segment_catalog
                .push(expected_catalog.clone());
        }
        if matches!(fault, CatalogFault::Epoch) {
            fixture
                .fixture
                .runtime_trace
                .lock()
                .unwrap()
                .segment_trim_released = 1;
            assert!(fixture
                .lane
                .trim_reusable_executables_if_quiescent()
                .unwrap());
            fresh_catalog(&fixture, expected_catalog);
        }
        assert!(!ready.publish(&retirement).unwrap(), "{fault:?}");
        assert!(fixture
            .lane
            .lookup_segment_binding_recipe(&submitted.recipe.program_id)
            .unwrap()
            .is_none());
        drop(submitted.recipe);
        drop(submitted.handle);
        fixture.close(false);
    }
}

#[test]
fn capture_wave_publication_does_not_retain_step_or_lane() {
    let mut fixture = PublicationFixture::new();
    fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None;
    let lane = Arc::downgrade(&fixture.lane);
    let step = Arc::downgrade(fixture.step.as_ref().unwrap());
    let dropped = Arc::new(AtomicU64::new(0));
    let probe = Arc::new(DropProbe {
        lane: Arc::downgrade(&fixture.lane),
        dropped: Arc::clone(&dropped),
    });
    let submitted = submit_capture(&mut fixture, probe);
    fresh_catalog(&fixture, submitted.catalog_row());
    let receipt = terminal(&submitted.handle);
    let ready = submitted.ticket.bind_completion(&receipt).unwrap();
    let retirement = fixture.retire();
    drop(submitted.recipe);
    drop(submitted.handle);
    assert!(step.upgrade().is_none());
    fixture.close(false);
    assert!(lane.upgrade().is_none());
    assert!(!ready.publish(&retirement).unwrap());
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
}
