//! Real admitted waves and actual fence/retirement receipts exercise publication.
//! The command is a TestRuntime compute; this is ownership, not a CUDA output oracle.
use super::*;
use crate::vnext::completion::LaneSubmitOutcome;
use crate::vnext::operation::segment_compile::CompiledSegmentBindingRecipe;
use std::any::Any;
use std::sync::Weak;
use vnext_device_operation_wave_contract::{setup_with_fixture, wave_active_bindings};

struct PublicationFixture {
    fixture: Fixture,
    sequence: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
}

impl PublicationFixture {
    fn new() -> Self {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(
            fixture_with_retained_dependencies(16, DependencyMode::Valid),
        );
        fixture.runtime_trace.lock().unwrap().segment_entry =
            Some(DeviceReusableExecutionEntryIdentity::new());
        let lane = Arc::clone(step.execution_lane());
        Self {
            fixture,
            sequence,
            session,
            batch,
            lane,
            step: Some(step),
            reaper: CompletionReaper::new(),
        }
    }

    fn submit(
        &mut self,
        state: Arc<dyn Any + Send + Sync>,
    ) -> (
        Arc<CompiledSegmentBindingRecipe>,
        SegmentBindingPublication<TestRuntime>,
        OperationCompletionReceipt,
    ) {
        if self.step.is_none() {
            self.step = Some(begin_single_participant_step_on_lane_with_bucket(
                &self.batch,
                &self.lane,
                self.fixture.reusable_execution_bucket.as_ref(),
            ));
        }
        let mut wave = prepare_wave(
            &self.fixture.plan_resources,
            &self.fixture.plan,
            self.step.as_ref().unwrap(),
        );
        let providers = self
            .fixture
            .registry
            .bind_plan(&self.fixture.resolved)
            .unwrap();
        let program = OperationDispatch::reusable_execution_program_id_for_wave(
            providers.providers(),
            &self.fixture.resolved,
            &wave,
            &self.lane,
        )
        .unwrap()
        .expect("real reusable admission");
        let recipe = Arc::new(
            CompiledSegmentBindingRecipe::compile(
                &self.fixture.resolved,
                program.clone(),
                self.fixture
                    .plan
                    .payload()
                    .nodes()
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        let declaration = SegmentBindingDeclaration::new(
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
                        .unwrap();
                        (index, declaration)
                    })
                    .collect(),
            )
            .unwrap(),
        );
        drop(state);
        let active = wave_active_bindings(&wave, &self.session);
        let identity = OperationDispatch::bind_submission_wave_identity(
            &self.fixture.resolved,
            active.iter(),
            &wave,
            &self.lane,
        )
        .unwrap();
        let (entry, epoch) = self
            .lane
            .try_segment_binding_entry(&program)
            .unwrap()
            .unwrap();
        wave.begin_dispatch().unwrap();
        let mut reservation =
            CompletionReaper::reserve_wave(&self.reaper, wave, Arc::clone(&self.lane), identity)
                .unwrap();
        reservation
            .set_segment_binding_candidate(Arc::clone(&recipe), entry, epoch)
            .unwrap();
        let mut commands = DeviceCommandBatch::with_capacity(8);
        reservation
            .encode_backing_initializations(self.fixture.runtime.as_ref(), &mut commands)
            .unwrap();
        commands.push_compute(TestCommand::Provider);
        let mut enqueue = self.lane.reserve_enqueue().unwrap();
        reservation.mark_submission_started();
        let fence = match enqueue.submit(commands) {
            LaneSubmitOutcome::Submitted(fence) => fence,
            _ => panic!("fixture submission did not return a fence"),
        };
        drop(enqueue);
        let handle = match reservation.arm(fence, DeviceTimingMode::Off) {
            Ok(handle) => handle,
            Err((error, _)) => panic!("fixture fence installation failed: {error}"),
        };
        let clone = handle.clone();
        let ticket = clone.take_segment_binding_publication().unwrap();
        assert!(
            handle.take_segment_binding_publication().is_none(),
            "clones share take-once publication"
        );
        let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
            panic!("fixture fence did not terminalize")
        };
        assert!(matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        assert_eq!(self.lane.in_flight_count(), 0);
        assert_eq!(self.reaper.retained_count(), 0);
        (recipe, ticket, receipt)
    }

    fn retire(&mut self) -> StepRetirementReceipt {
        self.step.take().unwrap().try_retire_normal().unwrap()
    }

    fn close(self, cancelled: bool) {
        assert!(self.step.is_none());
        let Self {
            fixture,
            sequence,
            session,
            batch,
            lane,
            step: _,
            reaper,
        } = self;
        drop(batch);
        if cancelled {
            session.try_abort().unwrap();
        } else {
            session.try_complete().unwrap();
        }
        drop(session);
        drop(sequence);
        drop(reaper);
        drop(lane);
        drop(fixture.registry);
        drop(fixture.impostor_registry);
        drop(fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

#[test]
fn segment_publication_requires_terminal_and_exact_committed_retirement() {
    let mut fixture = PublicationFixture::new();
    let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&recipe.program_id)
        .unwrap()
        .is_none());
    let ready = ticket.bind_completion(&receipt).unwrap();
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&recipe.program_id)
        .unwrap()
        .is_none());
    let retirement = fixture.retire();
    assert!(retirement
        .participants()
        .iter()
        .all(|p| p.disposition() == StepParticipantRetirementDisposition::Committed));
    assert!(ready.publish(&retirement).unwrap());
    let cached = fixture
        .lane
        .lookup_segment_binding_recipe(&recipe.program_id)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&cached, &recipe));
    let enqueue = fixture.lane.reserve_enqueue().unwrap();
    enqueue.validate_segment_binding_recipe(&recipe).unwrap();
    drop(enqueue);
    let submitted_before = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;
    fixture.fixture.runtime_trace.lock().unwrap().segment_entry =
        Some(DeviceReusableExecutionEntryIdentity::new());
    let enqueue = fixture.lane.reserve_enqueue().unwrap();
    assert!(enqueue.validate_segment_binding_recipe(&cached).is_err());
    drop(enqueue);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted_before
    );
    drop(cached);
    drop(recipe);
    fixture.close(false);

    let mut fixture = PublicationFixture::new();
    let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
    fixture.session.request_cancel().unwrap();
    let retirement = fixture.retire();
    assert_eq!(
        retirement.participants()[0].disposition(),
        StepParticipantRetirementDisposition::DiscardedCancelled
    );
    assert!(ticket
        .bind_completion(&receipt)
        .unwrap()
        .publish(&retirement)
        .is_err());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&recipe.program_id)
        .unwrap()
        .is_none());
    drop(recipe);
    fixture.close(true);
}

#[test]
fn segment_publication_rejects_previous_slot_and_frame_receipts() {
    let mut fixture = PublicationFixture::new();
    let (recipe, ticket, old_receipt) = fixture.submit(Arc::new(()));
    drop(ticket);
    drop(recipe);
    let old_retirement = fixture.retire();
    let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
    assert_ne!(
        receipt.submission().slot_id(),
        old_receipt.submission().slot_id()
    );
    assert!(ticket.bind_completion(&old_receipt).is_none());
    drop(recipe);
    fixture.retire();
    let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
    assert!(ticket
        .bind_completion(&receipt)
        .unwrap()
        .publish(&old_retirement)
        .is_err());
    fixture.retire();
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&recipe.program_id)
        .unwrap()
        .is_none());
    drop(recipe);
    fixture.close(false);
}

#[test]
fn segment_publication_rechecks_exact_entry_and_trim_epoch() {
    for invalidate in 0..3 {
        let mut fixture = PublicationFixture::new();
        let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
        let ready = ticket.bind_completion(&receipt).unwrap();
        let retirement = fixture.retire();
        let old_epoch = fixture.lane.reusable_execution_epoch();
        match invalidate {
            0 => {
                fixture.fixture.runtime_trace.lock().unwrap().segment_entry =
                    Some(DeviceReusableExecutionEntryIdentity::new())
            }
            1 => fixture.fixture.runtime_trace.lock().unwrap().segment_entry = None,
            2 => {
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
                assert!(fixture.lane.reusable_execution_epoch() > old_epoch);
            }
            _ => unreachable!(),
        }
        assert!(!ready.publish(&retirement).unwrap());
        assert!(fixture
            .lane
            .lookup_segment_binding_recipe(&recipe.program_id)
            .unwrap()
            .is_none());
        drop(recipe);
        fixture.close(false);
    }
}

struct DropProbe {
    lane: Weak<ExecutionLane<TestRuntime>>,
    dropped: Arc<AtomicU64>,
}
impl Drop for DropProbe {
    fn drop(&mut self) {
        if let Some(lane) = self.lane.upgrade() {
            assert!(
                lane.segment_binding_locks_available_for_test(),
                "recipe Drop ran under a lane or cache lock"
            );
        }
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn segment_publication_cache_replacement_eviction_and_trim_drop_outside_locks() {
    let mut fixture = PublicationFixture::new();
    let dropped = Arc::new(AtomicU64::new(0));
    for index in 0..3 {
        let state = Arc::new(DropProbe {
            lane: Arc::downgrade(&fixture.lane),
            dropped: Arc::clone(&dropped),
        });
        let (recipe, ticket, receipt) = fixture.submit(state);
        let retirement = fixture.retire();
        assert!(ticket
            .bind_completion(&receipt)
            .unwrap()
            .publish(&retirement)
            .unwrap());
        let program = recipe.program_id.clone();
        drop(recipe);
        if index == 0 {
            assert_eq!(dropped.load(Ordering::Relaxed), 0);
            // Publishing the next real frame replaces this sole cached owner.
        } else if index == 1 {
            assert_eq!(dropped.load(Ordering::Relaxed), 1);
            fixture.fixture.runtime_trace.lock().unwrap().segment_entry =
                Some(DeviceReusableExecutionEntryIdentity::new());
            assert!(fixture
                .lane
                .lookup_segment_binding_recipe(&program)
                .unwrap()
                .is_none());
            assert_eq!(dropped.load(Ordering::Relaxed), 2);
        } else {
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
            assert_eq!(dropped.load(Ordering::Relaxed), 3);
        }
    }
    fixture.close(false);
}

#[test]
fn segment_publication_pending_does_not_keep_lane_or_step_alive() {
    let mut fixture = PublicationFixture::new();
    let lane = Arc::downgrade(&fixture.lane);
    let step = Arc::downgrade(fixture.step.as_ref().unwrap());
    let (recipe, ticket, receipt) = fixture.submit(Arc::new(()));
    let ready = ticket.bind_completion(&receipt).unwrap();
    let retirement = fixture.retire();
    assert!(step.upgrade().is_none());
    drop(recipe);
    fixture.close(false);
    assert!(lane.upgrade().is_none());
    assert!(!ready.publish(&retirement).unwrap());
}
