//! Real admitted waves and actual fence/retirement receipts exercise publication.
//! The command is a TestRuntime compute; this is ownership, not a CUDA output oracle.
use super::*;
use crate::vnext::completion::LaneSubmitOutcome;
use crate::vnext::operation::segment_compile::CompiledSegmentBindingRecipe;
use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::Weak;
use vnext_device_operation_wave_contract::setup_with_fixture;

struct PublicationFixture {
    fixture: Fixture,
    sequence: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    additional_sequences: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    additional_sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
}

impl PublicationFixture {
    fn new() -> Self {
        Self::with_fixture(fixture_with_retained_dependencies(
            16,
            DependencyMode::Valid,
        ))
    }

    fn with_fixture(fixture: Fixture) -> Self {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture);
        fixture.runtime_trace.lock().unwrap().segment_entry =
            Some(DeviceReusableExecutionEntryIdentity::new());
        let lane = Arc::clone(step.execution_lane());
        Self {
            fixture,
            sequence,
            session,
            additional_sequences: Vec::new(),
            additional_sessions: Vec::new(),
            batch,
            lane,
            step: Some(step),
            reaper: CompletionReaper::new(),
        }
    }

    fn multi_shape(participant_count: usize) -> Self {
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("execution.segment-publication-shapes").unwrap(),
            ReusableExecutionCapacity::new(3, 3, 1).unwrap(),
        )
        .unwrap();
        let mut this = Self::with_fixture(fixture_with_retained_dependencies_and_bucket(
            16,
            DependencyMode::Valid,
            Some(bucket),
        ));
        this.lane
            .configure_segment_binding_recipe_capacity_for_test(NonZeroUsize::new(2).unwrap())
            .unwrap();
        {
            let mut trace = this.fixture.runtime_trace.lock().unwrap();
            trace.segment_entries = Some(BTreeMap::new());
            trace.segment_metadata_enabled = true;
            trace.segment_encoder_enabled = true;
        }
        // Only admit sessions exercised by this case: an unused session has
        // no retired frame and therefore cannot be completed successfully.
        for index in 1..participant_count {
            let sequence = logical_resources(
                &this.fixture.plan_resources,
                &format!("run.segment-publication.{index}"),
                &format!("request.segment-publication.{index}"),
            );
            this.additional_sessions
                .push(sequence.open_session().unwrap());
            this.additional_sequences.push(sequence);
        }
        this
    }

    fn set_width(&mut self, width: usize) {
        assert!(self.step.is_none());
        let sessions = std::iter::once(&self.session)
            .chain(&self.additional_sessions)
            .take(width)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(sessions.len(), width);
        self.batch = ExecutionBatchParticipants::new(sessions).unwrap();
    }

    fn ensure_step(&mut self) {
        if self.step.is_some() {
            return;
        }
        if self.batch.sessions().len() == 1 {
            self.step = Some(begin_single_participant_step_on_lane_with_bucket(
                &self.batch,
                &self.lane,
                self.fixture.reusable_execution_bucket.as_ref(),
            ));
            return;
        }
        let request = StepResourceAdmissionRequest::new(
            self.batch
                .bind_work_shape(vec![one_token_span(); self.batch.sessions().len()])
                .unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
        .with_reusable_execution_bucket(
            self.fixture
                .reusable_execution_bucket
                .as_ref()
                .unwrap()
                .bucket_id()
                .clone(),
        );
        for attempt in 0..=3 {
            match self
                .batch
                .try_begin_step(request.clone(), &self.lane)
                .unwrap()
            {
                StepResourceAdmissionDecision::Admitted(step) => {
                    self.step = Some(step);
                    return;
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("publication fixture step admission did not converge"),
            }
        }
        unreachable!("bounded admission returns or fails")
    }

    fn submit(
        &mut self,
        state: Arc<dyn Any + Send + Sync>,
    ) -> (
        Arc<CompiledSegmentBindingRecipe>,
        SegmentBindingPublication<TestRuntime>,
        OperationCompletionReceipt,
    ) {
        self.ensure_step();
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
        // Entries belong to programs derived from real admitted waves.
        if let Some(entries) = &mut self.fixture.runtime_trace.lock().unwrap().segment_entries {
            entries
                .entry(program.clone())
                .or_insert_with(DeviceReusableExecutionEntryIdentity::new);
        }
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
        let active = self
            .batch
            .sessions()
            .iter()
            .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
            .collect::<Vec<_>>();
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

    fn encode_cached(
        &mut self,
        expected: &Arc<CompiledSegmentBindingRecipe>,
    ) -> (BatchStepId, Vec<Arc<()>>) {
        self.ensure_step();
        let wave = prepare_wave(
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
        .unwrap();
        drop(providers);
        assert_eq!(program, expected.program_id);
        let selected = self
            .lane
            .lookup_segment_binding_recipe(&program)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&selected, expected));
        let active = self
            .batch
            .sessions()
            .iter()
            .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
            .collect::<Vec<_>>();
        let identity =
            super::segment_authority_tests::segment_identity(&self.fixture, &wave, &active);
        let before = self.fixture.provider_trace.lock().unwrap().encode_calls;
        let (encoded, _) = crate::vnext::operation::segment_dispatch::encode_segment_wave(
            self.fixture.runtime.as_ref(),
            &self.fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &selected,
        )
        .unwrap()
        .unwrap();
        assert_eq!(encoded.len(), selected.nodes.len());
        assert_eq!(
            self.fixture.provider_trace.lock().unwrap().encode_calls,
            before
        );
        assert_eq!(
            self.fixture
                .runtime_trace
                .lock()
                .unwrap()
                .segment_regions
                .len(),
            selected
                .nodes
                .iter()
                .map(|node| node.regions.len())
                .sum::<usize>()
                * self.batch.sessions().len()
        );
        let enqueue = self.lane.reserve_enqueue().unwrap();
        enqueue.validate_segment_binding_recipe(&selected).unwrap();
        drop(enqueue);
        let step_id = wave.batch_step_id();
        let scopes = encoded.iter().map(|node| Arc::clone(&node.scope)).collect();
        drop(encoded);
        drop(identity);
        drop(active);
        drop(wave);
        self.retire();
        (step_id, scopes)
    }

    fn retire(&mut self) -> StepRetirementReceipt {
        self.step.take().unwrap().try_retire_normal().unwrap()
    }

    fn publish(&mut self, state: Arc<dyn Any + Send + Sync>) -> Arc<CompiledSegmentBindingRecipe> {
        let (recipe, ticket, receipt) = self.submit(state);
        let retirement = self.retire();
        assert!(ticket
            .bind_completion(&receipt)
            .unwrap()
            .publish(&retirement)
            .unwrap());
        recipe
    }

    fn close(self, cancelled: bool) {
        assert!(self.step.is_none());
        let Self {
            fixture,
            sequence,
            session,
            additional_sequences,
            additional_sessions,
            batch,
            lane,
            step: _,
            reaper,
        } = self;
        drop(batch);
        for additional in &additional_sessions {
            additional.try_complete().unwrap();
        }
        drop(additional_sessions);
        drop(additional_sequences);
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

#[test]
fn segment_publication_multi_shape_reuses_fresh_frames_and_evicts_only_lru_outside_locks() {
    let mut fixture = PublicationFixture::multi_shape(3);
    let dropped = Arc::new(AtomicU64::new(0));
    let state = || {
        Arc::new(DropProbe {
            lane: Arc::downgrade(&fixture.lane),
            dropped: Arc::clone(&dropped),
        })
    };
    let first_state = state();
    let second_state = state();
    let third_state = state();
    let a = fixture.publish(first_state);
    fixture.set_width(2);
    let b = fixture.publish(second_state);
    assert_ne!(a.program_id, b.program_id);
    assert_eq!(a.program_id.immediate_sequences(), 1);
    assert_eq!(b.program_id.immediate_sequences(), 2);
    let a_program = a.program_id.clone();
    let b_program = b.program_id.clone();
    let a_weak = Arc::downgrade(&a);
    let b_weak = Arc::downgrade(&b);

    // Each lookup is followed by the actual whole-segment encoder consuming
    // a fresh admitted wave. No provider encode callback manufactures a hit.
    fixture.set_width(1);
    let (first_a_frame, first_a_scopes) = fixture.encode_cached(&a);
    fixture.set_width(2);
    let (b_frame, _) = fixture.encode_cached(&b);
    fixture
        .lane
        .exhaust_segment_binding_recipe_age_for_test()
        .unwrap();
    fixture.set_width(1);
    let (second_a_frame, second_a_scopes) = fixture.encode_cached(&a);
    assert_ne!(first_a_frame, b_frame);
    assert_ne!(first_a_frame, second_a_frame);
    for (previous, current) in first_a_scopes.iter().zip(&second_a_scopes) {
        assert!(!Arc::ptr_eq(previous, current));
    }
    drop(a);
    drop(b);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);

    fixture.set_width(3);
    let c = fixture.publish(third_state);
    let c_program = c.program_id.clone();
    assert_ne!(c_program, a_program);
    assert_ne!(c_program, b_program);
    assert_eq!(c_program.immediate_sequences(), 3);
    drop(c);
    assert!(b_weak.upgrade().is_none());
    assert!(a_weak.upgrade().is_some());
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&b_program)
        .unwrap()
        .is_none());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&a_program)
        .unwrap()
        .is_some());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&c_program)
        .unwrap()
        .is_some());

    // Replacing A's actual runtime entry invalidates A alone; C survives.
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_entries
        .as_mut()
        .unwrap()
        .insert(
            a_program.clone(),
            DeviceReusableExecutionEntryIdentity::new(),
        );
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&a_program)
        .unwrap()
        .is_none());
    assert_eq!(dropped.load(Ordering::Relaxed), 2);
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&c_program)
        .unwrap()
        .is_some());
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_trim_released = 3;
    assert!(fixture
        .lane
        .trim_reusable_executables_if_quiescent()
        .unwrap());
    assert_eq!(dropped.load(Ordering::Relaxed), 3);
    fixture.close(false);
}

#[test]
fn segment_publication_multi_shape_old_entry_ticket_and_trim_epoch_cannot_publish() {
    let mut fixture = PublicationFixture::multi_shape(2);
    let (a, ticket, receipt) = fixture.submit(Arc::new(()));
    let ready_a = ticket.bind_completion(&receipt).unwrap();
    let retirement_a = fixture.retire();
    fixture.set_width(2);
    let b = fixture.publish(Arc::new(()));
    assert_ne!(a.program_id, b.program_id);
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_entries
        .as_mut()
        .unwrap()
        .insert(
            a.program_id.clone(),
            DeviceReusableExecutionEntryIdentity::new(),
        );
    assert!(!ready_a.publish(&retirement_a).unwrap());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&a.program_id)
        .unwrap()
        .is_none());
    assert!(Arc::ptr_eq(
        &fixture
            .lane
            .lookup_segment_binding_recipe(&b.program_id)
            .unwrap()
            .unwrap(),
        &b
    ));

    fixture.set_width(1);
    let (new_a, ticket, receipt) = fixture.submit(Arc::new(()));
    assert_eq!(new_a.program_id, a.program_id);
    let ready_a = ticket.bind_completion(&receipt).unwrap();
    let retirement_a = fixture.retire();
    let old_epoch = fixture.lane.reusable_execution_epoch();
    fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_trim_released = 2;
    assert!(fixture
        .lane
        .trim_reusable_executables_if_quiescent()
        .unwrap());
    assert!(fixture.lane.reusable_execution_epoch() > old_epoch);
    assert!(!ready_a.publish(&retirement_a).unwrap());
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&b.program_id)
        .unwrap()
        .is_none());
    drop(new_a);
    drop(a);
    drop(b);
    fixture.close(false);
}

#[test]
fn segment_publication_evicted_recipe_arc_cannot_reauthorize_enqueue() {
    let mut fixture = PublicationFixture::multi_shape(3);
    let a = fixture.publish(Arc::new(()));
    fixture.set_width(2);
    let b = fixture.publish(Arc::new(()));
    assert!(fixture
        .lane
        .lookup_segment_binding_recipe(&a.program_id)
        .unwrap()
        .is_some());
    fixture.set_width(3);
    let c = fixture.publish(Arc::new(()));
    // B's runtime entry is still resident, but its old caller-held recipe is
    // no longer authorized by this cache after real capacity-driven eviction.
    assert!(fixture
        .lane
        .reusable_execution_entry_identity(&b.program_id)
        .unwrap()
        .is_some());
    let submitted_before = fixture.fixture.runtime_trace.lock().unwrap().submit_calls;
    let enqueue = fixture.lane.reserve_enqueue().unwrap();
    enqueue.validate_segment_binding_recipe(&a).unwrap();
    enqueue.validate_segment_binding_recipe(&c).unwrap();
    assert!(enqueue.validate_segment_binding_recipe(&b).is_err());
    drop(enqueue);
    assert_eq!(
        fixture.fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted_before
    );
    drop(a);
    drop(b);
    drop(c);
    fixture.close(false);
}
